"""Interactive axisymmetric FEM solenoid stress explorer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import factorized

import cfsem
from cfsem.solenoid_stress.axisymmetric_fem import (
    apply_dirichlet,
    assemble_axisymmetric,
    cfsem_radial_material,
    element_measures_axisymmetric,
    element_quadrature_axisymmetric,
)
from cfsem.solenoid_stress.solenoid_1d import (
    SolenoidStress1D,
    solenoid_1d_structural_factor,
    solenoid_1d_structural_rhs,
)

TESTING = bool(os.getenv("CFSEM_TESTING"))

SOLENOID_INNER_RADIUS = 0.5  # [m]
YOUNGS_MODULUS = 200.0e9  # [Pa]
POISSON_RATIO = 0.27  # [-]
QUADRATURE = "2x2"

DEFAULT_WIDTH = 0.18  # [m]
DEFAULT_HEIGHT = 0.24  # [m]
DEFAULT_CURRENT_DENSITY_MA = 75.0  # [MA/m^2]
DEFAULT_SOURCE_RADIUS = 0.32  # [m]
DEFAULT_SOURCE_Z = 0.0  # [m]
DEFAULT_SOURCE_CURRENT_MA = 1.2  # [MA-turn]

WIDTH_RANGE = (0.05, 0.35)
HEIGHT_RANGE = (0.05, 0.50)
CURRENT_DENSITY_RANGE_MA = (0.0, 200.0)
SOURCE_RADIUS_RANGE = (0.05, 1.20)
SOURCE_Z_RANGE = (-0.60, 0.60)
SOURCE_CURRENT_RANGE_MA = (0.0, 5.0)

SECTION_TARGET_FRACTIONS = (0.2, 0.5, 0.8)
SECTION_LABELS = ("Lower", "Middle", "Upper")
SECTION_COLORS = ("firebrick", "royalblue", "darkgreen")

MESH_LONG_SIDE_ELEMENTS = 14 if TESTING else 28
FIELD_GRID_R = 121 if TESTING else 241
FIELD_GRID_Z = 141 if TESTING else 281
LOG10_FLOOR = -16.0

DOCS_EXAMPLE_HTML = (
    Path(__file__).resolve().parents[1] / "docs/python/example_outputs/solenoid_stress_axisymmetric_fem.html"
)


@dataclass(frozen=True, slots=True)
class SectionComparison:
    label: str
    color: str
    z_value: float
    radius: np.ndarray
    b_z_fe: np.ndarray
    b_z_1d: np.ndarray
    u_r_fe: np.ndarray
    u_r_1d: np.ndarray
    e_rr_fe: np.ndarray
    e_rr_1d: np.ndarray
    e_tt_fe: np.ndarray
    e_tt_1d: np.ndarray
    s_rr_fe: np.ndarray
    s_rr_1d: np.ndarray
    s_tt_fe: np.ndarray
    s_tt_1d: np.ndarray


@dataclass(frozen=True, slots=True)
class CaseResult:
    width: float
    height: float
    current_density: float
    source_radius: float
    source_z: float
    source_current: float
    balance_axial_load: bool
    ri: float
    ro: float
    z_min: float
    z_max: float
    nr: int
    nz: int
    ndof: int
    stiffness_nnz: int
    field_r: np.ndarray
    field_z: np.ndarray
    bmag_field: np.ndarray
    bz_field: np.ndarray
    elem_r_centers: np.ndarray
    elem_z_centers: np.ndarray
    body_force_r: np.ndarray
    body_force_z: np.ndarray
    outline_r: np.ndarray
    outline_z: np.ndarray
    section_z_values: tuple[float, ...]
    sections: tuple[SectionComparison, ...]
    net_body_force_z: float
    pressure_top: float
    pressure_bottom: float
    net_total_force_z: float
    peak_body_force_density: float


def export_docs_example_figure(fig) -> None:
    DOCS_EXAMPLE_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(DOCS_EXAMPLE_HTML),
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True},
        auto_open=False,
    )


def normalize_float(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return float(np.clip(float(value), lower, upper))


def build_annulus_strip_mesh(
    ri: float,
    ro: float,
    height: float,
    nr: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radii = np.linspace(ri, ro, nr + 1, dtype=np.float64)
    zs = np.linspace(-0.5 * height, 0.5 * height, nz + 1, dtype=np.float64)
    nodes = np.array([[r, z] for z in zs for r in radii], dtype=np.float64)

    def node_id(i: int, j: int) -> int:
        return j * (nr + 1) + i

    elements = []
    for j in range(nz):
        for i in range(nr):
            elements.append(
                [
                    node_id(i, j),
                    node_id(i + 1, j),
                    node_id(i + 1, j + 1),
                    node_id(i, j + 1),
                ]
            )

    return nodes, np.asarray(elements, dtype=np.uint64), radii, zs


def choose_mesh_counts(width: float, height: float) -> tuple[int, int]:
    width = max(width, 1.0e-12)
    height = max(height, 1.0e-12)
    if width >= height:
        nr = MESH_LONG_SIDE_ELEMENTS
        nz = max(6, int(round(MESH_LONG_SIDE_ELEMENTS * height / width)))
    else:
        nz = MESH_LONG_SIDE_ELEMENTS
        nr = max(6, int(round(MESH_LONG_SIDE_ELEMENTS * width / height)))
    return nr, nz


def top_bottom_pressure_faces(nr: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    bottom = np.asarray([[i, 0] for i in range(nr)], dtype=np.uint64)
    top = np.asarray([[(nz - 1) * nr + i, 2] for i in range(nr)], dtype=np.uint64)
    return bottom, top


def section_rows(nz: int) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    seen: set[int] = set()
    for fraction, label, color in zip(SECTION_TARGET_FRACTIONS, SECTION_LABELS, SECTION_COLORS, strict=True):
        row = int(round((nz - 1) * fraction))
        row = max(0, min(nz - 1, row))
        if row in seen:
            continue
        seen.add(row)
        rows.append((label, color, row))
    return rows


def solenoid_outline(ri: float, ro: float, z_min: float, z_max: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([ri, ro, ro, ri, ri], dtype=np.float64),
        np.array([z_min, z_min, z_max, z_max, z_min], dtype=np.float64),
    )


def source_intersects_solenoid(ri: float, ro: float, z_min: float, z_max: float, source_r: float, source_z: float) -> bool:
    return ri <= source_r <= ro and z_min <= source_z <= z_max


def quad4_center_point_strain_stress(
    coords: np.ndarray,
    displacement_local: np.ndarray,
    material: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = 0.25 * np.ones(4, dtype=np.float64)
    grad_ref = 0.25 * np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ],
        dtype=np.float64,
    )
    jac = np.array(
        [
            [coords[:, 0] @ grad_ref[:, 0], coords[:, 0] @ grad_ref[:, 1]],
            [coords[:, 1] @ grad_ref[:, 0], coords[:, 1] @ grad_ref[:, 1]],
        ],
        dtype=np.float64,
    )
    inv_j = np.linalg.inv(jac)
    grad_phys = np.column_stack(
        [
            inv_j[0, 0] * grad_ref[:, 0] + inv_j[1, 0] * grad_ref[:, 1],
            inv_j[0, 1] * grad_ref[:, 0] + inv_j[1, 1] * grad_ref[:, 1],
        ]
    )

    point = n @ coords
    radius = float(point[0])
    if radius <= np.finfo(np.float64).eps:
        raise ValueError(f"section sample radius {radius} is too close to zero")

    b = np.zeros((4, 8), dtype=np.float64)
    for i in range(4):
        col_r = 2 * i
        col_z = col_r + 1
        b[0, col_r] = grad_phys[i, 0]
        b[1, col_z] = grad_phys[i, 1]
        b[2, col_r] = n[i] / radius
        b[3, col_r] = grad_phys[i, 1]
        b[3, col_z] = grad_phys[i, 0]

    u_center = np.sum(n[:, None] * displacement_local, axis=0)
    eps_center = b @ displacement_local.reshape(8)
    sig_center = material @ eps_center
    return point, u_center, eps_center, sig_center


def field_grid(source_r: float, source_z: float, source_current: float, ro: float, height: float) -> tuple[np.ndarray, ...]:
    r_max = max(1.1 * ro, 1.3 * source_r, ro + 0.25)
    z_extent = max(0.8 * height, abs(source_z) + 0.6 * height, 0.25)
    r = np.linspace(0.0, r_max, FIELD_GRID_R, dtype=np.float64)
    z = np.linspace(-z_extent, z_extent, FIELD_GRID_Z, dtype=np.float64)
    rr, zz = np.meshgrid(r, z, indexing="xy")
    br, bz = cfsem.flux_density_circular_filament(
        [source_current],
        [source_r],
        [source_z],
        rr.ravel(),
        zz.ravel(),
        par=True,
    )
    bmag = np.sqrt(br * br + bz * bz).reshape(rr.shape)
    bz_grid = np.asarray(bz, dtype=np.float64).reshape(rr.shape)

    if r.size > 1 and z.size > 1:
        dr = r[1] - r[0]
        dz = z[1] - z[0]
        near_source = (np.abs(rr - source_r) <= 0.55 * dr) & (np.abs(zz - source_z) <= 0.55 * dz)
        bmag = np.where(near_source, np.nan, bmag)
        bz_grid = np.where(near_source, np.nan, bz_grid)

    return r, z, bmag, bz_grid


def max_relative_error_percent(fe_values: np.ndarray, ref_values: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(fe_values))), float(np.max(np.abs(ref_values))), 1.0e-30)
    if scale <= 1.0e-30:
        return 0.0
    return 100.0 * float(np.max(np.abs(fe_values - ref_values))) / scale


def build_section_comparisons(
    nodes: np.ndarray,
    elements: np.ndarray,
    radii: np.ndarray,
    zs: np.ndarray,
    nr: int,
    nz: int,
    displacement: np.ndarray,
    material: np.ndarray,
    source_radius: float,
    source_z: float,
    source_current: float,
    current_density: float,
    body_force_r: np.ndarray,
) -> tuple[SectionComparison, ...]:
    section_list: list[SectionComparison] = []
    row_centers = 0.5 * (zs[:-1] + zs[1:])
    c_struct = solenoid_1d_structural_factor(YOUNGS_MODULUS, POISSON_RATIO)

    for label, color, row in section_rows(nz):
        element_indices = row * nr + np.arange(nr, dtype=np.int64)
        radius = np.zeros(nr, dtype=np.float64)
        u_r_fe = np.zeros(nr, dtype=np.float64)
        e_rr_fe = np.zeros(nr, dtype=np.float64)
        e_tt_fe = np.zeros(nr, dtype=np.float64)
        s_rr_fe = np.zeros(nr, dtype=np.float64)
        s_tt_fe = np.zeros(nr, dtype=np.float64)

        for i_local, element_index in enumerate(element_indices):
            conn = elements[element_index]
            coords = nodes[conn]
            displacement_local = displacement[conn]
            point, u_center, eps_center, sig_center = quad4_center_point_strain_stress(
                coords,
                displacement_local,
                material,
            )
            radius[i_local] = point[0]
            u_r_fe[i_local] = u_center[0]
            e_rr_fe[i_local] = eps_center[0]
            e_tt_fe[i_local] = eps_center[2]
            s_rr_fe[i_local] = sig_center[0]
            s_tt_fe[i_local] = sig_center[2]

        z_value = float(row_centers[row])
        b_z_fe = body_force_r[row] / current_density if current_density > 0.0 else np.zeros_like(radius)

        nudge = 1.0e-6
        rgrid = np.concatenate([[radii[0] - nudge], radius, [radii[-1] + nudge]])
        zgrid = np.full_like(rgrid, z_value)
        _br, bz_grid = cfsem.flux_density_circular_filament(
            [source_current],
            [source_radius],
            [source_z],
            rgrid,
            zgrid,
            par=True,
        )
        rhs = solenoid_1d_structural_rhs(c_struct, np.full_like(rgrid, current_density), bz_grid)
        reference = SolenoidStress1D(
            rgrid=rgrid,
            elasticity_modulus=YOUNGS_MODULUS,
            poisson_ratio=POISSON_RATIO,
            direct_inverse=False,
        )
        u_r_1d_full = np.asarray(reference.displacement_solver(rhs), dtype=np.float64).reshape(-1)
        strain_1d_full = np.asarray(reference.operators.a_eu @ u_r_1d_full, dtype=np.float64).reshape(-1)
        stress_1d_full = np.asarray(reference.operators.a_se @ strain_1d_full, dtype=np.float64).reshape(-1)
        n_1d = rgrid.size

        section_list.append(
            SectionComparison(
                label=label,
                color=color,
                z_value=z_value,
                radius=radius,
                b_z_fe=np.asarray(b_z_fe, dtype=np.float64),
                b_z_1d=np.asarray(bz_grid[1:-1], dtype=np.float64),
                u_r_fe=u_r_fe,
                u_r_1d=u_r_1d_full[1:-1],
                e_rr_fe=e_rr_fe,
                e_rr_1d=strain_1d_full[:n_1d][1:-1],
                e_tt_fe=e_tt_fe,
                e_tt_1d=strain_1d_full[n_1d:][1:-1],
                s_rr_fe=s_rr_fe,
                s_rr_1d=stress_1d_full[:n_1d][1:-1],
                s_tt_fe=s_tt_fe,
                s_tt_1d=stress_1d_full[n_1d:][1:-1],
            )
        )

    return tuple(section_list)


@lru_cache(maxsize=128)
def solve_case(
    width: float,
    height: float,
    current_density_ma: float,
    source_radius: float,
    source_z: float,
    source_current_ma: float,
    balance_axial_load: bool,
) -> CaseResult:
    width = normalize_float(width, WIDTH_RANGE)
    height = normalize_float(height, HEIGHT_RANGE)
    current_density = 1.0e6 * normalize_float(current_density_ma, CURRENT_DENSITY_RANGE_MA)
    source_radius = normalize_float(source_radius, SOURCE_RADIUS_RANGE)
    source_z = normalize_float(source_z, SOURCE_Z_RANGE)
    source_current = 1.0e6 * normalize_float(source_current_ma, SOURCE_CURRENT_RANGE_MA)
    balance_axial_load = bool(balance_axial_load)

    ri = SOLENOID_INNER_RADIUS
    ro = ri + width
    z_min = -0.5 * height
    z_max = 0.5 * height
    if source_intersects_solenoid(ri, ro, z_min, z_max, source_radius, source_z):
        raise ValueError("Move the source loop outside the solenoid conductor cross-section.")

    nr, nz = choose_mesh_counts(width, height)
    nodes, elements, radii, zs = build_annulus_strip_mesh(ri, ro, height, nr, nz)
    material = cfsem_radial_material(YOUNGS_MODULUS, POISSON_RATIO)
    material_table = np.asarray([material], dtype=np.float64)
    material_ids = np.zeros(elements.shape[0], dtype=np.uint64)

    quadrature_data = element_quadrature_axisymmetric(nodes, elements, quadrature=QUADRATURE)
    quadrature_points = quadrature_data.points_rz.reshape(-1, 2)
    br_q, bz_q = cfsem.flux_density_circular_filament(
        [source_current],
        [source_radius],
        [source_z],
        quadrature_points[:, 0],
        quadrature_points[:, 1],
        par=True,
    )
    nelem = elements.shape[0]
    nq = quadrature_data.nq_per_element
    weights = np.asarray(quadrature_data.weights_volume, dtype=np.float64)
    weights_sum = np.sum(weights, axis=1)
    br_mean = np.sum(np.asarray(br_q, dtype=np.float64).reshape(nelem, nq) * weights, axis=1) / weights_sum
    bz_mean = np.sum(np.asarray(bz_q, dtype=np.float64).reshape(nelem, nq) * weights, axis=1) / weights_sum
    body_force = np.column_stack((current_density * bz_mean, -current_density * br_mean))

    measures = element_measures_axisymmetric(nodes, elements, quadrature=QUADRATURE)
    net_body_force_z = float(np.sum(body_force[:, 1] * measures.swept_volumes))
    top_area = np.pi * (ro**2 - ri**2)
    pressure_top = net_body_force_z / (2.0 * top_area) if balance_axial_load else 0.0
    pressure_bottom = -pressure_top

    pressure_faces = None
    pressure_values = None
    if balance_axial_load:
        bottom_faces, top_faces = top_bottom_pressure_faces(nr, nz)
        pressure_faces = np.vstack([bottom_faces, top_faces])
        pressure_values = np.concatenate(
            [
                np.full(bottom_faces.shape[0], pressure_bottom, dtype=np.float64),
                np.full(top_faces.shape[0], pressure_top, dtype=np.float64),
            ]
        )

    assembly = assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=material_ids,
        material_table=material_table,
        body_force=body_force,
        pressure_faces=pressure_faces,
        pressure_values=pressure_values,
        quadrature=QUADRATURE,
    )
    stiffness = assembly.to_csr()
    reduced = apply_dirichlet(stiffness, assembly.rhs, prescribed={1: 0.0})
    displacement = reduced.recover(factorized(reduced.matrix.tocsc())(reduced.rhs)).reshape(nodes.shape[0], 2)

    body_force_r = body_force[:, 0].reshape(nz, nr)
    body_force_z = body_force[:, 1].reshape(nz, nr)
    elem_r_centers = 0.5 * (radii[:-1] + radii[1:])
    elem_z_centers = 0.5 * (zs[:-1] + zs[1:])
    outline_r, outline_z = solenoid_outline(ri, ro, z_min, z_max)
    sections = build_section_comparisons(
        nodes=nodes,
        elements=elements,
        radii=radii,
        zs=zs,
        nr=nr,
        nz=nz,
        displacement=displacement,
        material=material,
        source_radius=source_radius,
        source_z=source_z,
        source_current=source_current,
        current_density=current_density,
        body_force_r=body_force_r,
    )
    field_r, field_z, bmag_field, bz_field = field_grid(source_radius, source_z, source_current, ro, height)

    return CaseResult(
        width=width,
        height=height,
        current_density=current_density,
        source_radius=source_radius,
        source_z=source_z,
        source_current=source_current,
        balance_axial_load=balance_axial_load,
        ri=ri,
        ro=ro,
        z_min=z_min,
        z_max=z_max,
        nr=nr,
        nz=nz,
        ndof=assembly.ndof,
        stiffness_nnz=stiffness.nnz,
        field_r=field_r,
        field_z=field_z,
        bmag_field=bmag_field,
        bz_field=bz_field,
        elem_r_centers=elem_r_centers,
        elem_z_centers=elem_z_centers,
        body_force_r=body_force_r,
        body_force_z=body_force_z,
        outline_r=outline_r,
        outline_z=outline_z,
        section_z_values=tuple(section.z_value for section in sections),
        sections=sections,
        net_body_force_z=net_body_force_z,
        pressure_top=pressure_top,
        pressure_bottom=pressure_bottom,
        net_total_force_z=float(np.sum(assembly.rhs[1::2])),
        peak_body_force_density=float(np.max(np.sqrt(body_force[:, 0] ** 2 + body_force[:, 1] ** 2))),
    )


def message_figure(title: str, message: str):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 18},
        align="center",
    )
    fig.update_layout(
        title=title,
        height=520,
        margin={"l": 40, "r": 20, "t": 60, "b": 30},
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def build_overview_figure(case: CaseResult):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    bmag_log = np.maximum(np.log10(np.asarray(case.bmag_field, dtype=np.float64) + 1.0e-30), LOG10_FLOOR)
    bmag_finite = bmag_log[np.isfinite(bmag_log)]
    bmag_range = (
        float(np.nanmin(bmag_finite)) if bmag_finite.size else LOG10_FLOOR,
        float(np.nanpercentile(bmag_finite, 99.0)) if bmag_finite.size else 0.0,
    )

    bz_clip = float(np.nanpercentile(np.abs(case.bz_field), 99.0)) if np.isfinite(case.bz_field).any() else 1.0
    force_r_clip = float(np.nanpercentile(np.abs(case.body_force_r), 99.0))
    force_z_clip = float(np.nanpercentile(np.abs(case.body_force_z), 99.0))
    force_r_clip = force_r_clip if force_r_clip > 0.0 else 1.0
    force_z_clip = force_z_clip if force_z_clip > 0.0 else 1.0

    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.12,
        vertical_spacing=0.16,
        subplot_titles=[
            "Loop-source |B| [T] (log10)",
            "Loop-source B_z [T]",
            "Element body force f_r = J_theta B_z [N/m^3]",
            "Element body force f_z = -J_theta B_r [N/m^3]",
        ],
    )

    fig.add_trace(
        go.Heatmap(
            x=case.field_r,
            y=case.field_z,
            z=bmag_log,
            colorscale="Magma",
            zmin=bmag_range[0],
            zmax=bmag_range[1],
            colorbar={"title": "log10(|B|)", "thickness": 14, "x": -0.14, "xanchor": "left"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=case.field_r,
            y=case.field_z,
            z=case.bz_field,
            colorscale="RdBu",
            zmid=0.0,
            zmin=-bz_clip,
            zmax=bz_clip,
            colorbar={"title": "B_z [T]", "thickness": 14},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            x=case.elem_r_centers,
            y=case.elem_z_centers,
            z=case.body_force_r,
            colorscale="RdBu",
            zmid=0.0,
            zmin=-force_r_clip,
            zmax=force_r_clip,
            colorbar={"title": "f_r [N/m^3]", "thickness": 14, "x": -0.14, "xanchor": "left"},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=case.elem_r_centers,
            y=case.elem_z_centers,
            z=case.body_force_z,
            colorscale="RdBu",
            zmid=0.0,
            zmin=-force_z_clip,
            zmax=force_z_clip,
            colorbar={"title": "f_z [N/m^3]", "thickness": 14},
        ),
        row=2,
        col=2,
    )

    for row, col in ((1, 1), (1, 2), (2, 1), (2, 2)):
        fig.add_trace(
            go.Scatter(
                x=case.outline_r,
                y=case.outline_z,
                mode="lines",
                line={"color": "white" if row == 1 else "black", "width": 2},
                name="Solenoid section",
                legendgroup="solenoid",
                showlegend=(row == 1 and col == 1),
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        for section in case.sections:
            fig.add_trace(
                go.Scatter(
                    x=np.array([case.ri, case.ro], dtype=np.float64),
                    y=np.array([section.z_value, section.z_value], dtype=np.float64),
                    mode="lines",
                    line={"color": section.color, "width": 2, "dash": "dot"},
                    name=f"{section.label} section",
                    legendgroup=section.label,
                    showlegend=(row == 1 and col == 1),
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
        fig.add_trace(
            go.Scatter(
                x=[case.source_radius],
                y=[case.source_z],
                mode="markers",
                marker={
                    "size": 11,
                    "color": "cyan" if row == 1 else "black",
                    "line": {"color": "black" if row == 1 else "white", "width": 1},
                    "symbol": "circle",
                },
                name="Loop source",
                legendgroup="source",
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )

    for row, col in ((1, 1), (1, 2), (2, 1), (2, 2)):
        fig.update_xaxes(title_text="r [m]", row=row, col=col)
        fig.update_yaxes(title_text="z [m]", row=row, col=col)

    fig.update_layout(
        height=920,
        title=(
            "Axisymmetric FEM source/load overview | "
            f"ri={case.ri:.3f} m, width={case.width:.3f} m, height={case.height:.3f} m"
        ),
        margin={"l": 60, "r": 20, "t": 110, "b": 50},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.08,
            "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0.8)",
        },
    )
    return fig


def build_profile_figure(case: CaseResult):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3,
        cols=2,
        horizontal_spacing=0.12,
        vertical_spacing=0.14,
        subplot_titles=[
            "Radial displacement u_r [m]",
            "Radial strain e_rr [-]",
            "Hoop strain e_tt [-]",
            "Radial stress s_rr [Pa]",
            "Hoop stress s_tt [Pa]",
            "B_z used for loading [T]",
        ],
    )

    subplot_specs = [
        ("u_r_fe", "u_r_1d", 1, 1),
        ("e_rr_fe", "e_rr_1d", 1, 2),
        ("e_tt_fe", "e_tt_1d", 2, 1),
        ("s_rr_fe", "s_rr_1d", 2, 2),
        ("s_tt_fe", "s_tt_1d", 3, 1),
        ("b_z_fe", "b_z_1d", 3, 2),
    ]

    for i_subplot, (fe_name, ref_name, row, col) in enumerate(subplot_specs):
        for section in case.sections:
            fig.add_trace(
                go.Scatter(
                    x=section.radius,
                    y=getattr(section, fe_name),
                    mode="lines+markers",
                    line={"color": section.color, "width": 2},
                    marker={"size": 5},
                    name=f"{section.label} FEM",
                    legendgroup=f"{section.label}-fem",
                    showlegend=i_subplot == 0,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=section.radius,
                    y=getattr(section, ref_name),
                    mode="lines",
                    line={"color": section.color, "width": 2, "dash": "dash"},
                    name=f"{section.label} 1D",
                    legendgroup=f"{section.label}-1d",
                    showlegend=i_subplot == 0,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="r [m]", row=row, col=col)

    fig.update_layout(
        height=1060,
        title=(
            "Radial section comparison | "
            "FEM uses the full axisymmetric body force; 1D uses local B_z(r, z_section)"
        ),
        margin={"l": 55, "r": 20, "t": 110, "b": 50},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.08,
            "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0.8)",
        },
    )
    return fig


def build_error_figure(case: CaseResult):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3,
        cols=2,
        horizontal_spacing=0.12,
        vertical_spacing=0.14,
        subplot_titles=[
            "Absolute error |u_r,FEM - u_r,1D| [m]",
            "Absolute error |e_rr,FEM - e_rr,1D| [-]",
            "Absolute error |e_tt,FEM - e_tt,1D| [-]",
            "Absolute error |s_rr,FEM - s_rr,1D| [Pa]",
            "Absolute error |s_tt,FEM - s_tt,1D| [Pa]",
            "Max relative error by section [%]",
        ],
    )

    error_specs = [
        ("u_r_fe", "u_r_1d", 1, 1),
        ("e_rr_fe", "e_rr_1d", 1, 2),
        ("e_tt_fe", "e_tt_1d", 2, 1),
        ("s_rr_fe", "s_rr_1d", 2, 2),
        ("s_tt_fe", "s_tt_1d", 3, 1),
    ]

    for i_subplot, (fe_name, ref_name, row, col) in enumerate(error_specs):
        for section in case.sections:
            fig.add_trace(
                go.Scatter(
                    x=section.radius,
                    y=np.abs(getattr(section, fe_name) - getattr(section, ref_name)),
                    mode="lines",
                    line={"color": section.color, "width": 2},
                    name=section.label,
                    legendgroup=section.label,
                    showlegend=i_subplot == 0,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="r [m]", row=row, col=col)

    quantity_labels = ["u_r", "e_rr", "e_tt", "s_rr", "s_tt"]
    for section in case.sections:
        max_rel = [
            max_relative_error_percent(section.u_r_fe, section.u_r_1d),
            max_relative_error_percent(section.e_rr_fe, section.e_rr_1d),
            max_relative_error_percent(section.e_tt_fe, section.e_tt_1d),
            max_relative_error_percent(section.s_rr_fe, section.s_rr_1d),
            max_relative_error_percent(section.s_tt_fe, section.s_tt_1d),
        ]
        fig.add_trace(
            go.Bar(
                x=quantity_labels,
                y=max_rel,
                marker={"color": section.color},
                name=section.label,
                legendgroup=section.label,
                showlegend=False,
            ),
            row=3,
            col=2,
        )

    fig.update_xaxes(title_text="Quantity", row=3, col=2)
    fig.update_yaxes(title_text="Max relative error [%]", row=3, col=2)
    fig.update_layout(
        height=1060,
        title="Radial section errors against the 1D finite-difference reference",
        margin={"l": 55, "r": 20, "t": 110, "b": 50},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.08,
            "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0.8)",
        },
        barmode="group",
    )
    return fig


def build_summary(case: CaseResult) -> str:
    section_text = ", ".join(f"{section.label}: z={section.z_value:.3f} m" for section in case.sections)
    return (
        f"Mesh {case.nr}x{case.nz} ({case.ndof} dof, K nnz={case.stiffness_nnz}) | "
        f"J_theta={case.current_density:.3e} A/m^2 | "
        f"source I={case.source_current:.3e} A-turn at (r={case.source_radius:.3f} m, z={case.source_z:.3f} m) | "
        f"net axial body force={case.net_body_force_z:.3e} N | "
        f"top/bottom pressure values=({case.pressure_top:.3e}, {case.pressure_bottom:.3e}) Pa | "
        f"net axial load after balance={case.net_total_force_z:.3e} N | "
        f"peak |JxB|={case.peak_body_force_density:.3e} N/m^3 | "
        f"sections: {section_text}"
    )


def create_app():
    from dash import Dash, Input, Output, dcc, html

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H3("CFSEM Axisymmetric FEM Solenoid Stress Explorer"),
            html.P(
                "The solenoid inner radius is fixed at 0.5 m. "
                "Move the source loop outside the conductor cross-section to avoid singular loading."
            ),
            html.P(
                "The 1D reference uses the local B_z(r, z_section) on each radial section and ignores "
                "axial/shear coupling."
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.P("Solenoid width [m]", style={"marginTop": "0.25rem", "marginBottom": "0.25rem"}),
                            dcc.Slider(
                                id="solenoid-width",
                                min=WIDTH_RANGE[0],
                                max=WIDTH_RANGE[1],
                                step=0.005,
                                value=DEFAULT_WIDTH,
                                marks={0.05: "0.05", 0.15: "0.15", 0.25: "0.25", 0.35: "0.35"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.P("Solenoid height [m]", style={"marginTop": "0.25rem", "marginBottom": "0.25rem"}),
                            dcc.Slider(
                                id="solenoid-height",
                                min=HEIGHT_RANGE[0],
                                max=HEIGHT_RANGE[1],
                                step=0.005,
                                value=DEFAULT_HEIGHT,
                                marks={0.05: "0.05", 0.20: "0.20", 0.35: "0.35", 0.50: "0.50"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.P(
                                "Current density J_theta [MA/m^2]",
                                style={"marginTop": "0.25rem", "marginBottom": "0.25rem"},
                            ),
                            dcc.Slider(
                                id="current-density",
                                min=CURRENT_DENSITY_RANGE_MA[0],
                                max=CURRENT_DENSITY_RANGE_MA[1],
                                step=5.0,
                                value=DEFAULT_CURRENT_DENSITY_MA,
                                marks={0: "0", 50: "50", 100: "100", 150: "150", 200: "200"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.P(
                                "Loop source radius [m]",
                                style={"marginTop": "0.25rem", "marginBottom": "0.25rem"},
                            ),
                            dcc.Slider(
                                id="source-radius",
                                min=SOURCE_RADIUS_RANGE[0],
                                max=SOURCE_RADIUS_RANGE[1],
                                step=0.01,
                                value=DEFAULT_SOURCE_RADIUS,
                                marks={0.05: "0.05", 0.30: "0.30", 0.60: "0.60", 0.90: "0.90", 1.20: "1.20"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.P("Loop source z [m]", style={"marginTop": "0.25rem", "marginBottom": "0.25rem"}),
                            dcc.Slider(
                                id="source-z",
                                min=SOURCE_Z_RANGE[0],
                                max=SOURCE_Z_RANGE[1],
                                step=0.01,
                                value=DEFAULT_SOURCE_Z,
                                marks={-0.6: "-0.6", -0.3: "-0.3", 0.0: "0.0", 0.3: "0.3", 0.6: "0.6"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.P(
                                "Loop current [MA-turn]",
                                style={"marginTop": "0.25rem", "marginBottom": "0.25rem"},
                            ),
                            dcc.Slider(
                                id="source-current",
                                min=SOURCE_CURRENT_RANGE_MA[0],
                                max=SOURCE_CURRENT_RANGE_MA[1],
                                step=0.05,
                                value=DEFAULT_SOURCE_CURRENT_MA,
                                marks={0: "0", 1: "1", 2: "2", 3: "3", 5: "5"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, minmax(320px, 1fr))",
                    "columnGap": "1rem",
                    "rowGap": "1rem",
                    "paddingBottom": "0.5rem",
                },
            ),
            html.Div(
                [
                    dcc.Checklist(
                        id="balance-axial-load",
                        options=[
                            {
                                "label": "Balance net axial body force with equal-and-opposite top/bottom pressure values",
                                "value": "balance",
                            }
                        ],
                        value=["balance"],
                    )
                ],
                style={"marginTop": "0.5rem", "marginBottom": "0.75rem"},
            ),
            html.Div(
                id="case-summary",
                style={"marginBottom": "0.75rem", "fontFamily": "monospace", "fontSize": "0.92rem"},
            ),
            dcc.Tabs(
                children=[
                    dcc.Tab(
                        label="Overview",
                        children=[
                            html.Div(
                                dcc.Loading(type="circle", children=dcc.Graph(id="overview-figure")),
                                style={"marginTop": "0.5rem"},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Profiles",
                        children=[
                            html.Div(
                                dcc.Loading(type="circle", children=dcc.Graph(id="profile-figure")),
                                style={"marginTop": "0.5rem"},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Errors",
                        children=[
                            html.Div(
                                dcc.Loading(type="circle", children=dcc.Graph(id="error-figure")),
                                style={"marginTop": "0.5rem"},
                            )
                        ],
                    ),
                ]
            ),
        ],
        style={"maxWidth": "1280px", "margin": "0 auto", "padding": "1rem"},
    )

    @app.callback(
        Output("case-summary", "children"),
        Output("overview-figure", "figure"),
        Output("profile-figure", "figure"),
        Output("error-figure", "figure"),
        Input("solenoid-width", "value"),
        Input("solenoid-height", "value"),
        Input("current-density", "value"),
        Input("source-radius", "value"),
        Input("source-z", "value"),
        Input("source-current", "value"),
        Input("balance-axial-load", "value"),
    )
    def update_figures(
        width: float,
        height: float,
        current_density: float,
        source_radius: float,
        source_z: float,
        source_current: float,
        balance_axial_load: list[str],
    ):
        try:
            case = solve_case(
                width,
                height,
                current_density,
                source_radius,
                source_z,
                source_current,
                "balance" in balance_axial_load,
            )
        except ValueError as exc:
            message = str(exc)
            fig = message_figure("Invalid source placement", message)
            return message, fig, fig, fig

        return (
            build_summary(case),
            build_overview_figure(case),
            build_profile_figure(case),
            build_error_figure(case),
        )

    return app


def main() -> None:
    if not TESTING:
        create_app().run(debug=True)
        return

    create_app()
    case = solve_case(
        DEFAULT_WIDTH,
        DEFAULT_HEIGHT,
        DEFAULT_CURRENT_DENSITY_MA,
        DEFAULT_SOURCE_RADIUS,
        DEFAULT_SOURCE_Z,
        DEFAULT_SOURCE_CURRENT_MA,
        True,
    )
    overview = build_overview_figure(case)
    export_docs_example_figure(overview)
    build_profile_figure(case)
    build_error_figure(case)


if __name__ == "__main__":
    main()
