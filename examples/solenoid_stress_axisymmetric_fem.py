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
SECTION_WIDTHS = (4.5, 3.25, 2.0)
SECTION_MARKERS = ("circle", "square", "diamond")
SECTION_DASHES = ("solid", "dashdot", "dot")

MESH_LONG_SIDE_ELEMENTS = 14 if TESTING else 28
MESH_MIN_SHORT_SIDE_ELEMENTS = 6
FIELD_GRID_LONG_SIDE_POINTS = 141 if TESTING else 281
FIELD_GRID_MIN_SHORT_SIDE_POINTS = 41 if TESTING else 81
FD_REFERENCE_SPACING = 1.0e-3  # [m]
GRID_NUDGE = 1.0e-6  # [m]
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
    s_zz_fe: np.ndarray
    s_zz_1d: np.ndarray
    s_tt_fe: np.ndarray
    s_tt_1d: np.ndarray
    s_vm_fe: np.ndarray
    s_vm_1d: np.ndarray


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
    vm_stress_fem: np.ndarray
    vm_stress_1d: np.ndarray


@dataclass(frozen=True, slots=True)
class Reference1DProfile:
    b_z: np.ndarray
    u_r: np.ndarray
    e_rr: np.ndarray
    e_tt: np.ndarray
    s_rr: np.ndarray
    s_zz: np.ndarray
    s_tt: np.ndarray
    s_vm: np.ndarray


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


def section_style(label: str) -> dict[str, object]:
    for candidate, width, marker, dash in zip(
        SECTION_LABELS,
        SECTION_WIDTHS,
        SECTION_MARKERS,
        SECTION_DASHES,
        strict=True,
    ):
        if label == candidate:
            return {
                "width": width,
                "marker_symbol": marker,
                "dash": dash,
            }
    return {
        "width": 2.5,
        "marker_symbol": "circle",
        "dash": "solid",
    }


def choose_mesh_counts(width: float, height: float) -> tuple[int, int]:
    width = max(width, 1.0e-12)
    height = max(height, 1.0e-12)
    long_side = max(width, height)
    short_side = min(width, height)
    target_size = min(
        long_side / MESH_LONG_SIDE_ELEMENTS,
        short_side / MESH_MIN_SHORT_SIDE_ELEMENTS,
    )
    nr = max(1, int(np.ceil(width / target_size)))
    nz = max(1, int(np.ceil(height / target_size)))
    return nr, nz


def choose_field_grid_counts(r_extent: float, z_extent: float) -> tuple[int, int]:
    r_extent = max(r_extent, 1.0e-12)
    z_extent = max(z_extent, 1.0e-12)
    long_side = max(r_extent, z_extent)
    short_side = min(r_extent, z_extent)
    target_spacing = min(
        long_side / max(FIELD_GRID_LONG_SIDE_POINTS - 1, 1),
        short_side / max(FIELD_GRID_MIN_SHORT_SIDE_POINTS - 1, 1),
    )
    nr = max(2, int(np.ceil(r_extent / target_spacing)) + 1)
    nz = max(2, int(np.ceil(z_extent / target_spacing)) + 1)
    return nr, nz


def build_uniform_interval_grid(start: float, stop: float, spacing: float) -> np.ndarray:
    start = float(start)
    stop = float(stop)
    spacing = max(float(spacing), 1.0e-12)
    n = max(2, int(np.ceil((stop - start) / spacing)) + 1)
    return np.linspace(start, stop, n, dtype=np.float64)


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


def source_intersects_solenoid(
    ri: float,
    ro: float,
    z_min: float,
    z_max: float,
    source_r: float,
    source_z: float,
) -> bool:
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


def field_grid(
    source_r: float,
    source_z: float,
    source_current: float,
    ro: float,
    height: float,
) -> tuple[np.ndarray, ...]:
    r_max = max(1.1 * ro, 1.3 * source_r, ro + 0.25)
    z_extent = max(0.8 * height, abs(source_z) + 0.6 * height, 0.25)
    nr_field, nz_field = choose_field_grid_counts(r_max, 2.0 * z_extent)
    r = np.linspace(0.0, r_max, nr_field, dtype=np.float64)
    z = np.linspace(-z_extent, z_extent, nz_field, dtype=np.float64)
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


def normalized_error_percent(fe_values: np.ndarray, ref_values: np.ndarray) -> np.ndarray:
    x_max = max(float(np.max(np.abs(fe_values))), float(np.max(np.abs(ref_values))), 1.0e-30)
    return 100.0 * (fe_values - ref_values) / x_max


def peak_magnitude_error_percent(fe_values: np.ndarray, ref_values: np.ndarray) -> float:
    fe_peak = float(np.max(np.abs(fe_values)))
    ref_peak = float(np.max(np.abs(ref_values)))
    x_peak = max(fe_peak, ref_peak, 1.0e-30)
    return 100.0 * (fe_peak - ref_peak) / x_peak


def sample_loop_bz(
    sample_r: np.ndarray,
    z_value: float,
    source_radius: float,
    source_z: float,
    source_current: float,
) -> np.ndarray:
    _br, bz = cfsem.flux_density_circular_filament(
        [source_current],
        [source_radius],
        [source_z],
        sample_r,
        np.full_like(sample_r, z_value),
        par=True,
    )
    return np.asarray(bz, dtype=np.float64)


def solve_reference_profile_1d(
    sample_r: np.ndarray,
    sample_bz: np.ndarray,
    current_density: float,
    *,
    pi: float = 0.0,
    po: float = 0.0,
) -> Reference1DProfile:
    fine_r = build_uniform_interval_grid(sample_r[0], sample_r[-1], FD_REFERENCE_SPACING)
    fine_bz = np.interp(fine_r, sample_r, sample_bz)
    fine_j = np.full_like(fine_r, current_density)

    rgrid = np.concatenate([[fine_r[0] - GRID_NUDGE], fine_r, [fine_r[-1] + GRID_NUDGE]])
    jgrid = np.concatenate([[fine_j[0]], fine_j, [fine_j[-1]]])
    bz_grid = np.concatenate([[fine_bz[0]], fine_bz, [fine_bz[-1]]])

    c_struct = solenoid_1d_structural_factor(YOUNGS_MODULUS, POISSON_RATIO)
    rhs = solenoid_1d_structural_rhs(c_struct, jgrid, bz_grid, pi=pi, po=po)
    reference = SolenoidStress1D(
        rgrid=rgrid,
        elasticity_modulus=YOUNGS_MODULUS,
        poisson_ratio=POISSON_RATIO,
        direct_inverse=False,
    )
    u_r_full = np.asarray(reference.displacement_solver(rhs), dtype=np.float64).reshape(-1)
    strain_full = np.asarray(reference.operators.a_eu @ u_r_full, dtype=np.float64).reshape(-1)
    stress_full = np.asarray(reference.operators.a_se @ strain_full, dtype=np.float64).reshape(-1)
    n_fine = rgrid.size

    u_r = np.interp(sample_r, fine_r, u_r_full[1:-1])
    e_rr = np.interp(sample_r, fine_r, strain_full[:n_fine][1:-1])
    e_tt = np.interp(sample_r, fine_r, strain_full[n_fine:][1:-1])
    s_rr = np.interp(sample_r, fine_r, stress_full[:n_fine][1:-1])
    s_tt = np.interp(sample_r, fine_r, stress_full[n_fine:][1:-1])
    s_zz = np.zeros_like(sample_r)
    s_vm = von_mises_stress(s_rr, s_zz, s_tt)

    return Reference1DProfile(
        b_z=np.interp(sample_r, fine_r, fine_bz),
        u_r=u_r,
        e_rr=e_rr,
        e_tt=e_tt,
        s_rr=s_rr,
        s_zz=s_zz,
        s_tt=s_tt,
        s_vm=s_vm,
    )


def von_mises_stress(
    s_rr: np.ndarray | float,
    s_zz: np.ndarray | float,
    s_tt: np.ndarray | float,
    tau_rz: np.ndarray | float = 0.0,
) -> np.ndarray:
    return np.sqrt(
        0.5 * ((s_rr - s_zz) ** 2 + (s_zz - s_tt) ** 2 + (s_tt - s_rr) ** 2)
        + 3.0 * tau_rz**2
    )


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

    for label, color, row in section_rows(nz):
        element_indices = row * nr + np.arange(nr, dtype=np.int64)
        radius = np.zeros(nr, dtype=np.float64)
        u_r_fe = np.zeros(nr, dtype=np.float64)
        e_rr_fe = np.zeros(nr, dtype=np.float64)
        e_tt_fe = np.zeros(nr, dtype=np.float64)
        s_rr_fe = np.zeros(nr, dtype=np.float64)
        s_zz_fe = np.zeros(nr, dtype=np.float64)
        s_tt_fe = np.zeros(nr, dtype=np.float64)
        s_vm_fe = np.zeros(nr, dtype=np.float64)

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
            s_zz_fe[i_local] = sig_center[1]
            s_tt_fe[i_local] = sig_center[2]
            s_vm_fe[i_local] = von_mises_stress(sig_center[0], sig_center[1], sig_center[2], sig_center[3])

        z_value = float(row_centers[row])
        b_z_fe = body_force_r[row] / current_density if current_density > 0.0 else np.zeros_like(radius)
        b_z_section = sample_loop_bz(radius, z_value, source_radius, source_z, source_current)
        reference = solve_reference_profile_1d(radius, b_z_section, current_density)

        section_list.append(
            SectionComparison(
                label=label,
                color=color,
                z_value=z_value,
                radius=radius,
                b_z_fe=np.asarray(b_z_fe, dtype=np.float64),
                b_z_1d=reference.b_z,
                u_r_fe=u_r_fe,
                u_r_1d=reference.u_r,
                e_rr_fe=e_rr_fe,
                e_rr_1d=reference.e_rr,
                e_tt_fe=e_tt_fe,
                e_tt_1d=reference.e_tt,
                s_rr_fe=s_rr_fe,
                s_rr_1d=reference.s_rr,
                s_zz_fe=s_zz_fe,
                s_zz_1d=reference.s_zz,
                s_tt_fe=s_tt_fe,
                s_tt_1d=reference.s_tt,
                s_vm_fe=s_vm_fe,
                s_vm_1d=reference.s_vm,
            )
        )

    return tuple(section_list)


def build_vm_stress_grids(
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
) -> tuple[np.ndarray, np.ndarray]:
    vm_fem = np.zeros((nz, nr), dtype=np.float64)
    row_centers = 0.5 * (zs[:-1] + zs[1:])
    elem_r_centers = 0.5 * (radii[:-1] + radii[1:])

    for row in range(nz):
        for col in range(nr):
            element_index = row * nr + col
            conn = elements[element_index]
            coords = nodes[conn]
            displacement_local = displacement[conn]
            _point, _u_center, _eps_center, sig_center = quad4_center_point_strain_stress(
                coords,
                displacement_local,
                material,
            )
            vm_fem[row, col] = von_mises_stress(sig_center[0], sig_center[1], sig_center[2], sig_center[3])

    vm_1d = np.zeros((nz, nr), dtype=np.float64)
    for row, z_value in enumerate(row_centers):
        b_z_row = sample_loop_bz(elem_r_centers, z_value, source_radius, source_z, source_current)
        reference = solve_reference_profile_1d(elem_r_centers, b_z_row, current_density)
        vm_1d[row, :] = reference.s_vm

    return vm_fem, vm_1d


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
    vm_stress_fem, vm_stress_1d = build_vm_stress_grids(
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
        vm_stress_fem=vm_stress_fem,
        vm_stress_1d=vm_stress_1d,
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


def build_heatmap_figure(
    case: CaseResult,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    title: str,
    colorbar_title: str,
    colorscale: str,
    zmin: float | None = None,
    zmax: float | None = None,
    zmid: float | None = None,
    outline_color: str = "black",
    source_color: str = "black",
    source_line_color: str = "white",
):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x,
            y=y,
            z=z,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            zmid=zmid,
            colorbar={"title": colorbar_title, "thickness": 14},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=case.outline_r,
            y=case.outline_z,
            mode="lines",
            line={"color": outline_color, "width": 2},
            name="Solenoid section",
            hoverinfo="skip",
        )
    )
    for section in case.sections:
        fig.add_trace(
            go.Scatter(
                x=np.array([case.ri, case.ro], dtype=np.float64),
                y=np.array([section.z_value, section.z_value], dtype=np.float64),
                mode="lines",
                line={"color": section.color, "width": 2, "dash": "dot"},
                name=f"{section.label} section",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[case.source_radius],
            y=[case.source_z],
            mode="markers",
            marker={
                "size": 11,
                "color": source_color,
                "line": {"color": source_line_color, "width": 1},
                "symbol": "circle",
            },
            name="Loop source",
        )
    )
    fig.update_xaxes(title_text="r [m]")
    fig.update_yaxes(title_text="z [m]")
    fig.update_layout(
        height=430,
        title={
            "text": title,
            "pad": {"b": 18},
            "y": 0.985,
            "yanchor": "top",
            "x": 0.5,
            "xanchor": "center",
        },
        margin={"l": 60, "r": 20, "t": 110, "b": 50},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.01,
            "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0.8)",
        },
    )
    return fig


def build_profile_figure(case: CaseResult):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=4,
        cols=2,
        horizontal_spacing=0.12,
        vertical_spacing=0.14,
        subplot_titles=[
            "Radial displacement u_r [m]",
            "Radial strain e_rr [-]",
            "Hoop strain e_tt [-]",
            "Radial stress s_rr [Pa]",
            "Axial stress s_zz [Pa]",
            "Hoop stress s_tt [Pa]",
            "Von Mises stress [Pa]",
            "B_z used for loading [T]",
        ],
    )

    subplot_specs = [
        ("u_r_fe", "u_r_1d", 1, 1),
        ("e_rr_fe", "e_rr_1d", 1, 2),
        ("e_tt_fe", "e_tt_1d", 2, 1),
        ("s_rr_fe", "s_rr_1d", 2, 2),
        ("s_zz_fe", "s_zz_1d", 3, 1),
        ("s_tt_fe", "s_tt_1d", 3, 2),
        ("s_vm_fe", "s_vm_1d", 4, 1),
        ("b_z_fe", "b_z_1d", 4, 2),
    ]

    for i_subplot, (fe_name, ref_name, row, col) in enumerate(subplot_specs):
        for section in case.sections:
            style = section_style(section.label)
            fig.add_trace(
                go.Scatter(
                    x=section.radius,
                    y=getattr(section, fe_name),
                    mode="lines+markers",
                    line={"color": section.color, "width": style["width"]},
                    marker={"size": 5.5, "symbol": style["marker_symbol"]},
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
                    line={
                        "color": section.color,
                        "width": style["width"],
                        "dash": "dash" if style["dash"] == "solid" else style["dash"],
                    },
                    name=f"{section.label} 1D",
                    legendgroup=f"{section.label}-1d",
                    showlegend=i_subplot == 0,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="r [m]", row=row, col=col)

    fig.update_layout(
        height=1380,
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
        rows=4,
        cols=2,
        specs=[
            [{}, {}],
            [{}, {}],
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.14,
        subplot_titles=[
            "Normalized error in u_r [%]",
            "Normalized error in e_rr [%]",
            "Normalized error in e_tt [%]",
            "Normalized error in s_rr [%]",
            "Normalized error in VM stress [%]",
            "Normalized error in s_tt [%]",
            "Peak-magnitude error by section [%]",
        ],
    )

    error_specs = [
        ("u_r_fe", "u_r_1d", 1, 1),
        ("e_rr_fe", "e_rr_1d", 1, 2),
        ("e_tt_fe", "e_tt_1d", 2, 1),
        ("s_rr_fe", "s_rr_1d", 2, 2),
        ("s_vm_fe", "s_vm_1d", 3, 1),
        ("s_tt_fe", "s_tt_1d", 3, 2),
    ]

    for i_subplot, (fe_name, ref_name, row, col) in enumerate(error_specs):
        for section in case.sections:
            style = section_style(section.label)
            fig.add_trace(
                go.Scatter(
                    x=section.radius,
                    y=normalized_error_percent(getattr(section, fe_name), getattr(section, ref_name)),
                    mode="lines+markers",
                    line={"color": section.color, "width": style["width"], "dash": style["dash"]},
                    marker={"size": 5.5, "symbol": style["marker_symbol"]},
                    name=section.label,
                    legendgroup=section.label,
                    showlegend=i_subplot == 0,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="r [m]", row=row, col=col)

    quantity_labels = ["u_r", "e_rr", "e_tt", "s_rr", "s_vm", "s_tt"]
    for section in case.sections:
        peak_error = [
            peak_magnitude_error_percent(section.u_r_fe, section.u_r_1d),
            peak_magnitude_error_percent(section.e_rr_fe, section.e_rr_1d),
            peak_magnitude_error_percent(section.e_tt_fe, section.e_tt_1d),
            peak_magnitude_error_percent(section.s_rr_fe, section.s_rr_1d),
            peak_magnitude_error_percent(section.s_vm_fe, section.s_vm_1d),
            peak_magnitude_error_percent(section.s_tt_fe, section.s_tt_1d),
        ]
        fig.add_trace(
            go.Bar(
                x=quantity_labels,
                y=peak_error,
                marker={"color": section.color},
                name=section.label,
                legendgroup=section.label,
                showlegend=False,
            ),
            row=4,
            col=1,
        )

    fig.update_xaxes(title_text="Quantity", row=4, col=1)
    fig.update_yaxes(title_text="Peak-magnitude error [%]", row=4, col=1)
    fig.update_layout(
        height=1320,
        title={
            "text": (
                "Radial section errors against the 1D finite-difference reference<br>"
                "curves: normalized error = 100 * (x_FEM - x_FD) / x_max,<br>"
                "x_max = max(max|x_FEM|, max|x_FD|) over each section<br>"
                "bars: peak-magnitude error = 100 * (x_peak,FEM - x_peak,FD) / "
                "max(x_peak,FEM, x_peak,FD),<br>"
                "x_peak = max|x| over each section"
            ),
            "pad": {"b": 28},
            "y": 0.985,
            "yanchor": "top",
            "x": 0.5,
            "xanchor": "center",
        },
        margin={"l": 55, "r": 20, "t": 230, "b": 50},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.02,
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
        f"source I={case.source_current:.3e} A-turn at "
        f"(r={case.source_radius:.3f} m, z={case.source_z:.3f} m) | "
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
                            html.P(
                                "Solenoid width [m]",
                                style={"marginTop": "0.25rem", "marginBottom": "0.25rem"},
                            ),
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
                            html.P(
                                "Solenoid height [m]",
                                style={"marginTop": "0.25rem", "marginBottom": "0.25rem"},
                            ),
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
                            html.P(
                                "Loop source z [m]",
                                style={"marginTop": "0.25rem", "marginBottom": "0.25rem"},
                            ),
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
                                "label": (
                                    "Balance net axial body force with equal-and-opposite "
                                    "top/bottom pressure values"
                                ),
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
                                [
                                    html.Div(
                                        dcc.Loading(
                                            type="circle",
                                            children=dcc.Graph(id="overview-bmag-figure"),
                                        )
                                    ),
                                    html.Div(
                                        dcc.Loading(
                                            type="circle",
                                            children=dcc.Graph(id="overview-bz-figure"),
                                        )
                                    ),
                                    html.Div(
                                        dcc.Loading(
                                            type="circle",
                                            children=dcc.Graph(id="overview-force-r-figure"),
                                        )
                                    ),
                                    html.Div(
                                        dcc.Loading(
                                            type="circle",
                                            children=dcc.Graph(id="overview-force-z-figure"),
                                        )
                                    ),
                                    html.Div(
                                        dcc.Loading(
                                            type="circle",
                                            children=dcc.Graph(id="overview-vm-fem-figure"),
                                        )
                                    ),
                                    html.Div(
                                        dcc.Loading(
                                            type="circle",
                                            children=dcc.Graph(id="overview-vm-1d-figure"),
                                        )
                                    ),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(2, minmax(320px, 1fr))",
                                    "columnGap": "1rem",
                                    "rowGap": "1rem",
                                    "marginTop": "0.5rem",
                                },
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
        Output("overview-bmag-figure", "figure"),
        Output("overview-bz-figure", "figure"),
        Output("overview-force-r-figure", "figure"),
        Output("overview-force-z-figure", "figure"),
        Output("overview-vm-fem-figure", "figure"),
        Output("overview-vm-1d-figure", "figure"),
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
            return message, fig, fig, fig, fig, fig, fig, fig, fig

        bmag_log = np.maximum(np.log10(np.asarray(case.bmag_field, dtype=np.float64) + 1.0e-30), LOG10_FLOOR)
        bmag_finite = bmag_log[np.isfinite(bmag_log)]
        bmag_zmin = float(np.nanmin(bmag_finite)) if bmag_finite.size else LOG10_FLOOR
        bmag_zmax = float(np.nanpercentile(bmag_finite, 99.0)) if bmag_finite.size else 0.0
        bz_clip = (
            float(np.nanpercentile(np.abs(case.bz_field), 99.0))
            if np.isfinite(case.bz_field).any()
            else 1.0
        )
        force_r_clip = float(np.nanpercentile(np.abs(case.body_force_r), 99.0))
        force_z_clip = float(np.nanpercentile(np.abs(case.body_force_z), 99.0))
        vm_fem_clip = float(np.nanpercentile(case.vm_stress_fem, 99.0))
        vm_1d_clip = float(np.nanpercentile(case.vm_stress_1d, 99.0))
        force_r_clip = force_r_clip if force_r_clip > 0.0 else 1.0
        force_z_clip = force_z_clip if force_z_clip > 0.0 else 1.0
        vm_fem_clip = vm_fem_clip if vm_fem_clip > 0.0 else 1.0
        vm_1d_clip = vm_1d_clip if vm_1d_clip > 0.0 else 1.0

        return (
            build_summary(case),
            build_heatmap_figure(
                case,
                case.field_r,
                case.field_z,
                bmag_log,
                title="Loop-source |B| [T] (log10)",
                colorbar_title="log10(|B|)",
                colorscale="Magma",
                zmin=bmag_zmin,
                zmax=bmag_zmax,
                outline_color="white",
                source_color="cyan",
                source_line_color="black",
            ),
            build_heatmap_figure(
                case,
                case.field_r,
                case.field_z,
                case.bz_field,
                title="Loop-source B_z [T]",
                colorbar_title="B_z [T]",
                colorscale="RdBu",
                zmin=-bz_clip,
                zmax=bz_clip,
                zmid=0.0,
                outline_color="white",
                source_color="cyan",
                source_line_color="black",
            ),
            build_heatmap_figure(
                case,
                case.elem_r_centers,
                case.elem_z_centers,
                case.body_force_r,
                title="Element body force f_r = J_theta B_z [N/m^3]",
                colorbar_title="f_r [N/m^3]",
                colorscale="RdBu",
                zmin=-force_r_clip,
                zmax=force_r_clip,
                zmid=0.0,
                outline_color="black",
                source_color="black",
                source_line_color="white",
            ),
            build_heatmap_figure(
                case,
                case.elem_r_centers,
                case.elem_z_centers,
                case.body_force_z,
                title="Element body force f_z = -J_theta B_r [N/m^3]",
                colorbar_title="f_z [N/m^3]",
                colorscale="RdBu",
                zmin=-force_z_clip,
                zmax=force_z_clip,
                zmid=0.0,
                outline_color="black",
                source_color="black",
                source_line_color="white",
            ),
            build_heatmap_figure(
                case,
                case.elem_r_centers,
                case.elem_z_centers,
                case.vm_stress_fem,
                title="Von Mises stress from FEM [Pa]",
                colorbar_title="VM FEM [Pa]",
                colorscale="Cividis",
                zmin=0.0,
                zmax=vm_fem_clip,
                outline_color="black",
                source_color="black",
                source_line_color="white",
            ),
            build_heatmap_figure(
                case,
                case.elem_r_centers,
                case.elem_z_centers,
                case.vm_stress_1d,
                title="Von Mises stress from row-wise 1D reference [Pa]",
                colorbar_title="VM 1D [Pa]",
                colorscale="Cividis",
                zmin=0.0,
                zmax=vm_1d_clip,
                outline_color="black",
                source_color="black",
                source_line_color="white",
            ),
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
    overview = build_heatmap_figure(
        case,
        case.field_r,
        case.field_z,
        np.maximum(np.log10(np.asarray(case.bmag_field, dtype=np.float64) + 1.0e-30), LOG10_FLOOR),
        title="Loop-source |B| [T] (log10)",
        colorbar_title="log10(|B|)",
        colorscale="Magma",
        outline_color="white",
        source_color="cyan",
        source_line_color="black",
    )
    export_docs_example_figure(overview)
    build_profile_figure(case)
    build_error_figure(case)


if __name__ == "__main__":
    main()
