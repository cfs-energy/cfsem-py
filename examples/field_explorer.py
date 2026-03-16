from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

import cfsem

GRID_SIZE = 201 if os.getenv("CFSEM_TESTING") else 1001
EQUIV_GRID_SIZE = 21 if os.getenv("CFSEM_TESTING") else 1001
SECTION_COMPARE_GRID_SIZE = 17 if os.getenv("CFSEM_TESTING") else 161
DEFAULT_WIRE_RADIUS = 0.02
PATH_RADIUS = 0.7
DOMAIN = 1.0
CURRENT = 1.0
LOG10_FLOOR = -16.0
DEFAULT_SECTION_GRID_N = 50
MAX_SECTION_GRID_N = 400
DEFAULT_POINT_SEGMENT_SUBDIVISIONS = 24
MAX_POINT_SEGMENT_SUBDIVISIONS = 400
DEFAULT_BOUNDARY_ELEMENT_LENGTH_NODES = 4
MAX_BOUNDARY_ELEMENT_LENGTH_NODES = 20
BOUNDARY_COMPARE_GRID_SIZE = 17 if os.getenv("CFSEM_TESTING") else 101
DOCS_FIELD_EXPLORER_SVG = (
    Path(__file__).resolve().parents[1] / "docs/python/example_outputs/field_explorer.svg"
)
DOCS_FIELD_EXPLORER_HTML = (
    Path(__file__).resolve().parents[1] / "docs/python/example_outputs/field_explorer.html"
)


def describe_geometry(n_sides: int) -> str:
    if n_sides == 1:
        return "Straight line"
    if n_sides == 2:
        return "Two-segment path"
    return f"{n_sides}-sided polygon"


def build_plot_context_summary(
    n_sides: int,
    wire_radius: float,
    rotation_deg: float,
    n_subdivisions: int,
    *,
    point_segment_subdivisions: int | None = None,
    section_grid_n: int | None = None,
    section_filaments_per_segment: int | None = None,
    distributed_radius_mode: str | None = None,
    boundary_strip_length_nodes: int | None = None,
) -> str:
    parts = [
        describe_geometry(n_sides),
        f"wire radius: {wire_radius:.3f} m",
        f"rotation: {rotation_deg:.0f} deg",
        f"sub-divisions: {n_subdivisions}",
    ]
    if point_segment_subdivisions is not None:
        parts.append(f"point-segment sub-divisions: {point_segment_subdivisions}")
    if section_grid_n is not None:
        grid_summary = f"section grid: {section_grid_n}x{section_grid_n}"
        if section_filaments_per_segment is not None:
            grid_summary += f" -> {section_filaments_per_segment} fil/segment"
        parts.append(grid_summary)
    if distributed_radius_mode is not None:
        parts.append(f"distributed radius mode: {distributed_radius_mode}")
    if boundary_strip_length_nodes is not None:
        parts.append(f"strip nodes/segment: {boundary_strip_length_nodes}")
    return " | ".join(parts)


def export_docs_example_figure(fig) -> None:
    DOCS_FIELD_EXPLORER_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(DOCS_FIELD_EXPLORER_HTML),
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True},
        auto_open=False,
    )
    DOCS_FIELD_EXPLORER_SVG.unlink(missing_ok=True)


def normalize_section_grid_n(section_grid_n: int) -> int:
    n = int(section_grid_n)
    n = max(2, min(MAX_SECTION_GRID_N, n))
    if n % 2 != 0:
        n += 1
        if n > MAX_SECTION_GRID_N:
            n -= 2
    return n


def normalize_point_segment_subdivisions(n_point_subdivisions: int) -> int:
    return max(1, min(MAX_POINT_SEGMENT_SUBDIVISIONS, int(n_point_subdivisions)))


def normalize_boundary_element_length_nodes(n_strip_nodes: int) -> int:
    return max(2, min(MAX_BOUNDARY_ELEMENT_LENGTH_NODES, int(n_strip_nodes)))


def finite_positive_max(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    vmax = float(np.max(finite))
    return vmax if vmax > 0.0 else None


def build_path_vertices(n_sides: int) -> np.ndarray:
    if n_sides == 1:
        return np.array([[-PATH_RADIUS, 0.0, 0.0], [PATH_RADIUS, 0.0, 0.0]])
    if n_sides == 2:
        return np.array(
            [
                [-0.85 * PATH_RADIUS, 0.0, -0.5 * PATH_RADIUS],
                [0.0, 0.0, 0.85 * PATH_RADIUS],
                [0.85 * PATH_RADIUS, 0.0, -0.5 * PATH_RADIUS],
            ]
        )

    theta = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False) + 0.5 * np.pi
    x = PATH_RADIUS * np.cos(theta)
    z = PATH_RADIUS * np.sin(theta)
    y = np.zeros_like(x)
    return np.column_stack((x, y, z))


def rotate_vertices_y(vertices: np.ndarray, rotation_deg: float) -> np.ndarray:
    theta = np.deg2rad(rotation_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    xr = c * x + s * z
    zr = -s * x + c * z
    return np.column_stack((xr, y, zr))


def build_linear_filaments(
    n_sides: int,
    rotation_deg: float,
    n_subdivisions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray]:
    vertices = rotate_vertices_y(build_path_vertices(n_sides), rotation_deg)
    if n_sides >= 3:
        starts_base = vertices
        ends_base = np.roll(vertices, -1, axis=0)
    else:
        starts_base = vertices[:-1]
        ends_base = vertices[1:]

    n_sub = max(1, int(n_subdivisions))
    if n_sub == 1:
        starts = starts_base
        ends = ends_base
    else:
        dvec = ends_base - starts_base
        dsub = dvec / n_sub
        base = np.repeat(starts_base, n_sub, axis=0)
        frac = np.tile(np.arange(n_sub, dtype=float), starts_base.shape[0])[:, None]
        starts = base + frac * np.repeat(dsub, n_sub, axis=0)
        ends = starts + np.repeat(dsub, n_sub, axis=0)

    dl = ends - starts
    xyzfil = (starts[:, 0], starts[:, 1], starts[:, 2])
    dlxyzfil = (dl[:, 0], dl[:, 1], dl[:, 2])
    ifil = np.full(starts.shape[0], CURRENT)
    return vertices, starts, ends, xyzfil, dlxyzfil, ifil


def discretize_point_segments(
    starts: np.ndarray,
    dl: np.ndarray,
    current: np.ndarray,
    n_lengthwise_discretizations: int,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray]:
    nseg_per_filament = normalize_point_segment_subdivisions(n_lengthwise_discretizations)

    xfil_ps = []
    yfil_ps = []
    zfil_ps = []
    dlx_ps = []
    dly_ps = []
    dlz_ps = []
    ifil_ps = []

    for start, dvec, amp in zip(starts, dl, current, strict=True):
        dseg = dvec / nseg_per_filament
        seg_starts = start + dseg * np.arange(nseg_per_filament)[:, None]
        xfil_ps.append(seg_starts[:, 0])
        yfil_ps.append(seg_starts[:, 1])
        zfil_ps.append(seg_starts[:, 2])
        dlx_ps.append(np.full(nseg_per_filament, dseg[0]))
        dly_ps.append(np.full(nseg_per_filament, dseg[1]))
        dlz_ps.append(np.full(nseg_per_filament, dseg[2]))
        ifil_ps.append(np.full(nseg_per_filament, amp))

    xyzfil_ps = (
        np.concatenate(xfil_ps),
        np.concatenate(yfil_ps),
        np.concatenate(zfil_ps),
    )
    dlxyzfil_ps = (
        np.concatenate(dlx_ps),
        np.concatenate(dly_ps),
        np.concatenate(dlz_ps),
    )
    return xyzfil_ps, dlxyzfil_ps, np.concatenate(ifil_ps)


@lru_cache(maxsize=256)
def rectangular_disk_offsets(n_side: int) -> np.ndarray:
    n_side = max(1, int(n_side))
    cell = 2.0 / n_side
    coords = -1.0 + (np.arange(n_side, dtype=float) + 0.5) * cell
    uu, vv = np.meshgrid(coords, coords, indexing="xy")
    offsets = np.column_stack((uu.ravel(), vv.ravel()))
    mask = np.sum(offsets * offsets, axis=1) <= 1.0
    return offsets[mask]


def distributed_section_summary_text(section_grid_n: int) -> str:
    n_grid = normalize_section_grid_n(section_grid_n)
    n_filaments = rectangular_disk_offsets(n_grid).shape[0]
    return f"Distributed section: {n_grid}x{n_grid} grid -> {n_filaments} filaments per segment"


def orthonormal_section_basis(dvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dnorm = np.linalg.norm(dvec)
    if dnorm <= 0.0:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    d_hat = dvec / dnorm

    # Prefer the projection of +y into the filament cross-section plane so the
    # local section coordinate maps cleanly away from the y=0 evaluation plane.
    y_hat = np.array([0.0, 1.0, 0.0])
    u = y_hat - np.dot(y_hat, d_hat) * d_hat
    unorm = np.linalg.norm(u)
    if unorm <= 1e-15:
        ref = np.array([1.0, 0.0, 0.0]) if abs(d_hat[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        u = np.cross(d_hat, ref)
        unorm = np.linalg.norm(u)
        if unorm <= 1e-15:
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    u = u / unorm
    v = np.cross(d_hat, u)
    return u, v


def distribute_filaments_in_cylinder_section(
    starts: np.ndarray,
    dl: np.ndarray,
    current: np.ndarray,
    wire_radius: float,
    section_grid_n: int,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray, int]:
    n_grid = normalize_section_grid_n(section_grid_n)
    if wire_radius <= 0.0:
        return (
            (starts[:, 0].copy(), starts[:, 1].copy(), starts[:, 2].copy()),
            (dl[:, 0].copy(), dl[:, 1].copy(), dl[:, 2].copy()),
            current.copy(),
            1,
        )

    offsets_uv = rectangular_disk_offsets(n_grid)
    n_offsets = offsets_uv.shape[0]

    starts_ref = []
    dl_ref = []
    current_ref = []

    for start, dvec, amp in zip(starts, dl, current, strict=True):
        u, v = orthonormal_section_basis(dvec)
        offsets_xyz = wire_radius * (
            offsets_uv[:, 0:1] * u.reshape(1, 3) + offsets_uv[:, 1:2] * v.reshape(1, 3)
        )
        starts_i = start.reshape(1, 3) + offsets_xyz

        # Nudge any exact y=0 starts by an extremely small amount within the
        # local section plane to avoid sampling singular/near-singular points.
        on_eval_plane = np.isclose(starts_i[:, 1], 0.0, atol=1e-15)
        if np.any(on_eval_plane):
            use_u_dir = abs(u[1]) >= abs(v[1])
            nudge_dir = u if use_u_dir else v
            nudge_coord = 0 if use_u_dir else 1
            nudge_mag = max(1e-12 * max(1.0, wire_radius), 1e-15)
            if abs(nudge_dir[1]) > 1e-15:
                signs = np.where(offsets_uv[on_eval_plane, nudge_coord] >= 0.0, 1.0, -1.0)
                starts_i[on_eval_plane] += (nudge_mag * signs)[:, None] * nudge_dir.reshape(1, 3)
            else:
                dnorm = np.linalg.norm(dvec)
                if dnorm > 0.0 and abs(dvec[1]) > 1e-15:
                    d_hat = dvec / dnorm
                    signs = np.where(offsets_uv[on_eval_plane, nudge_coord] >= 0.0, 1.0, -1.0)
                    starts_i[on_eval_plane] += (nudge_mag * signs)[:, None] * d_hat.reshape(1, 3)

        starts_ref.append(starts_i)
        dl_ref.append(np.repeat(dvec.reshape(1, 3), n_offsets, axis=0))
        current_ref.append(np.full(n_offsets, amp / n_offsets))

    starts_ref_a = np.concatenate(starts_ref, axis=0)
    dl_ref_a = np.concatenate(dl_ref, axis=0)
    current_ref_a = np.concatenate(current_ref)

    return (
        (starts_ref_a[:, 0], starts_ref_a[:, 1], starts_ref_a[:, 2]),
        (dl_ref_a[:, 0], dl_ref_a[:, 1], dl_ref_a[:, 2]),
        current_ref_a,
        n_offsets,
    )


def build_boundary_element_strips(
    starts: np.ndarray,
    ends: np.ndarray,
    current: np.ndarray,
    wire_radius: float,
    n_length_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    n_length_nodes = normalize_boundary_element_length_nodes(n_length_nodes)
    half_width = max(float(wire_radius), 1e-6)
    y_offset = np.array([0.0, half_width, 0.0])

    nodes: list[np.ndarray] = []
    triangles: list[list[int]] = []
    svals: list[np.ndarray] = []
    upper_paths: list[np.ndarray] = []
    lower_paths: list[np.ndarray] = []
    end_connectors: list[np.ndarray] = []
    node_offset = 0

    for start, end, amp in zip(starts, ends, current, strict=True):
        centerline = start + np.linspace(0.0, 1.0, n_length_nodes)[:, None] * (end - start)
        lower = centerline - y_offset.reshape(1, 3)
        upper = centerline + y_offset.reshape(1, 3)

        nodes.extend([lower, upper])
        svals.extend(
            [
                np.full(n_length_nodes, -amp, dtype=float),
                np.full(n_length_nodes, amp, dtype=float),
            ]
        )

        for i in range(n_length_nodes - 1):
            lower0 = node_offset + i
            lower1 = node_offset + i + 1
            upper0 = node_offset + n_length_nodes + i
            upper1 = node_offset + n_length_nodes + i + 1

            # Winding is chosen so that s = +I on the upper edge and s = -I on the
            # lower edge produces current along the segment direction.
            triangles.append([lower0, upper1, lower1])
            triangles.append([lower0, upper0, upper1])

        upper_paths.append(upper)
        lower_paths.append(lower)
        end_connectors.append(np.vstack((lower[0], upper[0])))
        end_connectors.append(np.vstack((lower[-1], upper[-1])))
        node_offset += 2 * n_length_nodes

    return (
        np.ascontiguousarray(np.vstack(nodes), dtype=np.float64),
        np.ascontiguousarray(np.array(triangles), dtype=np.int64),
        np.ascontiguousarray(np.concatenate(svals), dtype=np.float64),
        upper_paths,
        lower_paths,
        end_connectors,
    )


def segment_lines_xyz(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = starts.shape[0]
    x = np.empty(3 * n, dtype=float)
    y = np.empty(3 * n, dtype=float)
    z = np.empty(3 * n, dtype=float)

    x[0::3] = starts[:, 0]
    x[1::3] = ends[:, 0]
    x[2::3] = np.nan
    y[0::3] = starts[:, 1]
    y[1::3] = ends[:, 1]
    y[2::3] = np.nan
    z[0::3] = starts[:, 2]
    z[1::3] = ends[:, 2]
    z[2::3] = np.nan
    return x, y, z


def polyline_collection_xyz(paths: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = sum(path.shape[0] + 1 for path in paths)
    x = np.empty(n, dtype=float)
    y = np.empty(n, dtype=float)
    z = np.empty(n, dtype=float)

    idx = 0
    for path in paths:
        m = path.shape[0]
        x[idx : idx + m] = path[:, 0]
        y[idx : idx + m] = path[:, 1]
        z[idx : idx + m] = path[:, 2]
        x[idx + m] = np.nan
        y[idx + m] = np.nan
        z[idx + m] = np.nan
        idx += m + 1

    return x, y, z


def triangle_mesh_edge_lines_xyz(
    nodes: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_set: set[tuple[int, int]] = set()
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []

    for tri in triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        for ia, ib in ((i0, i1), (i1, i2), (i2, i0)):
            edge = (ia, ib) if ia < ib else (ib, ia)
            if edge in edge_set:
                continue
            edge_set.add(edge)
            starts.append(nodes[edge[0]])
            ends.append(nodes[edge[1]])

    return segment_lines_xyz(np.vstack(starts), np.vstack(ends))


@lru_cache(maxsize=8)
def compute_field(
    mode: str,
    n_sides: int,
    wire_radius: float,
    rotation_deg: float,
    n_subdivisions: int,
    point_segment_subdivisions: int,
) -> dict[str, np.ndarray | float]:
    vertices, starts, ends, xyzfil, dlxyzfil, ifil = build_linear_filaments(
        n_sides, rotation_deg, n_subdivisions
    )
    dl = ends - starts

    x = np.linspace(-DOMAIN, DOMAIN, GRID_SIZE)
    z = np.linspace(-DOMAIN, DOMAIN, GRID_SIZE)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    yy = np.zeros_like(xx)
    xyzp = (xx.ravel(), yy.ravel(), zz.ravel())

    t0 = time.perf_counter()
    if mode == "b":
        vx, vy, vz = cfsem.flux_density_linear_filament(
            xyzp, xyzfil, dlxyzfil, ifil, wire_radius=wire_radius, par=True
        )
    else:
        vx, vy, vz = cfsem.vector_potential_linear_filament(
            xyzp, xyzfil, dlxyzfil, ifil, wire_radius=wire_radius, par=True
        )
    t_linear = time.perf_counter() - t0

    xyzfil_ps, dlxyzfil_ps, ifil_ps = discretize_point_segments(starts, dl, ifil, point_segment_subdivisions)

    t0 = time.perf_counter()
    if mode == "b":
        vx_ps, vy_ps, vz_ps = cfsem.flux_density_point_segment(
            xyzp, xyzfil_ps, dlxyzfil_ps, ifil_ps, par=True
        )
    else:
        vx_ps, vy_ps, vz_ps = cfsem.vector_potential_point_segment(
            xyzp, xyzfil_ps, dlxyzfil_ps, ifil_ps, par=True
        )
    t_point = time.perf_counter() - t0

    mag_linear = np.sqrt(vx * vx + vy * vy + vz * vz).reshape(xx.shape)
    mag_point = np.sqrt(vx_ps * vx_ps + vy_ps * vy_ps + vz_ps * vz_ps).reshape(xx.shape)

    err = np.abs(mag_linear - mag_point)

    return {
        "x": x,
        "z": z,
        "mag_linear": mag_linear,
        "mag_point": mag_point,
        "err": err,
        "path_x": np.r_[vertices[:, 0], vertices[0, 0]] if n_sides >= 3 else vertices[:, 0],
        "path_z": np.r_[vertices[:, 2], vertices[0, 2]] if n_sides >= 3 else vertices[:, 2],
        "t_linear": t_linear,
        "t_point": t_point,
        "n_linear": float(ifil.size * xyzp[0].size),
        "n_point": float(ifil_ps.size * xyzp[0].size),
    }


@lru_cache(maxsize=8)
def compute_field_equivalence(
    n_sides: int, wire_radius: float, rotation_deg: float, n_subdivisions: int
) -> dict[str, np.ndarray | float]:
    vertices, starts, ends, xyzfil, dlxyzfil, ifil = build_linear_filaments(
        n_sides, rotation_deg, n_subdivisions
    )

    x = np.linspace(-DOMAIN, DOMAIN, EQUIV_GRID_SIZE)
    z = np.linspace(-DOMAIN, DOMAIN, EQUIV_GRID_SIZE)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    yy = np.zeros_like(xx)
    xyzp = (xx.ravel(), yy.ravel(), zz.ravel())
    eps = 1e-7
    inv_2eps = 0.5 / eps

    t0 = time.perf_counter()
    bx, by, bz = cfsem.flux_density_linear_filament(
        xyzp, xyzfil, dlxyzfil, ifil, wire_radius=wire_radius, par=True
    )
    t_b = time.perf_counter() - t0

    def eval_a(
        dx_shift: float, dy_shift: float, dz_shift: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xyzp_shift = (xyzp[0] + dx_shift, xyzp[1] + dy_shift, xyzp[2] + dz_shift)
        return cfsem.vector_potential_linear_filament(
            xyzp_shift, xyzfil, dlxyzfil, ifil, wire_radius=wire_radius, par=True
        )

    t0 = time.perf_counter()
    ax_xm, ay_xm, az_xm = eval_a(-eps, 0.0, 0.0)
    ax_xp, ay_xp, az_xp = eval_a(eps, 0.0, 0.0)
    ax_ym, ay_ym, az_ym = eval_a(0.0, -eps, 0.0)
    ax_yp, ay_yp, az_yp = eval_a(0.0, eps, 0.0)
    ax_zm, ay_zm, az_zm = eval_a(0.0, 0.0, -eps)
    ax_zp, ay_zp, az_zp = eval_a(0.0, 0.0, eps)
    t_curl = time.perf_counter() - t0

    daz_dy = (az_yp - az_ym) * inv_2eps
    day_dz = (ay_zp - ay_zm) * inv_2eps
    daz_dx = (az_xp - az_xm) * inv_2eps
    dax_dz = (ax_zp - ax_zm) * inv_2eps
    day_dx = (ay_xp - ay_xm) * inv_2eps
    dax_dy = (ax_yp - ax_ym) * inv_2eps

    curl_x = daz_dy - day_dz
    curl_y = dax_dz - daz_dx
    curl_z = day_dx - dax_dy

    bmag = np.sqrt(bx * bx + by * by + bz * bz).reshape(xx.shape)
    curl_mag = np.sqrt(curl_x * curl_x + curl_y * curl_y + curl_z * curl_z).reshape(xx.shape)
    err = np.sqrt((bx - curl_x) ** 2 + (by - curl_y) ** 2 + (bz - curl_z) ** 2).reshape(xx.shape)

    npts = xyzp[0].size
    return {
        "x": x,
        "z": z,
        "bmag": bmag,
        "curl_mag": curl_mag,
        "err": err,
        "path_x": np.r_[vertices[:, 0], vertices[0, 0]] if n_sides >= 3 else vertices[:, 0],
        "path_z": np.r_[vertices[:, 2], vertices[0, 2]] if n_sides >= 3 else vertices[:, 2],
        "t_b": t_b,
        "t_curl": t_curl,
        "n_b": float(ifil.size * npts),
        "n_curl": float(6 * ifil.size * npts),
    }


@lru_cache(maxsize=8)
def compute_section_comparison_field(
    mode: str,
    n_sides: int,
    wire_radius: float,
    rotation_deg: float,
    n_subdivisions: int,
    section_grid_n: int,
    distributed_use_area_radius: bool,
) -> dict[str, np.ndarray | float]:
    vertices, starts, ends, xyzfil, dlxyzfil, ifil = build_linear_filaments(
        n_sides, rotation_deg, n_subdivisions
    )
    dl = ends - starts

    x = np.linspace(-DOMAIN, DOMAIN, SECTION_COMPARE_GRID_SIZE)
    z = np.linspace(-DOMAIN, DOMAIN, SECTION_COMPARE_GRID_SIZE)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    yy = np.zeros_like(xx)
    xyzp = (xx.ravel(), yy.ravel(), zz.ravel())

    t0 = time.perf_counter()
    if mode == "b":
        vx, vy, vz = cfsem.flux_density_linear_filament(
            xyzp, xyzfil, dlxyzfil, ifil, wire_radius=wire_radius, par=True
        )
    else:
        vx, vy, vz = cfsem.vector_potential_linear_filament(
            xyzp, xyzfil, dlxyzfil, ifil, wire_radius=wire_radius, par=True
        )
    t_model = time.perf_counter() - t0

    xyzfil_ref, dlxyzfil_ref, ifil_ref, n_offsets = distribute_filaments_in_cylinder_section(
        starts, dl, ifil, wire_radius, section_grid_n
    )
    ref_wire_radius = (
        wire_radius / np.sqrt(n_offsets)
        if distributed_use_area_radius and wire_radius > 0.0 and n_offsets > 0
        else 0.0
    )

    t0 = time.perf_counter()
    if mode == "b":
        vx_ref, vy_ref, vz_ref = cfsem.flux_density_linear_filament(
            xyzp, xyzfil_ref, dlxyzfil_ref, ifil_ref, wire_radius=ref_wire_radius, par=True
        )
    else:
        vx_ref, vy_ref, vz_ref = cfsem.vector_potential_linear_filament(
            xyzp, xyzfil_ref, dlxyzfil_ref, ifil_ref, wire_radius=ref_wire_radius, par=True
        )
    t_ref = time.perf_counter() - t0

    mag_model = np.sqrt(vx * vx + vy * vy + vz * vz).reshape(xx.shape)
    mag_ref = np.sqrt(vx_ref * vx_ref + vy_ref * vy_ref + vz_ref * vz_ref).reshape(xx.shape)
    err = np.abs(mag_model - mag_ref)

    npts = xyzp[0].size
    return {
        "x": x,
        "z": z,
        "mag_model": mag_model,
        "mag_ref": mag_ref,
        "err": err,
        "path_x": np.r_[vertices[:, 0], vertices[0, 0]] if n_sides >= 3 else vertices[:, 0],
        "path_z": np.r_[vertices[:, 2], vertices[0, 2]] if n_sides >= 3 else vertices[:, 2],
        "t_model": t_model,
        "t_ref": t_ref,
        "n_model": float(ifil.size * npts),
        "n_ref": float(ifil_ref.size * npts),
        "n_offsets": float(n_offsets),
        "section_grid_n": float(normalize_section_grid_n(section_grid_n)),
        "ref_wire_radius": float(ref_wire_radius),
    }


def build_figures(
    mode: str,
    n_sides: int,
    wire_radius: float,
    rotation_deg: float,
    n_subdivisions: int,
    point_segment_subdivisions: int,
    mask_axis_spikes: bool,
    show_filament_line: bool,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = compute_field(mode, n_sides, wire_radius, rotation_deg, n_subdivisions, point_segment_subdivisions)
    x = data["x"]
    z = data["z"]
    mag_linear = data["mag_linear"]
    mag_point = data["mag_point"]
    err = data["err"]
    path_x = data["path_x"]
    path_z = data["path_z"]

    if mask_axis_spikes:
        mag_linear = np.where(mag_linear > 1e2, np.nan, mag_linear)
        err = np.where(err > 1e2, np.nan, err)
    mag_log10 = np.maximum(np.log10(mag_linear + 1e-30), LOG10_FLOOR)
    err_log10 = np.where(np.isnan(err), np.nan, np.maximum(np.log10(err + 1e-30), LOG10_FLOOR))
    mid = GRID_SIZE // 2
    x_slice_ymax = finite_positive_max(mag_linear[mid, :])
    z_slice_ymax = finite_positive_max(mag_linear[:, mid])

    value_title = "|B| [T]" if mode == "b" else "|A| [T m]"
    title_prefix = "B-Field" if mode == "b" else "Vector Potential"

    top_fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.15,
        subplot_titles=[
            f"{title_prefix} magnitude (log10)",
            "Slice along x (z = 0)",
        ],
    )
    top_fig.add_trace(
        go.Heatmap(
            x=x,
            y=z,
            z=mag_log10,
            colorscale="Magma",
            colorbar={
                "title": f"log10({value_title})",
                "thickness": 14,
                "x": -0.15,
                "xanchor": "left",
            },
            zmin=np.nanmin(mag_log10),
            zmax=np.nanmax(mag_log10),
        ),
        row=1,
        col=1,
    )
    if show_filament_line:
        top_fig.add_trace(
            go.Scatter(
                x=path_x,
                y=path_z,
                mode="lines",
                line={"color": "white", "width": 3},
                name="Path geometry",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    top_fig.add_trace(
        go.Scatter(
            x=x,
            y=mag_linear[mid, :],
            mode="lines",
            line={"color": "black", "width": 2},
            name="Linear filament",
        ),
        row=1,
        col=2,
    )
    top_fig.add_trace(
        go.Scatter(
            x=x,
            y=mag_point[mid, :],
            mode="lines",
            line={"color": "deepskyblue", "width": 2, "dash": "dash"},
            name="Point segment",
        ),
        row=1,
        col=2,
    )
    top_fig.update_xaxes(title_text="x [m]", row=1, col=1)
    top_fig.update_yaxes(title_text="z [m]", row=1, col=1, scaleanchor="x", scaleratio=1.0)
    top_fig.update_xaxes(title_text="x [m]", row=1, col=2)
    top_fig.update_yaxes(title_text=value_title, row=1, col=2)
    if x_slice_ymax is not None:
        top_fig.update_yaxes(range=[0.0, 2.0 * x_slice_ymax], row=1, col=2)
    top_fig.update_xaxes(showgrid=False)
    top_fig.update_yaxes(showgrid=False)
    top_fig.update_layout(
        height=460,
        title=title_prefix,
        margin={"l": 50, "r": 20, "t": 110, "b": 45},
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

    bottom_fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.15,
        subplot_titles=[
            "Point-segment error (log10)",
            "Slice along z (x = 0)",
        ],
    )
    bottom_fig.add_trace(
        go.Heatmap(
            x=x,
            y=z,
            z=err_log10,
            colorscale="Viridis",
            colorbar={
                "title": f"log10(delta {value_title})",
                "thickness": 14,
                "x": -0.15,
                "xanchor": "left",
            },
            zmin=np.nanmin(err_log10),
            zmax=np.nanmax(err_log10),
        ),
        row=1,
        col=1,
    )
    bottom_fig.add_trace(
        go.Scatter(
            x=z,
            y=mag_linear[:, mid],
            mode="lines",
            line={"color": "black", "width": 2},
            name="Linear filament (z-slice)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    bottom_fig.add_trace(
        go.Scatter(
            x=z,
            y=mag_point[:, mid],
            mode="lines",
            line={"color": "deepskyblue", "width": 2, "dash": "dash"},
            name="Point segment (z-slice)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    bottom_fig.update_xaxes(title_text="x [m]", row=1, col=1)
    bottom_fig.update_yaxes(title_text="z [m]", row=1, col=1, scaleanchor="x", scaleratio=1.0)
    bottom_fig.update_xaxes(title_text="z [m]", row=1, col=2)
    bottom_fig.update_yaxes(title_text=value_title, row=1, col=2)
    if z_slice_ymax is not None:
        bottom_fig.update_yaxes(range=[0.0, 2.0 * z_slice_ymax], row=1, col=2)
    bottom_fig.update_xaxes(showgrid=False)
    bottom_fig.update_yaxes(showgrid=False)
    bottom_fig.update_layout(
        height=460,
        margin={"l": 50, "r": 20, "t": 50, "b": 60},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return top_fig, bottom_fig


def build_equivalence_figures(
    n_sides: int,
    wire_radius: float,
    rotation_deg: float,
    n_subdivisions: int,
    mask_axis_spikes: bool,
    show_filament_line: bool,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = compute_field_equivalence(n_sides, wire_radius, rotation_deg, n_subdivisions)
    x = data["x"]
    z = data["z"]
    bmag = data["bmag"]
    curl_mag = data["curl_mag"]
    err = data["err"]
    path_x = data["path_x"]
    path_z = data["path_z"]

    if mask_axis_spikes:
        bmag = np.where(bmag > 1e2, np.nan, bmag)
        err = np.where(err > 1e2, np.nan, err)
    b_log10 = np.maximum(np.log10(bmag + 1e-30), LOG10_FLOOR)
    err_log10 = np.where(np.isnan(err), np.nan, np.maximum(np.log10(err + 1e-30), LOG10_FLOOR))
    mid = EQUIV_GRID_SIZE // 2
    top_fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.15,
        subplot_titles=[
            "|B| from linear filament (log10)",
            "Slice along x (z = 0): |B| vs |curl(A)|",
        ],
    )
    top_fig.add_trace(
        go.Heatmap(
            x=x,
            y=z,
            z=b_log10,
            colorscale="Magma",
            colorbar={
                "title": "log10(|B| [T])",
                "thickness": 14,
                "x": -0.15,
                "xanchor": "left",
            },
            zmin=np.nanmin(b_log10),
            zmax=np.nanmax(b_log10),
        ),
        row=1,
        col=1,
    )
    if show_filament_line:
        top_fig.add_trace(
            go.Scatter(
                x=path_x,
                y=path_z,
                mode="lines",
                line={"color": "white", "width": 3},
                name="Path geometry",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    top_fig.add_trace(
        go.Scatter(
            x=x,
            y=bmag[mid, :],
            mode="lines",
            line={"color": "black", "width": 2},
            name="|B| (linear filament)",
        ),
        row=1,
        col=2,
    )
    top_fig.add_trace(
        go.Scatter(
            x=x,
            y=curl_mag[mid, :],
            mode="lines",
            line={"color": "deepskyblue", "width": 2, "dash": "dash"},
            name="|curl(A)|",
        ),
        row=1,
        col=2,
    )
    top_fig.update_xaxes(title_text="x [m]", row=1, col=1)
    top_fig.update_yaxes(title_text="z [m]", row=1, col=1, scaleanchor="x", scaleratio=1.0)
    top_fig.update_xaxes(title_text="x [m]", row=1, col=2)
    top_fig.update_yaxes(title_text="Magnitude", row=1, col=2)
    top_fig.update_xaxes(showgrid=False)
    top_fig.update_yaxes(showgrid=False)
    top_fig.update_layout(
        height=460,
        title="Field Equivalence",
        margin={"l": 50, "r": 20, "t": 110, "b": 45},
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

    bottom_fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.15,
        subplot_titles=[
            "|B - curl(A)| error (log10)",
            "Error slice along z (x = 0)",
        ],
    )
    bottom_fig.add_trace(
        go.Heatmap(
            x=x,
            y=z,
            z=err_log10,
            colorscale="Viridis",
            colorbar={
                "title": "log10(|B-curl(A)|)",
                "thickness": 14,
                "x": -0.15,
                "xanchor": "left",
            },
            zmin=np.nanmin(err_log10),
            zmax=np.nanmax(err_log10),
        ),
        row=1,
        col=1,
    )
    bottom_fig.add_trace(
        go.Scatter(
            x=z,
            y=err[:, mid],
            mode="lines",
            line={"color": "firebrick", "width": 2},
            name="|B-curl(A)| (z-slice)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    bottom_fig.update_xaxes(title_text="x [m]", row=1, col=1)
    bottom_fig.update_yaxes(title_text="z [m]", row=1, col=1, scaleanchor="x", scaleratio=1.0)
    bottom_fig.update_xaxes(title_text="z [m]", row=1, col=2)
    bottom_fig.update_yaxes(title_text="Error magnitude", row=1, col=2)
    bottom_fig.update_xaxes(showgrid=False)
    bottom_fig.update_yaxes(showgrid=False)
    bottom_fig.update_layout(
        height=460,
        margin={"l": 50, "r": 20, "t": 50, "b": 60},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return top_fig, bottom_fig


def build_section_geometry_figure(
    n_sides: int,
    wire_radius: float,
    rotation_deg: float,
    n_subdivisions: int,
    section_grid_n: int,
):
    import plotly.graph_objects as go

    section_grid_n = normalize_section_grid_n(section_grid_n)

    vertices, starts, ends, _xyzfil, _dlxyzfil, ifil = build_linear_filaments(
        n_sides, rotation_deg, n_subdivisions
    )
    dl = ends - starts
    xyzfil_ref, dlxyzfil_ref, _ifil_ref, n_offsets = distribute_filaments_in_cylinder_section(
        starts, dl, ifil, wire_radius, section_grid_n
    )
    starts_ref = np.column_stack(xyzfil_ref)
    ends_ref = starts_ref + np.column_stack(dlxyzfil_ref)

    x_single, y_single, z_single = segment_lines_xyz(starts, ends)
    x_dist, y_dist, z_dist = segment_lines_xyz(starts_ref, ends_ref)

    path = np.vstack((vertices, vertices[0])) if n_sides >= 3 else vertices

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x_single,
            y=y_single,
            z=z_single,
            mode="lines",
            line={"color": "black", "width": 4},
            name="Single filaments",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x_dist,
            y=y_dist,
            z=z_dist,
            mode="lines",
            line={"color": "deepskyblue", "width": 2},
            opacity=0.4,
            name="Distributed filaments",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=path[:, 2],
            mode="lines",
            line={"color": "firebrick", "width": 6},
            name="Path centerline",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        height=620,
        title={
            "text": (
                "Filament geometry overlay | "
                f"single: {starts.shape[0]} segments, "
                f"distributed: {starts_ref.shape[0]} filaments ({section_grid_n}x{section_grid_n} grid, "
                f"{n_offsets}/segment)"
            ),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"b": 5},
        },
        margin={"l": 40, "r": 20, "t": 45, "b": 30},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.93,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.8)",
        },
    )
    fig.update_scenes(
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        zaxis_title="z [m]",
        aspectmode="data",
        camera={"eye": {"x": 1.5, "y": 1.4, "z": 1.0}},
    )
    return fig


def build_section_comparison_figures(
    mode: str,
    n_sides: int,
    wire_radius: float,
    rotation_deg: float,
    n_subdivisions: int,
    section_grid_n: int,
    distributed_use_area_radius: bool,
    mask_axis_spikes: bool,
    show_filament_line: bool,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = compute_section_comparison_field(
        mode,
        n_sides,
        wire_radius,
        rotation_deg,
        n_subdivisions,
        section_grid_n,
        distributed_use_area_radius,
    )
    x = data["x"]
    z = data["z"]
    mag_model = data["mag_model"]
    mag_ref = data["mag_ref"]
    err = data["err"]
    path_x = data["path_x"]
    path_z = data["path_z"]
    n_offsets = int(data["n_offsets"])
    n_grid = int(data["section_grid_n"])
    ref_wire_radius = float(data["ref_wire_radius"])

    if mask_axis_spikes:
        mag_model = np.where(mag_model > 1e2, np.nan, mag_model)
        mag_ref = np.where(mag_ref > 1e2, np.nan, mag_ref)
        err = np.where(err > 1e2, np.nan, err)

    mag_log10 = np.maximum(np.log10(mag_model + 1e-30), LOG10_FLOOR)
    err_log10 = np.where(np.isnan(err), np.nan, np.maximum(np.log10(err + 1e-30), LOG10_FLOOR))
    mid = len(x) // 2

    value_title = "|B| [T]" if mode == "b" else "|A| [T m]"
    title_prefix = "B-Field" if mode == "b" else "Vector Potential"

    top_fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.15,
        subplot_titles=[
            f"{title_prefix} finite-thickness model (log10)",
            "Slice along x (z = 0): model vs distributed section",
        ],
    )
    top_fig.add_trace(
        go.Heatmap(
            x=x,
            y=z,
            z=mag_log10,
            colorscale="Magma",
            colorbar={
                "title": f"log10({value_title})",
                "thickness": 14,
                "x": -0.15,
                "xanchor": "left",
            },
            zmin=np.nanmin(mag_log10),
            zmax=np.nanmax(mag_log10),
        ),
        row=1,
        col=1,
    )
    if show_filament_line:
        top_fig.add_trace(
            go.Scatter(
                x=path_x,
                y=path_z,
                mode="lines",
                line={"color": "white", "width": 3},
                name="Path geometry",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    top_fig.add_trace(
        go.Scatter(
            x=x,
            y=mag_model[mid, :],
            mode="lines",
            line={"color": "black", "width": 2},
            name="Finite-thickness model",
        ),
        row=1,
        col=2,
    )
    top_fig.add_trace(
        go.Scatter(
            x=x,
            y=mag_ref[mid, :],
            mode="lines",
            line={"color": "deepskyblue", "width": 2, "dash": "dash"},
            name=(
                f"Uniform section ({n_grid}x{n_grid} grid, {n_offsets} fil/segment, "
                f"r={'0' if ref_wire_radius == 0.0 else f'{ref_wire_radius:.3g}'} m)"
            ),
        ),
        row=1,
        col=2,
    )
    top_fig.update_xaxes(title_text="x [m]", row=1, col=1)
    top_fig.update_yaxes(title_text="z [m]", row=1, col=1, scaleanchor="x", scaleratio=1.0)
    top_fig.update_xaxes(title_text="x [m]", row=1, col=2)
    top_fig.update_yaxes(title_text=value_title, row=1, col=2)
    top_fig.update_xaxes(showgrid=False)
    top_fig.update_yaxes(showgrid=False)
    top_fig.update_layout(
        height=460,
        title=f"{title_prefix} Conductor Model Check",
        margin={"l": 50, "r": 20, "t": 110, "b": 45},
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

    bottom_fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.15,
        subplot_titles=[
            "Model - distributed-section error (log10)",
            "Slice along z (x = 0): model vs distributed section",
        ],
    )
    bottom_fig.add_trace(
        go.Heatmap(
            x=x,
            y=z,
            z=err_log10,
            colorscale="Viridis",
            colorbar={
                "title": f"log10(delta {value_title})",
                "thickness": 14,
                "x": -0.15,
                "xanchor": "left",
            },
            zmin=np.nanmin(err_log10),
            zmax=np.nanmax(err_log10),
        ),
        row=1,
        col=1,
    )
    bottom_fig.add_trace(
        go.Scatter(
            x=z,
            y=mag_model[:, mid],
            mode="lines",
            line={"color": "black", "width": 2},
            name="Finite-thickness model (z-slice)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    bottom_fig.add_trace(
        go.Scatter(
            x=z,
            y=mag_ref[:, mid],
            mode="lines",
            line={"color": "deepskyblue", "width": 2, "dash": "dash"},
            name="Uniform section (z-slice)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    bottom_fig.update_xaxes(title_text="x [m]", row=1, col=1)
    bottom_fig.update_yaxes(title_text="z [m]", row=1, col=1, scaleanchor="x", scaleratio=1.0)
    bottom_fig.update_xaxes(title_text="z [m]", row=1, col=2)
    bottom_fig.update_yaxes(title_text=value_title, row=1, col=2)
    bottom_fig.update_xaxes(showgrid=False)
    bottom_fig.update_yaxes(showgrid=False)
    bottom_fig.update_layout(
        height=460,
        margin={"l": 50, "r": 20, "t": 50, "b": 60},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return top_fig, bottom_fig


@lru_cache(maxsize=8)
def compute_boundary_element_field(
    mode: str,
    n_sides: int,
    wire_radius: float,
    rotation_deg: float,
    n_subdivisions: int,
    boundary_strip_length_nodes: int,
) -> dict[str, np.ndarray | float]:
    vertices, starts, ends, xyzfil, dlxyzfil, ifil = build_linear_filaments(
        n_sides, rotation_deg, n_subdivisions
    )
    nodes, triangles, s, _upper_paths, _lower_paths, _connectors = build_boundary_element_strips(
        starts,
        ends,
        ifil,
        wire_radius,
        boundary_strip_length_nodes,
    )

    x = np.linspace(-DOMAIN, DOMAIN, BOUNDARY_COMPARE_GRID_SIZE)
    z = np.linspace(-DOMAIN, DOMAIN, BOUNDARY_COMPARE_GRID_SIZE)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    yy = np.zeros_like(xx)
    xyzp = (xx.ravel(), yy.ravel(), zz.ravel())
    obs = np.column_stack((xyzp[0], xyzp[1], xyzp[2]))

    t0 = time.perf_counter()
    if mode == "b":
        vx_model, vy_model, vz_model = cfsem.flux_density_linear_filament(
            xyzp, xyzfil, dlxyzfil, ifil, wire_radius=wire_radius, par=True
        )
    else:
        vx_model, vy_model, vz_model = cfsem.vector_potential_linear_filament(
            xyzp, xyzfil, dlxyzfil, ifil, wire_radius=wire_radius, par=True
        )
    t_model = time.perf_counter() - t0

    t0 = time.perf_counter()
    if mode == "b":
        vx_strip, vy_strip, vz_strip = cfsem.flux_density_triangle_mesh(
            obs, nodes, triangles, s, par=True, quad="gl3"
        )
    else:
        vx_strip, vy_strip, vz_strip = cfsem.vector_potential_triangle_mesh(
            obs, nodes, triangles, s, par=True, quad="gl3"
        )
    t_strip = time.perf_counter() - t0

    mag_model = np.sqrt(vx_model * vx_model + vy_model * vy_model + vz_model * vz_model).reshape(xx.shape)
    mag_strip = np.sqrt(vx_strip * vx_strip + vy_strip * vy_strip + vz_strip * vz_strip).reshape(xx.shape)
    err = np.abs(mag_model - mag_strip)

    npts = xyzp[0].size
    return {
        "x": x,
        "z": z,
        "mag_model": mag_model,
        "mag_strip": mag_strip,
        "err": err,
        "path_x": np.r_[vertices[:, 0], vertices[0, 0]] if n_sides >= 3 else vertices[:, 0],
        "path_z": np.r_[vertices[:, 2], vertices[0, 2]] if n_sides >= 3 else vertices[:, 2],
        "t_model": t_model,
        "t_strip": t_strip,
        "n_model": float(ifil.size * npts),
        "n_strip": float(triangles.shape[0] * npts),
        "n_triangles": float(triangles.shape[0]),
        "n_nodes": float(nodes.shape[0]),
        "strip_length_nodes": float(normalize_boundary_element_length_nodes(boundary_strip_length_nodes)),
    }


def build_boundary_element_figures(
    mode: str,
    n_sides: int,
    wire_radius: float,
    rotation_deg: float,
    n_subdivisions: int,
    boundary_strip_length_nodes: int,
    mask_axis_spikes: bool,
    show_filament_line: bool,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = compute_boundary_element_field(
        mode,
        n_sides,
        wire_radius,
        rotation_deg,
        n_subdivisions,
        boundary_strip_length_nodes,
    )
    x = data["x"]
    z = data["z"]
    mag_model = data["mag_model"]
    mag_strip = data["mag_strip"]
    err = data["err"]
    path_x = data["path_x"]
    path_z = data["path_z"]
    n_triangles = int(data["n_triangles"])
    n_strip_nodes = int(data["strip_length_nodes"])

    if mask_axis_spikes:
        mag_model = np.where(mag_model > 1e2, np.nan, mag_model)
        mag_strip = np.where(mag_strip > 1e2, np.nan, mag_strip)
        err = np.where(err > 1e2, np.nan, err)

    mag_log10 = np.maximum(np.log10(mag_model + 1e-30), LOG10_FLOOR)
    err_log10 = np.where(np.isnan(err), np.nan, np.maximum(np.log10(err + 1e-30), LOG10_FLOOR))
    mid = len(x) // 2

    value_title = "|B| [T]" if mode == "b" else "|A| [T m]"
    title_prefix = "B-Field" if mode == "b" else "Vector Potential"

    top_fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.15,
        subplot_titles=[
            f"{title_prefix} linear-filament model (log10)",
            "Slice along x (z = 0): linear filament vs triangle strip",
        ],
    )
    top_fig.add_trace(
        go.Heatmap(
            x=x,
            y=z,
            z=mag_log10,
            colorscale="Magma",
            colorbar={
                "title": f"log10({value_title})",
                "thickness": 14,
                "x": -0.15,
                "xanchor": "left",
            },
            zmin=np.nanmin(mag_log10),
            zmax=np.nanmax(mag_log10),
        ),
        row=1,
        col=1,
    )
    if show_filament_line:
        top_fig.add_trace(
            go.Scatter(
                x=path_x,
                y=path_z,
                mode="lines",
                line={"color": "white", "width": 3},
                name="Path geometry",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
    top_fig.add_trace(
        go.Scatter(
            x=x,
            y=mag_model[mid, :],
            mode="lines",
            line={"color": "black", "width": 2},
            name="Linear filament",
        ),
        row=1,
        col=2,
    )
    top_fig.add_trace(
        go.Scatter(
            x=x,
            y=mag_strip[mid, :],
            mode="lines",
            line={"color": "deepskyblue", "width": 2, "dash": "dash"},
            name=f"Triangle strip ({n_strip_nodes} nodes/seg, {n_triangles} tris)",
        ),
        row=1,
        col=2,
    )
    top_fig.update_xaxes(title_text="x [m]", row=1, col=1)
    top_fig.update_yaxes(title_text="z [m]", row=1, col=1, scaleanchor="x", scaleratio=1.0)
    top_fig.update_xaxes(title_text="x [m]", row=1, col=2)
    top_fig.update_yaxes(title_text=value_title, row=1, col=2)
    top_fig.update_xaxes(showgrid=False)
    top_fig.update_yaxes(showgrid=False)
    top_fig.update_layout(
        height=460,
        title=f"{title_prefix} Boundary-Element Check",
        margin={"l": 50, "r": 20, "t": 110, "b": 45},
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

    bottom_fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.15,
        subplot_titles=[
            "Linear - boundary-element error (log10)",
            "Slice along z (x = 0): linear filament vs triangle strip",
        ],
    )
    bottom_fig.add_trace(
        go.Heatmap(
            x=x,
            y=z,
            z=err_log10,
            colorscale="Viridis",
            colorbar={
                "title": f"log10(delta {value_title})",
                "thickness": 14,
                "x": -0.15,
                "xanchor": "left",
            },
            zmin=np.nanmin(err_log10),
            zmax=np.nanmax(err_log10),
        ),
        row=1,
        col=1,
    )
    bottom_fig.add_trace(
        go.Scatter(
            x=z,
            y=mag_model[:, mid],
            mode="lines",
            line={"color": "black", "width": 2},
            name="Linear filament (z-slice)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    bottom_fig.add_trace(
        go.Scatter(
            x=z,
            y=mag_strip[:, mid],
            mode="lines",
            line={"color": "deepskyblue", "width": 2, "dash": "dash"},
            name="Triangle strip (z-slice)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    bottom_fig.update_xaxes(title_text="x [m]", row=1, col=1)
    bottom_fig.update_yaxes(title_text="z [m]", row=1, col=1, scaleanchor="x", scaleratio=1.0)
    bottom_fig.update_xaxes(title_text="z [m]", row=1, col=2)
    bottom_fig.update_yaxes(title_text=value_title, row=1, col=2)
    bottom_fig.update_xaxes(showgrid=False)
    bottom_fig.update_yaxes(showgrid=False)
    bottom_fig.update_layout(
        height=460,
        margin={"l": 50, "r": 20, "t": 50, "b": 60},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return top_fig, bottom_fig


def build_boundary_element_geometry_figure(
    n_sides: int,
    wire_radius: float,
    rotation_deg: float,
    n_subdivisions: int,
    boundary_strip_length_nodes: int,
):
    import plotly.graph_objects as go

    vertices, starts, ends, _xyzfil, _dlxyzfil, ifil = build_linear_filaments(
        n_sides, rotation_deg, n_subdivisions
    )
    nodes, triangles, _s, upper_paths, lower_paths, end_connectors = build_boundary_element_strips(
        starts,
        ends,
        ifil,
        wire_radius,
        boundary_strip_length_nodes,
    )
    path = np.vstack((vertices, vertices[0])) if n_sides >= 3 else vertices
    x_upper, y_upper, z_upper = polyline_collection_xyz(upper_paths)
    x_lower, y_lower, z_lower = polyline_collection_xyz(lower_paths)
    x_conn, y_conn, z_conn = polyline_collection_xyz(end_connectors)
    x_mesh, y_mesh, z_mesh = triangle_mesh_edge_lines_xyz(nodes, triangles)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=x_mesh,
            y=y_mesh,
            z=z_mesh,
            mode="lines",
            line={"color": "black", "width": 2},
            opacity=0.7,
            name="Triangle mesh edges",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=nodes[:, 0],
            y=nodes[:, 1],
            z=nodes[:, 2],
            mode="markers",
            marker={"color": "black", "size": 3, "symbol": "circle"},
            name="Mesh nodes",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x_upper,
            y=y_upper,
            z=z_upper,
            mode="lines",
            line={"color": "deepskyblue", "width": 3},
            opacity=0.85,
            name="Upper strip edge",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x_lower,
            y=y_lower,
            z=z_lower,
            mode="lines",
            line={"color": "royalblue", "width": 3},
            opacity=0.85,
            name="Lower strip edge",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x_conn,
            y=y_conn,
            z=z_conn,
            mode="lines",
            line={"color": "lightsteelblue", "width": 2},
            name="Strip end connectors",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=path[:, 2],
            mode="lines",
            line={"color": "firebrick", "width": 6},
            name="Path centerline",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=620,
        title={
            "text": (
                "Boundary-element strip geometry | "
                f"{triangles.shape[0]} triangles, {nodes.shape[0]} nodes, "
                f"{normalize_boundary_element_length_nodes(boundary_strip_length_nodes)} nodes/segment"
            ),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"b": 5},
        },
        margin={"l": 40, "r": 20, "t": 45, "b": 30},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.93,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.8)",
        },
    )
    fig.update_scenes(
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        zaxis_title="z [m]",
        aspectmode="data",
        camera={"eye": {"x": 1.5, "y": 1.4, "z": 1.0}},
    )
    return fig


def build_perf_summary(
    mode: str,
    n_sides: int,
    wire_radius: float,
    rotation_deg: float,
    n_subdivisions: int,
    point_segment_subdivisions: int = DEFAULT_POINT_SEGMENT_SUBDIVISIONS,
    section_grid_n: int = DEFAULT_SECTION_GRID_N,
    distributed_use_area_radius: bool = False,
    boundary_strip_length_nodes: int = DEFAULT_BOUNDARY_ELEMENT_LENGTH_NODES,
) -> str:
    if mode in ("b", "a"):
        data = compute_field(
            mode, n_sides, wire_radius, rotation_deg, n_subdivisions, point_segment_subdivisions
        )
        label = "B-field" if mode == "b" else "Vector potential"
        return (
            f"{label} | "
            f"{build_plot_context_summary(
                n_sides,
                wire_radius,
                rotation_deg,
                n_subdivisions,
                point_segment_subdivisions=point_segment_subdivisions,
            )} | "
            f"linear: {data['t_linear']:.3f}s / {data['n_linear']:.2e} interactions, "
            f"point-segment: {data['t_point']:.3f}s / {data['n_point']:.2e} interactions"
        )

    if mode in ("cb", "ca"):
        field_mode = "b" if mode == "cb" else "a"
        data = compute_section_comparison_field(
            field_mode,
            n_sides,
            wire_radius,
            rotation_deg,
            n_subdivisions,
            section_grid_n,
            distributed_use_area_radius,
        )
        label = "B-field" if field_mode == "b" else "Vector potential"
        radius_mode = "area-equivalent" if distributed_use_area_radius else "zero-radius"
        n_grid = int(data["section_grid_n"])
        return (
            f"{label} conductor-model check | "
            f"{build_plot_context_summary(
                n_sides,
                wire_radius,
                rotation_deg,
                n_subdivisions,
                section_grid_n=n_grid,
                section_filaments_per_segment=int(data['n_offsets']),
                distributed_radius_mode=radius_mode,
            )} | "
            f"finite-thickness: {data['t_model']:.3f}s / {data['n_model']:.2e} interactions, "
            f"distributed section: {data['t_ref']:.3f}s / {data['n_ref']:.2e} interactions"
        )

    if mode in ("bb", "ba"):
        field_mode = "b" if mode == "bb" else "a"
        data = compute_boundary_element_field(
            field_mode,
            n_sides,
            wire_radius,
            rotation_deg,
            n_subdivisions,
            boundary_strip_length_nodes,
        )
        label = "B-field" if field_mode == "b" else "Vector potential"
        n_strip_nodes = int(data["strip_length_nodes"])
        return (
            f"{label} boundary-element check | "
            f"{build_plot_context_summary(
                n_sides,
                wire_radius,
                rotation_deg,
                n_subdivisions,
                boundary_strip_length_nodes=n_strip_nodes,
            )} | "
            f"linear: {data['t_model']:.3f}s / {data['n_model']:.2e} interactions, "
            f"triangle strip: {data['t_strip']:.3f}s / {data['n_strip']:.2e} interactions "
            f"({int(data['n_triangles'])} tris)"
        )

    data = compute_field_equivalence(n_sides, wire_radius, rotation_deg, n_subdivisions)
    return (
        f"Field equivalence (B vs curl(A)) | "
        f"{build_plot_context_summary(n_sides, wire_radius, rotation_deg, n_subdivisions)} | "
        f"B: {data['t_b']:.3f}s / {data['n_b']:.2e} interactions, "
        f"curl(A): {data['t_curl']:.3f}s / {data['n_curl']:.2e} interactions"
    )


def create_app():
    from dash import Dash, Input, Output, dcc, html, no_update

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H3("CFSEM Biot-Savart and Vector Potential"),
            html.P("Use the slider to set geometry: 1 is a straight line, 3-50 are closed polygons."),
            html.Div(
                [
                    html.Div(
                        [
                            html.P(
                                "Path geometry (sides)",
                                style={"marginTop": "0.25rem", "marginBottom": "0.25rem"},
                            ),
                            dcc.Slider(
                                id="polygon-sides",
                                min=1,
                                max=50,
                                step=1,
                                value=3,
                                marks={1: "1", 10: "10", 20: "20", 30: "30", 40: "40", 50: "50"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.P(
                                "Sub-divisions per segment",
                                style={"marginTop": "0.25rem", "marginBottom": "0.25rem"},
                            ),
                            dcc.Slider(
                                id="segment-subdivisions",
                                min=1,
                                max=10,
                                step=1,
                                value=1,
                                marks={1: "1", 3: "3", 5: "5", 7: "7", 10: "10"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.P(
                                "Point-segment lengthwise discretizations",
                                style={"marginTop": "0.25rem", "marginBottom": "0.25rem"},
                            ),
                            dcc.Slider(
                                id="point-segment-subdivisions",
                                min=1,
                                max=MAX_POINT_SEGMENT_SUBDIVISIONS,
                                step=1,
                                value=DEFAULT_POINT_SEGMENT_SUBDIVISIONS,
                                marks={
                                    1: "1",
                                    100: "100",
                                    200: "200",
                                    300: "300",
                                    400: "400",
                                },
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.P(
                                "Boundary-element strip nodes / segment",
                                style={"marginTop": "0.25rem", "marginBottom": "0.25rem"},
                            ),
                            dcc.Slider(
                                id="boundary-strip-length-nodes",
                                min=2,
                                max=MAX_BOUNDARY_ELEMENT_LENGTH_NODES,
                                step=1,
                                value=DEFAULT_BOUNDARY_ELEMENT_LENGTH_NODES,
                                marks={2: "2", 4: "4", 8: "8", 12: "12", 16: "16", 20: "20"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.P(
                                "Wire radius [m]", style={"marginTop": "0.25rem", "marginBottom": "0.25rem"}
                            ),
                            dcc.Slider(
                                id="wire-radius",
                                min=0.0,
                                max=0.1,
                                step=0.001,
                                value=DEFAULT_WIRE_RADIUS,
                                marks={0.0: "0.00", 0.02: "0.02", 0.05: "0.05", 0.08: "0.08", 0.1: "0.10"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.P(
                                "Rotation [deg]", style={"marginTop": "0.25rem", "marginBottom": "0.25rem"}
                            ),
                            dcc.Slider(
                                id="rotation-deg",
                                min=0,
                                max=360,
                                step=1,
                                value=0,
                                marks={0: "0", 90: "90", 180: "180", 270: "270", 360: "360"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.P(
                                "Distributed section grid size (n x n)",
                                style={"marginTop": "0.25rem", "marginBottom": "0.25rem"},
                            ),
                            dcc.Slider(
                                id="section-grid-size",
                                min=2,
                                max=MAX_SECTION_GRID_N,
                                step=2,
                                value=DEFAULT_SECTION_GRID_N,
                                marks={2: "2", 50: "50", 100: "100", 200: "200", 300: "300", 400: "400"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Div(
                                id="section-grid-summary",
                                children=distributed_section_summary_text(DEFAULT_SECTION_GRID_N),
                                style={
                                    "marginTop": "0.35rem",
                                    "fontFamily": "monospace",
                                    "fontSize": "0.9rem",
                                },
                            ),
                        ]
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, minmax(280px, 1fr))",
                    "columnGap": "1rem",
                    "rowGap": "1.25rem",
                    "paddingBottom": "0.5rem",
                },
            ),
            html.Div(
                id="perf-summary",
                style={"marginTop": "1.5rem", "marginBottom": "1.0rem", "fontFamily": "monospace"},
            ),
            html.Div(
                [
                    dcc.Checklist(
                        id="show-filament-line",
                        options=[{"label": "Show filament line", "value": "show"}],
                        value=["show"],
                    ),
                    dcc.Checklist(
                        id="mask-axis-spikes",
                        options=[{"label": "Mask axis spikes > 1e2", "value": "mask"}],
                        value=[],
                    ),
                    dcc.Checklist(
                        id="section-radius-mode",
                        options=[
                            {
                                "label": "Distributed filaments use area-equivalent radius",
                                "value": "area",
                            }
                        ],
                        value=["area"],
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "row",
                    "flexWrap": "wrap",
                    "columnGap": "1.25rem",
                    "rowGap": "0.5rem",
                    "marginBottom": "0.75rem",
                },
            ),
            dcc.Tabs(
                id="field-tab",
                value="b",
                children=[
                    dcc.Tab(
                        label="B-field",
                        value="b",
                        children=[
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-b-top"),
                                ),
                                style={"marginBottom": "0.5rem"},
                            ),
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-b-bottom"),
                                )
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Vector potential",
                        value="a",
                        children=[
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-a-top"),
                                ),
                                style={"marginBottom": "0.5rem"},
                            ),
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-a-bottom"),
                                )
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Conductor model (B)",
                        value="cb",
                        children=[
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-cb-top"),
                                ),
                                style={"marginBottom": "0.5rem"},
                            ),
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-cb-bottom"),
                                )
                            ),
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-cb-geom"),
                                ),
                                style={"marginTop": "0.5rem"},
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Conductor model (A)",
                        value="ca",
                        children=[
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-ca-top"),
                                ),
                                style={"marginBottom": "0.5rem"},
                            ),
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-ca-bottom"),
                                )
                            ),
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-ca-geom"),
                                ),
                                style={"marginTop": "0.5rem"},
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Boundary element (B)",
                        value="bb",
                        children=[
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-bb-top"),
                                ),
                                style={"marginBottom": "0.5rem"},
                            ),
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-bb-bottom"),
                                )
                            ),
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-bb-geom"),
                                ),
                                style={"marginTop": "0.5rem"},
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Boundary element (A)",
                        value="ba",
                        children=[
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-ba-top"),
                                ),
                                style={"marginBottom": "0.5rem"},
                            ),
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-ba-bottom"),
                                )
                            ),
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-ba-geom"),
                                ),
                                style={"marginTop": "0.5rem"},
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Field equivalence",
                        value="eq",
                        children=[
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-eq-top"),
                                ),
                                style={"marginBottom": "0.5rem"},
                            ),
                            html.Div(
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(id="field-figure-eq-bottom"),
                                )
                            ),
                        ],
                    ),
                ],
            ),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "1rem"},
    )

    @app.callback(
        Output("field-figure-b-top", "figure"),
        Output("field-figure-b-bottom", "figure"),
        Output("perf-summary", "children"),
        Input("polygon-sides", "value"),
        Input("segment-subdivisions", "value"),
        Input("point-segment-subdivisions", "value"),
        Input("boundary-strip-length-nodes", "value"),
        Input("wire-radius", "value"),
        Input("rotation-deg", "value"),
        Input("mask-axis-spikes", "value"),
        Input("show-filament-line", "value"),
        Input("section-grid-size", "value"),
        Input("section-radius-mode", "value"),
        Input("field-tab", "value"),
    )
    def update_b_figure(
        n_sides: int,
        n_subdivisions: int,
        point_segment_subdivisions: int,
        boundary_strip_length_nodes: int,
        wire_radius: float,
        rotation_deg: float,
        mask_axis_spikes: list[str],
        show_filament_line: list[str],
        section_grid_size: int,
        section_radius_mode: list[str],
        field_tab: str,
    ):
        sides = int(n_sides)
        n_sub = int(np.clip(n_subdivisions, 1, 10))
        n_point_sub = normalize_point_segment_subdivisions(point_segment_subdivisions)
        n_strip_nodes = normalize_boundary_element_length_nodes(boundary_strip_length_nodes)
        radius = float(np.clip(wire_radius, 0.0, 0.1))
        rotation = float(np.mod(rotation_deg, 360.0))
        section_grid_n = normalize_section_grid_n(section_grid_size)
        use_area_radius = "area" in section_radius_mode
        mask_spikes = "mask" in mask_axis_spikes
        show_line = "show" in show_filament_line
        if field_tab != "b":
            return (
                no_update,
                no_update,
                build_perf_summary(
                    field_tab,
                    sides,
                    radius,
                    rotation,
                    n_sub,
                    n_point_sub,
                    section_grid_n,
                    use_area_radius,
                    n_strip_nodes,
                ),
            )
        top_fig, bottom_fig = build_figures(
            "b", sides, radius, rotation, n_sub, n_point_sub, mask_spikes, show_line
        )
        return (
            top_fig,
            bottom_fig,
            build_perf_summary(
                "b",
                sides,
                radius,
                rotation,
                n_sub,
                n_point_sub,
                section_grid_n,
                use_area_radius,
                n_strip_nodes,
            ),
        )

    @app.callback(
        Output("section-grid-summary", "children"),
        Input("section-grid-size", "value"),
    )
    def update_section_grid_summary(section_grid_size: int) -> str:
        section_grid_n = normalize_section_grid_n(section_grid_size)
        return distributed_section_summary_text(section_grid_n)

    @app.callback(
        Output("field-figure-a-top", "figure"),
        Output("field-figure-a-bottom", "figure"),
        Input("polygon-sides", "value"),
        Input("segment-subdivisions", "value"),
        Input("point-segment-subdivisions", "value"),
        Input("wire-radius", "value"),
        Input("rotation-deg", "value"),
        Input("mask-axis-spikes", "value"),
        Input("show-filament-line", "value"),
        Input("field-tab", "value"),
    )
    def update_a_figure(
        n_sides: int,
        n_subdivisions: int,
        point_segment_subdivisions: int,
        wire_radius: float,
        rotation_deg: float,
        mask_axis_spikes: list[str],
        show_filament_line: list[str],
        field_tab: str,
    ):
        if field_tab != "a":
            return no_update, no_update
        sides = int(n_sides)
        n_sub = int(np.clip(n_subdivisions, 1, 10))
        n_point_sub = normalize_point_segment_subdivisions(point_segment_subdivisions)
        radius = float(np.clip(wire_radius, 0.0, 0.1))
        rotation = float(np.mod(rotation_deg, 360.0))
        mask_spikes = "mask" in mask_axis_spikes
        show_line = "show" in show_filament_line
        return build_figures("a", sides, radius, rotation, n_sub, n_point_sub, mask_spikes, show_line)

    @app.callback(
        Output("field-figure-cb-top", "figure"),
        Output("field-figure-cb-bottom", "figure"),
        Output("field-figure-cb-geom", "figure"),
        Input("polygon-sides", "value"),
        Input("segment-subdivisions", "value"),
        Input("wire-radius", "value"),
        Input("rotation-deg", "value"),
        Input("mask-axis-spikes", "value"),
        Input("show-filament-line", "value"),
        Input("section-grid-size", "value"),
        Input("section-radius-mode", "value"),
        Input("field-tab", "value"),
    )
    def update_conductor_model_b_figure(
        n_sides: int,
        n_subdivisions: int,
        wire_radius: float,
        rotation_deg: float,
        mask_axis_spikes: list[str],
        show_filament_line: list[str],
        section_grid_size: int,
        section_radius_mode: list[str],
        field_tab: str,
    ):
        if field_tab != "cb":
            return no_update, no_update, no_update
        sides = int(n_sides)
        n_sub = int(np.clip(n_subdivisions, 1, 10))
        radius = float(np.clip(wire_radius, 0.0, 0.1))
        rotation = float(np.mod(rotation_deg, 360.0))
        section_grid_n = normalize_section_grid_n(section_grid_size)
        use_area_radius = "area" in section_radius_mode
        mask_spikes = "mask" in mask_axis_spikes
        show_line = "show" in show_filament_line
        top_fig, bottom_fig = build_section_comparison_figures(
            "b", sides, radius, rotation, n_sub, section_grid_n, use_area_radius, mask_spikes, show_line
        )
        geom_fig = build_section_geometry_figure(sides, radius, rotation, n_sub, section_grid_n)
        return top_fig, bottom_fig, geom_fig

    @app.callback(
        Output("field-figure-ca-top", "figure"),
        Output("field-figure-ca-bottom", "figure"),
        Output("field-figure-ca-geom", "figure"),
        Input("polygon-sides", "value"),
        Input("segment-subdivisions", "value"),
        Input("wire-radius", "value"),
        Input("rotation-deg", "value"),
        Input("mask-axis-spikes", "value"),
        Input("show-filament-line", "value"),
        Input("section-grid-size", "value"),
        Input("section-radius-mode", "value"),
        Input("field-tab", "value"),
    )
    def update_conductor_model_a_figure(
        n_sides: int,
        n_subdivisions: int,
        wire_radius: float,
        rotation_deg: float,
        mask_axis_spikes: list[str],
        show_filament_line: list[str],
        section_grid_size: int,
        section_radius_mode: list[str],
        field_tab: str,
    ):
        if field_tab != "ca":
            return no_update, no_update, no_update
        sides = int(n_sides)
        n_sub = int(np.clip(n_subdivisions, 1, 10))
        radius = float(np.clip(wire_radius, 0.0, 0.1))
        rotation = float(np.mod(rotation_deg, 360.0))
        section_grid_n = normalize_section_grid_n(section_grid_size)
        use_area_radius = "area" in section_radius_mode
        mask_spikes = "mask" in mask_axis_spikes
        show_line = "show" in show_filament_line
        top_fig, bottom_fig = build_section_comparison_figures(
            "a", sides, radius, rotation, n_sub, section_grid_n, use_area_radius, mask_spikes, show_line
        )
        geom_fig = build_section_geometry_figure(sides, radius, rotation, n_sub, section_grid_n)
        return top_fig, bottom_fig, geom_fig

    @app.callback(
        Output("field-figure-bb-top", "figure"),
        Output("field-figure-bb-bottom", "figure"),
        Output("field-figure-bb-geom", "figure"),
        Input("polygon-sides", "value"),
        Input("segment-subdivisions", "value"),
        Input("boundary-strip-length-nodes", "value"),
        Input("wire-radius", "value"),
        Input("rotation-deg", "value"),
        Input("mask-axis-spikes", "value"),
        Input("show-filament-line", "value"),
        Input("field-tab", "value"),
    )
    def update_boundary_element_b_figure(
        n_sides: int,
        n_subdivisions: int,
        boundary_strip_length_nodes: int,
        wire_radius: float,
        rotation_deg: float,
        mask_axis_spikes: list[str],
        show_filament_line: list[str],
        field_tab: str,
    ):
        if field_tab != "bb":
            return no_update, no_update, no_update
        sides = int(n_sides)
        n_sub = int(np.clip(n_subdivisions, 1, 10))
        n_strip_nodes = normalize_boundary_element_length_nodes(boundary_strip_length_nodes)
        radius = float(np.clip(wire_radius, 0.0, 0.1))
        rotation = float(np.mod(rotation_deg, 360.0))
        mask_spikes = "mask" in mask_axis_spikes
        show_line = "show" in show_filament_line
        top_fig, bottom_fig = build_boundary_element_figures(
            "b",
            sides,
            radius,
            rotation,
            n_sub,
            n_strip_nodes,
            mask_spikes,
            show_line,
        )
        geom_fig = build_boundary_element_geometry_figure(
            sides,
            radius,
            rotation,
            n_sub,
            n_strip_nodes,
        )
        return top_fig, bottom_fig, geom_fig

    @app.callback(
        Output("field-figure-ba-top", "figure"),
        Output("field-figure-ba-bottom", "figure"),
        Output("field-figure-ba-geom", "figure"),
        Input("polygon-sides", "value"),
        Input("segment-subdivisions", "value"),
        Input("boundary-strip-length-nodes", "value"),
        Input("wire-radius", "value"),
        Input("rotation-deg", "value"),
        Input("mask-axis-spikes", "value"),
        Input("show-filament-line", "value"),
        Input("field-tab", "value"),
    )
    def update_boundary_element_a_figure(
        n_sides: int,
        n_subdivisions: int,
        boundary_strip_length_nodes: int,
        wire_radius: float,
        rotation_deg: float,
        mask_axis_spikes: list[str],
        show_filament_line: list[str],
        field_tab: str,
    ):
        if field_tab != "ba":
            return no_update, no_update, no_update
        sides = int(n_sides)
        n_sub = int(np.clip(n_subdivisions, 1, 10))
        n_strip_nodes = normalize_boundary_element_length_nodes(boundary_strip_length_nodes)
        radius = float(np.clip(wire_radius, 0.0, 0.1))
        rotation = float(np.mod(rotation_deg, 360.0))
        mask_spikes = "mask" in mask_axis_spikes
        show_line = "show" in show_filament_line
        top_fig, bottom_fig = build_boundary_element_figures(
            "a",
            sides,
            radius,
            rotation,
            n_sub,
            n_strip_nodes,
            mask_spikes,
            show_line,
        )
        geom_fig = build_boundary_element_geometry_figure(
            sides,
            radius,
            rotation,
            n_sub,
            n_strip_nodes,
        )
        return top_fig, bottom_fig, geom_fig

    @app.callback(
        Output("field-figure-eq-top", "figure"),
        Output("field-figure-eq-bottom", "figure"),
        Input("polygon-sides", "value"),
        Input("segment-subdivisions", "value"),
        Input("wire-radius", "value"),
        Input("rotation-deg", "value"),
        Input("mask-axis-spikes", "value"),
        Input("show-filament-line", "value"),
        Input("field-tab", "value"),
    )
    def update_equivalence_figure(
        n_sides: int,
        n_subdivisions: int,
        wire_radius: float,
        rotation_deg: float,
        mask_axis_spikes: list[str],
        show_filament_line: list[str],
        field_tab: str,
    ):
        if field_tab != "eq":
            return no_update, no_update
        sides = int(n_sides)
        n_sub = int(np.clip(n_subdivisions, 1, 10))
        radius = float(np.clip(wire_radius, 0.0, 0.1))
        rotation = float(np.mod(rotation_deg, 360.0))
        mask_spikes = "mask" in mask_axis_spikes
        show_line = "show" in show_filament_line
        return build_equivalence_figures(sides, radius, rotation, n_sub, mask_spikes, show_line)

    return app


def main() -> None:
    app = create_app()

    if not os.getenv("CFSEM_TESTING"):
        app.run(debug=True)
    else:
        # smoketest figures if we're not running the full gui
        top_fig, _bottom_fig = build_figures(
            "b",
            3,
            DEFAULT_WIRE_RADIUS,
            0.0,
            1,
            DEFAULT_POINT_SEGMENT_SUBDIVISIONS,
            False,
            True,
        )
        export_docs_example_figure(top_fig)
        build_figures(
            "a",
            3,
            DEFAULT_WIRE_RADIUS,
            0.0,
            1,
            DEFAULT_POINT_SEGMENT_SUBDIVISIONS,
            False,
            True,
        )
        build_section_comparison_figures(
            "b", 3, DEFAULT_WIRE_RADIUS, 0.0, 1, DEFAULT_SECTION_GRID_N, False, False, True
        )
        build_section_comparison_figures(
            "a", 3, DEFAULT_WIRE_RADIUS, 0.0, 1, DEFAULT_SECTION_GRID_N, False, False, True
        )
        build_section_geometry_figure(3, DEFAULT_WIRE_RADIUS, 0.0, 1, DEFAULT_SECTION_GRID_N)
        build_boundary_element_figures(
            "b", 3, DEFAULT_WIRE_RADIUS, 0.0, 1, DEFAULT_BOUNDARY_ELEMENT_LENGTH_NODES, False, True
        )
        build_boundary_element_figures(
            "a", 3, DEFAULT_WIRE_RADIUS, 0.0, 1, DEFAULT_BOUNDARY_ELEMENT_LENGTH_NODES, False, True
        )
        build_boundary_element_geometry_figure(
            3, DEFAULT_WIRE_RADIUS, 0.0, 1, DEFAULT_BOUNDARY_ELEMENT_LENGTH_NODES
        )
        build_equivalence_figures(3, DEFAULT_WIRE_RADIUS, 0.0, 1, False, True)


if __name__ == "__main__":
    main()
