from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

import cfsem

GRID_N = 124 if os.getenv("CFSEM_TESTING") else 228
DEFAULT_SOURCE_COUNT = 63 if os.getenv("CFSEM_TESTING") else 159
MIN_SOURCE_COUNT = 8
MAX_SOURCE_COUNT = 255 if os.getenv("CFSEM_TESTING") else 100_000
MAX_DIRECT_SELF_INTERACTIONS = 1_000_000_000
CURRENT = 1.0
WIRE_RADIUS = 0.015
HELICAL_WIRE_RADIUS = 0.055
TRIANGLE_STRIP_WIDTH = 0.08
SOURCE_SPAN = 1.4
DISTRIBUTED_SOURCE_RADIUS = 0.5 * SOURCE_SPAN
DISTRIBUTED_SOURCE_CENTER_OFFSET = 2.5 * SOURCE_SPAN
DEFAULT_TWIST_PITCH = 0.36
DEFAULT_HELIX_WIDTH = HELICAL_WIRE_RADIUS
DEFAULT_BEND_CURVATURE = 2.0 / SOURCE_SPAN
DEFAULT_LOOP_FRACTION = 0.5
DEFAULT_THETA = 0.05
LOG10_MIN_SOURCE_COUNT = float(np.log10(MIN_SOURCE_COUNT))
LOG10_DEFAULT_SOURCE_COUNT = float(np.log10(DEFAULT_SOURCE_COUNT))
LOG10_MAX_SOURCE_COUNT = float(np.log10(MAX_SOURCE_COUNT))
MAX_PLOTTED_PATH_POINTS = 400
MAX_PLOTTED_AABBS = 500
DISTRIBUTED_SOURCE_SEED = 1729


@dataclass(frozen=True)
class Geometry:
    centerline: np.ndarray
    helix: np.ndarray
    xyzfil: tuple[np.ndarray, np.ndarray, np.ndarray]
    dlxyzfil: tuple[np.ndarray, np.ndarray, np.ndarray]
    current: np.ndarray
    wire_radius: np.ndarray
    volume_xyzfil: tuple[np.ndarray, np.ndarray, np.ndarray]
    volume_dlxyzfil: tuple[np.ndarray, np.ndarray, np.ndarray]
    volume_current: np.ndarray
    volume_wire_radius: np.ndarray
    dipole_loc: tuple[np.ndarray, np.ndarray, np.ndarray]
    dipole_moment: tuple[np.ndarray, np.ndarray, np.ndarray]
    dipole_outer_radius: np.ndarray
    volume_dipole_loc: tuple[np.ndarray, np.ndarray, np.ndarray]
    volume_dipole_moment: tuple[np.ndarray, np.ndarray, np.ndarray]
    volume_dipole_outer_radius: np.ndarray
    strip_nodes: np.ndarray
    strip_triangles: np.ndarray
    strip_stream_function: np.ndarray
    volume_strip_nodes: np.ndarray
    volume_strip_triangles: np.ndarray
    volume_strip_stream_function: np.ndarray
    obs: tuple[np.ndarray, np.ndarray, np.ndarray]
    obs_grid: tuple[np.ndarray, np.ndarray]
    extent: float


def fixed_span_arc_centerline(span: float, curvature: float, loop_fraction: float, n: int) -> np.ndarray:
    curvature = max(0.0, float(curvature))
    x = np.linspace(-0.5 * span, 0.5 * span, n, endpoint=True)
    loop_fraction = max(0.0, min(1.0, float(loop_fraction)))
    if curvature <= 1.0e-12 or loop_fraction <= 1.0e-12:
        return np.vstack((x, np.zeros_like(x), np.zeros_like(x)))

    max_curvature = 2.0 / span
    curvature = min(curvature, max_curvature)
    radius = 1.0 / curvature
    half_angle = np.arcsin(0.5 * span / radius)
    half_angle = min(np.pi, half_angle * loop_fraction / DEFAULT_LOOP_FRACTION)
    if half_angle <= 1.0e-12:
        return np.vstack((x, np.zeros_like(x), np.zeros_like(x)))
    theta = np.linspace(-half_angle, half_angle, n, endpoint=True)
    x = radius * np.sin(theta)
    z = radius * np.cos(theta)
    return np.vstack((x, np.zeros_like(x), z))


def section_observation_plane(
    extent: float,
    n: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    x = np.linspace(-extent, extent, n)
    z = np.linspace(-extent, extent, n)
    xg, zg = np.meshgrid(x, z, indexing="xy")
    yg = np.zeros_like(xg)
    return (xg.ravel(), yg.ravel(), zg.ravel()), (xg, zg)


def build_geometry(
    geometry_layout: str,
    n_centerline: int,
    grid_n: int,
    twist_pitch: float,
    helix_width: float,
    bend_curvature: float,
    loop_fraction: float,
) -> Geometry:
    centerline = fixed_span_arc_centerline(SOURCE_SPAN, bend_curvature, loop_fraction, n_centerline)
    helix = np.asarray(
        cfsem.filament_helix_path(
            path=centerline,
            helix_start_offset=(0.0, helix_width, 0.0),
            twist_pitch=float(twist_pitch),
            angle_offset=0.0,
        )
    )
    segment_centers = 0.5 * (helix[:, :-1] + helix[:, 1:])
    source_centroid = np.mean(segment_centers, axis=1, keepdims=True)
    centerline = centerline - source_centroid
    helix = helix - source_centroid
    starts = helix[:, :-1].T
    ends = helix[:, 1:].T
    dl = ends - starts
    xyzfil = (starts[:, 0], starts[:, 1], starts[:, 2])
    dlxyzfil = (dl[:, 0], dl[:, 1], dl[:, 2])
    current = np.full(starts.shape[0], CURRENT)
    wire_radius = np.full(starts.shape[0], WIRE_RADIUS)
    dipole_loc, dipole_moment, dipole_outer_radius = build_segment_dipoles(starts, dl, current)
    (
        volume_dipole_loc,
        volume_dipole_moment,
        volume_dipole_outer_radius,
        volume_xyzfil,
        volume_dlxyzfil,
        volume_current,
        volume_wire_radius,
        volume_strip_nodes,
        volume_strip_triangles,
        volume_strip_stream_function,
    ) = build_volume_sources(
        max(1, starts.shape[0]),
        DISTRIBUTED_SOURCE_RADIUS,
    )
    strip_nodes, strip_triangles, strip_stream_function = build_triangle_strip(
        helix,
        TRIANGLE_STRIP_WIDTH,
        CURRENT,
    )
    extent = 8.0 * (SOURCE_SPAN + 3.0 * helix_width)
    obs, obs_grid = section_observation_plane(extent, grid_n)
    return Geometry(
        centerline,
        helix,
        xyzfil,
        dlxyzfil,
        current,
        wire_radius,
        volume_xyzfil,
        volume_dlxyzfil,
        volume_current,
        volume_wire_radius,
        dipole_loc,
        dipole_moment,
        dipole_outer_radius,
        volume_dipole_loc,
        volume_dipole_moment,
        volume_dipole_outer_radius,
        strip_nodes,
        strip_triangles,
        strip_stream_function,
        volume_strip_nodes,
        volume_strip_triangles,
        volume_strip_stream_function,
        obs,
        obs_grid,
        extent,
    )


def build_volume_sources(
    n: int,
    radius: float,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(DISTRIBUTED_SOURCE_SEED)
    angles = 2.0 * np.pi * rng.random(n)
    radii = radius * np.sqrt(rng.random(n))
    disk_offsets = np.array(
        [
            [-DISTRIBUTED_SOURCE_CENTER_OFFSET, 0.0, -DISTRIBUTED_SOURCE_CENTER_OFFSET],
            [DISTRIBUTED_SOURCE_CENTER_OFFSET, 0.0, -DISTRIBUTED_SOURCE_CENTER_OFFSET],
            [-DISTRIBUTED_SOURCE_CENTER_OFFSET, 0.0, DISTRIBUTED_SOURCE_CENTER_OFFSET],
            [DISTRIBUTED_SOURCE_CENTER_OFFSET, 0.0, DISTRIBUTED_SOURCE_CENTER_OFFSET],
        ]
    )
    disk_ids = np.arange(n) % disk_offsets.shape[0]
    loc = np.column_stack((radii * np.cos(angles), np.zeros(n), radii * np.sin(angles)))
    loc = loc + disk_offsets[disk_ids]

    central_moment = np.array([0.0, 0.0, 1.0])
    r_norm = np.linalg.norm(loc, axis=1)
    r_hat = loc / np.maximum(r_norm[:, None], 1.0e-30)
    m_dot_r = r_hat @ central_moment
    direction = 3.0 * r_hat * m_dot_r[:, None] - central_moment[None, :]
    direction_norm = np.linalg.norm(direction, axis=1)
    direction = direction / np.maximum(direction_norm[:, None], 1.0e-30)
    moment_scale = CURRENT * radius**3 / max(1, n)
    moments = moment_scale * direction
    segment_length = 2.0 * radius / max(4.0, n ** (1.0 / 3.0))
    starts = loc - 0.5 * segment_length * direction
    dl = segment_length * direction
    strip_nodes, strip_triangles, strip_stream_function = build_distributed_triangle_strips(
        loc,
        direction,
        segment_length,
        TRIANGLE_STRIP_WIDTH,
        CURRENT,
    )
    return (
        (loc[:, 0], loc[:, 1], loc[:, 2]),
        (moments[:, 0], moments[:, 1], moments[:, 2]),
        np.zeros(n),
        (starts[:, 0], starts[:, 1], starts[:, 2]),
        (dl[:, 0], dl[:, 1], dl[:, 2]),
        np.full(n, CURRENT),
        np.full(n, WIRE_RADIUS),
        strip_nodes,
        strip_triangles,
        strip_stream_function,
    )


def build_distributed_triangle_strips(
    centers: np.ndarray,
    directions: np.ndarray,
    length: float,
    strip_width: float,
    stream_function_jump: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = centers.shape[0]
    reference = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    parallel = np.abs(np.sum(reference * directions, axis=1)) > 0.9
    reference[parallel] = np.array([0.0, 1.0, 0.0])
    width_dir = np.cross(directions, reference)
    width_norm = np.linalg.norm(width_dir, axis=1)
    width_dir = width_dir / np.maximum(width_norm[:, None], 1.0e-30)

    half_length = 0.5 * length * directions
    half_width = 0.5 * strip_width * width_dir
    nodes = np.empty((4 * n, 3))
    nodes[0::4] = centers - half_length - half_width
    nodes[1::4] = centers - half_length + half_width
    nodes[2::4] = centers + half_length - half_width
    nodes[3::4] = centers + half_length + half_width

    triangles = np.empty((2 * n, 3), dtype=np.int64)
    for i in range(n):
        base = 4 * i
        triangles[2 * i] = [base, base + 2, base + 1]
        triangles[2 * i + 1] = [base + 1, base + 2, base + 3]

    stream_function = np.empty(4 * n)
    stream_function[0::4] = 0.0
    stream_function[2::4] = 0.0
    stream_function[1::4] = stream_function_jump
    stream_function[3::4] = stream_function_jump
    return nodes, triangles, stream_function


def build_segment_dipoles(
    starts: np.ndarray,
    dl: np.ndarray,
    current: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    centers = starts + 0.5 * dl
    # This is a compact source distribution for comparing direct and hierarchical dipole kernels.
    moments = current[:, None] * dl
    return (
        (centers[:, 0], centers[:, 1], centers[:, 2]),
        (moments[:, 0], moments[:, 1], moments[:, 2]),
        np.zeros(starts.shape[0]),
    )


def build_triangle_strip(
    helix: np.ndarray,
    strip_width: float,
    stream_function_jump: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offset = np.array([[0.0], [0.5 * strip_width], [0.0]])
    lower = (helix - offset).T
    upper = (helix + offset).T
    n_path = helix.shape[1]
    nodes = np.empty((2 * n_path, 3))
    nodes[0::2] = lower
    nodes[1::2] = upper

    triangles = np.empty((2 * (n_path - 1), 3), dtype=np.int64)
    for i in range(n_path - 1):
        lower0 = 2 * i
        upper0 = lower0 + 1
        lower1 = lower0 + 2
        upper1 = lower0 + 3
        triangles[2 * i] = [lower0, lower1, upper0]
        triangles[2 * i + 1] = [upper0, lower1, upper1]

    stream_function = np.empty(2 * n_path)
    stream_function[0::2] = 0.0
    stream_function[1::2] = stream_function_jump
    return nodes, triangles, stream_function


def source_count_from_log10(log10_source_count: float) -> int:
    count = int(round(10.0 ** float(log10_source_count)))
    return max(MIN_SOURCE_COUNT, min(MAX_SOURCE_COUNT, count))


def field_magnitude(field: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return np.sqrt(field[0] * field[0] + field[1] * field[1] + field[2] * field[2])


def relative_error(
    hierarchical: tuple[np.ndarray, np.ndarray, np.ndarray],
    direct: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    err = field_magnitude(
        (
            hierarchical[0] - direct[0],
            hierarchical[1] - direct[1],
            hierarchical[2] - direct[2],
        )
    )
    ref = np.maximum(field_magnitude(direct), 1e-30)
    return err / ref


def linear_filament_centers(
    geometry: Geometry,
    geometry_layout: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyzfil, dlxyzfil, _current, _wire_radius = filament_source_arrays(geometry, geometry_layout)
    return (
        xyzfil[0] + 0.5 * dlxyzfil[0],
        xyzfil[1] + 0.5 * dlxyzfil[1],
        xyzfil[2] + 0.5 * dlxyzfil[2],
    )


def triangle_centroids(
    geometry: Geometry,
    geometry_layout: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, triangles, _stream_function = boundary_source_arrays(geometry, geometry_layout)
    tri_nodes = nodes[triangles]
    centroids = np.mean(tri_nodes, axis=1)
    return (centroids[:, 0], centroids[:, 1], centroids[:, 2])


def dipole_source_arrays(
    geometry: Geometry,
    geometry_layout: str,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    if geometry_layout == "distributed":
        return (
            geometry.volume_dipole_loc,
            geometry.volume_dipole_moment,
            geometry.volume_dipole_outer_radius,
        )
    return geometry.dipole_loc, geometry.dipole_moment, geometry.dipole_outer_radius


def filament_source_arrays(
    geometry: Geometry,
    geometry_layout: str,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    if geometry_layout == "distributed":
        return (
            geometry.volume_xyzfil,
            geometry.volume_dlxyzfil,
            geometry.volume_current,
            geometry.volume_wire_radius,
        )
    return geometry.xyzfil, geometry.dlxyzfil, geometry.current, geometry.wire_radius


def boundary_source_arrays(
    geometry: Geometry,
    geometry_layout: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if geometry_layout == "distributed":
        return (
            geometry.volume_strip_nodes,
            geometry.volume_strip_triangles,
            geometry.volume_strip_stream_function,
        )
    return geometry.strip_nodes, geometry.strip_triangles, geometry.strip_stream_function


def boundary_triangle_values(
    stream_function: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.ascontiguousarray(stream_function[triangles[:, 0]]),
        np.ascontiguousarray(stream_function[triangles[:, 1]]),
        np.ascontiguousarray(stream_function[triangles[:, 2]]),
    )


def solve_self_fields(
    geometry: Geometry,
    geometry_layout: str,
    source_geometry: str,
    construction_method: str,
    theta: float,
    par: bool,
) -> dict[str, object]:
    if source_geometry == "dipole":
        dipole_loc, dipole_moment, dipole_outer_radius = dipole_source_arrays(geometry, geometry_layout)
        self_obs = dipole_loc
        source_count = dipole_outer_radius.size
    elif source_geometry == "boundary":
        strip_nodes, strip_triangles, strip_stream_function = boundary_source_arrays(
            geometry, geometry_layout
        )
        self_obs = triangle_centroids(geometry, geometry_layout)
        source_count = strip_triangles.shape[0]
    else:
        xyzfil, dlxyzfil, current, wire_radius = filament_source_arrays(geometry, geometry_layout)
        self_obs = linear_filament_centers(geometry, geometry_layout)
        source_count = current.size

    interactions = source_count * source_count
    direct_time: float | None = None
    direct_b = None
    direct_a = None
    if interactions <= MAX_DIRECT_SELF_INTERACTIONS:
        t0 = time.perf_counter()
        if source_geometry == "dipole":
            direct_b = cfsem.flux_density_dipole(
                dipole_loc,
                dipole_moment,
                self_obs,
                par=par,
                outer_radius=dipole_outer_radius,
            )
            direct_a = cfsem.vector_potential_dipole(
                dipole_loc,
                dipole_moment,
                self_obs,
                par=par,
                outer_radius=dipole_outer_radius,
            )
        elif source_geometry == "boundary":
            self_obs_array = np.column_stack(self_obs)
            direct_b = cfsem.flux_density_triangle_mesh(
                self_obs_array,
                strip_nodes,
                strip_triangles,
                strip_stream_function,
                par=par,
            )
            direct_a = cfsem.vector_potential_triangle_mesh(
                self_obs_array,
                strip_nodes,
                strip_triangles,
                strip_stream_function,
                par=par,
            )
        else:
            direct_b = cfsem.flux_density_linear_filament(
                self_obs,
                xyzfil,
                dlxyzfil,
                current,
                wire_radius,
                par=par,
            )
            direct_a = cfsem.vector_potential_linear_filament(
                self_obs,
                xyzfil,
                dlxyzfil,
                current,
                wire_radius,
                par=par,
            )
        direct_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    if source_geometry == "dipole":
        dipole_loc, dipole_moment, dipole_outer_radius = dipole_source_arrays(geometry, geometry_layout)
        solver = cfsem.HierarchicalDipoles(theta=theta, construction_method=construction_method)
        solver.set_sources(dipole_loc, dipole_outer_radius)
        build_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        hierarchical_b = solver.flux_density(self_obs, dipole_moment, par=par)
        hierarchical_a = solver.vector_potential(self_obs, dipole_moment, par=par)
    elif source_geometry == "boundary":
        strip_nodes, strip_triangles, strip_stream_function = boundary_source_arrays(
            geometry, geometry_layout
        )
        strip_triangle_values = boundary_triangle_values(strip_stream_function, strip_triangles)
        solver = cfsem.HierarchicalBoundaryElements(
            theta=theta,
            construction_method=construction_method,
        )
        solver.set_sources(strip_nodes, strip_triangles)
        build_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        hierarchical_b = solver.flux_density(self_obs, strip_triangle_values, par=par)
        hierarchical_a = solver.vector_potential(self_obs, strip_triangle_values, par=par)
    else:
        solver = cfsem.HierarchicalLinearFilaments(
            theta=theta,
            construction_method=construction_method,
        )
        solver.set_sources(xyzfil, dlxyzfil, wire_radius)
        build_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        hierarchical_b = solver.flux_density(self_obs, current, par=par)
        hierarchical_a = solver.vector_potential(self_obs, current, par=par)
    eval_time = time.perf_counter() - t0

    return {
        "direct_b": direct_b,
        "direct_a": direct_a,
        "hierarchical_b": hierarchical_b,
        "hierarchical_a": hierarchical_a,
        "interactions": interactions,
        "direct_time": direct_time,
        "direct_skipped": direct_time is None,
        "hierarchical_build_time": build_time,
        "hierarchical_eval_time": eval_time,
    }


def solve_fields(
    geometry: Geometry,
    geometry_layout: str,
    source_geometry: str,
    construction_method: str,
    theta: float,
    par: bool,
    calc_self_field: bool,
    overlay_aabbs: bool,
) -> dict[str, object]:
    direct_build_time = 0.0

    t0 = time.perf_counter()
    if source_geometry == "dipole":
        dipole_loc, dipole_moment, dipole_outer_radius = dipole_source_arrays(geometry, geometry_layout)
        direct_b = cfsem.flux_density_dipole(
            dipole_loc,
            dipole_moment,
            geometry.obs,
            par=par,
            outer_radius=dipole_outer_radius,
        )
        direct_a = cfsem.vector_potential_dipole(
            dipole_loc,
            dipole_moment,
            geometry.obs,
            par=par,
            outer_radius=dipole_outer_radius,
        )
    elif source_geometry == "boundary":
        strip_nodes, strip_triangles, strip_stream_function = boundary_source_arrays(
            geometry, geometry_layout
        )
        obs_array = np.column_stack(geometry.obs)
        direct_b = cfsem.flux_density_triangle_mesh(
            obs_array,
            strip_nodes,
            strip_triangles,
            strip_stream_function,
            par=par,
        )
        direct_a = cfsem.vector_potential_triangle_mesh(
            obs_array,
            strip_nodes,
            strip_triangles,
            strip_stream_function,
            par=par,
        )
    else:
        xyzfil, dlxyzfil, current, wire_radius = filament_source_arrays(geometry, geometry_layout)
        direct_b = cfsem.flux_density_linear_filament(
            geometry.obs,
            xyzfil,
            dlxyzfil,
            current,
            wire_radius,
            par=par,
        )
        direct_a = cfsem.vector_potential_linear_filament(
            geometry.obs,
            xyzfil,
            dlxyzfil,
            current,
            wire_radius,
            par=par,
        )
    direct_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    if source_geometry == "dipole":
        dipole_loc, dipole_moment, dipole_outer_radius = dipole_source_arrays(geometry, geometry_layout)
        solver = cfsem.HierarchicalDipoles(
            theta=theta,
            construction_method=construction_method,
        )
        solver.set_sources(dipole_loc, dipole_outer_radius)
        source_count = dipole_outer_radius.size
        build_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        hierarchical_b = solver.flux_density(geometry.obs, dipole_moment, par=par)
        hierarchical_a = solver.vector_potential(geometry.obs, dipole_moment, par=par)
        accepted_source_levels_b = solver.accepted_source_levels(geometry.obs, dipole_moment, field="b")
        accepted_source_levels_a = solver.accepted_source_levels(geometry.obs, dipole_moment, field="a")
    elif source_geometry == "boundary":
        strip_nodes, strip_triangles, strip_stream_function = boundary_source_arrays(
            geometry, geometry_layout
        )
        strip_triangle_values = boundary_triangle_values(strip_stream_function, strip_triangles)
        solver = cfsem.HierarchicalBoundaryElements(
            theta=theta,
            construction_method=construction_method,
        )
        solver.set_sources(strip_nodes, strip_triangles)
        source_count = strip_triangles.shape[0]
        build_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        hierarchical_b = solver.flux_density(geometry.obs, strip_triangle_values, par=par)
        hierarchical_a = solver.vector_potential(geometry.obs, strip_triangle_values, par=par)
        accepted_source_levels_b = solver.accepted_source_levels(
            geometry.obs, strip_triangle_values, field="b"
        )
        accepted_source_levels_a = solver.accepted_source_levels(
            geometry.obs, strip_triangle_values, field="a"
        )
    else:
        xyzfil, dlxyzfil, current, wire_radius = filament_source_arrays(geometry, geometry_layout)
        solver = cfsem.HierarchicalLinearFilaments(
            theta=theta,
            construction_method=construction_method,
        )
        solver.set_sources(xyzfil, dlxyzfil, wire_radius)
        source_count = current.size
        build_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        hierarchical_b = solver.flux_density(geometry.obs, current, par=par)
        hierarchical_a = solver.vector_potential(geometry.obs, current, par=par)
        accepted_source_levels_b = solver.accepted_source_levels(geometry.obs, current, field="b")
        accepted_source_levels_a = solver.accepted_source_levels(geometry.obs, current, field="a")
    eval_time = time.perf_counter() - t0
    source_tree_aabbs = solver.source_tree_aabbs() if overlay_aabbs else None

    results: dict[str, object] = {
        "direct_b": direct_b,
        "direct_a": direct_a,
        "direct_build_time": direct_build_time,
        "hierarchical_b": hierarchical_b,
        "hierarchical_a": hierarchical_a,
        "direct_time": direct_time,
        "build_time": build_time,
        "eval_time": eval_time,
        "source_count": source_count,
        "source_target_interactions": source_count * geometry.obs[0].size,
        "geometry_layout": geometry_layout,
    }
    results["accepted_source_levels_b"] = accepted_source_levels_b
    results["accepted_source_levels_a"] = accepted_source_levels_a
    if source_tree_aabbs is not None:
        results["source_tree_aabbs"] = source_tree_aabbs
    if calc_self_field:
        results["self_field"] = solve_self_fields(
            geometry,
            geometry_layout=geometry_layout,
            source_geometry=source_geometry,
            construction_method=construction_method,
            theta=theta,
            par=par,
        )
    return results


def heatmap_values(values: np.ndarray, geometry: Geometry) -> np.ndarray:
    return values.reshape(geometry.obs_grid[0].shape)


def decimate_path_for_plot(path: np.ndarray, max_points: int = MAX_PLOTTED_PATH_POINTS) -> np.ndarray:
    if path.shape[1] <= max_points:
        return path
    step = max(1, int(np.ceil(path.shape[1] / max_points)))
    decimated = path[:, ::step]
    if decimated.shape[1] == 0 or not np.array_equal(decimated[:, -1], path[:, -1]):
        decimated = np.column_stack((decimated, path[:, -1]))
    return decimated


def aabb_overlay_path(
    aabbs: tuple[np.ndarray, ...],
    max_boxes: int = MAX_PLOTTED_AABBS,
) -> tuple[list[float | None], list[float | None]]:
    min_x, _min_y, min_z, max_x, _max_y, max_z, levels = aabbs
    if min_x.size == 0:
        return [], []
    indices_by_level: list[np.ndarray] = []
    count = 0
    for level in np.unique(levels.astype(np.int64)):
        level_indices = np.flatnonzero(levels == float(level))
        if count + level_indices.size > max_boxes:
            break
        indices_by_level.append(level_indices)
        count += level_indices.size
    indices = np.concatenate(indices_by_level) if indices_by_level else np.array([0])

    xs: list[float | None] = []
    zs: list[float | None] = []
    for idx in indices:
        i = int(idx)
        xs.extend([min_x[i], max_x[i], max_x[i], min_x[i], min_x[i], None])
        zs.extend([min_z[i], min_z[i], max_z[i], max_z[i], min_z[i], None])
    return xs, zs


def source_geometry_overlay_path(
    geometry: Geometry,
    geometry_layout: str,
) -> tuple[list[float | None], list[float | None], list[float | None], list[float | None]]:
    if geometry_layout == "distributed":
        xyzfil, dlxyzfil, _current, _wire_radius = filament_source_arrays(geometry, geometry_layout)
        count = xyzfil[0].size
        if count <= MAX_PLOTTED_PATH_POINTS:
            indices = np.arange(count)
        else:
            rng = np.random.default_rng(DISTRIBUTED_SOURCE_SEED)
            indices = np.sort(rng.choice(count, size=MAX_PLOTTED_PATH_POINTS, replace=False))

        segment_x: list[float | None] = []
        segment_z: list[float | None] = []
        marker_x: list[float | None] = []
        marker_z: list[float | None] = []
        for idx in indices:
            i = int(idx)
            x0 = float(xyzfil[0][i])
            z0 = float(xyzfil[2][i])
            x1 = float(xyzfil[0][i] + dlxyzfil[0][i])
            z1 = float(xyzfil[2][i] + dlxyzfil[2][i])
            segment_x.extend([x0, x1, None])
            segment_z.extend([z0, z1, None])
            marker_x.append(0.5 * (x0 + x1))
            marker_z.append(0.5 * (z0 + z1))
        return segment_x, segment_z, marker_x, marker_z

    centerline_plot = decimate_path_for_plot(geometry.centerline)
    helix_plot = decimate_path_for_plot(geometry.helix)
    return (
        list(centerline_plot[0]),
        list(centerline_plot[2]),
        list(helix_plot[0]),
        list(helix_plot[2]),
    )


def make_figure(
    geometry: Geometry,
    results: dict[str, object],
    field: str,
    show_error: bool,
    overlay_aabbs: bool,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    direct = results[f"direct_{field}"]
    hierarchical = results[f"hierarchical_{field}"]
    assert isinstance(direct, tuple)
    assert isinstance(hierarchical, tuple)

    left = np.log10(np.maximum(field_magnitude(direct), 1e-30))
    middle = np.log10(np.maximum(field_magnitude(hierarchical), 1e-30))
    right = np.log10(np.maximum(relative_error(hierarchical, direct), 1e-16)) if show_error else middle - left
    right_title = "log10 relative error" if show_error else "log10 magnitude difference"
    accepted_source_levels = results.get(f"accepted_source_levels_{field}")
    show_level_diagnostic = isinstance(accepted_source_levels, np.ndarray)
    col_count = 4 if show_level_diagnostic else 3

    fig = make_subplots(
        rows=1,
        cols=col_count,
        subplot_titles=(
            ("Direct", "Hierarchical", right_title, "Accepted Source Level")
            if show_level_diagnostic
            else ("Direct", "Hierarchical", right_title)
        ),
        horizontal_spacing=0.055,
    )
    xg, zg = geometry.obs_grid
    geometry_layout = str(results.get("geometry_layout", "helical"))
    geom_x0, geom_z0, geom_x1, geom_z1 = source_geometry_overlay_path(geometry, geometry_layout)
    aabb_overlay = results.get("source_tree_aabbs") if overlay_aabbs else None
    aabb_x: list[float | None] = []
    aabb_z: list[float | None] = []
    if isinstance(aabb_overlay, tuple):
        aabb_x, aabb_z = aabb_overlay_path(aabb_overlay)
    traces = [
        (left, "log10 |direct|"),
        (middle, "log10 |hierarchical|"),
        (right, right_title),
    ]
    if show_level_diagnostic:
        traces.append((accepted_source_levels, "mean terminal source level"))
    for col, (values, title) in enumerate(traces, start=1):
        colorscale = "Viridis"
        if col == 3:
            colorscale = [[0.0, "#2c7bb6"], [0.5, "#ffffbf"], [1.0, "#d7191c"]] if show_error else "RdBu"
        if col == 4:
            colorscale = "Cividis"
        fig.add_trace(
            go.Heatmap(
                x=xg[0, :],
                y=zg[:, 0],
                z=heatmap_values(values, geometry),
                showscale=col == 3,
                colorbar={"title": title} if col == 3 else None,
                colorscale=colorscale,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=geom_x0,
                y=geom_z0,
                mode="lines",
                line={"color": "white", "width": 2, "dash": "dash"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=geom_x1,
                y=geom_z1,
                mode="markers" if geometry_layout == "distributed" else "lines",
                line={"color": "black", "width": 1},
                marker={"color": "black", "size": 3} if geometry_layout == "distributed" else None,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        if aabb_x:
            fig.add_trace(
                go.Scatter(
                    x=aabb_x,
                    y=aabb_z,
                    mode="lines",
                    line={"color": "rgba(20, 20, 20, 0.32)", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )

    for axis in fig.select_xaxes():
        axis.update(title="x [m]")
    for axis in fig.select_yaxes():
        axis.update(title="z [m]")
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin={"l": 40, "r": 40, "t": 54, "b": 8},
        title=f"{'B-field' if field == 'b' else 'A-field'} comparison on the centerline plane",
    )
    return fig


def make_self_field_figure(results: dict[str, object], field: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    self_field = results.get("self_field")
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Self Magnitude", "Self Relative Error"),
        horizontal_spacing=0.08,
    )
    fig.update_layout(
        template="plotly_white",
        height=190,
        margin={"l": 40, "r": 40, "t": 48, "b": 24},
        title=f"Self-field {'B' if field == 'b' else 'A'} by source index",
    )
    for axis in fig.select_xaxes():
        axis.update(title="source index")
    for axis in fig.select_yaxes():
        axis.update(title="log10 magnitude")

    if not isinstance(self_field, dict):
        fig.add_annotation(
            text="Enable Calculate self-field to show source-index diagnostics",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    direct = self_field[f"direct_{field}"]
    hierarchical = self_field[f"hierarchical_{field}"]
    assert isinstance(hierarchical, tuple)
    source_index = np.arange(hierarchical[0].size)
    hierarchical_log = np.log10(np.maximum(field_magnitude(hierarchical), 1e-30))
    fig.add_trace(
        go.Scatter(
            x=source_index,
            y=hierarchical_log,
            mode="lines",
            name="Hierarchical",
            line={"color": "#3b6fb6"},
        ),
        row=1,
        col=1,
    )

    if direct is None:
        fig.add_annotation(
            text="Direct self-field skipped",
            xref="x2 domain",
            yref="y2 domain",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    assert isinstance(direct, tuple)
    direct_log = np.log10(np.maximum(field_magnitude(direct), 1e-30))
    error_log = np.log10(np.maximum(relative_error(hierarchical, direct), 1e-16))
    fig.add_trace(
        go.Scatter(
            x=source_index,
            y=direct_log,
            mode="lines",
            name="Direct",
            line={"color": "#555"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=source_index,
            y=error_log,
            mode="lines",
            name="Error",
            line={"color": "#b63b4a"},
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(title="log10 relative error", row=1, col=2)
    return fig


def speedup_text(direct_time: object, hierarchical_time: object) -> str:
    direct = float(direct_time)
    hierarchical = float(hierarchical_time)
    if hierarchical <= 0.0:
        return "n/a"
    return f"{direct / hierarchical:.2f}x"


def make_app():
    from dash import Dash, Input, Output, dcc, html

    app = Dash(__name__)
    sidebar_style = {
        "width": "320px",
        "minWidth": "320px",
        "height": "100vh",
        "overflowY": "auto",
        "padding": "16px",
        "borderRight": "1px solid #d9dde3",
        "boxSizing": "border-box",
        "background": "#f7f8fa",
    }
    content_style = {
        "flex": "1 1 auto",
        "minWidth": "0",
    }
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Label("Source kernel"),
                    dcc.Dropdown(
                        id="source-geometry",
                        value="linear",
                        clearable=False,
                        options=[
                            {"label": "Linear filament", "value": "linear"},
                            {"label": "Dipole", "value": "dipole"},
                            {"label": "Boundary-element triangle strip", "value": "boundary"},
                        ],
                    ),
                    html.Label("Geometry"),
                    dcc.Dropdown(
                        id="geometry-layout",
                        value="helical",
                        clearable=False,
                        options=[
                            {"label": "Helical loop", "value": "helical"},
                            {"label": "Distributed disk", "value": "distributed"},
                        ],
                    ),
                    html.Label("Twist pitch"),
                    dcc.Slider(
                        id="twist-pitch",
                        min=0.12,
                        max=0.8,
                        step=0.01,
                        value=DEFAULT_TWIST_PITCH,
                        marks={0.12: "0.12", 0.36: "0.36", 0.8: "0.8"},
                    ),
                    html.Label("Helix width"),
                    dcc.Slider(
                        id="helix-width",
                        min=0.0,
                        max=0.16,
                        step=0.005,
                        value=DEFAULT_HELIX_WIDTH,
                        marks={0.0: "0", 0.055: "0.055", 0.16: "0.16"},
                    ),
                    html.Label("Bend curvature"),
                    dcc.Slider(
                        id="bend-curvature",
                        min=0.0,
                        max=DEFAULT_BEND_CURVATURE,
                        step=0.05,
                        value=DEFAULT_BEND_CURVATURE,
                        marks={
                            0.0: "0",
                            0.5: "0.5",
                            1.0: "1.0",
                            round(DEFAULT_BEND_CURVATURE, 2): f"{DEFAULT_BEND_CURVATURE:.2f}",
                        },
                    ),
                    html.Label("Loop fraction"),
                    dcc.Slider(
                        id="loop-fraction",
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        value=DEFAULT_LOOP_FRACTION,
                        marks={0.0: "0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1.0: "1"},
                    ),
                    html.Label("Construction"),
                    dcc.Dropdown(
                        id="construction",
                        value="recursive",
                        clearable=False,
                        options=[
                            {"label": "Recursive source tree", "value": "recursive"},
                            {"label": "Morton/LBVH source tree", "value": "morton_lbvh"},
                        ],
                    ),
                    html.Label("Field"),
                    dcc.RadioItems(
                        id="field",
                        value="b",
                        inline=True,
                        options=[
                            {"label": "B", "value": "b"},
                            {"label": "A", "value": "a"},
                        ],
                    ),
                    html.Label("Theta"),
                    dcc.Slider(
                        id="theta",
                        min=0.0,
                        max=0.6,
                        step=0.01,
                        value=DEFAULT_THETA,
                        marks={round(i * 0.1, 1): f"{i * 0.1:.1f}" for i in range(7)},
                    ),
                    html.Label("Sources"),
                    dcc.Slider(
                        id="source-count",
                        min=LOG10_MIN_SOURCE_COUNT,
                        max=LOG10_MAX_SOURCE_COUNT,
                        step=0.01,
                        value=LOG10_DEFAULT_SOURCE_COUNT,
                        marks={
                            LOG10_MIN_SOURCE_COUNT: f"{MIN_SOURCE_COUNT}",
                            LOG10_DEFAULT_SOURCE_COUNT: f"{DEFAULT_SOURCE_COUNT}",
                            LOG10_MAX_SOURCE_COUNT: f"{MAX_SOURCE_COUNT:.0E}",
                        },
                    ),
                    dcc.Checklist(
                        id="options",
                        value=["parallel", "relative-error"],
                        options=[
                            {"label": "Parallel evaluation", "value": "parallel"},
                            {"label": "Show relative error", "value": "relative-error"},
                            {"label": "Calculate self-field", "value": "self-field"},
                            {"label": "Overlay source AABBs", "value": "aabbs"},
                        ],
                    ),
                ],
                style=sidebar_style,
            ),
            html.Div(
                [
                    html.Div(
                        id="timing",
                        style={
                            "fontFamily": "monospace",
                            "padding": "16px 16px 0",
                            "whiteSpace": "pre-wrap",
                        },
                    ),
                    dcc.Graph(
                        id="field-figure",
                        config={"responsive": True},
                        style={"height": "360px", "marginBottom": "0"},
                    ),
                    dcc.Graph(
                        id="self-field-figure",
                        config={"responsive": True},
                        style={"height": "190px", "marginTop": "0"},
                    ),
                ],
                style=content_style,
            ),
        ],
        style={
            "fontFamily": "system-ui, sans-serif",
            "display": "flex",
            "minHeight": "100vh",
        },
    )

    @app.callback(
        Output("field-figure", "figure"),
        Output("self-field-figure", "figure"),
        Output("timing", "children"),
        Input("source-geometry", "value"),
        Input("geometry-layout", "value"),
        Input("twist-pitch", "value"),
        Input("helix-width", "value"),
        Input("bend-curvature", "value"),
        Input("loop-fraction", "value"),
        Input("construction", "value"),
        Input("field", "value"),
        Input("theta", "value"),
        Input("source-count", "value"),
        Input("options", "value"),
    )
    def update(
        source_geometry,
        geometry_layout,
        twist_pitch,
        helix_width,
        bend_curvature,
        loop_fraction,
        construction,
        field,
        theta,
        log10_source_count,
        options,
    ):
        source_count = source_count_from_log10(float(log10_source_count))
        geometry = build_geometry(
            geometry_layout,
            int(source_count) + 1,
            GRID_N,
            float(twist_pitch),
            float(helix_width),
            float(bend_curvature),
            float(loop_fraction),
        )
        opts = set(options or [])
        results = solve_fields(
            geometry,
            geometry_layout=geometry_layout,
            source_geometry=source_geometry,
            construction_method=construction,
            theta=float(theta),
            par="parallel" in opts,
            calc_self_field="self-field" in opts,
            overlay_aabbs="aabbs" in opts,
        )
        fig = make_figure(geometry, results, field, "relative-error" in opts, "aabbs" in opts)
        self_fig = make_self_field_figure(results, field)
        self_field = results.get("self_field")
        self_text = ""
        if isinstance(self_field, dict):
            direct_self = (
                "skipped"
                if self_field["direct_skipped"]
                else f"{float(self_field['direct_time']):.3f}s"
            )
            self_speedup = (
                "n/a"
                if self_field["direct_skipped"]
                else speedup_text(self_field["direct_time"], self_field["hierarchical_eval_time"])
            )
            self_text = (
                f"\nself-field source-source interactions={self_field['interactions']:.1E}\n"
                f"self direct:       evaluation={direct_self}\n"
                f"self hierarchical: construction={self_field['hierarchical_build_time']:.3f}s, "
                f"evaluation={self_field['hierarchical_eval_time']:.3f}s, "
                f"speedup={self_speedup}"
            )
        timing = (
            f"nsrc={results['source_count']}, nobs={geometry.obs[0].size}, "
            f"plane={geometry.obs_grid[0].shape[0]}x{geometry.obs_grid[0].shape[1]}\n"
            f"theta={float(theta):.2f}\n"
            f"geometry={geometry_layout}, kernel={source_geometry}\n"
            f"twist_pitch={float(twist_pitch):.3f}, helix_width={float(helix_width):.3f}, "
            f"bend_curvature={float(bend_curvature):.3f}, loop_fraction={float(loop_fraction):.2f}\n"
            f"original source-target interactions={results['source_target_interactions']:.1E}\n"
            f"direct:       construction={results['direct_build_time']:.3f}s, "
            f"evaluation={results['direct_time']:.3f}s\n"
            f"hierarchical: construction={results['build_time']:.3f}s, "
            f"evaluation={results['eval_time']:.3f}s, "
            f"speedup={speedup_text(results['direct_time'], results['eval_time'])}"
            f"{self_text}"
        )
        return fig, self_fig, timing

    return app


if __name__ == "__main__":
    if not os.getenv("CFSEM_TESTING"):
        make_app().run(debug=True)
