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
DEFAULT_TWIST_PITCH = 0.36
DEFAULT_HELIX_WIDTH = HELICAL_WIRE_RADIUS
DEFAULT_BEND_CURVATURE = 2.0 / SOURCE_SPAN
DEFAULT_LOOP_FRACTION = 0.5
DEFAULT_THETA = 0.05
LOG10_MIN_SOURCE_COUNT = float(np.log10(MIN_SOURCE_COUNT))
LOG10_DEFAULT_SOURCE_COUNT = float(np.log10(DEFAULT_SOURCE_COUNT))
LOG10_MAX_SOURCE_COUNT = float(np.log10(MAX_SOURCE_COUNT))
MAX_PLOTTED_PATH_POINTS = 400


@dataclass(frozen=True)
class Geometry:
    centerline: np.ndarray
    helix: np.ndarray
    xyzfil: tuple[np.ndarray, np.ndarray, np.ndarray]
    dlxyzfil: tuple[np.ndarray, np.ndarray, np.ndarray]
    current: np.ndarray
    wire_radius: np.ndarray
    dipole_loc: tuple[np.ndarray, np.ndarray, np.ndarray]
    dipole_moment: tuple[np.ndarray, np.ndarray, np.ndarray]
    dipole_outer_radius: np.ndarray
    strip_nodes: np.ndarray
    strip_triangles: np.ndarray
    strip_stream_function: np.ndarray
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
        dipole_loc,
        dipole_moment,
        dipole_outer_radius,
        strip_nodes,
        strip_triangles,
        strip_stream_function,
        obs,
        obs_grid,
        extent,
    )


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


def linear_filament_centers(geometry: Geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        geometry.xyzfil[0] + 0.5 * geometry.dlxyzfil[0],
        geometry.xyzfil[1] + 0.5 * geometry.dlxyzfil[1],
        geometry.xyzfil[2] + 0.5 * geometry.dlxyzfil[2],
    )


def triangle_centroids(geometry: Geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri_nodes = geometry.strip_nodes[geometry.strip_triangles]
    centroids = np.mean(tri_nodes, axis=1)
    return (centroids[:, 0], centroids[:, 1], centroids[:, 2])


def solve_self_fields(
    geometry: Geometry,
    source_geometry: str,
    construction_method: str,
    theta: float,
    par: bool,
) -> dict[str, object]:
    if source_geometry == "dipole":
        self_obs = geometry.dipole_loc
        source_count = geometry.dipole_outer_radius.size
    elif source_geometry == "boundary":
        self_obs = triangle_centroids(geometry)
        source_count = geometry.strip_triangles.shape[0]
    else:
        self_obs = linear_filament_centers(geometry)
        source_count = geometry.current.size

    interactions = source_count * source_count
    direct_time: float | None = None
    direct_b = None
    direct_a = None
    if interactions <= MAX_DIRECT_SELF_INTERACTIONS:
        t0 = time.perf_counter()
        if source_geometry == "dipole":
            direct_b = cfsem.flux_density_dipole(
                geometry.dipole_loc,
                geometry.dipole_moment,
                self_obs,
                par=par,
                outer_radius=geometry.dipole_outer_radius,
            )
            direct_a = cfsem.vector_potential_dipole(
                geometry.dipole_loc,
                geometry.dipole_moment,
                self_obs,
                par=par,
                outer_radius=geometry.dipole_outer_radius,
            )
        elif source_geometry == "boundary":
            self_obs_array = np.column_stack(self_obs)
            direct_b = cfsem.flux_density_triangle_mesh(
                self_obs_array,
                geometry.strip_nodes,
                geometry.strip_triangles,
                geometry.strip_stream_function,
                par=par,
            )
            direct_a = cfsem.vector_potential_triangle_mesh(
                self_obs_array,
                geometry.strip_nodes,
                geometry.strip_triangles,
                geometry.strip_stream_function,
                par=par,
            )
        else:
            direct_b = cfsem.flux_density_linear_filament(
                self_obs,
                geometry.xyzfil,
                geometry.dlxyzfil,
                geometry.current,
                geometry.wire_radius,
                par=par,
            )
            direct_a = cfsem.vector_potential_linear_filament(
                self_obs,
                geometry.xyzfil,
                geometry.dlxyzfil,
                geometry.current,
                geometry.wire_radius,
                par=par,
            )
        direct_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    if source_geometry == "dipole":
        solver = cfsem.HierarchicalDipoles(theta=theta, construction_method=construction_method)
        solver.build(geometry.dipole_loc, self_obs, geometry.dipole_outer_radius)
        build_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        hierarchical_b = solver.flux_density(geometry.dipole_moment, par=par)
        hierarchical_a = solver.vector_potential(geometry.dipole_moment, par=par)
    elif source_geometry == "boundary":
        solver = cfsem.HierarchicalBoundaryElements(
            theta=theta,
            construction_method=construction_method,
        )
        solver.build(geometry.strip_nodes, geometry.strip_triangles, self_obs)
        build_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        hierarchical_b = solver.flux_density(geometry.strip_stream_function, par=par)
        hierarchical_a = solver.vector_potential(geometry.strip_stream_function, par=par)
    else:
        solver = cfsem.HierarchicalLinearFilaments(
            theta=theta,
            construction_method=construction_method,
        )
        solver.build(geometry.xyzfil, geometry.dlxyzfil, geometry.wire_radius, self_obs)
        build_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        hierarchical_b = solver.flux_density(geometry.current, par=par)
        hierarchical_a = solver.vector_potential(geometry.current, par=par)
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
    source_geometry: str,
    construction_method: str,
    theta: float,
    par: bool,
    calc_self_field: bool,
) -> dict[str, object]:
    direct_build_time = 0.0

    t0 = time.perf_counter()
    if source_geometry == "dipole":
        direct_b = cfsem.flux_density_dipole(
            geometry.dipole_loc,
            geometry.dipole_moment,
            geometry.obs,
            par=par,
            outer_radius=geometry.dipole_outer_radius,
        )
        direct_a = cfsem.vector_potential_dipole(
            geometry.dipole_loc,
            geometry.dipole_moment,
            geometry.obs,
            par=par,
            outer_radius=geometry.dipole_outer_radius,
        )
    elif source_geometry == "boundary":
        obs_array = np.column_stack(geometry.obs)
        direct_b = cfsem.flux_density_triangle_mesh(
            obs_array,
            geometry.strip_nodes,
            geometry.strip_triangles,
            geometry.strip_stream_function,
            par=par,
        )
        direct_a = cfsem.vector_potential_triangle_mesh(
            obs_array,
            geometry.strip_nodes,
            geometry.strip_triangles,
            geometry.strip_stream_function,
            par=par,
        )
    else:
        direct_b = cfsem.flux_density_linear_filament(
            geometry.obs,
            geometry.xyzfil,
            geometry.dlxyzfil,
            geometry.current,
            geometry.wire_radius,
            par=par,
        )
        direct_a = cfsem.vector_potential_linear_filament(
            geometry.obs,
            geometry.xyzfil,
            geometry.dlxyzfil,
            geometry.current,
            geometry.wire_radius,
            par=par,
        )
    direct_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    if source_geometry == "dipole":
        solver = cfsem.HierarchicalDipoles(
            theta=theta,
            construction_method=construction_method,
        )
        solver.build(geometry.dipole_loc, geometry.obs, geometry.dipole_outer_radius)
        source_count = geometry.dipole_outer_radius.size
        build_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        hierarchical_b = solver.flux_density(geometry.dipole_moment, par=par)
        hierarchical_a = solver.vector_potential(geometry.dipole_moment, par=par)
    elif source_geometry == "boundary":
        solver = cfsem.HierarchicalBoundaryElements(
            theta=theta,
            construction_method=construction_method,
        )
        solver.build(geometry.strip_nodes, geometry.strip_triangles, geometry.obs)
        source_count = geometry.strip_triangles.shape[0]
        build_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        hierarchical_b = solver.flux_density(geometry.strip_stream_function, par=par)
        hierarchical_a = solver.vector_potential(geometry.strip_stream_function, par=par)
    else:
        solver = cfsem.HierarchicalLinearFilaments(
            theta=theta,
            construction_method=construction_method,
        )
        solver.build(geometry.xyzfil, geometry.dlxyzfil, geometry.wire_radius, geometry.obs)
        source_count = geometry.current.size
        build_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        hierarchical_b = solver.flux_density(geometry.current, par=par)
        hierarchical_a = solver.vector_potential(geometry.current, par=par)
    eval_time = time.perf_counter() - t0

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
    }
    if calc_self_field:
        results["self_field"] = solve_self_fields(
            geometry,
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


def make_figure(
    geometry: Geometry,
    results: dict[str, object],
    field: str,
    show_error: bool,
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

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Direct", "Hierarchical", right_title),
        horizontal_spacing=0.055,
    )
    xg, zg = geometry.obs_grid
    centerline_plot = decimate_path_for_plot(geometry.centerline)
    helix_plot = decimate_path_for_plot(geometry.helix)
    traces = [
        (left, "log10 |direct|"),
        (middle, "log10 |hierarchical|"),
        (right, right_title),
    ]
    for col, (values, title) in enumerate(traces, start=1):
        colorscale = "Viridis"
        if col == 3:
            colorscale = [[0.0, "#2c7bb6"], [0.5, "#ffffbf"], [1.0, "#d7191c"]] if show_error else "RdBu"
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
                x=centerline_plot[0],
                y=centerline_plot[2],
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
                x=helix_plot[0],
                y=helix_plot[2],
                mode="lines",
                line={"color": "black", "width": 1},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )

    for axis in fig.select_xaxes():
        axis.update(title="x [m]", scaleanchor=None)
    for axis in fig.select_yaxes():
        axis.update(title="z [m]", scaleanchor="x")
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
        go.Scatter(x=source_index, y=error_log, mode="lines", line={"color": "#b63b4a"}),
        row=1,
        col=2,
    )
    fig.update_yaxes(title="log10 relative error", row=1, col=2)
    return fig


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
                    html.Label("Field geometry"),
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
            source_geometry=source_geometry,
            construction_method=construction,
            theta=float(theta),
            par="parallel" in opts,
            calc_self_field="self-field" in opts,
        )
        fig = make_figure(geometry, results, field, "relative-error" in opts)
        self_fig = make_self_field_figure(results, field)
        self_field = results.get("self_field")
        self_text = ""
        if isinstance(self_field, dict):
            direct_self = (
                "skipped"
                if self_field["direct_skipped"]
                else f"{float(self_field['direct_time']):.3f}s"
            )
            self_text = (
                f"\nself-field source-source interactions={self_field['interactions']:.1E}\n"
                f"self direct:       evaluation={direct_self}\n"
                f"self hierarchical: construction={self_field['hierarchical_build_time']:.3f}s, "
                f"evaluation={self_field['hierarchical_eval_time']:.3f}s"
            )
        timing = (
            f"nsrc={results['source_count']}, nobs={geometry.obs[0].size}, "
            f"plane={geometry.obs_grid[0].shape[0]}x{geometry.obs_grid[0].shape[1]}\n"
            f"theta={float(theta):.2f}\n"
            f"twist_pitch={float(twist_pitch):.3f}, helix_width={float(helix_width):.3f}, "
            f"bend_curvature={float(bend_curvature):.3f}, loop_fraction={float(loop_fraction):.2f}\n"
            f"original source-target interactions={results['source_target_interactions']:.1E}\n"
            f"direct:       construction={results['direct_build_time']:.3f}s, "
            f"evaluation={results['direct_time']:.3f}s\n"
            f"hierarchical: construction={results['build_time']:.3f}s, "
            f"evaluation={results['eval_time']:.3f}s"
            f"{self_text}"
        )
        return fig, self_fig, timing

    return app


if __name__ == "__main__":
    if not os.getenv("CFSEM_TESTING"):
        make_app().run(debug=True)
