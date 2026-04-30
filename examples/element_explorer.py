"""Interactive quadrilateral element interpolation and quadrature explorer."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from cfsem.solenoid_stress.fem2d import (
    infer_quad9_mesh,
    quad_mesh_interpolation_operator,
    query_quad_mesh,
)

TESTING = bool(os.getenv("CFSEM_TESTING"))
OUTPUT_HTML = Path(__file__).resolve().parents[1] / "docs/python/example_outputs/element_explorer.html"

ELEMENT_TYPES = ("quad4", "quad9")
QUADRATURES = {"gl3": 3, "gl4": 4}
CENTER_ELEMENT_INDEX = 4
SURFACE_Z_OFFSET = 0.9
SURFACE_VALUE_SCALE = 0.35
HISTORY_LENGTH = 5

LOCAL_NODE_LABELS = (
    "0 bottom-left",
    "1 bottom-right",
    "2 top-right",
    "3 top-left",
    "4 bottom mid",
    "5 right mid",
    "6 top mid",
    "7 left mid",
    "8 center",
)
DEFAULT_NODE_VALUES = np.array([0.0, 0.8, -0.1, 0.2, 1.1, 0.4, 0.7, -0.4, 1.6], dtype=np.float64)


def _export_docs_example_figure(fig) -> None:
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        OUTPUT_HTML,
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )


def _quad4_shape(xi: float, eta: float) -> np.ndarray:
    return 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=np.float64,
    )


def _q2_shape_1d(x: float) -> np.ndarray:
    return np.array([0.5 * x * (x - 1.0), 1.0 - x * x, 0.5 * x * (x + 1.0)], dtype=np.float64)


def _quad9_shape(xi: float, eta: float) -> np.ndarray:
    lx = _q2_shape_1d(xi)
    ly = _q2_shape_1d(eta)
    return np.array(
        [
            lx[0] * ly[0],
            lx[2] * ly[0],
            lx[2] * ly[2],
            lx[0] * ly[2],
            lx[1] * ly[0],
            lx[2] * ly[1],
            lx[1] * ly[2],
            lx[0] * ly[1],
            lx[1] * ly[1],
        ],
        dtype=np.float64,
    )


def _shape_functions(element_type: str, xi: float, eta: float) -> np.ndarray:
    return _quad4_shape(xi, eta) if element_type == "quad4" else _quad9_shape(xi, eta)


def _face_reference(local_face: int, s: float) -> tuple[float, float]:
    if local_face == 0:
        return s, -1.0
    if local_face == 1:
        return 1.0, s
    if local_face == 2:
        return -s, 1.0
    if local_face == 3:
        return -1.0, -s
    raise ValueError(f"unsupported local face {local_face}")


def _base_quad4_patch() -> tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(-1.5, 1.5, 4, dtype=np.float64)
    nodes = np.array([[x, y] for y in coords for x in coords], dtype=np.float64)

    def node_id(i: int, j: int) -> int:
        return j * 4 + i

    elements = []
    for j in range(3):
        for i in range(3):
            elements.append(
                [
                    node_id(i, j),
                    node_id(i + 1, j),
                    node_id(i + 1, j + 1),
                    node_id(i, j + 1),
                ]
            )
    return nodes, np.asarray(elements, dtype=np.uint64)


def _analysis_mesh(element_type: str) -> tuple[np.ndarray, np.ndarray]:
    nodes, elements = _base_quad4_patch()
    if element_type == "quad4":
        return nodes, elements
    elevated = infer_quad9_mesh(nodes, elements)
    return elevated.analysis_nodes, elevated.analysis_elements


def _configured_mesh(
    element_type: str,
    values: np.ndarray,
    offsets_x: np.ndarray,
    offsets_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes, elements = _analysis_mesh(element_type)
    nodes = np.array(nodes, dtype=np.float64, copy=True)
    nodal_values = np.zeros(nodes.shape[0], dtype=np.float64)
    central_nodes = np.asarray(elements[CENTER_ELEMENT_INDEX], dtype=np.int64)
    active_count = 4 if element_type == "quad4" else 9
    active_nodes = central_nodes[:active_count]
    nodes[active_nodes, 0] += offsets_x[:active_count]
    nodes[active_nodes, 1] += offsets_y[:active_count]
    nodal_values[active_nodes] = values[:active_count]
    return nodes, elements, nodal_values, active_nodes


def _map_reference_points(
    nodes: np.ndarray,
    element: np.ndarray,
    element_type: str,
    reference_points: np.ndarray,
) -> np.ndarray:
    """Map element-local reference points into the embedded 2D analysis mesh."""
    coords = nodes[np.asarray(element, dtype=np.int64)]
    mapped = np.zeros((reference_points.shape[0], 2), dtype=np.float64)
    for i, (xi, eta) in enumerate(reference_points):
        mapped[i] = _shape_functions(element_type, float(xi), float(eta)) @ coords
    return mapped


def _interpolation_values(
    nodes: np.ndarray,
    elements: np.ndarray,
    points: np.ndarray,
    nodal_values: np.ndarray,
    element_type: str,
) -> np.ndarray:
    """Evaluate nodal values at arbitrary points through the reusable mesh-query operator."""
    query = query_quad_mesh(nodes, elements, points, element_type=element_type)
    operator = quad_mesh_interpolation_operator(query)
    return np.asarray(operator @ nodal_values, dtype=np.float64)


def _surface_samples(
    nodes: np.ndarray,
    elements: np.ndarray,
    element_type: str,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample each element in reference space, then triangulate those samples for Plotly."""
    grid = np.linspace(-1.0, 1.0, int(resolution), dtype=np.float64)
    reference_grid = np.array([[xi, eta] for eta in grid for xi in grid], dtype=np.float64)
    points: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    ngrid = int(resolution)
    for element in elements:
        start = sum(part.shape[0] for part in points)
        points.append(_map_reference_points(nodes, element, element_type, reference_grid))
        for j in range(ngrid - 1):
            for i in range(ngrid - 1):
                n0 = start + j * ngrid + i
                n1 = n0 + 1
                n3 = n0 + ngrid
                n2 = n3 + 1
                triangles.append((n0, n1, n2))
                triangles.append((n0, n2, n3))
    return np.vstack(points), np.asarray(triangles, dtype=np.int64)


def _quadrature_points(
    nodes: np.ndarray,
    elements: np.ndarray,
    element_type: str,
    quadrature: str,
) -> np.ndarray:
    """Return tensor-product Gauss-Legendre point locations for every element."""
    points_1d, _weights_1d = np.polynomial.legendre.leggauss(QUADRATURES[quadrature])
    reference_points = np.array([[xi, eta] for eta in points_1d for xi in points_1d], dtype=np.float64)
    return np.vstack(
        [_map_reference_points(nodes, element, element_type, reference_points) for element in elements]
    )


def _central_quadrature_values(
    element_type: str,
    quadrature: str,
    values: np.ndarray,
    offsets_x: np.ndarray,
    offsets_y: np.ndarray,
) -> np.ndarray:
    """Interpolate the central element nodal values at its quadrature points."""
    nodes, elements, nodal_values, _active_nodes = _configured_mesh(
        element_type,
        values,
        offsets_x,
        offsets_y,
    )
    points_1d, _weights_1d = np.polynomial.legendre.leggauss(QUADRATURES[quadrature])
    reference_points = np.array([[xi, eta] for eta in points_1d for xi in points_1d], dtype=np.float64)
    central_points = _map_reference_points(
        nodes,
        elements[CENTER_ELEMENT_INDEX],
        element_type,
        reference_points,
    )
    return _interpolation_values(nodes, elements, central_points, nodal_values, element_type)


def _updated_history(
    history_data: dict[str, object] | None,
    element_type: str,
    quadrature: str,
    values: np.ndarray,
    x_node: int,
    x_value: float,
) -> dict[str, object]:
    """Append one central-element quadrature sample and keep the last few updates."""
    key = f"{element_type}:{quadrature}"
    samples: list[list[float]]
    x_values: list[float]
    if history_data is None or history_data.get("key") != key or history_data.get("x_node") != x_node:
        samples = []
        x_values = []
    else:
        samples = list(history_data.get("samples", []))
        x_values = [float(value) for value in history_data.get("x_values", [])]
    samples.append([float(value) for value in values])
    x_values.append(float(x_value))
    return {
        "key": key,
        "x_node": int(x_node),
        "x_values": x_values[-HISTORY_LENGTH:],
        "samples": samples[-HISTORY_LENGTH:],
    }


def _build_history_figure(history_data: dict[str, object] | None):
    import plotly.graph_objects as go

    samples_raw = [] if history_data is None else history_data.get("samples", [])
    x_values_raw = [] if history_data is None else history_data.get("x_values", [])
    x_node = 0 if history_data is None else int(history_data.get("x_node", 0))
    samples = np.asarray(samples_raw, dtype=np.float64)
    x_values = np.asarray(x_values_raw, dtype=np.float64)
    fig = go.Figure()
    if samples.ndim == 2 and samples.shape[0] > 0 and x_values.shape[0] == samples.shape[0]:
        delta_samples = samples - np.min(samples, axis=0)
        for point_index in range(samples.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=delta_samples[:, point_index],
                    mode="lines+markers",
                    name=f"q{point_index}",
                    hovertemplate="node value=%{x:.4f}<br>delta value=%{y:.4f}<extra></extra>",
                )
            )
    fig.update_layout(
        title=f"Central element quadrature value delta vs node {x_node}",
        margin={"l": 50, "r": 12, "t": 36, "b": 42},
        height=260,
        xaxis={"title": f"node {x_node} value"},
        yaxis={"title": "interpolated value delta<br>from history minimum"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
    )
    return fig


def _active_history_node(
    history_data: dict[str, object] | None,
    triggered_id: object,
    element_type: str,
) -> int:
    """Select the nodal value slider that should define the history x-axis."""
    active_count = 4 if element_type == "quad4" else 9
    if isinstance(triggered_id, str) and triggered_id.startswith("value-"):
        try:
            node = int(triggered_id.removeprefix("value-"))
        except ValueError:
            node = 0
    elif history_data is None:
        node = 0
    else:
        node = int(history_data.get("x_node", 0))
    return node if 0 <= node < active_count else 0


def _mesh_line_points(
    nodes: np.ndarray,
    elements: np.ndarray,
    element_type: str,
    *,
    element_indices: tuple[int, ...] | None = None,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Build edge polylines, using curved quad9 interpolation for midside geometry."""
    selected = range(elements.shape[0]) if element_indices is None else element_indices
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    edge_coord = np.linspace(-1.0, 1.0, 12, dtype=np.float64)
    for element_index in selected:
        element = elements[element_index]
        for local_face in range(4):
            reference_points = np.array([_face_reference(local_face, float(s)) for s in edge_coord])
            edge_points = _map_reference_points(nodes, element, element_type, reference_points)
            x.extend(edge_points[:, 0].tolist())
            y.extend(edge_points[:, 1].tolist())
            z.extend([0.0] * edge_points.shape[0])
            x.append(None)
            y.append(None)
            z.append(None)
    return x, y, z


def _build_figure(
    element_type: str,
    quadrature: str,
    resolution: int,
    values: np.ndarray,
    offsets_x: np.ndarray,
    offsets_y: np.ndarray,
):
    import plotly.graph_objects as go

    element_type = element_type if element_type in ELEMENT_TYPES else "quad4"
    quadrature = quadrature if quadrature in QUADRATURES else "gl3"
    nodes, elements, nodal_values, active_nodes = _configured_mesh(
        element_type,
        values,
        offsets_x,
        offsets_y,
    )
    surface_points, triangles = _surface_samples(nodes, elements, element_type, resolution)
    surface_values = _interpolation_values(nodes, elements, surface_points, nodal_values, element_type)
    surface_z = SURFACE_Z_OFFSET + SURFACE_VALUE_SCALE * surface_values
    quad_points = _quadrature_points(nodes, elements, element_type, quadrature)
    quad_values = _interpolation_values(nodes, elements, quad_points, nodal_values, element_type)
    quad_surface_z = SURFACE_Z_OFFSET + SURFACE_VALUE_SCALE * quad_values
    mesh_x, mesh_y, mesh_z = _mesh_line_points(nodes, elements, element_type)
    center_x, center_y, center_z = _mesh_line_points(
        nodes,
        elements,
        element_type,
        element_indices=(CENTER_ELEMENT_INDEX,),
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=mesh_x,
            y=mesh_y,
            z=mesh_z,
            mode="lines",
            line={"color": "rgba(80,80,80,0.55)", "width": 3},
            name="analysis mesh",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=center_x,
            y=center_y,
            z=center_z,
            mode="lines",
            line={"color": "#e26d2f", "width": 7},
            name="controlled element",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Mesh3d(
            x=surface_points[:, 0],
            y=surface_points[:, 1],
            z=surface_z,
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            intensity=surface_values,
            colorscale="Viridis",
            opacity=0.74,
            colorbar={"title": "value"},
            name="interpolated surface",
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>value=%{intensity:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=nodes[:, 0],
            y=nodes[:, 1],
            z=np.zeros(nodes.shape[0]),
            mode="markers",
            marker={
                "size": 4,
                "color": nodal_values,
                "colorscale": "Viridis",
                "line": {"color": "black", "width": 1},
            },
            name="mesh nodes",
            hovertemplate="node value=%{marker.color:.3f}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=nodes[active_nodes, 0],
            y=nodes[active_nodes, 1],
            z=np.zeros(active_nodes.size),
            mode="markers",
            marker={"size": 7, "color": "#e26d2f", "symbol": "circle-open", "line": {"width": 3}},
            name="controlled nodes",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=quad_points[:, 0],
            y=quad_points[:, 1],
            z=np.zeros(quad_points.shape[0]),
            mode="markers",
            marker={"size": 3, "color": "black", "symbol": "diamond"},
            name=f"{quadrature} points on mesh",
            hovertemplate="quadrature point<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=quad_points[:, 0],
            y=quad_points[:, 1],
            z=quad_surface_z,
            mode="markers",
            marker={"size": 4, "color": quad_values, "colorscale": "Viridis", "symbol": "diamond"},
            name=f"{quadrature} points on surface",
            hovertemplate="quadrature value=%{marker.color:.3f}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{element_type.upper()} interpolation with {quadrature.upper()} quadrature",
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
        height=760,
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.0, "xanchor": "left", "x": 0.0},
        scene={
            "xaxis": {"title": "x"},
            "yaxis": {"title": "y"},
            "zaxis": {"title": "analysis plane / interpolated value"},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.45},
            "camera": {"eye": {"x": 1.55, "y": -1.9, "z": 1.15}},
        },
    )
    return fig


def _slider(label: str, slider_id: str, minimum: float, maximum: float, value: float, step: float):
    from dash import dcc, html

    return html.Div(
        [
            html.Label(label, style={"fontSize": "0.78rem", "fontWeight": 600}),
            dcc.Slider(
                minimum,
                maximum,
                step=step,
                value=value,
                id=slider_id,
                tooltip={"placement": "bottom", "always_visible": False},
                marks=None,
            ),
        ],
        style={"minWidth": "160px"},
    )


def _node_control(local_node: int):
    from dash import html

    return html.Div(
        [
            html.Div(LOCAL_NODE_LABELS[local_node], style={"fontWeight": 700, "fontSize": "0.82rem"}),
            _slider("value", f"value-{local_node}", -2.0, 2.0, float(DEFAULT_NODE_VALUES[local_node]), 0.05),
            _slider("x", f"dx-{local_node}", -0.24, 0.24, 0.0, 0.02),
            _slider("y", f"dy-{local_node}", -0.24, 0.24, 0.0, 0.02),
        ],
        style={
            "border": "1px solid #d9d9d9",
            "borderRadius": "6px",
            "padding": "8px",
            "background": "#fbfbfb",
        },
    )


def create_app():
    """Create the Dash app for interactively comparing quad4/quad9 interpolation."""
    from dash import Dash, Input, Output, State, ctx, dcc, html

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H3(
                        "CFSEM Quadrilateral Element Explorer",
                        style={"margin": "0 0 14px 0", "fontSize": "1.1rem"},
                    ),
                    html.Div(
                        [
                            html.Label("Element", style={"fontWeight": 700}),
                            dcc.Dropdown(
                                id="element-type",
                                options=[{"label": item, "value": item} for item in ELEMENT_TYPES],
                                value="quad9",
                                clearable=False,
                            ),
                        ],
                        style={"marginBottom": "12px"},
                    ),
                    html.Div(
                        [
                            html.Label("Quadrature", style={"fontWeight": 700}),
                            dcc.Dropdown(
                                id="quadrature",
                                options=[{"label": item, "value": item} for item in QUADRATURES],
                                value="gl3",
                                clearable=False,
                            ),
                        ],
                        style={"marginBottom": "12px"},
                    ),
                    _slider("Surface resolution", "resolution", 5, 50, 25, 1),
                    html.Div(
                        [_node_control(i) for i in range(9)],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr",
                            "gap": "8px",
                            "marginTop": "12px",
                            "paddingBottom": "16px",
                        },
                    ),
                ],
                style={
                    "width": "340px",
                    "minWidth": "300px",
                    "maxWidth": "380px",
                    "height": "100vh",
                    "overflowY": "auto",
                    "padding": "16px",
                    "borderRight": "1px solid #d9d9d9",
                    "background": "#f7f7f7",
                    "boxSizing": "border-box",
                },
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="element-figure",
                        config={"responsive": True, "displaylogo": False},
                        style={"height": "calc(100vh - 260px)"},
                    ),
                    dcc.Graph(
                        id="quadrature-history",
                        config={"responsive": True, "displaylogo": False},
                        style={"height": "260px"},
                    ),
                    dcc.Store(id="quadrature-history-data"),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "flex": "1 1 auto",
                    "minWidth": 0,
                },
            ),
        ],
        style={
            "display": "flex",
            "height": "100vh",
            "fontFamily": "system-ui, sans-serif",
            "margin": 0,
        },
    )

    inputs = [
        Input("element-type", "value"),
        Input("quadrature", "value"),
        Input("resolution", "value"),
        *[Input(f"value-{i}", "value") for i in range(9)],
        *[Input(f"dx-{i}", "value") for i in range(9)],
        *[Input(f"dy-{i}", "value") for i in range(9)],
    ]

    @app.callback(
        Output("element-figure", "figure"),
        Output("quadrature-history", "figure"),
        Output("quadrature-history-data", "data"),
        inputs,
        State("quadrature-history-data", "data"),
    )
    def update_figure(element_type, quadrature, resolution, *node_args_and_history):
        history_data = node_args_and_history[-1]
        node_args = node_args_and_history[:-1]
        values = np.asarray(node_args[:9], dtype=np.float64)
        offsets_x = np.asarray(node_args[9:18], dtype=np.float64)
        offsets_y = np.asarray(node_args[18:27], dtype=np.float64)
        normalized_element_type = str(element_type)
        normalized_quadrature = str(quadrature)
        element_figure = _build_figure(
            normalized_element_type,
            normalized_quadrature,
            int(resolution),
            values,
            offsets_x,
            offsets_y,
        )
        quadrature_values = _central_quadrature_values(
            normalized_element_type,
            normalized_quadrature,
            values,
            offsets_x,
            offsets_y,
        )
        x_node = _active_history_node(history_data, ctx.triggered_id, normalized_element_type)
        updated_history = _updated_history(
            history_data,
            normalized_element_type,
            normalized_quadrature,
            quadrature_values,
            x_node,
            float(values[x_node]),
        )
        return element_figure, _build_history_figure(updated_history), updated_history

    return app


def main() -> None:
    """Run the Dash server, or export the static docs figure under CFSEM_TESTING."""
    if not TESTING:
        create_app().run(debug=True)
        return

    fig = _build_figure(
        "quad9",
        "gl3",
        11,
        DEFAULT_NODE_VALUES,
        np.zeros(9, dtype=np.float64),
        np.zeros(9, dtype=np.float64),
    )
    _export_docs_example_figure(fig)
    create_app()


if __name__ == "__main__":
    main()
