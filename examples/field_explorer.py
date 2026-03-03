from __future__ import annotations

import os
import time
from functools import lru_cache

import numpy as np

import cfsem

GRID_SIZE = 30 if os.getenv("CFSEM_TESTING") else 1000
DEFAULT_WIRE_RADIUS = 0.02
PATH_RADIUS = 0.7
DOMAIN = 1.0
CURRENT = 1.0


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray]:
    vertices = rotate_vertices_y(build_path_vertices(n_sides), rotation_deg)
    if n_sides >= 3:
        starts = vertices
        ends = np.roll(vertices, -1, axis=0)
    else:
        starts = vertices[:-1]
        ends = vertices[1:]

    dl = ends - starts
    xyzfil = (starts[:, 0], starts[:, 1], starts[:, 2])
    dlxyzfil = (dl[:, 0], dl[:, 1], dl[:, 2])
    ifil = np.full(starts.shape[0], CURRENT)
    return vertices, starts, ends, xyzfil, dlxyzfil, ifil


def discretize_point_segments(
    starts: np.ndarray,
    dl: np.ndarray,
    current: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray]:
    n_filaments = starts.shape[0]
    nseg_per_filament = max(6, min(120, 900 // max(1, n_filaments)))

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


def segment_distance_map(xx: np.ndarray, zz: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    dist = np.full_like(xx, np.inf, dtype=float)
    for start, end in zip(starts, ends, strict=True):
        vx = end[0] - start[0]
        vz = end[2] - start[2]
        denom = vx * vx + vz * vz
        if denom == 0.0:
            continue
        t = ((xx - start[0]) * vx + (zz - start[2]) * vz) / denom
        t = np.clip(t, 0.0, 1.0)
        projx = start[0] + t * vx
        projz = start[2] + t * vz
        dist = np.minimum(dist, np.sqrt((xx - projx) ** 2 + (zz - projz) ** 2))
    return dist


@lru_cache(maxsize=8)
def compute_field(
    mode: str, n_sides: int, wire_radius: float, rotation_deg: float
) -> dict[str, np.ndarray | float]:
    vertices, starts, ends, xyzfil, dlxyzfil, ifil = build_linear_filaments(n_sides, rotation_deg)
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

    xyzfil_ps, dlxyzfil_ps, ifil_ps = discretize_point_segments(starts, dl, ifil)

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
    near_wire = segment_distance_map(xx, zz, starts, ends) < wire_radius
    err = np.where(near_wire, np.nan, err)

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


def build_figure(
    mode: str,
    n_sides: int,
    wire_radius: float,
    rotation_deg: float,
    show_filament_line: bool,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = compute_field(mode, n_sides, wire_radius, rotation_deg)
    x = data["x"]
    z = data["z"]
    mag_linear = data["mag_linear"]
    mag_point = data["mag_point"]
    err = data["err"]
    path_x = data["path_x"]
    path_z = data["path_z"]

    mag_log10 = np.log10(mag_linear + 1e-30)
    err_log10 = np.where(np.isnan(err), np.nan, np.log10(err + 1e-30))
    mid = GRID_SIZE // 2

    value_title = "|B| [T]" if mode == "b" else "|A| [T m]"
    title_prefix = "B-field" if mode == "b" else "Vector Potential"

    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.25,
        vertical_spacing=0.22,
        subplot_titles=[
            f"{title_prefix} magnitude (log10)",
            "Slice along x (z = 0)",
            "Point-segment error (log10)",
            "Slice along z (x = 0)",
        ],
    )
    fig.add_trace(
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
        fig.add_trace(
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
    fig.add_trace(
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
    fig.add_trace(
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
    fig.add_trace(
        go.Heatmap(
            x=x,
            y=z,
            z=err_log10,
            colorscale="Viridis",
            colorbar={
                "title": f"log10(delta {value_title})",
                "thickness": 14,
                "x": 0.6,
                "xanchor": "right",
            },
            zmin=np.nanmin(err_log10),
            zmax=np.nanmax(err_log10),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=z,
            y=mag_linear[:, mid],
            mode="lines",
            line={"color": "black", "width": 2},
            name="Linear filament (z-slice)",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=z,
            y=mag_point[:, mid],
            mode="lines",
            line={"color": "deepskyblue", "width": 2, "dash": "dash"},
            name="Point segment (z-slice)",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="x [m]", row=1, col=1)
    fig.update_yaxes(title_text="z [m]", row=1, col=1, scaleanchor="x", scaleratio=1.0)
    fig.update_xaxes(title_text="x [m]", row=1, col=2)
    fig.update_yaxes(title_text=value_title, row=1, col=2)
    fig.update_xaxes(title_text="x [m]", row=2, col=1)
    fig.update_yaxes(title_text="z [m]", row=2, col=1, scaleanchor="x3", scaleratio=1.0)
    fig.update_xaxes(title_text="z [m]", row=2, col=2)
    fig.update_yaxes(title_text=value_title, row=2, col=2)

    geometry_label = (
        "Straight line"
        if n_sides == 1
        else ("Two-segment path" if n_sides == 2 else f"{n_sides}-sided polygon")
    )
    fig.update_layout(
        height=920,
        title=f"{title_prefix}: {geometry_label}, rotation {rotation_deg:.0f} deg",
        margin={"l": 50, "r": 20, "t": 130, "b": 60},
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


def build_perf_summary(mode: str, n_sides: int, wire_radius: float, rotation_deg: float) -> str:
    data = compute_field(mode, n_sides, wire_radius, rotation_deg)
    label = "B-field" if mode == "b" else "Vector potential"
    return (
        f"{label} | wire radius: {wire_radius:.3f} m | rotation: {rotation_deg:.0f} deg | "
        f"linear: {data['t_linear']:.3f}s / {data['n_linear']:.2e} interactions, "
        f"point-segment: {data['t_point']:.3f}s / {data['n_point']:.2e} interactions"
    )


def create_app():
    from dash import Dash, Input, Output, dcc, html

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H3("CFSEM Biot-Savart and Vector Potential"),
            html.P("Use the slider to set geometry: 1 is a straight line, 3-50 are closed polygons."),
            html.Div(
                dcc.Slider(
                    id="polygon-sides",
                    min=1,
                    max=50,
                    step=1,
                    value=3,
                    marks={1: "1", 10: "10", 20: "20", 30: "30", 40: "40", 50: "50"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                style={"paddingBottom": "0.5rem"},
            ),
            html.P("Wire radius [m]", style={"marginTop": "0.75rem", "marginBottom": "0.25rem"}),
            html.Div(
                dcc.Slider(
                    id="wire-radius",
                    min=0.0,
                    max=0.1,
                    step=0.001,
                    value=DEFAULT_WIRE_RADIUS,
                    marks={0.0: "0.00", 0.02: "0.02", 0.05: "0.05", 0.08: "0.08", 0.1: "0.10"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                style={"paddingBottom": "0.5rem"},
            ),
            html.P("Rotation [deg]", style={"marginTop": "0.5rem", "marginBottom": "0.25rem"}),
            html.Div(
                dcc.Slider(
                    id="rotation-deg",
                    min=0,
                    max=360,
                    step=1,
                    value=0,
                    marks={0: "0", 90: "90", 180: "180", 270: "270", 360: "360"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                style={"paddingBottom": "0.5rem"},
            ),
            html.Div(
                id="perf-summary",
                style={"marginTop": "1.5rem", "marginBottom": "1.0rem", "fontFamily": "monospace"},
            ),
            dcc.Checklist(
                id="show-filament-line",
                options=[{"label": "Show filament line", "value": "show"}],
                value=["show"],
                style={"marginBottom": "0.75rem"},
            ),
            dcc.Tabs(
                id="field-tab",
                value="b",
                children=[
                    dcc.Tab(label="B-field", value="b"),
                    dcc.Tab(label="Vector potential", value="a"),
                ],
            ),
            dcc.Graph(id="field-figure"),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "1rem"},
    )

    @app.callback(
        Output("field-figure", "figure"),
        Output("perf-summary", "children"),
        Input("polygon-sides", "value"),
        Input("wire-radius", "value"),
        Input("rotation-deg", "value"),
        Input("show-filament-line", "value"),
        Input("field-tab", "value"),
    )
    def update_figure(
        n_sides: int,
        wire_radius: float,
        rotation_deg: float,
        show_filament_line: list[str],
        field_tab: str,
    ):
        sides = int(n_sides)
        radius = float(np.clip(wire_radius, 0.0, 0.1))
        rotation = float(np.mod(rotation_deg, 360.0))
        show_line = "show" in show_filament_line
        return (
            build_figure(field_tab, sides, radius, rotation, show_line),
            build_perf_summary(field_tab, sides, radius, rotation),
        )

    return app


def main() -> None:
    if os.getenv("CFSEM_TESTING"):
        try:
            build_figure("b", 3, DEFAULT_WIRE_RADIUS, 0.0, True)
            build_figure("a", 3, DEFAULT_WIRE_RADIUS, 0.0, True)
        except ModuleNotFoundError:
            return
        return

    app = create_app()
    app.run(debug=True)


if __name__ == "__main__":
    main()
