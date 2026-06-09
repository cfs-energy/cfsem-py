"""Convergence study comparing axisymmetric FEM and 1D FD solenoid stress solutions.

This example intentionally configures the 2D FEM model to mimic the 1D solver:

- the reduced `cfsem_radial_material()` constitutive law is used,
- the load is purely radial (`f_r = J_theta B_z`, `f_z = 0`),
- all axial displacement DOFs are fixed to zero to suppress any axial/shear mode.

With that setup, the only meaningful difference between the models is spatial
discretization. The load uses a linear `B_z(r)` profile so the radial and hoop
stresses can also be compared against the analytic long-solenoid formula from
Iwasa. The script sweeps radial resolution, compares both models against that
analytic stress truth at the FEM midplane sample locations, prints a
figure-of-merit table, and plots both the finest-resolution stress profiles and
the convergence curves.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

if os.getenv("CFSEM_TESTING"):
    import matplotlib

    matplotlib.use("Agg")

from matplotlib import pyplot as plt

from cfsem.solenoid_stress.fem2d import (
    assemble_structural_2d,
    cfsem_radial_material,
    infer_quad9_mesh,
    quad_mesh_interpolation_operator,
    quad_mesh_stress_operator,
    query_quad_mesh,
)
from cfsem.solenoid_stress.solenoid_handcalc import s_long_solenoid
from cfsem.solenoid_stress.solenoid_1d import (
    SolenoidStress1D,
    solenoid_1d_structural_factor,
    solenoid_1d_structural_rhs,
)

TESTING = bool(os.getenv("CFSEM_TESTING"))

RI = 0.5  # [m]
RO = 1.0  # [m]
HEIGHT = 0.1  # [m]
ELASTICITY_MODULUS = 200.0e9  # [Pa]
POISSON_RATIO = 0.27  # [-]
CURRENT_DENSITY = 0.2 * 390.0e6  # [A/m^2]
BZ_INNER = 27.0  # [T]
QUADRATURE = "gl3"
ELEMENT_TYPES = ("quad4", "quad9")
DISCRETIZATION_PANEL_ELEMENT_TYPE = "quad9"
NUDGE = 1.0e-6  # [m]
LEGEND_RIGHT_PAD_POINTS = 50.0
REPRESENTATIVE_DISCRETIZATION_NZ = 1
REPRESENTATIVE_DISCRETIZATION_NR = max(3, int(round((RO - RI) / HEIGHT)))
TARGET_DR_SWEEP_MM = np.array([50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 1.0], dtype=np.float64)
NR_SWEEP = np.asarray(np.ceil(1.0e3 * (RO - RI) / TARGET_DR_SWEEP_MM), dtype=int)
MIN_EXPECTED_FEM_STRESS_CONVERGENCE_RATE = 1.8


@dataclass(frozen=True, slots=True)
class Profile:
    radius: np.ndarray
    u_r: np.ndarray
    s_rr: np.ndarray
    s_tt: np.ndarray


@dataclass(frozen=True, slots=True)
class SweepResult:
    element_type: str
    nr: int
    dr_mm: float
    ndof: int
    fem_build_seconds: float
    fem_factorize_seconds: float
    fem_solve_seconds: float
    fd_profile: Profile
    fem_profile: Profile
    analytic_profile: Profile
    fd_errors_pct: dict[str, float]
    fem_errors_pct: dict[str, float]
    parity_errors_pct: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Save the figure without opening a Matplotlib window.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".png"),
        help="Path for the saved PNG figure.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def build_annulus_strip_mesh(
    ri: float,
    ro: float,
    height: float,
    nr: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray]:
    radii = np.linspace(ri, ro, nr + 1, dtype=np.float64)
    zs = np.linspace(0.0, height, nz + 1, dtype=np.float64)
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
    return nodes, np.asarray(elements, dtype=np.uint64)


def prescribed_z_dofs(node_count: int) -> dict[int, float]:
    return {2 * node + 1: 0.0 for node in range(node_count)}


def linear_bz_profile(radius: np.ndarray) -> np.ndarray:
    radius_arr = np.asarray(radius, dtype=np.float64)
    taper = 1.0 - (radius_arr - RI) / (RO - RI)
    return BZ_INNER * np.clip(taper, 0.0, 1.0)


def analytic_stress_profile(sample_r: np.ndarray) -> Profile:
    s_rr, s_tt = s_long_solenoid(
        np.asarray(sample_r, dtype=np.float64),
        RI,
        RO,
        CURRENT_DENSITY,
        BZ_INNER,
        0.0,
        POISSON_RATIO,
    )
    return Profile(
        radius=np.asarray(sample_r, dtype=np.float64),
        u_r=np.full_like(sample_r, np.nan, dtype=np.float64),
        s_rr=np.asarray(s_rr, dtype=np.float64),
        s_tt=np.asarray(s_tt, dtype=np.float64),
    )


def recover_axisymmetric_midplane_profile(
    nodes: np.ndarray,
    elements: np.ndarray,
    sample_radius: np.ndarray,
    displacement: np.ndarray,
    material: np.ndarray,
    element_type: str,
) -> Profile:
    points = np.column_stack([sample_radius, np.full_like(sample_radius, 0.5 * HEIGHT)])
    query = query_quad_mesh(nodes, elements, points, element_type=element_type)
    interpolation_operator = quad_mesh_interpolation_operator(query)
    stress_operator = quad_mesh_stress_operator(
        query,
        np.zeros(elements.shape[0], dtype=np.uint64),
        np.asarray([material], dtype=np.float64),
        formulation="axisymmetric",
    )
    displacement_2d = np.asarray(displacement, dtype=np.float64).reshape(nodes.shape[0], 2)
    displacement_at_points = np.asarray(interpolation_operator @ displacement_2d, dtype=np.float64)
    stress = np.asarray(
        stress_operator @ displacement_2d.reshape(-1),
        dtype=np.float64,
    ).reshape(-1, 4)
    return Profile(
        radius=np.asarray(sample_radius, dtype=np.float64),
        u_r=displacement_at_points[:, 0],
        s_rr=stress[:, 0],
        s_tt=stress[:, 2],
    )


def build_1d_grid(spacing: float) -> np.ndarray:
    n = int(np.ceil((RO - RI) / spacing)) + 1
    r_interior = np.linspace(RI, RO, n, dtype=np.float64)
    return np.concatenate([[RI - NUDGE], r_interior, [RO + NUDGE]])


def solve_1d_field(rgrid: np.ndarray) -> Profile:
    j = np.full_like(rgrid, CURRENT_DENSITY)
    bz = np.concatenate(
        [
            np.array([BZ_INNER], dtype=np.float64),
            linear_bz_profile(rgrid[1:-1]),
            np.array([0.0], dtype=np.float64),
        ]
    )
    c_struct = solenoid_1d_structural_factor(ELASTICITY_MODULUS, POISSON_RATIO)
    rhs = solenoid_1d_structural_rhs(c_struct, j, bz)
    model = SolenoidStress1D(
        rgrid=rgrid,
        elasticity_modulus=ELASTICITY_MODULUS,
        poisson_ratio=POISSON_RATIO,
        order=4,
        direct_inverse=False,
    )
    u_r = np.asarray(model.displacement_solver(rhs), dtype=np.float64).reshape(-1)
    strain = np.asarray(model.operators.a_eu @ u_r, dtype=np.float64).reshape(-1)
    stress = np.asarray(model.operators.a_se @ strain, dtype=np.float64).reshape(-1)
    n = rgrid.size
    return Profile(
        radius=np.asarray(rgrid[1:-1], dtype=np.float64),
        u_r=np.asarray(u_r[1:-1], dtype=np.float64),
        s_rr=np.asarray(stress[:n][1:-1], dtype=np.float64),
        s_tt=np.asarray(stress[n:][1:-1], dtype=np.float64),
    )


def sample_profile(profile: Profile, sample_r: np.ndarray) -> Profile:
    return Profile(
        radius=np.asarray(sample_r, dtype=np.float64),
        u_r=np.interp(sample_r, profile.radius, profile.u_r),
        s_rr=np.interp(sample_r, profile.radius, profile.s_rr),
        s_tt=np.interp(sample_r, profile.radius, profile.s_tt),
    )


def max_normalized_error_percent(values: np.ndarray, reference: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(values))), float(np.max(np.abs(reference))), 1.0e-30)
    return 100.0 * float(np.max(np.abs(values - reference))) / scale


def solve_fem_midplane_profile(
    nr: int,
    element_type: str,
) -> tuple[Profile, int, float, float, float]:
    build_start = perf_counter()
    nodes, elements = build_annulus_strip_mesh(RI, RO, HEIGHT, nr=nr, nz=1)
    elevated = infer_quad9_mesh(nodes, elements) if element_type == "quad9" else None
    analysis_nodes = elevated.analysis_nodes if elevated is not None else nodes
    analysis_elements = elevated.analysis_elements if elevated is not None else elements
    nelem = elements.shape[0]
    material = cfsem_radial_material(ELASTICITY_MODULUS, POISSON_RATIO)
    prescribed = prescribed_z_dofs(analysis_nodes.shape[0])
    model = assemble_structural_2d(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(nelem, dtype=np.uint64),
        material_table=np.asarray([material]),
        prescribed=prescribed,
        quadrature=QUADRATURE,
        element_type=element_type,
    )
    quadrature_data = model.element_quadrature()
    points = quadrature_data.points.reshape(-1, 2)
    nq = quadrature_data.nq_per_element
    weights = np.asarray(quadrature_data.weights_volume, dtype=np.float64)
    bz_weighted = linear_bz_profile(points[:, 0]).reshape(nelem, nq) * weights
    bz_mean = np.sum(bz_weighted, axis=1) / np.sum(weights, axis=1)
    body_force = np.column_stack((CURRENT_DENSITY * bz_mean, np.zeros(nelem, dtype=np.float64)))
    rhs = model.build_rhs(body_force=body_force)
    fem_build_seconds = perf_counter() - build_start

    factorize_start = perf_counter()
    _ = model.solve(np.zeros_like(rhs))
    fem_factorize_seconds = perf_counter() - factorize_start

    solve_start = perf_counter()
    displacement = model.solve(rhs).reshape(analysis_nodes.shape[0], 2)
    fem_solve_seconds = perf_counter() - solve_start

    return (
        recover_axisymmetric_midplane_profile(
            analysis_nodes,
            analysis_elements,
            0.5 * (nodes[:nr, 0] + nodes[1 : nr + 1, 0]),
            displacement,
            material,
            element_type,
        ),
        model.ndof,
        fem_build_seconds,
        fem_factorize_seconds,
        fem_solve_seconds,
    )


def run_study() -> dict[str, list[SweepResult]]:
    results_by_type: dict[str, list[SweepResult]] = {element_type: [] for element_type in ELEMENT_TYPES}

    for element_type in ELEMENT_TYPES:
        for nr in NR_SWEEP:
            fem_profile, ndof, fem_build_seconds, fem_factorize_seconds, fem_solve_seconds = (
                solve_fem_midplane_profile(int(nr), element_type)
            )
            coarse_1d = solve_1d_field(build_1d_grid((RO - RI) / int(nr)))
            fd_profile = sample_profile(coarse_1d, fem_profile.radius)
            analytic_profile = analytic_stress_profile(fem_profile.radius)

            fd_errors_pct = {
                "s_rr": max_normalized_error_percent(fd_profile.s_rr, analytic_profile.s_rr),
                "s_tt": max_normalized_error_percent(fd_profile.s_tt, analytic_profile.s_tt),
            }
            fem_errors_pct = {
                "s_rr": max_normalized_error_percent(fem_profile.s_rr, analytic_profile.s_rr),
                "s_tt": max_normalized_error_percent(fem_profile.s_tt, analytic_profile.s_tt),
            }
            parity_errors_pct = {
                "u_r": max_normalized_error_percent(fem_profile.u_r, fd_profile.u_r),
                "s_rr": max_normalized_error_percent(fem_profile.s_rr, fd_profile.s_rr),
                "s_tt": max_normalized_error_percent(fem_profile.s_tt, fd_profile.s_tt),
            }

            results_by_type[element_type].append(
                SweepResult(
                    element_type=element_type,
                    nr=int(nr),
                    dr_mm=1.0e3 * (RO - RI) / int(nr),
                    ndof=ndof,
                    fem_build_seconds=fem_build_seconds,
                    fem_factorize_seconds=fem_factorize_seconds,
                    fem_solve_seconds=fem_solve_seconds,
                    fd_profile=fd_profile,
                    fem_profile=fem_profile,
                    analytic_profile=analytic_profile,
                    fd_errors_pct=fd_errors_pct,
                    fem_errors_pct=fem_errors_pct,
                    parity_errors_pct=parity_errors_pct,
                )
            )

    return results_by_type


def plot_discretization_panel(ax, nr: int, nz: int, element_type: str) -> None:
    nodes, elements = build_annulus_strip_mesh(RI, RO, HEIGHT, nr=nr, nz=nz)
    elevated = infer_quad9_mesh(nodes, elements) if element_type == "quad9" else None
    analysis_nodes = elevated.analysis_nodes if elevated is not None else nodes
    analysis_elements = elevated.analysis_elements if elevated is not None else elements
    model = assemble_structural_2d(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([cfsem_radial_material(ELASTICITY_MODULUS, POISSON_RATIO)]),
        prescribed=prescribed_z_dofs(analysis_nodes.shape[0]),
        quadrature=QUADRATURE,
        element_type=element_type,
    )
    quadrature_data = model.element_quadrature()
    quadrature_points = quadrature_data.points.reshape(-1, 2)
    fd_grid = build_1d_grid((RO - RI) / nr)[1:-1]

    if quadrature_points.shape[0] <= 1_000:
        quadrature_marker_size = 10.0
    elif quadrature_points.shape[0] <= 20_000:
        quadrature_marker_size = 4.0
    else:
        quadrature_marker_size = 1.5
    fd_marker_size = 28.0 if fd_grid.size <= 200 else 10.0 if fd_grid.size <= 2_000 else 4.0

    for conn in analysis_elements:
        coords = analysis_nodes[conn]
        edge_cycles = ((0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0))
        for i0, im, i1 in edge_cycles:
            ax.plot(
                coords[[i0, im, i1], 0],
                coords[[i0, im, i1], 1],
                color="0.78",
                linewidth=0.9,
                alpha=0.9,
            )
    ax.plot([], [], color="0.78", linewidth=0.9, label=f"FEM mesh ({element_type})")
    ax.scatter(
        quadrature_points[:, 0],
        quadrature_points[:, 1],
        s=quadrature_marker_size,
        color="tab:red",
        alpha=0.85,
        label=f"FEM quadrature points ({element_type}, {QUADRATURE})",
        rasterized=quadrature_points.shape[0] > 5_000,
    )
    ax.scatter(
        fd_grid,
        np.full_like(fd_grid, 0.5 * HEIGHT),
        s=fd_marker_size,
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=0.7,
        marker="o",
        label="1D FD physical grid",
        rasterized=fd_grid.size > 5_000,
    )
    ax.set_title(f"Representative matched-grid discretization ({element_type}, nr={nr}, nz={nz})")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_xlim(RI - 0.01 * (RO - RI), RO + 0.01 * (RO - RI))
    ax.set_ylim(-0.05 * HEIGHT, 1.05 * HEIGHT)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.legend(loc="upper right", frameon=True)


def build_figure(results_by_type: dict[str, list[SweepResult]]):
    fig = plt.figure(figsize=(14.5, 10.0))
    right_pad_fraction = LEGEND_RIGHT_PAD_POINTS / (72.0 * fig.get_size_inches()[0])
    layout_right = max(0.0, 0.86 - right_pad_fraction)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.8])
    profile_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    error_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    discretization_ax = fig.add_subplot(grid[2, :])
    finest_results = {element_type: results[-1] for element_type, results in results_by_type.items()}
    finest_reference = finest_results["quad9"]
    xconv = np.array([result.dr_mm for result in results_by_type["quad4"]], dtype=np.float64)
    element_plot_specs = {
        "quad4": {"color": "tab:red", "marker": "s"},
        "quad9": {"color": "tab:green", "marker": "o"},
    }

    profile_specs = [
        ("s_rr", "Radial stress $s_{rr}$ [Pa]"),
        ("s_tt", "Hoop stress $s_{tt}$ [Pa]"),
    ]
    for ax, (field_name, title) in zip(profile_axes, profile_specs, strict=True):
        ax.plot(
            finest_reference.analytic_profile.radius,
            getattr(finest_reference.analytic_profile, field_name),
            color="black",
            linewidth=1.8,
            label="Analytic long-solenoid stress",
        )
        ax.plot(
            finest_reference.fd_profile.radius,
            getattr(finest_reference.fd_profile, field_name),
            color="tab:blue",
            linestyle="--",
            linewidth=1.4,
            label=f"1D matched grid (nr={finest_reference.nr})",
        )
        for element_type in ELEMENT_TYPES:
            finest = finest_results[element_type]
            spec = element_plot_specs[element_type]
            ax.plot(
                finest.fem_profile.radius,
                getattr(finest.fem_profile, field_name),
                color=spec["color"],
                marker=spec["marker"],
                markersize=3.2,
                linewidth=1.2,
                label=f"{element_type.upper()} FEM midplane (nr={finest.nr})",
            )
        ax.set_title(title)
        ax.set_xlabel("r [m]")
        ax.grid(True, linestyle=":", linewidth=0.7)

    error_specs = [
        ("s_rr", "Max normalized error in $s_{rr}$ vs. analytic [%]"),
        ("s_tt", "Max normalized error in $s_{tt}$ vs. analytic [%]"),
    ]
    for ax, (field_name, title) in zip(error_axes, error_specs, strict=True):
        ax.loglog(
            xconv,
            [result.fd_errors_pct[field_name] for result in results_by_type["quad4"]],
            color="tab:blue",
            marker="s",
            linewidth=1.4,
            label="1D FD vs. analytic",
        )
        for element_type in ELEMENT_TYPES:
            spec = element_plot_specs[element_type]
            y_error = np.array(
                [result.fem_errors_pct[field_name] for result in results_by_type[element_type]],
                dtype=np.float64,
            )
            x_fit = xconv
            y_fit = y_error
            slope, intercept = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
            ax.loglog(
                xconv,
                y_error,
                color=spec["color"],
                marker=spec["marker"],
                linewidth=1.4,
                label=f"{element_type.upper()} FEM vs. analytic",
            )
            ax.loglog(
                x_fit,
                np.exp(intercept) * x_fit**slope,
                color=spec["color"],
                linestyle=":",
                linewidth=1.1,
                alpha=0.95,
                label=f"{element_type.upper()} fit $O(h^{{{slope:.2f}}})$",
            )
        ax.set_title(title)
        ax.set_xlabel("Radial element size [mm]")
        ax.grid(True, which="both", linestyle=":", linewidth=0.7)
        ax.invert_xaxis()

    plot_discretization_panel(
        discretization_ax,
        REPRESENTATIVE_DISCRETIZATION_NR,
        REPRESENTATIVE_DISCRETIZATION_NZ,
        DISCRETIZATION_PANEL_ELEMENT_TYPE,
    )

    handles, labels = profile_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(0.985, 0.70),
        bbox_transform=fig.transFigure,
        ncol=1,
        frameon=True,
    )
    fig.suptitle(
        "Axisymmetric FEM vs. 1D FD convergence study\n"
        "Analytic truth: linear-$B_z$ long-solenoid stress; parity setup: reduced radial material, "
        f"z-DOFs fixed, radial body force only, QUAD4/QUAD9 + {QUADRATURE} Gauss",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, layout_right, 0.94])
    return fig


def print_results(results_by_type: dict[str, list[SweepResult]]) -> None:
    print(
        "Study configuration: "
        f"ri={RI:.3f} m, ro={RO:.3f} m, height={HEIGHT:.3f} m, "
        f"E={ELASTICITY_MODULUS / 1.0e9:.1f} GPa, nu={POISSON_RATIO:.3f}, "
        f"J_theta={CURRENT_DENSITY:.3e} A/m^2, "
        f"Bz(ri)={BZ_INNER:.1f} T, Bz(ro)=0.0 T, "
        f"element_types={','.join(ELEMENT_TYPES)}, quadrature={QUADRATURE}"
    )
    for element_type in ELEMENT_TYPES:
        results = results_by_type[element_type]
        print()
        print(f"Element family: {element_type}")
        print(
            "Columns: nr, dr_mm, ndof, fem_build_ms, fem_factorize_ms, fem_solve_ms, "
            "1D-vs-analytic[s_rr,s_tt] %, FEM-vs-analytic[s_rr,s_tt] %, FEM-vs-1D[u_r,s_rr,s_tt] %"
        )
        for result in results:
            print(
                f"nr={result.nr:5d}, dr={result.dr_mm:8.4f} mm, ndof={result.ndof:6d}, "
                f"build={1.0e3 * result.fem_build_seconds:8.4f} ms, "
                f"factorize={1.0e3 * result.fem_factorize_seconds:8.4f} ms, "
                f"solve={1.0e3 * result.fem_solve_seconds:8.4f} ms | "
                f"1D=({result.fd_errors_pct['s_rr']:8.4f}, "
                f"{result.fd_errors_pct['s_tt']:8.4f}) | "
                f"FEM=({result.fem_errors_pct['s_rr']:8.4f}, "
                f"{result.fem_errors_pct['s_tt']:8.4f}) | "
                f"parity=({result.parity_errors_pct['u_r']:8.4f}, "
                f"{result.parity_errors_pct['s_rr']:8.4f}, "
                f"{result.parity_errors_pct['s_tt']:8.4f})"
            )

        finest = results[-1]
        fem_fom = max(finest.fem_errors_pct.values())
        fd_fom = max(finest.fd_errors_pct.values())
        parity_fom = max(finest.parity_errors_pct.values())
        print(
            f"Primary figure-of-merit: finest FEM max normalized midplane stress error "
            f"vs. analytic = {fem_fom:.6f} %"
        )
        print(f"Finest 1D FD max normalized midplane stress error " f"vs. analytic = {fd_fom:.6f} %")
        print(
            f"Finest matched-grid parity max normalized midplane error " f"(FEM vs. 1D) = {parity_fom:.6f} %"
        )
        print(f"Finest FEM system build time = {1.0e3 * finest.fem_build_seconds:.4f} ms")
        print(f"Finest FEM factorization time = {1.0e3 * finest.fem_factorize_seconds:.4f} ms")
        print(f"Finest FEM linear solve time = {1.0e3 * finest.fem_solve_seconds:.4f} ms")
        dr_mm = np.array([result.dr_mm for result in results], dtype=np.float64)
        stress_fom_error = np.maximum(
            np.array([result.fem_errors_pct["s_rr"] for result in results], dtype=np.float64),
            np.array([result.fem_errors_pct["s_tt"] for result in results], dtype=np.float64),
        )
        x_fit = dr_mm
        y_fit = stress_fom_error
        slope, _ = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
        assert slope >= MIN_EXPECTED_FEM_STRESS_CONVERGENCE_RATE, (
            f"{element_type} FEM stress convergence fit dropped below the expected polynomial rate: "
            f"fitted slope={slope:.3f}, required>={MIN_EXPECTED_FEM_STRESS_CONVERGENCE_RATE:.3f}"
        )
        print("Fitted FEM convergence rate over the full radial sweep: " f"stress FOM~O(h^{slope:.3f})")


def main() -> None:
    args = parse_args()
    results_by_type = run_study()
    print_results(results_by_type)
    fig = build_figure(results_by_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    print(f"Saved plot to {args.output}")
    if not args.no_plot and not TESTING:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
