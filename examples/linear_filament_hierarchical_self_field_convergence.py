"""Self-field convergence study for hierarchical linear-filament loop fields.

This example compares the hierarchical single-source-tree linear-filament
`B`-field solve against the serial direct linear-filament solve for the
self-field of a circular current loop. The targets are the segment centers, so
the study exercises the same source-to-source pattern used when estimating
self-field error in coil discretizations.

Two segment-length targets are shown:

- 10 mm, representing a coarse filamentization,
- 1 mm, representing a finer filamentization.

For each discretization, the direct solution is computed once and used as the
reference while the hierarchical opening angle `theta` is swept from 1.0 down
to 0.01. A second plot row changes the loop discretization to sweep dense
interaction counts from about 1e5 to 1e10 and records hierarchical build and
evaluation time at several fixed theta values.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

if os.getenv("CFSEM_TESTING"):
    import matplotlib

    matplotlib.use("Agg")

from matplotlib import pyplot as plt

import cfsem

TESTING = bool(os.getenv("CFSEM_TESTING"))

LOOP_RADIUS = 1.0  # [m]
CURRENT = 1.0  # [A]
WIRE_RADIUS = 1.0e-3  # [m]
TARGET_SEGMENT_LENGTHS = (0.01, 0.001)  # [m]
THETA_SWEEP = np.logspace(0.0, -2.0, 10, dtype=np.float64)
SCALING_THETAS = (0.05, 0.3, 0.6)
SCALING_INTERACTION_TARGETS = np.logspace(5.0, 10.0, 6, dtype=np.float64)
TESTING_SCALING_INTERACTION_TARGETS = np.logspace(5.0, 6.0, 2, dtype=np.float64)
DIRECT_SCALING_MAX_INTERACTIONS = 1.0e9
DIRECT_TRACE_COLOR = "tab:green"
CONSTRUCTION_METHOD = "recursive"


@dataclass(frozen=True, slots=True)
class LoopDiscretization:
    """Piecewise-linear representation of one circular current loop."""

    target_ds: float
    actual_ds: float
    starts: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    deltas: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    centers: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    current: NDArray[np.float64]
    wire_radius: NDArray[np.float64]

    @property
    def segment_count(self) -> int:
        """Number of straight filament segments in the loop."""

        return self.current.size

    @property
    def interaction_count(self) -> int:
        """Dense source-target interaction count for the self-field problem."""

        return self.segment_count * self.segment_count


@dataclass(frozen=True, slots=True)
class ThetaResult:
    """Error and timing result for one hierarchical theta value."""

    theta: float
    build_seconds: float
    eval_seconds: float
    rms_relative_error: float
    max_relative_error: float


@dataclass(frozen=True, slots=True)
class StudyResult:
    """Convergence result for one loop discretization."""

    discretization: LoopDiscretization
    direct_seconds: float
    direct_field: NDArray[np.float64]
    theta_results: list[ThetaResult]


@dataclass(frozen=True, slots=True)
class ScalingResult:
    """Timing result for one self-field interaction-count target."""

    theta: float
    target_interactions: float
    discretization: LoopDiscretization
    build_seconds: float
    eval_seconds: float
    direct_seconds: float | None


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the convergence example."""

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
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel direct and hierarchical evaluation.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def circular_loop_discretization(target_ds: float) -> LoopDiscretization:
    """Build a closed circular loop with approximately `target_ds` segment length."""

    segment_count = max(8, int(np.ceil(2.0 * np.pi * LOOP_RADIUS / target_ds)))
    theta = np.linspace(0.0, 2.0 * np.pi, segment_count + 1, dtype=np.float64)
    points = np.vstack(
        [
            LOOP_RADIUS * np.cos(theta),
            LOOP_RADIUS * np.sin(theta),
            np.zeros_like(theta),
        ]
    )
    starts_arr = points[:, :-1]
    ends_arr = points[:, 1:]
    deltas_arr = ends_arr - starts_arr
    centers_arr = 0.5 * (starts_arr + ends_arr)
    actual_ds = float(np.mean(np.linalg.norm(deltas_arr, axis=0)))

    return LoopDiscretization(
        target_ds=target_ds,
        actual_ds=actual_ds,
        starts=(starts_arr[0], starts_arr[1], starts_arr[2]),
        deltas=(deltas_arr[0], deltas_arr[1], deltas_arr[2]),
        centers=(centers_arr[0], centers_arr[1], centers_arr[2]),
        current=np.full(segment_count, CURRENT, dtype=np.float64),
        wire_radius=np.full(segment_count, WIRE_RADIUS, dtype=np.float64),
    )


def circular_loop_discretization_for_interactions(target_interactions: float) -> LoopDiscretization:
    """Build a loop whose self-field has about `target_interactions` dense pairs."""

    segment_count = max(8, int(np.ceil(np.sqrt(target_interactions))))
    target_ds = 2.0 * np.pi * LOOP_RADIUS / segment_count
    return circular_loop_discretization(target_ds)


def stack_field(
    field: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Convert a `(x, y, z)` field tuple to a contiguous `(3, n)` array."""

    return np.vstack(field).astype(np.float64, copy=False)


def relative_error_metrics(
    candidate: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> tuple[float, float]:
    """Return vector RMS and max pointwise relative errors."""

    diff = candidate - reference
    reference_norm = np.linalg.norm(reference, axis=0)
    diff_norm = np.linalg.norm(diff, axis=0)
    rms_denominator = max(float(np.sqrt(np.mean(reference_norm * reference_norm))), np.finfo(np.float64).tiny)
    max_denominator = np.maximum(reference_norm, np.finfo(np.float64).tiny)
    rms_relative = float(np.sqrt(np.mean(diff_norm * diff_norm)) / rms_denominator)
    max_relative = float(np.max(diff_norm / max_denominator))
    return rms_relative, max_relative


def direct_self_field(discretization: LoopDiscretization, par: bool) -> tuple[NDArray[np.float64], float]:
    """Evaluate the direct linear-filament self-field at segment centers."""

    start = perf_counter()
    field = stack_field(
        cfsem.flux_density_linear_filament(
            discretization.centers,
            discretization.starts,
            discretization.deltas,
            discretization.current,
            discretization.wire_radius,
            par=par,
        )
    )
    elapsed = perf_counter() - start
    return field, elapsed


def hierarchical_self_field(
    discretization: LoopDiscretization,
    theta: float,
    par: bool,
) -> tuple[NDArray[np.float64], float, float]:
    """Build and evaluate the hierarchical linear-filament self-field solve."""

    solver = cfsem.HierarchicalLinearFilaments(theta=theta, construction_method=CONSTRUCTION_METHOD)
    start = perf_counter()
    solver.build(
        discretization.starts,
        discretization.deltas,
        discretization.wire_radius,
        discretization.centers,
        par=par,
    )
    build_seconds = perf_counter() - start

    start = perf_counter()
    field = stack_field(solver.flux_density(discretization.current, par=par))
    eval_seconds = perf_counter() - start
    return field, build_seconds, eval_seconds


def run_study(par: bool) -> list[StudyResult]:
    """Run the theta sweep for each requested segment length."""

    results: list[StudyResult] = []
    for target_ds in TARGET_SEGMENT_LENGTHS:
        discretization = circular_loop_discretization(target_ds)
        direct_field, direct_seconds = direct_self_field(discretization, par=par)
        theta_results: list[ThetaResult] = []
        for theta in THETA_SWEEP:
            hierarchical_field, build_seconds, eval_seconds = hierarchical_self_field(
                discretization,
                theta,
                par=par,
            )
            rms_relative, max_relative = relative_error_metrics(hierarchical_field, direct_field)
            theta_results.append(
                ThetaResult(
                    theta=float(theta),
                    build_seconds=build_seconds,
                    eval_seconds=eval_seconds,
                    rms_relative_error=rms_relative,
                    max_relative_error=max_relative,
                )
            )
        results.append(
            StudyResult(
                discretization=discretization,
                direct_seconds=direct_seconds,
                direct_field=direct_field,
                theta_results=theta_results,
            )
        )
    return results


def run_scaling_study(par: bool) -> dict[float, list[ScalingResult]]:
    """Run hierarchical self-field timing over a dense interaction-count sweep."""

    interaction_targets = TESTING_SCALING_INTERACTION_TARGETS if TESTING else SCALING_INTERACTION_TARGETS
    discretizations = [
        circular_loop_discretization_for_interactions(float(target_interactions))
        for target_interactions in interaction_targets
    ]
    direct_seconds_by_interactions: dict[int, float | None] = {}
    for discretization in discretizations:
        direct_seconds = None
        if discretization.interaction_count <= DIRECT_SCALING_MAX_INTERACTIONS:
            _direct_field, direct_seconds = direct_self_field(discretization, par=par)
        direct_seconds_by_interactions[discretization.interaction_count] = direct_seconds

    results_by_theta: dict[float, list[ScalingResult]] = {}
    for theta in SCALING_THETAS:
        theta_results: list[ScalingResult] = []
        for target_interactions, discretization in zip(interaction_targets, discretizations, strict=True):
            _field, build_seconds, eval_seconds = hierarchical_self_field(
                discretization,
                theta,
                par=par,
            )
            theta_results.append(
                ScalingResult(
                    theta=float(theta),
                    target_interactions=float(target_interactions),
                    discretization=discretization,
                    build_seconds=build_seconds,
                    eval_seconds=eval_seconds,
                    direct_seconds=direct_seconds_by_interactions[discretization.interaction_count],
                )
            )
        results_by_theta[float(theta)] = theta_results
    return results_by_theta


def direct_time_fit(scaling_results: list[ScalingResult]) -> tuple[float, float] | None:
    """Fit direct self-field time as `seconds = slope * interactions + intercept`."""

    measured = [item for item in scaling_results if item.direct_seconds is not None]
    if len(measured) < 2:
        return None
    interactions = np.array([item.discretization.interaction_count for item in measured], dtype=np.float64)
    seconds = np.array([float(item.direct_seconds) for item in measured], dtype=np.float64)
    slope, intercept = np.polyfit(interactions, seconds, 1)
    return float(slope), float(intercept)


def build_figure(
    results: list[StudyResult],
    scaling_results_by_theta: dict[float, list[ScalingResult]],
) -> plt.Figure:
    """Plot self-field convergence and interaction-count timing."""

    fig = plt.figure(figsize=(17.0, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    error_ax = fig.add_subplot(grid[0, 0])
    theta_time_ax = fig.add_subplot(grid[0, 1])
    legend_ax = fig.add_subplot(grid[0, 2])
    scaling_axes = [fig.add_subplot(grid[1, i]) for i in range(3)]

    for result in results:
        theta = np.array([item.theta for item in result.theta_results], dtype=np.float64)
        rms_error = np.array([item.rms_relative_error for item in result.theta_results], dtype=np.float64)
        max_error = np.array([item.max_relative_error for item in result.theta_results], dtype=np.float64)
        eval_seconds = np.array([item.eval_seconds for item in result.theta_results], dtype=np.float64)
        build_seconds = np.array([item.build_seconds for item in result.theta_results], dtype=np.float64)
        label = (
            f"target ds={1.0e3 * result.discretization.target_ds:.0f} mm "
            f"(n={result.discretization.segment_count})"
        )

        error_ax.loglog(theta, rms_error, marker="o", label=f"RMS, {label}")
        error_ax.loglog(theta, max_error, marker="s", linestyle="--", label=f"Max, {label}")
        theta_time_ax.loglog(theta, eval_seconds, marker="o", label=f"Eval, {label}")
        theta_time_ax.loglog(
            theta,
            build_seconds + eval_seconds,
            marker="s",
            linestyle="--",
            label=f"Build+eval, {label}",
        )
        theta_time_ax.axhline(result.direct_seconds, color="0.65", linewidth=1.0, linestyle=":")

    for ax in (error_ax, theta_time_ax):
        ax.set_xlabel(r"Opening angle $\theta$")
        ax.invert_xaxis()
        ax.grid(True, which="both", linewidth=0.5, alpha=0.35)

    for ax, theta in zip(scaling_axes, SCALING_THETAS, strict=True):
        scaling_results = scaling_results_by_theta[float(theta)]
        plot_scaling_axis(ax, scaling_results)
        ax.set_title(rf"Scaling at $\theta={theta:g}$")
        if ax is scaling_axes[0]:
            ax.set_ylabel("Time [s]")
        else:
            ax.set_ylabel("")

    error_ax.set_ylabel("Relative error vs. direct self-field")
    error_ax.set_title("Hierarchical self-field convergence")
    error_ax.legend(fontsize="small")
    theta_time_ax.set_ylabel("Time [s]")
    theta_time_ax.set_title("Hierarchical timing; dotted lines are direct references")
    theta_time_ax.legend(fontsize="small")
    legend_ax.axis("off")
    handles, labels = scaling_axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", frameon=True, title="Scaling traces")
    fig.suptitle(
        "Linear-filament circular-loop self-field convergence\n"
        f"radius={LOOP_RADIUS:g} m, current={CURRENT:g} A, wire_radius={WIRE_RADIUS:g} m, "
        f"construction={CONSTRUCTION_METHOD}"
    )
    return fig


def plot_scaling_axis(ax: plt.Axes, scaling_results: list[ScalingResult]) -> None:
    """Plot hierarchical and direct timing for one fixed theta value."""

    scaling_interactions = np.array(
        [item.discretization.interaction_count for item in scaling_results],
        dtype=np.float64,
    )
    scaling_build_seconds = np.array([item.build_seconds for item in scaling_results], dtype=np.float64)
    scaling_eval_seconds = np.array([item.eval_seconds for item in scaling_results], dtype=np.float64)
    ax.loglog(scaling_interactions, scaling_eval_seconds, marker="o", label="Hierarchical eval")
    ax.loglog(
        scaling_interactions,
        scaling_build_seconds + scaling_eval_seconds,
        marker="s",
        linestyle="--",
        label="Hierarchical build+eval",
    )

    direct_measured = [item for item in scaling_results if item.direct_seconds is not None]
    if direct_measured:
        direct_interactions = np.array(
            [item.discretization.interaction_count for item in direct_measured],
            dtype=np.float64,
        )
        direct_seconds = np.array([float(item.direct_seconds) for item in direct_measured], dtype=np.float64)
        ax.loglog(direct_interactions, direct_seconds, marker="^", color=DIRECT_TRACE_COLOR, label="Direct")

    direct_fit = direct_time_fit(scaling_results)
    if direct_fit is not None:
        slope, intercept = direct_fit
        fit_mask = scaling_interactions > DIRECT_SCALING_MAX_INTERACTIONS
        if np.any(fit_mask) and direct_measured:
            last_measured = max(direct_measured, key=lambda item: item.discretization.interaction_count)
            last_interaction = float(last_measured.discretization.interaction_count)
            fit_interactions = np.concatenate(([last_interaction], scaling_interactions[fit_mask]))
            fit_seconds = np.maximum(slope * fit_interactions + intercept, np.finfo(np.float64).tiny)
            ax.loglog(
                fit_interactions,
                fit_seconds,
                color=DIRECT_TRACE_COLOR,
                linestyle=":",
                label="Direct linear fit",
            )

    ax.set_xlabel(r"Dense interactions $n_\mathrm{src} n_\mathrm{target}$")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)


def print_results(
    results: list[StudyResult],
    scaling_results_by_theta: dict[float, list[ScalingResult]],
) -> None:
    """Print a compact table of convergence and timing results."""

    print(
        "Study configuration: "
        f"radius={LOOP_RADIUS:.6g} m, current={CURRENT:.6g} A, wire_radius={WIRE_RADIUS:.6g} m, "
        f"construction_method={CONSTRUCTION_METHOD}, theta=[{THETA_SWEEP[0]:.3g}, {THETA_SWEEP[-1]:.3g}]"
    )
    for result in results:
        disc = result.discretization
        print()
        print(
            f"Loop discretization: target_ds={1.0e3 * disc.target_ds:.3f} mm, "
            f"actual_ds={1.0e3 * disc.actual_ds:.3f} mm, segments={disc.segment_count}, "
            f"dense_interactions={disc.interaction_count:.3e}, direct={1.0e3 * result.direct_seconds:.3f} ms"
        )
        print("theta, build_ms, eval_ms, speedup_vs_direct_eval, rms_rel_error, max_point_rel_error")
        for item in result.theta_results:
            speedup = result.direct_seconds / item.eval_seconds if item.eval_seconds > 0.0 else np.inf
            print(
                f"{item.theta:6.3f}, "
                f"{1.0e3 * item.build_seconds:9.3f}, "
                f"{1.0e3 * item.eval_seconds:8.3f}, "
                f"{speedup:10.3f}, "
                f"{item.rms_relative_error:12.5e}, "
                f"{item.max_relative_error:12.5e}"
            )

    for theta in SCALING_THETAS:
        scaling_results = scaling_results_by_theta[float(theta)]
        print()
        print(f"Interaction-count scaling at theta={theta:.3g}")
        print("target_interactions, actual_interactions, segments, ds_mm, build_ms, eval_ms, direct_ms")
        direct_fit = direct_time_fit(scaling_results)
        for item in scaling_results:
            disc = item.discretization
            direct_label = "fit skipped"
            if item.direct_seconds is not None:
                direct_label = f"{1.0e3 * item.direct_seconds:10.3f}"
            elif direct_fit is not None:
                slope, intercept = direct_fit
                fit_seconds = max(slope * disc.interaction_count + intercept, np.finfo(np.float64).tiny)
                direct_label = f"{1.0e3 * fit_seconds:10.3f} fit"
            print(
                f"{item.target_interactions:12.3e}, "
                f"{disc.interaction_count:12.3e}, "
                f"{disc.segment_count:8d}, "
                f"{1.0e3 * disc.actual_ds:10.5f}, "
                f"{1.0e3 * item.build_seconds:10.3f}, "
                f"{1.0e3 * item.eval_seconds:10.3f}, "
                f"{direct_label}"
            )


def main() -> None:
    """Run the convergence study and save the resulting plot."""

    args = parse_args()
    results = run_study(par=args.parallel)
    scaling_results_by_theta = run_scaling_study(par=args.parallel)
    print_results(results, scaling_results_by_theta)
    fig = build_figure(results, scaling_results_by_theta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    print(f"Saved plot to {args.output}")
    if not args.no_plot and not TESTING:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
