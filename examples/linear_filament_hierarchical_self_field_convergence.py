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
reference while the hierarchical opening angle `theta` is swept from 0.5 down
to 0.005. A second plot row changes the loop discretization to sweep dense
interaction counts from about 1e5 to 1e12 and records hierarchical build and
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
from scipy.ndimage import gaussian_filter

if os.getenv("CFSEM_TESTING"):
    import matplotlib

    matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import cfsem

TESTING = bool(os.getenv("CFSEM_TESTING"))

LOOP_RADIUS = 1.0  # [m]
CURRENT = 1.0  # [A]
WIRE_RADIUS = 1.0e-3  # [m]
TARGET_SEGMENT_LENGTHS = (0.01, 0.001)  # [m]
THETA_SWEEP = 0.5 * np.logspace(0.0, -2.0, 10, dtype=np.float64)
NEAR_FIELD_THETA_SWEEP = 0.5 * np.logspace(0.0, -2.0, 40, dtype=np.float64)
SCALING_THETAS = (0.0625, 0.175, 0.5)
FIRST_FIGURE_SCALING_THETAS = (0.5, 0.0625)
SCALING_INTERACTION_TARGETS = np.logspace(5.0, 12.0, 8, dtype=np.float64)
TESTING_SCALING_INTERACTION_TARGETS = np.logspace(5.0, 6.0, 2, dtype=np.float64)
MIN_SCALING_BENCH_SECONDS = 0.1
MIN_NEAR_FIELD_BENCH_SECONDS = 0.1
MIN_SELF_FIELD_TRADEOFF_BENCH_SECONDS = 0.1
DIRECT_SCALING_MAX_INTERACTIONS = 2.0e10
DIRECT_ASYMPTOTIC_FIT_POINTS = 3
DIRECT_TRACE_COLOR = "tab:green"
TREE_AABB_COLOR = "tab:blue"
CFSEM_HIERARCHICAL_TRACE_COLOR = "black"
MAX_AABB_PLOT_LEVELS = 6
NEAR_FIELD_INBOARD_FROM_FIRST_ORIGIN = 0.05  # [m]
NEAR_FIELD_INBOARD_RADIUS = 0.001  # [m]
NEAR_FIELD_OUTBOARD_FROM_FIRST_ORIGIN = 0.05  # [m]
NEAR_FIELD_OUTBOARD_RADIUS = 1.5  # [m]
NEAR_FIELD_TARGET_COUNT = 100
TESTING_NEAR_FIELD_TARGET_COUNT = 30
NEAR_FIELD_DS_SWEEP = np.logspace(np.log10(0.001), np.log10(0.05), 32, dtype=np.float64)
TESTING_NEAR_FIELD_DS_SWEEP = np.array([0.02, 0.005, 0.001], dtype=np.float64)
NEAR_FIELD_ERROR_TARGET = 1.0e-3
ACCURACY_RUNTIME_BAND_COLOR = "tab:red"
ACCURACY_RUNTIME_X_LABEL = "Build+eval time [s]"
ACCURACY_RUNTIME_Y_LABEL = "Relative error"
ACCURACY_RUNTIME_LEGEND_FONTSIZE = "x-small"
ACCURACY_RUNTIME_ANNOTATION_OFFSET = (4, 4)
ACCURACY_RUNTIME_ANNOTATION_FONTSIZE = 7


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


@dataclass(frozen=True, slots=True)
class SelfFieldTradeoffResult:
    """Self-field accuracy values with separately batched timing samples."""

    discretization: LoopDiscretization
    theta_values: NDArray[np.float64]
    build_eval_seconds: NDArray[np.float64]
    rms_relative_error: NDArray[np.float64]
    max_relative_error: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class NearFieldStudy:
    """Near-field error and timing values on a theta/discretization grid."""

    ds_values: NDArray[np.float64]
    theta_values: NDArray[np.float64]
    rms_relative_error: NDArray[np.float64]
    max_relative_error: NDArray[np.float64]
    run_seconds: NDArray[np.float64]


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
        "--serial",
        action="store_true",
        help="Use serial direct and hierarchical evaluation.",
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


def near_field_observation_points() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Observation points on radial lines inside and outside the loop."""

    target_count = TESTING_NEAR_FIELD_TARGET_COUNT if TESTING else NEAR_FIELD_TARGET_COUNT
    inboard_radius = np.linspace(
        NEAR_FIELD_INBOARD_RADIUS,
        LOOP_RADIUS - NEAR_FIELD_INBOARD_FROM_FIRST_ORIGIN,
        target_count,
        dtype=np.float64,
    )
    outboard_radius = np.linspace(
        LOOP_RADIUS + NEAR_FIELD_OUTBOARD_FROM_FIRST_ORIGIN,
        NEAR_FIELD_OUTBOARD_RADIUS,
        target_count,
        dtype=np.float64,
    )
    radius = np.concatenate((inboard_radius, outboard_radius))
    return (
        radius,
        np.zeros_like(radius),
        np.zeros_like(radius),
    )


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

    result = cfsem.flux_density_linear_filament_hierarchical(
        discretization.centers,
        discretization.starts,
        discretization.deltas,
        discretization.current,
        discretization.wire_radius,
        theta=theta,
        par=par,
    )
    return (
        stack_field(result.field),
        float(result.diagnostics.construction_time),
        float(result.diagnostics.evaluation_time),
    )


def scaling_direct_self_field_seconds(discretization: LoopDiscretization, par: bool) -> float:
    """Return the best direct timing from a scaling benchmark batch."""

    best_seconds = np.inf
    batch_start = perf_counter()
    while True:
        _field, seconds = direct_self_field(discretization, par=par)
        best_seconds = min(best_seconds, seconds)
        if perf_counter() - batch_start >= MIN_SCALING_BENCH_SECONDS:
            return float(best_seconds)


def scaling_hierarchical_self_field_seconds(
    discretization: LoopDiscretization,
    theta: float,
    par: bool,
) -> tuple[float, float]:
    """Return the build/eval timings from the best hierarchical scaling batch sample."""

    best_build_seconds = np.inf
    best_eval_seconds = np.inf
    best_total_seconds = np.inf
    batch_start = perf_counter()
    while True:
        _field, build_seconds, eval_seconds = hierarchical_self_field(discretization, theta, par=par)
        total_seconds = build_seconds + eval_seconds
        if total_seconds < best_total_seconds:
            best_build_seconds = build_seconds
            best_eval_seconds = eval_seconds
            best_total_seconds = total_seconds
        if perf_counter() - batch_start >= MIN_SCALING_BENCH_SECONDS:
            return float(best_build_seconds), float(best_eval_seconds)


def near_field_hierarchical_field(
    discretization: LoopDiscretization,
    targets: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    theta: float,
    par: bool,
) -> tuple[NDArray[np.float64], float]:
    """Return the best hierarchical near-field timing from a benchmark batch."""

    best_field: NDArray[np.float64] | None = None
    best_seconds = np.inf
    batch_start = perf_counter()
    while True:
        start = perf_counter()
        result = cfsem.flux_density_linear_filament_hierarchical(
            targets,
            discretization.starts,
            discretization.deltas,
            discretization.current,
            discretization.wire_radius,
            theta=float(theta),
            par=par,
        )
        elapsed = perf_counter() - start
        if elapsed < best_seconds:
            best_field = stack_field(result.field)
            best_seconds = elapsed
        if perf_counter() - batch_start >= MIN_NEAR_FIELD_BENCH_SECONDS:
            assert best_field is not None
            return best_field, float(best_seconds)


def self_field_hierarchical_build_eval_seconds(
    discretization: LoopDiscretization,
    theta: float,
    par: bool,
) -> float:
    """Return the best hierarchical build+eval timing from a tradeoff benchmark batch."""

    best_seconds = np.inf
    batch_start = perf_counter()
    while True:
        _field, build_seconds, eval_seconds = hierarchical_self_field(discretization, theta, par=par)
        best_seconds = min(best_seconds, build_seconds + eval_seconds)
        if perf_counter() - batch_start >= MIN_SELF_FIELD_TRADEOFF_BENCH_SECONDS:
            return float(best_seconds)


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
            print(
                "Scaling run: direct "
                f"interactions={discretization.interaction_count:.3e}, "
                f"segments={discretization.segment_count}",
                flush=True,
            )
            direct_seconds = scaling_direct_self_field_seconds(discretization, par=par)
        direct_seconds_by_interactions[discretization.interaction_count] = direct_seconds

    results_by_theta: dict[float, list[ScalingResult]] = {}
    for theta in SCALING_THETAS:
        theta_results: list[ScalingResult] = []
        for target_interactions, discretization in zip(interaction_targets, discretizations, strict=True):
            print(
                "Scaling run: hierarchical "
                f"theta={theta:.3g}, interactions={discretization.interaction_count:.3e}, "
                f"segments={discretization.segment_count}",
                flush=True,
            )
            build_seconds, eval_seconds = scaling_hierarchical_self_field_seconds(
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


def run_near_field_study(par: bool) -> NearFieldStudy:
    """Compare hierarchical filamentized-loop fields against the analytic circular loop."""

    ds_values = TESTING_NEAR_FIELD_DS_SWEEP if TESTING else NEAR_FIELD_DS_SWEEP
    obs = near_field_observation_points()
    reference = stack_field(
        cfsem.flux_density_circular_filament_cartesian(
            np.array([CURRENT], dtype=np.float64),
            np.array([LOOP_RADIUS], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            obs,
            par=par,
        )
    )
    theta_values = NEAR_FIELD_THETA_SWEEP
    error = np.empty((theta_values.size, ds_values.size), dtype=np.float64)
    max_error = np.empty_like(error)
    run_seconds = np.empty_like(error)

    for j, ds in enumerate(ds_values):
        discretization = circular_loop_discretization(float(ds))
        for i, theta in enumerate(theta_values):
            field, run_seconds[i, j] = near_field_hierarchical_field(
                discretization,
                obs,
                theta=float(theta),
                par=par,
            )
            error[i, j], max_error[i, j] = relative_error_metrics(field, reference)

    return NearFieldStudy(
        ds_values=np.asarray(ds_values, dtype=np.float64),
        theta_values=theta_values.copy(),
        rms_relative_error=error,
        max_relative_error=max_error,
        run_seconds=run_seconds,
    )


def run_self_field_tradeoff_study(
    results: list[StudyResult],
    par: bool,
) -> list[SelfFieldTradeoffResult]:
    """Remeasure 1 mm self-field tradeoff runtimes with at least 100 ms per data point."""

    tradeoff_results: list[SelfFieldTradeoffResult] = []
    selected_results = [result for result in results if np.isclose(result.discretization.target_ds, 0.001)]
    for result in selected_results:
        discretization = result.discretization
        seconds = np.empty(len(result.theta_results), dtype=np.float64)
        for i, theta_result in enumerate(result.theta_results):
            print(
                "Self-field tradeoff run: hierarchical "
                f"theta={theta_result.theta:.3g}, interactions={discretization.interaction_count:.3e}, "
                f"segments={discretization.segment_count}",
                flush=True,
            )
            seconds[i] = self_field_hierarchical_build_eval_seconds(
                discretization,
                theta_result.theta,
                par=par,
            )
        tradeoff_results.append(
            SelfFieldTradeoffResult(
                discretization=discretization,
                theta_values=np.array([item.theta for item in result.theta_results], dtype=np.float64),
                build_eval_seconds=seconds,
                rms_relative_error=np.array(
                    [item.rms_relative_error for item in result.theta_results],
                    dtype=np.float64,
                ),
                max_relative_error=np.array(
                    [item.max_relative_error for item in result.theta_results],
                    dtype=np.float64,
                ),
            )
        )
    return tradeoff_results


def direct_time_fit(scaling_results: list[ScalingResult]) -> tuple[float, float] | None:
    """Fit direct self-field time from the largest measured interaction counts."""

    measured = sorted(
        (item for item in scaling_results if item.direct_seconds is not None),
        key=lambda item: item.discretization.interaction_count,
    )
    if len(measured) < 2:
        return None
    measured = measured[-DIRECT_ASYMPTOTIC_FIT_POINTS:]
    interactions = np.array([item.discretization.interaction_count for item in measured], dtype=np.float64)
    seconds = np.array([float(item.direct_seconds) for item in measured], dtype=np.float64)
    slope, intercept = np.polyfit(interactions, seconds, 1)
    return float(slope), float(intercept)


def loglog_power_fit(
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
) -> tuple[float, float] | None:
    """Fit positive samples as `y = coefficient * x ** exponent`."""

    valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0.0) & (y_values > 0.0)
    if np.count_nonzero(valid) < 2:
        return None
    exponent, log_coefficient = np.polyfit(np.log(x_values[valid]), np.log(y_values[valid]), 1)
    return float(exponent), float(np.exp(log_coefficient))


def plot_runtime_scaling_fit(
    ax: plt.Axes,
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
    color: str,
) -> float | None:
    """Draw a log-log runtime scaling fit and return its exponent."""

    fit = loglog_power_fit(x_values, y_values)
    if fit is None:
        return None
    exponent, coefficient = fit
    fit_x = np.array([float(np.min(x_values)), float(np.max(x_values))], dtype=np.float64)
    fit_y = coefficient * fit_x**exponent
    ax.loglog(fit_x, fit_y, color=color, linestyle="-.", linewidth=1.0, alpha=0.8, label="_nolegend_")
    return exponent


def plot_runtime_scaling_fit_on_axis(
    ax: plt.Axes,
    fit_x_values: NDArray[np.float64],
    plot_x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
    color: str,
) -> float | None:
    """Fit against one size variable and draw the fit against another x-axis variable."""

    fit = loglog_power_fit(fit_x_values, y_values)
    if fit is None:
        return None
    exponent, coefficient = fit
    fit_x = np.array([float(np.min(fit_x_values)), float(np.max(fit_x_values))], dtype=np.float64)
    plot_x = np.array([float(np.min(plot_x_values)), float(np.max(plot_x_values))], dtype=np.float64)
    fit_y = coefficient * fit_x**exponent
    ax.loglog(plot_x, fit_y, color=color, linestyle="-.", linewidth=1.0, alpha=0.8, label="_nolegend_")
    return exponent


def source_tree_aabbs(discretization: LoopDiscretization) -> tuple[NDArray[np.float64], ...]:
    """Build the coarse source tree and return its AABB arrays for plotting."""

    result = cfsem.flux_density_linear_filament_hierarchical(
        discretization.centers,
        discretization.starts,
        discretization.deltas,
        discretization.current,
        discretization.wire_radius,
        theta=float(THETA_SWEEP[0]),
        par=False,
        extra_diagnostics=True,
    )
    source_tree = result.diagnostics.source_tree
    assert source_tree is not None
    return source_tree


def build_figure(
    results: list[StudyResult],
    scaling_results_by_theta: dict[float, list[ScalingResult]],
    self_field_tradeoff_results: list[SelfFieldTradeoffResult],
) -> plt.Figure:
    """Plot self-field convergence and interaction-count timing."""

    fig = plt.figure(figsize=(18.5, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.34])
    error_ax = fig.add_subplot(grid[0, 0])
    theta_time_ax = fig.add_subplot(grid[0, 1])
    domain_ax = fig.add_subplot(grid[0, 2])
    scaling_axes = [fig.add_subplot(grid[1, i]) for i in range(2)]
    tradeoff_ax = fig.add_subplot(grid[1, 2])
    scaling_legend_ax = fig.add_subplot(grid[1, 3])

    for result in results:
        theta = np.array([item.theta for item in result.theta_results], dtype=np.float64)
        rms_error = np.array([item.rms_relative_error for item in result.theta_results], dtype=np.float64)
        max_error = np.array([item.max_relative_error for item in result.theta_results], dtype=np.float64)
        eval_seconds = np.array([item.eval_seconds for item in result.theta_results], dtype=np.float64)
        build_seconds = np.array([item.build_seconds for item in result.theta_results], dtype=np.float64)
        label = f"target ds={1.0e3 * result.discretization.target_ds:.0f} mm"

        error_ax.loglog(theta, rms_error, marker="o", label=f"cfsem RMS, {label}")
        max_error_color = (
            CFSEM_HIERARCHICAL_TRACE_COLOR if np.isclose(result.discretization.target_ds, 0.001) else None
        )
        error_ax.loglog(
            theta,
            max_error,
            marker="s",
            linestyle="--",
            color=max_error_color,
            label=f"cfsem max, {label}",
        )
        theta_time_ax.loglog(theta, eval_seconds, marker="o", label=f"cfsem eval, {label}")
        theta_time_ax.loglog(
            theta,
            build_seconds + eval_seconds,
            marker="s",
            linestyle="--",
            color=CFSEM_HIERARCHICAL_TRACE_COLOR,
            label=f"cfsem build+eval, {label}",
        )
        theta_time_ax.axhline(result.direct_seconds, color="0.65", linewidth=1.0, linestyle=":")

    for ax in (error_ax, theta_time_ax):
        ax.set_xlabel(r"Opening angle $\theta$")
        ax.invert_xaxis()
        ax.grid(True, which="both", linewidth=0.5, alpha=0.35)

    plot_domain_axis(domain_ax, results[0].discretization)

    for ax, theta in zip(scaling_axes, FIRST_FIGURE_SCALING_THETAS, strict=True):
        scaling_results = scaling_results_by_theta[float(theta)]
        plot_scaling_axis(ax, scaling_results)
        ax.set_title(rf"Scaling at $\theta={theta:g}$")
        if ax is scaling_axes[0]:
            ax.set_ylabel("Time [s]")
        else:
            ax.set_ylabel("")
    if self_field_tradeoff_results:
        plot_self_field_tradeoff_axis(tradeoff_ax, self_field_tradeoff_results[0], show_ylabel=False)

    error_ax.set_ylabel("Relative error vs. direct self-field")
    error_ax.set_title("Hierarchical self-field convergence")
    error_ax.legend(fontsize="small")
    theta_time_ax.set_ylabel("Time [s]")
    theta_time_ax.set_title("Hierarchical timing; dotted lines are direct references")
    theta_time_ax.legend(fontsize="small")
    scaling_legend_ax.axis("off")
    handles, labels = scaling_axes[0].get_legend_handles_labels()
    scaling_legend = scaling_legend_ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=True,
        title="Scaling traces",
    )
    scaling_legend_ax.add_artist(scaling_legend)
    if self_field_tradeoff_results:
        handles, labels = tradeoff_ax.get_legend_handles_labels()
        scaling_legend_ax.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0.0, 0.0),
            fontsize=ACCURACY_RUNTIME_LEGEND_FONTSIZE,
            frameon=True,
            title="Accuracy traces",
        )
    fig.suptitle(
        "Linear-filament circular-loop self-field convergence\n"
        f"radius={LOOP_RADIUS:g} m, current={CURRENT:g} A, wire_radius={WIRE_RADIUS:g} m"
    )
    return fig


def plot_self_field_tradeoff_axis(
    ax: plt.Axes,
    result: SelfFieldTradeoffResult,
    *,
    show_ylabel: bool,
) -> None:
    """Plot self-field accuracy against build+eval runtime on one axis."""

    seconds = result.build_eval_seconds.copy()
    rms_error = result.rms_relative_error.copy()
    max_error = result.max_relative_error.copy()
    theta = result.theta_values.copy()
    order = np.argsort(seconds)
    seconds = seconds[order]
    rms_error = rms_error[order]
    max_error = max_error[order]
    theta = theta[order]
    marker_count = min(4, seconds.size)
    marker_indices = np.unique(np.linspace(0, seconds.size - 1, marker_count, dtype=np.int64))

    ax.fill_between(
        seconds,
        rms_error,
        max_error,
        color=ACCURACY_RUNTIME_BAND_COLOR,
        alpha=0.10,
        label="cfsem RMS-max",
    )
    ax.loglog(
        seconds,
        rms_error,
        marker="o",
        markevery=marker_indices,
        color=CFSEM_HIERARCHICAL_TRACE_COLOR,
        label="cfsem hierarchical RMS",
    )
    ax.loglog(
        seconds,
        max_error,
        marker="s",
        markevery=marker_indices,
        linestyle="--",
        color=CFSEM_HIERARCHICAL_TRACE_COLOR,
        label="cfsem hierarchical max",
    )
    for annotation_index, marker_index in enumerate(marker_indices):
        theta_label = (
            f"theta={theta[marker_index]:.2g}" if annotation_index == 0 else f"{theta[marker_index]:.2g}"
        )
        ax.annotate(
            theta_label,
            (seconds[marker_index], max_error[marker_index]),
            textcoords="offset points",
            xytext=ACCURACY_RUNTIME_ANNOTATION_OFFSET,
            fontsize=ACCURACY_RUNTIME_ANNOTATION_FONTSIZE,
            color=CFSEM_HIERARCHICAL_TRACE_COLOR,
        )
    ax.set_xlabel(ACCURACY_RUNTIME_X_LABEL)
    if show_ylabel:
        ax.set_ylabel(ACCURACY_RUNTIME_Y_LABEL)
    ax.set_title(
        "Self-field accuracy vs. build+eval time\n"
        f"ds={1.0e3 * result.discretization.target_ds:.3g} mm, "
        f"N_src*N_targ={result.discretization.interaction_count:.3e}"
    )
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)


def build_self_field_tradeoff_figure(results: list[SelfFieldTradeoffResult]) -> plt.Figure:
    """Plot self-field accuracy against build+eval runtime as a standalone figure."""

    fig, axes_array = plt.subplots(
        1,
        len(results),
        figsize=(8.4 * max(1, len(results)), 5.2),
        constrained_layout=False,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.10, right=0.70, bottom=0.12, top=0.82)
    axes = axes_array.ravel()

    for ax, result in zip(axes, results, strict=True):
        plot_self_field_tradeoff_axis(ax, result, show_ylabel=True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.73, 0.5),
        fontsize=ACCURACY_RUNTIME_LEGEND_FONTSIZE,
        frameon=True,
    )
    fig.suptitle("Self-field accuracy vs. build+eval runtime")
    return fig


def log_edges(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return logarithmic cell edges for monotonically increasing positive centers."""

    values = np.asarray(values, dtype=np.float64)
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = np.sqrt(values[:-1] * values[1:])
    edges[0] = values[0] * np.sqrt(values[0] / values[1])
    edges[-1] = values[-1] * np.sqrt(values[-1] / values[-2])
    return edges


def log_norm(values: NDArray[np.float64]) -> LogNorm:
    """Return a valid logarithmic color normalization, including for flat arrays."""

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def fastest_acceptable_cell(error: NDArray[np.float64], seconds: NDArray[np.float64]) -> tuple[int, int]:
    """Return the theta/discretization cell with the shortest time under the error target."""

    acceptable = error <= NEAR_FIELD_ERROR_TARGET
    if np.any(acceptable):
        masked_seconds = np.where(acceptable, seconds, np.inf)
        return np.unravel_index(int(np.argmin(masked_seconds)), seconds.shape)
    return np.unravel_index(int(np.argmin(error)), error.shape)


def blurred_contour_values(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a lightly blurred copy for smoother contour lines."""

    return np.asarray(gaussian_filter(values, sigma=1.0, mode="nearest"), dtype=np.float64)


def plot_near_field_accuracy_runtime_axis(
    ax: plt.Axes,
    study: NearFieldStudy,
    *,
    show_ylabel: bool,
    show_legend: bool,
) -> NDArray[np.float64]:
    """Plot near-field accuracy against build+eval runtime on one axis."""

    ds_mm = 1.0e3 * study.ds_values
    theta = study.theta_values
    error = np.maximum(study.rms_relative_error, np.finfo(np.float64).tiny)
    seconds = np.maximum(study.run_seconds, np.finfo(np.float64).tiny)
    one_mm_ds_idx = int(np.argmin(np.abs(ds_mm - 1.0)))
    one_mm_error = error[:, one_mm_ds_idx]
    one_mm_max_error = np.maximum(study.max_relative_error[:, one_mm_ds_idx], np.finfo(np.float64).tiny)
    one_mm_seconds = seconds[:, one_mm_ds_idx]
    one_mm_order = np.argsort(one_mm_seconds)
    one_mm_seconds = one_mm_seconds[one_mm_order]
    one_mm_error = one_mm_error[one_mm_order]
    one_mm_max_error = one_mm_max_error[one_mm_order]
    one_mm_theta = theta[one_mm_order]
    marker_count = min(4, one_mm_seconds.size)
    marker_indices = np.unique(np.linspace(0, one_mm_seconds.size - 1, marker_count, dtype=np.int64))
    ax.fill_between(
        one_mm_seconds,
        one_mm_error,
        one_mm_max_error,
        color=ACCURACY_RUNTIME_BAND_COLOR,
        alpha=0.10,
        label="cfsem RMS-max",
    )
    ax.loglog(
        one_mm_seconds,
        one_mm_error,
        marker="o",
        markevery=marker_indices,
        color=CFSEM_HIERARCHICAL_TRACE_COLOR,
        label="cfsem hierarchical RMS",
    )
    ax.loglog(
        one_mm_seconds,
        one_mm_max_error,
        marker="s",
        markevery=marker_indices,
        linestyle="--",
        color=CFSEM_HIERARCHICAL_TRACE_COLOR,
        label="cfsem hierarchical max",
    )
    for annotation_index, marker_index in enumerate(marker_indices):
        theta_label = (
            f"theta={one_mm_theta[marker_index]:.2g}"
            if annotation_index == 0
            else f"{one_mm_theta[marker_index]:.2g}"
        )
        ax.annotate(
            theta_label,
            (one_mm_seconds[marker_index], one_mm_max_error[marker_index]),
            textcoords="offset points",
            xytext=ACCURACY_RUNTIME_ANNOTATION_OFFSET,
            fontsize=ACCURACY_RUNTIME_ANNOTATION_FONTSIZE,
            color=CFSEM_HIERARCHICAL_TRACE_COLOR,
        )
    ax.axhline(NEAR_FIELD_ERROR_TARGET, color="black", linestyle=":", linewidth=1.0)
    ax.set_xlabel(ACCURACY_RUNTIME_X_LABEL)
    if show_ylabel:
        ax.set_ylabel(ACCURACY_RUNTIME_Y_LABEL)
    one_mm_discretization = circular_loop_discretization(float(study.ds_values[one_mm_ds_idx]))
    near_field_target_count = near_field_observation_points()[0].size
    near_field_interactions = one_mm_discretization.segment_count * near_field_target_count
    ax.set_title(
        "Near-field accuracy vs. build+eval time\n"
        f"ds={ds_mm[one_mm_ds_idx]:.3g} mm, N_src*N_targ={near_field_interactions:.3e}"
    )
    if show_legend:
        ax.legend(fontsize=ACCURACY_RUNTIME_LEGEND_FONTSIZE, frameon=True)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.25)
    return np.concatenate((one_mm_error, one_mm_max_error))


def build_near_field_figure(study: NearFieldStudy) -> plt.Figure:
    """Plot near-field error and runtime heatmaps versus theta and discretization."""

    ds_mm = 1.0e3 * study.ds_values
    theta = study.theta_values
    ds_edges = log_edges(ds_mm)
    theta_edges = log_edges(theta[::-1])[::-1]
    error = np.maximum(study.rms_relative_error, np.finfo(np.float64).tiny)
    seconds = np.maximum(study.run_seconds, np.finfo(np.float64).tiny)
    contour_seconds = blurred_contour_values(seconds)
    selected_theta_idx, selected_ds_idx = fastest_acceptable_cell(error, seconds)
    selected_theta = theta[selected_theta_idx]
    selected_ds_mm = ds_mm[selected_ds_idx]

    fig = plt.figure(figsize=(17.0, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.05])
    error_ax = fig.add_subplot(grid[0, 0])
    time_ax = fig.add_subplot(grid[0, 1])
    domain_ax = fig.add_subplot(grid[0, 2])
    theta_slice_ax = fig.add_subplot(grid[1, 0])
    ds_slice_ax = fig.add_subplot(grid[1, 1])
    accuracy_runtime_ax = fig.add_subplot(grid[1, 2])
    error_mesh = error_ax.pcolormesh(
        ds_edges,
        theta_edges,
        error,
        norm=log_norm(error),
        shading="auto",
    )
    time_mesh = time_ax.pcolormesh(
        ds_edges,
        theta_edges,
        seconds,
        norm=log_norm(seconds),
        shading="auto",
    )

    error_contour_levels = [
        level
        for level in (1.0e-5, 1.0e-4, NEAR_FIELD_ERROR_TARGET)
        if np.min(error) <= level <= np.max(error)
    ]
    if error_contour_levels:
        contours = error_ax.contour(
            ds_mm,
            theta,
            error,
            levels=error_contour_levels,
            colors="black",
            linewidths=0.8,
        )
        error_ax.clabel(contours, inline=True, fontsize=8, fmt=lambda value: f"{value:.0e}")
    if np.min(error) <= NEAR_FIELD_ERROR_TARGET <= np.max(error):
        time_error_contours = time_ax.contour(
            ds_mm,
            theta,
            error,
            levels=[NEAR_FIELD_ERROR_TARGET],
            colors="white",
            linestyles="--",
            linewidths=1.0,
        )
        time_ax.clabel(
            time_error_contours,
            inline=True,
            fontsize=8,
            fmt=lambda _value: "err = 1e-3",
            colors="white",
        )
    if float(np.max(contour_seconds)) > float(np.min(contour_seconds)):
        time_contour_levels = np.logspace(
            np.log10(float(np.min(contour_seconds))),
            np.log10(float(np.max(contour_seconds))),
            6,
            dtype=np.float64,
        )[1:-1]
        time_contours = time_ax.contour(
            ds_mm,
            theta,
            contour_seconds,
            levels=time_contour_levels,
            colors="white",
            linewidths=0.8,
        )
        time_ax.clabel(time_contours, inline=True, fontsize=8, fmt=lambda value: f"{value:.1e} s")
    selected_marker = {
        "marker": "o",
        "color": "white",
        "markeredgecolor": "black",
        "markersize": 7,
    }
    error_ax.plot(selected_ds_mm, selected_theta, **selected_marker)
    time_ax.plot(selected_ds_mm, selected_theta, **selected_marker)

    theta_slice_ax.loglog(theta, error[:, selected_ds_idx], color="black", linewidth=2.0)
    theta_slice_ax.axhline(NEAR_FIELD_ERROR_TARGET, color="black", linestyle=":", linewidth=1.0)
    theta_slice_ax.axvline(selected_theta, color="black", linestyle="--", linewidth=1.0)
    theta_slice_ax.plot(selected_theta, error[selected_theta_idx, selected_ds_idx], marker="o", color="black")
    theta_slice_ax.invert_xaxis()
    theta_slice_ax.set_xlabel(r"Opening angle $\theta$")
    theta_slice_ax.set_ylabel("RMS relative error")
    theta_slice_ax.set_title(f"Error vs. theta at ds={selected_ds_mm:.3g} mm")

    ds_slice_ax.loglog(ds_mm, error[selected_theta_idx, :], color="black", linewidth=2.0)
    ds_slice_ax.axhline(NEAR_FIELD_ERROR_TARGET, color="black", linestyle=":", linewidth=1.0)
    ds_slice_ax.axvline(selected_ds_mm, color="black", linestyle="--", linewidth=1.0)
    ds_slice_ax.plot(selected_ds_mm, error[selected_theta_idx, selected_ds_idx], marker="o", color="black")
    ds_slice_ax.invert_xaxis()
    ds_slice_ax.set_xlabel("Segment length target [mm]")
    ds_slice_ax.set_ylabel("RMS relative error")
    ds_slice_ax.set_title(rf"Error vs. ds at $\theta={selected_theta:.3g}$")

    accuracy_runtime_trace_values = plot_near_field_accuracy_runtime_axis(
        accuracy_runtime_ax,
        study,
        show_ylabel=True,
        show_legend=True,
    )

    lower_trace_values = np.concatenate(
        (
            error[:, selected_ds_idx],
            error[selected_theta_idx, :],
            accuracy_runtime_trace_values,
        )
    )
    lower_y_min = float(np.min(lower_trace_values))
    lower_y_max = float(np.max(lower_trace_values))
    lower_y_pad = np.sqrt(10.0)
    lower_y_limits = (lower_y_min / lower_y_pad, lower_y_max * lower_y_pad)
    for ax in (error_ax, time_ax):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel("Segment length target [mm]")
        ax.grid(True, which="both", linewidth=0.4, alpha=0.25)

    error_ax.set_ylabel(r"Opening angle $\theta$")
    error_ax.set_title("Near-field RMS relative error")
    time_ax.set_title("Hierarchical run time")
    for ax in (theta_slice_ax, ds_slice_ax, accuracy_runtime_ax):
        ax.set_ylim(*lower_y_limits)
        ax.grid(True, which="both", linewidth=0.4, alpha=0.25)
    plot_near_field_domain_axis(domain_ax)
    fig.colorbar(error_mesh, ax=error_ax, label="RMS relative error")
    fig.colorbar(time_mesh, ax=time_ax, label="Time [s]")
    fig.suptitle(
        "Near-field convergence against analytic circular-filament field\n"
        f"targets on x-axis from r={NEAR_FIELD_INBOARD_RADIUS:.3f} m "
        f"to r={LOOP_RADIUS - NEAR_FIELD_INBOARD_FROM_FIRST_ORIGIN:.3f} m and "
        f"r={LOOP_RADIUS + NEAR_FIELD_OUTBOARD_FROM_FIRST_ORIGIN:.3f} m "
        f"to r={NEAR_FIELD_OUTBOARD_RADIUS:.3f} m"
    )
    return fig


def build_near_field_accuracy_runtime_figure(study: NearFieldStudy) -> plt.Figure:
    """Plot near-field accuracy against build+eval runtime as a standalone figure."""

    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.70, bottom=0.12, top=0.82)
    plot_near_field_accuracy_runtime_axis(ax, study, show_ylabel=True, show_legend=False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.73, 0.5),
        fontsize=ACCURACY_RUNTIME_LEGEND_FONTSIZE,
        frameon=True,
    )
    fig.suptitle("Near-field accuracy vs. build+eval runtime")
    return fig


def plot_near_field_domain_axis(ax: plt.Axes) -> None:
    """Plot the ds=10 mm source loop, near-field targets, and source-tree AABBs."""

    discretization = circular_loop_discretization(0.01)
    start_x, start_y, _start_z = discretization.starts
    delta_x, delta_y, _delta_z = discretization.deltas
    end_x = start_x + delta_x
    end_y = start_y + delta_y
    for i in range(discretization.segment_count):
        label = "sources" if i == 0 else None
        ax.plot([start_x[i], end_x[i]], [start_y[i], end_y[i]], color="black", linewidth=2.5, label=label)

    min_x, min_y, _min_z, max_x, max_y, _max_z, levels = source_tree_aabbs(discretization)[:7]
    for i in range(len(min_x)):
        if levels[i] > MAX_AABB_PLOT_LEVELS:
            continue
        xs = [min_x[i], max_x[i], max_x[i], min_x[i], min_x[i]]
        ys = [min_y[i], min_y[i], max_y[i], max_y[i], min_y[i]]
        label = "tree AABBs" if i == 0 else None
        ax.plot(xs, ys, color=TREE_AABB_COLOR, linewidth=0.8, alpha=0.7, label=label)

    target_x, target_y, _target_z = near_field_observation_points()
    ax.scatter(target_x, target_y, s=12, color="tab:red", label="targets", zorder=3)

    pad = 0.12 * LOOP_RADIUS
    ax.set_xlim(-LOOP_RADIUS - pad, NEAR_FIELD_OUTBOARD_RADIUS + pad)
    ax.set_ylim(-LOOP_RADIUS - pad, LOOP_RADIUS + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        "Near-field geometry\n"
        f"ds=10 mm, AABBs through level {MAX_AABB_PLOT_LEVELS}"
    )
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper right", fontsize="small", frameon=True)


def plot_domain_axis(ax: plt.Axes, discretization: LoopDiscretization) -> None:
    """Plot the coarse filament loop and source-tree AABBs in the loop plane."""

    start_x, start_y, _start_z = discretization.starts
    delta_x, delta_y, _delta_z = discretization.deltas
    end_x = start_x + delta_x
    end_y = start_y + delta_y
    for i in range(discretization.segment_count):
        ax.plot([start_x[i], end_x[i]], [start_y[i], end_y[i]], color="black", linewidth=3.0)

    min_x, min_y, _min_z, max_x, max_y, _max_z, levels = source_tree_aabbs(discretization)[:7]
    for i in range(len(min_x)):
        if levels[i] > MAX_AABB_PLOT_LEVELS:
            continue
        xs = [min_x[i], max_x[i], max_x[i], min_x[i], min_x[i]]
        ys = [min_y[i], min_y[i], max_y[i], max_y[i], min_y[i]]
        ax.plot(xs, ys, color=TREE_AABB_COLOR, linewidth=1.0)

    pad = 0.12 * LOOP_RADIUS
    ax.set_xlim(-LOOP_RADIUS - pad, LOOP_RADIUS + pad)
    ax.set_ylim(-LOOP_RADIUS - pad, LOOP_RADIUS + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Coarse loop source tree\n"
        f"ds={1.0e3 * discretization.target_ds:.0f} mm, AABBs through level {MAX_AABB_PLOT_LEVELS}"
    )
    ax.grid(True, linewidth=0.5, alpha=0.35)


def plot_scaling_axis(ax: plt.Axes, scaling_results: list[ScalingResult]) -> None:
    """Plot hierarchical and direct timing for one fixed theta value."""

    scaling_interactions = np.array(
        [item.discretization.interaction_count for item in scaling_results],
        dtype=np.float64,
    )
    scaling_sources = np.array(
        [item.discretization.segment_count for item in scaling_results],
        dtype=np.float64,
    )
    scaling_build_seconds = np.array([item.build_seconds for item in scaling_results], dtype=np.float64)
    scaling_eval_seconds = np.array([item.eval_seconds for item in scaling_results], dtype=np.float64)
    eval_line = ax.loglog(
        scaling_interactions,
        scaling_eval_seconds,
        marker="o",
        label="cfsem hierarchical eval",
    )[0]
    build_eval_seconds = scaling_build_seconds + scaling_eval_seconds
    build_eval_line = ax.loglog(
        scaling_interactions,
        build_eval_seconds,
        marker="s",
        linestyle="--",
        color=CFSEM_HIERARCHICAL_TRACE_COLOR,
        label="cfsem hierarchical build+eval",
    )[0]
    scaling_annotations: list[str] = []
    eval_exponent = plot_runtime_scaling_fit_on_axis(
        ax,
        scaling_sources,
        scaling_interactions,
        scaling_eval_seconds,
        eval_line.get_color(),
    )
    if eval_exponent is not None:
        scaling_annotations.append(rf"eval $\sim N^{{{eval_exponent:.2f}}}$")
    build_eval_exponent = plot_runtime_scaling_fit_on_axis(
        ax,
        scaling_sources,
        scaling_interactions,
        build_eval_seconds,
        build_eval_line.get_color(),
    )
    if build_eval_exponent is not None:
        scaling_annotations.append(rf"build+eval $\sim N^{{{build_eval_exponent:.2f}}}$")

    direct_measured = [item for item in scaling_results if item.direct_seconds is not None]
    if direct_measured:
        direct_interactions = np.array(
            [item.discretization.interaction_count for item in direct_measured],
            dtype=np.float64,
        )
        direct_sources = np.array(
            [item.discretization.segment_count for item in direct_measured],
            dtype=np.float64,
        )
        direct_seconds = np.array([float(item.direct_seconds) for item in direct_measured], dtype=np.float64)
        ax.loglog(
            direct_interactions,
            direct_seconds,
            marker="^",
            color=DIRECT_TRACE_COLOR,
            label="cfsem direct",
        )
        direct_exponent = plot_runtime_scaling_fit_on_axis(
            ax,
            direct_sources,
            direct_interactions,
            direct_seconds,
            DIRECT_TRACE_COLOR,
        )
        if direct_exponent is not None:
            scaling_annotations.append(rf"direct $\sim N^{{{direct_exponent:.2f}}}$")

    direct_fit = direct_time_fit(scaling_results)
    if direct_fit is not None:
        slope, intercept = direct_fit
        fit_mask = scaling_interactions > DIRECT_SCALING_MAX_INTERACTIONS
        if np.any(fit_mask) and direct_measured:
            last_measured = max(direct_measured, key=lambda item: item.discretization.interaction_count)
            last_interaction_count = float(last_measured.discretization.interaction_count)
            fit_interactions = np.concatenate(([last_interaction_count], scaling_interactions[fit_mask]))
            fit_seconds = np.maximum(slope * fit_interactions + intercept, np.finfo(np.float64).tiny)
            ax.loglog(
                fit_interactions,
                fit_seconds,
                color=DIRECT_TRACE_COLOR,
                linestyle=":",
                label="cfsem direct linear fit",
            )

    if scaling_annotations:
        ax.text(
            0.04,
            0.96,
            "\n".join(scaling_annotations),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
    ax.set_xlabel(r"Number of interactions $N^2$")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)


def print_results(
    results: list[StudyResult],
    scaling_results_by_theta: dict[float, list[ScalingResult]],
) -> None:
    """Print a compact table of convergence and timing results."""

    print(
        "Study configuration: "
        f"radius={LOOP_RADIUS:.6g} m, current={CURRENT:.6g} A, wire_radius={WIRE_RADIUS:.6g} m, "
        f"theta=[{THETA_SWEEP[0]:.3g}, {THETA_SWEEP[-1]:.3g}]"
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
    use_parallel = not args.serial
    results = run_study(par=use_parallel)
    scaling_results_by_theta = run_scaling_study(par=use_parallel)
    near_field_study = run_near_field_study(par=use_parallel)
    self_field_tradeoff_results = run_self_field_tradeoff_study(results, par=use_parallel)
    print_results(results, scaling_results_by_theta)
    fig = build_figure(results, scaling_results_by_theta, self_field_tradeoff_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    svg_output = args.output.with_suffix(".svg")
    fig.savefig(svg_output)
    print(f"Saved plot to {args.output}")
    print(f"Saved plot to {svg_output}")
    near_field_output = args.output.with_name(f"{args.output.stem}_near_field{args.output.suffix}")
    near_field_svg_output = near_field_output.with_suffix(".svg")
    near_field_fig = build_near_field_figure(near_field_study)
    near_field_fig.savefig(near_field_output, dpi=300)
    near_field_fig.savefig(near_field_svg_output)
    print(f"Saved near-field plot to {near_field_output}")
    print(f"Saved near-field plot to {near_field_svg_output}")
    near_field_tradeoff_output = args.output.with_name(
        f"{args.output.stem}_near_field_accuracy_runtime{args.output.suffix}"
    )
    near_field_tradeoff_svg_output = near_field_tradeoff_output.with_suffix(".svg")
    near_field_tradeoff_fig = build_near_field_accuracy_runtime_figure(near_field_study)
    near_field_tradeoff_fig.savefig(near_field_tradeoff_output, dpi=300)
    near_field_tradeoff_fig.savefig(near_field_tradeoff_svg_output)
    print(f"Saved near-field accuracy/runtime plot to {near_field_tradeoff_output}")
    print(f"Saved near-field accuracy/runtime plot to {near_field_tradeoff_svg_output}")
    tradeoff_output = args.output.with_name(f"{args.output.stem}_self_field_tradeoff{args.output.suffix}")
    tradeoff_svg_output = tradeoff_output.with_suffix(".svg")
    tradeoff_fig = build_self_field_tradeoff_figure(self_field_tradeoff_results)
    tradeoff_fig.savefig(tradeoff_output, dpi=300)
    tradeoff_fig.savefig(tradeoff_svg_output)
    print(f"Saved self-field tradeoff plot to {tradeoff_output}")
    print(f"Saved self-field tradeoff plot to {tradeoff_svg_output}")
    if not args.no_plot and not TESTING:
        plt.show()
    plt.close(fig)
    plt.close(near_field_fig)
    plt.close(near_field_tradeoff_fig)
    plt.close(tradeoff_fig)


if __name__ == "__main__":
    main()
