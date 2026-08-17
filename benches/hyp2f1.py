"""Compare Python-level hyp2f1 throughput for cfsem, mpmath, and SciPy.

The complex workload exercises several numerical regions and compares cfsem's
array interface with mpmath's scalar complex implementation. The real workload
uses ``z < 1`` so that SciPy and cfsem evaluate the same real branch.

Run the default 65,536-element benchmark with::

    uv run python benches/hyp2f1.py

mpmath is much slower than the native array implementations, so the default is
one timed pass. Use ``--repeats 3`` when more stable timings are worth the wait.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import mpmath as mp
import numpy as np
from scipy.special import hyp2f1 as scipy_hyp2f1

import cfsem

DEFAULT_SIZE = 1 << 16

# These points cover the direct series, a terminating polynomial, Pfaff's
# transformation, expansions near one and infinity, Taylor continuation, and
# a near-integer parameter difference.
COMPLEX_CASES = np.asarray(
    [
        (0.5 + 0.25j, 1.25 - 0.5j, 2.0 + 0.75j, 0.1 + 0.2j),
        (-2.0 + 0.0j, 1.2 + 0.4j, 3.5 - 0.2j, 2.0 + 0.5j),
        (0.7 + 0.2j, 1.3 - 0.1j, 2.4 + 0.3j, -3.0 + 0.4j),
        (0.4 + 0.2j, 1.1 + 0.3j, 2.5 + 0.5j, 0.98 + 0.03j),
        (0.4 + 0.2j, 1.1 + 0.3j, 2.7 - 0.2j, 4.0 + 2.0j),
        (0.7 + 0.2j, 1.2 - 0.3j, 2.1 + 0.1j, 0.5 + 0.866_025_403_784_438_6j),
        (0.4 + 0.2j, 0.9 - 0.1j, 3.3 + 0.100_000_001j, 0.97 + 0.02j),
    ],
    dtype=np.complex128,
)

# SciPy's real-valued interface does not return the complex limiting value for
# real z > 1. Keep this comparison below the cut while retaining direct,
# transformed, near-one, polynomial, and moderate-parameter cases.
REAL_CASES = np.asarray(
    [
        (0.5, 1.1, 2.4, 0.2),
        (1.2, 0.7, 2.8, -0.5),
        (0.4, 1.1, 2.5, 0.95),
        (0.7, 1.3, 3.1, -3.0),
        (-3.0, 1.4, 2.5, 0.8),
        (12.5, 9.25, 17.0, 0.4),
    ],
    dtype=np.float64,
)

T = TypeVar("T")


@dataclass(frozen=True)
class Timing:
    """Median wall time and derived element throughput for one implementation."""

    name: str
    seconds: float
    size: int

    @property
    def values_per_second(self) -> float:
        return self.size / self.seconds


def tiled_arguments(cases: np.ndarray, size: int, dtype: np.dtype) -> tuple[np.ndarray, ...]:
    """Tile case rows into four contiguous argument arrays of exactly ``size`` elements."""

    indices = np.arange(size) % len(cases)
    return tuple(np.ascontiguousarray(cases[indices, column], dtype=dtype) for column in range(4))


def time_call(name: str, function: Callable[[], T], size: int, repeats: int) -> tuple[Timing, T]:
    """Time a no-argument callable while excluding cyclic-GC bookkeeping."""

    samples = []
    result: T | None = None
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            result = function()
            samples.append(time.perf_counter() - start)
    finally:
        if gc_was_enabled:
            gc.enable()
    assert result is not None
    return Timing(name, statistics.median(samples), size), result


def print_timings(title: str, timings: Sequence[Timing]) -> None:
    """Print a compact throughput table."""

    print(f"\n{title}")
    print(f"{'implementation':<30} {'seconds':>10} {'values/s':>15}")
    print(f"{'-' * 30} {'-' * 10} {'-' * 15}")
    for timing in timings:
        print(f"{timing.name:<30} {timing.seconds:>10.4f} {timing.values_per_second:>15,.0f}")


def mpmath_arguments(arguments: tuple[np.ndarray, ...]) -> tuple[tuple[mp.mpc, ...], ...]:
    """Convert NumPy inputs once so mpmath conversion is outside the timed loop."""

    return tuple(tuple(mp.mpc(value) for value in argument) for argument in arguments)


def benchmark_complex(size: int, repeats: int, dps: int) -> None:
    """Benchmark complex128 cfsem arrays against scalar mpmath evaluation."""

    arguments = tiled_arguments(COMPLEX_CASES, size, np.dtype(np.complex128))
    parallel_out = np.empty(size, dtype=np.complex128)
    serial_out = np.empty(size, dtype=np.complex128)
    mp.mp.dps = dps
    mp_arguments = mpmath_arguments(arguments)

    # Initialize the Rayon pool and mpmath caches without evaluating the full
    # workload twice.
    cfsem.hyp2f1(*(argument[: len(COMPLEX_CASES)] for argument in arguments), par=True)
    mp.hyp2f1(*(argument[0] for argument in mp_arguments))

    parallel, parallel_values = time_call(
        "cfsem (parallel)",
        lambda: cfsem.hyp2f1(*arguments, par=True, out=parallel_out),
        size,
        repeats,
    )
    serial, _ = time_call(
        "cfsem (serial)",
        lambda: cfsem.hyp2f1(*arguments, par=False, out=serial_out),
        size,
        repeats,
    )
    mpmath, mpmath_values = time_call(
        "mpmath (scalar loop)",
        lambda: [mp.hyp2f1(a, b, c, z) for a, b, c, z in zip(*mp_arguments, strict=True)],
        size,
        repeats,
    )

    sample = np.linspace(0, size - 1, min(size, 32), dtype=np.intp)
    reference = np.asarray([complex(mpmath_values[index]) for index in sample])
    np.testing.assert_allclose(parallel_values[sample], reference, rtol=3e-10, atol=3e-11)
    print_timings("Complex inputs", [parallel, serial, mpmath])


def benchmark_real(size: int, repeats: int) -> None:
    """Benchmark cfsem and SciPy for wholly real input arrays below the cut."""

    real_arguments = tiled_arguments(REAL_CASES, size, np.dtype(np.float64))
    complex_arguments = tuple(argument.astype(np.complex128) for argument in real_arguments)
    parallel_out = np.empty(size, dtype=np.complex128)
    serial_out = np.empty(size, dtype=np.complex128)

    cfsem.hyp2f1(*(argument[: len(REAL_CASES)] for argument in complex_arguments), par=True)
    scipy_hyp2f1(*(argument[: len(REAL_CASES)] for argument in real_arguments))

    parallel, parallel_values = time_call(
        "cfsem (parallel)",
        lambda: cfsem.hyp2f1(*complex_arguments, par=True, out=parallel_out),
        size,
        repeats,
    )
    serial, _ = time_call(
        "cfsem (serial)",
        lambda: cfsem.hyp2f1(*complex_arguments, par=False, out=serial_out),
        size,
        repeats,
    )
    scipy, scipy_values = time_call(
        "scipy.special.hyp2f1", lambda: scipy_hyp2f1(*real_arguments), size, repeats
    )

    np.testing.assert_allclose(parallel_values.real, scipy_values, rtol=3e-10, atol=3e-11)
    np.testing.assert_allclose(parallel_values.imag, 0.0, atol=3e-11)
    print_timings("Real inputs (z < 1)", [parallel, serial, scipy])


def positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    """Parse benchmark settings and run both workloads."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--size", type=positive_integer, default=DEFAULT_SIZE, help="elements per workload")
    parser.add_argument("--repeats", type=positive_integer, default=1, help="timed passes per implementation")
    parser.add_argument("--dps", type=positive_integer, default=17, help="mpmath decimal precision")
    args = parser.parse_args()

    print(f"hyp2f1 throughput: size={args.size:,}, repeats={args.repeats}, mpmath dps={args.dps}")
    benchmark_complex(args.size, args.repeats, args.dps)
    benchmark_real(args.size, args.repeats)


if __name__ == "__main__":
    main()
