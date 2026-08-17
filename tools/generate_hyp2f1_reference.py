"""Regenerate complex hyp2f1 and gamma reference fixtures with mpmath 1.3.0."""

from __future__ import annotations

import csv
from pathlib import Path

import mpmath as mp

MPMATH_VERSION = "1.3.0"
PRECISION = 100
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "test" / "data"


def parts(value: mp.mpc) -> tuple[str, str]:
    return mp.nstr(value.real, 80), mp.nstr(value.imag, 80)


def hyp_rows() -> list[tuple[complex, complex, complex, complex, str, float]]:
    points = [
        (0.5 + 0.25j, 1.25 - 0.5j, 2.0 + 0.75j, 0.1 + 0.2j, "direct", 2e-13),
        (-2.0 + 0j, 1.2 + 0.4j, 3.5 - 0.2j, 2.0 + 0.5j, "polynomial", 2e-13),
        (0.7 + 0.2j, 1.3 - 0.1j, 2.4 + 0.3j, -3.0 + 0.4j, "pfaff", 2e-12),
        (0.4 + 0.2j, 1.1 + 0.3j, 2.7 - 0.2j, 4.0 + 2.0j, "infinity", 3e-12),
        (0.4 + 0.2j, 1.1 + 0.3j, 2.5 + 0.5j, 0.98 + 0.03j, "one", 3e-12),
        (
            0.4 + 0.2j,
            0.9 - 0.1j,
            3.3 + 0.100000001j,
            0.97 + 0.02j,
            "one-near-integer",
            8e-11,
        ),
        (0.4 + 0.2j, 0.9 - 0.1j, 3.300000001 + 0.1j, 0.97 + 0.02j, "one-near-integer", 8e-11),
        (0.4 + 0.2j, 0.9 - 0.1j, 3.299999999 + 0.1j, 0.97 + 0.02j, "one-near-integer", 8e-11),
        (0.4 + 0.2j, 0.9 - 0.1j, 3.3 + 0.099999999j, 0.97 + 0.02j, "one-near-integer", 8e-11),
        (
            0.4 + 0.2j,
            2.400000001 + 0.200000001j,
            3.1 - 0.3j,
            5.0 + 1.0j,
            "infinity-near-integer",
            8e-11,
        ),
        (0.4 + 0.2j, 2.400000001 + 0.2j, 3.1 - 0.3j, 5.0 + 1.0j, "infinity-near-integer", 8e-11),
        (0.4 + 0.2j, 2.399999999 + 0.2j, 3.1 - 0.3j, 5.0 + 1.0j, "infinity-near-integer", 8e-11),
        (0.4 + 0.2j, 2.4 + 0.199999999j, 3.1 - 0.3j, 5.0 + 1.0j, "infinity-near-integer", 8e-11),
        (1.4 + 0.2j, 1.2 - 0.3j, 1.1 + 0.4j, 0.6 + 0.2j, "euler", 8e-11),
        (12.5 + 2.0j, 9.25 - 1.5j, 17.0 + 0.5j, 0.4 + 0.2j, "moderate", 2e-10),
        (0.7 + 0.2j, 1.2 - 0.3j, 2.1 + 0.1j, 0.5 + 0.8660254037844386j, "taylor", 5e-12),
        (0.7 + 0.2j, 1.2 - 0.3j, 2.1 + 0.1j, 0.5 - 0.8660254037844386j, "taylor", 5e-12),
        (0.7 + 0.2j, 1.2 - 0.3j, 2.1 + 0.1j, 0.495 + 0.8573651497465942j, "taylor", 5e-12),
        (0.7 + 0.2j, 1.2 - 0.3j, 2.1 + 0.1j, 0.495 - 0.8573651497465942j, "taylor", 5e-12),
        (0.7 + 0.2j, 1.2 - 0.3j, 2.1 + 0.1j, 0.505 + 0.874685657822283j, "taylor", 5e-12),
        (0.7 + 0.2j, 1.2 - 0.3j, 2.1 + 0.1j, 0.505 - 0.874685657822283j, "taylor", 5e-12),
        (1.0 + 0j, 1.0 + 0j, 4.0 + 0j, 3.0 + 4.0j, "scipy-1561", 3e-12),
        (1.2 + 0.3j, 0.7 - 0.1j, 2.8 + 0.2j, 2.0 + 0.0j, "upper-cut", 5e-12),
        (1.2 + 0.3j, 0.7 - 0.1j, 2.8 + 0.2j, 2.0 - 0.0j, "lower-cut", 5e-12),
    ]
    return points


def generate_hyp2f1() -> None:
    path = DATA / "hyp2f1_reference.csv"
    with path.open("w", newline="") as stream:
        stream.write(f"# mpmath={MPMATH_VERSION}, dps={PRECISION}\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "a_re",
                "a_im",
                "b_re",
                "b_im",
                "c_re",
                "c_im",
                "z_re",
                "z_im",
                "expected_re",
                "expected_im",
                "rtol",
                "label",
            ]
        )
        for a, b, c, z, label, rtol in hyp_rows():
            mz = mp.mpc(z.real, z.imag)
            if label == "upper-cut":
                mz = mp.mpc(z.real, mp.mpf("1e-80"))
            elif label == "lower-cut":
                mz = mp.mpc(z.real, -mp.mpf("1e-80"))
            expected = mp.hyp2f1(mp.mpc(a), mp.mpc(b), mp.mpc(c), mz)
            writer.writerow(
                [
                    a.real,
                    a.imag,
                    b.real,
                    b.imag,
                    c.real,
                    c.imag,
                    z.real,
                    "-0.0" if label == "lower-cut" else z.imag,
                    *parts(expected),
                    rtol,
                    label,
                ]
            )


def generate_gamma() -> None:
    path = DATA / "complex_gamma_reference.csv"
    points = [
        0.2 + 0.3j,
        0.2 - 0.3j,
        1.2 - 2.5j,
        1.2 + 2.5j,
        8.5 + 1.25j,
        -0.3 + 0.7j,
        -0.3 - 0.7j,
        -4.0 + 1e-8j,
        -4.0 - 1e-8j,
    ]
    with path.open("w", newline="") as stream:
        stream.write(f"# mpmath={MPMATH_VERSION}, dps={PRECISION}\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "z_re",
                "z_im",
                "gamma_re",
                "gamma_im",
                "rgamma_re",
                "rgamma_im",
                "digamma_re",
                "digamma_im",
            ]
        )
        for z in points:
            value = mp.mpc(z)
            writer.writerow(
                [
                    z.real,
                    z.imag,
                    *parts(mp.gamma(value)),
                    *parts(mp.rgamma(value)),
                    *parts(mp.digamma(value)),
                ]
            )


if __name__ == "__main__":
    if mp.__version__ != MPMATH_VERSION:
        raise RuntimeError(f"expected mpmath {MPMATH_VERSION}, found {mp.__version__}")
    mp.mp.dps = PRECISION
    DATA.mkdir(parents=True, exist_ok=True)
    generate_hyp2f1()
    generate_gamma()
