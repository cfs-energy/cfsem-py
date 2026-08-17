"""Test rusteq.math module against scipy/numpy"""

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ellipe, ellipk, hyp2f1 as scipy_hyp2f1

import cfsem

DATA = Path(__file__).parent / "data"


def test_ellipe():
    # 64-bit version
    xs = np.linspace(0.0, 1.0 - 1e-7, 100)
    assert np.allclose(ellipe(xs), np.array([cfsem.ellipe(x) for x in xs]))


def test_ellipk():
    # 64-bit version
    xs = np.linspace(0.0, 1.0 - 1e-7, 100)
    assert np.allclose(ellipk(xs), np.array([cfsem.ellipk(x) for x in xs]))


def test_hyp2f1_complex128_binding():
    a = np.array([0.5 + 0.2j, 1.2 - 0.3j], dtype=np.complex128)
    b = np.array([1.1 - 0.1j, 0.7 + 0.4j], dtype=np.complex128)
    c = np.array([2.4 + 0.3j, 2.8 - 0.2j], dtype=np.complex128)
    z = np.array([0.0 - 0.0j, 2.0 + 0.5j], dtype=np.complex128)
    serial = cfsem.hyp2f1(a, b, c, z, par=False)
    parallel = cfsem.hyp2f1(a, b, c, z)
    assert serial.dtype == np.complex128
    assert serial.shape == (2,)
    assert serial[0] == 1.0 + 0.0j
    np.testing.assert_array_equal(parallel, serial)


@pytest.mark.parametrize(
    "bad",
    [
        np.ones(3, dtype=np.float64),
        np.ones(3, dtype=np.complex64),
        np.ones((1, 3), dtype=np.complex128),
        np.ones(6, dtype=np.complex128)[::2],
    ],
)
def test_hyp2f1_rejects_non_complex128_vector_inputs(bad):
    valid = np.ones(3, dtype=np.complex128)
    with pytest.raises((TypeError, ValueError)):
        cfsem.hyp2f1(bad, valid, valid, valid)


def test_hyp2f1_rejects_unequal_lengths():
    short = np.ones(2, dtype=np.complex128)
    long = np.ones(3, dtype=np.complex128)
    with pytest.raises(ValueError, match="Length mismatch"):
        cfsem.hyp2f1(short, long, short, short)


def _hyp2f1_reference_rows():
    with (DATA / "hyp2f1_reference.csv").open(newline="") as stream:
        records = (line for line in stream if not line.startswith("#"))
        yield from csv.DictReader(records)


def test_hyp2f1_full_complex_mpmath_reference():
    rows = list(_hyp2f1_reference_rows())

    def values(prefix):
        return np.array(
            [complex(float(row[f"{prefix}_re"]), float(row[f"{prefix}_im"])) for row in rows],
            dtype=np.complex128,
        )

    actual = cfsem.hyp2f1(values("a"), values("b"), values("c"), values("z"))
    expected = values("expected")
    for index, row in enumerate(rows):
        tolerance = float(row["rtol"])
        np.testing.assert_allclose(
            actual[index],
            expected[index],
            rtol=tolerance,
            atol=tolerance,
            err_msg=row["label"],
        )


def test_hyp2f1_agrees_with_scipy_real_parameter_subset():
    a = np.array([0.5, 1.2, 1.0, -3.0, 0.7], dtype=np.complex128)
    b = np.array([1.1, 0.7, 1.0, 1.4, 1.3], dtype=np.complex128)
    c = np.array([2.4, 2.8, 4.0, 2.5, 3.1], dtype=np.complex128)
    z = np.array([0.2 + 0.1j, 2.0 + 0.4j, 3.0 + 4.0j, 2.0 + 0.5j, -3.0 + 0.2j])
    actual = cfsem.hyp2f1(a, b, c, z)
    expected = scipy_hyp2f1(a.real, b.real, c.real, z)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)


def test_hyp2f1_scipy_1561_regression():
    one = np.array([1.0 + 0.0j], dtype=np.complex128)
    c = np.array([4.0 + 0.0j], dtype=np.complex128)
    z = np.array([3.0 + 4.0j], dtype=np.complex128)
    actual = cfsem.hyp2f1(one, one, c, z)[0]
    expected = scipy_hyp2f1(1.0, 1.0, 4.0, 3.0 + 4.0j)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)


def test_hyp2f1_conjugation_and_branch_approach():
    a = np.array([1.2 + 0.3j], dtype=np.complex128)
    b = np.array([0.7 - 0.1j], dtype=np.complex128)
    c = np.array([2.8 + 0.2j], dtype=np.complex128)
    z = np.array([0.4 + 0.6j], dtype=np.complex128)
    value = cfsem.hyp2f1(a, b, c, z)[0]
    conjugate = cfsem.hyp2f1(a.conj(), b.conj(), c.conj(), z.conj())[0]
    np.testing.assert_allclose(conjugate, value.conjugate(), rtol=2e-12, atol=2e-12)

    upper_cut = cfsem.hyp2f1(a, b, c, np.array([complex(2.0, 0.0)]))[0]
    lower_cut = cfsem.hyp2f1(a, b, c, np.array([complex(2.0, -0.0)]))[0]
    assert upper_cut != lower_cut
    upper_near = cfsem.hyp2f1(a, b, c, np.array([2.0 + 1e-10j]))[0]
    lower_near = cfsem.hyp2f1(a, b, c, np.array([2.0 - 1e-10j]))[0]
    np.testing.assert_allclose(upper_near, upper_cut, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(lower_near, lower_cut, rtol=2e-9, atol=2e-9)


def test_hyp2f1_derivative_identity():
    a = np.array([0.5 + 0.2j], dtype=np.complex128)
    b = np.array([1.1 - 0.1j], dtype=np.complex128)
    c = np.array([2.4 + 0.3j], dtype=np.complex128)
    z = np.array([0.3 + 0.2j], dtype=np.complex128)
    step = 1e-5
    plus = cfsem.hyp2f1(a, b, c, z + step)[0]
    minus = cfsem.hyp2f1(a, b, c, z - step)[0]
    numerical = (plus - minus) / (2.0 * step)
    shifted = cfsem.hyp2f1(a + 1.0, b + 1.0, c + 1.0, z)[0]
    expected = (a[0] * b[0] / c[0]) * shifted
    np.testing.assert_allclose(numerical, expected, rtol=2e-9, atol=2e-10)
