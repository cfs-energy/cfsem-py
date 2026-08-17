"""Test rusteq.math module against scipy/numpy"""

import numpy as np
import pytest
from scipy.special import ellipe, ellipk

import cfsem


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
