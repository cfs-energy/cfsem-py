"""Test rusteq.math module against scipy/numpy"""

import numpy as np
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
