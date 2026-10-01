"""Reproduce the branch-free D(1-c) fit in src/math.rs.

Run with the project's development environment. Fits P(c) - log(c)*Q(c)
to scipy.special.elliprd(0, c, 1)/3 using relative least squares, then checks
a separate dense grid and high-precision mpmath values. NIST DLMF 19.25.1
defines the reference; 19.5.3 and 19.12.1–2 supply the endpoint constraints.
The measured errors are empirical checks, not rigorous bounds.
"""

from math import log, pi

import mpmath as mp
import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import elliprd


def main():
    degree = 5
    c = np.unique(np.r_[np.linspace(1e-8, 1, 5001), np.geomspace(1e-30, 1, 5001)])
    reference = elliprd(0, c, 1) / 3
    # P(0)=ln(4)-1, P(1)=pi/4, Q(0)=1/2, enforced before fitting.
    base = log(4) - 1 + (pi / 4 - log(4) + 1) * c - 0.5 * np.log(c)
    columns = [c**i * c * (1 - c) for i in range(degree - 1)]
    columns += [-np.log(c) * c**i for i in range(1, degree + 1)]
    design = np.array(columns).T
    fit = np.linalg.lstsq(design / reference[:, None], (reference - base) / reference, rcond=None)[0]
    p = Polynomial([log(4) - 1, pi / 4 - log(4) + 1])
    p += Polynomial([0, 1, -1]) * Polynomial(fit[: degree - 1])
    q = Polynomial(np.r_[0.5, fit[degree - 1 :]])
    print("P:", p.coef.tolist())
    print("Q:", q.coef.tolist())

    c = np.unique(np.r_[np.linspace(1e-10, 1, 100001), np.geomspace(1e-30, 1, 10001)])
    reference = elliprd(0, c, 1) / 3
    actual = p(c) - np.log(c) * q(c)
    print("Dense-grid maximum relative error:", np.max(np.abs(actual / reference - 1)))
    print("Dense-grid maximum absolute error:", np.max(np.abs(actual - reference)))
    assert np.all(np.abs(actual / reference - 1) < 4e-9)

    with mp.workdps(80):
        # Includes complements too small for subtraction from 1 or SciPy R_D.
        points = np.r_[np.linspace(0.001, 1, 257), np.geomspace(1e-310, 1, 257), np.nextafter(0, 1)]
        error = max(
            float(abs((p(c) - np.log(c) * q(c)) / (mp.elliprd(0, mp.mpf(float(c)), 1) / 3) - 1))
            for c in points
        )
    print("High-precision maximum relative error:", error)
    assert error < 4e-9


if __name__ == "__main__":
    main()
