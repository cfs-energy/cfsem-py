import os
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix

from cfsem import gs_operator_order2, gs_operator_order4

HERE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))


def test_gs_operators():
    """Check that both Jardin's 2nd-order operator and our homebrewed 4th-order operator give similar results"""

    # Need high enough resolution that finite differences should be well-converged
    n = 200
    m = 201

    xmin, xmax = (5.0, 100.0)  # Must not cross zero
    ymin, ymax = (-50.0, 50.0)

    z1func = lambda x, y: 3 * x + 7 * y
    z2func = lambda x, y: 3 * x**2 + 7 * y**2

    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, m)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z1 = z1func(X, Y)
    Z2 = z2func(X, Y)

    gs_op_order2_triplet = gs_operator_order2(xs, ys)
    gs_op_order4_triplet = gs_operator_order4(xs, ys)

    gs_op_order2 = csc_matrix(
        (gs_op_order2_triplet[0], (gs_op_order2_triplet[1], gs_op_order2_triplet[2])),
        shape=(n * m, n * m),
    )
    gs_op_order4 = csc_matrix(
        (gs_op_order4_triplet[0], (gs_op_order4_triplet[1], gs_op_order4_triplet[2])),
        shape=(n * m, n * m),
    )

    z1_op_order2 = gs_op_order2 @ Z1.flatten()
    z1_op_order4 = gs_op_order4 @ Z1.flatten()

    assert np.allclose(z1_op_order2, z1_op_order4, rtol=5e-3)

    z2_op_order2 = gs_op_order2 @ Z2.flatten()
    z2_op_order4 = gs_op_order4 @ Z2.flatten()

    assert np.allclose(z2_op_order2, z2_op_order4, rtol=1e-6)
