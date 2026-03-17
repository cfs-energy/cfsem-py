"""Tests of triangle boundary-element interfaces"""

import numpy as np
from pytest import mark, raises

import cfsem

GL2_TRI_QUAD = np.array(
    [
        [0.05283121635, 0.1666666667, 0.7886751346],
        [0.1971687836, 0.6220084679, 0.2113248654],
        [0.05283121635, 0.04465819874, 0.7886751346],
        [0.1971687836, 0.1666666667, 0.2113248654],
    ],
    dtype=np.float64,
)


def _triangle_strip_mesh(
    radius: float,
    height: float,
    s0: float,
    nphi: int,
    z_center: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phis = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    x = radius * np.cos(phis)
    y = radius * np.sin(phis)
    z_lower = np.full(nphi, z_center - height / 2.0)
    z_upper = np.full(nphi, z_center + height / 2.0)

    lower = np.column_stack((x, y, z_lower))
    upper = np.column_stack((x, y, z_upper))
    nodes = np.ascontiguousarray(np.vstack((lower, upper)), dtype=np.float64)

    triangles = np.empty((2 * nphi, 3), dtype=np.int64)
    for i in range(nphi):
        i1 = (i + 1) % nphi
        lower0 = i
        lower1 = i1
        upper0 = i + nphi
        upper1 = i1 + nphi
        triangles[2 * i] = [lower0, lower1, upper1]
        triangles[2 * i + 1] = [lower0, upper1, upper0]

    s = np.ascontiguousarray(
        np.concatenate((-s0 * np.ones(nphi), s0 * np.ones(nphi))),
        dtype=np.float64,
    )

    return nodes, triangles, s


def _loop_vector_potential_cartesian(
    radius: float,
    current: float,
    obs: np.ndarray,
    par: bool,
) -> np.ndarray:
    r = np.sqrt(obs[:, 0] ** 2 + obs[:, 1] ** 2)
    phi = np.arctan2(obs[:, 1], obs[:, 0])
    a_phi = cfsem.vector_potential_circular_filament(
        np.array([current], dtype=np.float64),
        np.array([radius], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        r,
        obs[:, 2],
        par,
    )
    return np.column_stack(
        (-a_phi * np.sin(phi), a_phi * np.cos(phi), np.zeros(obs.shape[0]))
    )


def _triangle_current_density_reference(
    nodes: np.ndarray,
    triangles: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    out = np.empty((triangles.shape[0], 3), dtype=np.float64)
    for i, (i0, i1, i2) in enumerate(triangles):
        n0 = nodes[i0]
        n1 = nodes[i1]
        n2 = nodes[i2]
        area = 0.5 * np.linalg.norm(np.cross(n1 - n0, n2 - n0))
        out[i] = (
            s[i0] * (n2 - n1) + s[i1] * (n0 - n2) + s[i2] * (n1 - n0)
        ) / (2.0 * area)
    return out


def test_triangle_mesh_quadrature_points_and_current_density():
    nodes, triangles, s = _triangle_strip_mesh(0.73, 7.3e-4, 1.7, nphi=32)

    j = cfsem.triangle_mesh_current_density(nodes, triangles, s)
    j_ref = _triangle_current_density_reference(nodes, triangles, s)
    points, weights = cfsem.triangle_mesh_quadrature_points(nodes, triangles, quad="gl2")
    dunavant_points, dunavant_weights = cfsem.triangle_mesh_quadrature_points(
        nodes, triangles, quad="dunavant5"
    )

    assert j.shape == (triangles.shape[0], 3)
    assert points.shape == (triangles.shape[0], GL2_TRI_QUAD.shape[0], 3)
    assert weights.shape == (triangles.shape[0], GL2_TRI_QUAD.shape[0])
    assert dunavant_points.shape == (triangles.shape[0], 7, 3)
    assert dunavant_weights.shape == (triangles.shape[0], 7)
    assert np.allclose(j, j_ref, rtol=1e-13, atol=1e-13)

    for i, (i0, i1, i2) in enumerate(triangles):
        n0 = nodes[i0]
        n1 = nodes[i1]
        n2 = nodes[i2]
        area = 0.5 * np.linalg.norm(np.cross(n1 - n0, n2 - n0))
        expected_points = (
            (1.0 - GL2_TRI_QUAD[:, 1] - GL2_TRI_QUAD[:, 2])[:, None] * n0[None, :]
            + GL2_TRI_QUAD[:, 1][:, None] * n1[None, :]
            + GL2_TRI_QUAD[:, 2][:, None] * n2[None, :]
        )
        expected_weights = GL2_TRI_QUAD[:, 0] * area
        assert np.allclose(points[i], expected_points, rtol=0.0, atol=1e-13)
        assert np.allclose(weights[i], expected_weights, rtol=0.0, atol=1e-13)


@mark.parametrize("par", [True, False])
@mark.parametrize("quad", ["gl3", "dunavant5"])
def test_triangle_mesh_far_field_against_circular_filament(par, quad):
    radius = 0.7312345987
    height = radius * 1e-3
    loop_current = 1.7
    nodes, triangles, s = _triangle_strip_mesh(radius, height, loop_current, nphi=256)
    obs = np.array(
        [
            [2.70, 0.95, 0.85],
            [3.10, -1.15, 1.05],
            [3.45, 0.75, -1.25],
            [3.80, 1.30, 1.60],
            [4.20, -0.90, -1.55],
            [4.55, 1.10, -2.05],
        ],
        dtype=np.float64,
    )

    bx, by, bz = cfsem.flux_density_triangle_mesh(obs, nodes, triangles, s, par=par, quad=quad)
    ax, ay, az = cfsem.vector_potential_triangle_mesh(
        obs, nodes, triangles, s, par=par, quad=quad
    )
    b_ref = np.column_stack(
        cfsem.flux_density_circular_filament_cartesian(
            np.array([loop_current], dtype=np.float64),
            np.array([radius], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            (obs[:, 0], obs[:, 1], obs[:, 2]),
            par,
        )
    )
    a_ref = _loop_vector_potential_cartesian(radius, loop_current, obs, par)

    b = np.column_stack((bx, by, bz))
    a = np.column_stack((ax, ay, az))
    b_atol = np.max(np.abs(b_ref)) * 1e-12
    a_atol = np.max(np.abs(a_ref)) * 1e-12

    assert np.allclose(b, b_ref, rtol=1e-3, atol=b_atol)
    assert np.allclose(a, a_ref, rtol=1e-3, atol=a_atol)


def test_triangle_mesh_serial_vs_parallel():
    radius = 0.7312345987
    height = radius * 1e-3
    loop_current = 1.7
    nodes, triangles, s = _triangle_strip_mesh(radius, height, loop_current, nphi=128)
    obs = np.array(
        [
            [0.4, -0.2, 1.1],
            [1.7, 0.8, -0.5],
            [2.8, -0.7, 0.9],
            [3.6, 1.2, -1.4],
        ],
        dtype=np.float64,
    )

    b_serial = np.column_stack(
        cfsem.flux_density_triangle_mesh(obs, nodes, triangles, s, par=False)
    )
    b_parallel = np.column_stack(
        cfsem.flux_density_triangle_mesh(obs, nodes, triangles, s, par=True)
    )
    a_serial = np.column_stack(
        cfsem.vector_potential_triangle_mesh(obs, nodes, triangles, s, par=False)
    )
    a_parallel = np.column_stack(
        cfsem.vector_potential_triangle_mesh(obs, nodes, triangles, s, par=True)
    )

    assert np.allclose(b_serial, b_parallel, rtol=1e-12, atol=1e-12)
    assert np.allclose(a_serial, a_parallel, rtol=1e-12, atol=1e-12)


def test_triangle_mesh_invalid_inputs():
    nodes, triangles, s = _triangle_strip_mesh(0.7, 7e-4, 1.2, nphi=32)
    obs = np.array([[0.4, -0.2, 1.1], [1.7, 0.8, -0.5]], dtype=np.float64)

    with raises(ValueError, match="obs must have shape"):
        cfsem.flux_density_triangle_mesh(obs[:, :2], nodes, triangles, s)

    with raises(ValueError, match="triangles must have shape"):
        cfsem.vector_potential_triangle_mesh(obs, nodes, triangles[:, :2], s)

    bad_triangles = triangles.copy()
    bad_triangles[0, 0] = -1
    with raises(ValueError, match="nonnegative node indices"):
        cfsem.flux_density_triangle_mesh(obs, nodes, bad_triangles, s)

    with raises(ValueError, match="Unsupported triangle quadrature rule"):
        cfsem.vector_potential_triangle_mesh(obs, nodes, triangles, s, quad="bad")

    with raises(ValueError, match="Unsupported triangle quadrature rule"):
        cfsem.triangle_mesh_quadrature_points(nodes, triangles, quad="bad")
