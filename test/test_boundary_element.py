"""Tests of triangle boundary-element interfaces"""

import numpy as np
from pytest import approx, mark, raises

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
    return np.column_stack((-a_phi * np.sin(phi), a_phi * np.cos(phi), np.zeros(obs.shape[0])))


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
        out[i] = (s[i0] * (n2 - n1) + s[i1] * (n0 - n2) + s[i2] * (n1 - n0)) / (2.0 * area)
    return out


def _contract_force_mapping(
    fx: np.ndarray,
    fy: np.ndarray,
    fz: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.float64)
    return np.column_stack((fx @ coeffs, fy @ coeffs, fz @ coeffs))


def _force_from_bfield_on_target(
    points: np.ndarray,
    weights: np.ndarray,
    j_tgt: np.ndarray,
    b_qp: np.ndarray,
) -> np.ndarray:
    return np.sum(np.cross(j_tgt[:, None, :], b_qp) * weights[:, :, None], axis=1)


def _linear_filament_loop(
    radius: float,
    z: float,
    ndiscr: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    phi = np.linspace(0.0, 2.0 * np.pi, ndiscr)
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    zvals = np.full_like(x, z)
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(zvals)
    return ((x[:-1], y[:-1], zvals[:-1]), (dx, dy, dz))


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
    ax, ay, az = cfsem.vector_potential_triangle_mesh(obs, nodes, triangles, s, par=par, quad=quad)
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

    b_serial = np.column_stack(cfsem.flux_density_triangle_mesh(obs, nodes, triangles, s, par=False))
    b_parallel = np.column_stack(cfsem.flux_density_triangle_mesh(obs, nodes, triangles, s, par=True))
    a_serial = np.column_stack(cfsem.vector_potential_triangle_mesh(obs, nodes, triangles, s, par=False))
    a_parallel = np.column_stack(cfsem.vector_potential_triangle_mesh(obs, nodes, triangles, s, par=True))

    assert np.allclose(b_serial, b_parallel, rtol=1e-12, atol=1e-12)
    assert np.allclose(a_serial, a_parallel, rtol=1e-12, atol=1e-12)


def test_triangle_mesh_inductance_matrix_strip_self_inductance_against_wien_and_lyle():
    major_radius = 0.5  # [m]
    minor_radius = 5e-3  # [m]
    height = 2.0 * minor_radius  # [m]
    nodes, triangles, s = _triangle_strip_mesh(major_radius, height, 1.0, nphi=96)
    target_current = 3.7  # [A]

    # Calculate full-mesh nodal mutual inductance matrix
    lmat = cfsem.triangle_mesh_inductance_matrix(nodes, triangles, par=False, quad="gl3")  # [H]

    # Total self-inductance is 2 * stored energy per amp^2,
    # so we can calculate the self-inductance by taking 2 times the stored energy at unit current.
    # The default values of `s` on the test strip correspond to unit current.
    l_from_matrix = float(s @ lmat @ s)  # [H]

    # Stored energy is 1/2 s^T @ L @ s.
    # To make sure we handle non-unit current, we test with some
    s_at_target_current = target_current * s  # [A]
    energy_from_matrix = float(0.5 * s_at_target_current @ lmat @ s_at_target_current)  # [J]

    # Analytic self-inductance and energy calculations for comparison
    l_wien = float(cfsem.self_inductance_circular_ring_wien(major_radius, minor_radius))  # [H]
    l_lyle = float(
        cfsem.self_inductance_lyle6(
            r=major_radius,
            dr=height / 2.0,
            dz=height,
            n=1.0,
        )
    )  # [H]
    energy_wien = 0.5 * l_wien * target_current**2  # [J]
    energy_lyle = 0.5 * l_lyle * target_current**2  # [J]

    # Lyle's calc should match a little better than Wien's,
    # because a strip is more like a rectangle than a circle.
    assert lmat.shape == (nodes.shape[0], nodes.shape[0])
    assert np.allclose(lmat, lmat.T, rtol=1e-12, atol=1e-12)
    assert l_from_matrix == approx(l_wien, rel=0.12)
    assert l_from_matrix == approx(l_lyle, rel=0.09)
    assert energy_from_matrix == approx(energy_wien, rel=0.12)
    assert energy_from_matrix == approx(energy_lyle, rel=0.09)


@mark.parametrize("par", [True, False])
def test_triangle_mesh_force_mapping_against_direct_target_quadrature(par):
    radius = 0.63
    height = radius * 1e-3
    nodes_src, triangles_src, s_src = _triangle_strip_mesh(radius, height, 1.1, nphi=24, z_center=-0.19)
    nodes_tgt, triangles_tgt, s_tgt = _triangle_strip_mesh(radius, height, 0.8, nphi=24, z_center=0.27)

    fx, fy, fz = cfsem.triangle_mesh_force_mapping(
        nodes_src,
        triangles_src,
        nodes_tgt,
        triangles_tgt,
        s_tgt,
        par=par,
        quad="gl3",
    )
    tri_forces = _contract_force_mapping(fx, fy, fz, s_src)

    points, weights = cfsem.triangle_mesh_quadrature_points(nodes_tgt, triangles_tgt, quad="gl3")
    j_tgt = cfsem.triangle_mesh_current_density(nodes_tgt, triangles_tgt, s_tgt)
    b_qp = np.column_stack(
        cfsem.flux_density_triangle_mesh(
            points.reshape(-1, 3), nodes_src, triangles_src, s_src, par=False, quad="gl3"
        )
    ).reshape(points.shape)
    tri_forces_ref = _force_from_bfield_on_target(points, weights, j_tgt, b_qp)

    assert fx.shape == (triangles_tgt.shape[0], nodes_src.shape[0])
    assert fy.shape == (triangles_tgt.shape[0], nodes_src.shape[0])
    assert fz.shape == (triangles_tgt.shape[0], nodes_src.shape[0])
    assert np.allclose(tri_forces, tri_forces_ref, rtol=1e-11, atol=1e-12)


@mark.parametrize("par", [True, False])
def test_triangle_mesh_self_force_mapping_shapes_and_serial_parallel_agree(par):
    radius = 0.71
    height = radius * 1e-3
    nodes, triangles, s = _triangle_strip_mesh(radius, height, 1.0, nphi=32)

    fx, fy, fz = cfsem.triangle_mesh_self_force_mapping(nodes, triangles, s, par=par, quad="gl3")
    tri_forces = _contract_force_mapping(fx, fy, fz, s)
    total_force = np.sum(tri_forces, axis=0)
    fx_ref, fy_ref, fz_ref = cfsem.triangle_mesh_self_force_mapping(
        nodes, triangles, s, par=not par, quad="gl3"
    )
    tri_forces_ref = _contract_force_mapping(fx_ref, fy_ref, fz_ref, s)

    assert fx.shape == (triangles.shape[0], nodes.shape[0])
    assert fy.shape == (triangles.shape[0], nodes.shape[0])
    assert fz.shape == (triangles.shape[0], nodes.shape[0])
    assert np.all(np.isfinite(total_force))
    assert np.allclose(tri_forces, tri_forces_ref, rtol=1e-12, atol=1e-12)


@mark.parametrize("par", [True, False])
def test_triangle_mesh_force_mappings_from_other_source_models(par):
    radius = 0.58
    height = radius * 1e-3
    nodes_tgt, triangles_tgt, s_tgt = _triangle_strip_mesh(radius, height, 0.9, nphi=20, z_center=0.14)
    points, weights = cfsem.triangle_mesh_quadrature_points(nodes_tgt, triangles_tgt, quad="gl3")
    j_tgt = cfsem.triangle_mesh_current_density(nodes_tgt, triangles_tgt, s_tgt)
    xyzp = (points[..., 0].ravel(), points[..., 1].ravel(), points[..., 2].ravel())

    xyzfil = (
        np.array([0.15, -0.12], dtype=np.float64),
        np.array([-0.35, 0.28], dtype=np.float64),
        np.array([0.42, -0.31], dtype=np.float64),
    )
    dlxyzfil = (
        np.array([0.27, -0.18], dtype=np.float64),
        np.array([0.16, 0.21], dtype=np.float64),
        np.array([-0.11, 0.14], dtype=np.float64),
    )
    ifil = np.array([1.4, -0.7], dtype=np.float64)
    wire_radius = np.array([2.5e-3, 1.5e-3], dtype=np.float64)

    fx_lin, fy_lin, fz_lin = cfsem.triangle_mesh_force_mapping_from_linear_filaments(
        xyzfil,
        dlxyzfil,
        nodes_tgt,
        triangles_tgt,
        s_tgt,
        wire_radius=wire_radius,
        par=par,
        quad="gl3",
    )
    tri_forces_lin = _contract_force_mapping(fx_lin, fy_lin, fz_lin, ifil)
    b_lin = np.column_stack(
        cfsem.flux_density_linear_filament(
            xyzp, xyzfil, dlxyzfil, ifil, wire_radius=wire_radius, par=False
        )
    ).reshape(points.shape)
    tri_forces_lin_ref = _force_from_bfield_on_target(points, weights, j_tgt, b_lin)

    rfil = np.array([0.47, 0.63], dtype=np.float64)
    zfil = np.array([-0.16, 0.29], dtype=np.float64)
    icirc = np.array([0.8, -1.1], dtype=np.float64)
    fx_circ, fy_circ, fz_circ = cfsem.triangle_mesh_force_mapping_from_circular_filaments(
        rfil, zfil, nodes_tgt, triangles_tgt, s_tgt, par=par, quad="gl3"
    )
    tri_forces_circ = _contract_force_mapping(fx_circ, fy_circ, fz_circ, icirc)
    b_circ = np.column_stack(
        cfsem.flux_density_circular_filament_cartesian(
            icirc,
            rfil,
            zfil,
            xyzp,
            False,
        )
    ).reshape(points.shape)
    tri_forces_circ_ref = _force_from_bfield_on_target(points, weights, j_tgt, b_circ)

    loc = (
        np.array([0.21, -0.38], dtype=np.float64),
        np.array([0.12, 0.25], dtype=np.float64),
        np.array([-0.27, 0.34], dtype=np.float64),
    )
    moment_dir = (
        np.array([0.0, 0.8], dtype=np.float64),
        np.array([0.0, -0.4], dtype=np.float64),
        np.array([1.0, 0.45], dtype=np.float64),
    )
    dip_amp = np.array([0.35, -0.6], dtype=np.float64)
    outer_radius = np.array([0.0, 0.0], dtype=np.float64)
    fx_dip, fy_dip, fz_dip = cfsem.triangle_mesh_force_mapping_from_dipoles(
        loc,
        moment_dir,
        nodes_tgt,
        triangles_tgt,
        s_tgt,
        par=par,
        outer_radius=outer_radius,
        quad="gl3",
    )
    tri_forces_dip = _contract_force_mapping(fx_dip, fy_dip, fz_dip, dip_amp)
    moment = tuple(dip_amp * comp for comp in moment_dir)
    b_dip = np.column_stack(
        cfsem.flux_density_dipole(
            loc,
            moment,
            xyzp,
            par=False,
            outer_radius=outer_radius,
        )
    ).reshape(points.shape)
    tri_forces_dip_ref = _force_from_bfield_on_target(points, weights, j_tgt, b_dip)

    assert fx_lin.shape == (triangles_tgt.shape[0], ifil.size)
    assert fy_lin.shape == (triangles_tgt.shape[0], ifil.size)
    assert fz_lin.shape == (triangles_tgt.shape[0], ifil.size)
    assert np.allclose(tri_forces_lin, tri_forces_lin_ref, rtol=1e-11, atol=1e-12)

    assert fx_circ.shape == (triangles_tgt.shape[0], icirc.size)
    assert fy_circ.shape == (triangles_tgt.shape[0], icirc.size)
    assert fz_circ.shape == (triangles_tgt.shape[0], icirc.size)
    assert np.allclose(tri_forces_circ, tri_forces_circ_ref, rtol=1e-11, atol=1e-12)

    assert fx_dip.shape == (triangles_tgt.shape[0], dip_amp.size)
    assert fy_dip.shape == (triangles_tgt.shape[0], dip_amp.size)
    assert fz_dip.shape == (triangles_tgt.shape[0], dip_amp.size)
    assert np.allclose(tri_forces_dip, tri_forces_dip_ref, rtol=1e-11, atol=1e-12)


@mark.parametrize("par", [True, False])
def test_triangle_mesh_force_on_strip_against_linear_filament_loop(par):
    radius = 0.63
    height = radius * 1e-3
    z_loop = -0.18
    z_strip = 0.24
    loop_current = 1.3
    nseg = 256
    nodes_tgt, triangles_tgt, s_tgt = _triangle_strip_mesh(radius, height, 0.9, nphi=32, z_center=z_strip)
    xyzfil, dlxyzfil = _linear_filament_loop(radius, z_loop, nseg)
    ifil = np.full(nseg - 1, loop_current, dtype=np.float64)

    fx, fy, fz = cfsem.triangle_mesh_force_mapping_from_linear_filaments(
        xyzfil,
        dlxyzfil,
        nodes_tgt,
        triangles_tgt,
        s_tgt,
        wire_radius=0.0,
        par=par,
        quad="gl3",
    )
    tri_forces = _contract_force_mapping(fx, fy, fz, ifil)

    points, weights = cfsem.triangle_mesh_quadrature_points(nodes_tgt, triangles_tgt, quad="gl3")
    j_tgt = cfsem.triangle_mesh_current_density(nodes_tgt, triangles_tgt, s_tgt)
    j_qp = np.repeat(j_tgt[:, None, :], points.shape[1], axis=1).reshape(-1, 3)
    fx_ref, fy_ref, fz_ref = cfsem.body_force_density_linear_filament(
        xyzfil,
        dlxyzfil,
        ifil,
        (points[..., 0].ravel(), points[..., 1].ravel(), points[..., 2].ravel()),
        (j_qp[:, 0], j_qp[:, 1], j_qp[:, 2]),
        wire_radius=0.0,
        par=False,
    )
    tri_forces_ref = np.sum(
        np.column_stack((fx_ref, fy_ref, fz_ref)).reshape(points.shape) * weights[:, :, None],
        axis=1,
    )

    total_force = np.sum(tri_forces, axis=0)
    total_force_ref = np.sum(tri_forces_ref, axis=0)

    assert fx.shape == (triangles_tgt.shape[0], ifil.size)
    assert fy.shape == (triangles_tgt.shape[0], ifil.size)
    assert fz.shape == (triangles_tgt.shape[0], ifil.size)
    assert np.allclose(tri_forces, tri_forces_ref, rtol=1e-11, atol=1e-12)
    assert np.allclose(total_force, total_force_ref, rtol=1e-11, atol=1e-12)


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
