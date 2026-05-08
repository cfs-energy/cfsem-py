"""Tests for reusable hierarchical field solvers."""

import numpy as np

import cfsem


def _assert_vec_close(actual, expected, rtol=1e-12, atol=1e-18):
    for actual_component, expected_component in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_component, expected_component, rtol=rtol, atol=atol)


def test_hierarchical_dipoles_match_direct_and_reuse_tree():
    loc = (
        np.array([0.0, 0.5, -0.25]),
        np.array([0.0, 0.2, 0.1]),
        np.array([0.0, 0.1, 0.3]),
    )
    moment0 = (
        np.array([0.0, 0.2, 0.5]),
        np.array([0.0, 0.1, -0.1]),
        np.array([1.0, 0.0, 0.25]),
    )
    moment1 = (2.0 * moment0[0], 2.0 * moment0[1], 2.0 * moment0[2])
    outer_radius = np.zeros(3)
    obs = (
        np.array([1.0, 1.3, -0.7, 0.1]),
        np.array([0.0, -0.4, 0.9, 1.2]),
        np.array([0.5, 0.2, -0.2, 0.7]),
    )

    solver = cfsem.HierarchicalDipoles(theta=0.0)
    solver.build_targets(obs)
    solver.build_sources(loc, outer_radius)

    direct_b0 = cfsem.flux_density_dipole(loc, moment0, obs, par=False, outer_radius=outer_radius)
    direct_a0 = cfsem.vector_potential_dipole(loc, moment0, obs, par=False, outer_radius=outer_radius)
    _assert_vec_close(solver.flux_density(moment0, par=False), direct_b0)
    _assert_vec_close(solver.flux_density(moment0, par=True), direct_b0)
    _assert_vec_close(solver.vector_potential(moment0, par=False), direct_a0)

    direct_b1 = cfsem.flux_density_dipole(loc, moment1, obs, par=False, outer_radius=outer_radius)
    out = (np.empty_like(obs[0]), np.empty_like(obs[0]), np.empty_like(obs[0]))
    solver.flux_density_into(moment1, out, par=False)
    _assert_vec_close(out, direct_b1)


def test_hierarchical_linear_filaments_match_direct():
    xyzfil = (
        np.array([0.0, 0.5]),
        np.array([0.0, 0.2]),
        np.array([0.0, 0.1]),
    )
    dlxyzfil = (
        np.array([0.0, 0.1]),
        np.array([0.4, -0.2]),
        np.array([0.2, 0.5]),
    )
    current = np.array([2.0, -1.5])
    wire_radius = np.zeros(2)
    obs = (
        np.array([1.0, 1.3, -0.7]),
        np.array([0.0, -0.4, 0.9]),
        np.array([0.5, 0.2, -0.2]),
    )

    solver = cfsem.HierarchicalLinearFilaments(theta=0.0)
    solver.build(xyzfil, dlxyzfil, wire_radius, obs)

    direct_b = cfsem.flux_density_linear_filament(obs, xyzfil, dlxyzfil, current, wire_radius, par=False)
    direct_a = cfsem.vector_potential_linear_filament(obs, xyzfil, dlxyzfil, current, wire_radius, par=False)
    _assert_vec_close(solver.flux_density(current, par=False), direct_b)
    _assert_vec_close(solver.flux_density(current, par=True), direct_b)
    _assert_vec_close(solver.vector_potential(current, par=False), direct_a)


def test_hierarchical_boundary_elements_match_direct_triangle_mesh():
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    stream_function = np.array([0.0, 1.0, 0.25, -0.5])
    obs = (
        np.array([0.25, 1.5, -0.4]),
        np.array([0.25, -0.2, 1.2]),
        np.array([0.5, 0.8, -0.7]),
    )
    obs_array = np.column_stack(obs)

    solver = cfsem.HierarchicalBoundaryElements(theta=0.0, quad="dunavant3")
    solver.build(nodes, triangles, obs)

    direct_b = cfsem.flux_density_triangle_mesh(
        obs_array, nodes, triangles, stream_function, par=False, quad="dunavant3"
    )
    direct_a = cfsem.vector_potential_triangle_mesh(
        obs_array, nodes, triangles, stream_function, par=False, quad="dunavant3"
    )
    _assert_vec_close(solver.flux_density(stream_function, par=False), direct_b)
    _assert_vec_close(solver.flux_density(stream_function, par=True), direct_b)
    _assert_vec_close(solver.vector_potential(stream_function, par=False), direct_a)
