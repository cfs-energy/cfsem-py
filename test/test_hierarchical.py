"""Tests for reusable hierarchical field solvers."""

import numpy as np
import pytest

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


def test_hierarchical_one_shot_wrappers_match_reusable_solvers():
    loc = (
        np.array([0.0, 0.5]),
        np.array([0.0, 0.2]),
        np.array([0.0, 0.1]),
    )
    moment = (
        np.array([0.0, 0.2]),
        np.array([0.0, 0.1]),
        np.array([1.0, -0.3]),
    )
    outer_radius = np.zeros(2)
    obs = (
        np.array([1.0, 1.3, -0.7]),
        np.array([0.0, -0.4, 0.9]),
        np.array([0.5, 0.2, -0.2]),
    )
    dipole_solver = cfsem.HierarchicalDipoles(theta=0.0)
    dipole_solver.build(loc, obs, outer_radius=outer_radius)
    _assert_vec_close(
        cfsem.flux_density_dipole_hierarchical(
            loc,
            moment,
            obs,
            theta=0.0,
            par=False,
            outer_radius=outer_radius,
        ),
        dipole_solver.flux_density(moment, par=False),
    )
    _assert_vec_close(
        cfsem.vector_potential_dipole_hierarchical(
            loc, moment, obs, theta=0.0, par=False, outer_radius=outer_radius
        ),
        dipole_solver.vector_potential(moment, par=False),
    )

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
    filament_solver = cfsem.HierarchicalLinearFilaments(theta=0.0)
    filament_solver.build(xyzfil, dlxyzfil, wire_radius, obs)
    _assert_vec_close(
        cfsem.flux_density_linear_filament_hierarchical(
            obs, xyzfil, dlxyzfil, current, wire_radius, theta=0.0, par=False
        ),
        filament_solver.flux_density(current, par=False),
    )
    _assert_vec_close(
        cfsem.vector_potential_linear_filament_hierarchical(
            obs, xyzfil, dlxyzfil, current, wire_radius, theta=0.0, par=False
        ),
        filament_solver.vector_potential(current, par=False),
    )

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
    obs_array = np.column_stack(obs)
    boundary_solver = cfsem.HierarchicalBoundaryElements(theta=0.0, quad="dunavant3")
    boundary_solver.build(nodes, triangles, obs)
    _assert_vec_close(
        cfsem.flux_density_triangle_mesh_hierarchical(
            obs_array, nodes, triangles, stream_function, theta=0.0, par=False, quad="dunavant3"
        ),
        boundary_solver.flux_density(stream_function, par=False),
    )
    _assert_vec_close(
        cfsem.vector_potential_triangle_mesh_hierarchical(
            obs_array, nodes, triangles, stream_function, theta=0.0, par=False, quad="dunavant3"
        ),
        boundary_solver.vector_potential(stream_function, par=False),
    )


def test_hierarchical_dipole_wrapper_update_and_into_methods():
    loc = np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.1]])
    obs = np.array([[1.0, 0.0, 0.5], [0.4, -0.3, 0.2], [-0.2, 0.7, -0.1]])
    moment = (
        np.array([0.0, 0.2]),
        np.array([0.0, -0.1]),
        np.array([1.0, 0.3]),
    )

    solver = cfsem.HierarchicalDipoles(theta=0.0)
    solver.build(loc, obs)
    solver.update_sources(loc)
    solver.update_targets(obs)

    expected_a = cfsem.vector_potential_dipole(loc, moment, obs, par=False)
    out = (
        np.empty(obs.shape[0]),
        np.empty(obs.shape[0]),
        np.empty(obs.shape[0]),
    )
    solver.vector_potential_into(moment, out, par=False)
    _assert_vec_close(out, expected_a)


def test_hierarchical_linear_filament_wrapper_update_methods():
    xyzfil = np.array([[0.0, 0.0, 0.0], [0.5, 0.2, 0.1]])
    dlxyzfil = np.array([[0.0, 0.4, 0.2], [0.1, -0.2, 0.5]])
    current = np.array([2.0, -1.5])
    wire_radius = np.zeros(2)
    obs = np.array([[1.0, 0.0, 0.5], [1.3, -0.4, 0.2], [-0.7, 0.9, -0.2]])

    solver = cfsem.HierarchicalLinearFilaments(theta=0.0)
    solver.build_sources(xyzfil, dlxyzfil, wire_radius)
    solver.update_sources(xyzfil, dlxyzfil, wire_radius)
    solver.build_targets(obs)
    solver.update_targets(obs)

    direct_b = cfsem.flux_density_linear_filament(obs, xyzfil, dlxyzfil, current, wire_radius, par=False)
    direct_a = cfsem.vector_potential_linear_filament(obs, xyzfil, dlxyzfil, current, wire_radius, par=False)
    _assert_vec_close(solver.flux_density(current, par=False), direct_b)
    _assert_vec_close(solver.vector_potential(current, par=False), direct_a)


def test_hierarchical_boundary_element_wrapper_update_and_source_value_methods():
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
    triangle_values = (
        stream_function[triangles[:, 0]],
        stream_function[triangles[:, 1]],
        stream_function[triangles[:, 2]],
    )
    obs = np.array([[0.25, 0.25, 0.5], [1.5, -0.2, 0.8], [-0.4, 1.2, -0.7]])
    obs_tuple = (obs[:, 0], obs[:, 1], obs[:, 2])

    unbuilt = cfsem.HierarchicalBoundaryElements(theta=0.0, quad="dunavant3")
    with pytest.raises(ValueError, match="sources must be built"):
        unbuilt.flux_density(stream_function)

    solver = cfsem.HierarchicalBoundaryElements(theta=0.0, quad="dunavant3")
    solver.build_sources(nodes, triangles)
    solver.update_sources(nodes, triangles)
    solver.build_targets(obs)
    solver.update_targets(obs_tuple)

    direct_b = cfsem.flux_density_triangle_mesh(
        obs, nodes, triangles, stream_function, par=False, quad="dunavant3"
    )
    direct_a = cfsem.vector_potential_triangle_mesh(
        obs, nodes, triangles, stream_function, par=False, quad="dunavant3"
    )
    _assert_vec_close(solver.flux_density(stream_function, par=False), direct_b)
    _assert_vec_close(solver.flux_density(triangle_values, par=False), direct_b)
    _assert_vec_close(solver.vector_potential(stream_function, par=False), direct_a)


def test_coordinate_tuple_conversion_rejects_invalid_shape():
    solver = cfsem.HierarchicalDipoles(theta=0.0)
    loc = np.zeros((2, 3))
    obs = np.zeros((3, 2))
    bad_moment = np.zeros((2, 2))
    solver.build(loc, obs)

    with pytest.raises(ValueError, match="Expected a tuple of three coordinate arrays"):
        solver.flux_density(bad_moment)
