"""Tests for reusable hierarchical field solvers."""

import numpy as np
import pytest

import cfsem


def _assert_vec_close(actual, expected, rtol=1e-12, atol=1e-18):
    for actual_component, expected_component in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_component, expected_component, rtol=rtol, atol=atol)


def _tuple_columns(values):
    return (
        np.ascontiguousarray(values[:, 0]),
        np.ascontiguousarray(values[:, 1]),
        np.ascontiguousarray(values[:, 2]),
    )


def _assert_returns_output_views(returned, out):
    for returned_component, out_component in zip(returned, out, strict=True):
        assert np.shares_memory(returned_component, out_component)


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
    assert solver.accepted_source_levels(moment0, field="b").shape == obs[0].shape
    assert solver.source_tree_aabbs()[0].size > 0

    direct_b1 = cfsem.flux_density_dipole(loc, moment1, obs, par=False, outer_radius=outer_radius)
    out = (np.empty_like(obs[0]), np.empty_like(obs[0]), np.empty_like(obs[0]))
    _assert_returns_output_views(solver.flux_density(moment1, par=False, out=out), out)
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
    assert solver.accepted_source_levels(current, field="a").shape == obs[0].shape
    assert solver.source_tree_aabbs()[0].size > 0


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
    triangle_values = (
        stream_function[triangles[:, 0]],
        stream_function[triangles[:, 1]],
        stream_function[triangles[:, 2]],
    )
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
    _assert_vec_close(solver.flux_density(triangle_values, par=False), direct_b)
    _assert_vec_close(solver.flux_density(triangle_values, par=True), direct_b)
    _assert_vec_close(solver.vector_potential(triangle_values, par=False), direct_a)
    assert solver.accepted_source_levels(triangle_values, field="b").shape == obs[0].shape
    assert solver.source_tree_aabbs()[0].size > 0


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
            obs, xyzfil, dlxyzfil, current, 0.0, theta=0.0, par=False
        ),
        filament_solver.flux_density(current, par=False),
    )
    _assert_vec_close(
        cfsem.vector_potential_linear_filament_hierarchical(
            obs, xyzfil, dlxyzfil, current, 0.0, theta=0.0, par=False
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
    triangle_values = (
        stream_function[triangles[:, 0]],
        stream_function[triangles[:, 1]],
        stream_function[triangles[:, 2]],
    )
    obs_array = np.column_stack(obs)
    boundary_solver = cfsem.HierarchicalBoundaryElements(theta=0.0, quad="dunavant3")
    boundary_solver.build(nodes, triangles, obs)
    _assert_vec_close(
        cfsem.flux_density_triangle_mesh_hierarchical(
            obs_array, nodes, triangles, stream_function, theta=0.0, par=False, quad="dunavant3"
        ),
        boundary_solver.flux_density(triangle_values, par=False),
    )
    _assert_vec_close(
        cfsem.vector_potential_triangle_mesh_hierarchical(
            obs_array, nodes, triangles, stream_function, theta=0.0, par=False, quad="dunavant3"
        ),
        boundary_solver.vector_potential(triangle_values, par=False),
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
    loc_tuple = _tuple_columns(loc)
    obs_tuple = _tuple_columns(obs)
    solver.build(loc_tuple, obs_tuple)
    solver.update_sources(loc_tuple)
    solver.update_targets(obs_tuple)

    expected_a = cfsem.vector_potential_dipole(loc, moment, obs, par=False)
    out = (
        np.empty(obs.shape[0]),
        np.empty(obs.shape[0]),
        np.empty(obs.shape[0]),
    )
    _assert_returns_output_views(solver.vector_potential(moment, par=False, out=out), out)
    _assert_vec_close(out, expected_a)

    expected_b = cfsem.flux_density_dipole(loc, moment, obs, par=False)
    out_b = (
        np.empty(obs.shape[0]),
        np.empty(obs.shape[0]),
        np.empty(obs.shape[0]),
    )
    out_a = (
        np.empty(obs.shape[0]),
        np.empty(obs.shape[0]),
        np.empty(obs.shape[0]),
    )
    _assert_returns_output_views(solver.flux_density(moment, par=False, out=out_b), out_b)
    _assert_returns_output_views(solver.vector_potential(moment, par=False, out=out_a), out_a)
    _assert_vec_close(out_b, expected_b)
    _assert_vec_close(out_a, expected_a)


def test_hierarchical_linear_filament_wrapper_update_methods():
    xyzfil = np.array([[0.0, 0.0, 0.0], [0.5, 0.2, 0.1]])
    dlxyzfil = np.array([[0.0, 0.4, 0.2], [0.1, -0.2, 0.5]])
    current = np.array([2.0, -1.5])
    wire_radius = np.zeros(2)
    obs = np.array([[1.0, 0.0, 0.5], [1.3, -0.4, 0.2], [-0.7, 0.9, -0.2]])

    solver = cfsem.HierarchicalLinearFilaments(theta=0.0)
    xyzfil_tuple = _tuple_columns(xyzfil)
    dlxyzfil_tuple = _tuple_columns(dlxyzfil)
    obs_tuple = _tuple_columns(obs)
    solver.build_sources(xyzfil_tuple, dlxyzfil_tuple, wire_radius)
    solver.update_sources(xyzfil_tuple, dlxyzfil_tuple, wire_radius)
    solver.build_targets(obs_tuple)
    solver.update_targets(obs_tuple)

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
    obs_tuple = _tuple_columns(obs)

    unbuilt = cfsem.HierarchicalBoundaryElements(theta=0.0, quad="dunavant3")
    with pytest.raises(ValueError, match="sources have not been built"):
        unbuilt.flux_density(triangle_values)

    solver = cfsem.HierarchicalBoundaryElements(theta=0.0, quad="dunavant3")
    solver.build_sources(nodes, triangles)
    solver.update_sources(nodes, triangles)
    solver.build_targets(obs_tuple)
    solver.update_targets(obs_tuple)

    direct_b = cfsem.flux_density_triangle_mesh(
        obs, nodes, triangles, stream_function, par=False, quad="dunavant3"
    )
    direct_a = cfsem.vector_potential_triangle_mesh(
        obs, nodes, triangles, stream_function, par=False, quad="dunavant3"
    )
    _assert_vec_close(solver.flux_density(triangle_values, par=False), direct_b)
    _assert_vec_close(solver.vector_potential(triangle_values, par=False), direct_a)
    assert solver.accepted_source_levels(triangle_values, field="a").shape == obs_tuple[0].shape
    assert solver.source_tree_aabbs()[0].size > 0


def test_coordinate_tuple_conversion_rejects_invalid_shape():
    solver = cfsem.HierarchicalDipoles(theta=0.0)
    loc = (np.zeros(2), np.zeros(2), np.zeros(2))
    obs = (np.zeros(3), np.zeros(3), np.zeros(3))
    bad_moment = (np.zeros(2), np.zeros(3), np.zeros(2))
    solver.build(loc, obs)

    with pytest.raises(ValueError, match="component arrays must have matching lengths"):
        solver.flux_density(bad_moment)


def test_direct_wrapper_rejects_array_without_coordinate_dimension():
    obs = np.zeros((2, 2))
    xyzfil = (np.zeros(1), np.zeros(1), np.zeros(1))
    dlxyzfil = (np.ones(1), np.zeros(1), np.zeros(1))
    current = np.ones(1)
    wire_radius = np.zeros(1)

    with pytest.raises(ValueError, match="one dimension of length 3"):
        cfsem.flux_density_linear_filament(obs, xyzfil, dlxyzfil, current, wire_radius, par=False)
