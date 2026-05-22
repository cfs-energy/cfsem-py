"""Tests for stateless hierarchical field solvers."""

import numpy as np
import pytest

import cfsem


def _assert_vec_close(actual, expected, rtol=1e-12, atol=1e-18):
    if hasattr(actual, "field"):
        actual = actual.field
    if hasattr(expected, "field"):
        expected = expected.field
    for actual_component, expected_component in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_component, expected_component, rtol=rtol, atol=atol)


def _tuple_columns(values):
    return (
        np.ascontiguousarray(values[:, 0]),
        np.ascontiguousarray(values[:, 1]),
        np.ascontiguousarray(values[:, 2]),
    )


def _assert_returns_output_views(returned, out):
    if hasattr(returned, "field"):
        returned = returned.field
    for returned_component, out_component in zip(returned, out, strict=True):
        assert np.shares_memory(returned_component, out_component)


def _assert_diagnostics(result, nsource, ntarget):
    assert result.diagnostics.construction_time >= 0.0
    assert result.diagnostics.evaluation_time >= 0.0
    assert result.diagnostics.source_count == nsource
    assert result.diagnostics.target_count == ntarget
    assert result.diagnostics.source_tree is not None
    assert result.diagnostics.source_tree[0].size > 0
    assert result.diagnostics.accepted_levels is not None
    assert result.diagnostics.accepted_levels.shape == (ntarget,)


def test_hierarchical_dipoles_match_direct():
    loc = (
        np.array([0.0, 0.5, -0.25]),
        np.array([0.0, 0.2, 0.1]),
        np.array([0.0, 0.1, 0.3]),
    )
    moment = (
        np.array([0.0, 0.2, 0.5]),
        np.array([0.0, 0.1, -0.1]),
        np.array([1.0, 0.0, 0.25]),
    )
    outer_radius = np.zeros(3)
    obs = (
        np.array([1.0, 1.3, -0.7, 0.1]),
        np.array([0.0, -0.4, 0.9, 1.2]),
        np.array([0.5, 0.2, -0.2, 0.7]),
    )

    direct_b = cfsem.flux_density_dipole(loc, moment, obs, par=False, outer_radius=outer_radius)
    direct_a = cfsem.vector_potential_dipole(loc, moment, obs, par=False, outer_radius=outer_radius)
    result_b = cfsem.flux_density_dipole_hierarchical(
        loc, moment, outer_radius, obs, theta=0.0, par=False, extra_diagnostics=True
    )
    result_a = cfsem.vector_potential_dipole_hierarchical(
        loc, moment, outer_radius, obs, theta=0.0, par=True, extra_diagnostics=True
    )
    _assert_vec_close(result_b, direct_b)
    _assert_vec_close(result_a, direct_a)
    _assert_diagnostics(result_b, nsource=3, ntarget=4)
    _assert_diagnostics(result_a, nsource=3, ntarget=4)

    out = (np.empty_like(obs[0]), np.empty_like(obs[0]), np.empty_like(obs[0]))
    returned = cfsem.flux_density_dipole_hierarchical(
        loc, moment, outer_radius, obs, theta=0.0, par=False, out=out
    )
    _assert_returns_output_views(returned, out)
    _assert_vec_close(out, direct_b)


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

    direct_b = cfsem.flux_density_linear_filament(obs, xyzfil, dlxyzfil, current, wire_radius, par=False)
    direct_a = cfsem.vector_potential_linear_filament(obs, xyzfil, dlxyzfil, current, wire_radius, par=False)
    result_b = cfsem.flux_density_linear_filament_hierarchical(
        xyzfil, dlxyzfil, current, wire_radius, obs, theta=0.0, par=False, extra_diagnostics=True
    )
    result_a = cfsem.vector_potential_linear_filament_hierarchical(
        xyzfil, dlxyzfil, current, wire_radius, obs, theta=0.0, par=True, extra_diagnostics=True
    )
    _assert_vec_close(result_b, direct_b)
    _assert_vec_close(result_a, direct_a)
    _assert_diagnostics(result_b, nsource=2, ntarget=3)
    _assert_diagnostics(result_a, nsource=2, ntarget=3)


def test_hierarchical_construction_method_is_exposed():
    xyzfil = (
        np.array([0.0, 0.5, -0.2]),
        np.array([0.0, 0.2, 0.1]),
        np.array([0.0, 0.1, 0.4]),
    )
    dlxyzfil = (
        np.array([0.0, 0.1, 0.2]),
        np.array([0.4, -0.2, 0.1]),
        np.array([0.2, 0.5, -0.3]),
    )
    current = np.array([2.0, -1.5, 0.7])
    wire_radius = np.zeros(3)
    obs = (
        np.array([1.0, 1.3, -0.7]),
        np.array([0.0, -0.4, 0.9]),
        np.array([0.5, 0.2, -0.2]),
    )

    direct = cfsem.flux_density_linear_filament(obs, xyzfil, dlxyzfil, current, wire_radius, par=False)
    result = cfsem.flux_density_linear_filament_hierarchical(
        xyzfil,
        dlxyzfil,
        current,
        wire_radius,
        obs,
        theta=0.0,
        construction_method="morton_lbvh",
        par=False,
        extra_diagnostics=True,
    )
    _assert_vec_close(result, direct)
    _assert_diagnostics(result, nsource=3, ntarget=3)

    with pytest.raises(ValueError, match="Unsupported hierarchical construction method"):
        cfsem.flux_density_linear_filament_hierarchical(
            xyzfil,
            dlxyzfil,
            current,
            wire_radius,
            obs,
            construction_method="not-a-method",
            par=False,
        )


def test_hierarchical_rejects_noncontiguous_output():
    xyzfil = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
    dlxyzfil = (np.array([0.1]), np.array([0.0]), np.array([0.0]))
    current = np.array([1.0])
    wire_radius = np.zeros(1)
    obs = (np.array([0.0, 0.2, 0.4]), np.zeros(3), np.ones(3))
    out = (np.empty(6)[::2], np.empty(6)[::2], np.empty(6)[::2])

    with pytest.raises(ValueError, match="output arrays must be contiguous"):
        cfsem.flux_density_linear_filament_hierarchical(
            xyzfil,
            dlxyzfil,
            current,
            wire_radius,
            obs,
            theta=0.0,
            par=False,
            out=out,
        )


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
    obs = np.array([[0.25, 0.25, 0.5], [1.5, -0.2, 0.8], [-0.4, 1.2, -0.7]])

    direct_b = cfsem.flux_density_triangle_mesh(
        obs, nodes, triangles, stream_function, par=False, quad="dunavant3"
    )
    direct_a = cfsem.vector_potential_triangle_mesh(
        obs, nodes, triangles, stream_function, par=False, quad="dunavant3"
    )
    result_b = cfsem.flux_density_triangle_mesh_hierarchical(
        nodes,
        triangles,
        stream_function,
        obs,
        theta=0.0,
        quad="dunavant3",
        par=False,
        extra_diagnostics=True,
    )
    result_a = cfsem.vector_potential_triangle_mesh_hierarchical(
        nodes,
        triangles,
        stream_function,
        obs,
        theta=0.0,
        quad="dunavant3",
        par=True,
        extra_diagnostics=True,
    )
    _assert_vec_close(result_b, direct_b)
    _assert_vec_close(result_a, direct_a)
    _assert_diagnostics(result_b, nsource=2, ntarget=3)
    _assert_diagnostics(result_a, nsource=2, ntarget=3)


def test_hierarchical_accepts_tuple_columns_from_2d_inputs():
    loc = np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.1]])
    obs = np.array([[1.0, 0.0, 0.5], [0.4, -0.3, 0.2], [-0.2, 0.7, -0.1]])
    moment = (
        np.array([0.0, 0.2]),
        np.array([0.0, -0.1]),
        np.array([1.0, 0.3]),
    )
    outer_radius = np.zeros(2)

    expected = cfsem.vector_potential_dipole(loc, moment, obs, par=False, outer_radius=outer_radius)
    result = cfsem.vector_potential_dipole_hierarchical(
        _tuple_columns(loc),
        moment,
        outer_radius,
        _tuple_columns(obs),
        theta=0.0,
        par=False,
    )
    _assert_vec_close(result, expected)


def test_coordinate_tuple_conversion_rejects_invalid_shape():
    loc = (np.zeros(2), np.zeros(2), np.zeros(2))
    obs = (np.zeros(3), np.zeros(3), np.zeros(3))
    bad_moment = (np.zeros(2), np.zeros(3), np.zeros(2))
    outer_radius = np.zeros(2)

    with pytest.raises(ValueError, match="component arrays must have matching lengths"):
        cfsem.flux_density_dipole_hierarchical(
            loc, bad_moment, outer_radius, obs, theta=0.0, par=False
        )


def test_direct_wrapper_rejects_array_without_coordinate_dimension():
    obs = np.zeros((2, 2))
    xyzfil = (np.zeros(1), np.zeros(1), np.zeros(1))
    dlxyzfil = (np.ones(1), np.zeros(1), np.zeros(1))
    current = np.ones(1)
    wire_radius = np.zeros(1)

    with pytest.raises(ValueError, match="one dimension of length 3"):
        cfsem.flux_density_linear_filament(obs, xyzfil, dlxyzfil, current, wire_radius, par=False)
