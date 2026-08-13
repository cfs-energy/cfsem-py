"""Tests for stateless hierarchical field solvers."""

import numpy as np
import pytest
import scipy.sparse as sp

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


def _assert_vec_zero(result):
    for component in result.field:
        np.testing.assert_array_equal(component, np.zeros_like(component))


def _add_vec3(lhs, rhs):
    return tuple(left + right for left, right in zip(lhs, rhs, strict=True))


def _assert_diagnostics(result, nsource, ntarget):
    assert result.diagnostics.construction_time >= 0.0
    assert result.diagnostics.evaluation_time >= 0.0
    assert result.diagnostics.source_count == nsource
    assert result.diagnostics.target_count == ntarget
    assert result.diagnostics.source_tree is not None
    assert len(result.diagnostics.source_tree) == 9
    assert result.diagnostics.source_tree[0].size > 0
    assert result.diagnostics.source_tree[7].shape == result.diagnostics.source_tree[0].shape
    assert result.diagnostics.source_tree[8].shape == result.diagnostics.source_tree[0].shape
    assert result.diagnostics.accepted_levels is not None
    assert result.diagnostics.accepted_levels.shape == (ntarget,)
    interaction_map = result.diagnostics.near_field_interaction_map
    assert sp.isspmatrix_csc(interaction_map)
    assert interaction_map.shape == (nsource, ntarget)
    assert interaction_map.has_canonical_format


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
        loc, moment, obs, outer_radius, theta=0.0, par=False, extra_diagnostics=True
    )
    result_a = cfsem.vector_potential_dipole_hierarchical(
        loc, moment, obs, outer_radius, theta=0.0, par=True, extra_diagnostics=True
    )
    _assert_vec_close(result_b, direct_b)
    _assert_vec_close(result_a, direct_a)
    _assert_diagnostics(result_b, nsource=3, ntarget=4)
    _assert_diagnostics(result_a, nsource=3, ntarget=4)
    interaction_map = result_a.diagnostics.near_field_interaction_map
    assert interaction_map.nnz == 12
    np.testing.assert_array_equal(interaction_map.indices, np.tile(np.arange(3), 4))
    np.testing.assert_array_equal(interaction_map.indptr, np.arange(0, 13, 3))

    out = (np.empty_like(obs[0]), np.empty_like(obs[0]), np.empty_like(obs[0]))
    returned = cfsem.flux_density_dipole_hierarchical(
        loc, moment, obs, outer_radius, theta=0.0, par=False, out=out
    )
    _assert_returns_output_views(returned, out)
    _assert_vec_close(out, direct_b)
    assert returned.diagnostics.near_field_interaction_map is None


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
        obs, xyzfil, dlxyzfil, current, wire_radius, theta=0.0, par=False, extra_diagnostics=True
    )
    result_a = cfsem.vector_potential_linear_filament_hierarchical(
        obs, xyzfil, dlxyzfil, current, wire_radius, theta=0.0, par=True, extra_diagnostics=True
    )
    _assert_vec_close(result_b, direct_b)
    _assert_vec_close(result_a, direct_a)
    _assert_diagnostics(result_b, nsource=2, ntarget=3)
    _assert_diagnostics(result_a, nsource=2, ntarget=3)


@pytest.mark.parametrize("par", [False, True])
def test_hierarchical_skip_decomposes_near_and_far_fields(par):
    loc = (
        np.array([0.0, 0.1, 0.2, 10.0, 10.1, 10.2]),
        np.zeros(6),
        np.zeros(6),
    )
    moment = (np.zeros(6), np.ones(6), np.ones(6))
    obs = (np.array([0.4]), np.array([0.3]), np.array([0.2]))
    # Finite source bounds force the nearby leaves down the direct path while
    # the compact cluster near x=10 is still accepted as far field.
    outer_radius = np.full(6, 0.05)

    full = cfsem.vector_potential_dipole_hierarchical(
        loc, moment, obs, outer_radius, theta=0.2, par=par, extra_diagnostics=True
    )
    far_only = cfsem.vector_potential_dipole_hierarchical(
        loc,
        moment,
        obs,
        outer_radius,
        theta=0.2,
        par=par,
        skip="near",
        extra_diagnostics=True,
    )
    near_only = cfsem.vector_potential_dipole_hierarchical(
        loc,
        moment,
        obs,
        outer_radius,
        theta=0.2,
        par=par,
        skip="far",
        extra_diagnostics=True,
    )
    diagnostics_only = cfsem.vector_potential_dipole_hierarchical(
        loc,
        moment,
        obs,
        outer_radius,
        theta=0.2,
        par=par,
        skip="both",
        extra_diagnostics=True,
    )

    _assert_vec_close(full, _add_vec3(far_only.field, near_only.field))
    assert any(np.any(component != 0.0) for component in far_only.field)
    assert any(np.any(component != 0.0) for component in near_only.field)
    _assert_vec_zero(diagnostics_only)
    for filtered in (far_only, near_only, diagnostics_only):
        np.testing.assert_array_equal(
            filtered.diagnostics.near_field_interaction_map.indices,
            full.diagnostics.near_field_interaction_map.indices,
        )
        np.testing.assert_array_equal(
            filtered.diagnostics.near_field_interaction_map.indptr,
            full.diagnostics.near_field_interaction_map.indptr,
        )

    zero_only = cfsem.vector_potential_dipole_hierarchical(
        loc, moment, obs, outer_radius, theta=0.2, par=par, skip="both"
    )
    _assert_vec_zero(zero_only)
    assert zero_only.diagnostics.near_field_interaction_map is None


def test_hierarchical_skip_rejects_unknown_value():
    loc = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
    moment = (np.array([0.0]), np.array([0.0]), np.array([1.0]))
    obs = (np.array([1.0]), np.array([0.0]), np.array([0.0]))

    with pytest.raises(ValueError, match="Unsupported hierarchical skip value"):
        cfsem.vector_potential_dipole_hierarchical(loc, moment, obs, np.zeros(1), skip="not-an-interaction")


def test_near_field_interaction_map_uses_original_source_rows():
    loc = (
        np.array([10.0, 0.0]),
        np.zeros(2),
        np.zeros(2),
    )
    moment = (np.zeros(2), np.ones(2), np.ones(2))
    obs = (
        np.array([0.0, 10.0, 5.0]),
        np.zeros(3),
        np.zeros(3),
    )
    result = cfsem.vector_potential_dipole_hierarchical(
        loc,
        moment,
        obs,
        np.zeros(2),
        theta=0.5,
        par=False,
        extra_diagnostics=True,
    )

    interaction_map = result.diagnostics.near_field_interaction_map
    np.testing.assert_array_equal(interaction_map.indices, np.array([1, 0]))
    np.testing.assert_array_equal(interaction_map.indptr, np.array([0, 1, 2, 2]))
    np.testing.assert_array_equal(interaction_map.data, np.ones(2, dtype=bool))


def test_all_hierarchical_methods_accept_skip_both():
    loc = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
    moment = (np.array([0.0]), np.array([0.0]), np.array([1.0]))
    obs = (np.array([1.0]), np.array([0.2]), np.array([0.3]))
    radius = np.zeros(1)
    xyzfil = loc
    dlxyzfil = (np.array([0.0]), np.array([0.0]), np.array([0.5]))
    current = np.ones(1)
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    stream_function = np.array([0.0, 1.0, 0.25])
    obs_rows = np.array([[0.2, 0.2, 0.5]])

    results = (
        cfsem.flux_density_dipole_hierarchical(loc, moment, obs, radius, theta=0.0, skip="both"),
        cfsem.vector_potential_dipole_hierarchical(loc, moment, obs, radius, theta=0.0, skip="both"),
        cfsem.flux_density_linear_filament_hierarchical(
            obs, xyzfil, dlxyzfil, current, radius, theta=0.0, skip="both"
        ),
        cfsem.vector_potential_linear_filament_hierarchical(
            obs, xyzfil, dlxyzfil, current, radius, theta=0.0, skip="both"
        ),
        cfsem.flux_density_triangle_mesh_hierarchical(
            obs_rows, nodes, triangles, stream_function, theta=0.0, skip="both"
        ),
        cfsem.vector_potential_triangle_mesh_hierarchical(
            obs_rows, nodes, triangles, stream_function, theta=0.0, skip="both"
        ),
    )
    for result in results:
        _assert_vec_zero(result)


def test_skip_both_zeroes_and_returns_supplied_output_arrays():
    loc = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
    moment = (np.array([0.0]), np.array([0.0]), np.array([1.0]))
    obs = (np.array([1.0, 2.0]), np.zeros(2), np.zeros(2))
    out = tuple(np.full(2, np.nan) for _ in range(3))

    result = cfsem.vector_potential_dipole_hierarchical(
        loc, moment, obs, np.zeros(1), skip="both", out=out
    )

    _assert_returns_output_views(result, out)
    _assert_vec_zero(result)


@pytest.mark.parametrize("par", [False, True])
def test_near_field_map_drives_sparse_filament_inductance(par):
    nsegment = 16
    phi = np.linspace(0.0, 2.0 * np.pi, nsegment + 1)
    points = np.column_stack((np.cos(phi), np.sin(phi), np.zeros_like(phi)))
    starts = points[:-1]
    deltas = np.diff(points, axis=0)
    midpoints = starts + 0.5 * deltas
    xyzfil = _tuple_columns(starts)
    dlxyzfil = _tuple_columns(deltas)
    targets = _tuple_columns(midpoints)
    current = np.linspace(0.8, 1.2, nsegment)
    wire_radius = np.full(nsegment, 0.02)

    full = cfsem.vector_potential_linear_filament_hierarchical(
        targets,
        xyzfil,
        dlxyzfil,
        current,
        wire_radius,
        theta=0.35,
        par=par,
    )
    diagnostics_only = cfsem.vector_potential_linear_filament_hierarchical(
        targets,
        xyzfil,
        dlxyzfil,
        current,
        wire_radius,
        theta=0.35,
        par=par,
        extra_diagnostics=True,
        skip="both",
    )
    far_only = cfsem.vector_potential_linear_filament_hierarchical(
        targets,
        xyzfil,
        dlxyzfil,
        current,
        wire_radius,
        theta=0.35,
        par=par,
        skip="near",
    )
    near_only = cfsem.vector_potential_linear_filament_hierarchical(
        targets,
        xyzfil,
        dlxyzfil,
        current,
        wire_radius,
        theta=0.35,
        par=par,
        skip="far",
    )
    _assert_vec_close(full, _add_vec3(far_only.field, near_only.field))
    _assert_vec_zero(diagnostics_only)

    interaction_map = diagnostics_only.diagnostics.near_field_interaction_map
    assert 0 < interaction_map.nnz < nsegment * nsegment
    sparse_inductance = cfsem.inductance_linear_filaments_sparse(
        xyzfil,
        dlxyzfil,
        xyzfil,
        dlxyzfil,
        interaction_map,
        wire_radius_src=wire_radius,
        par=par,
    )
    dense_inductance = cfsem.inductance_linear_filaments(
        xyzfil,
        dlxyzfil,
        xyzfil,
        dlxyzfil,
        wire_radius_src=wire_radius,
        par=par,
        output="matrix",
    )
    for target in range(nsegment):
        start, end = sparse_inductance.indptr[target : target + 2]
        rows = sparse_inductance.indices[start:end]
        np.testing.assert_allclose(
            sparse_inductance.data[start:end],
            dense_inductance[rows, target],
            rtol=1e-14,
            atol=1e-18,
        )

    expected_contraction = np.zeros(nsegment)
    for target in range(nsegment):
        start, end = interaction_map.indptr[target : target + 2]
        rows = interaction_map.indices[start:end]
        expected_contraction[target] = dense_inductance[rows, target] @ current[rows]
    np.testing.assert_allclose(
        sparse_inductance.T @ current,
        expected_contraction,
        rtol=1e-14,
        atol=1e-18,
    )


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
        obs,
        xyzfil,
        dlxyzfil,
        current,
        wire_radius,
        theta=0.0,
        construction_method="morton_lbvh",
        par=False,
        extra_diagnostics=True,
    )
    _assert_vec_close(result, direct)
    _assert_diagnostics(result, nsource=3, ntarget=3)

    with pytest.raises(ValueError, match="Unsupported hierarchical construction method"):
        cfsem.flux_density_linear_filament_hierarchical(
            obs,
            xyzfil,
            dlxyzfil,
            current,
            wire_radius,
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
            obs,
            xyzfil,
            dlxyzfil,
            current,
            wire_radius,
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

    direct_b = cfsem.flux_density_triangle_mesh(obs, nodes, triangles, stream_function, par=False)
    direct_a = cfsem.vector_potential_triangle_mesh(obs, nodes, triangles, stream_function, par=False)
    result_b = cfsem.flux_density_triangle_mesh_hierarchical(
        obs,
        nodes,
        triangles,
        stream_function,
        theta=0.0,
        par=False,
        extra_diagnostics=True,
    )
    result_a = cfsem.vector_potential_triangle_mesh_hierarchical(
        obs,
        nodes,
        triangles,
        stream_function,
        theta=0.0,
        par=True,
        extra_diagnostics=True,
    )
    _assert_vec_close(result_b, direct_b)
    _assert_vec_close(result_a, direct_a)
    _assert_diagnostics(result_b, nsource=2, ntarget=3)
    _assert_diagnostics(result_a, nsource=2, ntarget=3)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        cfsem.vector_potential_triangle_mesh_hierarchical(
            obs,
            nodes,
            triangles,
            stream_function,
            quad="dunavant3",
        )


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
        _tuple_columns(obs),
        outer_radius,
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
        cfsem.flux_density_dipole_hierarchical(loc, bad_moment, obs, outer_radius, theta=0.0, par=False)


def test_direct_wrapper_rejects_array_without_coordinate_dimension():
    obs = np.zeros((2, 2))
    xyzfil = (np.zeros(1), np.zeros(1), np.zeros(1))
    dlxyzfil = (np.ones(1), np.zeros(1), np.zeros(1))
    current = np.ones(1)
    wire_radius = np.zeros(1)

    with pytest.raises(ValueError, match="one dimension of length 3"):
        cfsem.flux_density_linear_filament(obs, xyzfil, dlxyzfil, current, wire_radius, par=False)
