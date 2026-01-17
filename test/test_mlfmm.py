import numpy as np
import pytest
from pytest import mark

import cfsem


def _build_helix_sources():
    n_path = int(1e4)
    z = np.linspace(0.0, 1.0, n_path)
    path = (np.zeros_like(z), np.zeros_like(z), z)
    helix = cfsem.filament_helix_path(
        path=path,
        helix_start_offset=(5.0, 0.0, 0.0),
        twist_pitch=1.0,
        angle_offset=0.0,
    )
    helix = np.array(helix)
    xyzfil = helix[:, :-1]
    dlxyzfil = helix[:, 1:] - helix[:, :-1]
    ifil = np.ones(dlxyzfil.shape[1])
    eps = np.full_like(ifil, 1e-6)
    return xyzfil, dlxyzfil, ifil, eps


def _cube_targets(center, half_len, points_per_axis=5):
    grid = np.linspace(-half_len, half_len, points_per_axis)
    xg, yg, zg = np.meshgrid(grid, grid, grid, indexing="ij")
    x = xg.ravel() + center[0]
    y = yg.ravel() + center[1]
    z = zg.ravel() + center[2]
    return x, y, z


@mark.parametrize("half_len", [5.1, 100.0])
@mark.parametrize("direct_threshold", [0, int(1e3), int(1e12)])
@mark.parametrize("use_van_lanen", [False, True])
def test_mlfmm_fields_against_direct(half_len, direct_threshold, use_van_lanen):
    try:
        fields_linear_filament_mlfmm = cfsem.fields_linear_filament_mlfmm
    except AttributeError:
        pytest.skip("rat-mlfmm feature is not enabled in this build")

    xyzfil, dlxyzfil, ifil, eps = _build_helix_sources()
    center = (0.0, 0.0, 0.5)
    xyzp = _cube_targets(center=center, half_len=half_len, points_per_axis=5)

    b_direct = cfsem.flux_density_linear_filament(xyzp, xyzfil, dlxyzfil, ifil, par=False)
    a_direct = cfsem.vector_potential_linear_filament(xyzp, xyzfil, dlxyzfil, ifil, par=False)

    b_mlfmm, a_mlfmm = fields_linear_filament_mlfmm(
        xyzp,
        xyzfil,
        dlxyzfil,
        ifil,
        eps,
        use_van_lanen=use_van_lanen,
        direct_threshold=direct_threshold,
        order=None,
    )

    for got, exp in zip(b_mlfmm, b_direct, strict=True):
        assert np.allclose(got, exp, rtol=1e-6, atol=1e-10)
    for got, exp in zip(a_mlfmm, a_direct, strict=True):
        assert np.allclose(got, exp, rtol=1e-6, atol=1e-10)


def test_mlfmm_van_lanen_single_segment_matches_subdivided_direct():
    try:
        fields_linear_filament_mlfmm = cfsem.fields_linear_filament_mlfmm
    except AttributeError:
        pytest.skip("rat-mlfmm feature is not enabled in this build")

    xyzfil = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
    dlxyzfil = (np.array([1.0]), np.array([0.0]), np.array([0.0]))
    ifil = np.array([1.0])
    eps = np.array([1e-3])

    t = np.linspace(0.0, 1.0, 101)
    x0 = t[:-1]
    dx = np.diff(t)
    xyzfil_sub = (x0, np.zeros_like(x0), np.zeros_like(x0))
    dlxyzfil_sub = (dx, np.zeros_like(dx), np.zeros_like(dx))
    ifil_sub = np.full_like(dx, ifil[0])

    xyzp = (
        np.array([0.5, 0.5, 0.5, 0.25, 0.75]),
        np.array([0.2, -0.2, 0.3, -0.4, 0.1]),
        np.array([0.1, 0.1, -0.2, 0.2, -0.3]),
    )

    b_direct = cfsem.flux_density_linear_filament(xyzp, xyzfil_sub, dlxyzfil_sub, ifil_sub, par=False)
    a_direct = cfsem.vector_potential_linear_filament(xyzp, xyzfil_sub, dlxyzfil_sub, ifil_sub, par=False)

    b_mlfmm, a_mlfmm = fields_linear_filament_mlfmm(
        xyzp,
        xyzfil,
        dlxyzfil,
        ifil,
        eps,
        use_van_lanen=True,
        direct_threshold=1,
        order=None,
    )

    for got, exp in zip(b_mlfmm, b_direct, strict=True):
        assert np.allclose(got, exp, rtol=1e-6, atol=1e-10)
    for got, exp in zip(a_mlfmm, a_direct, strict=True):
        assert np.allclose(got, exp, rtol=1e-6, atol=1e-10)
