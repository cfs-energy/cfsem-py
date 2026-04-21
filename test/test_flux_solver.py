import numpy as np
from pytest import raises

import cfsem
from cfsem.flux_solver import gradient_order4

ValidationError = (AssertionError, ValueError)


def test_gradient_order4():
    """Make sure order 2 and order 4 gradients converge to the same value"""

    def f(x, y):
        # Some bivariate function without important features under the grid size
        return np.sin(x / 6.0) + np.cos(y / 6.0)

    # Coarse grid
    xgrid1 = np.linspace(-5.0, 5.0, 10)
    ygrid1 = np.linspace(-7.0, 7.0, 10)
    xmesh1, ymesh1 = np.meshgrid(xgrid1, ygrid1, indexing="ij")
    z1 = f(xmesh1, ymesh1)
    grad1_order2 = np.gradient(z1, xgrid1, ygrid1)
    grad1_order4 = gradient_order4(z1, xmesh1, ymesh1)
    assert np.allclose(grad1_order2, grad1_order4, rtol=0.1)  # Related, but not great match
    with raises(AssertionError):
        # Order 2 method should not be very good at this resolution
        assert np.allclose(grad1_order2, grad1_order4, rtol=0.01)

    # Fine grid
    xgrid2 = np.linspace(-5.0, 5.0, 100)
    ygrid2 = np.linspace(-7.0, 7.0, 100)
    xmesh2, ymesh2 = np.meshgrid(xgrid2, ygrid2, indexing="ij")
    z2 = f(xmesh2, ymesh2)
    grad2_order2 = np.gradient(z2, xgrid2, ygrid2)
    grad2_order4 = gradient_order4(z2, xmesh2, ymesh2)
    assert np.allclose(grad2_order2, grad2_order4, rtol=0.01)  # Reasonably close
    with raises(AssertionError):
        # There's still significant benefit to using the order 4 method here
        assert np.allclose(grad1_order2, grad1_order4, rtol=0.005)


def test_gradient_order4_supports_documented_minimum_grid_size():
    xgrid = np.linspace(-2.0, 2.0, 6)
    ygrid = np.linspace(-3.0, 3.0, 6)
    xmesh, ymesh = np.meshgrid(xgrid, ygrid, indexing="ij")
    z = 2.0 * xmesh - 3.0 * ymesh

    dzdx, dzdy = gradient_order4(z, xmesh, ymesh)

    assert np.allclose(dzdx, 2.0)
    assert np.allclose(dzdy, -3.0)


def test_gradient_order4_supports_integer_inputs_by_accumulating_in_float():
    xgrid = np.arange(6)
    ygrid = np.arange(6)
    xmesh, ymesh = np.meshgrid(xgrid, ygrid, indexing="ij")
    z = 2 * xmesh - 3 * ymesh

    dzdx, dzdy = gradient_order4(z, xmesh, ymesh)

    assert np.issubdtype(dzdx.dtype, np.floating)
    assert np.issubdtype(dzdy.dtype, np.floating)
    assert np.allclose(dzdx, 2.0)
    assert np.allclose(dzdy, -3.0)


def test_gradient_order4_rejects_5x5_grid():
    xgrid = np.linspace(-2.0, 2.0, 5)
    ygrid = np.linspace(-3.0, 3.0, 5)
    xmesh, ymesh = np.meshgrid(xgrid, ygrid, indexing="ij")
    z = 2.0 * xmesh - 3.0 * ymesh

    with raises(ValidationError, match="at least 6 points"):
        gradient_order4(z, xmesh, ymesh)


def test_calc_flux_density_from_flux_matches_direct_filament_field():
    rgrid = np.linspace(0.5, 1.5, 141)
    zgrid = np.linspace(-0.8, 0.8, 161)
    rmesh, zmesh = np.meshgrid(rgrid, zgrid, indexing="ij")

    ifil = np.array([1.0, -0.7])
    rfil = np.array([0.85, 1.2])
    zfil = np.array([-0.15, 0.25])

    psi = cfsem.flux_circular_filament(ifil, rfil, zfil, rmesh.flatten(), zmesh.flatten(), par=False).reshape(
        rmesh.shape
    )
    br_ref, bz_ref = cfsem.flux_density_circular_filament(
        ifil, rfil, zfil, rmesh.flatten(), zmesh.flatten(), par=False
    )
    br_ref = br_ref.reshape(rmesh.shape)
    bz_ref = bz_ref.reshape(rmesh.shape)

    br_from_psi, bz_from_psi = cfsem.calc_flux_density_from_flux(psi, rmesh, zmesh)

    mask = np.ones_like(rmesh, dtype=bool)
    for rf, zf in zip(rfil, zfil, strict=True):
        mask &= ((rmesh - rf) ** 2 + (zmesh - zf) ** 2) > (0.08**2)

    assert np.allclose(br_from_psi[mask], br_ref[mask], rtol=2e-2, atol=1e-7)
    assert np.allclose(bz_from_psi[mask], bz_ref[mask], rtol=2e-2, atol=1e-7)


def test_calc_flux_density_from_flux_rejects_axis_inclusive_mesh():
    rgrid = np.linspace(0.0, 1.0, 11)
    zgrid = np.linspace(-0.5, 0.5, 13)
    rmesh, zmesh = np.meshgrid(rgrid, zgrid, indexing="ij")
    psi = rmesh**2 + zmesh

    with raises(ValidationError, match="strictly positive"):
        cfsem.calc_flux_density_from_flux(psi, rmesh, zmesh)


def test_flux_solver_reuse_and_boundary_condition():
    rgrid = np.linspace(0.6, 1.4, 17)
    zgrid = np.linspace(-0.4, 0.4, 19)
    rmesh, zmesh = np.meshgrid(rgrid, zgrid, indexing="ij")
    current_density = np.zeros_like(rmesh)
    current_density[6:11, 7:12] = 2.5e6

    solver = cfsem.flux_solver((rgrid, zgrid))
    psi = cfsem.solve_flux_axisymmetric((rgrid, zgrid), (rmesh, zmesh), current_density)
    psi_reused = cfsem.solve_flux_axisymmetric((rgrid, zgrid), (rmesh, zmesh), current_density, solver=solver)
    assert np.allclose(psi, psi_reused)

    dr = rgrid[1] - rgrid[0]
    dz = zgrid[1] - zgrid[0]
    nonzero = np.where(current_density != 0.0)
    ifil = (dr * dz * current_density[nonzero]).flatten()
    rfil = rmesh[nonzero].flatten()
    zfil = zmesh[nonzero].flatten()

    for r_slice, z_slice in ((0, slice(None)), (-1, slice(None)), (slice(None), 0), (slice(None), -1)):
        psi_expected = cfsem.flux_circular_filament(
            ifil,
            rfil,
            zfil,
            rmesh[r_slice, z_slice],
            zmesh[r_slice, z_slice],
        )
        assert np.allclose(psi[r_slice, z_slice], psi_expected)


def test_flux_solver_rejects_non_increasing_and_axis_inclusive_grids():
    zgrid = np.linspace(-0.4, 0.4, 19)
    with raises(ValidationError, match="strictly positive"):
        cfsem.flux_solver((np.linspace(0.0, 1.4, 17), zgrid))

    with raises(ValidationError, match="rgrid must be strictly increasing"):
        cfsem.flux_solver((np.array([0.6, 0.65, 0.65, *np.linspace(0.7, 1.4, 14)]), zgrid))

    with raises(ValidationError, match="strictly increasing"):
        cfsem.flux_solver((np.linspace(0.6, 1.4, 17), zgrid[::-1]))

    irregular_rgrid = np.linspace(0.6, 1.4, 17)
    irregular_rgrid[5] += 1.0e-3
    with raises(ValidationError, match="rgrid must be regular"):
        cfsem.flux_solver((irregular_rgrid, zgrid))

    irregular_zgrid = np.linspace(-0.4, 0.4, 19)
    irregular_zgrid[7] += 1.0e-3
    with raises(ValidationError, match="zgrid must be regular"):
        cfsem.flux_solver((np.linspace(0.6, 1.4, 17), irregular_zgrid))


def test_flux_solver_rejects_grids_too_short_for_order4_gs_stencil():
    with raises(ValidationError, match="at least 7 points"):
        cfsem.flux_solver((np.linspace(0.6, 1.4, 6), np.linspace(-0.4, 0.4, 19)))

    with raises(ValidationError, match="at least 7 points"):
        cfsem.flux_solver((np.linspace(0.6, 1.4, 17), np.linspace(-0.4, 0.4, 6)))


def test_flux_solver_rejects_source_current_on_boundary():
    rgrid = np.linspace(0.6, 1.4, 17)
    zgrid = np.linspace(-0.4, 0.4, 19)
    rmesh, zmesh = np.meshgrid(rgrid, zgrid, indexing="ij")
    current_density = np.zeros_like(rmesh)
    current_density[0, 5] = 2.5e6

    with raises(ValidationError, match="finite-difference boundary"):
        cfsem.solve_flux_axisymmetric((rgrid, zgrid), (rmesh, zmesh), current_density)


def test_flux_solver_rejects_default_meshgrid_layout_when_unambiguous():
    rgrid = np.linspace(0.6, 1.4, 17)
    zgrid = np.linspace(-0.4, 0.4, 19)
    rmesh, zmesh = np.meshgrid(rgrid, zgrid)
    current_density = np.zeros_like(rmesh)

    with raises(ValidationError, match="transposed|indexing='ij'"):
        cfsem.solve_flux_axisymmetric((rgrid, zgrid), (rmesh, zmesh), current_density)


def test_flux_solver_rejects_default_meshgrid_layout_for_square_grids():
    rgrid = np.linspace(0.6, 1.4, 17)
    zgrid = np.linspace(-0.4, 0.4, 17)
    rmesh, zmesh = np.meshgrid(rgrid, zgrid)
    current_density = np.zeros_like(rmesh)

    with raises(ValidationError, match="consistent with grids|indexing='ij'"):
        cfsem.solve_flux_axisymmetric((rgrid, zgrid), (rmesh, zmesh), current_density)


def test_flux_solver_rejects_mesh_shape_and_axis_content_mismatches():
    rgrid = np.linspace(0.6, 1.4, 17)
    zgrid = np.linspace(-0.4, 0.4, 19)
    rmesh, zmesh = np.meshgrid(rgrid, zgrid, indexing="ij")
    current_density = np.zeros_like(rmesh)

    with raises(ValidationError, match=r"must all have shape \(17, 19\)"):
        cfsem.solve_flux_axisymmetric((rgrid, zgrid), (rmesh, zmesh), current_density[:-1, :])

    bad_rmesh = rmesh.copy()
    bad_rmesh[:, 0] += 0.1
    with raises(ValidationError, match="consistent with grids"):
        cfsem.solve_flux_axisymmetric((rgrid, zgrid), (bad_rmesh, zmesh), current_density)
