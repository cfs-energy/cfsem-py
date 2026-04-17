"""Axisymmetric finite-difference flux solve helpers."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.constants import mu_0
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized

from .bindings import flux_circular_filament, gs_operator_order4

# Finite-difference coeffs
# https://en.wikipedia.org/wiki/Finite_difference_coefficient
_DDX_CENTRAL_ORDER4 = np.array(
    [
        # 4th-order central difference for first derivative
        (-2, 1 / 12),
        (-1, -2 / 3),
        # (0, 0.0),
        (1, 2 / 3),
        (2, -1 / 12),
    ]
)

_DDX_FWD_ORDER4 = np.array(
    [
        # 4th-order forward difference for first derivative
        (0, -25 / 12),
        (1, 4),
        (2, -3),
        (3, 4 / 3),
        (4, -1 / 4),
    ]
)

_DDX_BWD_ORDER4 = -_DDX_FWD_ORDER4  # Reverse & flip signs


def gradient_order4(z: NDArray, xmesh: NDArray, ymesh: NDArray) -> tuple[NDArray, NDArray]:
    """
    Calculate gradient by 4th-order finite difference.

    `numpy.gradient` exists and is fast and convenient, but only uses a second-order difference,
    which produces unacceptable error in B-fields (well over 1% for typical geometries).

    ## Errors

        * If the input grids are not regular
        * If any input grid dimensions have size less than 6

    ## References

        * [1] “Finite difference coefficient,” Wikipedia. Aug. 22, 2023.
              Accessed: Mar. 29, 2024. [Online].
              Available: https://en.wikipedia.org/w/index.php?title=Finite_difference_coefficient

    Args:
        z: [<xunits>] 2D array of values on which to calculate the gradient
        xmesh: [m] 2D array of coordinates of first dimension
        ymesh: [m] 2D array of coordinates of second dimension

    Returns:
        (dzdx, dzdy) [<xunits>/m] 2D arrays of gradient components
    """
    nx, ny = z.shape
    assert nx >= 6 and ny >= 6, "gradient_order4 requires each grid dimension to have at least 6 points"
    dx = xmesh[1][0] - xmesh[0][0]
    dy = ymesh[0][1] - ymesh[0][0]

    # Check regular grid assumption
    assert np.all(
        np.abs(np.diff(xmesh[:, 0]) - dx) / dx < 1e-6
    ), "This method is only implemented for a regular grid"
    assert np.all(
        np.abs(np.diff(ymesh[0, :]) - dy) / dy < 1e-6
    ), "This method is only implemented for a regular grid"

    accumulator_dtype = np.result_type(z, np.float64)
    dzdx = np.zeros(z.shape, dtype=accumulator_dtype)
    for offs, w in _DDX_CENTRAL_ORDER4:
        start = int(2 + offs)
        end = int(nx - 2 + offs)
        dzdx[2:-2, :] += w * z[start:end, :] / dx  # Central difference on interior points
    left_rows = np.arange(2)
    for offs, w in _DDX_FWD_ORDER4:
        dzdx[0:2, :] += w * z[left_rows + int(offs), :] / dx  # One-sided difference on left side
    right_rows = np.arange(nx - 2, nx)
    for offs, w in _DDX_BWD_ORDER4:
        dzdx[-2:, :] += w * z[right_rows + int(offs), :] / dx  # One-sided difference on right side

    dzdy = np.zeros(z.shape, dtype=accumulator_dtype)
    for offs, w in _DDX_CENTRAL_ORDER4:
        start = int(2 + offs)
        end = int(ny - 2 + offs)
        dzdy[:, 2:-2] += w * z[:, start:end] / dy  # Interior points
    bottom_cols = np.arange(2)
    for offs, w in _DDX_FWD_ORDER4:
        dzdy[:, 0:2] += w * z[:, bottom_cols + int(offs)] / dy  # One-sided difference on bottom
    top_cols = np.arange(ny - 2, ny)
    for offs, w in _DDX_BWD_ORDER4:
        dzdy[:, -2:] += w * z[:, top_cols + int(offs)] / dy  # One-sided difference on top

    return dzdx, dzdy


def calc_flux_density_from_flux(psi: NDArray, rmesh: NDArray, zmesh: NDArray) -> tuple[NDArray, NDArray]:
    """
    Back-calculate B-field from poloidal flux per Wesson eqn 3.2.2 by 4th-order finite difference,
    modified to use total poloidal flux instead of flux per radian.

    This avoids an expensive sum over filamentized contributions at the expense of some numerical error.

    # Errors

        * If the input grids are not regular
        * If any input grid dimensions have size less than 6
        * If `rmesh` contains non-positive radii

    # References

        * [1] J. Wesson, Tokamaks. Oxford, New York: Clarendon Press, 1987.

    Args:
        psi: [Wb] poloidal flux
        rmesh: [m] 2D r-coordinates
        zmesh: [m] 2D z-coordinates

    Returns:
        (br, bz) [T] 2D arrays of poloidal flux density
    """
    assert not np.any(rmesh <= 0.0), "rmesh must be strictly positive"

    dpsidr, dpsidz = gradient_order4(psi, rmesh, zmesh)

    r_inv = rmesh**-1

    br = -r_inv * dpsidz / (2.0 * np.pi)  # [T]
    bz = r_inv * dpsidr / (2.0 * np.pi)  # [T]

    return (br, bz)


def flux_solver(grids: tuple[NDArray, NDArray]) -> Callable[[NDArray], NDArray]:
    """
    Linear solver for extracting a flux field from a toroidal current density distribution
    using a 4th-order finite difference approximation of the Grad-Shafranov PDE.
    For `jtor` toroidal current density shaped like (nr, nz), call like `psi = flux_solver(rhs)`
    to get `psi` in [Wb] or [V-s], where `rhs = -2.0 * np.pi * mu_0 * rmesh * jtor` with the boundary
    values set to the circular-filament solved flux.

    Args:
        grids: [m] regular 1D r,z grids. The fourth-order Grad-Shafranov stencil
            requires strictly increasing grids with positive radius and at least
            7 points on each axis.

    Returns:
        solver: factorized solver for Grad-Shafranov differential operator
    """
    # Build Grad-Shafranov Delta* linear operator for finite difference
    # as a sparse matrix
    _ = _check_regular(grids, min_points=7)
    rgrid, zgrid = grids
    nr = rgrid.size
    nz = zgrid.size
    vals, rows, cols = gs_operator_order4(*grids)
    operator = csc_matrix((vals, (rows, cols)), shape=(nr * nz, nr * nz))
    # Store LU factorization of operator matrix to allow fast, reusable
    # solves using different right-hand-side (different current density)
    return factorized(operator)


def _validate_flux_mesh_inputs(
    grids: tuple[NDArray, NDArray],
    meshes: tuple[NDArray, NDArray],
    current_density: NDArray,
    tol: float = 1e-6,
) -> None:
    """Best-effort validation that the mesh arrays are consistent with the FD grid ordering."""
    rgrid, zgrid = grids
    rmesh, zmesh = meshes
    expected_shape = (rgrid.size, zgrid.size)
    transposed_shape = (zgrid.size, rgrid.size)

    if (
        rmesh.shape != expected_shape
        or zmesh.shape != expected_shape
        or current_density.shape != expected_shape
    ):
        assert not (
            rgrid.size != zgrid.size
            and rmesh.shape == transposed_shape
            and zmesh.shape == transposed_shape
            and current_density.shape == transposed_shape
        ), "meshes and current_density appear transposed; use np.meshgrid(..., indexing='ij')"
        raise AssertionError(f"meshes and current_density must all have shape {expected_shape}")

    # If the two axes have different lengths, the expected `indexing="ij"` layout
    # is no longer ambiguous, so we can validate the mesh-axis content directly.
    if rgrid.size != zgrid.size:
        r_axis_matches = np.allclose(rmesh[:, 0], rgrid, rtol=tol, atol=tol)
        z_axis_matches = np.allclose(zmesh[0, :], zgrid, rtol=tol, atol=tol)
        assert (
            r_axis_matches and z_axis_matches
        ), "meshes must be consistent with grids and use np.meshgrid(..., indexing='ij')"


def solve_flux_axisymmetric(
    grids: tuple[NDArray, NDArray],
    meshes: tuple[NDArray, NDArray],
    current_density: NDArray,
    solver: Callable[[NDArray], NDArray] | None = None,
) -> NDArray:
    """
    Calculate the flux field associated with a given toroidal current density distribution,
    by solving the Grad-Shafranov PDE.

    This calculation is most commonly used for the plasma, but is in fact more general,
    and applies to anything with an equivalent toroidal current density and axisymmetry.

    Args:
        grids: [m] 1D r,z regular coordinate grids. The fourth-order solve requires
            strictly increasing grids with positive radius and at least 7 points
            on each axis.
        meshes: [m] 2D meshgrids made from grids like np.meshgrid(*grids, indexing="ij")
        current_density: [A/m^2], shape (nr, nz), toroidal current density on finite-difference mesh
            with zero values on the finite-difference boundary
        solver: Optionally, provide a pre-initialized linear solver. See `cfsem.flux_solver`.

    Returns:
        poloidal flux field, [Wb] with shape (nr, nz)
    """
    _ = _check_regular(grids, min_points=7)
    _validate_flux_mesh_inputs(grids, meshes, current_density)

    # Build the differential operator, if needed
    solver = solver or flux_solver(grids)

    # Unpack and filter down to just useful inputs
    dr, dz = _check_regular(grids, min_points=7)  # [m] grid spacing
    area = dr * dz  # [m^2]
    rmesh, zmesh = meshes  # [m]
    assert not (
        np.any(current_density[0, :] != 0.0)
        or np.any(current_density[-1, :] != 0.0)
        or np.any(current_density[:, 0] != 0.0)
        or np.any(current_density[:, -1] != 0.0)
    ), "current_density must be zero on the finite-difference boundary"
    nonzero_inds = np.where(current_density != 0.0)
    current_density_nonzero = np.ascontiguousarray(current_density[nonzero_inds])  # [A/m^2]
    rmesh_nonzero = np.ascontiguousarray(rmesh[nonzero_inds])  # [m]
    zmesh_nonzero = np.ascontiguousarray(zmesh[nonzero_inds])  # [m]
    # Solve `Delta* @ psi = -mu_0 * 2pi * rmesh * jtor`
    #   Set up right-hand-side of Grad-Shafranov
    rhs = -(2.0 * np.pi * mu_0) * rmesh * current_density  # [Wb/m^2]
    #   Set flux boundary condition
    #   For most relevant grid sizes (up to 500 X 500), doing the O(N^3)
    #   circular-filament flux calc is faster than the linear solve
    #   and therefore faster than doing an extra fixed-boundary linear solve
    #   in order to use Von Hagenow's asymptotically-O(N^2logN) method.
    ifil = (area * current_density_nonzero).flatten()  # [A] plasma filament current
    rfil = rmesh_nonzero.flatten()
    zfil = zmesh_nonzero.flatten()
    for s in [[0, ...], [-1, ...], [..., 0], [..., -1]]:  # All boundary slices
        rhs[s[0], s[1]] = flux_circular_filament(ifil, rfil, zfil, rmesh[s[0], s[1]], zmesh[s[0], s[1]])
    #   Do the actual linear solve
    psi = solver(rhs.flatten()).reshape(rmesh.shape)  # [Wb]

    return psi


def _check_regular(grids: tuple[NDArray, NDArray], tol=1e-6, min_points: int = 2) -> tuple[float, float]:
    """Check that grids are regular, strictly increasing, and at positive radius."""
    rgrid, zgrid = grids
    assert (
        rgrid.size >= min_points and zgrid.size >= min_points
    ), f"rgrid and zgrid must each have at least {min_points} points"
    assert not np.any(rgrid <= 0.0), "rgrid must be strictly positive"
    drs = np.diff(rgrid)
    dzs = np.diff(zgrid)
    assert not np.any(drs <= 0.0), "rgrid must be strictly increasing"
    assert not np.any(dzs <= 0.0), "zgrid must be strictly increasing"
    drmean = float(np.mean(drs))
    dzmean = float(np.mean(dzs))
    assert np.all(np.abs(drs - drmean) / drmean < tol), "rgrid must be regular"
    assert np.all(np.abs(dzs - dzmean) / dzmean < tol), "zgrid must be regular"

    return drmean, dzmean  # [m]


__all__ = [
    "gradient_order4",
    "calc_flux_density_from_flux",
    "flux_solver",
    "solve_flux_axisymmetric",
]
