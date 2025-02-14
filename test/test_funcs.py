"""Brute-force calculations for testing more efficient methods"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.constants import mu_0
from scipy.special import ellipe, ellipk
from interpn import MultilinearRectilinear
import cfsem


def _self_inductance_filamentized(
    r, z, dr, dz, nt, nr, nz, mask: Optional[tuple[NDArray, NDArray, NDArray]] = None
) -> float:
    """
    USE L_lyle6 INSTEAD! This function is for verification of L_lyle6 in
    unit testing, and is very slow

    Estimate self-inductance of filamentized coil pack
    using an approximation for the self-inductance of filaments as
    the mutual inductance between the center of the region of the pack
    associated with that filament and its inner edge

    Args:
        r (float): radius, coil center
        z (float): axial position, coil center
        dr (float): width of coil pack
        dz (float): height of coil pack
        nt (float): turns
        nr (int): radial discretizations
        nz (int): axial discretizations
        mask: (rgrid, zgrid, vals) mask describing where to keep or discard filaments

    Returns:
        float: [H], estimated self-inductance
    """

    # Make estimate of same coil's self-inductance based on filamentized model
    fs = cfsem.filament_coil(r, z, dr, dz, nt, nr, nz)  # Generate filaments
    # Filter filaments based on mask
    if mask is not None:
        interp = MultilinearRectilinear.new([mask[0], mask[1]], mask[2].flatten())
        fs = np.array(
            [f for f in fs if interp.eval([np.array([f[0]]), np.array([f[1]])]) > 0.0]
        )
        fs[:, 2] *= nt / np.sum(fs[:, 2])

    # Mutual inductances between filaments, with erroneous elements on the diagonal
    nfil = fs.shape[0]
    M = np.zeros((nfil, nfil))
    for i in range(nfil):
        for j in range(nfil):
            if i != j:
                # If this is a mutual inductance between two different filaments, use that calc
                M[i, j] = _mutual_inductance_of_circular_filaments(
                    fs[i, :], fs[j, :]
                )  # [H]
            else:
                # Self-inductance of this filament
                major_radius_filament = fs[i, :][0]  # [m] Filament radius
                minor_radius_filament = (
                    (dr / nr) / 2
                )  # [m] Heuristic approximation for effective wire radius of filament
                num_turns = fs[i, :][2]  # [] Filament number of turns
                L_f = num_turns**2 * cfsem.self_inductance_circular_ring_wien(
                    major_radius_filament, minor_radius_filament
                )  # [H]
                M[i, j] = (
                    L_f  # [H] Rough estimate of self-inductance of conductor cross-section assigned to this filament
                )

    # Since all current and turns values are equal across the filaments, effective L is just the sum of all elements of M
    L = np.sum(np.sum(M))  # [H]

    return L  # [H]


def _flux_density_circular_filament_numerical(
    I: float, a: float, r: float, z: float, n: int = 100
) -> tuple[float, float]:
    """
    DO NOT USE - for unit testing, slow and not very precise

    Numerical integration approach to calculating B-field of a current loop

    Based on 8.02 notes
    https://web.mit.edu/8.02t/www/802TEAL3D/visualizations/coursenotes/modules/guide09.pdf
    appendix 1

    This function only exists to validate flux_density_circular_filament

    Args:
        I (float): [A] current
        a (float): [m] radius of current loop
        r (float): [m] r-coord of point to evaluate, relative to loop center
        z (float): [m] z-coord of point to evaluate, relative to loop axis
        n (float): [] number of discretization points
    """

    mu_0_over_4pi = 1e-7  # Collapse some algebra to reduce float error
    a0 = mu_0_over_4pi * I * a  # Leading term of both

    # Window out a small region near zero to avoid singularity
    phis = np.linspace(0.0 + 1e-8, 2.0 * np.pi - 1e-8, n)

    # Evaluate elliptic integrals numerically from calculus statement of PDE
    a1 = (a**2 + r**2 + z**2 - 2.0 * r * a * np.sin(phis)) ** -1.5  # Shared denominator

    Brs = a0 * z * np.sin(phis) * a1
    Br = np.trapezoid(x=phis, y=Brs)  # [T]

    Bzs = a0 * (a - r * np.sin(phis)) * a1
    Bz = np.trapezoid(x=phis, y=Bzs)  # [T]

    return Br, Bz  # [T]


def _mutual_inductance_of_circular_filaments(rzn1: NDArray, rzn2: NDArray) -> float:
    """
    Analytic mutual inductance between ideal
    cylindrically-symmetric coaxial filament pair.

    This is equivalent to taking the flux produced by each circular filament from
    either collection of filaments to the other. Mutual inductance is reflexive,
    so the order of the inputs is not important.

    Args:
        rzn1 (array): 3x1 array (r [m], z [m], n []) coordinates and number of turns
        rzn2 (array): 3x1 array (r [m], z [m], n []) coordinates and number of turns

    Returns:
        float: [H] mutual inductance
    """

    r1, z1, n1 = rzn1
    r2, z2, n2 = rzn2

    k2 = 4 * r1 * r2 / ((r1 + r2) ** 2 + (z1 - z2) ** 2)
    amp = 2 * mu_0 * r1 * r2 / np.sqrt((r1 + r2) ** 2 + (z1 - z2) ** 2)
    M0 = n1 * n2 * amp * ((2 - k2) * ellipk(k2) - 2 * ellipe(k2)) / k2

    return M0  # [H]


def _mutual_inductance_of_cylindrical_coils(f1: NDArray, f2: NDArray) -> float:
    """
    Analytical mutual inductance between two coaxial collections of ideal coils
    Each collection typically represents a discretized "real" cylindrically-symmetric
    coil of rectangular cross-section, but could have any cross-section as long as it
    maintains symmetry.

    Args:
        f1: m x 3 array of filament definitions like (r [m], z [m], n [])
        f2: m x 3 array of filament definitions like (r [m], z [m], n [])

    Returns:
        [H] mutual inductance of the two discretized coils
    """
    M = 0.0
    for i in range(f1.shape[0]):
        for j in range(f2.shape[0]):
            M += _mutual_inductance_of_circular_filaments(f1[i, :], f2[j, :])
    return M  # [H]


def _filament_coil_comprehension(
    r: float, z: float, w: float, h: float, nt: float, nr: int, nz: int
) -> NDArray:
    """
    Create an array of filaments from coil cross-section, evenly spaced
    _inside_ the winding pack. No filaments are coincident with the coil surface.

    Args:
        r: [m] radius, coil center
        z: [m] axial position, coil center
        w: [m] width of coil pack
        h: [m] height of coil pack
        nt: turns
        nr: radial discretizations
        nz: axial discretizations

    Returns:
        (nr*nz) x 3, (r,z,n) of each filament
    """

    rs = np.linspace(r - w * (nr - 1) / nr / 2, r + w * (nr - 1) / nr / 2, nr)
    zs = np.linspace(z - h * (nz - 1) / nz / 2, z + h * (nz - 1) / nz / 2, nz)

    rz = [(rr, zz) for rr in rs for zz in zs]
    R = [x[0] for x in rz]
    Z = [x[1] for x in rz]
    N = np.full_like(R, float(nt) / (nr * nz))
    filaments = np.dstack([R, Z, N]).reshape(nr * nz, 3)

    return filaments


def _filament_loop(
    r: float, z: float, ndiscr: int
) -> tuple[tuple[NDArray, NDArray, NDArray], tuple[NDArray, NDArray, NDArray]]:
    """Make linear filaments from a circular filament"""
    phi = np.linspace(0.0, 2.0 * np.pi, ndiscr)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = z * np.ones_like(x)

    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)

    return ((x, y, z), (dx, dy, dz))
