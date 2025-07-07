"""Quasi-steady electromagnetics calcs"""

import numpy as np
from interpn import MulticubicRectilinear
from numpy.typing import NDArray

from cfsem.bindings import (
    body_force_density_circular_filament_cartesian,
    body_force_density_linear_filament,
    filament_helix_path,
    flux_circular_filament,
    flux_density_biot_savart,
    flux_density_circular_filament,
    flux_density_circular_filament_cartesian,
    flux_density_dipole,
    flux_density_linear_filament,
    gs_operator_order2,
    gs_operator_order4,
    inductance_piecewise_linear_filaments,
    mutual_inductance_circular_to_linear,
    rotate_filaments_about_path,
    vector_potential_circular_filament,
    vector_potential_linear_filament,
)
from cfsem.types import Array3xN

from .cfsem import ellipe, ellipk

MU_0 = 4.0 * np.pi * 1e-7 * (1.0 + 5.5e-10)
"""
Vacuum permeability with slight correction from latest measurements per NIST CODATA 2019 (SP-961),
https://www.physics.nist.gov/cuu/pdf/wall_2018.pdf .
"""

__all__ = [
    "flux_circular_filament",
    "flux_density_biot_savart",
    "flux_density_linear_filament",
    "flux_density_circular_filament",
    "gs_operator_order2",
    "gs_operator_order4",
    "filament_helix_path",
    "inductance_piecewise_linear_filaments",
    "self_inductance_piecewise_linear_filaments",
    "mutual_inductance_piecewise_linear_filaments",
    "flux_density_ideal_solenoid",
    "self_inductance_lyle6",
    "mutual_inductance_of_circular_filaments",
    "mutual_inductance_of_cylindrical_coils",
    "filament_coil",
    "self_inductance_circular_ring_wien",
    "self_inductance_annular_ring",
    "self_inductance_distributed_axisymmetric_conductor",
    "ellipe",
    "ellipk",
    "rotate_filaments_about_path",
    "vector_potential_linear_filament",
    "vector_potential_circular_filament",
    "flux_density_circular_filament_cartesian",
    "mutual_inductance_circular_to_linear",
    "flux_density_dipole",
    "body_force_density_circular_filament_cartesian",
    "body_force_density_linear_filament",
]


def self_inductance_piecewise_linear_filaments(xyzp: Array3xN) -> float:
    """
    Estimate the self-inductance of one piecewise-linear current filament.

    Uses Neumann's Formula for the mutual inductance of arbitrary loops
    for non-self-pairings, zeroes-out the contributions from self-pairings
    to resolve the thin-filament self-inductance singularity, and replaces the
    segment self-inductance term with an analytic value from [3].

    Assumes:

    * Thin, well-behaved filaments
    * Uniform current distribution within segments
        * Low frequency operation; no skin effect
          (which would reduce the segment self-field term)
    * Vacuum permeability everywhere
    * Each filament has a constant current in all segments
      (otherwise we need an inductance matrix)

    References:
        [1] “Inductance,” Wikipedia. Dec. 12, 2022. Accessed: Jan. 23, 2023. [Online].
            Available: <https://en.wikipedia.org/w/index.php?title=Inductance>

        [2] F. E. Neumann, “Allgemeine Gesetze der inducirten elektrischen Ströme,”
            Jan. 1846, doi: [10.1002/andp.18461430103](https://doi.org/10.1002/andp.18461430103)

        [3] R. Dengler, “Self inductance of a wire loop as a curve integral,”
            AEM, vol. 5, no. 1, p. 1, Jan. 2016, doi: [10.7716/aem.v5i1.331](https://doi.org/10.7716/aem.v5i1.331)

    Args:
        xyzp: [m] 3xN point series describing the filament

    Returns:
        [H] Scalar self-inductance
    """
    # Indexing numpy arrays here produces some `Any`-type hints and strips the element type
    # erroneously in the pyright output as of pyright 1.1.393.
    x, y, z = xyzp  # type: ignore
    xyzfil = (x[:-1], y[:-1], z[:-1])  # type: ignore
    dlxyzfil = (x[1:] - x[:-1], y[1:] - y[:-1], z[1:] - z[:-1])  # type: ignore

    self_inductance = inductance_piecewise_linear_filaments(
        xyzfil,  # type: ignore
        dlxyzfil,  # type: ignore
        xyzfil,  # type: ignore
        dlxyzfil,  # type: ignore
        True,
    )

    return self_inductance  # [H]


def mutual_inductance_piecewise_linear_filaments(
    xyz0: Array3xN,
    xyz1: Array3xN,
) -> float:
    """
    Estimate the mutual inductance between two piecewise-linear current filaments.

    Uses Neumann's Formula for the mutual inductance of arbitrary loops, which is
    originally from [2] and can be found in a more friendly format on wikipedia.

    Assumes:

    * Thin, well-behaved filaments
    * Vacuum permeability everywhere
    * Each filament has a constant current in all segments
      (otherwise we need an inductance matrix)
    * All segments between the two filaments are distinct; no identical pairs

    References:
        [1] “Inductance,” Wikipedia. Dec. 12, 2022. Accessed: Jan. 23, 2023. [Online].
            Available: <https://en.wikipedia.org/w/index.php?title=Inductance>

        [2] F. E. Neumann, “Allgemeine Gesetze der inducirten elektrischen Ströme,”
            Jan. 1846, doi: [10.1002/andp.18461430103](https://doi.org/10.1002/andp.18461430103)

    Args:
        xyz0: [m] 3xN point series describing the first filament
        xyz1: [m] 3xM point series describing the second filament

    Returns:
        [H] Scalar mutual inductance between the two filaments
    """
    # Indexing numpy arrays here produces some `Any`-type hints and strips the element type
    # erroneously in the pyright output as of pyright 1.1.393.
    x0, y0, z0 = xyz0  # type: ignore
    xyzfil0 = (x0[:-1], y0[:-1], z0[:-1])  # type: ignore
    dlxyzfil0 = (x0[1:] - x0[:-1], y0[1:] - y0[:-1], z0[1:] - z0[:-1])  # type: ignore

    x1, y1, z1 = xyz1  # type: ignore
    xyzfil1 = (x1[:-1], y1[:-1], z1[:-1])  # type: ignore
    dlxyzfil1 = (x1[1:] - x1[:-1], y1[1:] - y1[:-1], z1[1:] - z1[:-1])  # type: ignore

    inductance = inductance_piecewise_linear_filaments(
        xyzfil0,  # type: ignore
        dlxyzfil0,  # type: ignore
        xyzfil1,  # type: ignore
        dlxyzfil1,  # type: ignore
        False,
    )

    return inductance  # [H]


def flux_density_ideal_solenoid(current: float, num_turns: float, length: float) -> float:
    """
    Axial B-field on centerline of an ideal (infinitely long) solenoid.

    This calc converges reasonably well for coil L/D > 20.

    Args:
        current: [A] solenoid current
        num_turns: [#] number of conductor turns
        length: [m] length of winding pack

    Returns:
        [T] B-field on axis (in the direction aligned with the axis)
    """
    b_on_axis = MU_0 * num_turns * current / length
    return b_on_axis  # [T]


def self_inductance_lyle6(r: float, dr: float, dz: float, n: float) -> float:
    """
    Self-inductance of a cylindrically-symmetric coil of rectangular
    cross-section, estimated to 6th order.

    This estimate is viable up to an L/D of about 1.5, above which it
    rapidly accumulates error and eventually produces negative values.

    References:
        [1] T. R. Lyle,
        “IX. On the self-inductance of circular coils of rectangular section,”
        Philosophical Transactions of the Royal Society of London.
        Series A, Containing Papers of a Mathematical or Physical Character,
        vol. 213, no. 497-508, pp. 421-435, Jan. 1914, doi: [10.1098/rsta.1914.0009](https://doi.org/10.1098/rsta.1914.0009)

    Args:
        r: [m] radius, coil center
        dr: [m] radial width of coil
        dz: [m] cylindrical height of coil
        n: number of turns

    Returns:
        [H] self-inductance
    """

    assert dr < 2.0 * r, "Axisymmetric coil edges can't extend to negative R"
    assert dz / r <= 3.5, "Coil geometry outside domain of validity of Lyle's formula"

    # Guarantee 64-bit floats needed for 6th-order shape term
    a = np.float64(r)
    b = np.float64(dz)
    c = np.float64(dr)

    # Build up reusable terms for calculation of shape parameter
    d = np.sqrt(b**2 + c**2)  # [m] diagonal length
    u = ((b / c) ** 2) * 2 * np.log(d / b)
    v = ((c / b) ** 2) * 2 * np.log(d / c)
    w = (b / c) * np.arctan(c / b)
    ww = (c / b) * np.arctan(b / c)

    bd2 = (b / d) ** 2
    cd2 = (c / d) ** 2
    da2 = (d / a) ** 2
    ml = np.log(8 * a / d)

    f = (
        ml
        + (1 + u + v - 8 * (w + ww)) / 12.0  # 0th order in d/a
        + (da2 * (cd2 * (221 + 60 * ml - 6 * v) + 3 * bd2 * (69 + 60 * ml + 10 * u - 64 * w)))
        / 5760.0  # 2nd order
        + (
            da2**2
            * (
                2 * cd2**2 * (5721 + 3080 * ml - 345 * v)
                + 5 * bd2 * cd2 * (407 + 5880 * ml + 6720 * u - 14336 * w)
                - 10 * bd2**2 * (3659 + 2520 * ml + 805 * u - 6144 * w)
            )
        )
        / 2.58048e7  # 4th order
        + (
            da2**3
            * (
                3 * cd2**3 * (4308631 + 86520 * ml - 10052 * v)
                - 14 * bd2**2 * cd2 * (617423 + 289800 * ml + 579600 * u - 1474560 * w)
                + 21 * bd2**3 * (308779 + 63000 * ml + 43596 * u - 409600 * w)
                + 42 * bd2 * cd2**2 * (-8329 + 46200 * ml + 134400 * u - 172032 * w)
            )
        )
        / 1.73408256e10  # 6th order
    )  # [nondim] shape parameter

    self_inductance = MU_0 * (n**2) * a * f

    assert self_inductance >= 0.0, "Coil geometry outside domain of validity of Lyle's formula"

    return self_inductance  # [H]


def mutual_inductance_of_circular_filaments(rzn1: NDArray, rzn2: NDArray, par: bool = True) -> float:
    """
    Analytic mutual inductance between a pair of ideal cylindrically-symmetric coaxial filaments.

    This is equivalent to taking the flux produced by each circular filament from
    either collection of filaments to the other. Mutual inductance is reflexive,
    so the order of the inputs is not important.

    Args:
        rzn1 (array): 3x1 array (r [m], z [m], n []) coordinates and number of turns
        rzn2 (array): 3x1 array (r [m], z [m], n []) coordinates and number of turns
        par: Whether to use CPU parallelism

    Returns:
        float: [H] mutual inductance
    """

    m = mutual_inductance_of_cylindrical_coils(rzn1.reshape((3, 1)), rzn2.reshape((3, 1)), par)

    return m  # [H]


def mutual_inductance_of_cylindrical_coils(f1: NDArray, f2: NDArray, par: bool = True) -> float:
    """
    Analytical mutual inductance between two coaxial collections of ideal filaments.

    Each collection typically represents a discretized "real" cylindrically-symmetric
    coil of rectangular cross-section, but could have any cross-section as long as it
    maintains cylindrical symmetry.

    Args:
        f1: 3 x N array of filament definitions like (r [m], z [m], n [])
        f2: 3 x N array of filament definitions like (r [m], z [m], n [])
        par: Whether to use CPU parallelism

    Returns:
        [H] mutual inductance of the two discretized coils
    """
    r1, z1, n1 = f1
    r2, z2, n2 = f2

    # Using n2 as the current per filament is equivalent to examining a 1A reference current,
    # which gives us the flux per amp (inductance)
    m = np.sum(n1 * flux_circular_filament(n2, r2, z2, r1, z1, par))  # [H] total mutual inductance

    return m  # [H]


def filament_coil(r: float, z: float, w: float, h: float, nt: float, nr: int, nz: int) -> NDArray:
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

    # Build a 2D mesh of points evenly spaced in the interior of the bounding rectangle
    rs = np.linspace(r - w * (nr - 1) / nr / 2, r + w * (nr - 1) / nr / 2, nr)  # [m]
    zs = np.linspace(z - h * (nz - 1) / nz / 2, z + h * (nz - 1) / nz / 2, nz)  # [m]
    rmesh, zmesh = np.meshgrid(rs, zs, indexing="ij")  # [m]

    # Number of turns attributed to each point is not necessarily an integer
    n = np.full_like(rmesh.flatten(), float(nt) / (nr * nz))

    # Pack filament locations and number of turns into an array together
    filaments = np.dstack([rmesh.flatten(), zmesh.flatten(), n]).reshape(nr * nz, 3)

    return filaments  # [m], [m], [dimensionless]


def self_inductance_circular_ring_wien(major_radius: NDArray, minor_radius: NDArray) -> NDArray:
    """
    Wien's formula for the self-inductance of a circular ring
    with thin circular cross section.

    Uses equation 7 from reference [1].

    References:
        [1] E Rosa and L Cohen, "On the Self-Inductance of Circles,"
        Bulletin of the Bureau of Standards, 1908.
        [Online]. Available:
        <https://nvlpubs.nist.gov/nistpubs/bulletin/04/nbsbulletinv4n1p149_A2b.pdf>

    Args:
        major_radius: [m] Major radius of the ring.
        minor_radius: [m] Radius of the ring's cross section.

    Returns:
        [H] self-inductance
    """
    ar = major_radius / minor_radius  # [], a / rho, dimensionless
    ra2 = (minor_radius / major_radius) ** 2  # [], (rho / a)^2, dimensionless
    # Equation 7 in Rosa & Cohen lacks the factor of 1e-7
    # because it is not in SI base units.
    self_inductance = MU_0 * major_radius * ((1 + 0.125 * ra2) * np.log(8 * ar) - 0.0083 * ra2 - 1.75)  # [H]
    return self_inductance  # [H]


def self_inductance_distributed_axisymmetric_conductor(
    current: float,
    grid: tuple[NDArray, NDArray],
    mesh: tuple[NDArray, NDArray],
    b_part: tuple[NDArray, NDArray],
    psi_part: NDArray,
    mask: NDArray,
    edge_path: tuple[NDArray, NDArray],
) -> tuple[float, float, float]:
    """
    Calculation of a distributed conductor's self-inductance from two components:

    * External inductance: the portion related to the poloidal flux exactly at the conductor edge
    * Internal inductance: the portion related to the poloidal magnetic field inside the conductor,
      where the filamentized method used for coils does not apply due to the parallel arrangement
      and variable current density.

    Note: the B-field and flux inputs are _not_ the total from all sources - they are only the
    contribution from the distributed conductor under examination.

    This calculation was developed for use with tokamak plasmas, but applies similarly to
    other kinds of distributed axisymmetric conductor.

    Assumptions:

    * Cylindrically-symmetric, distributed, single-winding, contiguous conductor
    * No high-magnetic-permeability materials in the vicinity
    * Isopotential on the edge contour
        * This is slightly less restrictive than isopotential on the section,
          but notably does _not_ allow the calc to be used with, for example,
          large, shell conductors where different regions are meaningfully
          independent of each other.
    * Conductor interior does not touch the edge of the computational domain
        * At least one grid cell of padding is needed to support finite differences

    References:
        [1] S. Ejima, R. W. Callis, J. L. Luxon, R. D. Stambaugh, T. S. Taylor, and J. C. Wesley,
            “Volt-second analysis and consumption in Doublet III plasmas,”
            Nucl. Fusion, vol. 22, no. 10, pp. 1313-1319, Oct. 1982,
            doi: [10.1088/0029-5515/22/10/006](https://doi.org/10.1088/0029-5515/22/10/006)

        [2] J. A. Romero and J.-E. Contributors, “Plasma internal inductance dynamics in a tokamak,”
            arXiv.org. Accessed: Dec. 21, 2023. [Online]. Available: https://arxiv.org/abs/1009.1984v1
            doi: [10.1088/0029-5515/50/11/115002](https://doi.org/10.1088/0029-5515/50/11/115002)

        [3] J. T. Wai and E. Kolemen, “GSPD: An algorithm for time-dependent tokamak equilibria design.”
            arXiv, Jun. 22, 2023. Accessed: Sep. 15, 2023. [Online]. Available: https://arxiv.org/abs/2306.13163
            doi: [10.48550/arXiv.2306.13163](https://doi.org/10.48550/arXiv.2306.13163)

    Args:
        current: [A] total toroidal current in this conductor
        grid: [m] (1 X nr) grids of (R coords, Z coords)
        mesh: [m] (nr X nz) meshgrids of (R coords, Z coords)
        b_part: [T] (nr X nz) Flux density (R-component, Z-component) due to this conductor
        psi_part: [V-s] or [T-m^2] (nr X nz) this conductor's poloidal flux field
        mask: (nr X nz) positive mask of the conductor's interior region
        edge_path: [m] (2 x N) closed (r, z) path along conductor edge

    Returns:
        (Lt, Li, Le) Total, internal, and external self-inductance components
    """
    # Unpack
    rgrid, zgrid = grid  # [m]
    rmesh, zmesh = mesh  # [m]
    br, bz = b_part  # [T]

    # Set up
    nr = rgrid.size
    nz = zgrid.size
    psi_interpolator = MulticubicRectilinear.new([rgrid, zgrid], psi_part.flatten())

    # Same-length diffs assuming last grid cell is the same size
    # as the previous one. In general the conductor can't touch the
    # edge of the computational domain without breaking the solver,
    # so it's ok to allow some potential error at the last index.
    drmesh = np.zeros_like(rmesh)  # [m]
    drmesh[: nr - 1, :] = np.diff(rmesh, axis=0)
    drmesh[-1, :] = drmesh[-2, :]

    dzmesh = np.zeros_like(zmesh)  # [m]
    dzmesh[:, : nz - 1] = np.diff(zmesh, axis=1)
    dzmesh[:, -1] = dzmesh[:, -2]

    # Toroidal volume of each cell in the mesh.
    #
    # Ideally we'd use a two-sided difference for
    # calculating dr and dz in order to properly handle cell volume
    # for nonuniform grids, but (1) that's a lot of extra array handling
    # for minimal real benefit and (2) the grid is almost always uniform
    volmesh = 2.0 * np.pi * rmesh * drmesh * dzmesh  # [m^3]

    # Stored poloidal magnetic energy inside the conductor volume.
    #
    # Because E ~ B^2 and B = sqrt(Br^2 + Bz^2 + Btor^2) -> B^2 = Br^2 + Bz^2 + Btor^2,
    # the components of magnetic energy (and induction) due to a current on different axes
    # are separable, and we can ignore the toroidal field entirely when treating the
    # poloidal inductance.
    #
    # That doesn't mean there isn't store energy related to the conductor's toroidal field,
    # only that it can be separated from the poloidal inductance.
    wmag_pol = (1.0 / (2.0 * MU_0)) * np.sum((br**2 + bz**2) * volmesh * mask)  # [J]

    # Internal inductance.
    #
    # Because E = (1/2) * L * I^2, we can superpose multiple inductance terms
    # for a given current-carrying element, and say E = (1/2) * (Li + Le) * I^2 .
    #
    # Now that we have the internal stored magnetic energy, we can calculate
    # the part of the self-inductance related to that energy directly.
    internal_inductance = 2.0 * wmag_pol / current**2  # [H]

    # External inductance.
    #
    # This is a hack based on Poynting's Theorem relating surface conditions to
    # magnetic energy.
    #
    #   Do weighted average of flux along the edge contour
    #   to adjust for the length of individual segments
    edge_dr = np.diff(edge_path[0])  # [m]
    edge_dz = np.diff(edge_path[1])  # [m]
    edge_dl = (edge_dr**2 + edge_dz**2) ** 0.5  # [m] length of each segment
    edge_length = np.sum(edge_dl)  # [m] total length of conductor limit contour
    edge_psi = psi_interpolator.eval([x[:-1] for x in edge_path])  # [V-s]
    edge_psi_mean = np.sum(edge_psi * edge_dl) / edge_length  # [V-s]
    #   Take the inductance
    external_inductance = edge_psi_mean / current  # [H]

    # Total inductance.
    total_inductance = internal_inductance + external_inductance  # [H]

    return (
        float(total_inductance),
        float(internal_inductance),
        float(external_inductance),
    )


def self_inductance_annular_ring(r: float, a: float, b: float) -> float:
    """
    Low-frequency self-inductance of a thick-walled tube bent in a circle.

    Uses Wien's method, per the 1912 NIST handbook [1], Eqn. 64 on pg. 112,
    with a correction to a misprint in term 5 and a unit conversion factor
    in the final expression.

    This is an approximation that drops terms of order higher than `(a/r)^2`
    and `(b/r)^2`.

    References:
        [1] E. B. Rosa and F. W. Grover,
            “Formulas and tables for the calculation of mutual and self-inductance (Revised),”
            BULL. NATL. BUR. STAND., vol. 8, no. 1, p. 1, Jan. 1912,
            doi: [10.6028/bulletin.185](https://doi.org/10.6028/bulletin.185)

    Args:
        r: [m] major radius of loop
        a: [m] inner minor radius (tube inside radius)
        b: [m] outer minor radius (tube outside radius)

    Returns:
        [H] self-inductance [H]
    """

    if not ((r > 0.0) and (a > 0.0) and (b > 0.0)):
        raise ValueError("All radii must be positive and nonzero.")
    if not (a < b):
        raise ValueError("Inner minor radius must be less than outer minor radius.")
    if not (b < r):
        raise ValueError("Outer minor radius must be less than major radius.")

    term1 = (1.0 + (a**2 + b**2) / (8.0 * r**2)) * np.log(8.0 * r / b)
    term2 = -1.75 + (2.0 * b**2 + a**2) / (32.0 * r**2)
    term3 = -0.5 * a**2 / (b**2 - a**2)
    term4 = (a**4 / ((b**2 - a**2) ** 2)) * (1.0 + a**2 / (8.0 * r**2)) * np.log(b / a)
    term5 = -(a**4 + a**2 * b**2 + b**4) / (48.0 * r**2 * (b**2 - a**2))

    self_inductance = MU_0 * r * (term1 + term2 + term3 + term4 + term5)  # [H]

    return self_inductance  # [H]
