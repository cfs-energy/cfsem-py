"""
Symmetric bindings for backend calcs.

This fulfills the function of typing stubs, while also guaranteeing arrays are
passed as contiguous and reallocating into contiguous inputs if necessary.
"""

from cfsem.types import Array3xN

from numpy import ascontiguousarray, float64, zeros_like
from numpy.typing import NDArray

from ._cfsem import flux_circular_filament as em_flux_circular_filament
from ._cfsem import flux_density_biot_savart as em_flux_density_biot_savart
from ._cfsem import flux_density_circular_filament as em_flux_density_circular_filament
from ._cfsem import gs_operator_order2 as em_gs_operator_order2
from ._cfsem import gs_operator_order4 as em_gs_operator_order4
from ._cfsem import (
    inductance_piecewise_linear_filaments as em_inductance_piecewise_linear_filaments,
)
from ._cfsem import filament_helix_path as em_filament_helix_path
from ._cfsem import rotate_filaments_about_path as em_rotate_filaments_about_path


def flux_circular_filament(
    ifil: NDArray[float64],
    rfil: NDArray[float64],
    zfil: NDArray[float64],
    rprime: NDArray[float64],
    zprime: NDArray[float64],
) -> NDArray[float64]:
    """
    Flux contributions from some circular filaments to some observation points,
    which happens to be the Green's function for the Grad-Shafranov solve.

    This represents the integral of $\\vec{B} \\cdot \\hat{n} \\, dA$ from the z-axis to each
    (`rprime`, `zprime`) observation location with $\\hat{n}$ oriented parallel to the z-axis.

    A convenient interpretation of the flux is as the mutual inductance
    per secondary coil current between a filament at (`rfil`, `zfil`) and a secondary
    filament at (`rprime`, `zprime`); this can be used to get the mutual inductance
    between two filamentized coils as the sum of I1 * I2 * flux_from_coil_1_to_coil_2.
    Because mutual inductance is reflexive, the order of the coils can be reversed and
    the same result is obtained.

    This function is fairly well-optimized and runs about >10x faster than an
    equivalent implementation in numpy+scipy, with lower RAM usage
    (O(n) in number of observation locations) than what would be required
    for a tiled implementation. Numba does not have an implementation of
    elliptic integrals and will not accept scipy's.

    Args:
        ifil: [A] filament current
        rfil: [m] filament R-coord
        zfil: [m] filament Z-coord
        rprime: [m] Observation point R-coord
        zprime: [m] Observation point Z-coord

    Returns:
        [T-m^2] or [V-s] psi, poloidal flux at each observation point
    """
    ifil = ascontiguousarray(ifil)
    rfil = ascontiguousarray(rfil)
    zfil = ascontiguousarray(zfil)
    rprime = ascontiguousarray(rprime)
    zprime = ascontiguousarray(zprime)
    psi = em_flux_circular_filament(ifil, rfil, zfil, rprime, zprime)
    return psi  # [T-m^2] or [V-s]


def flux_density_circular_filament(
    ifil: NDArray[float64],
    rfil: NDArray[float64],
    zfil: NDArray[float64],
    rprime: NDArray[float64],
    zprime: NDArray[float64],
) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Off-axis Br,Bz components for a circular current filament in vacuum.

    Near-exact formula (except numerically-evaluated elliptic integrals)
    See eqns. 12, 13 pg. 34 in [1], eqn 9.8.7 in [2], and all of [3].

    Note the formula for Br as given by [1] is incorrect and does not satisfy the
    constraints of the calculation without correcting by a factor of ($z / r$).

    References:
        [1] D. B. Montgomery and J. Terrell,
            “Some Useful Information For The Design Of Aircore Solenoids,
            Part I. Relationships Between Magnetic Field, Power, Ampere-Turns
            And Current Density. Part II. Homogeneous Magnetic Fields,”
            Massachusetts Inst. Of Tech. Francis Bitter National Magnet Lab, Cambridge, MA,
            Nov. 1961. Accessed: May 18, 2021. [Online].
            Available: <https://apps.dtic.mil/sti/citations/tr/AD0269073>

        [2] 8.02 Course Notes. Available:
        <https://web.mit.edu/8.02t/www/802TEAL3D/visualizations/coursenotes/modules/guide09.pdf>

        [3] Eric Dennyson, "Magnet Formulas". Available:
        <https://tiggerntatie.github.io/emagnet-py/offaxis/off_axis_loop.html>

    Args:
        ifil: [A] filament current
        rfil: [m] filament R-coord
        zfil: [m] filament Z-coord
        rprime: [m] Observation point R-coord
        zprime: [m] Observation point Z-coord

    Returns:
        [T] (Br, Bz) flux density components
    """
    ifil = ascontiguousarray(ifil)
    rfil = ascontiguousarray(rfil)
    zfil = ascontiguousarray(zfil)
    rprime = ascontiguousarray(rprime)
    zprime = ascontiguousarray(zprime)
    Br, Bz = em_flux_density_circular_filament(ifil, rfil, zfil, rprime, zprime)
    return Br, Bz  # [T]


def flux_density_biot_savart(
    xyzp: Array3xN,
    xyzfil: Array3xN,
    dlxyzfil: Array3xN,
    ifil: NDArray[float64],
    par: bool = True,
) -> Array3xN:
    """
    Biot-Savart law calculation for B-field contributions from many filament segments
    to many observation points.

    This calc is fairly well-optimized under the constraint to keep peak RAM usage to
    O(n) in n observation points, which precludes tiling. It comes out about 20% faster
    than an equivalent implementation in numba, which is about 4x faster than an
    equivalent implementation in numpy.

    Args:
        xyzp: [m] x,y,z coords of observation points
        xyzfil: [m] x,y,z coords of current filament origins (start of segment)
        dlxyzfil: [m] x,y,z length delta of current filaments
        ifil: [A] current in each filament segment
        par: Whether to use CPU parallelism

    Returns:
        [T] (Bx, By, Bz) magnetic flux density at observation points
    """
    xyzp = (
        ascontiguousarray(xyzp[0]),
        ascontiguousarray(xyzp[1]),
        ascontiguousarray(xyzp[2]),
    )
    xyzfil = (
        ascontiguousarray(xyzfil[0]),
        ascontiguousarray(xyzfil[1]),
        ascontiguousarray(xyzfil[2]),
    )
    dlxyzfil = (
        ascontiguousarray(dlxyzfil[0]),
        ascontiguousarray(dlxyzfil[1]),
        ascontiguousarray(dlxyzfil[2]),
    )
    ifil = ascontiguousarray(ifil)
    return em_flux_density_biot_savart(xyzp, xyzfil, dlxyzfil, ifil, par)


def inductance_piecewise_linear_filaments(
    xyzfil0: Array3xN,
    dlxyzfil0: Array3xN,
    xyzfil1: Array3xN,
    dlxyzfil1: Array3xN,
    self_inductance: bool = False,
) -> float:
    """
    Estimate the mutual inductance between two piecewise-linear current filaments,
    or estimate self-inductance by passing the same filaments twice and setting
    `self_inductance = True`.

    It may be easier to use wrappers of this function that are specialized for self- and mutual-inductance
    calculations:
    [`self_inductance_piecewise_linear_filaments`][cfsem.self_inductance_piecewise_linear_filaments]
    and [`mutual_inductance_piecewise_linear_filaments`][cfsem.mutual_inductance_piecewise_linear_filaments].

    Uses Neumann's Formula for the mutual inductance of arbitrary loops, which is
    originally from [2] and can be found in a more friendly format on wikipedia.

    When self_inductance flag is set, zeroes-out the contributions from self-pairings
    to resolve the thin-filament self-inductance singularity and replaces the
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
        xyzfil0: [m] Nx3 point series describing the filament origins
        dlxyzfil0: [m] Nx3 length vector of each filament
        xyzfil1: [m] Nx3 point series describing the filament origins
        dlxyzfil1: [m] Nx3 length vector of each filament
        self_inductance: Whether this is being used as a self-inductance calc

    Returns:
        [H] Scalar inductance
    """
    xyzfil0 = (
        ascontiguousarray(xyzfil0[0]),
        ascontiguousarray(xyzfil0[1]),
        ascontiguousarray(xyzfil0[2]),
    )
    dlxyzfil0 = (
        ascontiguousarray(dlxyzfil0[0]),
        ascontiguousarray(dlxyzfil0[1]),
        ascontiguousarray(dlxyzfil0[2]),
    )
    xyzfil1 = (
        ascontiguousarray(xyzfil1[0]),
        ascontiguousarray(xyzfil1[1]),
        ascontiguousarray(xyzfil1[2]),
    )
    dlxyzfil1 = (
        ascontiguousarray(dlxyzfil1[0]),
        ascontiguousarray(dlxyzfil1[1]),
        ascontiguousarray(dlxyzfil1[2]),
    )
    return em_inductance_piecewise_linear_filaments(
        xyzfil0, dlxyzfil0, xyzfil1, dlxyzfil1, self_inductance
    )


def gs_operator_order2(rs: NDArray[float64], zs: NDArray[float64]) -> Array3xN:
    """Build second-order Grad-Shafranov operator in triplet format.
    Assumes regular grid spacing.

    Args:
        rs: [m] r-coordinates of finite difference grid
        zs: [m] z-coordinates of finite difference grid

    Returns:
        Differential operator as triplet format sparse matrix
    """
    rs = ascontiguousarray(rs)
    zs = ascontiguousarray(zs)
    return em_gs_operator_order2(rs, zs)


def gs_operator_order4(rs: NDArray[float64], zs: NDArray[float64]) -> Array3xN:
    """
    Build fourth-order Grad-Shafranov operator in triplet format.
    Assumes regular grid spacing.

    Args:
        rs: [m] r-coordinates of finite difference grid
        zs: [m] z-coordinates of finite difference grid

    Returns:
        Differential operator as triplet format sparse matrix
    """
    rs = ascontiguousarray(rs)
    zs = ascontiguousarray(zs)
    return em_gs_operator_order4(rs, zs)


def filament_helix_path(
    path: Array3xN,
    helix_start_offset: tuple[float, float, float],
    twist_pitch: float,
    angle_offset: float,
) -> NDArray[float64]:
    """
    Filamentize a helix about an arbitrary piecewise-linear path.

    Assumes angle between sequential path segments is small and will fail
    if that angle approaches or exceeds 90 degrees.

    The helix initial position vector, helix_start_offset, must be in a plane normal to
    the first path segment in order to produce good results. If it is not in-plane,
    it will be projected on to that plane and then scaled to the magnitude of its
    original length s.t. the distance from the helix to the path center is preserved
    but its orientation is not.

    Description of the method:

    1. Translate [filament segment n-1] to the base of [path segment n]
        and call it [filament segment n]
    2. Take cross product of [path segment n] with [path segment n-1]
    3. Rotate [filament segment n] segment about the axis of that cross product
        to bring it into the plane defined by [path segment n] as a normal vector
    4. Rotate [filament seg. n] about [path seg. n] to continue the helix orbit

    Args:
        path: [m] 3xN Centerline points
        helix_start_offset: [m] (3x1) Initial position of helix rel. to centerline path
        twist_pitch: [m] (scalar) Centerline length per helix orbit
        angle_offset: [rad] (scalar) Initial rotation offset about centerline

    Returns:
        [m] 3xN array of points on the helix that twists around the path
    """

    # Make sure input is contiguous, reallocating only if necessary
    path = ascontiguousarray(path)

    # Allocate output
    helix = zeros_like(path)  # [m]

    # Calculate, mutating output
    em_filament_helix_path(
        (*path,),
        helix_start_offset,
        twist_pitch,
        angle_offset,
        (*helix,),
    )

    return helix  # [m]


def rotate_filaments_about_path(
    path: Array3xN, angle_offset: float, fils: Array3xN
) -> NDArray[float64]:
    """
    Rotate a path of point about another path.

    Intended for rotating a helix generated by [`filament_helix_path`][cfsem.filament_helix_path]
    about the centerline that was used to generate it.

    Args:
        path: [m] x,y,z Centerline points
        angle_offset: [rad] (scalar) Initial rotation offset about centerline
        fils: [m] x,y,z Filaments to rotate around centerline

    Returns:
        [m] 3xN array of points on the helix that twists around the path
    """

    # Make sure input is contiguous, reallocating only if necessary
    path = ascontiguousarray(path)

    new_fils = ascontiguousarray(fils).copy()

    em_rotate_filaments_about_path(
        (*path,),
        angle_offset,
        (*new_fils,),
    )

    return new_fils  # [m]
