"""
Symmetric bindings for backend calcs.

This fulfills the function of typing stubs, while also guaranteeing arrays are
passed as contiguous and reallocating into contiguous inputs if necessary.
"""

from typing import Literal

from numpy import asarray, ascontiguousarray, column_stack, float64, full, int64, zeros_like
from numpy.typing import NDArray

from cfsem.types import Array3xN

from .cfsem import (
    body_force_density_circular_filament_cartesian as em_body_force_density_circular_filament_cartesian,
)
from .cfsem import (
    body_force_density_linear_filament as em_body_force_density_linear_filament,
)
from .cfsem import filament_helix_path as em_filament_helix_path
from .cfsem import flux_circular_filament as em_flux_circular_filament
from .cfsem import flux_density_circular_filament as em_flux_density_circular_filament
from .cfsem import (
    flux_density_circular_filament_cartesian as em_flux_density_circular_filament_cartesian,
)
from .cfsem import flux_density_dipole as em_flux_density_dipole
from .cfsem import vector_potential_dipole as em_vector_potential_dipole
from .cfsem import flux_density_linear_filament as em_flux_density_linear_filament
from .cfsem import flux_density_triangle_mesh_mapping as em_flux_density_triangle_mesh_mapping
from .cfsem import flux_density_triangle_mesh as em_flux_density_triangle_mesh
from .cfsem import (
    flux_density_linear_filament_matrix as em_flux_density_linear_filament_matrix,
)
from .cfsem import flux_density_point_segment as em_flux_density_point_segment
from .cfsem import gs_operator_order2 as em_gs_operator_order2
from .cfsem import gs_operator_order4 as em_gs_operator_order4
from .cfsem import (
    inductance_linear_filaments as em_inductance_linear_filaments,
)
from .cfsem import (
    inductance_linear_filaments_matrix as em_inductance_linear_filaments_matrix,
)
from .cfsem import (
    inductance_piecewise_linear_filaments as em_inductance_piecewise_linear_filaments,
)
from .cfsem import (
    mutual_inductance_circular_to_linear as em_mutual_inductance_circular_to_linear,
)
from .cfsem import rotate_filaments_about_path as em_rotate_filaments_about_path
from .cfsem import triangle_mesh_current_density as em_triangle_mesh_current_density
from .cfsem import triangle_mesh_force_mapping as em_triangle_mesh_force_mapping
from .cfsem import (
    triangle_mesh_force_mapping_from_circular_filaments as em_triangle_mesh_force_mapping_from_circular_filaments,  # noqa: E501
)
from .cfsem import (
    triangle_mesh_force_mapping_from_dipoles as em_triangle_mesh_force_mapping_from_dipoles,
)
from .cfsem import (
    triangle_mesh_force_mapping_from_linear_filaments as em_triangle_mesh_force_mapping_from_linear_filaments,
)
from .cfsem import (
    triangle_mesh_flux_linkage_mapping_from_dipoles as em_triangle_mesh_flux_linkage_mapping_from_dipoles,
)
from .cfsem import (
    triangle_mesh_inductance_mapping_from_circular_filaments as em_triangle_mesh_inductance_mapping_from_circular_filaments,  # noqa: E501
)
from .cfsem import (
    triangle_mesh_inductance_mapping_from_linear_filaments as em_triangle_mesh_inductance_mapping_from_linear_filaments,  # noqa: E501
)
from .cfsem import triangle_mesh_inductance_matrix as em_triangle_mesh_inductance_matrix
from .cfsem import triangle_mesh_quadrature_points as em_triangle_mesh_quadrature_points
from .cfsem import triangle_mesh_self_force_mapping as em_triangle_mesh_self_force_mapping
from .cfsem import (
    vector_potential_circular_filament as em_vector_potential_circular_filament,
)
from .cfsem import (
    vector_potential_linear_filament as em_vector_potential_linear_filament,
)
from .cfsem import (
    vector_potential_triangle_mesh_mapping as em_vector_potential_triangle_mesh_mapping,
)
from .cfsem import (
    vector_potential_triangle_mesh as em_vector_potential_triangle_mesh,
    vector_potential_linear_filament_matrix as em_vector_potential_linear_filament_matrix,
)
from .cfsem import (
    vector_potential_point_segment as em_vector_potential_point_segment,
)


def flux_circular_filament(
    ifil: NDArray[float64],
    rfil: NDArray[float64],
    zfil: NDArray[float64],
    rprime: NDArray[float64],
    zprime: NDArray[float64],
    par: bool = True,
) -> NDArray[float64]:
    """
    Flux contributions from some circular filaments to some observation points,
    which happens to be the Green's function for the Grad-Shafranov solve.

    This represents the integral of $\\vec{B} \\cdot \\hat{n} \\, dA$ from the z-axis to each
    (`rprime`, `zprime`) observation location with $\\hat{n}$ oriented parallel to the z-axis.

    A convenient interpretation of the flux is as the mutual inductance
    between a circular filament at (`rfil`, `zfil`) and a second circular
    filament at (`rprime`, `zprime`); this can be used to get the mutual inductance
    between two filamentized coils as the sum of flux contributions between each coil's filaments.
    Because mutual inductance is reflexive, the order of the coils can be reversed and
    the same result is obtained.

    Args:
        ifil: [A] filament current
        rfil: [m] filament R-coord
        zfil: [m] filament Z-coord
        rprime: [m] Observation point R-coord
        zprime: [m] Observation point Z-coord
        par: Whether to use CPU parallelism

    Returns:
        [Wb] or [T-m^2] or [V-s] psi, poloidal flux at each observation point
    """
    ifil, rfil, zfil = _3tup_contig((ifil, rfil, zfil))
    rprime, zprime = _2tup_contig((rprime, zprime))
    psi = em_flux_circular_filament(ifil, rfil, zfil, rprime, zprime, par)
    return psi  # [Wb] or [T-m^2] or [V-s]


def vector_potential_circular_filament(
    ifil: NDArray[float64],
    rfil: NDArray[float64],
    zfil: NDArray[float64],
    rprime: NDArray[float64],
    zprime: NDArray[float64],
    par: bool = True,
) -> NDArray[float64]:
    """
    Vector potential contributions from some circular filaments to some observation points.
    Off-axis A_phi component for a circular current filament in vacuum.

    The vector potential of a loop has zero r- and z- components due to symmetry,
    and does not vary in the phi-direction.

    Note that to recover the B-field as the curl of A, the curl operator for cylindrical
    coordinates must be used with the output of this function incorporated into a full
    3D A-field like [A_r, A_phi, A_z].

    References:
        [1] J. C. Simpson, J. E. Lane, C. D. Immer, R. C. Youngquist, and T. Steinrock,
            “Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop,”
            Jan. 01, 2001. Accessed: Sep. 06, 2022. [Online]. Available: <https://ntrs.nasa.gov/citations/20010038494>

    Args:
        ifil: [A] filament current
        rfil: [m] filament R-coord
        zfil: [m] filament Z-coord
        rprime: [m] Observation point R-coord
        zprime: [m] Observation point Z-coord
        par: Whether to use CPU parallelism

    Returns:
        [Wb/m] or [V-s/m] a_phi, vector potential in the toroidal direction
    """
    ifil, rfil, zfil = _3tup_contig((ifil, rfil, zfil))
    rprime, zprime = _2tup_contig((rprime, zprime))
    a_phi = em_vector_potential_circular_filament(ifil, rfil, zfil, rprime, zprime, par)
    return a_phi  # [Wb/m] or [V-s/m]


def flux_density_circular_filament(
    ifil: NDArray[float64],
    rfil: NDArray[float64],
    zfil: NDArray[float64],
    rprime: NDArray[float64],
    zprime: NDArray[float64],
    par: bool = True,
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
        par: Whether to use CPU parallelism

    Returns:
        [T] (Br, Bz) flux density components
    """
    ifil, rfil, zfil = _3tup_contig((ifil, rfil, zfil))
    rprime, zprime = _2tup_contig((rprime, zprime))
    br, bz = em_flux_density_circular_filament(ifil, rfil, zfil, rprime, zprime, par)
    return br, bz  # [T]


def flux_density_linear_filament(
    xyzp: Array3xN,
    xyzfil: Array3xN,
    dlxyzfil: Array3xN,
    ifil: NDArray[float64],
    wire_radius: float | NDArray[float64] = 0.0,
    par: bool = True,
    output: Literal["vector", "matrix"] = "vector",
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """
    Biot-Savart law calculation for B-field contributions from many filament segments
    to many observation points.

    Args:
        xyzp: [m] x,y,z coords of observation points
        xyzfil: [m] x,y,z coords of filament segment start points
        dlxyzfil: [m] x,y,z deltas from segment start to segment end
        ifil: [A] current in each filament segment
        wire_radius: [m] filament radius, scalar or array of length `m`
        par: Whether to use CPU parallelism
        output: `"vector"` for contracted field values at each target point,
            or `"matrix"` for row-major `(nobs, nfil)` source-target interaction matrices

    Returns:
        [T] (Bx, By, Bz) magnetic flux density at observation points,
        or explicit `(nobs, nfil)` interaction matrices if `output="matrix"`
    """
    xyzp = _3tup_contig(xyzp)
    xyzfil = _3tup_contig(xyzfil)
    dlxyzfil = _3tup_contig(dlxyzfil)
    ifil = ascontiguousarray(ifil).ravel()
    if asarray(wire_radius).ndim == 0:
        wire_radius = full(ifil.size, float(wire_radius))
    wire_radius = ascontiguousarray(wire_radius).ravel()
    if output == "vector":
        return em_flux_density_linear_filament(xyzp, xyzfil, dlxyzfil, ifil, wire_radius, par)
    if output == "matrix":
        bx, by, bz = em_flux_density_linear_filament_matrix(xyzp, xyzfil, dlxyzfil, ifil, wire_radius, par)
        nobs = xyzp[0].size
        nfil = ifil.size
        return (
            bx.reshape((nobs, nfil)),
            by.reshape((nobs, nfil)),
            bz.reshape((nobs, nfil)),
        )
    raise ValueError("output must be 'vector' or 'matrix'")


def flux_density_point_segment(
    xyzp: Array3xN,
    xyzfil: Array3xN,
    dlxyzfil: Array3xN,
    ifil: NDArray[float64],
    par: bool = True,
) -> Array3xN:
    """
    Biot-Savart law calculation for B-field contributions from many filament segments
    to many observation points, treating each segment as a point source.

    Args:
        xyzp: [m] x,y,z coords of observation points
        xyzfil: [m] x,y,z coords of filament segment start points
        dlxyzfil: [m] x,y,z deltas from segment start to segment end
        ifil: [A] current in each filament segment
        par: Whether to use CPU parallelism

    Returns:
        [T] (Bx, By, Bz) magnetic flux density at observation points
    """
    xyzp = _3tup_contig(xyzp)
    xyzfil = _3tup_contig(xyzfil)
    dlxyzfil = _3tup_contig(dlxyzfil)
    ifil = ascontiguousarray(ifil).ravel()
    return em_flux_density_point_segment(xyzp, xyzfil, dlxyzfil, ifil, par)


def vector_potential_linear_filament(
    xyzp: Array3xN,
    xyzfil: Array3xN,
    dlxyzfil: Array3xN,
    ifil: NDArray[float64],
    wire_radius: float | NDArray[float64] = 0.0,
    par: bool = True,
    output: Literal["vector", "matrix"] = "vector",
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """
    Vector potential calculation for A-field contribution from many current filament
    segments to many observation points.

    Args:
        xyzp: [m] x,y,z coords of observation points
        xyzfil: [m] x,y,z coords of filament segment start points
        dlxyzfil: [m] x,y,z deltas from segment start to segment end
        ifil: [A] current in each filament segment
        wire_radius: [m] filament radius, scalar or array of length `m`
        par: Whether to use CPU parallelism
        output: `"vector"` for contracted field values at each target point,
            or `"matrix"` for row-major `(nobs, nfil)` source-target interaction matrices

    Returns:
        [Wb/m] or [V-s/m] (Ax, Ay, Az) magnetic vector potential at observation points,
        or explicit `(nobs, nfil)` interaction matrices if `output="matrix"`
    """
    xyzp = _3tup_contig(xyzp)
    xyzfil = _3tup_contig(xyzfil)
    dlxyzfil = _3tup_contig(dlxyzfil)
    ifil = ascontiguousarray(ifil).ravel()
    if asarray(wire_radius).ndim == 0:
        wire_radius = full(ifil.size, float(wire_radius))
    wire_radius = ascontiguousarray(wire_radius).ravel()
    if output == "vector":
        return em_vector_potential_linear_filament(xyzp, xyzfil, dlxyzfil, ifil, wire_radius, par)
    if output == "matrix":
        ax, ay, az = em_vector_potential_linear_filament_matrix(
            xyzp, xyzfil, dlxyzfil, ifil, wire_radius, par
        )
        nobs = xyzp[0].size
        nfil = ifil.size
        return (
            ax.reshape((nobs, nfil)),
            ay.reshape((nobs, nfil)),
            az.reshape((nobs, nfil)),
        )
    raise ValueError("output must be 'vector' or 'matrix'")


def vector_potential_point_segment(
    xyzp: Array3xN,
    xyzfil: Array3xN,
    dlxyzfil: Array3xN,
    ifil: NDArray[float64],
    par: bool = True,
) -> Array3xN:
    """
    Vector potential calculation for A-field contribution from many filament
    segments to many observation points, treating each segment as a point source.

    Args:
        xyzp: [m] x,y,z coords of observation points
        xyzfil: [m] x,y,z coords of filament segment start points
        dlxyzfil: [m] x,y,z deltas from segment start to segment end
        ifil: [A] current in each filament segment
        par: Whether to use CPU parallelism

    Returns:
        [Wb/m] or [V-s/m] (Ax, Ay, Az) magnetic vector potential at observation points
    """
    xyzp = _3tup_contig(xyzp)
    xyzfil = _3tup_contig(xyzfil)
    dlxyzfil = _3tup_contig(dlxyzfil)
    ifil = ascontiguousarray(ifil).ravel()
    return em_vector_potential_point_segment(xyzp, xyzfil, dlxyzfil, ifil, par)


def flux_density_triangle_mesh(
    obs: NDArray[float64],
    nodes: NDArray[float64],
    triangles: NDArray[int64],
    s: NDArray[float64],
    par: bool = True,
    quad: str = "gl3",
) -> Array3xN:
    """
    Biot-Savart law calculation for B-field contribution from a triangle mesh
    with one stream-function value per node.

    Args:
        obs: [m] observation points with shape `(nobs, 3)`
        nodes: [m] mesh node coordinates with shape `(nnode, 3)`
        triangles: node indices with shape `(ntri, 3)`
        s: nodal stream-function values with shape `(nnode,)`
        par: Whether to use CPU parallelism
        quad: Triangle quadrature rule, one of `"gl2"`, `"gl3"`, or `"dunavant5"`

    Returns:
        [T] (Bx, By, Bz) magnetic flux density at observation points
    """
    obs = ascontiguousarray(obs, dtype=float64)
    nodes = ascontiguousarray(nodes, dtype=float64)
    triangles = ascontiguousarray(triangles, dtype=int64)
    s = ascontiguousarray(s, dtype=float64).ravel()
    return em_flux_density_triangle_mesh(obs, nodes, triangles, s, par, quad)


def vector_potential_triangle_mesh(
    obs: NDArray[float64],
    nodes: NDArray[float64],
    triangles: NDArray[int64],
    s: NDArray[float64],
    par: bool = True,
    quad: str = "gl3",
) -> Array3xN:
    """
    Vector potential calculation for A-field contribution from a triangle mesh
    with one stream-function value per node.

    Args:
        obs: [m] observation points with shape `(nobs, 3)`
        nodes: [m] mesh node coordinates with shape `(nnode, 3)`
        triangles: node indices with shape `(ntri, 3)`
        s: nodal stream-function values with shape `(nnode,)`
        par: Whether to use CPU parallelism
        quad: Triangle quadrature rule, one of `"gl2"`, `"gl3"`, or `"dunavant5"`

    Returns:
        [Wb/m] or [V-s/m] (Ax, Ay, Az) magnetic vector potential at observation points
    """
    obs = ascontiguousarray(obs, dtype=float64)
    nodes = ascontiguousarray(nodes, dtype=float64)
    triangles = ascontiguousarray(triangles, dtype=int64)
    s = ascontiguousarray(s, dtype=float64).ravel()
    return em_vector_potential_triangle_mesh(obs, nodes, triangles, s, par, quad)


def flux_density_triangle_mesh_mapping(
    obs: NDArray[float64],
    nodes: NDArray[float64],
    triangles: NDArray[int64],
    par: bool = True,
    quad: str = "gl3",
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """
    Assemble the dense source-node to target-point B-field mapping for a triangle mesh.

    Args:
        obs: [m] observation points with shape `(nobs, 3)`
        nodes: [m] mesh node coordinates with shape `(nnode, 3)`
        triangles: node indices with shape `(ntri, 3)`
        par: Whether to use CPU parallelism
        quad: Triangle quadrature rule, one of `"gl2"`, `"gl3"`, or `"dunavant5"`

    Returns:
        [T/A] `(bx_map, by_map, bz_map)` with shape `(nobs, nnode)`
    """
    obs = ascontiguousarray(obs, dtype=float64)
    nodes = ascontiguousarray(nodes, dtype=float64)
    triangles = ascontiguousarray(triangles, dtype=int64)
    bx, by, bz = em_flux_density_triangle_mesh_mapping(obs, nodes, triangles, par, quad)
    nobs = obs.shape[0]
    nnode = nodes.shape[0]
    return (
        ascontiguousarray(bx).reshape(nobs, nnode),
        ascontiguousarray(by).reshape(nobs, nnode),
        ascontiguousarray(bz).reshape(nobs, nnode),
    )


def vector_potential_triangle_mesh_mapping(
    obs: NDArray[float64],
    nodes: NDArray[float64],
    triangles: NDArray[int64],
    par: bool = True,
    quad: str = "gl3",
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """
    Assemble the dense source-node to target-point A-field mapping for a triangle mesh.

    Args:
        obs: [m] observation points with shape `(nobs, 3)`
        nodes: [m] mesh node coordinates with shape `(nnode, 3)`
        triangles: node indices with shape `(ntri, 3)`
        par: Whether to use CPU parallelism
        quad: Triangle quadrature rule, one of `"gl2"`, `"gl3"`, or `"dunavant5"`

    Returns:
        [V*s/(m*A)] `(ax_map, ay_map, az_map)` with shape `(nobs, nnode)`
    """
    obs = ascontiguousarray(obs, dtype=float64)
    nodes = ascontiguousarray(nodes, dtype=float64)
    triangles = ascontiguousarray(triangles, dtype=int64)
    ax, ay, az = em_vector_potential_triangle_mesh_mapping(obs, nodes, triangles, par, quad)
    nobs = obs.shape[0]
    nnode = nodes.shape[0]
    return (
        ascontiguousarray(ax).reshape(nobs, nnode),
        ascontiguousarray(ay).reshape(nobs, nnode),
        ascontiguousarray(az).reshape(nobs, nnode),
    )


def triangle_mesh_current_density(
    nodes: NDArray[float64],
    triangles: NDArray[int64],
    s: NDArray[float64],
) -> NDArray[float64]:
    """
    Extract the constant physical surface current density on each triangle of a mesh.

    Args:
        nodes: [m] mesh node coordinates with shape `(nnode, 3)`
        triangles: node indices with shape `(ntri, 3)`
        s: nodal stream-function values with shape `(nnode,)`

    Returns:
        [A/m] triangle-wise surface current density with shape `(ntri, 3)`
    """
    nodes = ascontiguousarray(nodes, dtype=float64)
    triangles = ascontiguousarray(triangles, dtype=int64)
    s = ascontiguousarray(s, dtype=float64).ravel()
    jx, jy, jz = em_triangle_mesh_current_density(nodes, triangles, s)
    return column_stack((jx, jy, jz))


def triangle_mesh_quadrature_points(
    nodes: NDArray[float64],
    triangles: NDArray[int64],
    quad: str = "gl3",
) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Extract physical quadrature-point coordinates and area weights for each triangle.

    Args:
        nodes: [m] mesh node coordinates with shape `(nnode, 3)`
        triangles: node indices with shape `(ntri, 3)`
        quad: Triangle quadrature rule, one of `"gl2"`, `"gl3"`, or `"dunavant5"`

    Returns:
        points: [m] quadrature-point coordinates with shape `(ntri, nqp, 3)`
        weights: [m^2] physical quadrature weights with shape `(ntri, nqp)`
    """
    nodes = ascontiguousarray(nodes, dtype=float64)
    triangles = ascontiguousarray(triangles, dtype=int64)
    xq, yq, zq, wq, nqp = em_triangle_mesh_quadrature_points(nodes, triangles, quad)
    ntri = triangles.shape[0]
    points = column_stack((xq, yq, zq)).reshape(ntri, nqp, 3)
    weights = ascontiguousarray(wq).reshape(ntri, nqp)
    return points, weights


def triangle_mesh_inductance_matrix(
    nodes: NDArray[float64],
    triangles: NDArray[int64],
    par: bool = True,
    quad: str = "gl3",
) -> NDArray[float64]:
    """
    Assemble the dense nodal inductance matrix for a triangle stream-function mesh.

    If `par=True` and the per-worker scratch matrices cannot be allocated, the
    implementation falls back to the serial path instead of failing outright.

    Args:
        nodes: [m] mesh node coordinates with shape `(nnode, 3)`
        triangles: node indices with shape `(ntri, 3)`
        par: Whether to use CPU parallelism
        quad: Triangle quadrature rule, one of `"gl2"`, `"gl3"`, or `"dunavant5"`

    Returns:
        [H] dense nodal inductance matrix with shape `(nnode, nnode)`
    """
    nodes = ascontiguousarray(nodes, dtype=float64)
    triangles = ascontiguousarray(triangles, dtype=int64)
    lmat = em_triangle_mesh_inductance_matrix(nodes, triangles, par, quad)
    nnode = nodes.shape[0]
    return ascontiguousarray(lmat).reshape(nnode, nnode)


def triangle_mesh_inductance_mapping_from_linear_filaments(
    xyzfil: Array3xN,
    dlxyzfil: Array3xN,
    nodes_tgt: NDArray[float64],
    triangles_tgt: NDArray[int64],
    wire_radius: float | NDArray[float64] = 0.0,
    par: bool = True,
    quad: str = "gl3",
) -> NDArray[float64]:
    """
    Assemble the source-current to target-node inductance mapping from linear filaments.

    Args:
        xyzfil: [m] x,y,z filament start coordinates
        dlxyzfil: [m] x,y,z filament segment deltas
        nodes_tgt: [m] target mesh node coordinates with shape `(nnode_tgt, 3)`
        triangles_tgt: target node indices with shape `(ntri_tgt, 3)`
        wire_radius: [m] filament radius, scalar or array of length `nfil`
        par: Whether to use CPU parallelism
        quad: Triangle quadrature rule, one of `"gl2"`, `"gl3"`, or `"dunavant5"`

    Returns:
        [H] mapping matrix with shape `(nnode_tgt, nfil)`
    """
    xyzfil = _3tup_contig(xyzfil)
    dlxyzfil = _3tup_contig(dlxyzfil)
    nodes_tgt = ascontiguousarray(nodes_tgt, dtype=float64)
    triangles_tgt = ascontiguousarray(triangles_tgt, dtype=int64)
    if asarray(wire_radius).ndim == 0:
        wire_radius = full(xyzfil[0].size, float(wire_radius))
    wire_radius = ascontiguousarray(wire_radius).ravel()
    out = em_triangle_mesh_inductance_mapping_from_linear_filaments(
        xyzfil, dlxyzfil, wire_radius, nodes_tgt, triangles_tgt, par, quad
    )
    return ascontiguousarray(out).reshape(nodes_tgt.shape[0], xyzfil[0].size)


def triangle_mesh_inductance_mapping_from_circular_filaments(
    rfil: NDArray[float64],
    zfil: NDArray[float64],
    nodes_tgt: NDArray[float64],
    triangles_tgt: NDArray[int64],
    par: bool = True,
    quad: str = "gl3",
) -> NDArray[float64]:
    """
    Assemble the source-current to target-node inductance mapping from circular filaments.

    Args:
        rfil: [m] circular filament radii
        zfil: [m] circular filament axial coordinates
        nodes_tgt: [m] target mesh node coordinates with shape `(nnode_tgt, 3)`
        triangles_tgt: target node indices with shape `(ntri_tgt, 3)`
        par: Whether to use CPU parallelism
        quad: Triangle quadrature rule, one of `"gl2"`, `"gl3"`, or `"dunavant5"`

    Returns:
        [H] mapping matrix with shape `(nnode_tgt, nfil)`
    """
    rfil = ascontiguousarray(rfil, dtype=float64).ravel()
    zfil = ascontiguousarray(zfil, dtype=float64).ravel()
    nodes_tgt = ascontiguousarray(nodes_tgt, dtype=float64)
    triangles_tgt = ascontiguousarray(triangles_tgt, dtype=int64)
    out = em_triangle_mesh_inductance_mapping_from_circular_filaments(
        rfil, zfil, nodes_tgt, triangles_tgt, par, quad
    )
    return ascontiguousarray(out).reshape(nodes_tgt.shape[0], rfil.size)


def triangle_mesh_flux_linkage_mapping_from_dipoles(
    loc: Array3xN,
    moment_dir: Array3xN,
    nodes_tgt: NDArray[float64],
    triangles_tgt: NDArray[int64],
    outer_radius: float | NDArray[float64] = 0.0,
    par: bool = True,
    quad: str = "gl3",
) -> NDArray[float64]:
    """
    Assemble the source-amplitude to target-node flux-linkage mapping from dipoles.

    Args:
        loc: [m] dipole locations
        moment_dir: dipole moment direction vectors
        nodes_tgt: [m] target mesh node coordinates with shape `(nnode_tgt, 3)`
        triangles_tgt: target node indices with shape `(ntri_tgt, 3)`
        outer_radius: [m] dipole finite-core radius, scalar or array of length `ndip`
        par: Whether to use CPU parallelism
        quad: Triangle quadrature rule, one of `"gl2"`, `"gl3"`, or `"dunavant5"`

    Returns:
        mapping matrix with shape `(nnode_tgt, ndip)`
    """
    loc = _3tup_contig(loc)
    moment_dir = _3tup_contig(moment_dir)
    nodes_tgt = ascontiguousarray(nodes_tgt, dtype=float64)
    triangles_tgt = ascontiguousarray(triangles_tgt, dtype=int64)
    if asarray(outer_radius).ndim == 0:
        outer_radius = full(loc[0].size, float(outer_radius))
    outer_radius = ascontiguousarray(outer_radius).ravel()
    out = em_triangle_mesh_flux_linkage_mapping_from_dipoles(
        loc, moment_dir, outer_radius, nodes_tgt, triangles_tgt, par, quad
    )
    return ascontiguousarray(out).reshape(nodes_tgt.shape[0], loc[0].size)


def triangle_mesh_force_mapping(
    nodes_src: NDArray[float64],
    triangles_src: NDArray[int64],
    nodes_tgt: NDArray[float64],
    triangles_tgt: NDArray[int64],
    s_tgt: NDArray[float64],
    par: bool = True,
    quad: str = "gl3",
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """
    Assemble the frozen-target source-node to target-triangle force mapping between two meshes.

    Args:
        nodes_src: [m] source mesh node coordinates with shape `(nnode_src, 3)`
        triangles_src: source node indices with shape `(ntri_src, 3)`
        nodes_tgt: [m] target mesh node coordinates with shape `(nnode_tgt, 3)`
        triangles_tgt: target node indices with shape `(ntri_tgt, 3)`
        s_tgt: [A] fixed target nodal current-potential values with shape `(nnode_tgt,)`
        par: Whether to use CPU parallelism
        quad: Triangle quadrature rule, one of `"gl2"`, `"gl3"`, or `"dunavant5"`

    Returns:
        [N/A] `(fx, fy, fz)` force mappings, each with shape `(ntri_tgt, nnode_src)`
    """
    nodes_src = ascontiguousarray(nodes_src, dtype=float64)
    triangles_src = ascontiguousarray(triangles_src, dtype=int64)
    nodes_tgt = ascontiguousarray(nodes_tgt, dtype=float64)
    triangles_tgt = ascontiguousarray(triangles_tgt, dtype=int64)
    s_tgt = ascontiguousarray(s_tgt, dtype=float64).ravel()
    fx, fy, fz = em_triangle_mesh_force_mapping(
        nodes_src, triangles_src, nodes_tgt, triangles_tgt, s_tgt, par, quad
    )
    ntri_tgt = triangles_tgt.shape[0]
    nnode_src = nodes_src.shape[0]
    return (
        ascontiguousarray(fx).reshape(ntri_tgt, nnode_src),
        ascontiguousarray(fy).reshape(ntri_tgt, nnode_src),
        ascontiguousarray(fz).reshape(ntri_tgt, nnode_src),
    )


def triangle_mesh_self_force_mapping(
    nodes: NDArray[float64],
    triangles: NDArray[int64],
    s: NDArray[float64],
    par: bool = True,
    quad: str = "gl3",
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """
    Assemble the self-excluded frozen-target source-node to target-triangle force mapping.

    Args:
        nodes: [m] mesh node coordinates with shape `(nnode, 3)`
        triangles: node indices with shape `(ntri, 3)`
        s: [A] fixed nodal current-potential values with shape `(nnode,)`
        par: Whether to use CPU parallelism
        quad: Triangle quadrature rule, one of `"gl2"`, `"gl3"`, or `"dunavant5"`

    Returns:
        [N/A] `(fx, fy, fz)` force mappings, each with shape `(ntri, nnode)`
    """
    nodes = ascontiguousarray(nodes, dtype=float64)
    triangles = ascontiguousarray(triangles, dtype=int64)
    s = ascontiguousarray(s, dtype=float64).ravel()
    fx, fy, fz = em_triangle_mesh_self_force_mapping(nodes, triangles, s, par, quad)
    ntri = triangles.shape[0]
    nnode = nodes.shape[0]
    return (
        ascontiguousarray(fx).reshape(ntri, nnode),
        ascontiguousarray(fy).reshape(ntri, nnode),
        ascontiguousarray(fz).reshape(ntri, nnode),
    )


def triangle_mesh_force_mapping_from_linear_filaments(
    xyzfil: Array3xN,
    dlxyzfil: Array3xN,
    nodes_tgt: NDArray[float64],
    triangles_tgt: NDArray[int64],
    s_tgt: NDArray[float64],
    wire_radius: float | NDArray[float64] = 0.0,
    par: bool = True,
    quad: str = "gl3",
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """
    Assemble the frozen-target source-current to target-triangle force mapping from linear filaments.

    Args:
        xyzfil: [m] x,y,z filament start coordinates
        dlxyzfil: [m] x,y,z filament segment deltas
        nodes_tgt: [m] target mesh node coordinates with shape `(nnode_tgt, 3)`
        triangles_tgt: target node indices with shape `(ntri_tgt, 3)`
        s_tgt: [A] fixed target nodal current-potential values with shape `(nnode_tgt,)`
        wire_radius: [m] filament radius, scalar or array of length `nfil`
        par: Whether to use CPU parallelism
        quad: Triangle quadrature rule, one of `"gl2"`, `"gl3"`, or `"dunavant5"`

    Returns:
        [N/A] `(fx, fy, fz)` force mappings, each with shape `(ntri_tgt, nfil)`
    """
    xyzfil = _3tup_contig(xyzfil)
    dlxyzfil = _3tup_contig(dlxyzfil)
    nodes_tgt = ascontiguousarray(nodes_tgt, dtype=float64)
    triangles_tgt = ascontiguousarray(triangles_tgt, dtype=int64)
    s_tgt = ascontiguousarray(s_tgt, dtype=float64).ravel()
    if asarray(wire_radius).ndim == 0:
        wire_radius = full(xyzfil[0].size, float(wire_radius))
    wire_radius = ascontiguousarray(wire_radius).ravel()
    fx, fy, fz = em_triangle_mesh_force_mapping_from_linear_filaments(
        xyzfil, dlxyzfil, wire_radius, nodes_tgt, triangles_tgt, s_tgt, par, quad
    )
    ntri_tgt = triangles_tgt.shape[0]
    nfil = xyzfil[0].size
    return (
        ascontiguousarray(fx).reshape(ntri_tgt, nfil),
        ascontiguousarray(fy).reshape(ntri_tgt, nfil),
        ascontiguousarray(fz).reshape(ntri_tgt, nfil),
    )


def triangle_mesh_force_mapping_from_circular_filaments(
    rfil: NDArray[float64],
    zfil: NDArray[float64],
    nodes_tgt: NDArray[float64],
    triangles_tgt: NDArray[int64],
    s_tgt: NDArray[float64],
    par: bool = True,
    quad: str = "gl3",
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """
    Assemble the frozen-target source-current to target-triangle force mapping from circular filaments.

    Args:
        rfil: [m] circular filament radii
        zfil: [m] circular filament axial coordinates
        nodes_tgt: [m] target mesh node coordinates with shape `(nnode_tgt, 3)`
        triangles_tgt: target node indices with shape `(ntri_tgt, 3)`
        s_tgt: [A] fixed target nodal current-potential values with shape `(nnode_tgt,)`
        par: Whether to use CPU parallelism
        quad: Triangle quadrature rule, one of `"gl2"`, `"gl3"`, or `"dunavant5"`

    Returns:
        [N/A] `(fx, fy, fz)` force mappings, each with shape `(ntri_tgt, nfil)`
    """
    rfil = ascontiguousarray(rfil, dtype=float64).ravel()
    zfil = ascontiguousarray(zfil, dtype=float64).ravel()
    nodes_tgt = ascontiguousarray(nodes_tgt, dtype=float64)
    triangles_tgt = ascontiguousarray(triangles_tgt, dtype=int64)
    s_tgt = ascontiguousarray(s_tgt, dtype=float64).ravel()
    fx, fy, fz = em_triangle_mesh_force_mapping_from_circular_filaments(
        rfil, zfil, nodes_tgt, triangles_tgt, s_tgt, par, quad
    )
    ntri_tgt = triangles_tgt.shape[0]
    nfil = rfil.size
    return (
        ascontiguousarray(fx).reshape(ntri_tgt, nfil),
        ascontiguousarray(fy).reshape(ntri_tgt, nfil),
        ascontiguousarray(fz).reshape(ntri_tgt, nfil),
    )


def triangle_mesh_force_mapping_from_dipoles(
    loc: Array3xN,
    moment_dir: Array3xN,
    nodes_tgt: NDArray[float64],
    triangles_tgt: NDArray[int64],
    s_tgt: NDArray[float64],
    par: bool = True,
    outer_radius: NDArray[float64] | None = None,
    quad: str = "gl3",
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """
    Assemble the frozen-target source-amplitude to target-triangle force mapping from dipoles.

    Args:
        loc: [m] x,y,z dipole locations
        moment_dir: dipole moment direction vectors, linear in scalar source amplitudes
        nodes_tgt: [m] target mesh node coordinates with shape `(nnode_tgt, 3)`
        triangles_tgt: target node indices with shape `(ntri_tgt, 3)`
        s_tgt: [A] fixed target nodal current-potential values with shape `(nnode_tgt,)`
        par: Whether to use CPU parallelism
        outer_radius: [m] radius inside which to defer to magnetized sphere calc. Defaults to zeroes.
        quad: Triangle quadrature rule, one of `"gl2"`, `"gl3"`, or `"dunavant5"`

    Returns:
        [N/source_amplitude] `(fx, fy, fz)` force mappings, each with shape `(ntri_tgt, ndip)`
    """
    loc = _3tup_contig(loc)
    moment_dir = _3tup_contig(moment_dir)
    nodes_tgt = ascontiguousarray(nodes_tgt, dtype=float64)
    triangles_tgt = ascontiguousarray(triangles_tgt, dtype=int64)
    s_tgt = ascontiguousarray(s_tgt, dtype=float64).ravel()
    outer_radius = outer_radius if outer_radius is not None else zeros_like(loc[0])
    outer_radius = ascontiguousarray(outer_radius).ravel()
    fx, fy, fz = em_triangle_mesh_force_mapping_from_dipoles(
        loc, moment_dir, outer_radius, nodes_tgt, triangles_tgt, s_tgt, par, quad
    )
    ntri_tgt = triangles_tgt.shape[0]
    ndip = loc[0].size
    return (
        ascontiguousarray(fx).reshape(ntri_tgt, ndip),
        ascontiguousarray(fy).reshape(ntri_tgt, ndip),
        ascontiguousarray(fz).reshape(ntri_tgt, ndip),
    )


def inductance_piecewise_linear_filaments(
    xyzfil0: Array3xN,
    dlxyzfil0: Array3xN,
    xyzfil1: Array3xN,
    dlxyzfil1: Array3xN,
    wire_radius: float | NDArray[float64] = 0.0,
) -> float:
    """
    Estimate the inductive coupling between two piecewise-linear current filaments.

    Uses the line-integral form `M = ∮ A_source · dl_target`, evaluated at the
    target segment midpoints with the finite-radius
    [`vector_potential_linear_filament`][cfsem.vector_potential_linear_filament] kernel.

    Assumes:

    * Thin, well-behaved filaments
    * Uniform current distribution within segments
        * Low frequency operation; no skin effect
    * Vacuum permeability everywhere
    * Each filament has a constant current in all segments
      (otherwise we need an interaction matrix)

    Args:
        xyzfil0: [m] Nx3 point series describing the filament origins
        dlxyzfil0: [m] Nx3 length vector of each filament
        xyzfil1: [m] Nx3 point series describing the filament origins
        dlxyzfil1: [m] Nx3 length vector of each filament
        wire_radius: [m] source filament radius, scalar or array of length `N`
    Returns:
        [H] Scalar inductance
    """
    xyzfil0 = _3tup_contig(xyzfil0)
    dlxyzfil0 = _3tup_contig(dlxyzfil0)
    xyzfil1 = _3tup_contig(xyzfil1)
    dlxyzfil1 = _3tup_contig(dlxyzfil1)
    nfil0 = xyzfil0[0].size
    if asarray(wire_radius).ndim == 0:
        wire_radius = full(nfil0, float(wire_radius))
    wire_radius = ascontiguousarray(wire_radius).ravel()

    return em_inductance_piecewise_linear_filaments(xyzfil0, dlxyzfil0, xyzfil1, dlxyzfil1, wire_radius)


def inductance_linear_filaments(
    xyzfil_tgt: Array3xN,
    dlxyzfil_tgt: Array3xN,
    xyzfil_src: Array3xN,
    dlxyzfil_src: Array3xN,
    wire_radius_src: float | NDArray[float64] = 0.0,
    par: bool = True,
    output: Literal["vector", "matrix"] = "vector",
) -> NDArray[float64]:
    """
    Estimate inductive coupling from source filament segments to target filament segments.

    Uses the same finite-radius `A·dl` kernel as
    [`inductance_piecewise_linear_filaments`][cfsem.inductance_piecewise_linear_filaments],
    but with a disjoint source/target segment API. The vector result is one inductive
    coupling value per target segment. The matrix result is row-major `(nsrc, ntgt)`.

    Args:
        xyzfil_tgt: [m] target filament segment start points
        dlxyzfil_tgt: [m] target filament segment deltas
        xyzfil_src: [m] source filament segment start points
        dlxyzfil_src: [m] source filament segment deltas
        wire_radius_src: [m] source filament radius, scalar or array of length `nsrc`
        par: Whether to use CPU parallelism for `output="matrix"`
        output: `"vector"` for contracted target couplings,
            or `"matrix"` for explicit row-major `(nsrc, ntgt)` source-target interaction matrix

    Returns:
        [H] Target coupling vector of length `ntgt`,
        or explicit `(nsrc, ntgt)` interaction matrix if `output="matrix"`
    """
    xyzfil_tgt = _3tup_contig(xyzfil_tgt)
    dlxyzfil_tgt = _3tup_contig(dlxyzfil_tgt)
    xyzfil_src = _3tup_contig(xyzfil_src)
    dlxyzfil_src = _3tup_contig(dlxyzfil_src)
    nsrc = xyzfil_src[0].size
    if asarray(wire_radius_src).ndim == 0:
        wire_radius_src = full(nsrc, float(wire_radius_src))
    wire_radius_src = ascontiguousarray(wire_radius_src).ravel()

    if output == "vector":
        return em_inductance_linear_filaments(
            xyzfil_tgt,
            dlxyzfil_tgt,
            xyzfil_src,
            dlxyzfil_src,
            wire_radius_src,
        )
    if output == "matrix":
        out = em_inductance_linear_filaments_matrix(
            xyzfil_tgt,
            dlxyzfil_tgt,
            xyzfil_src,
            dlxyzfil_src,
            wire_radius_src,
            par,
        )
        ntgt = xyzfil_tgt[0].size
        return out.reshape((nsrc, ntgt))
    raise ValueError("output must be 'vector' or 'matrix'")


def gs_operator_order2(rs: NDArray[float64], zs: NDArray[float64]) -> Array3xN:
    """Build second-order Grad-Shafranov operator in triplet format.
    Assumes regular grid spacing.

    Args:
        rs: [m] r-coordinates of finite difference grid
        zs: [m] z-coordinates of finite difference grid

    Returns:
        Differential operator as triplet format sparse matrix
    """
    rs, zs = _2tup_contig((rs, zs))
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
    rs, zs = _2tup_contig((rs, zs))
    return em_gs_operator_order4(rs, zs)


def filament_helix_path(
    path: Array3xN,
    helix_start_offset: tuple[float, float, float],
    twist_pitch: float,
    angle_offset: float,
) -> Array3xN:
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


def rotate_filaments_about_path(path: Array3xN, angle_offset: float, fils: Array3xN) -> Array3xN:
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


def flux_density_circular_filament_cartesian(
    ifil: NDArray[float64],
    rfil: NDArray[float64],
    zfil: NDArray[float64],
    xyzp: Array3xN,
    par: bool = True,
) -> Array3xN:
    """
    Flux density of a circular filament in cartesian form
    at a set of locations given in cartesian coordinates.

    Args:
        ifil: [A] filament current
        rfil: [m] filament R-coord
        zfil: [m] filament Z-coord
        xyzp: [m] x,y,z coords of observation points
        par: Whether to use CPU parallelism

    Returns:
        [T] flux density
    """
    ifil, rfil, zfil = _3tup_contig((ifil, rfil, zfil))
    xyzp = _3tup_contig(xyzp)
    bx, by, bz = em_flux_density_circular_filament_cartesian(ifil, rfil, zfil, xyzp, par)  # [T]

    return bx, by, bz  # type: ignore


def mutual_inductance_circular_to_linear(
    rfil: NDArray[float64],
    zfil: NDArray[float64],
    nfil: NDArray[float64],
    xyzfil: Array3xN,
    dlxyzfil: Array3xN,
    par: bool = True,
) -> NDArray[float64]:
    """
    Mutual inductance between a collection of circular filaments and a piecewise-linear filament.
    This method is much faster (~100x typically) than discretizing the circular loop
    into linear segments and using Neumann's formula.

    Args:
        rfil: [m] filament R-coord
        zfil: [m] filament Z-coord
        nfil: [dimensionless] filament number of turns
        xyzfil: [m] x,y,z coords of current filament origins (start of segment)
        dlxyzfil: [m] x,y,z length delta of current filaments
        par: Whether to use CPU parallelism

    Returns:
        [H] mutual inductance
    """
    rfil, zfil, nfil = _3tup_contig((rfil, zfil, nfil))
    xyzfil = _3tup_contig(xyzfil)
    dlxyzfil = _3tup_contig(dlxyzfil)
    m = em_mutual_inductance_circular_to_linear(rfil, zfil, nfil, xyzfil, dlxyzfil, par)

    return m  # [H]


def flux_density_dipole(
    loc: Array3xN,
    moment: Array3xN,
    xyzp: Array3xN,
    par: bool = True,  # Ordered for backwards compatibility
    outer_radius: NDArray[float64] | None = None,
) -> Array3xN:
    """
    Magnetic flux density of a dipole in cartesian coordiantes.

    Args:
        loc: [m] x,y,z coordinates of dipole
        moment: [A-m^2] dipole magnetic moment vector
        xyzp: [m] x,y,z coords of observation points
        par: Whether to use CPU parallelism
        outer_radius: [m] radius inside which to defer to magnetized sphere calc. Defaults to zeroes.


    Returns:
        [T] flux density
    """
    loc = _3tup_contig(loc)
    moment = _3tup_contig(moment)
    xyzp = _3tup_contig(xyzp)
    outer_radius = outer_radius if outer_radius is not None else zeros_like(loc[0])
    outer_radius = ascontiguousarray(outer_radius).ravel()

    bx, by, bz = em_flux_density_dipole(loc, moment, xyzp, outer_radius, par)  # [T]

    return bx, by, bz  # type: ignore


def vector_potential_dipole(
    loc: Array3xN,
    moment: Array3xN,
    xyzp: Array3xN,
    par: bool = True,  # Ordered for backwards compatibility
    outer_radius: NDArray[float64] | None = None,
) -> Array3xN:
    """
    Magnetic vector potential of a dipole in cartesian coordiantes.

    Args:
        loc: [m] x,y,z coordinates of dipole
        moment: [A-m^2] dipole magnetic moment vector
        xyzp: [m] x,y,z coords of observation points
        par: Whether to use CPU parallelism
        outer_radius: [m] radius inside which to defer to magnetized sphere calc. Defaults to zeroes.

    Returns:
        [V⋅s⋅m-1] vector potential
    """
    loc = _3tup_contig(loc)
    moment = _3tup_contig(moment)
    xyzp = _3tup_contig(xyzp)
    outer_radius = outer_radius if outer_radius is not None else zeros_like(loc[0])
    outer_radius = ascontiguousarray(outer_radius).ravel()

    ax, ay, az = em_vector_potential_dipole(loc, moment, xyzp, outer_radius, par)  # [T]

    return ax, ay, az  # type: ignore


def body_force_density_circular_filament_cartesian(
    ifil: NDArray[float64],
    rfil: NDArray[float64],
    zfil: NDArray[float64],
    obs: Array3xN,
    j: Array3xN,
    par: bool = True,
) -> Array3xN:
    """
    JxB (Lorentz) body force density (per volume) in cartesian form due to a circular current
    filament segment at an observation point in cartesian form with some current density (per area).

    Args:
        ifil: [A] filament current
        rfil: [m] filament R-coord
        zfil: [m] filament Z-coord
        obs: [m] x,y,z coords of observation locations
        j: [A/m^2] current density vector at observation locations
        par: Whether to use CPU parallelism

    Returns:
        [N/m^3] body force density
    """
    ifil, rfil, zfil = _3tup_contig((ifil, rfil, zfil))
    obs = _3tup_contig(obs)
    j = _3tup_contig(j)
    jxbx, jxby, jxbz = em_body_force_density_circular_filament_cartesian(
        ifil, rfil, zfil, obs, j, par
    )  # [N/m^3]

    return jxbx, jxby, jxbz  # type: ignore


def body_force_density_linear_filament(
    xyzfil: Array3xN,
    dlxyzfil: Array3xN,
    ifil: NDArray[float64],
    obs: Array3xN,
    j: Array3xN,
    wire_radius: float | NDArray[float64] = 0.0,
    par: bool = True,
) -> Array3xN:
    """
    JxB (Lorentz) body force density (per volume) due to a linear current
    filament segment at an observation point with some current density (per area).

    Args:
        xyzfil: [m] x,y,z coords of current filament origins (start of segment)
        dlxyzfil: [m] x,y,z length delta of current filaments
        ifil: [A] filament current
        obs: [m] x,y,z coords of observation locations
        j: [A/m^2] current density vector at observation locations
        wire_radius: [m] filament radius, scalar or array of length `m`
        par: Whether to use CPU parallelism

    Returns:
        [N/m^3] body force density
    """
    xyzfil = _3tup_contig(xyzfil)
    dlxyzfil = _3tup_contig(dlxyzfil)
    ifil = ascontiguousarray(ifil).ravel()
    obs = _3tup_contig(obs)
    j = _3tup_contig(j)
    if asarray(wire_radius).ndim == 0:
        wire_radius = full(ifil.size, float(wire_radius))
    wire_radius = ascontiguousarray(wire_radius).ravel()
    jxbx, jxby, jxbz = em_body_force_density_linear_filament(
        xyzfil, dlxyzfil, ifil, obs, j, wire_radius, par
    )  # [N/m^3]

    return jxbx, jxby, jxbz  # type: ignore


def _3tup_contig(
    t: Array3xN,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """Make contiguous references or copies to arrays in a 3-tuple.
    Only copies data if it is not already contiguous."""
    return (
        ascontiguousarray(t[0]).ravel(),
        ascontiguousarray(t[1]).ravel(),
        ascontiguousarray(t[2]).ravel(),
    )


def _2tup_contig(
    t: tuple[NDArray[float64], NDArray[float64]],
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Make contiguous references or copies to arrays in a 2-tuple.
    Only copies data if it is not already contiguous."""
    return (ascontiguousarray(t[0]).ravel(), ascontiguousarray(t[1]).ravel())
