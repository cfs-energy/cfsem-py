use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

use super::triangle_potential::UniformTriangle;
use super::{
    triangle_basis_current_densities, triangle_basis_current_density, triangle_current_density,
};
use crate::MU0_OVER_4PI;
use crate::chunksize;
use crate::macros::{check_length_3tup, mut_par_chunks_3tup, par_chunks_3tup};
use crate::math::Scalar;
use crate::math::scale3;
use crate::mesh::TriangleMeshView;

/// Apply the exact uniform-triangle scalar potential to one constant current density.
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21).
#[inline]
fn triangle_vector_potential_exact<T: Scalar>(
    triangle: &UniformTriangle<T>,
    current_density: [T; 3],
    obs: [T; 3],
) -> [T; 3] {
    scale3(
        current_density,
        crate::math::cast::<T>(MU0_OVER_4PI) * triangle.scalar_potential(obs),
    )
}

/// Apply one exact uniform-triangle scalar potential to all three basis currents.
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21).
#[inline]
fn triangle_vector_potential_bases_exact<T: Scalar>(
    triangle: &UniformTriangle<T>,
    obs: [T; 3],
) -> [[T; 3]; 3] {
    let nodes = triangle.nodes();
    let basis = triangle_basis_current_densities(nodes[0], nodes[1], nodes[2]);
    let factor = crate::math::cast::<T>(MU0_OVER_4PI) * triangle.scalar_potential(obs);
    [
        scale3(basis[0], factor),
        scale3(basis[1], factor),
        scale3(basis[2], factor),
    ]
}

/// Magnetic vector potential (A-field) contribution of a given triangle's basis
/// function with unit weighting to a given observation point.
///
/// Assumes a basis function living on the triangle's first node.
///
/// Uses the exact uniform-triangle scalar potential. The result is finite and
/// continuous on triangle interiors, edges, and vertices.
///
/// Args:
///     n0: Basis node coordinates `[x, y, z]` (m).
///     n1: Triangle vertex 1 coordinates `[x, y, z]` (m).
///     n2: Triangle vertex 2 coordinates `[x, y, z]` (m).
///     obs: Observation point `[x, y, z]` (m).
/// Returns:
///     Basis-function magnetic vector potential `[ax, ay, az]` (V*s/(A*m)).
///
/// References:
/// - \[5\], Eq. (3.24) for the stream-function surface current construction,
///   Eq. (4.6) for the constant current density on a linear triangle, and
///   Eqs. (5.3)-(5.5) for triangle vector-potential integrals.
/// - \[3\] for `1 / R` potential integrals on polygonal and polyhedral elements.
/// - \[2\] for numerical treatment of triangle `1 / R` and `∇(1 / R)`
///   integrals with linear shape functions.
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for the robust exact potential and
///   its limiting cases.
#[inline]
pub fn triangle_vector_potential_basis<T: Scalar>(
    n0: [T; 3],
    n1: [T; 3],
    n2: [T; 3],
    obs: [T; 3],
) -> [T; 3] {
    let (_, jref) = triangle_basis_current_density(n0, n1, n2); // [m^2], [1/m]
    triangle_vector_potential_exact(&UniformTriangle::new(n0, n1, n2), jref, obs)
}

/// Magnetic vector potential (A-field) of triangular surface current density
/// distribution at a target point due to scalar current density potential `s`
/// at each node.
///
/// For physical intuition, the current density is related to the difference
/// in potential between the nodes; for example, in a strip discretized into triangles
/// with s=s0 on one side of the strip and s=-s0 on the other side of the strip,
/// the total current on the strip (and its effective filament current) is equal to s0.
///
/// Uses the exact uniform-triangle scalar potential with the constant physical
/// current density induced by the three nodal values.
///
/// Args:
///     n0: Triangle vertex 0 coordinates `[x, y, z]` (m).
///     n1: Triangle vertex 1 coordinates `[x, y, z]` (m).
///     n2: Triangle vertex 2 coordinates `[x, y, z]` (m).
///     s: Nodal current-potential values `[s0, s1, s2]` (A).
///     obs: Observation point `[x, y, z]` (m).
/// Returns:
///     Magnetic vector potential `[ax, ay, az]` (V*s/m).
///
/// References:
/// - \[5\], Eq. (3.24), Eq. (4.6), and Eqs. (5.3)-(5.5).
/// - \[3\], pp. 276-281.
/// - \[2\], pp. 1448-1455.
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for the robust exact potential and
///   its limiting cases.
#[inline]
pub fn vector_potential_triangle<T: Scalar>(
    n0: [T; 3],
    n1: [T; 3],
    n2: [T; 3],
    s: [T; 3],
    obs: [T; 3],
) -> [T; 3] {
    let current_density = triangle_current_density(n0, n1, n2, s); // [A/m]
    triangle_vector_potential_exact(&UniformTriangle::new(n0, n1, n2), current_density, obs)
}

#[inline]
fn validate_vector_potential_mapping_inputs(
    outx: &[f64],
    outy: &[f64],
    outz: &[f64],
    nobs: usize,
    nnode: usize,
) -> Result<(), &'static str> {
    let expected = nobs
        .checked_mul(nnode)
        .ok_or("Vector-potential mapping size overflow")?;
    if outx.len() != expected || outy.len() != expected || outz.len() != expected {
        return Err("Output dimension mismatch");
    }
    Ok(())
}

#[inline]
fn vector_potential_triangle_mesh_mapping_chunk(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let nobs = obs.0.len(); // [-]
    check_length_3tup!(nobs, obs);
    validate_vector_potential_mapping_inputs(out.0, out.1, out.2, nobs, mesh.nnode())?;

    out.0.fill(0.0); // [V*s/(A*m)]
    out.1.fill(0.0); // [V*s/(A*m)]
    out.2.fill(0.0); // [V*s/(A*m)]

    for iobs in 0..nobs {
        let row_offset = iobs * mesh.nnode(); // [-]
        let obs_i = [obs.0[iobs], obs.1[iobs], obs.2[iobs]]; // [m]

        for itri in 0..mesh.len() {
            let (tri_nodes, idx) = mesh.triangle_nodes_and_indices(itri);
            let triangle = UniformTriangle::new(tri_nodes[0], tri_nodes[1], tri_nodes[2]);
            let [a0, a1, a2] = triangle_vector_potential_bases_exact(&triangle, obs_i); // [V*s/(A*m)]

            out.0[row_offset + idx[0]] += a0[0]; // [V*s/(A*m)]
            out.1[row_offset + idx[0]] += a0[1]; // [V*s/(A*m)]
            out.2[row_offset + idx[0]] += a0[2]; // [V*s/(A*m)]

            out.0[row_offset + idx[1]] += a1[0]; // [V*s/(A*m)]
            out.1[row_offset + idx[1]] += a1[1]; // [V*s/(A*m)]
            out.2[row_offset + idx[1]] += a1[2]; // [V*s/(A*m)]

            out.0[row_offset + idx[2]] += a2[0]; // [V*s/(A*m)]
            out.1[row_offset + idx[2]] += a2[1]; // [V*s/(A*m)]
            out.2[row_offset + idx[2]] += a2[2]; // [V*s/(A*m)]
        }
    }

    Ok(())
}

#[inline]
fn vector_potential_triangle_mesh_inner(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let nobs = obs.0.len();
    check_length_3tup!(nobs, obs);
    check_length_3tup!(nobs, out);
    mesh.validate_nodal_values(s)?;

    out.0.fill(0.0); // [V*s/m]
    out.1.fill(0.0); // [V*s/m]
    out.2.fill(0.0); // [V*s/m]

    for i in 0..nobs {
        let obs_i = [obs.0[i], obs.1[i], obs.2[i]];
        for j in 0..mesh.len() {
            let tri_nodes = mesh.triangle_nodes(j);
            let tri_s = mesh.triangle_scalars(j, s);
            let contrib =
                vector_potential_triangle(tri_nodes[0], tri_nodes[1], tri_nodes[2], tri_s, obs_i);
            out.0[i] += contrib[0]; // [V*s/m]
            out.1[i] += contrib[1]; // [V*s/m]
            out.2[i] += contrib[2]; // [V*s/m]
        }
    }

    Ok(())
}

/// Assemble the dense source-node to target-point vector-potential mapping for a triangle mesh.
///
/// Args:
///     obs: Observation point component slices `(x, y, z)` (m).
///     mesh: Borrowed triangle-mesh geometry view.
///     out: Output mapping buffers `(ax_map, ay_map, az_map)` (V*s/(m*A)), each row-major in
///         `(observation point, source node)` order.
///
/// Returns:
///     `Ok(())` after writing the dense mapping to `out`, or an error if the mesh
///     geometry or slice dimensions are inconsistent.
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for each exact triangle interaction.
#[inline]
pub fn vector_potential_triangle_mesh_mapping(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    vector_potential_triangle_mesh_mapping_chunk(obs, mesh, out)
}

/// Parallel variant of [`vector_potential_triangle_mesh_mapping`].
///
/// Args:
///     obs: Observation point component slices `(x, y, z)` (m).
///     mesh: Borrowed triangle-mesh geometry view.
///     out: Output mapping buffers `(ax_map, ay_map, az_map)` (V*s/(m*A)), each row-major in
///         `(observation point, source node)` order.
///
/// Returns:
///     `Ok(())` after writing the dense mapping to `out`, or an error if the mesh
///     geometry or slice dimensions are inconsistent.
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for each exact triangle interaction.
#[inline]
pub fn vector_potential_triangle_mesh_mapping_par(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let nobs = obs.0.len(); // [-]
    check_length_3tup!(nobs, obs);
    validate_vector_potential_mapping_inputs(out.0, out.1, out.2, nobs, mesh.nnode())?;

    if nobs == 0 || mesh.nnode() == 0 {
        return vector_potential_triangle_mesh_mapping_chunk(obs, mesh, out);
    }

    let nrow = chunksize(nobs); // [-]
    let nflat = nrow
        .checked_mul(mesh.nnode())
        .ok_or("Vector-potential mapping size overflow")?;
    let (xpc, ypc, zpc) = par_chunks_3tup!(obs, nrow);
    let axc = out.0.par_chunks_mut(nflat);
    let ayc = out.1.par_chunks_mut(nflat);
    let azc = out.2.par_chunks_mut(nflat);

    (axc, ayc, azc, xpc, ypc, zpc)
        .into_par_iter()
        .try_for_each(|(ax, ay, az, xp, yp, zp)| {
            vector_potential_triangle_mesh_mapping_chunk((xp, yp, zp), mesh, (ax, ay, az))
        })?;

    Ok(())
}

/// Apply a dense source-node to target-point vector-potential mapping to nodal current-potential values.
///
/// Args:
///     ax_map: Row-major `Ax` mapping in `(observation point, source node)` order (V*s/(m*A)).
///     ay_map: Row-major `Ay` mapping in `(observation point, source node)` order (V*s/(m*A)).
///     az_map: Row-major `Az` mapping in `(observation point, source node)` order (V*s/(m*A)).
///     s: Nodal current-potential values (A).
///     out: Output pointwise magnetic vector potential `(ax, ay, az)` (V*s/m).
///
/// Returns:
///     `Ok(())` after writing the contracted field to `out`, or an error if the mapping
///     or output dimensions are inconsistent.
#[inline]
pub fn triangle_mesh_vector_potential_from_potential_vectors(
    ax_map: &[f64],
    ay_map: &[f64],
    az_map: &[f64],
    s: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    if ay_map.len() != ax_map.len() || az_map.len() != ax_map.len() {
        return Err("Vector-potential mapping dimension mismatch");
    }
    if s.is_empty() {
        if ax_map.is_empty() && out.0.is_empty() && out.1.is_empty() && out.2.is_empty() {
            return Ok(());
        }
        return Err("Vector-potential mapping dimension mismatch");
    }
    if !ax_map.len().is_multiple_of(s.len()) {
        return Err("Vector-potential mapping dimension mismatch");
    }

    let nobs = ax_map.len() / s.len(); // [-]
    check_length_3tup!(nobs, out);

    for iobs in 0..nobs {
        let row_offset = iobs * s.len(); // [-]
        let mut ax = 0.0; // [V*s/m]
        let mut ay = 0.0; // [V*s/m]
        let mut az = 0.0; // [V*s/m]
        for inode in 0..s.len() {
            ax += ax_map[row_offset + inode] * s[inode]; // [V*s/m]
            ay += ay_map[row_offset + inode] * s[inode]; // [V*s/m]
            az += az_map[row_offset + inode] * s[inode]; // [V*s/m]
        }
        out.0[iobs] = ax; // [V*s/m]
        out.1[iobs] = ay; // [V*s/m]
        out.2[iobs] = az; // [V*s/m]
    }

    Ok(())
}

/// Vector potential contribution from a triangle mesh with nodal stream-function values.
///
/// Args:
///     obs: Observation point component slices `(x, y, z)` (m).
///     mesh: Borrowed triangle-mesh geometry view.
///     s: Nodal current-potential values (A).
///     out: Output buffers for magnetic vector potential `(ax, ay, az)` (V*s/m).
///
/// Returns:
///     `Ok(())` after writing the vector potential to `out`, or an error if the mesh
///     geometry or slice dimensions are inconsistent.
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for each exact triangle interaction.
#[inline]
pub fn vector_potential_triangle_mesh(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    vector_potential_triangle_mesh_inner(obs, mesh, s, out)
}

/// Vector potential contribution from a triangle mesh with nodal stream-function values.
/// This variant is parallelized over chunks of observation points.
///
/// Args:
///     obs: Observation point component slices `(x, y, z)` (m).
///     mesh: Borrowed triangle-mesh geometry view.
///     s: Nodal current-potential values (A).
///     out: Output buffers for magnetic vector potential `(ax, ay, az)` (V*s/m).
///
/// Returns:
///     `Ok(())` after writing the vector potential to `out`, or an error if the mesh
///     geometry or slice dimensions are inconsistent.
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for each exact triangle interaction.
#[inline]
pub fn vector_potential_triangle_mesh_par(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    mesh.validate_nodal_values(s)?;
    let n = chunksize(obs.0.len());
    let (xpc, ypc, zpc) = par_chunks_3tup!(obs, n);
    let (axc, ayc, azc) = mut_par_chunks_3tup!(out, n);

    (axc, ayc, azc, xpc, ypc, zpc)
        .into_par_iter()
        .try_for_each(|(ax, ay, az, xp, yp, zp)| {
            vector_potential_triangle_mesh_inner((xp, yp, zp), mesh, s, (ax, ay, az))
        })?;

    Ok(())
}
