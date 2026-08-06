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
use crate::math::{cross3, scale3};
use crate::mesh::TriangleMeshView;

/// Apply the exact uniform-triangle potential gradient to one constant current density.
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21).
#[inline]
fn triangle_flux_density_exact<T: Scalar>(
    triangle: &UniformTriangle<T>,
    current_density: [T; 3],
    obs: [T; 3],
) -> [T; 3] {
    if triangle.contains_on_surface(obs) {
        return [T::ZERO; 3];
    }
    scale3(
        cross3(triangle.scalar_potential_gradient(obs), current_density),
        crate::math::cast::<T>(MU0_OVER_4PI),
    )
}

/// Apply one exact uniform-triangle potential gradient to all three basis currents.
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21).
#[inline]
fn triangle_flux_density_bases_exact<T: Scalar>(
    triangle: &UniformTriangle<T>,
    obs: [T; 3],
) -> [[T; 3]; 3] {
    if triangle.contains_on_surface(obs) {
        return [[T::ZERO; 3]; 3];
    }
    let nodes = triangle.nodes();
    let basis = triangle_basis_current_densities(nodes[0], nodes[1], nodes[2]);
    let gradient = triangle.scalar_potential_gradient(obs);
    let scale = crate::math::cast::<T>(MU0_OVER_4PI);
    [
        scale3(cross3(gradient, basis[0]), scale),
        scale3(cross3(gradient, basis[1]), scale),
        scale3(cross3(gradient, basis[2]), scale),
    ]
}

/// Magnetic flux density (B-field) contribution of a given triangle's basis function
/// with unit weighting to a given observation point.
///
/// Assumes a basis function living on the triangle's first node.
///
/// Uses the exact gradient of the uniform-triangle scalar potential off the
/// source. Directly on the finite source triangle, this triangle's contribution
/// is defined to be exactly zero.
///
/// Args:
///     n0: Basis node coordinates `[x, y, z]` (m).
///     n1: Triangle vertex 1 coordinates `[x, y, z]` (m).
///     n2: Triangle vertex 2 coordinates `[x, y, z]` (m).
///     obs: Observation point `[x, y, z]` (m).
///
/// Returns:
///     Basis-function magnetic flux density `[bx, by, bz]` (T/A).
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for the exact potential gradient.
#[inline]
pub fn triangle_flux_density_basis<T: Scalar>(
    n0: [T; 3],
    n1: [T; 3],
    n2: [T; 3],
    obs: [T; 3],
) -> [T; 3] {
    let (_, jref) = triangle_basis_current_density(n0, n1, n2); // [m^2], [1/m]
    triangle_flux_density_exact(&UniformTriangle::new(n0, n1, n2), jref, obs)
}

/// Flux density (B-field) of triangular surface current density distribution
/// at a target point due to scalar current density potential `s`
/// at each node.
///
/// For physical intuition, the current density is related to the gradient
/// in potential between the nodes; for example, in a strip discretized into triangles
/// with s=s0 on one side of the strip and s=-s0 on the other side of the strip,
/// the total current on the strip (and its effective filament current) is equal to s0.
/// Off the finite triangle this uses the exact gradient of its uniform-source
/// potential. Directly on the triangle, this source contribution is defined as zero.
///
/// Args:
///     n0: Triangle vertex 0 coordinates `[x, y, z]` (m).
///     n1: Triangle vertex 1 coordinates `[x, y, z]` (m).
///     n2: Triangle vertex 2 coordinates `[x, y, z]` (m).
///     s: Nodal current-potential values `[s0, s1, s2]` (A).
///     obs: Observation point `[x, y, z]` (m).
///
/// Returns:
///     Magnetic flux density `[bx, by, bz]` (T).
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for the exact potential gradient.
#[inline]
pub fn flux_density_triangle<T: Scalar>(
    n0: [T; 3],
    n1: [T; 3],
    n2: [T; 3],
    s: [T; 3],
    obs: [T; 3],
) -> [T; 3] {
    let current_density = triangle_current_density(n0, n1, n2, s); // [A/m]
    triangle_flux_density_exact(&UniformTriangle::new(n0, n1, n2), current_density, obs)
}

#[inline]
fn validate_flux_density_mapping_inputs(
    outx: &[f64],
    outy: &[f64],
    outz: &[f64],
    nobs: usize,
    nnode: usize,
) -> Result<(), &'static str> {
    let expected = nobs
        .checked_mul(nnode)
        .ok_or("Flux-density mapping size overflow")?;
    if outx.len() != expected || outy.len() != expected || outz.len() != expected {
        return Err("Output dimension mismatch");
    }
    Ok(())
}

#[inline]
fn flux_density_triangle_mesh_mapping_chunk(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let nobs = obs.0.len(); // [-]
    check_length_3tup!(nobs, obs);
    validate_flux_density_mapping_inputs(out.0, out.1, out.2, nobs, mesh.nnode())?;

    out.0.fill(0.0); // [T/A]
    out.1.fill(0.0); // [T/A]
    out.2.fill(0.0); // [T/A]

    for iobs in 0..nobs {
        let row_offset = iobs * mesh.nnode(); // [-]
        let obs_i = [obs.0[iobs], obs.1[iobs], obs.2[iobs]]; // [m]

        for itri in 0..mesh.len() {
            let (tri_nodes, idx) = mesh.triangle_nodes_and_indices(itri);

            let triangle = UniformTriangle::new(tri_nodes[0], tri_nodes[1], tri_nodes[2]);
            let [b0, b1, b2] = triangle_flux_density_bases_exact(&triangle, obs_i); // [T/A]

            out.0[row_offset + idx[0]] += b0[0]; // [T/A]
            out.1[row_offset + idx[0]] += b0[1]; // [T/A]
            out.2[row_offset + idx[0]] += b0[2]; // [T/A]

            out.0[row_offset + idx[1]] += b1[0]; // [T/A]
            out.1[row_offset + idx[1]] += b1[1]; // [T/A]
            out.2[row_offset + idx[1]] += b1[2]; // [T/A]

            out.0[row_offset + idx[2]] += b2[0]; // [T/A]
            out.1[row_offset + idx[2]] += b2[1]; // [T/A]
            out.2[row_offset + idx[2]] += b2[2]; // [T/A]
        }
    }

    Ok(())
}

#[inline]
fn flux_density_triangle_mesh_inner(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let nobs = obs.0.len();
    check_length_3tup!(nobs, obs);
    check_length_3tup!(nobs, out);
    mesh.validate_nodal_values(s)?;

    out.0.fill(0.0); // [T]
    out.1.fill(0.0); // [T]
    out.2.fill(0.0); // [T]

    for i in 0..nobs {
        let obs_i = [obs.0[i], obs.1[i], obs.2[i]];
        for j in 0..mesh.len() {
            let tri_nodes = mesh.triangle_nodes(j);
            let tri_s = mesh.triangle_scalars(j, s);
            let contrib =
                flux_density_triangle(tri_nodes[0], tri_nodes[1], tri_nodes[2], tri_s, obs_i);
            out.0[i] += contrib[0]; // [T]
            out.1[i] += contrib[1]; // [T]
            out.2[i] += contrib[2]; // [T]
        }
    }

    Ok(())
}

/// Assemble the dense source-node to target-point flux-density mapping for a triangle mesh.
///
/// Args:
///     obs: Observation point component slices `(x, y, z)` (m).
///     mesh: Borrowed triangle-mesh geometry view.
///     out: Output mapping buffers `(bx_map, by_map, bz_map)` (T/A), each row-major in
///         `(observation point, source node)` order.
///
/// Returns:
///     `Ok(())` after writing the dense mapping to `out`, or an error if the mesh
///     geometry or slice dimensions are inconsistent.
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for each exact triangle interaction.
#[inline]
pub fn flux_density_triangle_mesh_mapping(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    flux_density_triangle_mesh_mapping_chunk(obs, mesh, out)
}

/// Parallel variant of [`flux_density_triangle_mesh_mapping`].
///
/// Args:
///     obs: Observation point component slices `(x, y, z)` (m).
///     mesh: Borrowed triangle-mesh geometry view.
///     out: Output mapping buffers `(bx_map, by_map, bz_map)` (T/A), each row-major in
///         `(observation point, source node)` order.
///
/// Returns:
///     `Ok(())` after writing the dense mapping to `out`, or an error if the mesh
///     geometry or slice dimensions are inconsistent.
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for each exact triangle interaction.
#[inline]
pub fn flux_density_triangle_mesh_mapping_par(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let nobs = obs.0.len(); // [-]
    check_length_3tup!(nobs, obs);
    validate_flux_density_mapping_inputs(out.0, out.1, out.2, nobs, mesh.nnode())?;

    if nobs == 0 || mesh.nnode() == 0 {
        return flux_density_triangle_mesh_mapping_chunk(obs, mesh, out);
    }

    let nrow = chunksize(nobs); // [-]
    let nflat = nrow
        .checked_mul(mesh.nnode())
        .ok_or("Flux-density mapping size overflow")?;
    let (xpc, ypc, zpc) = par_chunks_3tup!(obs, nrow);
    let bxc = out.0.par_chunks_mut(nflat);
    let byc = out.1.par_chunks_mut(nflat);
    let bzc = out.2.par_chunks_mut(nflat);

    (bxc, byc, bzc, xpc, ypc, zpc)
        .into_par_iter()
        .try_for_each(|(bx, by, bz, xp, yp, zp)| {
            flux_density_triangle_mesh_mapping_chunk((xp, yp, zp), mesh, (bx, by, bz))
        })?;

    Ok(())
}

/// Apply a dense source-node to target-point flux-density mapping to nodal current-potential values.
///
/// Args:
///     bx_map: Row-major `Bx` mapping in `(observation point, source node)` order (T/A).
///     by_map: Row-major `By` mapping in `(observation point, source node)` order (T/A).
///     bz_map: Row-major `Bz` mapping in `(observation point, source node)` order (T/A).
///     s: Nodal current-potential values (A).
///     out: Output pointwise magnetic flux density `(bx, by, bz)` (T).
///
/// Returns:
///     `Ok(())` after writing the contracted field to `out`, or an error if the mapping
///     or output dimensions are inconsistent.
#[inline]
pub fn triangle_mesh_flux_density_from_potential_vectors(
    bx_map: &[f64],
    by_map: &[f64],
    bz_map: &[f64],
    s: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    if by_map.len() != bx_map.len() || bz_map.len() != bx_map.len() {
        return Err("Flux-density mapping dimension mismatch");
    }
    if s.is_empty() {
        if bx_map.is_empty() && out.0.is_empty() && out.1.is_empty() && out.2.is_empty() {
            return Ok(());
        }
        return Err("Flux-density mapping dimension mismatch");
    }
    if !bx_map.len().is_multiple_of(s.len()) {
        return Err("Flux-density mapping dimension mismatch");
    }

    let nobs = bx_map.len() / s.len(); // [-]
    check_length_3tup!(nobs, out);

    for iobs in 0..nobs {
        let row_offset = iobs * s.len(); // [-]
        let mut bx = 0.0; // [T]
        let mut by = 0.0; // [T]
        let mut bz = 0.0; // [T]
        for inode in 0..s.len() {
            bx += bx_map[row_offset + inode] * s[inode]; // [T]
            by += by_map[row_offset + inode] * s[inode]; // [T]
            bz += bz_map[row_offset + inode] * s[inode]; // [T]
        }
        out.0[iobs] = bx; // [T]
        out.1[iobs] = by; // [T]
        out.2[iobs] = bz; // [T]
    }

    Ok(())
}

/// Flux density contribution from a triangle mesh with nodal stream-function values.
/// Each source triangle is evaluated analytically off its surface and contributes
/// exactly zero when an observation lies on that finite triangle.
///
/// Args:
///     obs: Observation point component slices `(x, y, z)` (m).
///     mesh: Borrowed triangle-mesh geometry view.
///     s: Nodal current-potential values (A).
///     out: Output buffers for magnetic flux density `(bx, by, bz)` (T).
///
/// Returns:
///     `Ok(())` after writing the flux density to `out`, or an error if the mesh
///     geometry or slice dimensions are inconsistent.
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for each exact triangle interaction.
#[inline]
pub fn flux_density_triangle_mesh(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    flux_density_triangle_mesh_inner(obs, mesh, s, out)
}

/// Flux density contribution from a triangle mesh with nodal stream-function values.
/// This variant is parallelized over chunks of observation points.
///
/// Args:
///     obs: Observation point component slices `(x, y, z)` (m).
///     mesh: Borrowed triangle-mesh geometry view.
///     s: Nodal current-potential values (A).
///     out: Output buffers for magnetic flux density `(bx, by, bz)` (T).
///
/// Returns:
///     `Ok(())` after writing the flux density to `out`, or an error if the mesh
///     geometry or slice dimensions are inconsistent.
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for each exact triangle interaction.
#[inline]
pub fn flux_density_triangle_mesh_par(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    mesh.validate_nodal_values(s)?;
    let n = chunksize(obs.0.len());
    let (xpc, ypc, zpc) = par_chunks_3tup!(obs, n);
    let (bxc, byc, bzc) = mut_par_chunks_3tup!(out, n);

    (bxc, byc, bzc, xpc, ypc, zpc)
        .into_par_iter()
        .try_for_each(|(bx, by, bz, xp, yp, zp)| {
            flux_density_triangle_mesh_inner((xp, yp, zp), mesh, s, (bx, by, bz))
        })?;

    Ok(())
}
