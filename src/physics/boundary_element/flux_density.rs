use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

use super::{
    QuadratureKind, TRIANGLE_NEAR_SUBDIVISION_DISTANCE_FACTOR, calc_tri_area, map_tri_uv,
    triangle_basis_current_density, triangle_quadrature_points,
};
use crate::MU0_OVER_4PI;
use crate::chunksize;
use crate::macros::{check_length_3tup, mut_par_chunks_3tup, par_chunks_3tup};
use crate::math::{add_scaled3, cross3, norm3, sub3};
use crate::mesh::TriangleMeshView;
use crate::mesh::elements::tri::tri3::{
    closest_point as triangle_closest_point,
    max_edge_length_squared as triangle_max_edge_length_squared,
    subdivide_about_point as triangle_subdivide_about_point,
};
use crate::physics::hierarchical::DualTreeScalar;
use crate::physics::point_source::current_element::flux_density_current_element_scalar;

/// Midpoint-rule samples for the Duffy-style transverse edge integral in
/// near-surface triangle B-field evaluations.
const TRIANGLE_B_DUFFY_EDGE_SAMPLES: usize = 32;

/// Observation offsets below this fraction of the maximum edge length are
/// treated as exactly on the source surface and evaluated as a principal value.
const TRIANGLE_B_DUFFY_SURFACE_TOL_FACTOR: f64 = 1e-12;

#[inline]
fn triangle_flux_density_inner<T: DualTreeScalar>(
    n0: [T; 3],
    n1: [T; 3],
    n2: [T; 3],
    current_density: [T; 3],
    obs: [T; 3],
    quad_kind: QuadratureKind,
) -> [T; 3] {
    let tri_area = calc_tri_area(n0, n1, n2); // [m^2]
    let quad_points = triangle_quadrature_points(quad_kind);

    let mut b = [T::ZERO; 3]; // [T/A]

    for qp in quad_points {
        let (c, u, v) = (T::from_f64(qp[0]), T::from_f64(qp[1]), T::from_f64(qp[2]));
        let src = map_tri_uv(n0, n1, n2, [u, v]); // [m]
        let moment = [
            current_density[0] * c * tri_area, // [m]
            current_density[1] * c * tri_area, // [m]
            current_density[2] * c * tri_area, // [m]
        ];
        let contrib = flux_density_current_element_scalar(src, moment, obs); // [T/A]
        b[0] = b[0] + contrib[0]; // [T/A]
        b[1] = b[1] + contrib[1]; // [T/A]
        b[2] = b[2] + contrib[2]; // [T/A]
    }

    b
}

#[inline]
fn accum_cross_scaled<T: DualTreeScalar>(out: &mut [T; 3], k: [T; 3], r: [T; 3], scale: T) {
    let k_cross_r = cross3(k, r);
    out[0] = out[0] + scale * k_cross_r[0];
    out[1] = out[1] + scale * k_cross_r[1];
    out[2] = out[2] + scale * k_cross_r[2];
}

#[inline]
fn triangle_flux_density_surface_duffy<T: DualTreeScalar>(
    n0: [T; 3],
    n1: [T; 3],
    n2: [T; 3],
    current_density: [T; 3],
    closest: [T; 3],
    min_sub_area: T,
    length_ref: T,
) -> [T; 3] {
    let mut b = [T::ZERO; 3]; // [T/A]

    for tri in triangle_subdivide_about_point(closest, n0, n1, n2) {
        let [p, va, vb] = tri;
        let area_sub = calc_tri_area(p, va, vb); // [m^2]
        if area_sub <= min_sub_area {
            continue;
        }

        let qa = sub3(va, closest); // [m]
        let qb = sub3(vb, closest); // [m]
        let dq = sub3(qb, qa); // [m]
        let mut sub_b = [T::ZERO; 3]; // [1/m]

        for i in 0..TRIANGLE_B_DUFFY_EDGE_SAMPLES {
            let eta = T::from_f64((i as f64 + 0.5) / TRIANGLE_B_DUFFY_EDGE_SAMPLES as f64);
            let q = add_scaled3(qa, dq, eta); // [m]
            let qnorm = norm3(q); // [m]
            if qnorm == T::ZERO {
                continue;
            }
            let scale = (qnorm / length_ref).ln() / qnorm.powf(3.0); // [1/m^3]
            accum_cross_scaled(&mut sub_b, current_density, q, scale); // [1/m^2]
        }

        let scale =
            T::from_f64(-MU0_OVER_4PI * 2.0 / TRIANGLE_B_DUFFY_EDGE_SAMPLES as f64) * area_sub; // [H/m * m^2]
        b[0] = b[0] + scale * sub_b[0]; // [T/A]
        b[1] = b[1] + scale * sub_b[1]; // [T/A]
        b[2] = b[2] + scale * sub_b[2]; // [T/A]
    }

    b
}

#[inline]
fn triangle_flux_density_duffy<T: DualTreeScalar>(
    n0: [T; 3],
    n1: [T; 3],
    n2: [T; 3],
    current_density: [T; 3],
    obs: [T; 3],
    closest: [T; 3],
    max_edge_sq: T,
) -> Option<[T; 3]> {
    let h = sub3(obs, closest); // [m]
    let surface_tol = T::from_f64(TRIANGLE_B_DUFFY_SURFACE_TOL_FACTOR);
    let surface_tol_sq = surface_tol * surface_tol * max_edge_sq; // [m^2]
    let min_sub_area = max_edge_sq * T::from_f64(1e-14); // [m^2]
    // The finite-part log needs a dimensionless argument. The reference length
    // is immaterial because the omitted log-divergent term cancels by angular
    // symmetry across the subtriangles around the singular point.
    let length_ref = max_edge_sq.sqrt(); // [m]

    if h[0].mul_add(h[0], h[1].mul_add(h[1], h[2] * h[2])) <= surface_tol_sq {
        Some(triangle_flux_density_surface_duffy(
            n0,
            n1,
            n2,
            current_density,
            closest,
            min_sub_area,
            length_ref,
        ))
    } else {
        None
    }
}

/// Magnetic flux density (B-field) contribution of a given triangle's basis function
/// with unit weighting to a given observation point.
///
/// Assumes a basis function living on the triangle's first node.
///
/// Method:
/// - The linear triangle basis induces a constant surface current density over the
///   element.
/// - Each quadrature point is treated as a point current element with moment
///   `m = K * ΔS_q`.
/// - When the target point is exactly on the triangle, a Duffy-style transform
///   centered at the closest point is used for the principal-value sheet
///   contribution without adding either one-sided `±mu0 / 2 K x n` jump term.
/// - Other near-field points use one level of closest-point subdivision before
///   applying the same quadrature rule on each subtriangle.
/// - Sum the Biot-Savart contributions with the full `μ0 / 4π` prefactor included.
///
/// Args:
///     n0: Basis node coordinates `[x, y, z]` (m).
///     n1: Triangle vertex 1 coordinates `[x, y, z]` (m).
///     n2: Triangle vertex 2 coordinates `[x, y, z]` (m).
///     obs: Observation point `[x, y, z]` (m).
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///
/// Returns:
///     Basis-function magnetic flux density `[bx, by, bz]` (T/A).
#[inline]
pub fn triangle_flux_density_basis<T: DualTreeScalar>(
    n0: [T; 3],
    n1: [T; 3],
    n2: [T; 3],
    obs: [T; 3],
    quad_kind: QuadratureKind,
) -> [T; 3] {
    let (_, jref) = triangle_basis_current_density(n0, n1, n2); // [m^2], [1/m]
    let max_edge_sq = triangle_max_edge_length_squared(n0, n1, n2); // [m^2]
    let closest = triangle_closest_point(obs, n0, n1, n2); // [m]
    let dx = obs[0] - closest[0]; // [m]
    let dy = obs[1] - closest[1]; // [m]
    let dz = obs[2] - closest[2]; // [m]
    let dist_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz)); // [m^2]
    let subdiv_factor = T::from_f64(TRIANGLE_NEAR_SUBDIVISION_DISTANCE_FACTOR);
    let subdiv_threshold_sq = subdiv_factor * subdiv_factor * max_edge_sq; // [m^2]

    if dist_sq > subdiv_threshold_sq {
        return triangle_flux_density_inner(n0, n1, n2, jref, obs, quad_kind);
    }

    if let Some(b) = triangle_flux_density_duffy(n0, n1, n2, jref, obs, closest, max_edge_sq) {
        return b;
    }

    // Single-level triangle subdivision for finite-offset near-field calcs
    // to keep quadrature points separated from the target point.
    let mut b = [T::ZERO; 3]; // [T/A]
    let min_sub_area = max_edge_sq * T::from_f64(1e-14); // [m^2]
    for tri in triangle_subdivide_about_point(closest, n0, n1, n2) {
        let [a, b0, c] = tri;
        if calc_tri_area(a, b0, c) <= min_sub_area {
            continue;
        }
        let contrib = triangle_flux_density_inner(a, b0, c, jref, obs, quad_kind);
        b[0] = b[0] + contrib[0]; // [T/A]
        b[1] = b[1] + contrib[1]; // [T/A]
        b[2] = b[2] + contrib[2]; // [T/A]
    }

    b
}

/// Flux density (B-field) of triangular surface current density distribution
/// at a target point due to scalar current density potential `s`
/// at each node.
///
/// For physical intuition, the current density is related to the gradient
/// in potential between the nodes; for example, in a strip discretized into triangles
/// with s=s0 on one side of the strip and s=-s0 on the other side of the strip,
/// the total current on the strip (and its effective filament current) is equal to s0.
///
/// Args:
///     n0: Triangle vertex 0 coordinates `[x, y, z]` (m).
///     n1: Triangle vertex 1 coordinates `[x, y, z]` (m).
///     n2: Triangle vertex 2 coordinates `[x, y, z]` (m).
///     s: Nodal current-potential values `[s0, s1, s2]` (A).
///     obs: Observation point `[x, y, z]` (m).
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///
/// Returns:
///     Magnetic flux density `[bx, by, bz]` (T).
#[inline]
pub fn flux_density_triangle<T: DualTreeScalar>(
    n0: [T; 3],
    n1: [T; 3],
    n2: [T; 3],
    s: [T; 3],
    obs: [T; 3],
    quad_kind: QuadratureKind,
) -> [T; 3] {
    let b_n0 = triangle_flux_density_basis(n0, n1, n2, obs, quad_kind);
    let b_n1 = triangle_flux_density_basis(n1, n2, n0, obs, quad_kind);
    let b_n2 = triangle_flux_density_basis(n2, n0, n1, obs, quad_kind);

    [
        s[0] * b_n0[0] + s[1] * b_n1[0] + s[2] * b_n2[0], // [T]
        s[0] * b_n0[1] + s[1] * b_n1[1] + s[2] * b_n2[1], // [T]
        s[0] * b_n0[2] + s[1] * b_n1[2] + s[2] * b_n2[2], // [T]
    ]
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
    quad_kind: QuadratureKind,
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

            let b0 = triangle_flux_density_basis(
                tri_nodes[0],
                tri_nodes[1],
                tri_nodes[2],
                obs_i,
                quad_kind,
            ); // [T/A]
            let b1 = triangle_flux_density_basis(
                tri_nodes[1],
                tri_nodes[2],
                tri_nodes[0],
                obs_i,
                quad_kind,
            ); // [T/A]
            let b2 = triangle_flux_density_basis(
                tri_nodes[2],
                tri_nodes[0],
                tri_nodes[1],
                obs_i,
                quad_kind,
            ); // [T/A]

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
    quad_kind: QuadratureKind,
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
            let contrib = flux_density_triangle(
                tri_nodes[0],
                tri_nodes[1],
                tri_nodes[2],
                tri_s,
                obs_i,
                quad_kind,
            );
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
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///     out: Output mapping buffers `(bx_map, by_map, bz_map)` (T/A), each row-major in
///         `(observation point, source node)` order.
///
/// Returns:
///     `Ok(())` after writing the dense mapping to `out`, or an error if the mesh
///     geometry or slice dimensions are inconsistent.
#[inline]
pub fn flux_density_triangle_mesh_mapping(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    flux_density_triangle_mesh_mapping_chunk(obs, mesh, quad_kind, out)
}

/// Parallel variant of [`flux_density_triangle_mesh_mapping`].
///
/// Args:
///     obs: Observation point component slices `(x, y, z)` (m).
///     mesh: Borrowed triangle-mesh geometry view.
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///     out: Output mapping buffers `(bx_map, by_map, bz_map)` (T/A), each row-major in
///         `(observation point, source node)` order.
///
/// Returns:
///     `Ok(())` after writing the dense mapping to `out`, or an error if the mesh
///     geometry or slice dimensions are inconsistent.
#[inline]
pub fn flux_density_triangle_mesh_mapping_par(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let nobs = obs.0.len(); // [-]
    check_length_3tup!(nobs, obs);
    validate_flux_density_mapping_inputs(out.0, out.1, out.2, nobs, mesh.nnode())?;

    if nobs == 0 || mesh.nnode() == 0 {
        return flux_density_triangle_mesh_mapping_chunk(obs, mesh, quad_kind, out);
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
            flux_density_triangle_mesh_mapping_chunk((xp, yp, zp), mesh, quad_kind, (bx, by, bz))
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
///
/// Args:
///     obs: Observation point component slices `(x, y, z)` (m).
///     mesh: Borrowed triangle-mesh geometry view.
///     s: Nodal current-potential values (A).
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///     out: Output buffers for magnetic flux density `(bx, by, bz)` (T).
///
/// Returns:
///     `Ok(())` after writing the flux density to `out`, or an error if the mesh
///     geometry or slice dimensions are inconsistent.
#[inline]
pub fn flux_density_triangle_mesh(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    flux_density_triangle_mesh_inner(obs, mesh, s, quad_kind, out)
}

/// Flux density contribution from a triangle mesh with nodal stream-function values.
/// This variant is parallelized over chunks of observation points.
///
/// Args:
///     obs: Observation point component slices `(x, y, z)` (m).
///     mesh: Borrowed triangle-mesh geometry view.
///     s: Nodal current-potential values (A).
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///     out: Output buffers for magnetic flux density `(bx, by, bz)` (T).
///
/// Returns:
///     `Ok(())` after writing the flux density to `out`, or an error if the mesh
///     geometry or slice dimensions are inconsistent.
#[inline]
pub fn flux_density_triangle_mesh_par(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    mesh.validate_nodal_values(s)?;
    let n = chunksize(obs.0.len());
    let (xpc, ypc, zpc) = par_chunks_3tup!(obs, n);
    let (bxc, byc, bzc) = mut_par_chunks_3tup!(out, n);

    (bxc, byc, bzc, xpc, ypc, zpc)
        .into_par_iter()
        .try_for_each(|(bx, by, bz, xp, yp, zp)| {
            flux_density_triangle_mesh_inner((xp, yp, zp), mesh, s, quad_kind, (bx, by, bz))
        })?;

    Ok(())
}
