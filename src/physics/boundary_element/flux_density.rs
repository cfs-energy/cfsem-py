use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

use super::{
    QuadratureKind, map_tri_uv, triangle_basis_current_density, triangle_quadrature_points,
};
use crate::chunksize;
use crate::macros::{check_length_3tup, mut_par_chunks_3tup, par_chunks_3tup};
use crate::mesh::TriangleMeshView;
use crate::physics::point_source::current_element::flux_density_current_element_scalar;

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
///     Basis-function magnetic flux density `[bx, by, bz]` (T).
#[inline]
pub fn triangle_flux_density_basis(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    obs: [f64; 3],
    quad_kind: QuadratureKind,
) -> [f64; 3] {
    let (tri_area, jref) = triangle_basis_current_density(n0, n1, n2); // [m^2], [1/m]
    let quad_points = triangle_quadrature_points(quad_kind);

    let mut b = [0.0; 3]; // [T/A]

    for qp in quad_points {
        let (c, u, v) = (qp[0], qp[1], qp[2]);
        let src = map_tri_uv(n0, n1, n2, [u, v]); // [m]
        let moment = [
            jref[0] * c * tri_area, // [m]
            jref[1] * c * tri_area, // [m]
            jref[2] * c * tri_area, // [m]
        ];
        let contrib = flux_density_current_element_scalar(src, moment, obs); // [T/A]
        b[0] += contrib[0]; // [T/A]
        b[1] += contrib[1]; // [T/A]
        b[2] += contrib[2]; // [T/A]
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
pub fn flux_density_triangle(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    s: [f64; 3],
    obs: [f64; 3],
    quad_kind: QuadratureKind,
) -> [f64; 3] {
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
fn flux_density_triangle_mesh_inner(
    obs: (&[f64], &[f64], &[f64]),
    mesh: TriangleMeshView<'_>,
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let nobs = obs.0.len();
    check_length_3tup!(nobs, obs);
    check_length_3tup!(nobs, out);

    out.0.fill(0.0); // [T]
    out.1.fill(0.0); // [T]
    out.2.fill(0.0); // [T]

    for i in 0..nobs {
        let obs_i = [obs.0[i], obs.1[i], obs.2[i]];
        for j in 0..mesh.len() {
            let (tri_nodes, tri_s) = mesh.triangle_nodes(j);
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

/// Flux density contribution from a triangle mesh with nodal stream-function values.
///
/// Args:
///     obs: Observation point component slices `(x, y, z)` (m).
///     nodes: Mesh node-coordinate component slices `(x, y, z)` (m).
///     triangles: Triangle-node index component slices `(i0, i1, i2)` (dimensionless).
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
    nodes: (&[f64], &[f64], &[f64]),
    triangles: (&[usize], &[usize], &[usize]),
    s: &[f64],
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let mesh = TriangleMeshView::new(nodes, triangles, s)?;
    flux_density_triangle_mesh_inner(obs, mesh, quad_kind, out)
}

/// Flux density contribution from a triangle mesh with nodal stream-function values.
/// This variant is parallelized over chunks of observation points.
///
/// Args:
///     obs: Observation point component slices `(x, y, z)` (m).
///     nodes: Mesh node-coordinate component slices `(x, y, z)` (m).
///     triangles: Triangle-node index component slices `(i0, i1, i2)` (dimensionless).
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
    nodes: (&[f64], &[f64], &[f64]),
    triangles: (&[usize], &[usize], &[usize]),
    s: &[f64],
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let mesh = TriangleMeshView::new(nodes, triangles, s)?;
    let n = chunksize(obs.0.len());
    let (xpc, ypc, zpc) = par_chunks_3tup!(obs, n);
    let (bxc, byc, bzc) = mut_par_chunks_3tup!(out, n);

    (bxc, byc, bzc, xpc, ypc, zpc)
        .into_par_iter()
        .try_for_each(|(bx, by, bz, xp, yp, zp)| {
            flux_density_triangle_mesh_inner((xp, yp, zp), mesh, quad_kind, (bx, by, bz))
        })?;

    Ok(())
}
