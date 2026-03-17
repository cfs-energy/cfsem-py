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
use crate::physics::point_source::current_element::vector_potential_current_element_scalar;

/// Magnetic vector potential (A-field) contribution of a given triangle's basis
/// function with unit weighting to a given observation point.
///
/// Assumes a basis function living on the triangle's first node.
///
/// Method:
/// - The linear triangle basis induces a constant surface current density over the
///   element.
/// - Each quadrature point is treated as a point current element with moment
///   `m = K * ΔS_q`.
/// - Sum the weighted contributions with the full `μ0 / 4π` prefactor included.
///
/// References:
/// - [5], Eq. (3.24) for the stream-function surface current construction,
///   Eq. (4.6) for the constant current density on a linear triangle, and
///   Eqs. (5.3)-(5.5) for triangle vector-potential integrals.
/// - [3] for `1 / R` potential integrals on polygonal and polyhedral elements.
/// - [2] for numerical treatment of triangle `1 / R` and `∇(1 / R)`
///   integrals with linear shape functions.
#[inline]
pub fn triangle_vector_potential_basis(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    obs: [f64; 3],
    quad_kind: QuadratureKind,
) -> [f64; 3] {
    let (tri_area, jref) = triangle_basis_current_density(n0, n1, n2);
    let quad_points = triangle_quadrature_points(quad_kind);

    let mut a = [0.0; 3];

    for qp in quad_points {
        let (c, u, v) = (qp[0], qp[1], qp[2]);
        let src = map_tri_uv(n0, n1, n2, [u, v]);
        let moment = [
            jref[0] * c * tri_area,
            jref[1] * c * tri_area,
            jref[2] * c * tri_area,
        ];
        let contrib = vector_potential_current_element_scalar(src, moment, obs);
        a[0] += contrib[0];
        a[1] += contrib[1];
        a[2] += contrib[2];
    }

    a
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
/// Method:
/// - Evaluate the three nodal basis-function vector potentials.
/// - Weight them by the nodal scalar potential values.
///
/// References:
/// - [5], Eq. (3.24), Eq. (4.6), and Eqs. (5.3)-(5.5).
/// - [3], pp. 276-281.
/// - [2], pp. 1448-1455.
#[inline]
pub fn vector_potential_triangle(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    s: [f64; 3],
    obs: [f64; 3],
    quad_kind: QuadratureKind,
) -> [f64; 3] {
    let a_n0 = triangle_vector_potential_basis(n0, n1, n2, obs, quad_kind);
    let a_n1 = triangle_vector_potential_basis(n1, n2, n0, obs, quad_kind);
    let a_n2 = triangle_vector_potential_basis(n2, n0, n1, obs, quad_kind);

    [
        s[0] * a_n0[0] + s[1] * a_n1[0] + s[2] * a_n2[0],
        s[0] * a_n0[1] + s[1] * a_n1[1] + s[2] * a_n2[1],
        s[0] * a_n0[2] + s[1] * a_n1[2] + s[2] * a_n2[2],
    ]
}

#[inline]
fn vector_potential_triangle_mesh_inner(
    obs: (&[f64], &[f64], &[f64]),
    mesh: TriangleMeshView<'_>,
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let nobs = obs.0.len();
    check_length_3tup!(nobs, obs);
    check_length_3tup!(nobs, out);

    out.0.fill(0.0);
    out.1.fill(0.0);
    out.2.fill(0.0);

    for i in 0..nobs {
        let obs_i = [obs.0[i], obs.1[i], obs.2[i]];
        for j in 0..mesh.len() {
            let (tri_nodes, tri_s) = mesh.triangle_nodes(j);
            let contrib = vector_potential_triangle(
                tri_nodes[0],
                tri_nodes[1],
                tri_nodes[2],
                tri_s,
                obs_i,
                quad_kind,
            );
            out.0[i] += contrib[0];
            out.1[i] += contrib[1];
            out.2[i] += contrib[2];
        }
    }

    Ok(())
}

/// Vector potential contribution from a triangle mesh with nodal stream-function values.
#[inline]
pub fn vector_potential_triangle_mesh(
    obs: (&[f64], &[f64], &[f64]),
    nodes: (&[f64], &[f64], &[f64]),
    triangles: (&[usize], &[usize], &[usize]),
    s: &[f64],
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let mesh = TriangleMeshView::new(nodes, triangles, s)?;
    vector_potential_triangle_mesh_inner(obs, mesh, quad_kind, out)
}

/// Vector potential contribution from a triangle mesh with nodal stream-function values.
/// This variant is parallelized over chunks of observation points.
#[inline]
pub fn vector_potential_triangle_mesh_par(
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
    let (axc, ayc, azc) = mut_par_chunks_3tup!(out, n);

    (axc, ayc, azc, xpc, ypc, zpc)
        .into_par_iter()
        .try_for_each(|(ax, ay, az, xp, yp, zp)| {
            vector_potential_triangle_mesh_inner((xp, yp, zp), mesh, quad_kind, (ax, ay, az))
        })?;

    Ok(())
}
