use super::{QuadratureKind, map_tri_uv, triangle_basis_current_density, triangle_quadrature_points};
use crate::MU0_OVER_4PI;
use crate::math::{cross3, rss3};

/// Magnetic flux density (B-field) contribution of a given triangle's basis function
/// with unit weighting to a given observation point.
///
/// Assumes a basis function living on the triangle's first node.
#[inline]
pub fn triangle_flux_density_basis(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    obs: [f64; 3],
    quad_kind: QuadratureKind,
) -> [f64; 3] {
    let (tri_area, jref) = triangle_basis_current_density(n0, n1, n2);
    let quad_points = triangle_quadrature_points(quad_kind);

    let mut b = [0.0; 3];

    for qp in quad_points {
        let (c, u, v) = (qp[0], qp[1], qp[2]);
        let src = map_tri_uv(n0, n1, n2, [u, v]);
        let r = [obs[0] - src[0], obs[1] - src[1], obs[2] - src[2]];
        let d = rss3(r[0], r[1], r[2]);
        let inv_d3 = 1.0 / (d * d * d);
        let j_cross_r = cross3(jref[0], jref[1], jref[2], r[0], r[1], r[2]);

        b[0] += c * j_cross_r.0 * inv_d3 * tri_area;
        b[1] += c * j_cross_r.1 * inv_d3 * tri_area;
        b[2] += c * j_cross_r.2 * inv_d3 * tri_area;
    }

    b
}

/// Flux density (B-field) of triangular surface current density distribution
/// at a target point due to scalar current density potential `s`
/// at each node.
///
/// For physical intuition, the current density is related to the difference
/// in potential between the nodes; for example, in a strip discretized into triangles
/// with s=s0 on one side of the strip and s=-s0 on the other side of the strip,
/// the total current on the strip (and its effective filament current) is equal to s0.
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
        (s[0] * b_n0[0] + s[1] * b_n1[0] + s[2] * b_n2[0]) * MU0_OVER_4PI,
        (s[0] * b_n0[1] + s[1] * b_n1[1] + s[2] * b_n2[1]) * MU0_OVER_4PI,
        (s[0] * b_n0[2] + s[1] * b_n1[2] + s[2] * b_n2[2]) * MU0_OVER_4PI,
    ]
}
