use super::{QuadratureKind, map_tri_uv, triangle_basis_current_density, triangle_quadrature_points};
use crate::MU0_OVER_4PI;
use crate::math::rss3;

/// Magnetic vector potential (A-field) contribution of a given triangle's basis
/// function with unit weighting to a given observation point.
///
/// Assumes a basis function living on the triangle's first node.
///
/// Method:
/// - The linear triangle basis induces a constant surface current density over the
///   element.
/// - Evaluate the Coulomb-gauge kernel `K / R` at triangle quadrature points.
/// - Sum the weighted contributions without the `μ0 / 4π` prefactor; that prefactor is
///   applied when basis functions are combined into a physical field.
///
/// References:
/// - [5], Eq. (3.24) on p. 70 for the stream-function surface current construction,
///   Eq. (4.6) on p. 93 for the constant current density on a linear triangle, and
///   Eqs. (5.3)-(5.5) on pp. 107-108 for triangle vector-potential integrals.
/// - [3], pp. 276-281, for classic `1 / R` potential integrals on polygonal and
///   polyhedral elements.
/// - [2], pp. 1448-1455, for numerical treatment of triangle `1 / R` and `∇(1 / R)`
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
        let r = rss3(obs[0] - src[0], obs[1] - src[1], obs[2] - src[2]);
        let weight = c * tri_area / r;

        a[0] += weight * jref[0];
        a[1] += weight * jref[1];
        a[2] += weight * jref[2];
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
/// - Apply the final `μ0 / 4π` prefactor to obtain the physical vector potential.
///
/// References:
/// - [5], Eq. (3.24) on p. 70, Eq. (4.6) on p. 93, and Eqs. (5.3)-(5.5) on pp. 107-108.
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
        (s[0] * a_n0[0] + s[1] * a_n1[0] + s[2] * a_n2[0]) * MU0_OVER_4PI,
        (s[0] * a_n0[1] + s[1] * a_n1[1] + s[2] * a_n2[1]) * MU0_OVER_4PI,
        (s[0] * a_n0[2] + s[1] * a_n1[2] + s[2] * a_n2[2]) * MU0_OVER_4PI,
    ]
}
