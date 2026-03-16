//! Boundary-element fields and formulas.
//!
//! References:
//! * \[1\] S. E. Mousavi and N. Sukumar, “Generalized Duffy transformation for integrating vertex singularities,” Comput Mech, vol. 45, no. 2–3, pp. 127–140, Jan. 2010, doi: 10.1007/s00466-009-0424-1.
//! * \[2\] R. D. Graglia, “On the numerical integration of the linear shape functions times the 3-D Green’s function or its gradient on a plane triangle,” IEEE Transactions on Antennas and Propagation, vol. 41, no. 10, pp. 1448–1455, Oct. 1993, doi: 10.1109/8.247786.
//! * \[3\] D. Wilton, S. Rao, A. Glisson, D. Schaubert, O. Al-Bundak, and C. Butler, “Potential integrals for uniform and linear source distributions on polygonal and polyhedral domains,” IEEE Transactions on Antennas and Propagation, vol. 32, no. 3, pp. 276–281, Mar. 1984, doi: 10.1109/TAP.1984.1143304.
//! * \[4\] M. G. Duffy, “Quadrature Over a Pyramid or Cube of Integrands with a Singularity at a Vertex,” SIAM Journal on Numerical Analysis, vol. 19, no. 6, pp. 1260–1262, 1982.
//! * \[5\] G. N. Peeren, “Stream function approach for determining optimal surface currents,” Phd Thesis 2 (Research NOT TU/e / Graduation TU/e), Technische Universiteit Eindhoven, Eindhoven, 2003. doi: 10.6100/IR570424.
//! * \[6\] F. Hussain, M. S. Karim, and R. Ahamad, “Appropriate Gaussian quadrature formulae for triangles”.

use crate::MU0_OVER_4PI;
use crate::math::{cross3, dot3, rss3};

/// Second-order quadrature integration weights on a triangular surface
/// Format is [Weights, U, V]
///
/// References:
/// * F. Hussain, M. S. Karim, and R. Ahamad, “Appropriate Gaussian quadrature formulae for triangles”.
const TABLE_GAUSS_LEGENDRE_2: [[f64; 3]; 4] = [
    [0.5283121635e-01, 0.1666666667e+00, 0.7886751346e+00],
    [0.1971687836e+00, 0.6220084679e+00, 0.2113248654e+00],
    [0.5283121635e-01, 0.4465819874e-01, 0.7886751346e+00],
    [0.1971687836e+00, 0.1666666667e+00, 0.2113248654e+00],
];

/// Third-order quadrature integration weights on a triangular surface
/// Format is [Weights, U, V]
///
/// References:
/// * F. Hussain, M. S. Karim, and R. Ahamad, “Appropriate Gaussian quadrature formulae for triangles”.
const TABLE_GAUSS_LEGENDRE_3: [[f64; 3]; 9] = [
    [0.9876542474e-01, 0.2500000000e+00, 0.5000000000e+00],
    [0.1391378575e-01, 0.5635083269e-01, 0.8872983346e+00],
    [0.1095430035e+00, 0.4436491673e+00, 0.1127016654e+00],
    [0.6172839460e-01, 0.4436491673e+00, 0.5000000000e+00],
    [0.8696116674e-02, 0.1000000000e+00, 0.8872983346e+00],
    [0.6846438175e-01, 0.7872983346e+00, 0.1127016654e+00],
    [0.6172839460e-01, 0.5635083269e-01, 0.5000000000e+00],
    [0.8696116674e-02, 0.1270166538e-01, 0.8872983346e+00],
    [0.6846438175e-01, 0.1000000000e+00, 0.1127016654e+00],
];

#[derive(Clone, Copy)]
pub enum QuadratureKind {
    GaussLegendre,
    Dunavant,
}

/// Isoparametric mapping of a point on a 3D triangle
/// from U-V coordinates on the triangle's surface.
#[inline]
pub fn map_tri_uv(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3], pin_uv: [f64; 2]) -> [f64; 3] {
    // Barycentric interpolation
    let mut pout = [0.0; 3];

    // Precalulate third uv component
    let w = 1.0 - pin_uv[0] - pin_uv[1];
    pout[0] = n0[0] * w + n1[0] * pin_uv[0] + n2[0] * pin_uv[1];
    pout[1] = n0[1] * w + n1[1] * pin_uv[0] + n2[1] * pin_uv[1];
    pout[2] = n0[2] * w + n1[2] * pin_uv[0] + n2[2] * pin_uv[1];

    return pout;
}

/// Area of a 3D triangle.
#[inline]
pub fn calc_tri_area(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> f64 {
    // Get two directional vectors from node 0 to nodes 1 and 2
    let v01 = [n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]];
    let v02 = [n2[0] - n0[0], n2[1] - n0[1], n2[2] - n0[2]];

    // Area of parallelgram is norm of crossproduct of the vectors.
    let cross = cross3(v01[0], v01[1], v01[2], v02[0], v02[1], v02[2]);
    let area = 0.5 * rss3(cross.0, cross.1, cross.2);
    return area;
}

/// Normal vector of a triangle.
///
/// Direction is non-unique; the order of the points determines whether
/// the returned normal points "up" or "down" relative to the triangle.
#[inline]
pub fn calc_tri_normal(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> [f64; 3] {
    // Get two directional vectors from node 0 to nodes 1 and 2
    let v01 = [n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]];
    let v02 = [n2[0] - n0[0], n2[1] - n0[1], n2[2] - n0[2]];

    // Get cross product for the two directional vectors
    let cross = cross3(v01[0], v01[1], v01[2], v02[0], v02[1], v02[2]);

    // Normalize the normal vector
    let norm = rss3(cross.0, cross.1, cross.2);
    let out = [cross.0 / norm, cross.1 / norm, cross.2 / norm];

    return out;
}

#[inline]
fn triangle_basis_current_density(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> (f64, [f64; 3]) {
    // Calculate directional vectors between nodes
    let v01 = [n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]];
    let v02 = [n2[0] - n0[0], n2[1] - n0[1], n2[2] - n0[2]];

    // Triangle area and basis current density vector
    let tri_area = calc_tri_area(n0, n1, n2);
    let jref = [
        (v02[0] - v01[0]) / (2.0 * tri_area),
        (v02[1] - v01[1]) / (2.0 * tri_area),
        (v02[2] - v01[2]) / (2.0 * tri_area),
    ];

    (tri_area, jref)
}

#[inline]
fn triangle_quadrature_points(quad_kind: QuadratureKind, quad_order: usize) -> &'static [[f64; 3]] {
    match (quad_kind, quad_order) {
        (QuadratureKind::GaussLegendre, 2) => &TABLE_GAUSS_LEGENDRE_2,
        (QuadratureKind::GaussLegendre, 3) => &TABLE_GAUSS_LEGENDRE_3,
        _ => panic!(),
    }
}

/// Magnetic flux density (B-field) contribution of a given triangle's basis function
/// with unit weighting to a given obseration point.
///
/// Assumes a basis function living on the triangle's first node.
#[inline]
pub fn triangle_flux_density_basis(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    obs: [f64; 3],
    quad_kind: QuadratureKind,
    quad_order: usize,
) -> [f64; 3] {
    let (tri_area, jref) = triangle_basis_current_density(n0, n1, n2);
    let quad_points = triangle_quadrature_points(quad_kind, quad_order);

    let mut b = [0.0; 3];

    for i in 0..quad_points.len() {
        // Unpack current quad point
        let (c, u, v) = (quad_points[i][0], quad_points[i][1], quad_points[i][2]);

        // Transform current quad points for the given triangle
        let qp = map_tri_uv(n0, n1, n2, [u, v]);

        // Biot-Savart kernel for a constant surface current density over the triangle
        let r = [obs[0] - qp[0], obs[1] - qp[1], obs[2] - qp[2]];
        let d = rss3(r[0], r[1], r[2]);
        let inv_d3 = 1.0 / (d * d * d);
        let j_cross_r = cross3(jref[0], jref[1], jref[2], r[0], r[1], r[2]);

        // Add B-field contribution for current quadrature point
        b[0] += c * j_cross_r.0 * inv_d3 * tri_area;
        b[1] += c * j_cross_r.1 * inv_d3 * tri_area;
        b[2] += c * j_cross_r.2 * inv_d3 * tri_area;
    }

    return b;
}

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
    quad_order: usize,
) -> [f64; 3] {
    let (tri_area, jref) = triangle_basis_current_density(n0, n1, n2);
    let quad_points = triangle_quadrature_points(quad_kind, quad_order);

    let mut a = [0.0; 3];

    for i in 0..quad_points.len() {
        // Unpack current quad point
        let (c, u, v) = (quad_points[i][0], quad_points[i][1], quad_points[i][2]);

        // Transform current quad points for the given triangle
        let qp = map_tri_uv(n0, n1, n2, [u, v]);

        // Coulomb-gauge kernel for a constant surface current density over the triangle
        let r = rss3(obs[0] - qp[0], obs[1] - qp[1], obs[2] - qp[2]);
        let weight = c * tri_area / r;

        // Add A-field contribution for current quadrature point
        a[0] += weight * jref[0];
        a[1] += weight * jref[1];
        a[2] += weight * jref[2];
    }

    a
}

/// Flux density (B-field) of triangular surface current density distribution
/// at a target point due to scalar current density potential `s`
/// at each node.
///
/// For phyisical intuition, the current density is related to the difference
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
    quad_order: usize,
) -> [f64; 3] {
    let mut out = [0.0; 3];

    // Collect B-field contributions for the three basis functions living on n0, n1, and n2
    let b_n0 = triangle_flux_density_basis(n0, n1, n2, obs, quad_kind, quad_order);
    let b_n1 = triangle_flux_density_basis(n1, n2, n0, obs, quad_kind, quad_order);
    let b_n2 = triangle_flux_density_basis(n2, n0, n1, obs, quad_kind, quad_order);

    // Sum contributions by each basis function weighted by the basis function value
    out[0] = (s[0] * b_n0[0] + s[1] * b_n1[0] + s[2] * b_n2[0]) * MU0_OVER_4PI;
    out[1] = (s[0] * b_n0[1] + s[1] * b_n1[1] + s[2] * b_n2[1]) * MU0_OVER_4PI;
    out[2] = (s[0] * b_n0[2] + s[1] * b_n1[2] + s[2] * b_n2[2]) * MU0_OVER_4PI;

    return out;
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
    quad_order: usize,
) -> [f64; 3] {
    let mut out = [0.0; 3];

    // Collect A-field contributions for the three basis functions living on n0, n1, and n2
    let a_n0 = triangle_vector_potential_basis(n0, n1, n2, obs, quad_kind, quad_order);
    let a_n1 = triangle_vector_potential_basis(n1, n2, n0, obs, quad_kind, quad_order);
    let a_n2 = triangle_vector_potential_basis(n2, n0, n1, obs, quad_kind, quad_order);

    // Sum contributions by each basis function weighted by the basis function value
    out[0] = (s[0] * a_n0[0] + s[1] * a_n1[0] + s[2] * a_n2[0]) * MU0_OVER_4PI;
    out[1] = (s[0] * a_n0[1] + s[1] * a_n1[1] + s[2] * a_n2[1]) * MU0_OVER_4PI;
    out[2] = (s[0] * a_n0[2] + s[1] * a_n1[2] + s[2] * a_n2[2]) * MU0_OVER_4PI;

    return out;
}

const TRIANGLE_SCALAR_POTENTIAL_NEAR_FACTOR: f64 = 2.0;
const TRIANGLE_SCALAR_POTENTIAL_MAX_DEPTH: usize = 8;
const TRIANGLE_SELF_DUFFY_SAMPLES: usize = 128;

#[inline]
fn triangle_basis_current_densities(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> [[f64; 3]; 3] {
    [
        triangle_basis_current_density(n0, n1, n2).1,
        triangle_basis_current_density(n1, n2, n0).1,
        triangle_basis_current_density(n2, n0, n1).1,
    ]
}

#[inline]
fn triangle_centroid(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> [f64; 3] {
    [
        (n0[0] + n1[0] + n2[0]) / 3.0,
        (n0[1] + n1[1] + n2[1]) / 3.0,
        (n0[2] + n1[2] + n2[2]) / 3.0,
    ]
}

#[inline]
fn triangle_max_edge_length(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> f64 {
    let d01 = rss3(n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]);
    let d12 = rss3(n2[0] - n1[0], n2[1] - n1[1], n2[2] - n1[2]);
    let d20 = rss3(n0[0] - n2[0], n0[1] - n2[1], n0[2] - n2[2]);
    d01.max(d12).max(d20)
}

#[inline]
fn midpoint(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        0.5 * (a[0] + b[0]),
        0.5 * (a[1] + b[1]),
        0.5 * (a[2] + b[2]),
    ]
}

#[inline]
fn subdivide_triangle(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> [[[f64; 3]; 3]; 4] {
    let m01 = midpoint(n0, n1);
    let m12 = midpoint(n1, n2);
    let m20 = midpoint(n2, n0);

    [
        [n0, m01, m20],
        [m01, n1, m12],
        [m20, m12, n2],
        [m01, m12, m20],
    ]
}

#[inline]
fn points_match(a: [f64; 3], b: [f64; 3]) -> bool {
    rss3(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1e-12
}

#[inline]
fn triangles_identical(
    src0: [f64; 3],
    src1: [f64; 3],
    src2: [f64; 3],
    tgt0: [f64; 3],
    tgt1: [f64; 3],
    tgt2: [f64; 3],
) -> bool {
    let src = [src0, src1, src2];
    let tgt = [tgt0, tgt1, tgt2];

    src.iter()
        .all(|&s| tgt.iter().any(|&t| points_match(s, t)))
}

/// Regular triangle evaluation of the scalar kernel integral `∫ dS / R` using plain
/// quadrature.
///
/// References:
/// - [3], pp. 276-281, for `1 / R` potential integrals on flat polygonal elements.
/// - [2], pp. 1448-1455, for triangle integration of Green-function kernels with linear
///   shape functions.
#[inline]
fn triangle_scalar_potential_regular(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    obs: [f64; 3],
    quad_kind: QuadratureKind,
    quad_order: usize,
) -> f64 {
    let tri_area = calc_tri_area(n0, n1, n2);
    let quad_points = triangle_quadrature_points(quad_kind, quad_order);

    let mut out = 0.0;
    for qp in quad_points {
        let src = map_tri_uv(n0, n1, n2, [qp[1], qp[2]]);
        let dist = rss3(obs[0] - src[0], obs[1] - src[1], obs[2] - src[2]);
        out += qp[0] * tri_area / dist;
    }

    out
}

/// Weakly singular single-triangle `∫ dS / R` evaluation for a target point on the
/// triangle itself.
///
/// Method:
/// - Split the parent triangle into three sub-triangles sharing `obs`.
/// - On each sub-triangle, use a Duffy-style collapse of the radial coordinate so the
///   `1 / R` singularity is canceled by the surface Jacobian.
/// - The remaining 1D integral along the opposite edge is smooth and is evaluated by a
///   midpoint rule.
///
/// References:
/// - [4], pp. 1260-1262, for the original Duffy transform for vertex singularities on
///   simplices.
/// - [1], abstract and Sec. 2, for the generalized Duffy mapping and the note that the
///   standard `1 / r` case corresponds to the classical Duffy choice.
/// - [2], pp. 1448-1455, and [3], pp. 276-281, for related weakly singular triangle
///   Green-function integrals.
#[inline]
fn triangle_scalar_potential_self_duffy(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    obs: [f64; 3],
) -> f64 {
    let edges = [(n0, n1), (n1, n2), (n2, n0)];
    let mut out = 0.0;

    for (va, vb) in edges {
        let area_sub = calc_tri_area(obs, va, vb);
        if area_sub == 0.0 {
            continue;
        }

        let mut line_integral = 0.0;
        for i in 0..TRIANGLE_SELF_DUFFY_SAMPLES {
            let eta = (i as f64 + 0.5) / TRIANGLE_SELF_DUFFY_SAMPLES as f64;
            let edge_vec = [
                (1.0 - eta).mul_add(va[0] - obs[0], eta * (vb[0] - obs[0])),
                (1.0 - eta).mul_add(va[1] - obs[1], eta * (vb[1] - obs[1])),
                (1.0 - eta).mul_add(va[2] - obs[2], eta * (vb[2] - obs[2])),
            ];
            line_integral += 1.0 / rss3(edge_vec[0], edge_vec[1], edge_vec[2]);
        }

        out += area_sub * line_integral / TRIANGLE_SELF_DUFFY_SAMPLES as f64;
    }

    out
}

/// Adaptive evaluation of `∫_triangle dS / R` for near-singular observation points.
///
/// Method:
/// - Use standard triangle quadrature when the observation point is well separated from
///   the triangle relative to its edge length.
/// - Otherwise recursively subdivide the source triangle into four children and sum the
///   child contributions until the pair is sufficiently separated or the depth limit is
///   reached.
///
/// References:
/// - [2], pp. 1448-1455, for numerical treatment of triangle Green-function integrals.
/// - [3], pp. 276-281, for the underlying `1 / R` element integrals.
/// - [1] and [4], as background on handling weak vertex singularities by variable
///   transformation when the observation point approaches the element.
#[inline]
fn triangle_scalar_potential_adaptive(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    obs: [f64; 3],
    quad_kind: QuadratureKind,
    quad_order: usize,
    depth: usize,
) -> f64 {
    let centroid = triangle_centroid(n0, n1, n2);
    let dist = rss3(
        obs[0] - centroid[0],
        obs[1] - centroid[1],
        obs[2] - centroid[2],
    );
    let tri_scale = triangle_max_edge_length(n0, n1, n2);

    if depth == 0 || dist > TRIANGLE_SCALAR_POTENTIAL_NEAR_FACTOR * tri_scale {
        return triangle_scalar_potential_regular(n0, n1, n2, obs, quad_kind, quad_order);
    }

    subdivide_triangle(n0, n1, n2)
        .iter()
        .map(|child| {
            triangle_scalar_potential_adaptive(
                child[0], child[1], child[2], obs, quad_kind, quad_order, depth - 1,
            )
        })
        .sum()
}

/// Double-surface geometric coupling
/// `∫_target ∫_source 1 / |r - r'| dS' dS`
/// for a well-separated triangle pair using plain nested quadrature.
///
/// References:
/// - [5], Eq. (3.16) on p. 68 for mutual inductance via `A · j`, Eq. (4.6) on p. 93
///   for constant triangle current density, and Eqs. (5.3)-(5.5) on pp. 107-108 for
///   triangle vector-potential integrals.
/// - [3], pp. 276-281.
/// - [2], pp. 1448-1455.
#[inline]
pub fn triangle_geometric_coupling_regular(
    src0: [f64; 3],
    src1: [f64; 3],
    src2: [f64; 3],
    tgt0: [f64; 3],
    tgt1: [f64; 3],
    tgt2: [f64; 3],
    quad_kind: QuadratureKind,
    quad_order: usize,
) -> f64 {
    let tri_area_tgt = calc_tri_area(tgt0, tgt1, tgt2);
    let quad_points_tgt = triangle_quadrature_points(quad_kind, quad_order);

    let mut out = 0.0;
    for qp in quad_points_tgt {
        let obs = map_tri_uv(tgt0, tgt1, tgt2, [qp[1], qp[2]]);
        out += qp[0]
            * tri_area_tgt
            * triangle_scalar_potential_regular(src0, src1, src2, obs, quad_kind, quad_order);
    }

    out
}

/// Double-surface self coupling for a triangle.
///
/// Method:
/// - Integrate over target quadrature points on the triangle.
/// - At each target point, evaluate the source-side weakly singular `∫ dS / R` term
///   with the Duffy-style helper above.
///
/// References:
/// - [5], discussion on p. 106 and Eqs. (5.3)-(5.5) on pp. 107-108 for evaluation of
///   vector potential on the source support.
/// - [4], pp. 1260-1262.
/// - [1], abstract and Sec. 2.
/// - [2], pp. 1448-1455, and [3], pp. 276-281.
#[inline]
fn triangle_geometric_coupling_self(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    quad_kind: QuadratureKind,
    quad_order: usize,
) -> f64 {
    let tri_area = calc_tri_area(n0, n1, n2);
    let quad_points = triangle_quadrature_points(quad_kind, quad_order);

    let mut out = 0.0;
    for qp in quad_points {
        let obs = map_tri_uv(n0, n1, n2, [qp[1], qp[2]]);
        out += qp[0] * tri_area * triangle_scalar_potential_self_duffy(n0, n1, n2, obs);
    }

    out
}

/// One directed evaluation of the triangle-pair geometric coupling.
///
/// Method:
/// - Integrate over target quadrature points.
/// - For each target point, evaluate the source triangle's `∫ dS / R` contribution with
///   adaptive subdivision so touching and near-touching pairs remain finite and accurate.
///
/// References:
/// - [5], Eq. (3.16) on p. 68 and discussion on p. 106 that mutual-inductance
///   evaluation requires vector potential on overlapping support.
/// - [2], pp. 1448-1455.
/// - [3], pp. 276-281.
/// - [1] and [4], for Duffy-type handling of weak singularities.
#[inline]
fn triangle_geometric_coupling_one_way(
    src0: [f64; 3],
    src1: [f64; 3],
    src2: [f64; 3],
    tgt0: [f64; 3],
    tgt1: [f64; 3],
    tgt2: [f64; 3],
    quad_kind: QuadratureKind,
    quad_order: usize,
) -> f64 {
    if triangles_identical(src0, src1, src2, tgt0, tgt1, tgt2) {
        return triangle_geometric_coupling_self(src0, src1, src2, quad_kind, quad_order);
    }

    let tri_area_tgt = calc_tri_area(tgt0, tgt1, tgt2);
    let quad_points_tgt = triangle_quadrature_points(quad_kind, quad_order);

    let mut out = 0.0;
    for qp in quad_points_tgt {
        let obs = map_tri_uv(tgt0, tgt1, tgt2, [qp[1], qp[2]]);
        out += qp[0]
            * tri_area_tgt
            * triangle_scalar_potential_adaptive(
                src0,
                src1,
                src2,
                obs,
                quad_kind,
                quad_order,
                TRIANGLE_SCALAR_POTENTIAL_MAX_DEPTH,
            );
    }

    out
}

/// Double-surface geometric coupling
/// `∫_target ∫_source 1 / |r - r'| dS' dS`
/// between two triangles.
///
/// Method:
/// - Use the dedicated self-term path when the two triangles are identical.
/// - Otherwise evaluate the pair in both source/target directions and average the two
///   results so the numerical coupling is explicitly symmetric.
///
/// References:
/// - [5], Eq. (3.16) on p. 68 and Sec. 3.5.1 on p. 85 for the symmetry of mutual
///   inductance, together with Eqs. (5.3)-(5.5) on pp. 107-108 for triangle
///   vector-potential evaluation.
/// - [2], pp. 1448-1455.
/// - [3], pp. 276-281.
/// - [1] and [4], for weakly singular integration background.
#[inline]
pub fn triangle_geometric_coupling(
    src0: [f64; 3],
    src1: [f64; 3],
    src2: [f64; 3],
    tgt0: [f64; 3],
    tgt1: [f64; 3],
    tgt2: [f64; 3],
    quad_kind: QuadratureKind,
    quad_order: usize,
) -> f64 {
    // NOTE: for smaller functions such as the linear and circular filaments,
    // having a branch in the middle of the scalar kernel like this would be
    // very bad for performance. However, because the total number of independent calculations
    // in this scalar kernel is so large, we're going to get good use out of SLP vectorization
    // and don't necessarily need to coddle the loop vectorizer or branch predictor.
    if triangles_identical(src0, src1, src2, tgt0, tgt1, tgt2) {
        return triangle_geometric_coupling_self(src0, src1, src2, quad_kind, quad_order);
    }

    0.5
        * (triangle_geometric_coupling_one_way(
            src0, src1, src2, tgt0, tgt1, tgt2, quad_kind, quad_order,
        ) + triangle_geometric_coupling_one_way(
            tgt0, tgt1, tgt2, src0, src1, src2, quad_kind, quad_order,
        ))
}

/// Mutual-inductance block for the three nodal basis functions on a source triangle and
/// the three nodal basis functions on a target triangle.
///
/// Method:
/// - Linear triangle basis current densities are constant over each triangle.
/// - Compute one scalar geometric coupling `G = ∫∫ 1 / R dS' dS`.
/// - Form the full `3x3` block as `μ0 / 4π * G * (K_src_i · K_tgt_j)`.
///
/// References:
/// - [5], Eq. (3.16) on p. 68 for `M_mn = ∬ A_m · j_n dS`, Eq. (3.24) on p. 70 for the
///   stream-function current representation, and Eq. (4.6) on p. 93 for the constant
///   current density induced by linear triangle nodal values.
/// - [3], pp. 276-281.
/// - [2], pp. 1448-1455.
#[inline]
pub fn triangle_basis_mutual_inductance_block(
    src0: [f64; 3],
    src1: [f64; 3],
    src2: [f64; 3],
    tgt0: [f64; 3],
    tgt1: [f64; 3],
    tgt2: [f64; 3],
    quad_kind: QuadratureKind,
    quad_order: usize,
) -> [[f64; 3]; 3] {
    let g = triangle_geometric_coupling(src0, src1, src2, tgt0, tgt1, tgt2, quad_kind, quad_order);
    let ksrc = triangle_basis_current_densities(src0, src1, src2);
    let ktgt = triangle_basis_current_densities(tgt0, tgt1, tgt2);

    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = MU0_OVER_4PI
                * g
                * dot3(
                    ksrc[i][0], ksrc[i][1], ksrc[i][2], ktgt[j][0], ktgt[j][1], ktgt[j][2],
                );
        }
    }

    out
}

/// Single entry from the triangle-basis mutual-inductance block.
///
/// References:
/// - [5], Eq. (3.16) on p. 68, Eq. (3.24) on p. 70, and Eq. (4.6) on p. 93.
#[inline]
pub fn triangle_basis_mutual_inductance(
    src0: [f64; 3],
    src1: [f64; 3],
    src2: [f64; 3],
    src_basis: usize,
    tgt0: [f64; 3],
    tgt1: [f64; 3],
    tgt2: [f64; 3],
    tgt_basis: usize,
    quad_kind: QuadratureKind,
    quad_order: usize,
) -> f64 {
    triangle_basis_mutual_inductance_block(
        src0, src1, src2, tgt0, tgt1, tgt2, quad_kind, quad_order,
    )[src_basis][tgt_basis]
}

/// Contract a triangle-pair inductance block with source and target nodal potential
/// vectors to obtain the total inductive coupling between the two triangle current
/// distributions.
///
/// References:
/// - [5], Eq. (3.16) on p. 68 for the mutual-inductance bilinear form, together with
///   Eq. (4.6) on p. 93 for the linear dependence of triangle current density on nodal
///   stream-function values.
#[inline]
pub fn triangle_inductance_from_potential_vectors(
    m_block: [[f64; 3]; 3],
    s_src: [f64; 3],
    s_tgt: [f64; 3],
) -> f64 {
    let mut out = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            out += s_src[i] * m_block[i][j] * s_tgt[j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use core::f64::consts::PI;

    use super::{
        QuadratureKind, calc_tri_area, calc_tri_normal, flux_density_triangle, map_tri_uv,
        triangle_basis_current_densities, triangle_basis_mutual_inductance_block,
        triangle_quadrature_points, triangle_inductance_from_potential_vectors,
        triangle_vector_potential_basis, vector_potential_triangle,
    };
    use crate::math::{cartesian_to_cylindrical, dot3};
    use crate::physics::circular_filament::{
        flux_circular_filament_scalar, flux_density_circular_filament_cartesian_scalar,
        vector_potential_circular_filament_scalar,
    };
    use crate::testing::approx;
    use crate::MU0_OVER_4PI;

    #[derive(Clone, Copy)]
    struct TrianglePatch {
        nodes: [[f64; 3]; 3],
        s: [f64; 3],
    }

    fn circular_strip_triangles_at_z(
        radius: f64,
        height: f64,
        s0: f64,
        nphi: usize,
        z_center: f64,
    ) -> Vec<TrianglePatch> {
        assert!(nphi >= 3);

        let dphi = 2.0 * PI / nphi as f64;
        let mut tris = Vec::with_capacity(2 * nphi);

        for i in 0..nphi {
            let phi0 = i as f64 * dphi;
            let phi1 = (i + 1) as f64 * dphi;

            let lower0 = [radius * phi0.cos(), radius * phi0.sin(), z_center - height / 2.0];
            let lower1 = [radius * phi1.cos(), radius * phi1.sin(), z_center - height / 2.0];
            let upper0 = [radius * phi0.cos(), radius * phi0.sin(), z_center + height / 2.0];
            let upper1 = [radius * phi1.cos(), radius * phi1.sin(), z_center + height / 2.0];

            let tri0 = TrianglePatch {
                nodes: [lower0, lower1, upper1],
                s: [-s0, -s0, s0],
            };
            let tri1 = TrianglePatch {
                nodes: [lower0, upper1, upper0],
                s: [-s0, s0, s0],
            };

            let radial = [(phi0 + 0.5 * dphi).cos(), (phi0 + 0.5 * dphi).sin(), 0.0];
            for tri in [tri0, tri1] {
                let normal = calc_tri_normal(tri.nodes[0], tri.nodes[1], tri.nodes[2]);
                let alignment = normal[0] * radial[0] + normal[1] * radial[1];
                assert!(
                    alignment > 0.0,
                    "triangle winding is not radially consistent"
                );
                tris.push(tri);
            }
        }

        tris
    }

    fn circular_strip_triangles(radius: f64, height: f64, s0: f64, nphi: usize) -> Vec<TrianglePatch> {
        circular_strip_triangles_at_z(radius, height, s0, nphi, 0.0)
    }

    fn strip_flux_density(tris: &[TrianglePatch], obs: [f64; 3]) -> [f64; 3] {
        let mut out = [0.0; 3];
        for tri in tris {
            let contrib = flux_density_triangle(
                tri.nodes[0],
                tri.nodes[1],
                tri.nodes[2],
                tri.s,
                obs,
                QuadratureKind::GaussLegendre,
                3,
            );
            out[0] += contrib[0];
            out[1] += contrib[1];
            out[2] += contrib[2];
        }
        out
    }

    fn strip_vector_potential(tris: &[TrianglePatch], obs: [f64; 3]) -> [f64; 3] {
        let mut out = [0.0; 3];
        for tri in tris {
            let contrib = vector_potential_triangle(
                tri.nodes[0],
                tri.nodes[1],
                tri.nodes[2],
                tri.s,
                obs,
                QuadratureKind::GaussLegendre,
                3,
            );
            out[0] += contrib[0];
            out[1] += contrib[1];
            out[2] += contrib[2];
        }
        out
    }

    fn max_abs_component(vectors: &[[f64; 3]]) -> f64 {
        vectors
            .iter()
            .flat_map(|v| v.iter())
            .map(|v| v.abs())
            .fold(0.0, f64::max)
    }

    fn triangle_nodes_for_basis(tri: [[f64; 3]; 3], basis_idx: usize) -> [[f64; 3]; 3] {
        match basis_idx {
            0 => [tri[0], tri[1], tri[2]],
            1 => [tri[1], tri[2], tri[0]],
            2 => [tri[2], tri[0], tri[1]],
            _ => panic!(),
        }
    }

    fn strip_mutual_inductance(src: &[TrianglePatch], tgt: &[TrianglePatch]) -> f64 {
        let mut out = 0.0;
        for source in src {
            for target in tgt {
                let block = triangle_basis_mutual_inductance_block(
                    source.nodes[0],
                    source.nodes[1],
                    source.nodes[2],
                    target.nodes[0],
                    target.nodes[1],
                    target.nodes[2],
                    QuadratureKind::GaussLegendre,
                    3,
                );
                out += triangle_inductance_from_potential_vectors(block, source.s, target.s);
            }
        }
        out
    }

    #[test]
    fn test_triangle_basis_mutual_inductance_block_matches_vector_potential_for_disjoint_triangles() {
        let src = [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.2, 0.8, 0.0]];
        let tgt = [[0.3, -0.2, 1.1], [1.1, 0.1, 1.4], [0.2, 0.9, 1.2]];
        let quad_kind = QuadratureKind::GaussLegendre;
        let quad_order = 3;

        let block = triangle_basis_mutual_inductance_block(
            src[0], src[1], src[2], tgt[0], tgt[1], tgt[2], quad_kind, quad_order,
        );
        let block_t = triangle_basis_mutual_inductance_block(
            tgt[0], tgt[1], tgt[2], src[0], src[1], src[2], quad_kind, quad_order,
        );

        let tri_area_tgt = calc_tri_area(tgt[0], tgt[1], tgt[2]);
        let quad_points_tgt = triangle_quadrature_points(quad_kind, quad_order);
        let ktgt = triangle_basis_current_densities(tgt[0], tgt[1], tgt[2]);

        for i in 0..3 {
            let src_basis = triangle_nodes_for_basis(src, i);
            for j in 0..3 {
                let mut via_a_dot_k = 0.0;
                for qp in quad_points_tgt {
                    let obs = map_tri_uv(tgt[0], tgt[1], tgt[2], [qp[1], qp[2]]);
                    let a_src = triangle_vector_potential_basis(
                        src_basis[0],
                        src_basis[1],
                        src_basis[2],
                        obs,
                        quad_kind,
                        quad_order,
                    );
                    via_a_dot_k += qp[0]
                        * tri_area_tgt
                        * MU0_OVER_4PI
                        * dot3(
                            a_src[0],
                            a_src[1],
                            a_src[2],
                            ktgt[j][0],
                            ktgt[j][1],
                            ktgt[j][2],
                        );
                }

                assert!(
                    approx(block[i][j], via_a_dot_k, 1e-10, 1e-12),
                    "A·K mismatch for block[{i}][{j}]: direct={:.6e}, via_A={:.6e}",
                    block[i][j],
                    via_a_dot_k,
                );
                assert!(
                    approx(block[i][j], block_t[j][i], 1e-10, 1e-12),
                    "reciprocity mismatch for block[{i}][{j}]: M12={:.6e}, M21^T={:.6e}",
                    block[i][j],
                    block_t[j][i],
                );
            }
        }
    }

    #[test]
    fn test_triangle_basis_self_inductance_block_is_symmetric_and_finite() {
        let tri = [[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.2, 0.9, 0.2]];
        let block = triangle_basis_mutual_inductance_block(
            tri[0],
            tri[1],
            tri[2],
            tri[0],
            tri[1],
            tri[2],
            QuadratureKind::GaussLegendre,
            3,
        );

        let mut max_entry: f64 = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                assert!(block[i][j].is_finite(), "self block contains non-finite entry at ({i},{j})");
                assert!(
                    approx(block[i][j], block[j][i], 1e-10, 1e-12),
                    "self block is not symmetric at ({i},{j}): {:.6e} vs {:.6e}",
                    block[i][j],
                    block[j][i],
                );
                max_entry = max_entry.max(block[i][j].abs());
            }
        }

        for s in [[1.0, -1.0, 0.0], [PI, -1.0, 0.5], [2.0, -0.75, -1.25]] {
            let energy_like = triangle_inductance_from_potential_vectors(block, s, s);
            assert!(
                energy_like > -(1e-10 * max_entry.max(1.0)),
                "self block is not positive semidefinite enough for s={s:?}: {:.6e}",
                energy_like,
            );
        }
    }

    #[test]
    fn test_triangle_basis_mutual_inductance_touching_pairs_are_finite_and_reciprocal() {
        let tri0 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let shared_edge = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let shared_vertex = [[0.0, 1.0, 0.0], [0.2, 1.7, 0.4], [-0.3, 1.4, -0.2]];

        for other in [shared_edge, shared_vertex] {
            let block12 = triangle_basis_mutual_inductance_block(
                tri0[0], tri0[1], tri0[2], other[0], other[1], other[2], QuadratureKind::GaussLegendre, 3,
            );
            let block21 = triangle_basis_mutual_inductance_block(
                other[0], other[1], other[2], tri0[0], tri0[1], tri0[2], QuadratureKind::GaussLegendre, 3,
            );

            for i in 0..3 {
                for j in 0..3 {
                    assert!(block12[i][j].is_finite(), "touching-pair entry is non-finite at ({i},{j})");
                    assert!(
                        approx(block12[i][j], block21[j][i], 1e-8, 1e-11),
                        "touching-pair reciprocity mismatch at ({i},{j}): {:.6e} vs {:.6e}",
                        block12[i][j],
                        block21[j][i],
                    );
                }
            }
        }
    }

    #[test]
    fn test_triangle_strip_mutual_inductance_against_circular_filament() {
        let radius = 0.71;
        let height = radius * 1e-3;
        let nphi = 32;
        let current = 1.0;
        let z_src = -0.37;
        let z_tgt = 0.41;

        let strip_src = circular_strip_triangles_at_z(radius, height, current, nphi, z_src);
        let strip_tgt = circular_strip_triangles_at_z(radius, height, current, nphi, z_tgt);

        let m_strip = strip_mutual_inductance(&strip_src, &strip_tgt);
        let m_strip_reverse = strip_mutual_inductance(&strip_tgt, &strip_src);
        let m_loop = flux_circular_filament_scalar((radius, z_src, 1.0), (radius, z_tgt));

        assert!(
            approx(m_loop, m_strip, 4e-2, 1e-12),
            "strip mutual inductance mismatch: strip={:.6e}, circular={:.6e}",
            m_strip,
            m_loop,
        );
        assert!(
            approx(m_strip, m_strip_reverse, 1e-10, 1e-12),
            "strip reciprocity mismatch: M12={:.6e}, M21={:.6e}",
            m_strip,
            m_strip_reverse,
        );
    }

    #[test]
    fn test_flux_density_triangle_circular_strip_matches_circular_filament_far_field() {
        let radius = 0.7312345987;
        let height = radius * 1e-3;
        let nphi = 256;

        let loop_current = 1.7; // Some not-special number to check current scaling
        let s0 = loop_current; // Potential jump for this strip construction matches the loop current

        let strip = circular_strip_triangles(radius, height, s0, nphi);
        let obs = [
            [2.70, 0.95, 0.85],
            [3.10, -1.15, 1.05],
            [3.45, 0.75, -1.25],
            [3.80, 1.30, 1.60],
            [4.20, -0.90, -1.55],
            [4.55, 1.10, -2.05],
        ];

        let mut b_strip = Vec::with_capacity(obs.len());
        let mut b_loop = Vec::with_capacity(obs.len());
        let mut a_strip = Vec::with_capacity(obs.len());
        let mut a_loop = Vec::with_capacity(obs.len());
        for point in obs {
            b_strip.push(strip_flux_density(&strip, point));
            let b_ref = flux_density_circular_filament_cartesian_scalar(
                (radius, 0.0, loop_current),
                (point[0], point[1], point[2]),
            );
            b_loop.push([b_ref.0, b_ref.1, b_ref.2]);

            a_strip.push(strip_vector_potential(&strip, point));
            let (r_obs, phi_obs, z_obs) = cartesian_to_cylindrical(point[0], point[1], point[2]);
            let a_phi = vector_potential_circular_filament_scalar(
                (radius, 0.0, loop_current),
                (r_obs, z_obs),
            );
            a_loop.push([-a_phi * libm::sin(phi_obs), a_phi * libm::cos(phi_obs), 0.0]);
        }

        let b_axis_names = ["Bx", "By", "Bz"];
        let a_axis_names = ["Ax", "Ay", "Az"];
        let bfield_rtol = 1e-3;
        let bfield_atol = max_abs_component(&b_loop) * 1e-12;
        let afield_rtol = 1e-3;
        let afield_atol = max_abs_component(&a_loop) * 1e-12;

        for i in 0..obs.len() {
            for axis in 0..3 {
                assert!(
                    approx(b_loop[i][axis], b_strip[i][axis], bfield_rtol, bfield_atol),
                    "{} mismatch at point {}: strip={:.6e}, reference={:.6e}, obs={:?}",
                    b_axis_names[axis],
                    i,
                    b_strip[i][axis],
                    b_loop[i][axis],
                    obs[i],
                );

                assert!(
                    approx(a_loop[i][axis], a_strip[i][axis], afield_rtol, afield_atol),
                    "{} mismatch at point {}: strip={:.6e}, reference={:.6e}, obs={:?}",
                    a_axis_names[axis],
                    i,
                    a_strip[i][axis],
                    a_loop[i][axis],
                    obs[i],
                );
            }
        }
    }

    #[test]
    fn test_flux_density_triangle_circular_strip_matches_circular_filament_near_axis() {
        let radius = 0.75;
        let height = radius * 1e-3;
        let nphi = 256;

        let loop_current = 1.7; // Some not-special number to check current scaling
        let s0 = loop_current; // Potential jump for this strip construction matches the loop current

        let strip = circular_strip_triangles(radius, height, s0, nphi);

        let mut obs = Vec::with_capacity(202);
        for i in 0..=200 {
            let z = -1.0 + i as f64 * 0.01;
            obs.push([1e-8, 0.0, z]);
        }
        obs.push([0.0, 1e-8, 0.0]);

        let axis_names = ["Bx", "By", "Bz"];
        let bz_rtol = 1e-3;
        let mut b_loop = Vec::with_capacity(obs.len());
        for point in &obs {
            let b_ref = flux_density_circular_filament_cartesian_scalar(
                (radius, 0.0, loop_current),
                (point[0], point[1], point[2]),
            );
            b_loop.push([b_ref.0, b_ref.1, b_ref.2]);
        }
        let transverse_atol = max_abs_component(&b_loop) * 1e-8 + 1e-14;

        for (i, point) in obs.iter().enumerate() {
            let b_strip = strip_flux_density(&strip, *point);

            assert!(
                approx(b_loop[i][2], b_strip[2], bz_rtol, 0.0),
                "Bz mismatch at point {}: strip={:.6e}, reference={:.6e}, obs={:?}",
                i,
                b_strip[2],
                b_loop[i][2],
                point,
            );

            for axis in 0..2 {
                assert!(
                    b_strip[axis].abs() < transverse_atol,
                    "{} should be near zero at point {}: strip={:.6e}, atol={:.6e}, obs={:?}",
                    axis_names[axis],
                    i,
                    b_strip[axis],
                    transverse_atol,
                    point,
                );
            }
        }
    }
}
