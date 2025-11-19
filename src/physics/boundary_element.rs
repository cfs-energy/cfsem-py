use crate::math::{cross3, rss3};
use crate::MU0_OVER_4PI;


/// Hard-coded quadrature points and weights for different integration schemes
/// Format is [Weights, U, V]
const TABLE_GAUSS_LEGENDRE_2: [[f64; 3]; 4] = [
    [0.5283121635e-01, 0.1666666667e+00, 0.7886751346e+00], 
    [0.1971687836e+00, 0.6220084679e+00, 0.2113248654e+00],
    [0.5283121635e-01, 0.4465819874e-01, 0.7886751346e+00],
    [0.1971687836e+00, 0.1666666667e+00, 0.2113248654e+00]
];

const TABLE_GAUSS_LEGENDRE_3: [[f64; 3]; 9] = [
    [0.9876542474e-01, 0.2500000000e+00, 0.5000000000e+00],
    [0.1391378575e-01, 0.5635083269e-01, 0.8872983346e+00],
    [0.1095430035e+00, 0.4436491673e+00, 0.1127016654e+00],
    [0.6172839460e-01, 0.4436491673e+00, 0.5000000000e+00],
    [0.8696116674e-02, 0.1000000000e+00, 0.8872983346e+00],
    [0.6846438175e-01, 0.7872983346e+00, 0.1127016654e+00],
    [0.6172839460e-01, 0.5635083269e-01, 0.5000000000e+00],
    [0.8696116674e-02, 0.1270166538e-01, 0.8872983346e+00],
    [0.6846438175e-01, 0.1000000000e+00, 0.1127016654e+00]
];

#[derive(Clone,Copy)]
pub enum QuadratureKind {
    GaussLegendre,
    Dunavant,
}

#[inline]
pub fn map_tri_uv(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    pin_uv: [f64; 2],
) -> [f64; 3] {
    // Barycentric interpolation
    let mut pout = [0.0; 3];

    // Precalulate third uv component
    let w = 1.0 - pin_uv[0] - pin_uv[1];
    pout[0] = n0[0] * w + n1[0] * pin_uv[0] + n2[0] * pin_uv[1];
    pout[1] = n0[1] * w + n1[1] * pin_uv[0] + n2[1] * pin_uv[1];
    pout[2] = n0[2] * w + n1[2] * pin_uv[0] + n2[2] * pin_uv[1];
    
    return pout
}

#[inline]
pub fn calc_tri_area(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
) -> f64 {
    // Get two directional vectors from node 0 to nodes 1 and 2
    let v01 = [
        n1[0] - n0[0], 
        n1[1] - n0[1], 
        n1[2] - n0[2],
    ];
    let v02 = [
        n2[0] - n0[0], 
        n2[1] - n0[1], 
        n2[2] - n0[2],
    ];

    // Area of parallelgram is norm of crossproduct of the vectors
    let cross = cross3(v01[0], v01[1], v01[2], v02[0], v02[1], v02[2]);
    let area = 0.5 * rss3(cross.0, cross.1, cross.2);
    return area
}

#[inline]
pub fn calc_tri_normal(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],  
) -> [f64; 3] {
    // Get two directional vectors from node 0 to nodes 1 and 2
    let v01 = [
        n1[0] - n0[0], 
        n1[1] - n0[1], 
        n1[2] - n0[2],
    ];
    let v02 = [
        n2[0] - n0[0], 
        n2[1] - n0[1], 
        n2[2] - n0[2],
    ];

    // Get cross product for the two directional vectors
    let cross = cross3(v01[0], v01[1], v01[2], v02[0], v02[1], v02[2]);
    
    // Normalize the normal vector
    let norm = rss3(cross.0, cross.1, cross.2);
    let out = [cross.0 / norm, cross.1 / norm, cross.2 / norm];

    return out;
}

/// Magnetic field contribution of a given triangle's basis function with unit weighting
/// to a given obseration point. Assumes a basis function living on the triangle's first node
/// Args:
///  - TODO
/// Returns:
///  - TODO
#[inline]
pub fn triangle_biot_savart_basis(
    n0: [f64;3],
    n1: [f64;3],
    n2: [f64;3],
    obs: [f64;3],
    quad_kind: QuadratureKind,
    quad_order: usize,
) -> [f64;3] {
    // Calculate directional vectors between nodes
    let v01 = [
        n1[0] - n0[0], 
        n1[1] - n0[1], 
        n1[2] - n0[2],
    ];
    let v02 = [
        n2[0] - n0[0], 
        n2[1] - n0[1], 
        n2[2] - n0[2],
    ];

    // Triangle area and normal vector
    let tri_area = calc_tri_area(n0, n1, n2);

    // Reference current density vector for this triangle
    let jref = [
        (v02[0] - v01[0])/(2.0*tri_area),
        (v02[1] - v01[1])/(2.0*tri_area),
        (v02[2] - v01[2])/(2.0*tri_area)
    ];

    // Match quadrature kind and order to list of quadrature points
    let quad_points: &[[f64; 3]] = match (quad_kind, quad_order) {
        (QuadratureKind::GaussLegendre, 2) => &TABLE_GAUSS_LEGENDRE_2,
        (QuadratureKind::GaussLegendre, 3) => &TABLE_GAUSS_LEGENDRE_3,
        _ => panic!()
    };

    let mut b = [0.0; 3];

    for i in 0..quad_points.len() {
        // Unpack current quad point
        let (c,u,v) = (quad_points[i][0], quad_points[i][1], quad_points[i][2]);

        // Transform current quad points for the given triangle
        let qp = map_tri_uv(n0, n1, n2, [u, v]);

        // Distance of quad point to observation point
        let d = rss3(obs[0] - qp[0], obs[1] - qp[1], obs[2] - qp[2]);

        // Add B-field contribution for current quadrature point
        b[0] += c * (jref[2] * (obs[1] - qp[1]) + jref[1] * (obs[2] - qp[2])) * d * tri_area;
        b[1] += c * (jref[0] * (obs[2] - qp[2]) + jref[2] * (obs[0] - qp[0])) * d * tri_area;
        b[2] += c * (jref[1] * (obs[0] - qp[0]) + jref[0] * (obs[1] - qp[1])) * d * tri_area;
    };   

    return b
}

pub fn triangle_biot_savart(
    n0: [f64;3],
    n1: [f64;3],
    n2: [f64;3],
    s: [f64;3],
    obs: [f64;3],
    quad_kind: QuadratureKind,
    quad_order: usize,   
) -> [f64;3] {
    let mut out = [0.0; 3];

    // Collect B-field contributions for the three basis functions living on n0, n1, and n2
    let b_n0 = triangle_biot_savart_basis(n0, n1, n2, obs, quad_kind, quad_order);
    let b_n1 = triangle_biot_savart_basis(n1, n2, n0, obs, quad_kind, quad_order);
    let b_n2 = triangle_biot_savart_basis(n2, n0, n1, obs, quad_kind, quad_order);

    // Sum contributions by each basis function weighted by the basis function value
    out[0] = (s[0] * b_n0[0] + s[1] * b_n1[0] + s[2] * b_n2[0]) * MU0_OVER_4PI;
    out[1] = (s[0] * b_n0[1] + s[1] * b_n1[1] + s[2] * b_n2[1]) * MU0_OVER_4PI;
    out[2] = (s[0] * b_n0[2] + s[1] * b_n1[2] + s[2] * b_n2[2]) * MU0_OVER_4PI;

    return out;
}
