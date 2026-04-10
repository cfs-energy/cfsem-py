//! Reference-triangle quadrature rules used by the boundary-element triangle kernels.

/// Second-order quadrature integration weights on a triangular surface.
/// Format is `[weight, u, v]`.
const TABLE_GAUSS_LEGENDRE_2: [[f64; 3]; 4] = [
    [0.5283121635e-01, 0.1666666667e+00, 0.7886751346e+00],
    [0.1971687836e+00, 0.6220084679e+00, 0.2113248654e+00],
    [0.5283121635e-01, 0.4465819874e-01, 0.7886751346e+00],
    [0.1971687836e+00, 0.1666666667e+00, 0.2113248654e+00],
];

/// Third-order quadrature integration weights on a triangular surface.
/// Format is `[weight, u, v]`.
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

/// Dunavant's 7-point degree-5 symmetric quadrature rule on a triangle.
/// Format is `[weight, u, v]`.
const TABLE_DUNAVANT_5: [[f64; 3]; 7] = [
    [0.112500000000000, 0.333333333333333, 0.333333333333333],
    [0.066197076394253, 0.470142064105115, 0.470142064105115],
    [0.066197076394253, 0.059715871789770, 0.470142064105115],
    [0.066197076394253, 0.470142064105115, 0.059715871789770],
    [0.062969590272414, 0.101286507323456, 0.101286507323456],
    [0.062969590272414, 0.797426985353087, 0.101286507323456],
    [0.062969590272414, 0.101286507323456, 0.797426985353087],
];

/// Maximum number of quadrature points among the supported triangle rules.
pub const TRIANGLE_MAX_QUADRATURE_POINTS: usize = TABLE_GAUSS_LEGENDRE_3.len();

#[derive(Clone, Copy)]
pub enum QuadratureKind {
    GaussLegendre2,
    GaussLegendre3,
    Dunavant5,
}

#[inline]
pub fn triangle_quadrature_points(quad_kind: QuadratureKind) -> &'static [[f64; 3]] {
    match quad_kind {
        QuadratureKind::GaussLegendre2 => &TABLE_GAUSS_LEGENDRE_2,
        QuadratureKind::GaussLegendre3 => &TABLE_GAUSS_LEGENDRE_3,
        QuadratureKind::Dunavant5 => &TABLE_DUNAVANT_5,
    }
}

#[inline]
pub fn triangle_quadrature_count(quad_kind: QuadratureKind) -> usize {
    triangle_quadrature_points(quad_kind).len()
}
