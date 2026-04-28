//! Reference-triangle quadrature rules used by the boundary-element triangle kernels.

/// Dunavant's one-point degree-1 centroid quadrature rule on a triangle.
/// Format is `[weight, u, v]`.
const TABLE_DUNAVANT_1: [[f64; 3]; 1] = [[0.5, 1.0 / 3.0, 1.0 / 3.0]];

/// Dunavant's 3-point degree-2 symmetric quadrature rule on a triangle.
/// Format is `[weight, u, v]`.
const TABLE_DUNAVANT_2: [[f64; 3]; 3] = [
    [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
];

/// Dunavant's 4-point degree-3 symmetric quadrature rule on a triangle.
/// Format is `[weight, u, v]`.
const TABLE_DUNAVANT_3: [[f64; 3]; 4] = [
    [-27.0 / 96.0, 1.0 / 3.0, 1.0 / 3.0],
    [25.0 / 96.0, 0.2, 0.2],
    [25.0 / 96.0, 0.6, 0.2],
    [25.0 / 96.0, 0.2, 0.6],
];

/// Dunavant's 6-point degree-4 symmetric quadrature rule on a triangle.
/// Format is `[weight, u, v]`.
const TABLE_DUNAVANT_4: [[f64; 3]; 6] = [
    [0.1116907948390055, 0.445948490915965, 0.445948490915965],
    [0.1116907948390055, 0.108103018168070, 0.445948490915965],
    [0.1116907948390055, 0.445948490915965, 0.108103018168070],
    [0.054975871827661, 0.091576213509771, 0.091576213509771],
    [0.054975871827661, 0.816847572980459, 0.091576213509771],
    [0.054975871827661, 0.091576213509771, 0.816847572980459],
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
pub const TRIANGLE_MAX_QUADRATURE_POINTS: usize = TABLE_DUNAVANT_5.len();

/// Supported reference-triangle quadrature rules for boundary-element kernels.
#[derive(Clone, Copy)]
pub enum QuadratureKind {
    /// One-point Dunavant rule, exact through total polynomial degree 1.
    Dunavant1,
    /// Three-point Dunavant rule, exact through total polynomial degree 2.
    Dunavant2,
    /// Four-point Dunavant rule, exact through total polynomial degree 3.
    Dunavant3,
    /// Six-point Dunavant rule, exact through total polynomial degree 4.
    Dunavant4,
    /// Seven-point Dunavant rule, exact through total polynomial degree 5.
    Dunavant5,
}

/// Return the reference-triangle quadrature table for the requested rule.
#[inline]
pub fn triangle_quadrature_points(quad_kind: QuadratureKind) -> &'static [[f64; 3]] {
    match quad_kind {
        QuadratureKind::Dunavant1 => &TABLE_DUNAVANT_1,
        QuadratureKind::Dunavant2 => &TABLE_DUNAVANT_2,
        QuadratureKind::Dunavant3 => &TABLE_DUNAVANT_3,
        QuadratureKind::Dunavant4 => &TABLE_DUNAVANT_4,
        QuadratureKind::Dunavant5 => &TABLE_DUNAVANT_5,
    }
}

/// Return the number of reference-triangle quadrature points for the requested rule.
#[inline]
pub fn triangle_quadrature_count(quad_kind: QuadratureKind) -> usize {
    triangle_quadrature_points(quad_kind).len()
}
