//! Gauss quadrature rules used by the 2D quadrilateral reference elements.
//!
//! The 1D rules in this module are the standard Gauss-Legendre rules on `[-1, 1]`: the nodes
//! are the roots of the degree-`n` Legendre polynomial and the weights are the corresponding
//! Christoffel numbers. The 2D square rules are then formed as tensor products of that 1D rule.
//!
//! # References
//!
//! - NIST Digital Library of Mathematical Functions, §3.5(v) "Gauss Quadrature", especially
//!   Eqs. 3.5.18-3.5.21 for the general Gauss quadrature formula and the Gauss-Legendre
//!   specialization.
//! - NIST Digital Library of Mathematical Functions, §18.3 "Definitions", for the Legendre
//!   polynomial family used by the Gauss-Legendre rule.

use crate::mesh::{Scalar, cast};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuadratureRule {
    /// Tensor-product 3-point rule in each parametric direction.
    Gauss3x3,
    /// Tensor-product 4-point rule in each parametric direction.
    Gauss4x4,
}

impl QuadratureRule {
    /// Parse the compact integer code exposed through the Python bindings.
    pub fn from_code(code: u8) -> Result<Self, String> {
        match code {
            3 => Ok(Self::Gauss3x3),
            4 => Ok(Self::Gauss4x4),
            _ => Err(format!(
                "unsupported quadrature code {code}; use 3 for 3x3 or 4 for 4x4"
            )),
        }
    }

    /// Number of quadrature points contributed by one element volume integral.
    pub fn points_per_element(self) -> usize {
        match self {
            Self::Gauss3x3 => 9,
            Self::Gauss4x4 => 16,
        }
    }
}

/// 1D Gauss-Legendre points and weights on `[-1, 1]`.
///
/// The hard-coded `3`- and `4`-point values are the exact Gauss-Legendre abscissas and weights
/// specialized from the general rule for the Legendre weight `w(x) = 1`.
///
/// # References
///
/// - NIST Digital Library of Mathematical Functions, §3.5(v), Eqs. 3.5.18-3.5.21.
/// - NIST Digital Library of Mathematical Functions, §18.3, for the Legendre polynomial family.
pub fn gauss_1d<F: Scalar>(rule: QuadratureRule) -> Vec<(F, F)> {
    match rule {
        QuadratureRule::Gauss3x3 => {
            let a = cast::<F>((3.0_f64 / 5.0).sqrt());
            vec![
                (-a, cast(5.0 / 9.0)),
                (F::zero(), cast(8.0 / 9.0)),
                (a, cast(5.0 / 9.0)),
            ]
        }
        QuadratureRule::Gauss4x4 => {
            let a1 = cast::<F>(0.861_136_311_594_052_6);
            let a2 = cast::<F>(0.339_981_043_584_856_26);
            let w1 = cast::<F>(0.347_854_845_137_453_85);
            let w2 = cast::<F>(0.652_145_154_862_546_1);
            vec![(-a1, w1), (-a2, w2), (a2, w2), (a1, w1)]
        }
    }
}

/// Tensor-product Gauss rule on the reference square `[-1, 1]^2`.
///
/// This is the Cartesian product of the selected 1D Gauss-Legendre rule in `\xi` and `\eta`.
pub fn gauss_volume<F: Scalar>(rule: QuadratureRule) -> Vec<([F; 2], F)> {
    let line = gauss_1d::<F>(rule);
    let mut out = Vec::with_capacity(line.len() * line.len());
    for (xi, wx) in &line {
        for (eta, wy) in &line {
            out.push(([*xi, *eta], *wx * *wy));
        }
    }
    out
}

/// 1D Gauss rule reused for integrating along a reference element face.
///
/// The face rule is exactly the same 1D Gauss-Legendre rule returned by [`gauss_1d()`].
pub fn gauss_face<F: Scalar>(rule: QuadratureRule) -> Vec<(F, F)> {
    gauss_1d::<F>(rule)
}

#[cfg(test)]
mod tests {
    use super::{QuadratureRule, gauss_volume};

    #[test]
    fn gauss_3x3_integrates_quadratic_exactly() {
        let integral: f64 = gauss_volume::<f64>(QuadratureRule::Gauss3x3)
            .into_iter()
            .map(|([xi, eta], w)| (xi * xi + eta * eta) * w)
            .sum();
        assert!((integral - (8.0 / 3.0)).abs() < 1.0e-12);
    }

    #[test]
    fn gauss_4x4_integrates_constant_to_four() {
        let weight_sum: f64 = gauss_volume::<f64>(QuadratureRule::Gauss4x4)
            .into_iter()
            .map(|(_, w)| w)
            .sum();
        assert!((weight_sum - 4.0).abs() < 1.0e-12);
    }

    #[test]
    fn gauss_4x4_integrates_degree_six_polynomial_exactly() {
        let integral: f64 = gauss_volume::<f64>(QuadratureRule::Gauss4x4)
            .into_iter()
            .map(|([xi, eta], w)| (xi.powi(6) + eta.powi(6)) * w)
            .sum();
        let exact = 8.0 / 7.0;
        assert!((integral - exact).abs() < 1.0e-12);
    }
}
