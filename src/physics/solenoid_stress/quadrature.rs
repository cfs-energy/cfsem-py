//! Gauss quadrature rules used by the axisymmetric Quad4 formulation.
//!
//! The volume rule is the tensor product of the 1D rule in `xi` and `eta`, while the face rule
//! reuses the same 1D points along a reference edge.

use crate::physics::solenoid_stress::types::{Real, cast};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuadratureRule {
    /// Tensor-product 2-point rule in each parametric direction.
    Gauss2x2,
    /// Tensor-product 3-point rule in each parametric direction.
    Gauss3x3,
    /// Tensor-product 4-point rule in each parametric direction.
    Gauss4x4,
}

impl QuadratureRule {
    /// Parse the compact integer code exposed through the Python bindings.
    pub fn from_code(code: u8) -> Result<Self, String> {
        match code {
            2 => Ok(Self::Gauss2x2),
            3 => Ok(Self::Gauss3x3),
            4 => Ok(Self::Gauss4x4),
            _ => Err(format!(
                "unsupported quadrature code {code}; use 2 for 2x2, 3 for 3x3, or 4 for 4x4"
            )),
        }
    }

    /// Number of quadrature points contributed by one element volume integral.
    pub fn points_per_element(self) -> usize {
        match self {
            Self::Gauss2x2 => 4,
            Self::Gauss3x3 => 9,
            Self::Gauss4x4 => 16,
        }
    }
}

/// 1D Gauss-Legendre points and weights on `[-1, 1]`.
pub fn gauss_1d<F: Real>(rule: QuadratureRule) -> Vec<(F, F)> {
    match rule {
        QuadratureRule::Gauss2x2 => {
            let a = cast::<F>(1.0 / 3.0_f64.sqrt());
            vec![(-a, F::one()), (a, F::one())]
        }
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
pub fn gauss_volume<F: Real>(rule: QuadratureRule) -> Vec<([F; 2], F)> {
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
pub fn gauss_face<F: Real>(rule: QuadratureRule) -> Vec<(F, F)> {
    gauss_1d::<F>(rule)
}

#[cfg(test)]
mod tests {
    use super::{QuadratureRule, gauss_volume};

    #[test]
    fn gauss_2x2_integrates_constant_to_four() {
        let weight_sum: f64 = gauss_volume::<f64>(QuadratureRule::Gauss2x2)
            .into_iter()
            .map(|(_, w)| w)
            .sum();
        assert!((weight_sum - 4.0).abs() < 1.0e-12);
    }

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
