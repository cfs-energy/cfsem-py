//! Generic quadrature rules that are shared across multiple element families and kernels.
//!
//! The most reusable rules in this crate are the 1D Gauss-Legendre rules on an interval. They
//! are used directly for line integrals and are also the building blocks for tensor-product
//! quadrature on quadrilateral reference cells.
//!
//! # References
//!
//! - Allan F. Bower, *Applied Mechanics of Solids*, CRC Press, 2009, Section 8.1.12, for the use
//!   of Gauss quadrature in finite-element integration.
//! - NIST Digital Library of Mathematical Functions, §3.5(v) "Gauss Quadrature", especially
//!   Eqs. 3.5.18-3.5.21 for the general Gauss quadrature formula and the Gauss-Legendre
//!   specialization.
//! - NIST Digital Library of Mathematical Functions, §18.3 "Definitions", for the Legendre
//!   polynomial family used by the Gauss-Legendre rule.

use crate::mesh::{Scalar, cast};

const GL2_INTERVAL: [[f64; 2]; 2] = [
    [-0.577_350_269_189_625_7, 1.0],
    [0.577_350_269_189_625_7, 1.0],
];

const GL3_INTERVAL: [[f64; 2]; 3] = [
    [-0.774_596_669_241_483_4, 0.555_555_555_555_555_6],
    [0.0, 0.888_888_888_888_888_8],
    [0.774_596_669_241_483_4, 0.555_555_555_555_555_6],
];

const GL4_INTERVAL: [[f64; 2]; 4] = [
    [-0.861_136_311_594_052_6, 0.347_854_845_137_453_85],
    [-0.339_981_043_584_856_26, 0.652_145_154_862_546_1],
    [0.339_981_043_584_856_26, 0.652_145_154_862_546_1],
    [0.861_136_311_594_052_6, 0.347_854_845_137_453_85],
];

const GL2_UNIT_INTERVAL: [[f64; 2]; 2] = [
    [0.211_324_865_405_187_13, 0.5],
    [0.788_675_134_594_812_9, 0.5],
];

const GL3_UNIT_INTERVAL: [[f64; 2]; 3] = [
    [0.112_701_665_379_258_31, 0.277_777_777_777_777_8],
    [0.5, 0.444_444_444_444_444_4],
    [0.887_298_334_620_741_7, 0.277_777_777_777_777_8],
];

const GL4_UNIT_INTERVAL: [[f64; 2]; 4] = [
    [0.069_431_844_202_973_71, 0.173_927_422_568_726_93],
    [0.330_009_478_207_571_87, 0.326_072_577_431_273_05],
    [0.669_990_521_792_428_1, 0.326_072_577_431_273_05],
    [0.930_568_155_797_026_2, 0.173_927_422_568_726_93],
];

/// Supported 1D Gauss-Legendre rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GaussLegendreRule {
    Gauss2,
    Gauss3,
    Gauss4,
}

impl GaussLegendreRule {
    /// Number of abscissas in the rule.
    pub const fn points(self) -> usize {
        match self {
            Self::Gauss2 => 2,
            Self::Gauss3 => 3,
            Self::Gauss4 => 4,
        }
    }
}

/// Tabulated Gauss-Legendre abscissas and weights on `[-1, 1]`.
///
/// Each entry is `[x_q, w_q]`.
pub fn gauss_legendre_interval_table(rule: GaussLegendreRule) -> &'static [[f64; 2]] {
    match rule {
        GaussLegendreRule::Gauss2 => &GL2_INTERVAL,
        GaussLegendreRule::Gauss3 => &GL3_INTERVAL,
        GaussLegendreRule::Gauss4 => &GL4_INTERVAL,
    }
}

/// Tabulated Gauss-Legendre abscissas and weights on `[0, 1]`.
///
/// Each entry is `[x_q, w_q]`. These are the affine map of the corresponding `[-1, 1]` rule.
pub fn gauss_legendre_unit_interval_table(rule: GaussLegendreRule) -> &'static [[f64; 2]] {
    match rule {
        GaussLegendreRule::Gauss2 => &GL2_UNIT_INTERVAL,
        GaussLegendreRule::Gauss3 => &GL3_UNIT_INTERVAL,
        GaussLegendreRule::Gauss4 => &GL4_UNIT_INTERVAL,
    }
}

/// Gauss-Legendre abscissas and weights on `[-1, 1]`, cast to the target scalar type.
pub fn gauss_legendre_interval<F: Scalar>(rule: GaussLegendreRule) -> Vec<(F, F)> {
    gauss_legendre_interval_table(rule)
        .iter()
        .map(|[x, w]| (cast::<F>(*x), cast::<F>(*w)))
        .collect()
}

/// Gauss-Legendre abscissas and weights on `[0, 1]`, cast to the target scalar type.
pub fn gauss_legendre_unit_interval<F: Scalar>(rule: GaussLegendreRule) -> Vec<(F, F)> {
    gauss_legendre_unit_interval_table(rule)
        .iter()
        .map(|[x, w]| (cast::<F>(*x), cast::<F>(*w)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{GaussLegendreRule, gauss_legendre_interval, gauss_legendre_unit_interval_table};

    #[test]
    fn unit_interval_rule_maps_from_symmetric_interval() {
        let interval = gauss_legendre_interval::<f64>(GaussLegendreRule::Gauss3);
        let unit = gauss_legendre_unit_interval_table(GaussLegendreRule::Gauss3);
        for ((x, w), [xu, wu]) in interval.into_iter().zip(unit.iter()) {
            assert!((xu - 0.5 * (x + 1.0)).abs() < 1.0e-15);
            assert!((wu - 0.5 * w).abs() < 1.0e-15);
        }
    }

    #[test]
    fn gauss4_unit_interval_integrates_quintic_exactly() {
        let integral: f64 = gauss_legendre_unit_interval_table(GaussLegendreRule::Gauss4)
            .iter()
            .map(|[x, w]| w * x.powi(5))
            .sum();
        assert!((integral - (1.0 / 6.0)).abs() < 1.0e-15);
    }
}
