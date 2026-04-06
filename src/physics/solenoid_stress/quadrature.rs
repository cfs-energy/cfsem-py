use crate::physics::solenoid_stress::types::{Real, cast};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuadratureRule {
    Gauss2x2,
    Gauss3x3,
}

impl QuadratureRule {
    pub fn from_code(code: u8) -> Result<Self, String> {
        match code {
            2 => Ok(Self::Gauss2x2),
            3 => Ok(Self::Gauss3x3),
            _ => Err(format!(
                "unsupported quadrature code {code}; use 2 for 2x2 or 3 for 3x3"
            )),
        }
    }

    pub fn points_per_element(self) -> usize {
        match self {
            Self::Gauss2x2 => 4,
            Self::Gauss3x3 => 9,
        }
    }
}

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
    }
}

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
}
