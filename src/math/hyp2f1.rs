//! Gauss hypergeometric function with complex parameters and argument.
// The private continuation primitives are introduced and validated before the
// region evaluators that consume them.
#![allow(dead_code)]

use num_complex::Complex64;
use rayon::prelude::*;

use crate::{chunksize, macros::check_length};

const NAN: Complex64 = Complex64::new(f64::NAN, f64::NAN);
const REL_TOL: f64 = 8.0 * f64::EPSILON;
const MAX_SERIES_ITERATIONS: usize = 10_000;
const LANCZOS_G_MINUS_HALF: f64 = 4.242_187_5;
const LOG_SQRT_TWO_PI: f64 = 0.918_938_533_204_672_7;
const LANCZOS_COEFFICIENTS: [f64; 15] = [
    0.999_999_999_999_997_1,
    57.156_235_665_862_92,
    -59.597_960_355_475_49,
    14.136_097_974_741_746,
    -0.491_913_816_097_620_2,
    0.000_033_994_649_984_811_89,
    0.000_046_523_628_927_048_58,
    -0.000_098_374_475_304_879_65,
    0.000_158_088_703_224_912_5,
    -0.000_210_264_441_724_104_88,
    0.000_217_439_618_115_212_64,
    -0.000_164_318_106_536_763_9,
    0.000_084_418_223_983_852_74,
    -0.000_026_190_838_401_581_067,
    0.000_003_689_918_265_953_162_5,
];

#[derive(Clone, Copy, Debug)]
struct EvalOutcome {
    value: Complex64,
    converged: bool,
}

impl EvalOutcome {
    #[inline]
    const fn success(value: Complex64) -> Self {
        Self {
            value,
            converged: true,
        }
    }

    #[inline]
    const fn failure() -> Self {
        Self {
            value: NAN,
            converged: false,
        }
    }
}

#[inline]
fn finite(z: Complex64) -> bool {
    z.re.is_finite() && z.im.is_finite()
}

#[inline]
fn complex_abs(z: Complex64) -> f64 {
    z.re.hypot(z.im)
}

#[inline]
fn complex_log1p(z: Complex64) -> Complex64 {
    if z == -Complex64::ONE {
        return Complex64::new(f64::NEG_INFINITY, z.im);
    }
    let quadratic = 2.0 * z.re + z.re * z.re + z.im * z.im;
    if quadratic > -1.0 {
        Complex64::new(0.5 * quadratic.ln_1p(), z.im.atan2(1.0 + z.re))
    } else {
        (Complex64::ONE + z).ln()
    }
}

#[inline]
fn complex_expm1(z: Complex64) -> Complex64 {
    let (sin_y, cos_y) = z.im.sin_cos();
    Complex64::new(
        z.re.exp_m1() * cos_y - 2.0 * (0.5 * z.im).sin().powi(2),
        z.re.exp() * sin_y,
    )
}

#[inline]
fn complex_pow(base: Complex64, exponent: Complex64) -> Complex64 {
    (exponent * base.ln()).exp()
}

#[inline]
fn is_nonpositive_integer(z: Complex64) -> bool {
    z.im == 0.0 && z.re <= 0.0 && z.re.is_finite() && z.re == z.re.trunc()
}

#[inline]
fn nearest_integer_difference(z: Complex64) -> (i32, Complex64) {
    let nearest = z.re.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32;
    (nearest, z - nearest as f64)
}

#[inline]
fn sin_pi(z: Complex64) -> Complex64 {
    let x = core::f64::consts::PI * (z.re % 2.0);
    let y = core::f64::consts::PI * z.im;
    Complex64::new(x.sin() * y.cosh(), x.cos() * y.sinh())
}

#[inline]
fn cos_pi(z: Complex64) -> Complex64 {
    let x = core::f64::consts::PI * (z.re % 2.0);
    let y = core::f64::consts::PI * z.im;
    Complex64::new(x.cos() * y.cosh(), -x.sin() * y.sinh())
}

fn log_sin_pi(z: Complex64) -> Complex64 {
    let y = core::f64::consts::PI * z.im;
    if y.abs() < 20.0 {
        return sin_pi(z).ln();
    }
    let x = core::f64::consts::PI * (z.re % 2.0);
    let phase = z.im.signum() * x.cos();
    Complex64::new(y.abs() - core::f64::consts::LN_2, phase.atan2(x.sin()))
}

#[inline]
fn cot_pi(z: Complex64) -> Complex64 {
    let two_x = 2.0 * core::f64::consts::PI * (z.re % 1.0);
    let two_y = 2.0 * core::f64::consts::PI * z.im;
    if two_y.abs() > 350.0 {
        return Complex64::new(0.0, -z.im.signum());
    }
    let denominator = two_y.cosh() - two_x.cos();
    Complex64::new(two_x.sin() / denominator, -two_y.sinh() / denominator)
}

#[inline]
fn sinc_pi(z: Complex64) -> Complex64 {
    if z == Complex64::ZERO {
        Complex64::ONE
    } else {
        sin_pi(z) / (core::f64::consts::PI * z)
    }
}

fn lanczos_sum(z: Complex64) -> Complex64 {
    let mut sum = Complex64::new(LANCZOS_COEFFICIENTS[0], 0.0);
    for (index, coefficient) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        sum += coefficient / (z + (index - 1) as f64);
    }
    sum
}

fn log_gamma(z: Complex64) -> Complex64 {
    if is_nonpositive_integer(z) {
        return Complex64::new(f64::INFINITY, f64::NAN);
    }
    if z.re < 0.5 {
        return Complex64::new(core::f64::consts::PI.ln(), 0.0)
            - log_sin_pi(z)
            - log_gamma(Complex64::ONE - z);
    }
    let shifted = z + LANCZOS_G_MINUS_HALF;
    Complex64::new(LOG_SQRT_TWO_PI, 0.0) + (z - 0.5) * shifted.ln() - shifted + lanczos_sum(z).ln()
}

#[inline]
fn gamma(z: Complex64) -> Complex64 {
    log_gamma(z).exp()
}

#[inline]
fn reciprocal_gamma(z: Complex64) -> Complex64 {
    if is_nonpositive_integer(z) {
        Complex64::ZERO
    } else {
        (-log_gamma(z)).exp()
    }
}

fn gamma_ratio(numerator: &[Complex64], denominator: &[Complex64]) -> Complex64 {
    let numerator_log = numerator
        .iter()
        .copied()
        .map(log_gamma)
        .fold(Complex64::ZERO, |sum, value| sum + value);
    let denominator_log = denominator
        .iter()
        .copied()
        .map(log_gamma)
        .fold(Complex64::ZERO, |sum, value| sum + value);
    (numerator_log - denominator_log).exp()
}

fn pochhammer(z: Complex64, order: i32) -> Complex64 {
    let mut result = Complex64::ONE;
    if order >= 0 {
        for n in 0..order {
            result *= z + n as f64;
        }
    } else {
        for n in order..0 {
            result /= z + n as f64;
        }
    }
    result
}

fn digamma(mut z: Complex64) -> Complex64 {
    if is_nonpositive_integer(z) {
        return NAN;
    }
    // Shifting is especially accurate next to a pole, where the reflection
    // formula subtracts a large cotangent term. Reserve reflection for inputs
    // so far left that a long recurrence would be needlessly expensive.
    if z.re < -64.0 {
        return digamma(Complex64::ONE - z) - core::f64::consts::PI * cot_pi(z);
    }
    let mut result = Complex64::ZERO;
    while z.re < 8.0 {
        result -= Complex64::ONE / z;
        z += 1.0;
    }
    let inverse = Complex64::ONE / z;
    let inverse_squared = inverse * inverse;
    let correction = 1.0 / 12.0
        + inverse_squared
            * (-1.0 / 120.0
                + inverse_squared
                    * (1.0 / 252.0
                        + inverse_squared * (-1.0 / 240.0 + inverse_squared * (5.0 / 660.0))));
    result + z.ln() - 0.5 * inverse - inverse_squared * correction
}

fn lanczos_ratio(z: Complex64, epsilon: Complex64) -> Complex64 {
    let mut numerator = Complex64::ZERO;
    let mut denominator = Complex64::new(LANCZOS_COEFFICIENTS[0], 0.0);
    for (index, coefficient) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        let offset = (index - 1) as f64;
        let inverse = Complex64::ONE / (z + offset);
        numerator += coefficient * inverse / (z + epsilon + offset);
        denominator += coefficient * inverse;
    }
    numerator / denominator
}

fn log_gamma_difference_over_epsilon(z: Complex64, epsilon: Complex64) -> Complex64 {
    let shifted = z + epsilon;
    let base = z - 0.5;
    let lanczos_argument = base + 4.742_187_5;
    if z.re >= 0.5 {
        if shifted == z {
            return base / lanczos_argument + lanczos_argument.ln()
                - 1.0
                - lanczos_ratio(z, epsilon);
        }
        let difference = base * complex_log1p(epsilon / lanczos_argument)
            + epsilon * (lanczos_argument + epsilon).ln()
            - epsilon
            + complex_log1p(-epsilon * lanczos_ratio(z, epsilon));
        return complex_expm1(difference) / epsilon;
    }

    let tangent = sin_pi(z) / cos_pi(z);
    if shifted == z {
        return log_gamma_difference_over_epsilon(Complex64::ONE - z, epsilon)
            - core::f64::consts::PI / tangent;
    }
    let mut value = (cos_pi(epsilon) + sin_pi(epsilon) / tangent)
        * log_gamma_difference_over_epsilon(Complex64::ONE - z, -epsilon)
        + 0.5 * epsilon * (core::f64::consts::PI * sinc_pi(0.5 * epsilon)).powi(2)
        - core::f64::consts::PI * sinc_pi(epsilon) / tangent;
    value /= Complex64::ONE - epsilon * value;
    value
}

fn gamma_difference_ratio(z: Complex64, epsilon: Complex64) -> Complex64 {
    let shifted = z + epsilon;
    if complex_abs(epsilon) > 0.1 {
        return (reciprocal_gamma(z) - reciprocal_gamma(shifted)) / epsilon;
    }
    if shifted == z {
        if is_nonpositive_integer(z) {
            let integer = z.re as i32;
            let sign = if (integer + 1).rem_euclid(2) == 0 {
                1.0
            } else {
                -1.0
            };
            return sign * gamma(Complex64::new((1 - integer) as f64, 0.0));
        }
        return digamma(z) * reciprocal_gamma(z);
    }
    if is_nonpositive_integer(z) {
        return -reciprocal_gamma(shifted) / epsilon;
    }
    if is_nonpositive_integer(shifted) {
        return reciprocal_gamma(z) / epsilon;
    }
    let (z_integer, _) = nearest_integer_difference(z);
    let (shifted_integer, _) = nearest_integer_difference(shifted);
    if complex_abs(z + (z_integer.abs() as f64))
        < complex_abs(shifted + (shifted_integer.abs() as f64))
    {
        log_gamma_difference_over_epsilon(z, epsilon) * reciprocal_gamma(shifted)
    } else {
        log_gamma_difference_over_epsilon(shifted, -epsilon) * reciprocal_gamma(z)
    }
}

fn pochhammer_difference_ratio(z: Complex64, epsilon: Complex64, order: i32) -> Complex64 {
    debug_assert!(order >= 0);
    if order == 0 {
        return Complex64::ZERO;
    }
    if epsilon == Complex64::ZERO {
        let mut derivative = Complex64::ZERO;
        for index in 0..order {
            let mut product = Complex64::ONE;
            for other in 0..order {
                if other != index {
                    product *= z + other as f64;
                }
            }
            derivative += product;
        }
        return derivative;
    }
    (pochhammer(z + epsilon, order) - pochhammer(z, order)) / epsilon
}

#[inline]
fn exponential_difference_ratio(z: Complex64, epsilon: Complex64) -> Complex64 {
    if epsilon == Complex64::ZERO {
        z
    } else {
        complex_expm1(epsilon * z) / epsilon
    }
}

#[inline]
fn direct_series(a: Complex64, b: Complex64, c: Complex64, z: Complex64) -> EvalOutcome {
    let mut term = Complex64::ONE;
    let mut sum = Complex64::ONE;
    let mut small_terms = 0;

    for n in 0..MAX_SERIES_ITERATIONS {
        let nf = n as f64;
        let denominator = (c + nf) * (nf + 1.0);
        if denominator == Complex64::ZERO {
            return EvalOutcome::failure();
        }
        term *= (a + nf) * (b + nf) * z / denominator;
        sum += term;
        if !finite(term) || !finite(sum) {
            return EvalOutcome::failure();
        }

        if term == Complex64::ZERO
            || (sum != Complex64::ZERO && term.norm() <= REL_TOL * sum.norm())
        {
            small_terms += 1;
            if small_terms >= 2 {
                return EvalOutcome::success(sum);
            }
        } else {
            small_terms = 0;
        }
    }
    EvalOutcome::failure()
}

/// Evaluate Gauss's hypergeometric function on its principal branch.
///
/// All four arguments may be complex. Values on the branch cut distinguish
/// positive and negative zero in `z.im`. Mathematical singularities,
/// non-finite inputs, and numerical nonconvergence return a complex NaN.
#[inline]
pub fn hyp2f1_scalar(a: Complex64, b: Complex64, c: Complex64, z: Complex64) -> Complex64 {
    if !finite(a) || !finite(b) || !finite(c) || !finite(z) {
        return NAN;
    }
    if z == Complex64::ZERO || a == Complex64::ZERO || b == Complex64::ZERO {
        return Complex64::ONE;
    }
    let result = direct_series(a, b, c, z);
    if result.converged { result.value } else { NAN }
}

/// Evaluate Gauss's hypergeometric function elementwise on equal-length slices.
pub fn hyp2f1(
    a: &[Complex64],
    b: &[Complex64],
    c: &[Complex64],
    z: &[Complex64],
    out: &mut [Complex64],
) -> Result<(), &'static str> {
    check_length!(out.len(), a, b, c, z);
    for i in 0..out.len() {
        out[i] = hyp2f1_scalar(a[i], b[i], c[i], z[i]);
    }
    Ok(())
}

/// Parallel elementwise evaluation of Gauss's hypergeometric function.
pub fn hyp2f1_par(
    a: &[Complex64],
    b: &[Complex64],
    c: &[Complex64],
    z: &[Complex64],
    out: &mut [Complex64],
) -> Result<(), &'static str> {
    check_length!(out.len(), a, b, c, z);
    let chunk = chunksize(out.len());
    out.par_chunks_mut(chunk)
        .zip(a.par_chunks(chunk))
        .zip(b.par_chunks(chunk))
        .zip(c.par_chunks(chunk))
        .zip(z.par_chunks(chunk))
        .for_each(|((((out, a), b), c), z)| {
            hyp2f1(a, b, c, z, out).expect("validated equal chunk lengths");
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Complex64, expected: Complex64, tolerance: f64) {
        let scale = complex_abs(expected).max(1.0);
        assert!(
            complex_abs(actual - expected) <= tolerance * scale,
            "actual={actual:?}, expected={expected:?}, tolerance={tolerance}"
        );
    }

    fn parse_gamma_fixture() -> Vec<[f64; 8]> {
        let fixture = include_str!("../../test/data/complex_gamma_reference.csv");
        let mut rows = Vec::new();
        for (line_index, line) in fixture.lines().enumerate() {
            if line.starts_with('#') || line_index == 1 || line.is_empty() {
                continue;
            }
            let fields: Vec<_> = line.split(',').collect();
            assert_eq!(
                fields.len(),
                8,
                "complex_gamma_reference.csv row {} has wrong field count",
                line_index + 1
            );
            let mut row = [0.0; 8];
            for (field_index, field) in fields.iter().enumerate() {
                row[field_index] = field.parse().unwrap_or_else(|error| {
                    panic!(
                        "complex_gamma_reference.csv row {}, field {} is not f64: {error}",
                        line_index + 1,
                        field_index + 1
                    )
                });
            }
            rows.push(row);
        }
        rows
    }

    #[test]
    fn complex_gamma_and_digamma_match_reference() {
        for [
            z_re,
            z_im,
            gamma_re,
            gamma_im,
            rgamma_re,
            rgamma_im,
            digamma_re,
            digamma_im,
        ] in parse_gamma_fixture()
        {
            let z = Complex64::new(z_re, z_im);
            assert_close(gamma(z), Complex64::new(gamma_re, gamma_im), 8e-13);
            assert_close(
                reciprocal_gamma(z),
                Complex64::new(rgamma_re, rgamma_im),
                8e-13,
            );
            assert_close(digamma(z), Complex64::new(digamma_re, digamma_im), 8e-13);
        }
    }

    #[test]
    fn elementary_helpers_preserve_small_complex_increments() {
        let z = Complex64::new(1e-12, -2e-12);
        assert_close(complex_log1p(z).exp() - 1.0, z, 2e-16);
        assert_close(complex_expm1(z), z, 2e-16);
        assert_close(sinc_pi(Complex64::ZERO), Complex64::ONE, 0.0);
        assert_close(
            complex_pow(Complex64::new(-2.0, 0.0), Complex64::new(0.5, 0.0)),
            Complex64::new(0.0, 2.0_f64.sqrt()),
            2e-15,
        );
    }

    #[test]
    fn gamma_helpers_obey_identities() {
        let z = Complex64::new(-0.3, 0.7);
        assert_close(gamma(z + 1.0), z * gamma(z), 2e-13);
        assert_close(
            gamma(z) * gamma(1.0 - z),
            core::f64::consts::PI / sin_pi(z),
            3e-13,
        );
        assert_close(gamma_ratio(&[z + 1.0], &[z]), z, 3e-13);
        assert_eq!(reciprocal_gamma(Complex64::new(-4.0, 0.0)), Complex64::ZERO);
    }

    #[test]
    fn stabilized_difference_helpers_have_finite_zero_limits() {
        let z = Complex64::new(1.2, -0.4);
        assert_close(
            gamma_difference_ratio(z, Complex64::ZERO),
            digamma(z) * reciprocal_gamma(z),
            2e-14,
        );
        assert_close(
            pochhammer_difference_ratio(z, Complex64::ZERO, 3),
            3.0 * z * z + 6.0 * z + 2.0,
            2e-14,
        );
        assert_eq!(exponential_difference_ratio(z, Complex64::ZERO), z);
    }

    #[test]
    fn zero_argument_is_one() {
        let value = hyp2f1_scalar(
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(4.0, 0.5),
            Complex64::new(0.0, -0.0),
        );
        assert_eq!(value, Complex64::ONE);
    }

    #[test]
    fn length_mismatch_is_transactional() {
        let input = [Complex64::ONE];
        let mut out = [Complex64::new(7.0, 8.0); 2];
        assert_eq!(
            hyp2f1(&input, &input, &input, &input, &mut out),
            Err("Length mismatch")
        );
        assert_eq!(out, [Complex64::new(7.0, 8.0); 2]);
    }
}
