//! Gauss hypergeometric function with complex parameters and argument.
//!
//! The scalar implementation is an allocation-free polyalgorithm. It combines
//! the defining Gauss series with Euler and Pfaff transformations, stabilized
//! connection expansions about `z = 1` and `z = infinity`, and Taylor
//! continuation through the region in which none of those series converges
//! rapidly. The same scalar kernel backs the serial and Rayon-parallel array
//! interfaces.
//!
//! Complex powers use their principal values. Consequently, values on the
//! conventional branch cut `[1, infinity)` depend on the sign of `z.im`,
//! including signed zero. See [`hyp2f1_scalar`] for the complete public
//! contract and references.

use num_complex::Complex64;
use rayon::prelude::*;

use crate::{chunksize, macros::check_length};

const NAN: Complex64 = Complex64::new(f64::NAN, f64::NAN);
const REL_TOL: f64 = 8.0 * f64::EPSILON;
const DIRECT_RADIUS: f64 = 0.9;
const MAX_SERIES_ITERATIONS: usize = 10_000;
const MAX_TAYLOR_ITERATIONS: usize = 512;
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

/// Evaluates Gauss's hypergeometric function on its principal branch.
///
/// This computes
///
/// `2F1(a, b; c; z) = sum((a)_n (b)_n z^n / ((c)_n n!), n = 0..infinity)`
///
/// by selecting among the defining series, Euler/Pfaff transformations,
/// stabilized connection expansions about `z = 1` and `z = infinity`, and a
/// Taylor continuation fallback. All four arguments may be complex, and `a`
/// and `b` are interchangeable.
///
/// Complex powers take their principal values. On the conventional branch cut
/// `z` in `[1, infinity)`, positive and negative zero in `z.im` therefore select
/// the upper and lower limiting values, respectively.
///
/// A complex NaN is returned for non-finite inputs, mathematical singularities
/// (including a nonpositive-integer `c` unless the series terminates before its
/// pole), or failure of an internal expansion to converge within its limit.
///
/// # References
///
/// \[1\] N. Michel and M. V. Stoitsov, “Fast computation of the Gauss
///       hypergeometric function with all its parameters complex with
///       application to the Pöschl–Teller–Ginocchio potential wave functions,”
///       *Computer Physics Communications*, vol. 178, no. 7, pp. 535–551,
///       Apr. 2008, doi:
///       [10.1016/j.cpc.2007.11.007](https://doi.org/10.1016/j.cpc.2007.11.007).
///
/// \[2\] NIST Digital Library of Mathematical Functions, “§15.2 Definitions
///       and Analytical Properties,” NIST. Accessed: Aug. 17, 2026. \[Online\].
///       Available: <https://dlmf.nist.gov/15.2>
///
/// \[3\] JuliaMath, “HypergeometricFunctions.jl,” ver. 0.3.30, GitHub.
///       Accessed: Aug. 17, 2026. \[Online\]. Available:
///       <https://github.com/JuliaMath/HypergeometricFunctions.jl>
///
/// \[4\] SciPy Developers, “scipy.special.hyp2f1,” *SciPy API Reference*.
///       Accessed: Aug. 17, 2026. \[Online\]. Available:
///       <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp2f1.html>
#[inline]
pub fn hyp2f1_scalar(a: Complex64, b: Complex64, c: Complex64, z: Complex64) -> Complex64 {
    if !finite(a) || !finite(b) || !finite(c) || !finite(z) {
        return NAN;
    }
    if z == Complex64::ZERO || a == Complex64::ZERO || b == Complex64::ZERO {
        return Complex64::ONE;
    }

    let terminating_degree = match (negative_integer_degree(a), negative_integer_degree(b)) {
        (Some(a_degree), Some(b_degree)) => Some(a_degree.min(b_degree)),
        (Some(degree), None) | (None, Some(degree)) => Some(degree),
        (None, None) => None,
    };
    if let Some(degree) = terminating_degree {
        if let Some(pole_degree) = negative_integer_degree(c)
            && degree > pole_degree
        {
            return NAN;
        }
        let result = terminating_series(a, b, c, z, degree);
        return if result.converged { result.value } else { NAN };
    }
    if is_nonpositive_integer(c) {
        return NAN;
    }
    if z == Complex64::ONE {
        let balance = c - a - b;
        if balance.re <= 0.0 {
            return NAN;
        }
        return gamma_ratio(&[c, balance], &[c - a, c - b]);
    }
    if c == a {
        return (-b * complex_log1p(-z)).exp();
    }
    if c == b {
        return (-a * complex_log1p(-z)).exp();
    }
    let result = general_evaluation(a, b, c, z);
    if result.converged { result.value } else { NAN }
}

/// Evaluates Gauss's hypergeometric function elementwise on equal-length,
/// contiguous slices.
///
/// Each output element is `hyp2f1_scalar(a[i], b[i], c[i], z[i])`. See
/// [`hyp2f1_scalar`] for the mathematical definition, branch convention,
/// failure policy, implementation notes, and references.
///
/// Returns `Err("Length mismatch")` without modifying `out` if any input slice
/// has a different length from `out`.
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

/// Evaluates Gauss's hypergeometric function elementwise in parallel on
/// equal-length, contiguous slices.
///
/// Work is split into Rayon chunks and each chunk is evaluated by [`hyp2f1`].
/// See [`hyp2f1_scalar`] for the mathematical definition, branch convention,
/// failure policy, implementation notes, and references.
///
/// Returns `Err("Length mismatch")` without modifying `out` if any input slice
/// has a different length from `out`.
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

/// A value returned by one candidate expansion together with convergence
/// status and a coarse measure of cancellation in its partial sums.
///
/// Keeping failure information separate from the value lets the path selector
/// map all internal singularities and iteration-limit failures to the public
/// complex-NaN policy in one place.
#[derive(Clone, Copy, Debug)]
struct EvalOutcome {
    value: Complex64,
    converged: bool,
    cancellation_estimate: f64,
}

impl EvalOutcome {
    #[inline]
    const fn success(value: Complex64) -> Self {
        Self {
            value,
            converged: true,
            cancellation_estimate: 1.0,
        }
    }

    #[inline]
    const fn success_with_cancellation(value: Complex64, cancellation_estimate: f64) -> Self {
        Self {
            value,
            converged: true,
            cancellation_estimate,
        }
    }

    #[inline]
    const fn failure() -> Self {
        Self {
            value: NAN,
            converged: false,
            cancellation_estimate: f64::INFINITY,
        }
    }
}

// Branch-aware complex primitives. These small helpers preserve precision or
// signed-zero information that the straightforward formulas can lose.

#[inline]
fn finite(z: Complex64) -> bool {
    z.re.is_finite() && z.im.is_finite()
}

#[inline]
fn complex_abs(z: Complex64) -> f64 {
    z.re.hypot(z.im)
}

/// Computes `ln(1 + z)` without first rounding `1 + z` near the origin.
///
/// The `atan2` expression also preserves which side of the negative real axis
/// was approached, which is required for the branch cut of `hyp2f1`.
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

/// Computes `1 / z` with scaled complex division to reduce avoidable overflow
/// and underflow when the real and imaginary components have unlike sizes.
#[inline]
fn complex_inverse(z: Complex64) -> Complex64 {
    if z.re.abs() >= z.im.abs() {
        let ratio = z.im / z.re;
        let denominator = z.re + z.im * ratio;
        Complex64::new(1.0 / denominator, -ratio / denominator)
    } else {
        let ratio = z.re / z.im;
        let denominator = z.im + z.re * ratio;
        Complex64::new(ratio / denominator, -1.0 / denominator)
    }
}

// Private gamma machinery. The connection formulas need complex gamma,
// digamma, and differences of nearly equal reciprocal-gamma values; keeping
// them local avoids adding a second public special-function API.

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
fn sin_cos_pi_real(x: f64) -> (f64, f64) {
    let nearest = x.round();
    let remainder = x - nearest;
    let sign = if (nearest % 2.0).abs() == 1.0 {
        -1.0
    } else {
        1.0
    };
    let (sine, cosine) = (core::f64::consts::PI * remainder).sin_cos();
    (sign * sine, sign * cosine)
}

#[inline]
fn sin_pi(z: Complex64) -> Complex64 {
    let y = core::f64::consts::PI * z.im;
    let (sine, cosine) = sin_cos_pi_real(z.re);
    Complex64::new(sine * y.cosh(), cosine * y.sinh())
}

#[inline]
fn cos_pi(z: Complex64) -> Complex64 {
    let y = core::f64::consts::PI * z.im;
    let (sine, cosine) = sin_cos_pi_real(z.re);
    Complex64::new(cosine * y.cosh(), -sine * y.sinh())
}

fn log_sin_pi(z: Complex64) -> Complex64 {
    let y = core::f64::consts::PI * z.im;
    if y.abs() < 20.0 {
        return sin_pi(z).ln();
    }
    // For large imaginary parts, evaluating sinh/cosh before taking the log
    // would overflow. Use the leading exponential form directly instead.
    let (sine, cosine) = sin_cos_pi_real(z.re);
    Complex64::new(
        y.abs() - core::f64::consts::LN_2,
        (z.im.signum() * cosine).atan2(sine),
    )
}

#[inline]
fn cot_pi(z: Complex64) -> Complex64 {
    let two_x = 2.0 * core::f64::consts::PI * (z.re - z.re.round());
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

/// Principal complex log-gamma from a 15-term Lanczos approximation, with the
/// reflection formula used to move arguments out of the left half-plane.
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

/// Evaluates `(Gamma(z + epsilon) / Gamma(z) - 1) / epsilon` in a form that
/// remains finite and accurate as `epsilon` tends to zero.
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

/// Evaluates `(1/Gamma(z) - 1/Gamma(z + epsilon)) / epsilon`, including the
/// removable limits at gamma poles and at `epsilon = 0`.
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
    let pole_index = -z.re.round() as i32;
    if epsilon == Complex64::ZERO {
        if z.im == 0.0 && (0..order).contains(&pole_index) {
            let mut product = Complex64::ONE;
            for index in 0..order {
                if index != pole_index {
                    product *= z + index as f64;
                }
            }
            return product;
        }
        let mut reciprocal_sum = Complex64::ZERO;
        for index in 0..order {
            reciprocal_sum += Complex64::ONE / (z + index as f64);
        }
        return pochhammer(z, order) * reciprocal_sum;
    }
    if z.im == 0.0 && (0..order).contains(&pole_index) {
        let mut shifted_product = Complex64::ONE;
        let mut log_sum = Complex64::ZERO;
        for index in 0..order {
            if index != pole_index {
                shifted_product *= z + epsilon + index as f64;
                log_sum += complex_log1p(epsilon / (z + index as f64));
            }
        }
        return shifted_product + pochhammer(z, order) * complex_expm1(log_sum) / epsilon;
    }
    let mut log_sum = Complex64::ZERO;
    for index in 0..order {
        log_sum += complex_log1p(epsilon / (z + index as f64));
    }
    pochhammer(z, order) * complex_expm1(log_sum) / epsilon
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
fn integer_sign(power: i32) -> f64 {
    if power.rem_euclid(2) == 0 { 1.0 } else { -1.0 }
}

// Stable connection formula about z = 1. Writing c - a - b = m + epsilon
// separates the finite polynomial part from the infinite tail. The paired
// beta/gamma recurrence evaluates their cancellation before epsilon reaches
// machine precision, rather than subtracting two singular connection terms.

fn one_alpha_zero(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    m: i32,
    epsilon: Complex64,
) -> Complex64 {
    if epsilon == Complex64::ZERO {
        integer_sign(m) * gamma(Complex64::new(m as f64, 0.0)) * gamma(c)
            / (gamma(a + m as f64) * gamma(b + m as f64))
    } else {
        gamma(c)
            / (epsilon
                * gamma(1.0 - m as f64 - epsilon)
                * gamma(a + m as f64 + epsilon)
                * gamma(b + m as f64 + epsilon))
    }
}

fn one_beta_zero(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    w: Complex64,
    m: i32,
    epsilon: Complex64,
) -> Complex64 {
    let mf = m as f64;
    if complex_abs(epsilon) > 0.1 {
        return (pochhammer(a, m) * pochhammer(b, m)
            / (gamma(1.0 - epsilon)
                * gamma(a + mf + epsilon)
                * gamma(b + mf + epsilon)
                * gamma(Complex64::new(mf + 1.0, 0.0)))
            - complex_pow(w, epsilon) / (gamma(a) * gamma(b) * gamma(mf + 1.0 + epsilon)))
            * gamma(c)
            * complex_pow(w, Complex64::new(mf, 0.0))
            / epsilon;
    }
    ((gamma_difference_ratio(Complex64::ONE, -epsilon) / gamma(Complex64::new(mf + 1.0, 0.0))
        + gamma_difference_ratio(Complex64::new(mf + 1.0, 0.0), epsilon))
        / (gamma(a + mf + epsilon) * gamma(b + mf + epsilon))
        - (gamma_difference_ratio(a + mf, epsilon) / gamma(b + mf + epsilon)
            + gamma_difference_ratio(b + mf, epsilon) / gamma(a + mf))
            / gamma(mf + 1.0 + epsilon)
        - exponential_difference_ratio(w.ln(), epsilon)
            / (gamma(a + mf) * gamma(b + mf) * gamma(mf + 1.0 + epsilon)))
        * gamma(c)
        * pochhammer(a, m)
        * pochhammer(b, m)
        * complex_pow(w, Complex64::new(mf, 0.0))
}

fn one_gamma_zero(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    w: Complex64,
    m: i32,
    epsilon: Complex64,
) -> Complex64 {
    let mf = m as f64;
    gamma(c) * pochhammer(a, m) * pochhammer(b, m) * complex_pow(w, Complex64::new(mf, 0.0))
        / (gamma(a + mf + epsilon)
            * gamma(b + mf + epsilon)
            * gamma(Complex64::new(mf + 1.0, 0.0))
            * gamma(1.0 - epsilon))
}

fn one_finite_part(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    w: Complex64,
    m: i32,
    epsilon: Complex64,
) -> EvalOutcome {
    if m <= 0 {
        return EvalOutcome::success(Complex64::ZERO);
    }
    let mut term = one_alpha_zero(a, b, c, m, epsilon);
    let mut sum = CompensatedSum::new(term);
    for n in 0..(m - 1) {
        let nf = n as f64;
        let denominator = (nf + 1.0) * (1.0 - m as f64 - epsilon + nf);
        if denominator == Complex64::ZERO {
            return EvalOutcome::failure();
        }
        term *= (a + nf) * (b + nf) * w / denominator;
        sum.add(term);
        if !finite(term) || !finite(sum.value()) {
            return EvalOutcome::failure();
        }
    }
    EvalOutcome::success(sum.value())
}

/// Evaluates the stabilized infinite tail of the connection expansion at
/// `w = 1 - z` using coupled beta and gamma recurrences.
fn one_infinite_part(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    w: Complex64,
    m: i32,
    epsilon: Complex64,
) -> EvalOutcome {
    let mf = m as f64;
    let mut beta = one_beta_zero(a, b, c, w, m, epsilon);
    let mut gamma_term = one_gamma_zero(a, b, c, w, m, epsilon) * w;
    let mut sum = CompensatedSum::new(beta);
    let mut small_terms = 0;
    for n in 0..MAX_SERIES_ITERATIONS {
        let nf = n as f64;
        let amn = a + mf + nf;
        let bmn = b + mf + nf;
        let shifted_a = amn + epsilon;
        let shifted_b = bmn + epsilon;
        let denominator = (mf + nf + 1.0 + epsilon) * (nf + 1.0);
        let correction_denominator = (mf + nf + 1.0 + epsilon) * (nf + 1.0 - epsilon);
        if denominator == Complex64::ZERO || correction_denominator == Complex64::ZERO {
            return EvalOutcome::failure();
        }
        beta = shifted_a * shifted_b * w * beta / denominator
            + (amn * bmn / (mf + nf + 1.0) - amn - bmn - epsilon
                + shifted_a * shifted_b / (nf + 1.0))
                * gamma_term
                / correction_denominator;
        sum.add(beta);
        gamma_term *= amn * bmn * w / ((mf + nf + 1.0) * (nf + 1.0 - epsilon));
        if !finite(beta) || !finite(gamma_term) || !finite(sum.value()) {
            return EvalOutcome::failure();
        }
        let sum_abs = complex_abs(sum.value());
        if sum_abs > 0.0 && complex_abs(beta) <= REL_TOL * sum_abs {
            small_terms += 1;
            if small_terms >= 2 {
                return EvalOutcome::success(sum.value());
            }
        } else {
            small_terms = 0;
        }
    }
    EvalOutcome::failure()
}

fn one_expansion(a: Complex64, b: Complex64, c: Complex64, z: Complex64) -> EvalOutcome {
    let (m, epsilon) = nearest_integer_difference(c - a - b);
    if m < 0 {
        return EvalOutcome::failure();
    }
    let w = Complex64::ONE - z;
    let finite_part = one_finite_part(a, b, c, w, m, epsilon);
    let infinite_part = one_infinite_part(a, b, c, w, m, epsilon);
    if !finite_part.converged || !infinite_part.converged {
        return EvalOutcome::failure();
    }
    let value = integer_sign(m) * (finite_part.value + infinite_part.value) / sinc_pi(epsilon);
    if finite(value) {
        EvalOutcome::success(value)
    } else {
        EvalOutcome::failure()
    }
}

// Stable connection formula about z = infinity. This has the same
// m + epsilon construction as the z = 1 expansion, now for b - a, and is
// evaluated in the reciprocal coordinate w = 1 / z.

fn infinity_alpha_zero(a: Complex64, c: Complex64, m: i32, epsilon: Complex64) -> Complex64 {
    if epsilon == Complex64::ZERO {
        integer_sign(m) * gamma(Complex64::new(m as f64, 0.0)) * gamma(c)
            / (gamma(a + m as f64) * gamma(c - a))
    } else {
        gamma(c)
            / (epsilon
                * gamma(1.0 - m as f64 - epsilon)
                * gamma(a + m as f64 + epsilon)
                * gamma(c - a))
    }
}

fn infinity_beta_zero(
    a: Complex64,
    c: Complex64,
    w: Complex64,
    m: i32,
    epsilon: Complex64,
) -> Complex64 {
    let mf = m as f64;
    let d = 1.0 - c + a;
    if complex_abs(epsilon) > 0.1 {
        return (pochhammer(a, m) * pochhammer(d, m)
            / (gamma(1.0 - epsilon)
                * gamma(a + mf + epsilon)
                * gamma(c - a)
                * gamma(Complex64::new(mf + 1.0, 0.0)))
            - complex_pow(-w, epsilon) * pochhammer(d + epsilon, m)
                / (gamma(a) * gamma(c - a - epsilon) * gamma(mf + 1.0 + epsilon)))
            * gamma(c)
            * complex_pow(w, Complex64::new(mf, 0.0))
            / epsilon;
    }
    ((pochhammer(d + epsilon, m) * gamma_difference_ratio(Complex64::ONE, -epsilon)
        - pochhammer_difference_ratio(d, epsilon, m) / gamma(1.0 - epsilon))
        / (gamma(c - a) * gamma(a + mf + epsilon) * gamma(Complex64::new(mf + 1.0, 0.0)))
        + pochhammer(d + epsilon, m)
            * ((gamma_difference_ratio(Complex64::new(mf + 1.0, 0.0), epsilon)
                / gamma(a + mf + epsilon)
                - gamma_difference_ratio(a + mf, epsilon) / gamma(mf + 1.0 + epsilon))
                / gamma(c - a)
                - (gamma_difference_ratio(c - a, -epsilon)
                    - exponential_difference_ratio(-(-w).ln(), -epsilon) / gamma(c - a - epsilon))
                    / (gamma(mf + 1.0 + epsilon) * gamma(a + mf))))
        * gamma(c)
        * pochhammer(a, m)
        * complex_pow(w, Complex64::new(mf, 0.0))
}

fn infinity_gamma_zero(
    a: Complex64,
    c: Complex64,
    w: Complex64,
    m: i32,
    epsilon: Complex64,
) -> Complex64 {
    let mf = m as f64;
    gamma(c)
        * pochhammer(a, m)
        * pochhammer(1.0 - c + a, m)
        * complex_pow(w, Complex64::new(mf, 0.0))
        / (gamma(a + mf + epsilon)
            * gamma(c - a)
            * gamma(Complex64::new(mf + 1.0, 0.0))
            * gamma(1.0 - epsilon))
}

fn infinity_finite_part(
    a: Complex64,
    c: Complex64,
    w: Complex64,
    m: i32,
    epsilon: Complex64,
) -> EvalOutcome {
    if m <= 0 {
        return EvalOutcome::success(Complex64::ZERO);
    }
    let mut term = infinity_alpha_zero(a, c, m, epsilon);
    let mut sum = CompensatedSum::new(term);
    for n in 0..(m - 1) {
        let nf = n as f64;
        let denominator = (nf + 1.0) * (1.0 - m as f64 - epsilon + nf);
        if denominator == Complex64::ZERO {
            return EvalOutcome::failure();
        }
        term *= (a + nf) * (1.0 - c + a + nf) * w / denominator;
        sum.add(term);
        if !finite(term) || !finite(sum.value()) {
            return EvalOutcome::failure();
        }
    }
    EvalOutcome::success(sum.value())
}

fn infinity_infinite_part(
    a: Complex64,
    c: Complex64,
    w: Complex64,
    m: i32,
    epsilon: Complex64,
) -> EvalOutcome {
    let mf = m as f64;
    let mut beta = infinity_beta_zero(a, c, w, m, epsilon);
    let mut gamma_term = infinity_gamma_zero(a, c, w, m, epsilon) * w;
    let mut sum = CompensatedSum::new(beta);
    let mut small_terms = 0;
    for n in 0..MAX_SERIES_ITERATIONS {
        let nf = n as f64;
        let amn = a + mf + nf;
        let dmn = 1.0 - c + a + mf + nf;
        let shifted_a = amn + epsilon;
        let shifted_d = dmn + epsilon;
        let denominator = (mf + nf + 1.0 + epsilon) * (nf + 1.0);
        let correction_denominator = (mf + nf + 1.0 + epsilon) * (nf + 1.0 - epsilon);
        if denominator == Complex64::ZERO || correction_denominator == Complex64::ZERO {
            return EvalOutcome::failure();
        }
        beta = shifted_a * shifted_d * w * beta / denominator
            + (amn * dmn / (mf + nf + 1.0) - amn - dmn - epsilon
                + shifted_a * shifted_d / (nf + 1.0))
                * gamma_term
                / correction_denominator;
        sum.add(beta);
        gamma_term *= amn * dmn * w / ((mf + nf + 1.0) * (nf + 1.0 - epsilon));
        if !finite(beta) || !finite(gamma_term) || !finite(sum.value()) {
            return EvalOutcome::failure();
        }
        let sum_abs = complex_abs(sum.value());
        if sum_abs > 0.0 && complex_abs(beta) <= REL_TOL * sum_abs {
            small_terms += 1;
            if small_terms >= 2 {
                return EvalOutcome::success(sum.value());
            }
        } else {
            small_terms = 0;
        }
    }
    EvalOutcome::failure()
}

/// Evaluates the reciprocal-coordinate connection expansion, swapping `a`
/// and `b` first so that the integer part of `b - a` is nonnegative.
fn infinity_expansion(
    mut a: Complex64,
    mut b: Complex64,
    c: Complex64,
    z: Complex64,
) -> EvalOutcome {
    if (b - a).re < 0.0 {
        core::mem::swap(&mut a, &mut b);
    }
    let (m, epsilon) = nearest_integer_difference(b - a);
    if m < 0 {
        return EvalOutcome::failure();
    }
    let w = complex_inverse(z);
    let finite_part = infinity_finite_part(a, c, w, m, epsilon);
    let infinite_part = infinity_infinite_part(a, c, w, m, epsilon);
    if !finite_part.converged || !infinite_part.converged {
        return EvalOutcome::failure();
    }
    let value = integer_sign(m) * complex_pow(-w, a) * (finite_part.value + infinite_part.value)
        / sinc_pi(epsilon);
    if finite(value) {
        EvalOutcome::success(value)
    } else {
        EvalOutcome::failure()
    }
}

// Direct and terminating series use compensated addition because intermediate
// terms can be much larger than the final answer for complex parameters.

/// Kahan-style compensated accumulation applied componentwise by complex
/// arithmetic.
#[derive(Clone, Copy, Debug)]
struct CompensatedSum {
    sum: Complex64,
    correction: Complex64,
}

impl CompensatedSum {
    #[inline]
    const fn new(value: Complex64) -> Self {
        Self {
            sum: value,
            correction: Complex64::ZERO,
        }
    }

    #[inline]
    fn add(&mut self, value: Complex64) {
        let adjusted = value - self.correction;
        let next = self.sum + adjusted;
        self.correction = (next - self.sum) - adjusted;
        self.sum = next;
    }

    #[inline]
    fn value(self) -> Complex64 {
        self.sum
    }
}

#[inline]
fn negative_integer_degree(z: Complex64) -> Option<usize> {
    if is_nonpositive_integer(z) && -z.re <= usize::MAX as f64 {
        Some((-z.re) as usize)
    } else {
        None
    }
}

fn terminating_series(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    z: Complex64,
    degree: usize,
) -> EvalOutcome {
    let mut term = Complex64::ONE;
    let mut sum = CompensatedSum::new(Complex64::ONE);
    for n in 0..degree {
        let nf = n as f64;
        let denominator = (c + nf) * (nf + 1.0);
        if denominator == Complex64::ZERO {
            return EvalOutcome::failure();
        }
        term *= (a + nf) * (b + nf) * z / denominator;
        sum.add(term);
        if !finite(term) || !finite(sum.value()) {
            return EvalOutcome::failure();
        }
    }
    EvalOutcome::success(sum.value())
}

/// Sums the defining Gauss series and records the largest-term/final-value
/// ratio as an inexpensive warning that the result suffered cancellation.
#[inline]
fn direct_series_with_limit(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    z: Complex64,
    iteration_limit: usize,
) -> EvalOutcome {
    let mut term = Complex64::ONE;
    let mut sum = CompensatedSum::new(Complex64::ONE);
    let mut small_terms = 0;
    let mut largest_term: f64 = 1.0;

    for n in 0..iteration_limit {
        let nf = n as f64;
        let denominator = (c + nf) * (nf + 1.0);
        if denominator == Complex64::ZERO {
            return EvalOutcome::failure();
        }
        term *= (a + nf) * (b + nf) * z / denominator;
        sum.add(term);
        let value = sum.value();
        let term_abs = complex_abs(term);
        largest_term = largest_term.max(term_abs);
        if !finite(term) || !finite(value) {
            return EvalOutcome::failure();
        }

        let value_abs = complex_abs(value);
        let converged =
            term == Complex64::ZERO || (value_abs > 0.0 && term_abs <= REL_TOL * value_abs);
        if converged {
            small_terms += 1;
            if small_terms >= 2 && n >= 1 {
                let cancellation = largest_term / value_abs.max(f64::MIN_POSITIVE);
                return EvalOutcome::success_with_cancellation(value, cancellation);
            }
        } else {
            small_terms = 0;
        }
    }
    EvalOutcome::failure()
}

#[inline]
fn direct_series(a: Complex64, b: Complex64, c: Complex64, z: Complex64) -> EvalOutcome {
    direct_series_with_limit(a, b, c, z, MAX_SERIES_ITERATIONS)
}

// Region selection and Taylor continuation. Each transformed candidate is
// scored by the modulus of its local series variable; the Taylor path fills
// the compact gap left when every candidate lies outside DIRECT_RADIUS.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvalPath {
    Direct,
    PfaffDirect,
    Infinity,
    PfaffInfinity,
    One,
    PfaffOne,
    Taylor,
}

fn select_path(z: Complex64) -> EvalPath {
    let one_minus_z = Complex64::ONE - z;
    let inverse_z = complex_inverse(z);
    let pfaff_z = Complex64::ONE + complex_inverse(z - 1.0);
    // The ordering is also the deterministic tie-break policy at region
    // boundaries, which prevents small roundoff changes from switching paths.
    let candidates = [
        (EvalPath::Direct, complex_abs(z)),
        (EvalPath::PfaffDirect, complex_abs(pfaff_z)),
        (EvalPath::Infinity, complex_abs(inverse_z)),
        (EvalPath::PfaffInfinity, complex_abs(1.0 - inverse_z)),
        (EvalPath::One, complex_abs(one_minus_z)),
        (
            EvalPath::PfaffOne,
            complex_abs(complex_inverse(one_minus_z)),
        ),
    ];
    let mut selected = EvalPath::Taylor;
    let mut selected_modulus = f64::INFINITY;
    for (path, modulus) in candidates {
        if modulus <= DIRECT_RADIUS {
            let tie = 32.0 * f64::EPSILON * selected_modulus.max(modulus).max(1.0);
            if selected == EvalPath::Taylor || modulus + tie < selected_modulus {
                selected = path;
                selected_modulus = modulus;
            }
        }
    }
    selected
}

fn forced_anchor_evaluation(a: Complex64, b: Complex64, c: Complex64, z: Complex64) -> EvalOutcome {
    // Anchors deliberately bypass select_path so Taylor fallback cannot recurse
    // back into itself.
    if complex_abs(z) <= 1.0 {
        direct_series(a, b, c, z)
    } else {
        infinity_expansion(a, b, c, z)
    }
}

fn taylor_continuation_with_limit(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    z: Complex64,
    iteration_limit: usize,
) -> EvalOutcome {
    let z_abs = complex_abs(z);
    if z_abs == 0.0 {
        return EvalOutcome::success(Complex64::ONE);
    }
    let anchor_radius = if z_abs < 1.0 { 0.9 } else { 1.1 };
    let z0 = (anchor_radius / z_abs) * z;
    let q0_outcome = forced_anchor_evaluation(a, b, c, z0);
    let shifted_outcome = forced_anchor_evaluation(a + 1.0, b + 1.0, c + 1.0, z0);
    if !q0_outcome.converged || !shifted_outcome.converged || c == Complex64::ZERO {
        return EvalOutcome::failure();
    }

    // q0 and q1 are the function and its first derivative at z0. Higher
    // derivatives follow from the hypergeometric differential equation.
    let mut q0 = q0_outcome.value;
    let mut q1 = a * b * shifted_outcome.value / c;
    let delta = z - z0;
    let mut delta_power = delta;
    let mut sum = CompensatedSum::new(q0);
    sum.add(q1 * delta);
    let differential_denominator = z0 * (1.0 - z0);
    if differential_denominator == Complex64::ZERO || !finite(sum.value()) {
        return EvalOutcome::failure();
    }

    let mut small_terms = 0;
    let mut largest_term = complex_abs(q0).max(complex_abs(q1 * delta));
    for n in 0..iteration_limit {
        let nf = n as f64;
        let q2 = ((nf * (2.0 * z0 - 1.0) - c + (a + b + 1.0) * z0) * q1
            + (a + nf) * (b + nf) * q0 / (nf + 1.0))
            / (differential_denominator * (nf + 2.0));
        delta_power *= delta;
        let term = q2 * delta_power;
        sum.add(term);
        let value = sum.value();
        if !finite(q2) || !finite(term) || !finite(value) {
            return EvalOutcome::failure();
        }
        let term_abs = complex_abs(term);
        let value_abs = complex_abs(value);
        largest_term = largest_term.max(term_abs);
        if value_abs > 0.0 && term_abs <= REL_TOL * value_abs {
            small_terms += 1;
            if small_terms >= 2 {
                return EvalOutcome::success_with_cancellation(
                    value,
                    largest_term / value_abs.max(f64::MIN_POSITIVE),
                );
            }
        } else {
            small_terms = 0;
        }
        q0 = q1;
        q1 = q2;
    }
    EvalOutcome::failure()
}

#[inline]
fn taylor_continuation(a: Complex64, b: Complex64, c: Complex64, z: Complex64) -> EvalOutcome {
    taylor_continuation_with_limit(a, b, c, z, MAX_TAYLOR_ITERATIONS)
}

fn general_evaluation(
    mut a: Complex64,
    mut b: Complex64,
    c: Complex64,
    z: Complex64,
) -> EvalOutcome {
    // Euler's transformation makes Re(c-a-b) nonnegative; symmetry in a and b
    // then gives the stable parameter ordering assumed by the expansions.
    let mut outer_prefactor = Complex64::ONE;
    let balance = c - a - b;
    if balance.re < 0.0 {
        outer_prefactor = (balance * complex_log1p(-z)).exp();
        a = c - a;
        b = c - b;
    }
    if (b - a).re < 0.0 {
        core::mem::swap(&mut a, &mut b);
    }

    let path = select_path(z);
    let pfaff_z = Complex64::ONE + complex_inverse(z - 1.0);
    let pfaff_prefactor = (-a * complex_log1p(-z)).exp();
    let result = match path {
        EvalPath::Direct => direct_series(a, b, c, z),
        EvalPath::PfaffDirect => direct_series(a, c - b, c, pfaff_z),
        EvalPath::Infinity => infinity_expansion(a, b, c, z),
        EvalPath::PfaffInfinity => infinity_expansion(a, c - b, c, pfaff_z),
        EvalPath::One => one_expansion(a, b, c, z),
        EvalPath::PfaffOne => one_expansion(a, c - b, c, pfaff_z),
        EvalPath::Taylor => taylor_continuation(a, b, c, z),
    };
    if !result.converged {
        return result;
    }
    let inner_prefactor = if matches!(
        path,
        EvalPath::PfaffDirect | EvalPath::PfaffInfinity | EvalPath::PfaffOne
    ) {
        pfaff_prefactor
    } else {
        Complex64::ONE
    };
    let value = outer_prefactor * inner_prefactor * result.value;
    if finite(value) {
        EvalOutcome::success_with_cancellation(value, result.cancellation_estimate)
    } else {
        EvalOutcome::failure()
    }
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

    #[derive(Debug)]
    struct HypFixtureRow<'a> {
        a: Complex64,
        b: Complex64,
        c: Complex64,
        z: Complex64,
        expected: Complex64,
        tolerance: f64,
        label: &'a str,
    }

    fn parse_hyp_fixture() -> Vec<HypFixtureRow<'static>> {
        let fixture = include_str!("../../test/data/hyp2f1_reference.csv");
        let mut rows = Vec::new();
        for (line_index, line) in fixture.lines().enumerate() {
            if line.starts_with('#') || line_index == 1 || line.is_empty() {
                continue;
            }
            let fields: Vec<_> = line.split(',').collect();
            assert_eq!(
                fields.len(),
                12,
                "hyp2f1_reference.csv row {} has wrong field count",
                line_index + 1
            );
            let mut values = [0.0; 11];
            for (field_index, field) in fields[..11].iter().enumerate() {
                values[field_index] = field.parse().unwrap_or_else(|error| {
                    panic!(
                        "hyp2f1_reference.csv row {}, field {} is not f64: {error}",
                        line_index + 1,
                        field_index + 1
                    )
                });
            }
            rows.push(HypFixtureRow {
                a: Complex64::new(values[0], values[1]),
                b: Complex64::new(values[2], values[3]),
                c: Complex64::new(values[4], values[5]),
                z: Complex64::new(values[6], values[7]),
                expected: Complex64::new(values[8], values[9]),
                tolerance: values[10],
                label: fields[11],
            });
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
        let tiny = Complex64::new(0.0, 1e-9);
        assert_close(
            gamma_difference_ratio(Complex64::ONE, -tiny),
            Complex64::new(-0.577_215_664_901_532_9, -6.558_780_715_202_539e-10),
            2e-14,
        );
        assert_close(
            gamma_difference_ratio(Complex64::new(3.0, 0.0), tiny),
            Complex64::new(0.461_392_167_549_233_57, -1.141_492_155_637_234e-10),
            2e-14,
        );
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
    fn direct_and_polynomial_fixtures_match_reference() {
        for row in parse_hyp_fixture()
            .into_iter()
            .filter(|row| matches!(row.label, "direct" | "polynomial"))
        {
            assert_close(
                hyp2f1_scalar(row.a, row.b, row.c, row.z),
                row.expected,
                row.tolerance,
            );
        }
    }

    #[test]
    fn transformed_fixtures_match_reference() {
        for row in parse_hyp_fixture().into_iter().filter(|row| {
            matches!(
                row.label,
                "pfaff"
                    | "infinity"
                    | "infinity-equal-m0"
                    | "one"
                    | "one-balanced-m0"
                    | "one-near-integer"
                    | "infinity-near-integer"
                    | "euler"
                    | "moderate"
                    | "scipy-1561"
                    | "upper-cut"
                    | "lower-cut"
            )
        }) {
            assert_close(
                hyp2f1_scalar(row.a, row.b, row.c, row.z),
                row.expected,
                row.tolerance,
            );
        }
    }

    #[test]
    fn taylor_gap_fixtures_match_reference() {
        for row in parse_hyp_fixture()
            .into_iter()
            .filter(|row| row.label == "taylor")
        {
            assert_eq!(select_path(row.z), EvalPath::Taylor);
            assert_close(
                hyp2f1_scalar(row.a, row.b, row.c, row.z),
                row.expected,
                row.tolerance,
            );
        }
    }

    #[test]
    fn taylor_cap_and_differential_equation_residual() {
        let a = Complex64::new(0.7, 0.2);
        let b = Complex64::new(1.2, -0.3);
        let c = Complex64::new(2.1, 0.1);
        let z = Complex64::new(0.5, 0.866_025_403_784_438_6);
        assert!(!taylor_continuation_with_limit(a, b, c, z, 1).converged);

        let step = 1e-4;
        let center = hyp2f1_scalar(a, b, c, z);
        let plus = hyp2f1_scalar(a, b, c, z + step);
        let minus = hyp2f1_scalar(a, b, c, z - step);
        let first = (plus - minus) / (2.0 * step);
        let second = (plus - 2.0 * center + minus) / (step * step);
        let residual = z * (1.0 - z) * second + (c - (a + b + 1.0) * z) * first - a * b * center;
        let scale = complex_abs(a * b * center).max(1.0);
        assert!(
            complex_abs(residual) <= 5e-7 * scale,
            "residual={residual:?}"
        );
    }

    #[test]
    fn every_transformed_selector_path_is_reachable() {
        assert_eq!(select_path(Complex64::new(0.2, 0.1)), EvalPath::Direct);
        assert_eq!(
            select_path(Complex64::new(-0.5, 0.1)),
            EvalPath::PfaffDirect
        );
        assert_eq!(select_path(Complex64::new(3.0, 4.0)), EvalPath::Infinity);
        assert_eq!(
            select_path(Complex64::new(1.2, 0.4)),
            EvalPath::PfaffInfinity
        );
        assert_eq!(select_path(Complex64::new(0.95, 0.05)), EvalPath::One);
        assert_eq!(select_path(Complex64::new(-3.0, 0.4)), EvalPath::PfaffOne);
    }

    #[test]
    fn symmetry_euler_and_pfaff_identities_agree() {
        let a = Complex64::new(0.4, 0.2);
        let b = Complex64::new(0.9, -0.1);
        let c = Complex64::new(2.4, 0.3);
        let z = Complex64::new(-0.7, 0.2);
        let value = hyp2f1_scalar(a, b, c, z);
        assert_close(hyp2f1_scalar(b, a, c, z), value, 5e-13);

        let euler = ((c - a - b) * complex_log1p(-z)).exp() * hyp2f1_scalar(c - a, c - b, c, z);
        assert_close(euler, value, 2e-12);

        let transformed_z = Complex64::ONE + complex_inverse(z - 1.0);
        let pfaff = (-a * complex_log1p(-z)).exp() * hyp2f1_scalar(a, c - b, c, transformed_z);
        assert_close(pfaff, value, 2e-12);
    }

    #[test]
    fn exceptional_cases_follow_documented_precedence() {
        let a = Complex64::new(1.2, 0.3);
        let b = Complex64::new(0.7, -0.2);
        assert_eq!(hyp2f1_scalar(a, b, a, Complex64::ZERO), Complex64::ONE);
        assert_close(
            hyp2f1_scalar(a, b, a, Complex64::new(0.2, -0.1)),
            (-b * complex_log1p(Complex64::new(-0.2, 0.1))).exp(),
            2e-14,
        );

        let valid = hyp2f1_scalar(
            Complex64::new(-2.0, 0.0),
            b,
            Complex64::new(-2.0, 0.0),
            Complex64::new(2.0, 0.5),
        );
        assert!(finite(valid));
        let invalid = hyp2f1_scalar(
            Complex64::new(-3.0, 0.0),
            b,
            Complex64::new(-2.0, 0.0),
            Complex64::new(0.2, 0.0),
        );
        assert!(invalid.re.is_nan() && invalid.im.is_nan());

        let non_finite = hyp2f1_scalar(Complex64::new(f64::INFINITY, 0.0), b, a, Complex64::ZERO);
        assert!(non_finite.re.is_nan() && non_finite.im.is_nan());
    }

    #[test]
    fn argument_unity_uses_gauss_ratio() {
        let a = Complex64::new(0.2, 0.1);
        let b = Complex64::new(0.3, -0.2);
        let c = Complex64::new(2.0, 0.4);
        let expected = gamma_ratio(&[c, c - a - b], &[c - a, c - b]);
        assert_close(hyp2f1_scalar(a, b, c, Complex64::ONE), expected, 2e-14);
        let divergent = hyp2f1_scalar(a, b, a + b, Complex64::ONE);
        assert!(divergent.re.is_nan() && divergent.im.is_nan());
    }

    #[test]
    fn direct_series_iteration_cap_reports_failure() {
        let result = direct_series_with_limit(
            Complex64::new(0.7, 0.2),
            Complex64::new(1.1, -0.3),
            Complex64::new(2.4, 0.1),
            Complex64::new(0.8, 0.1),
            1,
        );
        assert!(!result.converged);
        assert!(result.value.re.is_nan() && result.value.im.is_nan());
        assert!(result.cancellation_estimate.is_infinite());
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

        assert_eq!(
            hyp2f1_par(&input, &input, &input, &input, &mut out),
            Err("Length mismatch")
        );
        assert_eq!(out, [Complex64::new(7.0, 8.0); 2]);
    }

    #[test]
    fn vector_apis_match_scalar_across_regions() {
        let rows: Vec<_> = parse_hyp_fixture().into_iter().take(12).collect();
        let a: Vec<_> = rows.iter().map(|row| row.a).collect();
        let b: Vec<_> = rows.iter().map(|row| row.b).collect();
        let c: Vec<_> = rows.iter().map(|row| row.c).collect();
        let z: Vec<_> = rows.iter().map(|row| row.z).collect();
        let expected: Vec<_> = (0..rows.len())
            .map(|index| hyp2f1_scalar(a[index], b[index], c[index], z[index]))
            .collect();
        let mut serial = vec![NAN; rows.len()];
        let mut parallel = vec![NAN; rows.len()];
        hyp2f1(&a, &b, &c, &z, &mut serial).unwrap();
        hyp2f1_par(&a, &b, &c, &z, &mut parallel).unwrap();
        assert_eq!(serial, expected);
        assert_eq!(parallel, expected);
    }

    #[test]
    fn vector_apis_cover_empty_singleton_subslice_and_nan() {
        let empty: [Complex64; 0] = [];
        let mut empty_out = [];
        assert_eq!(
            hyp2f1(&empty, &empty, &empty, &empty, &mut empty_out),
            Ok(())
        );
        assert_eq!(
            hyp2f1_par(&empty, &empty, &empty, &empty, &mut empty_out),
            Ok(())
        );

        let a = [Complex64::new(0.5, 0.2); 3];
        let b = [Complex64::new(1.1, -0.3); 3];
        let c = [Complex64::new(2.4, 0.1); 3];
        let z = [
            Complex64::new(99.0, 0.0),
            Complex64::new(0.3, 0.2),
            Complex64::new(f64::NAN, 0.0),
        ];
        let mut out = [Complex64::new(7.0, 8.0); 3];
        hyp2f1(&a[1..], &b[1..], &c[1..], &z[1..], &mut out[1..]).unwrap();
        assert_eq!(out[0], Complex64::new(7.0, 8.0));
        assert_eq!(out[1], hyp2f1_scalar(a[1], b[1], c[1], z[1]));
        assert!(out[2].re.is_nan() && out[2].im.is_nan());

        let mut singleton = [NAN];
        hyp2f1_par(&a[1..2], &b[1..2], &c[1..2], &z[1..2], &mut singleton).unwrap();
        assert_eq!(singleton[0], out[1]);
    }
}
