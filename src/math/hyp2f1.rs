//! Gauss hypergeometric function with complex parameters and argument.

use num_complex::Complex64;
use rayon::prelude::*;

use crate::{chunksize, macros::check_length};

const NAN: Complex64 = Complex64::new(f64::NAN, f64::NAN);
const REL_TOL: f64 = 8.0 * f64::EPSILON;
const MAX_SERIES_ITERATIONS: usize = 10_000;

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
