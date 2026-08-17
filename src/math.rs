//! Pure-math functions supporting physics calculations.

mod hyp2f1;

pub use hyp2f1::{hyp2f1, hyp2f1_par, hyp2f1_scalar};

use core::ops::{Add, Div, Mul, Sub};
use num_traits::{Float, FromPrimitive};

/// Scalar type shared by generic math, mesh, field, and FEM infrastructure.
/// Floating-point operations, including [`Float::min`] and [`Float::max`], are
/// exposed through the [`Float`] supertrait.
pub trait Scalar:
    Float
    + FromPrimitive
    + Copy
    + Clone
    + Default
    + std::fmt::Debug
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Send
    + Sync
    + 'static
{
    const ZERO: Self;
    const ONE: Self;

    /// Return the smallest positive finite value for this scalar type.
    fn min_positive() -> Self;
    /// Convert this scalar value to `f64` for diagnostics and tolerances.
    fn to_f64(self) -> f64;
}

/// Cast a finite `f64` literal into the active scalar type.
#[inline]
pub(crate) fn cast<T: Scalar>(value: f64) -> T {
    T::from_f64(value).expect("finite f64 literal should cast to target scalar")
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    /// Return the smallest positive finite value for this scalar type.
    fn min_positive() -> Self {
        f32::MIN_POSITIVE
    }

    #[inline]
    /// Convert this scalar value to `f64` for diagnostics and tolerances.
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    /// Return the smallest positive finite value for this scalar type.
    fn min_positive() -> Self {
        f64::MIN_POSITIVE
    }

    #[inline]
    /// Convert this scalar value to `f64` for diagnostics and tolerances.
    fn to_f64(self) -> f64 {
        self
    }
}

// Curvefit coeffs for elliptic integrals
const ELLIPK_A: [f64; 5] = [
    1.38629436112,
    0.09666344259,
    0.03590092393,
    0.03742563713,
    0.01451196212,
];
const ELLIPK_B: [f64; 5] = [
    0.5,
    0.12498593597,
    0.06880248576,
    0.03328355346,
    0.00441787012,
];
const ELLIPE_A: [f64; 5] = [
    1.0,
    0.44325141463,
    0.06260601220,
    0.04757383546,
    0.01736506451,
];
const ELLIPE_B: [f64; 5] = [
    0.0,
    0.24998368310,
    0.09200180037,
    0.04069697526,
    0.00526449639,
];

/// Complete elliptic integral of the first kind.
///
/// Mirrors scipy's implementation using a blended 10th order polynomial fit from handbook section 17.3.34.
/// Scipy uses (1-m) as the parameter compared to the handbook's definition.
///
/// Per handbook, max absolute error is 2e-8.
///
/// # References
///
///    \[1\] M. Abramowitz and I. A. Stegun, *Handbook of mathematical functions: with formulas, graphs, and mathematical tables*. 1970.
#[inline]
pub fn ellipk(m: f64) -> f64 {
    let mut ellip: f64 = 0.0;
    let c: f64 = 1.0 - m;
    let logterm = c.powi(-1).ln();

    // NOTE: This loop is unrolled at compile-time automatically,
    // and the repeated calls to `powi` are de-duplicated by the compiler.
    for i in 0..5 {
        ellip = logterm
            .mul_add(ELLIPK_B[i], ELLIPK_A[i])
            .mul_add(c.powi(i as i32), ellip);
    }

    ellip
}

/// Complete elliptic integral of the second kind.
///
/// Mirrors scipy's implementation using a blended 10th order polynomial fit from handbook section 17.3.36.
/// Scipy uses (1-m) as the parameter compared to the handbook's definition.
///
/// Per handbook, max absolute error is 2e-8.
///
/// # References
///
///   \[1\] M. Abramowitz and I. A. Stegun, *Handbook of mathematical functions: with formulas, graphs, and mathematical tables*. 1970.
#[inline]
pub fn ellipe(m: f64) -> f64 {
    let mut ellip: f64 = 0.0;
    let c: f64 = 1.0 - m;
    let logterm = c.powi(-1).ln();

    // NOTE: This loop is unrolled at compile-time automatically,
    // and the repeated calls to `powi` are de-duplicated by the compiler.
    for i in 0..5 {
        ellip = logterm
            .mul_add(ELLIPE_B[i], ELLIPE_A[i])
            .mul_add(c.powi(i as i32), ellip);
    }

    ellip
}

/// 3D $(x^2 + y^2 + z^2)^{1/2}$ using `mul_add` to reduce roundoff error.
#[inline]
pub fn norm3<T: Scalar>(v: [T; 3]) -> T {
    dot3(v, v).sqrt()
}

/// Normalize a fixed-size 3D vector.
#[inline]
pub fn normalize3<T: Scalar>(v: [T; 3]) -> [T; 3] {
    scale3(v, T::ONE / norm3(v))
}

/// Evaluate the cross products for each axis component
/// separately using `mul_add` which would not be assumed usable
/// in a more general implementation.
#[inline]
pub fn cross3<T: Scalar>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [
        a[1].mul_add(b[2], (T::ZERO - b[1]) * a[2]),
        a[2].mul_add(b[0], (T::ZERO - b[2]) * a[0]),
        a[0].mul_add(b[1], (T::ZERO - b[0]) * a[1]),
    ]
}

/// Scalar dot product using `mul_add`.
#[inline]
pub fn dot3<T: Scalar>(a: [T; 3], b: [T; 3]) -> T {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

/// Elementwise subtraction of fixed-size 3D vectors.
#[inline]
pub fn sub3<T: Scalar>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Elementwise addition of fixed-size 3D vectors.
#[inline]
pub fn add3<T: Scalar>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// In-place elementwise addition of fixed-size 3D vectors.
#[inline]
pub fn add3_in_place<T: Scalar>(out: &mut [T; 3], value: [T; 3]) {
    for axis in 0..3 {
        out[axis] = out[axis] + value[axis];
    }
}

/// Scale a fixed-size 3D vector.
#[inline]
pub fn scale3<T: Scalar>(value: [T; 3], scale: T) -> [T; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

/// Affine combination `a + scale * b` of fixed-size 3D vectors using `mul_add`.
#[inline]
pub fn add_scaled3<T: Scalar>(a: [T; 3], b: [T; 3], scale: T) -> [T; 3] {
    [
        scale.mul_add(b[0], a[0]),
        scale.mul_add(b[1], a[1]),
        scale.mul_add(b[2], a[2]),
    ]
}

/// Convert a point from cartesian to cylindrical coordinates.
#[inline]
pub fn cartesian_to_cylindrical(point: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = point;
    let r = norm3([x, y, 0.0]);
    let phi = libm::atan2(y, x);
    [r, phi, z]
}

/// Convert a point in cylindrical coordinates to cartesian.
#[inline]
pub fn cylindrical_to_cartesian(point: [f64; 3]) -> [f64; 3] {
    let [r, phi, z] = point;
    let x = r * libm::cos(phi);
    let y = r * libm::sin(phi);
    [x, y, z]
}

/// Geometric components of the system of a filament and an observation point
/// to support the calculation of finite-length, finite-thickness filament field formulas.
///
///  ```text
///        p (target)
///        *
///       /|\
///      / | \
///   ap/  |  \bp
///    / ∠a|∠b \
///   /    |    \
///  a-----m-----b  -> I  
///```
pub(crate) struct PointLineDistance<T: Scalar> {
    /// Perpendicular distance from the infinite line defined by segment `ab` to the point `p`,
    /// clamped to the wire radius.
    pub(crate) perp: T,

    /// The normalized direction of the perpendicular distance.
    pub(crate) perp_hat: [T; 3],

    /// Length of segment `ap` using clamped perpendicular distance.
    pub(crate) dist_a: T,

    /// Length of segment `bp` using clamped perpendicular distance.
    pub(crate) dist_b: T,

    /// Fraction of perpendicular distance of point `p` from the filament axis to the wire radius,
    /// before clamping of the perpendicular distance.
    pub(crate) frac: T,

    /// Length of segment `am` using unclamped perpendicular distance.
    pub(crate) para_a: T,

    /// Length of segment `bm` using unclamped perpendicular distance.
    pub(crate) para_b: T,

    /// Length of the filament, segment `ab`.
    pub(crate) ab_norm: [T; 3],
}

/// Minimum perpendicular distance of point p to the infinite line defined by endpoints a and b,
/// distances to each endpoint, finite-thickness clamp fraction based on `r_min`,
/// and parallel distances from each endpoint to the target.
///
/// Finite-thickness clamping is based only on wire radius and does not
/// taper outside segment endpoint projections.
#[inline]
pub(crate) fn point_line_distance_with_endpoints<T: Scalar>(
    a: [T; 3],
    b: [T; 3],
    p: [T; 3],
    r_min: T,
) -> PointLineDistance<T> {
    // Vectors and distances between points.
    let ab = sub3(b, a);
    let ap = sub3(p, a);
    let bp = sub3(p, b);

    // Normalized segment vector.
    // This might be zero, and that will be handled as late as possible to avoid disrupting
    // calculations in nominal non-zero-length cases.
    let ab2 = dot3(ab, ab); // (m^2) squared length.
    // Handle zero-length special case before any division by segment length.
    if ab2 == T::ZERO {
        let r_min = r_min.max(T::ZERO);
        let r_min_frac = r_min.max(T::min_positive());
        let dist_a = norm3(ap);
        let dist_b = norm3(bp);
        let frac = (dist_a / r_min_frac).min(T::ONE);
        let dist_a = dist_a.max(r_min);
        let dist_b = dist_b.max(r_min);
        let perp = dist_a;
        return PointLineDistance {
            perp,
            perp_hat: [T::ZERO; 3],
            dist_a,
            dist_b,
            frac,
            para_a: T::ZERO,
            para_b: T::ZERO,
            ab_norm: [T::ZERO; 3],
        };
    }

    // Filament vector, length, and normalized direction
    let ab_len = ab2.sqrt();
    let ab_len_inv = T::ONE / ab_len;
    let ab_norm = scale3(ab, ab_len_inv);

    // Find the closest point on the infinite line defined by this segment to the target point.
    let t = dot3(ap, ab) / ab2; // Normed projected location
    let closest = add_scaled3(a, ab, t); // (m) closest point on infinite line
    let dp = sub3(p, closest); // (m) Vector from target to infinite line.
    let perp_raw = norm3(dp); // (m) Un-clamped perpendicular distance.
    let perp_hat = if perp_raw > T::ZERO {
        let inv = T::ONE / perp_raw;
        scale3(dp, inv)
    } else {
        [T::ZERO; 3]
    };

    // Clamp r_min to prevent div/0
    let r_min = r_min.max(T::min_positive());

    // Parallel distances from each endpoint to the target
    let para_a = dot3(ap, ab_norm);
    let para_b = para_a - ab_len;

    // Fraction used by field models to blend finite-thickness behavior to thin-wire behavior.
    let frac = (perp_raw / r_min).min(T::ONE);

    // Clamp distances only if we are inside the minimum radius.
    let perp = perp_raw.max(r_min);

    // Clamped dist_a and dist_b must be kept consistent with the clamped perpendicular distance
    let dist_a = perp.mul_add(perp, para_a * para_a).sqrt();
    let dist_b = perp.mul_add(perp, para_b * para_b).sqrt();

    PointLineDistance {
        perp,
        perp_hat,
        dist_a,
        dist_b,
        frac,
        para_a,
        para_b,
        ab_norm,
    }
}

/// Clip NaN values to the provided value.
/// This is carefully organized to avoid producing any `jmp`
/// instructions on modern-as-of-2025 systems.
#[inline]
pub fn clip_nan(x: f64, v: f64) -> f64 {
    if x.is_nan() { v } else { x }
}

/// Defer between two float values (left and right)
/// depending on some condition.
///
/// Evaluates like `left if cond else right`, clipping
/// either value to zero if it is non-finite.
///
/// If the condition is based on a float comparison, this
/// will evaluate without producing any `jmp`
/// instructions on modern-as-of-2025 systems.
#[inline(always)] // Must be inlined to eliminate jmp
pub fn switch_float(left: f64, right: f64, cond: bool) -> f64 {
    // Convert the boolean check to floating-point factors
    // for each branch without producing a true branch
    let left_factor: f64 = match cond {
        true => 1.0, // This does not produce a jmp!
        false => 0.0,
    };
    let right_factor: f64 = 1.0 - left_factor;

    // Use float factors instead of a true branch
    let left_part = left * left_factor;
    let right_part = right * right_factor;

    clip_nan(left_part, 0.0) + clip_nan(right_part, 0.0)
}
