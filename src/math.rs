//! Pure-math functions supporting physics calculations.

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
pub fn rss3(x: f64, y: f64, z: f64) -> f64 {
    x.mul_add(x, y.mul_add(y, z.powi(2))).sqrt()
}

/// Evaluate the cross products for each axis component
/// separately using `mul_add` which would not be assumed usable
/// in a more general implementation.
#[inline]
pub fn cross3(x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64) -> (f64, f64, f64) {
    let xy = -x1 * y0;
    let yz = -y1 * z0;
    let zx = -z1 * x0;
    let cx = y0.mul_add(z1, yz);
    let cy = z0.mul_add(x1, zx);
    let cz = x0.mul_add(y1, xy);

    (cx, cy, cz)
}

/// Evaluate the cross products for each axis component
/// separately using `mul_add` which would not be assumed usable
/// in a more general implementation.
/// 32-bit float variant.
#[inline]
pub fn cross3f(x0: f32, y0: f32, z0: f32, x1: f32, y1: f32, z1: f32) -> (f32, f32, f32) {
    let xy = -x1 * y0;
    let yz = -y1 * z0;
    let zx = -z1 * x0;
    let cx = y0.mul_add(z1, yz);
    let cy = z0.mul_add(x1, zx);
    let cz = x0.mul_add(y1, xy);

    (cx, cy, cz)
}

/// Scalar dot product using `mul_add`.
#[inline]
pub fn dot3(x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64) -> f64 {
    x0.mul_add(x1, y0.mul_add(y1, z0 * z1))
}

/// Scalar dot product using `mul_add`.
/// 32-bit float variant.
#[inline]
pub fn dot3f(x0: f32, y0: f32, z0: f32, x1: f32, y1: f32, z1: f32) -> f32 {
    x0.mul_add(x1, y0.mul_add(y1, z0 * z1))
}

/// Convert a point from cartesian to cylindrical coordinates.
#[inline]
pub fn cartesian_to_cylindrical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = rss3(x, y, 0.0);
    let phi = libm::atan2(y, x);
    (r, phi, z)
}

/// Convert a point in cylindrical coordinates to cartesian.
#[inline]
pub fn cylindrical_to_cartesian(r: f64, phi: f64, z: f64) -> (f64, f64, f64) {
    let x = r * libm::cos(phi);
    let y = r * libm::sin(phi);
    (x, y, z)
}

/// Decompose two filament endpoints into a midpoint and a length vector
#[inline]
pub(crate) fn decompose_filament(
    start: (f64, f64, f64),
    end: (f64, f64, f64),
) -> ((f64, f64, f64), (f64, f64, f64)) {
    // Evaluate
    let dl = (end.0 - start.0, end.1 - start.1, end.2 - start.2); // [m] filament vector
    let midpoint = (
        dl.0.mul_add(0.5, start.0),
        dl.1.mul_add(0.5, start.1),
        dl.2.mul_add(0.5, start.2),
    ); // [m] filament midpoint

    (midpoint, dl)
}

pub(crate) struct PointLineDistance {
    pub(crate) perp: f64,
    pub(crate) perp_hat: (f64, f64, f64),
    pub(crate) dist_a: f64,
    pub(crate) dist_b: f64,
    pub(crate) frac: f64,
    pub(crate) para_a: f64,
    pub(crate) para_b: f64,
    pub(crate) ab_norm: (f64, f64, f64),
}

/// Minimum perpendicular distance of point p to the infinite line defined by endpoints a and b,
/// distances to each endpoint, finite-thickness clamp fraction based on `r_min`,
/// and parallel distances from each endpoint to the target.
///
/// Finite-thickness clamping is based only on wire radius and does not
/// taper outside segment endpoint projections.
#[inline]
pub(crate) fn point_line_distance_with_endpoints(
    a: (f64, f64, f64),
    b: (f64, f64, f64),
    p: (f64, f64, f64),
    r_min: f64,
) -> PointLineDistance {
    // Vectors and distances between points.
    let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
    let ap = (p.0 - a.0, p.1 - a.1, p.2 - a.2);
    let bp = (p.0 - b.0, p.1 - b.1, p.2 - b.2);

    // Normalized segment vector.
    // This might be zero, and that will be handled as late as possible to avoid disrupting
    // calculations in nominal non-zero-length cases.
    let ab2 = dot3(ab.0, ab.1, ab.2, ab.0, ab.1, ab.2); // (m^2) squared length.
    // Handle zero-length special case before any division by segment length.
    if ab2 == 0.0 {
        let r_min = r_min.max(0.0);
        let r_min_frac = r_min.max(f64::MIN_POSITIVE);
        let dist_a = rss3(ap.0, ap.1, ap.2);
        let dist_b = rss3(bp.0, bp.1, bp.2);
        let frac = (dist_a / r_min_frac).min(1.0);
        let dist_a = dist_a.max(r_min);
        let dist_b = dist_b.max(r_min);
        let perp = dist_a;
        return PointLineDistance {
            perp,
            perp_hat: (0.0, 0.0, 0.0),
            dist_a,
            dist_b,
            frac,
            para_a: 0.0,
            para_b: 0.0,
            ab_norm: (0.0, 0.0, 0.0),
        };
    }

    // Filament vector, length, and normalized direction
    let ab_len = ab2.sqrt();
    let ab_len_inv = ab_len.recip();
    let ab_norm = (ab.0 * ab_len_inv, ab.1 * ab_len_inv, ab.2 * ab_len_inv);

    // Find the closest point on the infinite line defined by this segment to the target point.
    let t = dot3(ap.0, ap.1, ap.2, ab.0, ab.1, ab.2) / ab2; // Normed projected location
    let closest = (
        t.mul_add(ab.0, a.0),
        t.mul_add(ab.1, a.1),
        t.mul_add(ab.2, a.2),
    ); // (m) closest point on infinite line
    let dp = (p.0 - closest.0, p.1 - closest.1, p.2 - closest.2); // (m) Vector from target to infinite line.
    let perp_raw = rss3(dp.0, dp.1, dp.2); // (m) Un-clamped perpendicular distance.
    let perp_hat = if perp_raw > 0.0 {
        let inv = perp_raw.recip();
        (dp.0 * inv, dp.1 * inv, dp.2 * inv)
    } else {
        (0.0, 0.0, 0.0)
    };

    // Clamp r_min to prevent div/0
    let r_min = r_min.max(f64::MIN_POSITIVE);

    // Parallel distances from each endpoint to the target
    let para_a = dot3(ap.0, ap.1, ap.2, ab_norm.0, ab_norm.1, ab_norm.2);
    let para_b = para_a - ab_len;

    // Fraction used by field models to blend finite-thickness behavior to thin-wire behavior.
    let frac = (perp_raw / r_min).min(1.0);

    // Clamp distances only if we are inside the minimum radius.
    let perp = perp_raw.max(r_min);

    // Clamped dist_a and dist_b must be kept consistent with the clamped perpendicular distance
    let dist_a = (perp * perp + para_a * para_a).sqrt();
    let dist_b = (perp * perp + para_b * para_b).sqrt();

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
