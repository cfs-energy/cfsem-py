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
    for i in 0..5 {
        ellip = logterm
            .mul_add(ELLIPE_B[i], ELLIPE_A[i])
            .mul_add(c.powi(i as i32), ellip);
    }

    ellip
}

/// 3D $(x^2 + y^2 + z^2)^{1/2}$ using `mul_add` to perform all addition in extended registers.
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

/// Scalar dot product using `mul_add`.
#[inline]
pub fn dot3(x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64) -> f64 {
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
pub fn decompose_filament(
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
