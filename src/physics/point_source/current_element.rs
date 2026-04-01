//! Internal point-current-element kernels shared by point-segment and
//! boundary-element quadrature implementations.

use crate::MU0_OVER_4PI;
use crate::math::{cross3, dot3};

/// Minimum observation-point distance below which point-current-element kernels
/// return zero to avoid singular self-evaluation.
const CURRENT_ELEMENT_MIN_DISTANCE: f64 = 1e-14; // [m]

/// Squared form of [`CURRENT_ELEMENT_MIN_DISTANCE`] for early near-field checks
/// without an extra square root.
const CURRENT_ELEMENT_MIN_DISTANCE_SQ: f64 =
    CURRENT_ELEMENT_MIN_DISTANCE * CURRENT_ELEMENT_MIN_DISTANCE; // [m^2]

/// Magnetic flux density from a point current element with vector moment
/// `m = I Δl = K ΔS = J ΔV`.
#[inline]
pub(crate) fn flux_density_current_element_scalar(
    src: [f64; 3],
    moment: [f64; 3],
    obs: [f64; 3],
) -> [f64; 3] {
    let r = [obs[0] - src[0], obs[1] - src[1], obs[2] - src[2]]; // [m]
    let r_sq = dot3(r[0], r[1], r[2], r[0], r[1], r[2]); // [m^2]
    let near = r_sq < CURRENT_ELEMENT_MIN_DISTANCE_SQ; // [-]
    let rnorm3_inv = r_sq.max(CURRENT_ELEMENT_MIN_DISTANCE_SQ).powf(-1.5); // [m^-3]
    let m_cross_r = cross3(moment[0], moment[1], moment[2], r[0], r[1], r[2]); // [A*m^2]
    let out = [
        MU0_OVER_4PI * m_cross_r.0 * rnorm3_inv, // [T]
        MU0_OVER_4PI * m_cross_r.1 * rnorm3_inv, // [T]
        MU0_OVER_4PI * m_cross_r.2 * rnorm3_inv, // [T]
    ];
    if near { [0.0, 0.0, 0.0] } else { out }
}

/// Magnetic vector potential from a point current element with vector moment
/// `m = I Δl = K ΔS = J ΔV`.
#[inline]
pub(crate) fn vector_potential_current_element_scalar(
    src: [f64; 3],
    moment: [f64; 3],
    obs: [f64; 3],
) -> [f64; 3] {
    let r = [obs[0] - src[0], obs[1] - src[1], obs[2] - src[2]]; // [m]
    let r_sq = dot3(r[0], r[1], r[2], r[0], r[1], r[2]); // [m^2]
    let near = r_sq < CURRENT_ELEMENT_MIN_DISTANCE_SQ; // [-]
    let rmag = r_sq.sqrt().max(CURRENT_ELEMENT_MIN_DISTANCE); // [m]
    let out = [
        MU0_OVER_4PI * moment[0] / rmag, // [V*s/m]
        MU0_OVER_4PI * moment[1] / rmag, // [V*s/m]
        MU0_OVER_4PI * moment[2] / rmag, // [V*s/m]
    ];
    if near { [0.0, 0.0, 0.0] } else { out }
}
