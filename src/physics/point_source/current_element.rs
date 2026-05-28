//! Internal point-current-element kernels shared by point-segment and
//! boundary-element quadrature implementations.

use crate::MU0_OVER_4PI;
use crate::math::{cross3, dot3, max_scalar};
use crate::physics::hierarchical::Scalar;

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
pub(crate) fn flux_density_current_element_scalar<T: Scalar>(
    src: [T; 3],
    moment: [T; 3],
    obs: [T; 3],
) -> [T; 3] {
    let r = [obs[0] - src[0], obs[1] - src[1], obs[2] - src[2]]; // [m]
    let r_sq = dot3(r, r); // [m^2]
    let min_distance_sq = T::from_f64(CURRENT_ELEMENT_MIN_DISTANCE_SQ);
    let near = r_sq < min_distance_sq; // [-]
    let rnorm3_inv = max_scalar(r_sq, min_distance_sq).powf(-1.5); // [m^-3]
    let m_cross_r = cross3(moment, r); // [A*m^2]
    let c = T::from_f64(MU0_OVER_4PI);
    let out = [
        c * m_cross_r[0] * rnorm3_inv, // [T]
        c * m_cross_r[1] * rnorm3_inv, // [T]
        c * m_cross_r[2] * rnorm3_inv, // [T]
    ];
    if near { [T::ZERO; 3] } else { out }
}

/// Magnetic vector potential from a point current element with vector moment
/// `m = I Δl = K ΔS = J ΔV`.
#[inline]
pub(crate) fn vector_potential_current_element_scalar<T: Scalar>(
    src: [T; 3],
    moment: [T; 3],
    obs: [T; 3],
) -> [T; 3] {
    let r = [obs[0] - src[0], obs[1] - src[1], obs[2] - src[2]]; // [m]
    let r_sq = dot3(r, r); // [m^2]
    let min_distance = T::from_f64(CURRENT_ELEMENT_MIN_DISTANCE);
    let min_distance_sq = T::from_f64(CURRENT_ELEMENT_MIN_DISTANCE_SQ);
    let near = r_sq < min_distance_sq; // [-]
    let rmag = max_scalar(r_sq.sqrt(), min_distance); // [m]
    let c = T::from_f64(MU0_OVER_4PI);
    let out = [
        c * moment[0] / rmag, // [V*s/m]
        c * moment[1] / rmag, // [V*s/m]
        c * moment[2] / rmag, // [V*s/m]
    ];
    if near { [T::ZERO; 3] } else { out }
}
