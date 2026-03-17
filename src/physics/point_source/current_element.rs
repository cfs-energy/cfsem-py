//! Internal point-current-element kernels shared by point-segment and
//! boundary-element quadrature implementations.

use crate::MU0_OVER_4PI;
use crate::math::{cross3, dot3, rss3};

/// Magnetic flux density from a point current element with vector moment
/// `m = I Δl = K ΔS = J ΔV`.
#[inline]
pub(crate) fn flux_density_current_element_scalar(
    src: [f64; 3],
    moment: [f64; 3],
    obs: [f64; 3],
) -> [f64; 3] {
    let r = [obs[0] - src[0], obs[1] - src[1], obs[2] - src[2]];
    let sumsq = dot3(r[0], r[1], r[2], r[0], r[1], r[2]);
    let rnorm3_inv = sumsq.powf(-1.5);
    let m_cross_r = cross3(moment[0], moment[1], moment[2], r[0], r[1], r[2]);
    [
        MU0_OVER_4PI * m_cross_r.0 * rnorm3_inv,
        MU0_OVER_4PI * m_cross_r.1 * rnorm3_inv,
        MU0_OVER_4PI * m_cross_r.2 * rnorm3_inv,
    ]
}

/// Magnetic vector potential from a point current element with vector moment
/// `m = I Δl = K ΔS = J ΔV`.
#[inline]
pub(crate) fn vector_potential_current_element_scalar(
    src: [f64; 3],
    moment: [f64; 3],
    obs: [f64; 3],
) -> [f64; 3] {
    let rnorm = rss3(obs[0] - src[0], obs[1] - src[1], obs[2] - src[2]);
    [
        MU0_OVER_4PI * moment[0] / rnorm,
        MU0_OVER_4PI * moment[1] / rnorm,
        MU0_OVER_4PI * moment[2] / rnorm,
    ]
}
