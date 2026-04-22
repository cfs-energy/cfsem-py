//! Physical-space mapping helpers for 3-node triangles in 3D.

use crate::math::{cross3, rss3};

/// Map one reference-triangle point `(u, v)` into a physical triangle in 3D.
#[inline]
pub fn map_point(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3], uv: [f64; 2]) -> [f64; 3] {
    let w = 1.0 - uv[0] - uv[1];
    [
        n0[0] * w + n1[0] * uv[0] + n2[0] * uv[1],
        n0[1] * w + n1[1] * uv[0] + n2[1] * uv[1],
        n0[2] * w + n1[2] * uv[0] + n2[2] * uv[1],
    ]
}

/// Physical area of a 3D triangle.
#[inline]
pub fn area(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> f64 {
    let v01 = [n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]];
    let v02 = [n2[0] - n0[0], n2[1] - n0[1], n2[2] - n0[2]];
    let cross = cross3(v01[0], v01[1], v01[2], v02[0], v02[1], v02[2]);
    0.5 * rss3(cross.0, cross.1, cross.2)
}

/// Unit normal of a 3D triangle.
#[inline]
pub fn normal(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> [f64; 3] {
    let v01 = [n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]];
    let v02 = [n2[0] - n0[0], n2[1] - n0[1], n2[2] - n0[2]];
    let cross = cross3(v01[0], v01[1], v01[2], v02[0], v02[1], v02[2]);
    let norm = rss3(cross.0, cross.1, cross.2);
    [cross.0 / norm, cross.1 / norm, cross.2 / norm]
}
