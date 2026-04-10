//! Reference-element definitions and element-local geometry helpers for 3-node triangles.

use crate::math::{add_scaled3, dot3_arr, sub3};
use crate::mesh::Scalar;

/// Number of nodes in the linear triangular element.
pub const NODES_PER_ELEMENT: usize = 3;

/// Linear triangle shape functions on the reference simplex with vertices
/// `(u, v) = (0, 0), (1, 0), (0, 1)`.
pub fn shape<F: Scalar>(u: F, v: F) -> [F; NODES_PER_ELEMENT] {
    [F::one() - u - v, u, v]
}

/// Shape-function gradients with respect to the reference coordinates `(u, v)`.
pub fn grad_ref<F: Scalar>() -> [[F; 2]; NODES_PER_ELEMENT] {
    [
        [-F::one(), -F::one()],
        [F::one(), F::zero()],
        [F::zero(), F::one()],
    ]
}

/// Largest squared edge length of one triangle.
#[inline]
pub fn max_edge_length_squared(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> f64 {
    let e01 = sub3(n1, n0);
    let e12 = sub3(n2, n1);
    let e20 = sub3(n0, n2);
    dot3_arr(e01, e01)
        .max(dot3_arr(e12, e12))
        .max(dot3_arr(e20, e20))
}

/// Closest point on a triangle to an observation point.
#[inline]
pub fn closest_point(obs: [f64; 3], n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> [f64; 3] {
    let ab = sub3(n1, n0);
    let ac = sub3(n2, n0);
    let ap = sub3(obs, n0);
    let d1 = dot3_arr(ab, ap);
    let d2 = dot3_arr(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return n0;
    }

    let bp = sub3(obs, n1);
    let d3 = dot3_arr(ab, bp);
    let d4 = dot3_arr(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return n1;
    }

    let vc = d1.mul_add(d4, -(d3 * d2));
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return add_scaled3(n0, ab, d1 / (d1 - d3));
    }

    let cp = sub3(obs, n2);
    let d5 = dot3_arr(ab, cp);
    let d6 = dot3_arr(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return n2;
    }

    let vb = d5.mul_add(d2, -(d1 * d6));
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return add_scaled3(n0, ac, d2 / (d2 - d6));
    }

    let bc = sub3(n2, n1);
    let va = d3.mul_add(d6, -(d5 * d4));
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return add_scaled3(n1, bc, (d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }

    let denom_inv = 1.0 / (va + vb + vc);
    let v = vb * denom_inv;
    let w = vc * denom_inv;
    add_scaled3(add_scaled3(n0, ab, v), ac, w)
}

/// Split a triangle into three subtriangles sharing an interior point.
#[inline]
pub fn subdivide_about_point(
    point: [f64; 3],
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
) -> [[[f64; 3]; 3]; 3] {
    [[point, n0, n1], [point, n1, n2], [point, n2, n0]]
}

#[cfg(test)]
mod tests {
    use super::{grad_ref, shape};
    use crate::mesh::elements::tri::mapping::{area, map_point};

    #[test]
    fn shape_functions_sum_to_one() {
        let n = shape(0.2_f64, 0.3_f64);
        let sum: f64 = n.into_iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn gradients_sum_to_zero() {
        let grad = grad_ref::<f64>();
        let su: f64 = grad.iter().map(|g| g[0]).sum();
        let sv: f64 = grad.iter().map(|g| g[1]).sum();
        assert!(su.abs() < 1.0e-12);
        assert!(sv.abs() < 1.0e-12);
    }

    #[test]
    fn map_point_hits_vertices() {
        let coords = [[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        assert_eq!(
            map_point(coords[0], coords[1], coords[2], [0.0, 0.0]),
            coords[0]
        );
        assert_eq!(
            map_point(coords[0], coords[1], coords[2], [1.0, 0.0]),
            coords[1]
        );
        assert_eq!(
            map_point(coords[0], coords[1], coords[2], [0.0, 1.0]),
            coords[2]
        );
    }

    #[test]
    fn area_matches_right_triangle() {
        let n0 = [0.0, 0.0, 0.0];
        let n1 = [1.0, 0.0, 0.0];
        let n2 = [0.0, 2.0, 0.0];
        assert!((area(n0, n1, n2) - 1.0).abs() < 1.0e-12);
    }
}
