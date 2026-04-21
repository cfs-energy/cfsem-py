//! Reference-element definitions for 9-node biquadratic quadrilateral elements.
//!
//! This module uses the same isoparametric interpolation framework summarized by Bower in
//! *Applied Mechanics of Solids*, especially Section 8.1 and the quadrilateral interpolation
//! conventions summarized around Table 8.3.  The implementation here is the full tensor-product
//! biquadratic quadrilateral, with a center node in addition to the corner and midside nodes.

use crate::mesh::{Scalar, cast};

/// Number of nodes in the biquadratic quadrilateral element.
pub const NODES_PER_ELEMENT: usize = 9;

fn q2_lagrange_1d<F: Scalar>(x: F) -> [F; 3] {
    let half = cast::<F>(0.5);
    [
        half * x * (x - F::one()),
        F::one() - x * x,
        half * x * (x + F::one()),
    ]
}

fn q2_lagrange_grad_1d<F: Scalar>(x: F) -> [F; 3] {
    let half = cast::<F>(0.5);
    [x - half, -cast::<F>(2.0) * x, x + half]
}

/// Biquadratic shape functions on the reference square `[-1, 1]^2`.
pub fn shape<F: Scalar>(xi: F, eta: F) -> [F; NODES_PER_ELEMENT] {
    let lx = q2_lagrange_1d(xi);
    let ly = q2_lagrange_1d(eta);
    [
        lx[0] * ly[0],
        lx[2] * ly[0],
        lx[2] * ly[2],
        lx[0] * ly[2],
        lx[1] * ly[0],
        lx[2] * ly[1],
        lx[1] * ly[2],
        lx[0] * ly[1],
        lx[1] * ly[1],
    ]
}

/// Shape-function gradients with respect to the reference coordinates `(\xi, \eta)`.
pub fn grad_ref<F: Scalar>(xi: F, eta: F) -> [[F; 2]; NODES_PER_ELEMENT] {
    let lx = q2_lagrange_1d(xi);
    let ly = q2_lagrange_1d(eta);
    let dlx = q2_lagrange_grad_1d(xi);
    let dly = q2_lagrange_grad_1d(eta);
    [
        [dlx[0] * ly[0], lx[0] * dly[0]],
        [dlx[2] * ly[0], lx[2] * dly[0]],
        [dlx[2] * ly[2], lx[2] * dly[2]],
        [dlx[0] * ly[2], lx[0] * dly[2]],
        [dlx[1] * ly[0], lx[1] * dly[0]],
        [dlx[2] * ly[1], lx[2] * dly[1]],
        [dlx[1] * ly[2], lx[1] * dly[2]],
        [dlx[0] * ly[1], lx[0] * dly[1]],
        [dlx[1] * ly[1], lx[1] * dly[1]],
    ]
}

/// Map a 1D reference coordinate `s in [-1, 1]` onto a local element face.
pub fn face_reference<F: Scalar>(local_face: u8, s: F) -> Result<(F, F, [F; 2]), String> {
    match local_face {
        0 => Ok((s, -F::one(), [F::one(), F::zero()])),
        1 => Ok((F::one(), s, [F::zero(), F::one()])),
        2 => Ok((-s, F::one(), [-F::one(), F::zero()])),
        3 => Ok((-F::one(), -s, [F::zero(), -F::one()])),
        _ => Err(format!(
            "invalid local face {local_face}; expected 0, 1, 2, or 3"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{face_reference, grad_ref, shape};

    #[test]
    fn shape_functions_sum_to_one() {
        let n = shape(0.2_f64, -0.3_f64);
        let sum: f64 = n.into_iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn gradients_sum_to_zero() {
        let grad = grad_ref(-0.1_f64, 0.5_f64);
        let sx: f64 = grad.iter().map(|g| g[0]).sum();
        let sy: f64 = grad.iter().map(|g| g[1]).sum();
        assert!(sx.abs() < 1.0e-12);
        assert!(sy.abs() < 1.0e-12);
    }

    #[test]
    fn face_parameterization_matches_corner_and_midside_points() {
        let (xi0, eta0, _) = face_reference(0, -1.0_f64).expect("face 0 start");
        let (xim, etam, _) = face_reference(0, 0.0_f64).expect("face 0 midpoint");
        let (xi1, eta1, _) = face_reference(0, 1.0_f64).expect("face 0 end");
        assert_eq!((xi0, eta0), (-1.0, -1.0));
        assert_eq!((xim, etam), (0.0, -1.0));
        assert_eq!((xi1, eta1), (1.0, -1.0));
    }
}
