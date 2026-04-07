//! Reference-element definitions for bilinear 4-node quadrilateral elements.
//!
//! This module contains the purely geometric pieces of the formulation:
//! - scalar shape functions `N_i(\xi, \eta)`,
//! - their gradients in reference coordinates,
//! - the Jacobian that maps reference coordinates into physical `(r, z)` space, and
//! - the local-face parameterization used by pressure loads.

use crate::physics::solenoid_stress::types::{Real, cast};

/// Number of nodes in the bilinear quadrilateral element.
pub const NODES_PER_ELEMENT: usize = 4;
/// Number of displacement unknowns per node: radial and axial.
pub const DOF_PER_NODE: usize = 2;
/// Number of displacement unknowns per element.
pub const DOF_PER_ELEMENT: usize = NODES_PER_ELEMENT * DOF_PER_NODE;

/// Bilinear shape functions on the reference square `[-1, 1]^2`.
pub fn shape<F: Real>(xi: F, eta: F) -> [F; NODES_PER_ELEMENT] {
    let quarter = cast::<F>(0.25);
    [
        quarter * (F::one() - xi) * (F::one() - eta),
        quarter * (F::one() + xi) * (F::one() - eta),
        quarter * (F::one() + xi) * (F::one() + eta),
        quarter * (F::one() - xi) * (F::one() + eta),
    ]
}

/// Shape-function gradients with respect to the reference coordinates `(\xi, \eta)`.
pub fn grad_ref<F: Real>(xi: F, eta: F) -> [[F; 2]; NODES_PER_ELEMENT] {
    let quarter = cast::<F>(0.25);
    [
        [-quarter * (F::one() - eta), -quarter * (F::one() - xi)],
        [quarter * (F::one() - eta), -quarter * (F::one() + xi)],
        [quarter * (F::one() + eta), quarter * (F::one() + xi)],
        [-quarter * (F::one() + eta), quarter * (F::one() - xi)],
    ]
}

/// Map one reference point into physical `(r, z)` coordinates using the element interpolation.
pub fn map_point<F: Real>(
    coords: &[[F; 2]; NODES_PER_ELEMENT],
    n: &[F; NODES_PER_ELEMENT],
) -> [F; 2] {
    let mut point = [F::zero(); 2];
    for i in 0..NODES_PER_ELEMENT {
        point[0] = point[0] + n[i] * coords[i][0];
        point[1] = point[1] + n[i] * coords[i][1];
    }
    point
}

/// Jacobian matrix of the reference-to-physical map.
///
/// Rows correspond to physical coordinates `(r, z)` and columns to reference coordinates
/// `(\xi, \eta)`.
pub fn jacobian<F: Real>(
    coords: &[[F; 2]; NODES_PER_ELEMENT],
    grad: &[[F; 2]; NODES_PER_ELEMENT],
) -> [[F; 2]; 2] {
    let mut jac = [[F::zero(); 2]; 2];
    for i in 0..NODES_PER_ELEMENT {
        jac[0][0] = jac[0][0] + coords[i][0] * grad[i][0];
        jac[0][1] = jac[0][1] + coords[i][0] * grad[i][1];
        jac[1][0] = jac[1][0] + coords[i][1] * grad[i][0];
        jac[1][1] = jac[1][1] + coords[i][1] * grad[i][1];
    }
    jac
}

/// Determinant of the element Jacobian.
///
/// For a valid element this must stay positive at every quadrature point.
pub fn det_j<F: Real>(jac: &[[F; 2]; 2]) -> F {
    jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]
}

/// Inverse Jacobian used to transform reference gradients into physical gradients.
pub fn inv_j<F: Real>(jac: &[[F; 2]; 2]) -> Result<[[F; 2]; 2], String> {
    let det = det_j(jac);
    if det <= F::zero() {
        return Err(format!(
            "encountered non-positive element Jacobian determinant {det:?}"
        ));
    }
    let inv_det = F::one() / det;
    Ok([
        [jac[1][1] * inv_det, -jac[0][1] * inv_det],
        [-jac[1][0] * inv_det, jac[0][0] * inv_det],
    ])
}

/// Convert shape-function gradients from `(\xi, \eta)` to `(r, z)`.
pub fn grad_phys<F: Real>(
    grad_reference: &[[F; 2]; NODES_PER_ELEMENT],
    inv_jac: &[[F; 2]; 2],
) -> [[F; 2]; NODES_PER_ELEMENT] {
    let mut out = [[F::zero(); 2]; NODES_PER_ELEMENT];
    for i in 0..NODES_PER_ELEMENT {
        let dxi = grad_reference[i][0];
        let deta = grad_reference[i][1];
        out[i][0] = inv_jac[0][0] * dxi + inv_jac[1][0] * deta;
        out[i][1] = inv_jac[0][1] * dxi + inv_jac[1][1] * deta;
    }
    out
}

/// Map a 1D reference coordinate `s in [-1, 1]` onto a local element face.
///
/// Returns `(xi, eta, d[xi,eta]/ds)`.  The face numbering follows the usual counterclockwise
/// ordering:
/// - `0`: bottom edge from node 0 to node 1
/// - `1`: right edge from node 1 to node 2
/// - `2`: top edge from node 2 to node 3
/// - `3`: left edge from node 3 to node 0
pub fn face_reference<F: Real>(local_face: u8, s: F) -> Result<(F, F, [F; 2]), String> {
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
    use super::{grad_ref, shape};

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
}
