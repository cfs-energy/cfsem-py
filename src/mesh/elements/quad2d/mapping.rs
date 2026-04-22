//! Generic isoparametric mapping and Jacobian helpers for 2D quadrilateral elements.
//!
//! These routines implement the finite-element geometry pipeline for one 2D isoparametric
//! element:
//! - interpolate physical coordinates from nodal coordinates with the shape functions,
//! - differentiate that interpolation to form the element Jacobian `J`,
//! - use `J` and `J^{-1}` to convert between reference-space and physical-space derivatives, and
//! - map reference-edge directions into physical tangents for face integrals.
//!
//! The key geometric statement is that the element itself is interpolated with the same shape
//! functions used for the solution field. If `x_e = [[x_1, y_1], ..., [x_n, y_n]]` stores the
//! element nodal coordinates and `N_i(\xi, \eta)` are the reference-element shape functions, then
//! the physical point corresponding to one reference point `(\xi, \eta)` is
//! `x(\xi, \eta) = sum_i N_i(\xi, \eta) x_i`. The Jacobian is the matrix of partial derivatives
//! of that map:
//! - `J[0, 0] = dx / d\xi`
//! - `J[0, 1] = dx / d\eta`
//! - `J[1, 0] = dy / d\xi`
//! - `J[1, 1] = dy / d\eta`
//!
//! `det(J)` is the local area scaling from reference space to physical space, while `J^{-1}` is
//! the chain-rule map used to convert reference-coordinate gradients into physical gradients.
//!
//! # References
//!
//! - Allan F. Bower, *Applied Mechanics of Solids*, CRC Press, 2009, Section 8.1.11 and
//!   Section 8.1.13.

use crate::mesh::Scalar;

/// Interpolate one physical point from nodal coordinates and shape-function values.
///
/// This evaluates the isoparametric geometry map
/// `x(\xi, \eta) = sum_i N_i(\xi, \eta) x_i`
/// at one reference point. The returned coordinates live in the physical element domain.
pub fn map_point<F: Scalar, const NODES_PER_ELEMENT: usize>(
    coords: &[[F; 2]; NODES_PER_ELEMENT],
    n: &[F; NODES_PER_ELEMENT],
) -> [F; 2] {
    let mut point = [F::zero(); 2];
    for i in 0..NODES_PER_ELEMENT {
        // x = sum_i N_i x_i, y = sum_i N_i y_i
        point[0] = point[0] + n[i] * coords[i][0];
        point[1] = point[1] + n[i] * coords[i][1];
    }
    point
}

/// Build the 2 x 2 Jacobian of the reference-to-physical geometry map.
///
/// The rows correspond to the physical coordinates `(x, y)` and the columns correspond to the
/// reference coordinates `(\xi, \eta)`, so
/// `J = [[dx/d\xi, dx/d\eta], [dy/d\xi, dy/d\eta]]`.
pub fn jacobian<F: Scalar, const NODES_PER_ELEMENT: usize>(
    coords: &[[F; 2]; NODES_PER_ELEMENT],
    grad: &[[F; 2]; NODES_PER_ELEMENT],
) -> [[F; 2]; 2] {
    let mut jac = [[F::zero(); 2]; 2];
    for i in 0..NODES_PER_ELEMENT {
        // J = sum_i x_i \otimes grad_ref(N_i)
        jac[0][0] = jac[0][0] + coords[i][0] * grad[i][0];
        jac[0][1] = jac[0][1] + coords[i][0] * grad[i][1];
        jac[1][0] = jac[1][0] + coords[i][1] * grad[i][0];
        jac[1][1] = jac[1][1] + coords[i][1] * grad[i][1];
    }
    jac
}

/// Determinant of the 2 x 2 element Jacobian.
///
/// This quantity is the local area scaling between the reference element and the physical
/// element.
pub fn det_j<F: Scalar>(jac: &[[F; 2]; 2]) -> F {
    jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]
}

/// Inverse of the 2 x 2 element Jacobian.
///
/// The inverse Jacobian is the chain-rule map that converts reference-coordinate gradients into
/// physical-coordinate gradients. A non-positive determinant is rejected because it indicates an
/// inverted or degenerate element mapping.
pub fn inv_j<F: Scalar>(jac: &[[F; 2]; 2]) -> Result<[[F; 2]; 2], String> {
    let det = det_j(jac);
    if det <= F::zero() {
        return Err(format!(
            "encountered non-positive element Jacobian determinant {det:?}; quadrilateral elements must be non-degenerate and use counter-clockwise corner ordering in the (r, z) plane"
        ));
    }
    let inv_det = F::one() / det;
    Ok([
        [jac[1][1] * inv_det, -jac[0][1] * inv_det],
        [-jac[1][0] * inv_det, jac[0][0] * inv_det],
    ])
}

/// Map reference-space shape-function gradients into physical-space gradients.
///
/// If `grad_reference[i] = [dN_i/d\xi, dN_i/d\eta]`, this routine returns
/// `grad_phys[i] = [dN_i/dx, dN_i/dy]` using the inverse Jacobian and the multivariable chain
/// rule.
pub fn grad_phys<F: Scalar, const NODES_PER_ELEMENT: usize>(
    grad_reference: &[[F; 2]; NODES_PER_ELEMENT],
    inv_jac: &[[F; 2]; 2],
) -> [[F; 2]; NODES_PER_ELEMENT] {
    let mut out = [[F::zero(); 2]; NODES_PER_ELEMENT];
    for i in 0..NODES_PER_ELEMENT {
        let dxi = grad_reference[i][0];
        let deta = grad_reference[i][1];
        // [dN/dx, dN/dy]^T = J^{-T} [dN/d\xi, dN/d\eta]^T
        out[i][0] = inv_jac[0][0] * dxi + inv_jac[1][0] * deta;
        out[i][1] = inv_jac[0][1] * dxi + inv_jac[1][1] * deta;
    }
    out
}

/// Map a reference-edge direction into a physical tangent vector.
///
/// Face integration starts with a one-dimensional reference coordinate `s` on an edge.
/// If `ds_reference = d[\xi, \eta]/ds`, then the physical tangent follows from the chain rule:
/// `dx/ds = J d[\xi, \eta]/ds`.
///
/// This tangent carries both orientation and physical stretching of the face parameterization.
/// Surface/line measures are then built from its norm.
pub fn face_tangent<F: Scalar>(jac: &[[F; 2]; 2], ds_reference: [F; 2]) -> [F; 2] {
    [
        jac[0][0] * ds_reference[0] + jac[0][1] * ds_reference[1],
        jac[1][0] * ds_reference[0] + jac[1][1] * ds_reference[1],
    ]
}
