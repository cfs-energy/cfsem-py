//! Reference-element definitions for bilinear 4-node quadrilateral elements.
//!
//! This is the 4-node quadrilateral interpolation listed in Bower's *Applied Mechanics of
//! Solids*, Section 8.1, Table 8.3.

use crate::mesh::{Scalar, cast};

/// Number of nodes in the bilinear quadrilateral element.
pub const NODES_PER_ELEMENT: usize = 4;

/// Corner-node pairs for the four local faces.
///
/// The local corner-node ordering is
/// `[0, 1, 2, 3] = [bottom-left, bottom-right, top-right, top-left]`, which is the
/// counter-clockwise ordering required by the isoparametric mapping.
///
/// The matching local face numbering is:
/// - `0`: bottom edge from node `0` to node `1`
/// - `1`: right edge from node `1` to node `2`
/// - `2`: top edge from node `2` to node `3`
/// - `3`: left edge from node `3` to node `0`
pub const FACE_NODE_PAIRS: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

/// Bilinear shape functions on the reference square `[-1, 1]^2`.
///
/// The local node ordering is
/// `[0, 1, 2, 3] = [(-1, -1), (1, -1), (1, 1), (-1, 1)]`.
pub fn shape<F: Scalar>(xi: F, eta: F) -> [F; NODES_PER_ELEMENT] {
    let quarter = cast::<F>(0.25);
    [
        quarter * (F::one() - xi) * (F::one() - eta),
        quarter * (F::one() + xi) * (F::one() - eta),
        quarter * (F::one() + xi) * (F::one() + eta),
        quarter * (F::one() - xi) * (F::one() + eta),
    ]
}

/// Shape-function gradients with respect to the reference coordinates `(\xi, \eta)`.
///
/// The gradients are returned in the same local node ordering used by [`shape`].
pub fn grad_ref<F: Scalar>(xi: F, eta: F) -> [[F; 2]; NODES_PER_ELEMENT] {
    let quarter = cast::<F>(0.25);
    [
        [-quarter * (F::one() - eta), -quarter * (F::one() - xi)],
        [quarter * (F::one() - eta), -quarter * (F::one() + xi)],
        [quarter * (F::one() + eta), quarter * (F::one() + xi)],
        [-quarter * (F::one() + eta), quarter * (F::one() - xi)],
    ]
}

/// Map a 1D reference coordinate `s in [-1, 1]` onto a local element face.
///
/// The local face numbering follows [`FACE_NODE_PAIRS`]. For counter-clockwise physical
/// elements, the returned reference-edge direction produces a physical tangent whose clockwise
/// normal is the outward face normal used by the axisymmetric pressure assembly.
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
