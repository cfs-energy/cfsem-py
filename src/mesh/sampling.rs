//! Generic quadrature-point sampling of 2D quadrilateral elements.

use crate::mesh::Scalar;
use crate::mesh::elements::quad2d::{
    QuadratureRule,
    mapping::{det_j, face_tangent, grad_phys, inv_j, jacobian, map_point},
    quad4, quad9,
    quadrature::{gauss_face, gauss_volume},
};

#[derive(Debug, Clone, Copy)]
pub struct VolumeSample<F: Scalar, const NODES_PER_ELEMENT: usize> {
    /// Shape-function values at the quadrature point.
    pub n: [F; NODES_PER_ELEMENT],
    /// Physical gradients `[dN_i/dx, dN_i/dy]`.
    pub grad_phys: [[F; 2]; NODES_PER_ELEMENT],
    /// Jacobian determinant `det(J)` of the reference-to-physical map.
    pub det_j: F,
    /// Physical quadrature point `(x, y)`.
    pub point: [F; 2],
    /// Reference-space quadrature weight `w`.
    pub weight: F,
}

#[derive(Debug, Clone, Copy)]
pub struct FaceSample<F: Scalar, const NODES_PER_ELEMENT: usize> {
    /// Shape-function values on the element face.
    pub n: [F; NODES_PER_ELEMENT],
    /// Physical tangent vector corresponding to the reference edge direction.
    pub tangent: [F; 2],
    /// Physical quadrature point `(x, y)` on the face.
    pub point: [F; 2],
    /// Reference-edge quadrature weight.
    pub weight: F,
}

fn volume_samples_generic<F: Scalar, const NODES_PER_ELEMENT: usize>(
    coords: &[[F; 2]; NODES_PER_ELEMENT],
    quadrature: QuadratureRule,
    shape_fn: fn(F, F) -> [F; NODES_PER_ELEMENT],
    grad_ref_fn: fn(F, F) -> [[F; 2]; NODES_PER_ELEMENT],
) -> Result<Vec<VolumeSample<F, NODES_PER_ELEMENT>>, String> {
    let samples = gauss_volume::<F>(quadrature);
    let mut out = Vec::with_capacity(samples.len());
    for ([xi, eta], weight) in samples {
        let n = shape_fn(xi, eta);
        let grad_reference = grad_ref_fn(xi, eta);
        let jac = jacobian(coords, &grad_reference);
        let inv = inv_j(&jac)?;
        let det = det_j(&jac);
        let point = map_point(coords, &n);
        out.push(VolumeSample {
            n,
            grad_phys: grad_phys(&grad_reference, &inv),
            det_j: det,
            point,
            weight,
        });
    }
    Ok(out)
}

fn face_samples_generic<F: Scalar, const NODES_PER_ELEMENT: usize>(
    coords: &[[F; 2]; NODES_PER_ELEMENT],
    local_face: u8,
    quadrature: QuadratureRule,
    shape_fn: fn(F, F) -> [F; NODES_PER_ELEMENT],
    grad_ref_fn: fn(F, F) -> [[F; 2]; NODES_PER_ELEMENT],
    face_ref_fn: fn(u8, F) -> Result<(F, F, [F; 2]), String>,
) -> Result<Vec<FaceSample<F, NODES_PER_ELEMENT>>, String> {
    let samples = gauss_face::<F>(quadrature);
    let mut out = Vec::with_capacity(samples.len());
    for (s, weight) in samples {
        let (xi, eta, ds_reference) = face_ref_fn(local_face, s)?;
        let n = shape_fn(xi, eta);
        let grad_reference = grad_ref_fn(xi, eta);
        let jac = jacobian(coords, &grad_reference);
        let point = map_point(coords, &n);
        let tangent = face_tangent(&jac, ds_reference);
        let tangent_norm_sq = tangent[0] * tangent[0] + tangent[1] * tangent[1];
        if tangent_norm_sq <= F::zero() {
            return Err(format!(
                "degenerate face tangent on local face {local_face}; tangent squared norm is {tangent_norm_sq:?}"
            ));
        }
        out.push(FaceSample {
            n,
            tangent,
            point,
            weight,
        });
    }
    Ok(out)
}

pub fn volume_samples_quad4<F: Scalar>(
    coords: &[[F; 2]; quad4::NODES_PER_ELEMENT],
    quadrature: QuadratureRule,
) -> Result<Vec<VolumeSample<F, { quad4::NODES_PER_ELEMENT }>>, String> {
    volume_samples_generic(coords, quadrature, quad4::shape::<F>, quad4::grad_ref::<F>)
}

pub fn volume_samples_quad9<F: Scalar>(
    coords: &[[F; 2]; quad9::NODES_PER_ELEMENT],
    quadrature: QuadratureRule,
) -> Result<Vec<VolumeSample<F, { quad9::NODES_PER_ELEMENT }>>, String> {
    volume_samples_generic(coords, quadrature, quad9::shape::<F>, quad9::grad_ref::<F>)
}

pub fn face_samples_quad4<F: Scalar>(
    coords: &[[F; 2]; quad4::NODES_PER_ELEMENT],
    local_face: u8,
    quadrature: QuadratureRule,
) -> Result<Vec<FaceSample<F, { quad4::NODES_PER_ELEMENT }>>, String> {
    face_samples_generic(
        coords,
        local_face,
        quadrature,
        quad4::shape::<F>,
        quad4::grad_ref::<F>,
        quad4::face_reference::<F>,
    )
}

pub fn face_samples_quad9<F: Scalar>(
    coords: &[[F; 2]; quad9::NODES_PER_ELEMENT],
    local_face: u8,
    quadrature: QuadratureRule,
) -> Result<Vec<FaceSample<F, { quad9::NODES_PER_ELEMENT }>>, String> {
    face_samples_generic(
        coords,
        local_face,
        quadrature,
        quad9::shape::<F>,
        quad9::grad_ref::<F>,
        quad9::face_reference::<F>,
    )
}
