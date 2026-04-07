//! Geometry and quadrature helpers for axisymmetric Quad4 elements.
//!
//! These routines sit between the purely reference-element formulas in [`crate::physics::solenoid_stress::quad4`]
//! and the assembly kernels.  Their job is to evaluate everything at physical quadrature points:
//! position `(r, z)`, Jacobian-derived weights, and shape-function gradients in physical space.

use crate::physics::solenoid_stress::mesh::MeshView;
use crate::physics::solenoid_stress::quad4::{
    NODES_PER_ELEMENT, face_reference, grad_phys, grad_ref, inv_j, jacobian, map_point,
};
use crate::physics::solenoid_stress::quadrature::{QuadratureRule, gauss_face, gauss_volume};
use crate::physics::solenoid_stress::types::{Real, two_pi};

#[derive(Debug, Clone)]
pub struct ElementMeasures<F: Real> {
    /// Planar `(r, z)` area of each element before applying the `2*pi*r` revolution factor.
    pub areas: Vec<F>,
    /// Physical 3D volume swept out when each planar element is revolved about the axis.
    pub swept_volumes: Vec<F>,
}

#[derive(Debug, Clone)]
pub struct ElementQuadrature<F: Real> {
    /// Quadrature-point coordinates `(r, z)` in element-major order.
    pub points_rz: Vec<[F; 2]>,
    /// Quadrature weights for integrating over the planar `(r, z)` cross-section.
    pub weights_area: Vec<F>,
    /// Quadrature weights for integrating over the full axisymmetric 3D volume.
    pub weights_volume: Vec<F>,
    /// Number of quadrature points contributed by each element.
    pub nq_per_element: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct VolumeSample<F: Real> {
    /// Shape-function values at the quadrature point.
    pub n: [F; NODES_PER_ELEMENT],
    /// Physical gradients `[dN_i/dr, dN_i/dz]`.
    pub grad_phys: [[F; 2]; NODES_PER_ELEMENT],
    /// Jacobian determinant `det(J)` of the reference-to-physical map.
    pub det_j: F,
    /// Physical quadrature point `(r, z)`.
    pub point: [F; 2],
    /// Reference-space quadrature weight `w`.
    pub weight: F,
}

#[derive(Debug, Clone, Copy)]
pub struct FaceSample<F: Real> {
    /// Shape-function values on the element face.
    pub n: [F; NODES_PER_ELEMENT],
    /// Physical tangent vector corresponding to the reference edge direction.
    pub tangent: [F; 2],
    /// Physical quadrature point `(r, z)` on the face.
    pub point: [F; 2],
    /// Reference-edge quadrature weight.
    pub weight: F,
}

/// Evaluate all volume quadrature samples for one Quad4 element.
///
/// This converts the reference Gauss rule into physical-space samples suitable for both
/// stiffness integration and body-force integration.
pub fn volume_samples<F: Real>(
    coords: &[[F; 2]; NODES_PER_ELEMENT],
    quadrature: QuadratureRule,
) -> Result<Vec<VolumeSample<F>>, String> {
    let samples = gauss_volume::<F>(quadrature);
    let mut out = Vec::with_capacity(samples.len());
    for ([xi, eta], weight) in samples {
        let n = crate::physics::solenoid_stress::quad4::shape(xi, eta);
        let grad_reference = grad_ref(xi, eta);
        let jac = jacobian(coords, &grad_reference);
        let inv = inv_j(&jac)?;
        let det = crate::physics::solenoid_stress::quad4::det_j(&jac);
        let point = map_point(coords, &n);
        if point[0] < F::zero() {
            return Err(format!(
                "quadrature point has negative radius {:?}; axisymmetric radius must be nonnegative",
                point[0]
            ));
        }
        out.push(VolumeSample {
            n,
            // This is the chain-rule step that converts reference derivatives into
            // physical derivatives used by the elasticity equations.
            grad_phys: grad_phys(&grad_reference, &inv),
            det_j: det,
            point,
            weight,
        });
    }
    Ok(out)
}

/// Evaluate all quadrature samples on one local element face.
///
/// The returned tangent vector already includes the Jacobian mapping from the reference edge.
/// Rotating that tangent by 90 degrees yields the signed area-normal vector used for pressure
/// loads.
pub fn face_samples<F: Real>(
    coords: &[[F; 2]; NODES_PER_ELEMENT],
    local_face: u8,
    quadrature: QuadratureRule,
) -> Result<Vec<FaceSample<F>>, String> {
    let samples = gauss_face::<F>(quadrature);
    let mut out = Vec::with_capacity(samples.len());
    for (s, weight) in samples {
        let (xi, eta, ds_reference) = face_reference(local_face, s)?;
        let n = crate::physics::solenoid_stress::quad4::shape(xi, eta);
        let grad_reference = grad_ref(xi, eta);
        let jac = jacobian(coords, &grad_reference);
        let point = map_point(coords, &n);
        if point[0] < F::zero() {
            return Err(format!(
                "face quadrature point has negative radius {:?}; axisymmetric radius must be nonnegative",
                point[0]
            ));
        }
        let tangent = [
            jac[0][0] * ds_reference[0] + jac[0][1] * ds_reference[1],
            jac[1][0] * ds_reference[0] + jac[1][1] * ds_reference[1],
        ];
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

/// Compute per-element planar areas and revolved volumes.
///
/// This is mainly diagnostic/output data, but it also makes the axisymmetric measure explicit:
/// area integrals use `det(J) w`, while 3D volume integrals use `2*pi*r*det(J) w`.
pub fn element_measures<F: Real>(
    mesh: MeshView<'_, F>,
    quadrature: QuadratureRule,
) -> Result<ElementMeasures<F>, String> {
    mesh.validate_nodes()?;
    mesh.validate_connectivity()?;
    let mut areas = Vec::with_capacity(mesh.num_elements());
    let mut swept_volumes = Vec::with_capacity(mesh.num_elements());
    let two_pi = two_pi::<F>();
    for element_index in 0..mesh.num_elements() {
        let coords = mesh.element_coords(element_index)?;
        let mut area = F::zero();
        let mut swept = F::zero();
        for sample in volume_samples(&coords, quadrature)? {
            area = area + sample.det_j * sample.weight;
            swept = swept + two_pi * sample.point[0] * sample.det_j * sample.weight;
        }
        areas.push(area);
        swept_volumes.push(swept);
    }
    Ok(ElementMeasures {
        areas,
        swept_volumes,
    })
}

/// Export raw quadrature-point coordinates and weights for every element.
///
/// This is useful for tests and for Python-side diagnostics that want to inspect the integration
/// grid directly rather than only the assembled matrix.
pub fn element_quadrature<F: Real>(
    mesh: MeshView<'_, F>,
    quadrature: QuadratureRule,
) -> Result<ElementQuadrature<F>, String> {
    mesh.validate_nodes()?;
    mesh.validate_connectivity()?;
    let nq = quadrature.points_per_element();
    let mut points_rz = Vec::with_capacity(mesh.num_elements() * nq);
    let mut weights_area = Vec::with_capacity(mesh.num_elements() * nq);
    let mut weights_volume = Vec::with_capacity(mesh.num_elements() * nq);
    let two_pi = two_pi::<F>();
    for element_index in 0..mesh.num_elements() {
        let coords = mesh.element_coords(element_index)?;
        for sample in volume_samples(&coords, quadrature)? {
            points_rz.push(sample.point);
            weights_area.push(sample.det_j * sample.weight);
            weights_volume.push(two_pi * sample.point[0] * sample.det_j * sample.weight);
        }
    }
    Ok(ElementQuadrature {
        points_rz,
        weights_area,
        weights_volume,
        nq_per_element: nq,
    })
}
