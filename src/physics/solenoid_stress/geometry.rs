//! Geometry and quadrature helpers for axisymmetric quadrilateral elements.
//!
//! These routines sit between the purely reference-element formulas in the element modules and
//! the assembly kernels. Their job is to evaluate everything at physical quadrature points:
//! position `(r, z)`, Jacobian-derived weights, and shape-function gradients in physical space.

use crate::physics::solenoid_stress::mesh::MeshView;
use crate::physics::solenoid_stress::quadrature::{QuadratureRule, gauss_face, gauss_volume};
use crate::physics::solenoid_stress::types::{Real, two_pi};
use crate::physics::solenoid_stress::{quad4, quad9};

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
pub struct VolumeSample<F: Real, const NODES_PER_ELEMENT: usize> {
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
pub struct FaceSample<F: Real, const NODES_PER_ELEMENT: usize> {
    /// Shape-function values on the element face.
    pub n: [F; NODES_PER_ELEMENT],
    /// Physical tangent vector corresponding to the reference edge direction.
    pub tangent: [F; 2],
    /// Physical quadrature point `(r, z)` on the face.
    pub point: [F; 2],
    /// Reference-edge quadrature weight.
    pub weight: F,
}

fn map_point<F: Real, const NODES_PER_ELEMENT: usize>(
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

fn jacobian<F: Real, const NODES_PER_ELEMENT: usize>(
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

fn det_j<F: Real>(jac: &[[F; 2]; 2]) -> F {
    jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]
}

fn inv_j<F: Real>(jac: &[[F; 2]; 2]) -> Result<[[F; 2]; 2], String> {
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

fn grad_phys<F: Real, const NODES_PER_ELEMENT: usize>(
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

fn volume_samples_generic<F: Real, const NODES_PER_ELEMENT: usize>(
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
        if point[0] < F::zero() {
            return Err(format!(
                "quadrature point has negative radius {:?}; axisymmetric radius must be nonnegative",
                point[0]
            ));
        }
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

fn face_samples_generic<F: Real, const NODES_PER_ELEMENT: usize>(
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

pub fn volume_samples_quad4<F: Real>(
    coords: &[[F; 2]; quad4::NODES_PER_ELEMENT],
    quadrature: QuadratureRule,
) -> Result<Vec<VolumeSample<F, { quad4::NODES_PER_ELEMENT }>>, String> {
    volume_samples_generic(coords, quadrature, quad4::shape::<F>, quad4::grad_ref::<F>)
}

pub fn volume_samples_quad9<F: Real>(
    coords: &[[F; 2]; quad9::NODES_PER_ELEMENT],
    quadrature: QuadratureRule,
) -> Result<Vec<VolumeSample<F, { quad9::NODES_PER_ELEMENT }>>, String> {
    volume_samples_generic(coords, quadrature, quad9::shape::<F>, quad9::grad_ref::<F>)
}

pub fn face_samples_quad4<F: Real>(
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

pub fn face_samples_quad9<F: Real>(
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

fn element_measures_generic<F: Real, const NODES_PER_ELEMENT: usize>(
    mesh: MeshView<'_, F, NODES_PER_ELEMENT>,
    quadrature: QuadratureRule,
    volume_samples_fn: fn(
        &[[F; 2]; NODES_PER_ELEMENT],
        QuadratureRule,
    ) -> Result<Vec<VolumeSample<F, NODES_PER_ELEMENT>>, String>,
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
        for sample in volume_samples_fn(&coords, quadrature)? {
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

pub fn element_measures_quad4<F: Real>(
    mesh: MeshView<'_, F, { quad4::NODES_PER_ELEMENT }>,
    quadrature: QuadratureRule,
) -> Result<ElementMeasures<F>, String> {
    element_measures_generic(mesh, quadrature, volume_samples_quad4::<F>)
}

pub fn element_measures_quad9<F: Real>(
    mesh: MeshView<'_, F, { quad9::NODES_PER_ELEMENT }>,
    quadrature: QuadratureRule,
) -> Result<ElementMeasures<F>, String> {
    element_measures_generic(mesh, quadrature, volume_samples_quad9::<F>)
}

fn element_quadrature_generic<F: Real, const NODES_PER_ELEMENT: usize>(
    mesh: MeshView<'_, F, NODES_PER_ELEMENT>,
    quadrature: QuadratureRule,
    volume_samples_fn: fn(
        &[[F; 2]; NODES_PER_ELEMENT],
        QuadratureRule,
    ) -> Result<Vec<VolumeSample<F, NODES_PER_ELEMENT>>, String>,
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
        for sample in volume_samples_fn(&coords, quadrature)? {
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

pub fn element_quadrature_quad4<F: Real>(
    mesh: MeshView<'_, F, { quad4::NODES_PER_ELEMENT }>,
    quadrature: QuadratureRule,
) -> Result<ElementQuadrature<F>, String> {
    element_quadrature_generic(mesh, quadrature, volume_samples_quad4::<F>)
}

pub fn element_quadrature_quad9<F: Real>(
    mesh: MeshView<'_, F, { quad9::NODES_PER_ELEMENT }>,
    quadrature: QuadratureRule,
) -> Result<ElementQuadrature<F>, String> {
    element_quadrature_generic(mesh, quadrature, volume_samples_quad9::<F>)
}
