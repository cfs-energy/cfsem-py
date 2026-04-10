//! Axisymmetric wrappers around generic 2D quadrilateral geometry sampling.
//!
//! The reusable Jacobian, mapping, and quadrature sampling logic lives in [`crate::mesh`]. This
//! module adds the axisymmetric-specific validation and the `2*pi*r`-weighted element summaries
//! needed by the structural solver.

use crate::mesh::elements::quad2d::{quad4, quad9};
use crate::mesh::sampling;
use crate::mesh::{MeshView, QuadratureRule};
use crate::physics::solenoid_stress::types::{Real, two_pi};

pub use crate::mesh::{FaceSample, VolumeSample};

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

pub(crate) fn validate_axisymmetric_nodes<F: Real, const NODES_PER_ELEMENT: usize>(
    mesh: MeshView<'_, F, NODES_PER_ELEMENT>,
) -> Result<(), String> {
    for (index, node) in mesh.nodes_rz.iter().enumerate() {
        if node[0] < F::zero() {
            return Err(format!(
                "node {index} has negative radius {:?}; axisymmetric radius must be nonnegative",
                node[0]
            ));
        }
    }
    Ok(())
}

fn validate_axisymmetric_volume_samples<F: Real, const NODES_PER_ELEMENT: usize>(
    samples: Vec<VolumeSample<F, NODES_PER_ELEMENT>>,
) -> Result<Vec<VolumeSample<F, NODES_PER_ELEMENT>>, String> {
    for sample in &samples {
        if sample.point[0] < F::zero() {
            return Err(format!(
                "quadrature point has negative radius {:?}; axisymmetric radius must be nonnegative",
                sample.point[0]
            ));
        }
    }
    Ok(samples)
}

fn validate_axisymmetric_face_samples<F: Real, const NODES_PER_ELEMENT: usize>(
    samples: Vec<FaceSample<F, NODES_PER_ELEMENT>>,
) -> Result<Vec<FaceSample<F, NODES_PER_ELEMENT>>, String> {
    for sample in &samples {
        if sample.point[0] < F::zero() {
            return Err(format!(
                "face quadrature point has negative radius {:?}; axisymmetric radius must be nonnegative",
                sample.point[0]
            ));
        }
    }
    Ok(samples)
}

pub fn volume_samples_quad4<F: Real>(
    coords: &[[F; 2]; quad4::NODES_PER_ELEMENT],
    quadrature: QuadratureRule,
) -> Result<Vec<VolumeSample<F, { quad4::NODES_PER_ELEMENT }>>, String> {
    validate_axisymmetric_volume_samples(sampling::volume_samples_quad4(coords, quadrature)?)
}

pub fn volume_samples_quad9<F: Real>(
    coords: &[[F; 2]; quad9::NODES_PER_ELEMENT],
    quadrature: QuadratureRule,
) -> Result<Vec<VolumeSample<F, { quad9::NODES_PER_ELEMENT }>>, String> {
    validate_axisymmetric_volume_samples(sampling::volume_samples_quad9(coords, quadrature)?)
}

pub fn face_samples_quad4<F: Real>(
    coords: &[[F; 2]; quad4::NODES_PER_ELEMENT],
    local_face: u8,
    quadrature: QuadratureRule,
) -> Result<Vec<FaceSample<F, { quad4::NODES_PER_ELEMENT }>>, String> {
    validate_axisymmetric_face_samples(sampling::face_samples_quad4(
        coords, local_face, quadrature,
    )?)
}

pub fn face_samples_quad9<F: Real>(
    coords: &[[F; 2]; quad9::NODES_PER_ELEMENT],
    local_face: u8,
    quadrature: QuadratureRule,
) -> Result<Vec<FaceSample<F, { quad9::NODES_PER_ELEMENT }>>, String> {
    validate_axisymmetric_face_samples(sampling::face_samples_quad9(
        coords, local_face, quadrature,
    )?)
}

fn element_measures_generic<F: Real, const NODES_PER_ELEMENT: usize>(
    mesh: MeshView<'_, F, NODES_PER_ELEMENT>,
    quadrature: QuadratureRule,
    volume_samples_fn: fn(
        &[[F; 2]; NODES_PER_ELEMENT],
        QuadratureRule,
    ) -> Result<Vec<VolumeSample<F, NODES_PER_ELEMENT>>, String>,
) -> Result<ElementMeasures<F>, String> {
    validate_axisymmetric_nodes(mesh)?;
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
    validate_axisymmetric_nodes(mesh)?;
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
