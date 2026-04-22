//! Axisymmetric wrappers around generic 2D quadrilateral geometry sampling.
//!
//! The reusable Jacobian, mapping, and quadrature sampling logic lives in [`crate::mesh`]. This
//! module adds the axisymmetric-specific validation and the `2*pi*r`-weighted element summaries
//! needed by the structural solver.

use crate::mesh::elements::quad2d::{quad4, quad9};
use crate::mesh::sampling;
use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::types::Real;

pub use crate::mesh::{FaceSample, VolumeSample};

/// Validate that every node radius is admissible for an axisymmetric structural mesh.
pub(crate) fn validate_axisymmetric_nodes<F: Real, const NODES_PER_ELEMENT: usize>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
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

/// Validate mesh connectivity and the axisymmetric radius constraint together.
pub(crate) fn validate_axisymmetric_mesh<F: Real, const NODES_PER_ELEMENT: usize>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
) -> Result<(), String> {
    validate_axisymmetric_nodes(mesh)?;
    mesh.validate_connectivity()
}

/// Reject volume quadrature samples whose evaluated radius is invalid for axisymmetry.
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

/// Reject face quadrature samples whose evaluated radius is invalid for axisymmetry.
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

/// Evaluate axisymmetric volume quadrature samples for one `quad4` element.
///
/// Returned sample points live in meridian coordinates `(r, z)` with `r >= 0`.
pub fn volume_samples_quad4<F: Real>(
    coords: &[[F; 2]; quad4::NODES_PER_ELEMENT],
    quadrature: QuadratureRule,
) -> Result<Vec<VolumeSample<F, { quad4::NODES_PER_ELEMENT }>>, String> {
    validate_axisymmetric_volume_samples(sampling::volume_samples_quad4(coords, quadrature)?)
}

/// Evaluate axisymmetric volume quadrature samples for one `quad9` element.
///
/// Returned sample points live in meridian coordinates `(r, z)` with `r >= 0`.
pub fn volume_samples_quad9<F: Real>(
    coords: &[[F; 2]; quad9::NODES_PER_ELEMENT],
    quadrature: QuadratureRule,
) -> Result<Vec<VolumeSample<F, { quad9::NODES_PER_ELEMENT }>>, String> {
    validate_axisymmetric_volume_samples(sampling::volume_samples_quad9(coords, quadrature)?)
}

/// Evaluate axisymmetric face quadrature samples for one `quad4` element face.
///
/// Returned sample points live in meridian coordinates `(r, z)` with `r >= 0`.
pub fn face_samples_quad4<F: Real>(
    coords: &[[F; 2]; quad4::NODES_PER_ELEMENT],
    local_face: u8,
    quadrature: QuadratureRule,
) -> Result<Vec<FaceSample<F, { quad4::NODES_PER_ELEMENT }>>, String> {
    validate_axisymmetric_face_samples(sampling::face_samples_quad4(
        coords, local_face, quadrature,
    )?)
}

/// Evaluate axisymmetric face quadrature samples for one `quad9` element face.
///
/// Returned sample points live in meridian coordinates `(r, z)` with `r >= 0`.
pub fn face_samples_quad9<F: Real>(
    coords: &[[F; 2]; quad9::NODES_PER_ELEMENT],
    local_face: u8,
    quadrature: QuadratureRule,
) -> Result<Vec<FaceSample<F, { quad9::NODES_PER_ELEMENT }>>, String> {
    validate_axisymmetric_face_samples(sampling::face_samples_quad9(
        coords, local_face, quadrature,
    )?)
}
