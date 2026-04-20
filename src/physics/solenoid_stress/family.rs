//! Family-specific quadrilateral sampling and metadata for the axisymmetric FEM backend.

use crate::mesh::QuadratureRule;
use crate::mesh::elements::quad2d::{quad4, quad9};
use crate::physics::solenoid_stress::geometry::{
    FaceSample, VolumeSample, face_samples_quad4, face_samples_quad9, volume_samples_quad4,
    volume_samples_quad9,
};
use crate::physics::solenoid_stress::model::AxisymmetricElementType;
use crate::physics::solenoid_stress::types::Real;

/// Family-specific differences that remain after stiffness, load, and recovery assembly have been
/// written once generically.
///
/// This trait owns only:
/// - element-type metadata exposed through the public model,
/// - family-specific volume sampling,
/// - family-specific face sampling, and
/// - connectivity flattening for Python-facing metadata export.
pub(crate) trait QuadElementFamily<const NODES_PER_ELEMENT: usize> {
    /// Return the public element-family tag corresponding to this internal family marker.
    fn element_type() -> AxisymmetricElementType;

    /// Flatten element connectivity into one element-major `Vec<usize>` for metadata export.
    fn flatten_elements(elements: &[[usize; NODES_PER_ELEMENT]]) -> Vec<usize> {
        let mut flat = Vec::with_capacity(elements.len() * NODES_PER_ELEMENT);
        for conn in elements {
            flat.extend_from_slice(conn);
        }
        flat
    }

    /// Evaluate the family-specific volume quadrature samples for one element.
    fn volume_samples<F: Real>(
        coords: &[[F; 2]; NODES_PER_ELEMENT],
        quadrature: QuadratureRule,
    ) -> Result<Vec<VolumeSample<F, NODES_PER_ELEMENT>>, String>;

    /// Evaluate the family-specific face quadrature samples for one element face.
    fn face_samples<F: Real>(
        coords: &[[F; 2]; NODES_PER_ELEMENT],
        local_face: u8,
        quadrature: QuadratureRule,
    ) -> Result<Vec<FaceSample<F, NODES_PER_ELEMENT>>, String>;
}

/// Marker type for the bilinear four-node quadrilateral family.
pub(crate) struct Quad4Family;

impl QuadElementFamily<{ quad4::NODES_PER_ELEMENT }> for Quad4Family {
    /// Identify this marker as the public `quad4` family.
    fn element_type() -> AxisymmetricElementType {
        AxisymmetricElementType::Quad4
    }

    /// Delegate `quad4` volume sampling to the generic mesh sampling layer.
    fn volume_samples<F: Real>(
        coords: &[[F; 2]; quad4::NODES_PER_ELEMENT],
        quadrature: QuadratureRule,
    ) -> Result<Vec<VolumeSample<F, { quad4::NODES_PER_ELEMENT }>>, String> {
        volume_samples_quad4(coords, quadrature)
    }

    /// Delegate `quad4` face sampling to the generic mesh sampling layer.
    fn face_samples<F: Real>(
        coords: &[[F; 2]; quad4::NODES_PER_ELEMENT],
        local_face: u8,
        quadrature: QuadratureRule,
    ) -> Result<Vec<FaceSample<F, { quad4::NODES_PER_ELEMENT }>>, String> {
        face_samples_quad4(coords, local_face, quadrature)
    }
}

/// Marker type for the quadratic nine-node quadrilateral family.
pub(crate) struct Quad9Family;

impl QuadElementFamily<{ quad9::NODES_PER_ELEMENT }> for Quad9Family {
    /// Identify this marker as the public `quad9` family.
    fn element_type() -> AxisymmetricElementType {
        AxisymmetricElementType::Quad9
    }

    /// Delegate `quad9` volume sampling to the generic mesh sampling layer.
    fn volume_samples<F: Real>(
        coords: &[[F; 2]; quad9::NODES_PER_ELEMENT],
        quadrature: QuadratureRule,
    ) -> Result<Vec<VolumeSample<F, { quad9::NODES_PER_ELEMENT }>>, String> {
        volume_samples_quad9(coords, quadrature)
    }

    /// Delegate `quad9` face sampling to the generic mesh sampling layer.
    fn face_samples<F: Real>(
        coords: &[[F; 2]; quad9::NODES_PER_ELEMENT],
        local_face: u8,
        quadrature: QuadratureRule,
    ) -> Result<Vec<FaceSample<F, { quad9::NODES_PER_ELEMENT }>>, String> {
        face_samples_quad9(coords, local_face, quadrature)
    }
}
