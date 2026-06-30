//! Family-specific quadrilateral sampling and metadata for the structural 2D FEM backend.

use crate::mesh::elements::quad2d::quadrature::gauss_volume;
use crate::mesh::elements::quad2d::{quad4, quad9};
use crate::mesh::quad2d::{Quad4ReferenceElement, Quad9ReferenceElement, QuadReferenceElement};
use crate::mesh::{QuadratureRule, sampling};
use crate::physics::solenoid_stress::geometry::{FaceSample, VolumeSample};
use crate::physics::solenoid_stress::model::Structural2dElementType;

/// Family-specific differences that remain after stiffness, load, and recovery assembly have been
/// written once generically.
///
/// This trait owns only:
/// - element-type metadata exposed through the public model,
/// - family-specific volume sampling,
/// - family-specific face sampling, and
/// - connectivity flattening for Python-facing metadata export.
pub(crate) trait QuadElementFamily<const NODES_PER_ELEMENT: usize> {
    /// Reference-element implementation used by mesh query and recovery operators.
    type ReferenceElement: QuadReferenceElement<NODES_PER_ELEMENT>;

    /// Return the public element-family tag corresponding to this internal family marker.
    fn element_type() -> Structural2dElementType;

    /// Flatten element connectivity into one element-major `Vec<usize>` for metadata export.
    fn flatten_elements(elements: &[[usize; NODES_PER_ELEMENT]]) -> Vec<usize> {
        let mut flat = Vec::with_capacity(elements.len() * NODES_PER_ELEMENT);
        for conn in elements {
            flat.extend_from_slice(conn);
        }
        flat
    }

    /// Reference points whose Jacobian sign certifies that one element is non-degenerate.
    ///
    /// A bilinear quad4 maps the reference square with a Jacobian that is itself bilinear, so its
    /// sign is pinned by its four corner values: testing the corners is exactly sufficient. A
    /// biquadratic quad9 has a
    /// higher-order Jacobian that a mid-side node can drive negative while the corners stay
    /// positive, so it falls back to the volume Gauss points -- the same admissibility gate the
    /// assembler imposes when it integrates the element.
    fn jacobian_check_points(quadrature: QuadratureRule) -> Vec<[f64; 2]>;

    /// Evaluate the family-specific volume quadrature samples for one element.
    fn volume_samples(
        coords: &[[f64; 2]; NODES_PER_ELEMENT],
        quadrature: QuadratureRule,
    ) -> Result<Vec<VolumeSample<f64, NODES_PER_ELEMENT>>, String>;

    /// Evaluate the family-specific face quadrature samples for one element face.
    fn face_samples(
        coords: &[[f64; 2]; NODES_PER_ELEMENT],
        local_face: u8,
        quadrature: QuadratureRule,
    ) -> Result<Vec<FaceSample<f64, NODES_PER_ELEMENT>>, String>;
}

/// Marker type for the bilinear four-node quadrilateral family.
pub(crate) struct Quad4Family;

impl QuadElementFamily<{ quad4::NODES_PER_ELEMENT }> for Quad4Family {
    type ReferenceElement = Quad4ReferenceElement;

    /// Identify this marker as the public `quad4` family.
    fn element_type() -> Structural2dElementType {
        Structural2dElementType::Quad4
    }

    /// The four reference corners, walked counter-clockwise; their Jacobian sign is sufficient.
    fn jacobian_check_points(_quadrature: QuadratureRule) -> Vec<[f64; 2]> {
        vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    }

    /// Delegate `quad4` volume sampling to the generic mesh sampling layer.
    fn volume_samples(
        coords: &[[f64; 2]; quad4::NODES_PER_ELEMENT],
        quadrature: QuadratureRule,
    ) -> Result<Vec<VolumeSample<f64, { quad4::NODES_PER_ELEMENT }>>, String> {
        sampling::volume_samples_quad4(coords, quadrature)
    }

    /// Delegate `quad4` face sampling to the generic mesh sampling layer.
    fn face_samples(
        coords: &[[f64; 2]; quad4::NODES_PER_ELEMENT],
        local_face: u8,
        quadrature: QuadratureRule,
    ) -> Result<Vec<FaceSample<f64, { quad4::NODES_PER_ELEMENT }>>, String> {
        sampling::face_samples_quad4(coords, local_face, quadrature)
    }
}

/// Marker type for the quadratic nine-node quadrilateral family.
pub(crate) struct Quad9Family;

impl QuadElementFamily<{ quad9::NODES_PER_ELEMENT }> for Quad9Family {
    type ReferenceElement = Quad9ReferenceElement;

    /// Identify this marker as the public `quad9` family.
    fn element_type() -> Structural2dElementType {
        Structural2dElementType::Quad9
    }

    /// The volume Gauss points, matching the gate the assembler imposes during integration.
    fn jacobian_check_points(quadrature: QuadratureRule) -> Vec<[f64; 2]> {
        gauss_volume::<f64>(quadrature)
            .into_iter()
            .map(|(reference, _weight)| reference)
            .collect()
    }

    /// Delegate `quad9` volume sampling to the generic mesh sampling layer.
    fn volume_samples(
        coords: &[[f64; 2]; quad9::NODES_PER_ELEMENT],
        quadrature: QuadratureRule,
    ) -> Result<Vec<VolumeSample<f64, { quad9::NODES_PER_ELEMENT }>>, String> {
        sampling::volume_samples_quad9(coords, quadrature)
    }

    /// Delegate `quad9` face sampling to the generic mesh sampling layer.
    fn face_samples(
        coords: &[[f64; 2]; quad9::NODES_PER_ELEMENT],
        local_face: u8,
        quadrature: QuadratureRule,
    ) -> Result<Vec<FaceSample<f64, { quad9::NODES_PER_ELEMENT }>>, String> {
        sampling::face_samples_quad9(coords, local_face, quadrature)
    }
}
