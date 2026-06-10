//! Axisymmetric wrappers around generic 2D quadrilateral geometry sampling.
//!
//! The reusable Jacobian, mapping, and quadrature sampling logic lives in [`crate::mesh`]. This
//! module adds the axisymmetric-specific validation and the `2*pi*r`-weighted element summaries
//! needed by the structural solver.

use crate::mesh::QuadMeshView2d;
use crate::physics::solenoid_stress::types::Structural2dFormulation;

pub use crate::mesh::{FaceSample, VolumeSample};

/// Validate that every node radius is admissible for an axisymmetric structural mesh.
pub(crate) fn validate_axisymmetric_nodes<const NODES_PER_ELEMENT: usize>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
) -> Result<(), String> {
    for (index, node) in mesh.nodes_rz.iter().enumerate() {
        if node[0] < 0.0 {
            return Err(format!(
                "node {index} has negative radius {:?}; axisymmetric radius must be nonnegative",
                node[0]
            ));
        }
    }
    Ok(())
}

/// Validate mesh connectivity and the axisymmetric radius constraint together.
pub(crate) fn validate_axisymmetric_mesh<const NODES_PER_ELEMENT: usize>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
) -> Result<(), String> {
    validate_axisymmetric_nodes(mesh)?;
    mesh.validate_connectivity()
}

/// Validate mesh connectivity and formulation-specific coordinate constraints.
pub(crate) fn validate_structural_2d_mesh<const NODES_PER_ELEMENT: usize>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    formulation: Structural2dFormulation,
) -> Result<(), String> {
    match formulation {
        Structural2dFormulation::Axisymmetric => validate_axisymmetric_mesh(mesh),
        Structural2dFormulation::PlaneStrain { thickness } => {
            if thickness <= 0.0 {
                return Err(format!(
                    "plane-strain thickness must be positive; got {thickness:?}"
                ));
            }
            mesh.validate_connectivity()
        }
    }
}
