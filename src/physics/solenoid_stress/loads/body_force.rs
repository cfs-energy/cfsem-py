use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::family::QuadElementFamily;
use crate::physics::solenoid_stress::geometry::{VolumeSample, validate_structural_2d_mesh};
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, Real, Structural2dFormulation, local_dofs,
};

use super::{SparseOperator, scatter_local_matrix};

/// Build the local dense body-force operator for one element.
///
/// The returned block has shape `(DOF_PER_ELEMENT, 2)`. Its two columns correspond to:
/// - column `0`: unit radial body-force density `[force / volume]`
/// - column `1`: unit axial body-force density `[force / volume]`
///
/// Multiplying this block by `[b_r, b_z]^T` gives the element's consistent nodal load vector
/// `[energy / distance]`, which is force-like in the virtual-work sense.
///
/// Each block entry therefore has units of volume.
fn body_force_element_kernel<
    F: Real,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    samples: &[VolumeSample<F, NODES_PER_ELEMENT>],
    formulation: Structural2dFormulation<F>,
) -> Result<[[F; 2]; DOF_PER_ELEMENT], String> {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    let mut local = [[F::zero(); 2]; DOF_PER_ELEMENT];

    for sample in samples {
        let scale = formulation.volume_scale(sample.point, sample.det_j, sample.weight)?;
        for local_node in 0..NODES_PER_ELEMENT {
            // Even-numbered rows act on radial DOFs and odd-numbered rows act on axial DOFs.
            local[2 * local_node][0] = local[2 * local_node][0] + scale * sample.n[local_node];
            local[2 * local_node + 1][1] =
                local[2 * local_node + 1][1] + scale * sample.n[local_node];
        }
    }

    Ok(local)
}

/// Assemble the global body-force-to-RHS operator for one quadrilateral family.
///
/// Output shape: `(2 * mesh.num_nodes(), 2 * mesh.num_elements())`.
///
/// Row meaning:
/// - row `2*a` is the radial generalized-force equation for node `a`,
/// - row `2*a + 1` is the axial generalized-force equation for node `a`.
///
/// Column meaning:
/// - column `2*e` is unit radial body-force density on element `e`,
/// - column `2*e + 1` is unit axial body-force density on element `e`.
///
/// Entry units: `[volume]`.
pub(crate) fn body_force_operator_for_family<
    F: Real,
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
    formulation: Structural2dFormulation<F>,
    quadrature: QuadratureRule,
) -> Result<SparseOperator<F>, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    let ndof = mesh.num_nodes() * 2;
    let ncol = 2 * mesh.num_elements();
    let mut rows = Vec::with_capacity(mesh.num_elements() * DOF_PER_ELEMENT * 2);
    let mut cols = Vec::with_capacity(mesh.num_elements() * DOF_PER_ELEMENT * 2);
    let mut vals = Vec::with_capacity(mesh.num_elements() * DOF_PER_ELEMENT * 2);

    for element_index in 0..mesh.num_elements() {
        let coords = mesh.element_coords(element_index)?;
        let nodes = mesh.element_nodes(element_index)?;
        let samples = Family::volume_samples::<F>(&coords, quadrature)?;
        let local = body_force_element_kernel::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            &samples,
            formulation,
        )?;
        let global_rows = local_dofs::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(&nodes);
        let global_cols = [2 * element_index, 2 * element_index + 1];
        scatter_local_matrix(
            &mut rows,
            &mut cols,
            &mut vals,
            &global_rows,
            &global_cols,
            &local,
        );
    }

    Ok(SparseOperator {
        rows,
        cols,
        vals,
        nrow: ndof,
        ncol,
    })
}
