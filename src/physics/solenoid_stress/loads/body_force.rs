use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::family::QuadElementFamily;
use crate::physics::solenoid_stress::geometry::{VolumeSample, validate_structural_2d_mesh};
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, Structural2dFormulation, local_dofs, scatter_local_matrix,
};

use super::{SparseOperator, collect_sparse_operator_chunks, concat_sparse_operators};

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
fn body_force_element_kernel<const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    samples: &[VolumeSample<f64, NODES_PER_ELEMENT>],
    formulation: Structural2dFormulation,
) -> Result<[[f64; 2]; DOF_PER_ELEMENT], String> {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    let mut local = [[0.0; 2]; DOF_PER_ELEMENT];

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
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    formulation: Structural2dFormulation,
    quadrature: QuadratureRule,
) -> Result<SparseOperator, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    body_force_operator_range_for_family::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
        mesh,
        formulation,
        quadrature,
        0,
        mesh.num_elements(),
    )
}

/// Assemble the global body-force-to-RHS operator using element ranges split across Rayon workers.
pub(crate) fn body_force_operator_for_family_par<
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    formulation: Structural2dFormulation,
    quadrature: QuadratureRule,
) -> Result<SparseOperator, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    let nelem = mesh.num_elements();
    let chunks = collect_sparse_operator_chunks(nelem, |start, end| {
        body_force_operator_range_for_family::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            mesh,
            formulation,
            quadrature,
            start,
            end,
        )
    })?;

    Ok(concat_sparse_operators(
        chunks,
        mesh.num_nodes() * 2,
        2 * nelem,
        nelem * DOF_PER_ELEMENT * 2,
    ))
}

/// Apply body-force amplitudes directly to a reduced RHS without building a sparse operator.
pub(crate) fn apply_body_force_rhs_for_family<
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    formulation: Structural2dFormulation,
    quadrature: QuadratureRule,
    values: &[f64],
    global_to_reduced: &[usize],
    rhs: &mut [f64],
) -> Result<(), String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    let expected = 2 * mesh.num_elements();
    if values.len() != expected {
        return Err(format!(
            "body_force has length {}, but operator expects {expected} values",
            values.len()
        ));
    }
    for element_index in 0..mesh.num_elements() {
        let coords = mesh.element_coords(element_index)?;
        let nodes = mesh.element_nodes(element_index)?;
        let samples = Family::volume_samples(&coords, quadrature)?;
        let local =
            body_force_element_kernel::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(&samples, formulation)?;
        let global_rows = local_dofs::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(&nodes);
        let radial = values[2 * element_index];
        let axial = values[2 * element_index + 1];
        for local_dof in 0..DOF_PER_ELEMENT {
            let reduced_row = global_to_reduced[global_rows[local_dof]];
            if reduced_row != usize::MAX {
                rhs[reduced_row] += local[local_dof][0] * radial + local[local_dof][1] * axial;
            }
        }
    }
    Ok(())
}

fn body_force_operator_range_for_family<
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    formulation: Structural2dFormulation,
    quadrature: QuadratureRule,
    element_start: usize,
    element_end: usize,
) -> Result<SparseOperator, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    let ndof = mesh.num_nodes() * 2;
    let ncol = 2 * mesh.num_elements();
    let nelem = element_end - element_start;
    let mut rows = Vec::with_capacity(nelem * DOF_PER_ELEMENT * 2);
    let mut cols = Vec::with_capacity(nelem * DOF_PER_ELEMENT * 2);
    let mut vals = Vec::with_capacity(nelem * DOF_PER_ELEMENT * 2);

    for element_index in element_start..element_end {
        let coords = mesh.element_coords(element_index)?;
        let nodes = mesh.element_nodes(element_index)?;
        let samples = Family::volume_samples(&coords, quadrature)?;
        let local =
            body_force_element_kernel::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(&samples, formulation)?;
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
