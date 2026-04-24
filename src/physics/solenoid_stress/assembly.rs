//! Axisymmetric element assembly for small-strain elasticity.
//!
//! The element system matrix is assembled in Galerkin form
//! `K_e = integral(B^T D B 2*pi*r dA)`.

use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::axisym::{accumulate_stiffness, build_b_matrix};
use crate::physics::solenoid_stress::convenience::rotate_material_in_plane;
use crate::physics::solenoid_stress::family::QuadElementFamily;
use crate::physics::solenoid_stress::geometry::validate_structural_2d_mesh;
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, Real, StiffnessTriplets, Structural2dFormulation, local_dofs,
};

/// Assemble the full unconstrained stiffness matrix for one quadrilateral family.
///
/// The returned triplets describe the global structural stiffness operator before Dirichlet
/// reduction and CSC compression.  Its entries have units
/// `[generalized nodal force / displacement] = [energy / distance^2]`.
pub(crate) fn assemble_stiffness_for_family<
    F: Real,
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    material_orientation_angles: Option<&[F]>,
    formulation: Structural2dFormulation<F>,
    quadrature: QuadratureRule,
) -> Result<StiffnessTriplets<F>, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    if material_ids.len() != mesh.num_elements() {
        return Err(format!(
            "material_ids has length {}, but mesh has {} elements",
            material_ids.len(),
            mesh.num_elements()
        ));
    }
    if let Some(angles) = material_orientation_angles
        && angles.len() != mesh.num_elements()
    {
        return Err(format!(
            "material_orientation_angles has length {}, but mesh has {} elements",
            angles.len(),
            mesh.num_elements()
        ));
    }
    let mut rows = Vec::with_capacity(mesh.num_elements() * DOF_PER_ELEMENT * DOF_PER_ELEMENT);
    let mut cols = Vec::with_capacity(mesh.num_elements() * DOF_PER_ELEMENT * DOF_PER_ELEMENT);
    let mut vals = Vec::with_capacity(mesh.num_elements() * DOF_PER_ELEMENT * DOF_PER_ELEMENT);

    for element_index in 0..mesh.num_elements() {
        let coords = mesh.element_coords(element_index)?;
        let nodes = mesh.element_nodes(element_index)?;
        let material_id = material_ids[element_index];
        let material = material_table.get(material_id).ok_or_else(|| {
            format!("material_id {material_id} on element {element_index} is out of range")
        })?;
        let material_storage;
        let material = if let Some(angles) = material_orientation_angles {
            material_storage = rotate_material_in_plane(material, angles[element_index]);
            &material_storage
        } else {
            material
        };
        let mut ke = [[F::zero(); DOF_PER_ELEMENT]; DOF_PER_ELEMENT];

        for sample in Family::volume_samples::<F>(&coords, quadrature)? {
            let b = build_b_matrix::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                formulation,
                &sample.n,
                &sample.grad_phys,
                sample.point,
            )?;
            let scale = formulation.volume_scale(sample.point, sample.det_j, sample.weight)?;
            accumulate_stiffness(&mut ke, material, &b, scale);
        }

        let global_dofs = local_dofs::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(&nodes);

        for row in 0..DOF_PER_ELEMENT {
            for col in 0..DOF_PER_ELEMENT {
                rows.push(global_dofs[row]);
                cols.push(global_dofs[col]);
                vals.push(ke[row][col]);
            }
        }
    }

    Ok(StiffnessTriplets { rows, cols, vals })
}

#[cfg(test)]
mod tests {
    use super::assemble_stiffness_for_family;
    use crate::mesh::QuadratureRule;
    use crate::physics::solenoid_stress::convenience::isotropic_axisymmetric_material;
    use crate::physics::solenoid_stress::family::Quad4Family;
    use crate::physics::solenoid_stress::test_utils::single_element_quad4_mesh;
    use crate::physics::solenoid_stress::types::dof_per_element;

    #[test]
    /// Check that one assembled element produces a symmetric stiffness matrix.
    fn single_element_has_symmetric_stiffness() {
        let mesh = single_element_quad4_mesh();
        let material_ids = [0usize];
        let material_table = [isotropic_axisymmetric_material(200.0e9, 0.27)];
        let result = assemble_stiffness_for_family::<f64, Quad4Family, 4, { dof_per_element(4) }>(
            mesh,
            &material_ids,
            &material_table,
            None,
            crate::physics::solenoid_stress::types::Structural2dFormulation::Axisymmetric,
            QuadratureRule::GaussLegendre3,
        )
        .expect("assembly should succeed");

        let mut dense = [[0.0; 8]; 8];
        for ((row, col), val) in result.rows.iter().zip(&result.cols).zip(&result.vals) {
            dense[*row][*col] += *val;
        }
        for row in 0..8 {
            for col in 0..8 {
                let scale = dense[row][col].abs().max(dense[col][row].abs()).max(1.0);
                assert!((dense[row][col] - dense[col][row]).abs() / scale < 1.0e-12);
            }
        }
    }
}
