//! Axisymmetric element assembly for small-strain elasticity.
//!
//! The element system matrix is assembled in Galerkin form
//! `K_e = integral(B^T D B 2*pi*r dA)`.

use rayon::prelude::*;

use crate::chunksize;
use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::axisym::{accumulate_stiffness, build_b_matrix};
use crate::physics::solenoid_stress::convenience::rotate_material_in_plane;
use crate::physics::solenoid_stress::family::QuadElementFamily;
use crate::physics::solenoid_stress::geometry::validate_structural_2d_mesh;
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, Real, StiffnessTriplets, Structural2dFormulation, local_dofs,
    validate_element_material_inputs,
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
    validate_stiffness_inputs::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
        mesh,
        material_ids,
        material_orientation_angles,
        formulation,
    )?;
    assemble_stiffness_range_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
        mesh,
        material_ids,
        material_table,
        material_orientation_angles,
        formulation,
        quadrature,
        0,
        mesh.num_elements(),
    )
}

/// Assemble full-space stiffness triplets with element ranges split across Rayon workers.
///
/// Each worker owns an independent triplet buffer, so the element kernel has no shared mutable
/// state.  The per-range triplets are concatenated in element order before the existing
/// constrained-system reduction and sparse compression stages consume them.
pub(crate) fn assemble_stiffness_for_family_par<
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
    validate_stiffness_inputs::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
        mesh,
        material_ids,
        material_orientation_angles,
        formulation,
    )?;
    let nelem = mesh.num_elements();
    let chunk = chunksize(nelem);
    let mut ranges = Vec::with_capacity(nelem.div_ceil(chunk));
    let mut start = 0;
    while start < nelem {
        let end = (start + chunk).min(nelem);
        ranges.push((start, end));
        start = end;
    }

    let chunks = ranges
        .into_par_iter()
        .map(|(start, end)| {
            assemble_stiffness_range_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                material_ids,
                material_table,
                material_orientation_angles,
                formulation,
                quadrature,
                start,
                end,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(concat_stiffness_triplets(
        chunks,
        nelem * DOF_PER_ELEMENT * DOF_PER_ELEMENT,
    ))
}

/// Validate shape and material inputs shared by serial and threaded stiffness assembly.
fn validate_stiffness_inputs<
    F: Real,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
    material_ids: &[usize],
    material_orientation_angles: Option<&[F]>,
    formulation: Structural2dFormulation<F>,
) -> Result<(), String> {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    validate_element_material_inputs(
        mesh.num_elements(),
        material_ids,
        material_orientation_angles,
    )
}

#[allow(clippy::too_many_arguments)]
/// Assemble the contribution from a contiguous element range into an independent triplet buffer.
fn assemble_stiffness_range_for_family<
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
    element_start: usize,
    element_end: usize,
) -> Result<StiffnessTriplets<F>, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    let nelem = element_end - element_start;
    let mut rows = Vec::with_capacity(nelem * DOF_PER_ELEMENT * DOF_PER_ELEMENT);
    let mut cols = Vec::with_capacity(nelem * DOF_PER_ELEMENT * DOF_PER_ELEMENT);
    let mut vals = Vec::with_capacity(nelem * DOF_PER_ELEMENT * DOF_PER_ELEMENT);

    for element_index in element_start..element_end {
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

/// Concatenate independently assembled triplet chunks without changing element order.
fn concat_stiffness_triplets<F: Real>(
    chunks: Vec<StiffnessTriplets<F>>,
    capacity: usize,
) -> StiffnessTriplets<F> {
    let mut rows = Vec::with_capacity(capacity);
    let mut cols = Vec::with_capacity(capacity);
    let mut vals = Vec::with_capacity(capacity);

    for chunk in chunks {
        rows.extend(chunk.rows);
        cols.extend(chunk.cols);
        vals.extend(chunk.vals);
    }

    StiffnessTriplets { rows, cols, vals }
}

#[cfg(test)]
mod tests {
    use super::{assemble_stiffness_for_family, assemble_stiffness_for_family_par};
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

    #[test]
    /// Check that threaded assembly matches serial assembly exactly for the same element stream.
    fn parallel_stiffness_matches_serial_stiffness() {
        let mesh = single_element_quad4_mesh();
        let material_ids = [0usize];
        let material_table = [isotropic_axisymmetric_material(200.0e9, 0.27)];
        let serial = assemble_stiffness_for_family::<f64, Quad4Family, 4, { dof_per_element(4) }>(
            mesh,
            &material_ids,
            &material_table,
            None,
            crate::physics::solenoid_stress::types::Structural2dFormulation::Axisymmetric,
            QuadratureRule::GaussLegendre3,
        )
        .expect("serial assembly should succeed");
        let parallel =
            assemble_stiffness_for_family_par::<f64, Quad4Family, 4, { dof_per_element(4) }>(
                mesh,
                &material_ids,
                &material_table,
                None,
                crate::physics::solenoid_stress::types::Structural2dFormulation::Axisymmetric,
                QuadratureRule::GaussLegendre3,
            )
            .expect("parallel assembly should succeed");

        assert_eq!(serial.rows, parallel.rows);
        assert_eq!(serial.cols, parallel.cols);
        assert_eq!(serial.vals, parallel.vals);
    }
}
