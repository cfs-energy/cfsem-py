//! Axisymmetric element assembly for small-strain elasticity.
//!
//! The element system matrix is assembled in Galerkin form
//! `K_e = integral(B^T D B 2*pi*r dA)`.

use crate::mesh::elements::quad2d::{quad4, quad9};
use crate::mesh::{MeshView, QuadratureRule};
use crate::physics::solenoid_stress::axisym::{accumulate_stiffness, build_b_matrix};
use crate::physics::solenoid_stress::geometry::{
    VolumeSample, validate_axisymmetric_nodes, volume_samples_quad4, volume_samples_quad9,
};
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, Real, StiffnessTriplets, dof_per_element, local_dofs, two_pi,
};

fn assemble_axisymmetric_impl<
    F: Real,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: MeshView<'_, F, NODES_PER_ELEMENT>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    quadrature: QuadratureRule,
    volume_samples_fn: fn(
        &[[F; 2]; NODES_PER_ELEMENT],
        QuadratureRule,
    ) -> Result<Vec<VolumeSample<F, NODES_PER_ELEMENT>>, String>,
) -> Result<StiffnessTriplets<F>, String> {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_axisymmetric_nodes(mesh)?;
    mesh.validate_connectivity()?;
    if material_ids.len() != mesh.num_elements() {
        return Err(format!(
            "material_ids has length {}, but mesh has {} elements",
            material_ids.len(),
            mesh.num_elements()
        ));
    }
    let ndof = mesh.num_nodes() * 2;
    let mut rows = Vec::with_capacity(mesh.num_elements() * DOF_PER_ELEMENT * DOF_PER_ELEMENT);
    let mut cols = Vec::with_capacity(mesh.num_elements() * DOF_PER_ELEMENT * DOF_PER_ELEMENT);
    let mut vals = Vec::with_capacity(mesh.num_elements() * DOF_PER_ELEMENT * DOF_PER_ELEMENT);
    let two_pi = two_pi::<F>();

    for element_index in 0..mesh.num_elements() {
        let coords = mesh.element_coords(element_index)?;
        let nodes = mesh.element_nodes(element_index)?;
        let material_id = material_ids[element_index];
        let material = material_table.get(material_id).ok_or_else(|| {
            format!("material_id {material_id} on element {element_index} is out of range")
        })?;
        let mut ke = [[F::zero(); DOF_PER_ELEMENT]; DOF_PER_ELEMENT];

        for sample in volume_samples_fn(&coords, quadrature)? {
            let b = build_b_matrix::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                &sample.n,
                &sample.grad_phys,
                sample.point[0],
            )?;
            let scale = two_pi * sample.point[0] * sample.det_j * sample.weight;
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

    Ok(StiffnessTriplets {
        rows,
        cols,
        vals,
        ndof,
    })
}

/// Assemble the global axisymmetric Quad4 stiffness operator in COO triplet form.
pub fn assemble_stiffness_quad4<F: Real>(
    mesh: MeshView<'_, F, { quad4::NODES_PER_ELEMENT }>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    quadrature: QuadratureRule,
) -> Result<StiffnessTriplets<F>, String> {
    assemble_axisymmetric_impl::<
        F,
        { quad4::NODES_PER_ELEMENT },
        { dof_per_element(quad4::NODES_PER_ELEMENT) },
    >(
        mesh,
        material_ids,
        material_table,
        quadrature,
        volume_samples_quad4::<F>,
    )
}

/// Assemble the global axisymmetric Quad9 stiffness operator in COO triplet form.
pub fn assemble_stiffness_quad9<F: Real>(
    mesh: MeshView<'_, F, { quad9::NODES_PER_ELEMENT }>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    quadrature: QuadratureRule,
) -> Result<StiffnessTriplets<F>, String> {
    assemble_axisymmetric_impl::<
        F,
        { quad9::NODES_PER_ELEMENT },
        { dof_per_element(quad9::NODES_PER_ELEMENT) },
    >(
        mesh,
        material_ids,
        material_table,
        quadrature,
        volume_samples_quad9::<F>,
    )
}

#[cfg(test)]
mod tests {
    use super::assemble_stiffness_quad4;
    use crate::mesh::{MeshView, QuadratureRule};

    fn isotropic_material(e: f64, nu: f64) -> [[f64; 4]; 4] {
        let lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = e / (2.0 * (1.0 + nu));
        [
            [lam + 2.0 * mu, lam, lam, 0.0],
            [lam, lam + 2.0 * mu, lam, 0.0],
            [lam, lam, lam + 2.0 * mu, 0.0],
            [0.0, 0.0, 0.0, mu],
        ]
    }

    #[test]
    fn single_element_has_symmetric_stiffness() {
        let nodes = [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]];
        let elements = [[0usize, 1, 2, 3]];
        let mesh = MeshView {
            nodes_rz: &nodes,
            elements: &elements,
        };
        let material_ids = [0usize];
        let material_table = [isotropic_material(200.0e9, 0.27)];
        let result = assemble_stiffness_quad4(
            mesh,
            &material_ids,
            &material_table,
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
