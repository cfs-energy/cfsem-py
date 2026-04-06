//! Axisymmetric element assembly for small-strain elasticity.
//!
//! The element system matrix is assembled in the standard Galerkin form
//! `K_e = integral(B^T D B 2*pi*r dA)` and the consistent load vectors are assembled from the
//! same weak form.  This follows the conventional displacement-based finite-element construction
//! described in Hughes (1987), Bathe (1996), and Reddy (2005).

use crate::physics::solenoid_stress::axisym::{accumulate_stiffness, build_b_matrix};
use crate::physics::solenoid_stress::geometry::volume_samples;
use crate::physics::solenoid_stress::loads::{accumulate_body_force, pressure_element_load};
use crate::physics::solenoid_stress::mesh::{AssemblyResult, MeshView, PressureLoad};
use crate::physics::solenoid_stress::quad4::DOF_PER_ELEMENT;
use crate::physics::solenoid_stress::quadrature::QuadratureRule;
use crate::physics::solenoid_stress::types::{Real, two_pi};

pub fn assemble_axisymmetric_quad4<F: Real>(
    mesh: MeshView<'_, F>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    body_force: &[[F; 2]],
    pressure_loads: &[PressureLoad<F>],
    quadrature: QuadratureRule,
) -> Result<AssemblyResult<F>, String> {
    mesh.validate_nodes()?;
    mesh.validate_connectivity()?;
    if material_ids.len() != mesh.num_elements() {
        return Err(format!(
            "material_ids has length {}, but mesh has {} elements",
            material_ids.len(),
            mesh.num_elements()
        ));
    }
    if body_force.len() != mesh.num_elements() {
        return Err(format!(
            "body_force has length {}, but mesh has {} elements",
            body_force.len(),
            mesh.num_elements()
        ));
    }
    let ndof = mesh.num_nodes() * 2;
    let mut rows = Vec::with_capacity(mesh.num_elements() * DOF_PER_ELEMENT * DOF_PER_ELEMENT);
    let mut cols = Vec::with_capacity(mesh.num_elements() * DOF_PER_ELEMENT * DOF_PER_ELEMENT);
    let mut vals = Vec::with_capacity(mesh.num_elements() * DOF_PER_ELEMENT * DOF_PER_ELEMENT);
    let mut rhs = vec![F::zero(); ndof];
    let two_pi = two_pi::<F>();

    for element_index in 0..mesh.num_elements() {
        let coords = mesh.element_coords(element_index)?;
        let nodes = mesh.element_nodes(element_index)?;
        let material_id = material_ids[element_index];
        let material = material_table.get(material_id).ok_or_else(|| {
            format!("material_id {material_id} on element {element_index} is out of range")
        })?;
        let element_body_force = body_force[element_index];
        let mut ke = [[F::zero(); DOF_PER_ELEMENT]; DOF_PER_ELEMENT];
        let mut fe = [F::zero(); DOF_PER_ELEMENT];

        for sample in volume_samples(&coords, quadrature)? {
            // Axisymmetric element contribution:
            // K_e += B^T D B (2*pi*r det(J) w), f_e += N^T b (2*pi*r det(J) w)
            let b = build_b_matrix(&sample.n, &sample.grad_phys, sample.point[0])?;
            let scale = two_pi * sample.point[0] * sample.det_j * sample.weight;
            accumulate_stiffness(&mut ke, material, &b, scale);
            accumulate_body_force(&mut fe, element_body_force, &sample.n, scale);
        }

        let mut local_dofs = [0usize; DOF_PER_ELEMENT];
        for (local_node, global_node) in nodes.into_iter().enumerate() {
            local_dofs[2 * local_node] = 2 * global_node;
            local_dofs[2 * local_node + 1] = 2 * global_node + 1;
        }

        for row in 0..DOF_PER_ELEMENT {
            rhs[local_dofs[row]] = rhs[local_dofs[row]] + fe[row];
            for col in 0..DOF_PER_ELEMENT {
                rows.push(local_dofs[row]);
                cols.push(local_dofs[col]);
                vals.push(ke[row][col]);
            }
        }
    }

    for load in pressure_loads {
        if load.element >= mesh.num_elements() {
            return Err(format!(
                "pressure load references element {}, but mesh has only {} elements",
                load.element,
                mesh.num_elements()
            ));
        }
        let coords = mesh.element_coords(load.element)?;
        let nodes = mesh.element_nodes(load.element)?;
        let fe = pressure_element_load(&coords, load.local_face, load.value, quadrature)?;
        for (local_node, global_node) in nodes.into_iter().enumerate() {
            rhs[2 * global_node] = rhs[2 * global_node] + fe[2 * local_node];
            rhs[2 * global_node + 1] = rhs[2 * global_node + 1] + fe[2 * local_node + 1];
        }
    }

    Ok(AssemblyResult {
        rows,
        cols,
        vals,
        rhs,
        ndof,
    })
}

#[cfg(test)]
mod tests {
    use super::assemble_axisymmetric_quad4;
    use crate::physics::solenoid_stress::mesh::MeshView;
    use crate::physics::solenoid_stress::quadrature::QuadratureRule;

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
    fn single_element_zero_load_has_zero_rhs_and_symmetric_stiffness() {
        let nodes = [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]];
        let elements = [[0usize, 1, 2, 3]];
        let mesh = MeshView {
            nodes_rz: &nodes,
            elements: &elements,
        };
        let material_ids = [0usize];
        let material_table = [isotropic_material(200.0e9, 0.27)];
        let body_force = [[0.0, 0.0]];
        let result = assemble_axisymmetric_quad4(
            mesh,
            &material_ids,
            &material_table,
            &body_force,
            &[],
            QuadratureRule::Gauss2x2,
        )
        .expect("assembly should succeed");

        assert!(result.rhs.iter().all(|v| v.abs() < 1.0e-12));

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
