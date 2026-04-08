//! Sparse strain/stress recovery operators at element quadrature points.

use crate::physics::solenoid_stress::axisym::{build_b_matrix, constitutive_times_b};
use crate::physics::solenoid_stress::geometry::{
    VolumeSample, volume_samples_quad4, volume_samples_quad9,
};
use crate::physics::solenoid_stress::mesh::MeshView;
use crate::physics::solenoid_stress::quadrature::QuadratureRule;
use crate::physics::solenoid_stress::types::Real;
use crate::physics::solenoid_stress::{quad4, quad9};

#[derive(Debug, Clone)]
pub struct QuadratureFieldOperators<F: Real> {
    /// Quadrature-point coordinates `(r, z)` in element-major order.
    pub points_rz: Vec<[F; 2]>,
    /// Sparse row indices for the strain operator triplets.
    pub strain_rows: Vec<usize>,
    /// Sparse column indices for the strain operator triplets.
    pub strain_cols: Vec<usize>,
    /// Sparse values for the strain operator triplets.
    pub strain_vals: Vec<F>,
    /// Sparse row indices for the stress operator triplets.
    pub stress_rows: Vec<usize>,
    /// Sparse column indices for the stress operator triplets.
    pub stress_cols: Vec<usize>,
    /// Sparse values for the stress operator triplets.
    pub stress_vals: Vec<F>,
    /// Number of quadrature points contributed by each element.
    pub nq_per_element: usize,
    /// Number of global displacement DOFs the operators act on.
    pub ndof: usize,
}

fn quadrature_field_operators_impl<
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
) -> Result<QuadratureFieldOperators<F>, String> {
    debug_assert_eq!(DOF_PER_ELEMENT, 2 * NODES_PER_ELEMENT);
    mesh.validate_nodes()?;
    mesh.validate_connectivity()?;
    if material_ids.len() != mesh.num_elements() {
        return Err(format!(
            "material_ids has length {}, but mesh has {} elements",
            material_ids.len(),
            mesh.num_elements()
        ));
    }

    let ndof = mesh.num_nodes() * 2;
    let nq_per_element = quadrature.points_per_element();
    let nsamples = mesh.num_elements() * nq_per_element;
    let mut points_rz = Vec::with_capacity(nsamples);
    let mut strain_rows = Vec::with_capacity(nsamples * (DOF_PER_ELEMENT + 4));
    let mut strain_cols = Vec::with_capacity(nsamples * (DOF_PER_ELEMENT + 4));
    let mut strain_vals = Vec::with_capacity(nsamples * (DOF_PER_ELEMENT + 4));
    let mut stress_rows = Vec::with_capacity(nsamples * (DOF_PER_ELEMENT + 4));
    let mut stress_cols = Vec::with_capacity(nsamples * (DOF_PER_ELEMENT + 4));
    let mut stress_vals = Vec::with_capacity(nsamples * (DOF_PER_ELEMENT + 4));

    for element_index in 0..mesh.num_elements() {
        let coords = mesh.element_coords(element_index)?;
        let nodes = mesh.element_nodes(element_index)?;
        let material_id = material_ids[element_index];
        let material = material_table.get(material_id).ok_or_else(|| {
            format!("material_id {material_id} on element {element_index} is out of range")
        })?;
        let mut local_dofs = [0usize; DOF_PER_ELEMENT];
        for (local_node, global_node) in nodes.into_iter().enumerate() {
            local_dofs[2 * local_node] = 2 * global_node;
            local_dofs[2 * local_node + 1] = 2 * global_node + 1;
        }

        for (q_local, sample) in volume_samples_fn(&coords, quadrature)?
            .into_iter()
            .enumerate()
        {
            let b = build_b_matrix::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                &sample.n,
                &sample.grad_phys,
                sample.point[0],
            )?;
            let db = constitutive_times_b(material, &b);
            let row_base = 4 * (element_index * nq_per_element + q_local);
            points_rz.push(sample.point);

            for component in 0..4 {
                let global_row = row_base + component;
                for local_dof in 0..DOF_PER_ELEMENT {
                    let global_col = local_dofs[local_dof];
                    let strain_value = b[component][local_dof];
                    if strain_value != F::zero() {
                        strain_rows.push(global_row);
                        strain_cols.push(global_col);
                        strain_vals.push(strain_value);
                    }
                    let stress_value = db[component][local_dof];
                    if stress_value != F::zero() {
                        stress_rows.push(global_row);
                        stress_cols.push(global_col);
                        stress_vals.push(stress_value);
                    }
                }
            }
        }
    }

    Ok(QuadratureFieldOperators {
        points_rz,
        strain_rows,
        strain_cols,
        strain_vals,
        stress_rows,
        stress_cols,
        stress_vals,
        nq_per_element,
        ndof,
    })
}

/// Assemble sparse quadrature-point strain and stress operators for the Quad4 mesh.
pub fn quadrature_field_operators_quad4<F: Real>(
    mesh: MeshView<'_, F, { quad4::NODES_PER_ELEMENT }>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    quadrature: QuadratureRule,
) -> Result<QuadratureFieldOperators<F>, String> {
    quadrature_field_operators_impl::<F, { quad4::NODES_PER_ELEMENT }, { quad4::DOF_PER_ELEMENT }>(
        mesh,
        material_ids,
        material_table,
        quadrature,
        volume_samples_quad4::<F>,
    )
}

/// Assemble sparse quadrature-point strain and stress operators for the Quad9 mesh.
pub fn quadrature_field_operators_quad9<F: Real>(
    mesh: MeshView<'_, F, { quad9::NODES_PER_ELEMENT }>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    quadrature: QuadratureRule,
) -> Result<QuadratureFieldOperators<F>, String> {
    quadrature_field_operators_impl::<F, { quad9::NODES_PER_ELEMENT }, { quad9::DOF_PER_ELEMENT }>(
        mesh,
        material_ids,
        material_table,
        quadrature,
        volume_samples_quad9::<F>,
    )
}

#[cfg(test)]
mod tests {
    use super::quadrature_field_operators_quad4;
    use crate::physics::solenoid_stress::axisym::{build_b_matrix, constitutive_times_b};
    use crate::physics::solenoid_stress::geometry::volume_samples_quad4;
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

    fn apply_triplets(
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        nrow: usize,
        x: &[f64],
    ) -> Vec<f64> {
        let mut out = vec![0.0; nrow];
        for ((row, col), val) in rows.iter().zip(cols).zip(vals) {
            out[*row] += *val * x[*col];
        }
        out
    }

    #[test]
    fn quadrature_field_operators_match_direct_b_and_db_application() {
        let nodes = [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]];
        let elements = [[0usize, 1, 2, 3]];
        let mesh = MeshView {
            nodes_rz: &nodes,
            elements: &elements,
        };
        let material_ids = [0usize];
        let material_table = [isotropic_material(200.0e9, 0.27)];
        let operators = quadrature_field_operators_quad4(
            mesh,
            &material_ids,
            &material_table,
            QuadratureRule::Gauss3x3,
        )
        .expect("operator assembly should succeed");

        let u = [0.01, -0.02, 0.03, 0.01, 0.02, -0.01, -0.04, 0.02];
        let strain = apply_triplets(
            &operators.strain_rows,
            &operators.strain_cols,
            &operators.strain_vals,
            operators.points_rz.len() * 4,
            &u,
        );
        let stress = apply_triplets(
            &operators.stress_rows,
            &operators.stress_cols,
            &operators.stress_vals,
            operators.points_rz.len() * 4,
            &u,
        );

        let coords = mesh.element_coords(0).expect("element coords");
        let samples = volume_samples_quad4(&coords, QuadratureRule::Gauss3x3).expect("samples");
        for (q_local, sample) in samples.into_iter().enumerate() {
            let b = build_b_matrix::<f64, 4, 8>(&sample.n, &sample.grad_phys, sample.point[0])
                .expect("B matrix");
            let db = constitutive_times_b(&material_table[0], &b);
            for component in 0..4 {
                let row = 4 * q_local + component;
                let mut eps = 0.0;
                let mut sig = 0.0;
                for dof in 0..8 {
                    eps += b[component][dof] * u[dof];
                    sig += db[component][dof] * u[dof];
                }
                assert!((strain[row] - eps).abs() < 1.0e-12);
                assert!((stress[row] - sig).abs() < 1.0e-3);
            }
        }
    }
}
