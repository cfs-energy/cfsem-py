//! Sparse strain/stress recovery operators at element quadrature points.
//!
//! The displacement derivatives used in strain and stress recovery are implemented analytically
//! from the element shape functions, then mapped from reference-element coordinates into physical
//! meridian coordinates `(r, z)` with the element Jacobian. There is no finite-difference
//! approximation of the displacement field.
//!
//! Mechanical strain/stress recovery uses the same query-coordinate sparse operators that serve
//! arbitrary point probes. Quadrature recovery builds the known element-major
//! `(element_index, reference_point)` arrays directly, so it reuses that math without doing a
//! nearest-element search.
//!
//! The underlying chain is:
//! 1. Each element family defines closed-form shape functions `N_i(\xi,\eta)` and closed-form
//!    reference gradients.
//! 2. At each quadrature point, those reference gradients are mapped into physical-space
//!    gradients with the inverse Jacobian according to
//!    `partial N / partial (r, z) = J^{-T} partial N / partial (\xi, \eta)`.
//! 3. The strain-displacement matrix `B` is then built from those physical derivatives.
//!
//! In that `B` matrix:
//! - `e_rr = partial u_r / partial r`,
//! - `e_zz = partial u_z / partial z`,
//! - `g_rz = partial u_r / partial z + partial u_z / partial r`,
//! so those rows use the entries of `grad_phys` directly.
//!
//! The hoop term is slightly different:
//! - `e_tt = u_r / r`,
//! so that row uses `N_i / r`, not a spatial derivative.
//!
//! 4. Recovery then uses that same `B`:
//!    - strain recovery uses `B`,
//!    - stress recovery uses `D B`,
//!    where `D` is the local per-material `4 x 4` matrix for the elastic stress-strain law.
//!
//! Thermal recovery remains quadrature-specific because it maps nodal temperatures and material
//! reference temperatures to thermal strain/stress offsets.

use crate::mesh::elements::quad2d::quadrature::gauss_volume;
use crate::mesh::quad2d::{quad_mesh_strain_operator, quad_mesh_stress_operator};
use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::axisym::constitutive_times_strain;
use crate::physics::solenoid_stress::convenience::{
    rotate_material_in_plane, rotate_thermal_material_in_plane,
};
use crate::physics::solenoid_stress::family::QuadElementFamily;
use crate::physics::solenoid_stress::geometry::{VolumeSample, validate_structural_2d_mesh};
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, Real, Structural2dFormulation, ThermalMaterial, scatter_local_matrix,
    validate_element_material_inputs,
};

/// Sparse quadrature-point recovery operators before reduction into the model-owned CSR form.
///
/// Rows are stored in quadrature-point-major order with axisymmetric component ordering
/// `[rr, zz, tt, rz]`, so rows `4*q .. 4*q + 3` correspond to one quadrature point.
#[derive(Debug, Clone)]
pub struct QuadratureFieldOperators<F: Real> {
    /// Quadrature-point coordinates `(r, z)` in element-major order.
    ///
    /// Units: `[length]`.
    pub points: Vec<[F; 2]>,
    /// Sparse row indices for the strain operator triplets.
    pub strain_rows: Vec<usize>,
    /// Sparse column indices for the strain operator triplets.
    ///
    /// Columns index full displacement DOFs `[u_r1, u_z1, ...]`.
    pub strain_cols: Vec<usize>,
    /// Sparse values for the strain operator triplets.
    ///
    /// Units: `[strain / displacement] = [1 / length]`.
    pub strain_vals: Vec<F>,
    /// Sparse row indices for the stress operator triplets.
    pub stress_rows: Vec<usize>,
    /// Sparse column indices for the stress operator triplets.
    ///
    /// Columns index full displacement DOFs `[u_r1, u_z1, ...]`.
    pub stress_cols: Vec<usize>,
    /// Sparse values for the stress operator triplets.
    ///
    /// Units: `[stress / displacement] = [pressure / length]`.
    pub stress_vals: Vec<F>,
    /// Sparse row indices for the thermal-strain operator triplets.
    pub thermal_strain_rows: Vec<usize>,
    /// Sparse column indices for the thermal-strain operator triplets.
    ///
    /// Columns index nodal temperatures `[temperature]`.
    pub thermal_strain_cols: Vec<usize>,
    /// Sparse values for the thermal-strain operator triplets.
    ///
    /// Units: `[strain / temperature]`.
    pub thermal_strain_vals: Vec<F>,
    /// Sparse row indices for the thermal-stress operator triplets.
    pub thermal_stress_rows: Vec<usize>,
    /// Sparse column indices for the thermal-stress operator triplets.
    ///
    /// Columns index nodal temperatures `[temperature]`.
    pub thermal_stress_cols: Vec<usize>,
    /// Sparse values for the thermal-stress operator triplets.
    ///
    /// Units: `[stress / temperature]`.
    pub thermal_stress_vals: Vec<F>,
    /// Constant quadrature-point thermal strain contribution from per-material reference temperature.
    ///
    /// Units: `[strain]`.
    pub thermal_strain_constant: Vec<F>,
    /// Constant quadrature-point thermal stress contribution from per-material reference temperature.
    ///
    /// Units: `[stress]`.
    pub thermal_stress_constant: Vec<F>,
    /// Number of quadrature points contributed by each element.
    pub nq_per_element: usize,
    /// Number of nodal temperatures the thermal operators act on.
    pub ntemp: usize,
}

/// Dense thermal recovery operators for one quadrature point.
///
/// Units:
/// - `thermal_strain`: `[strain / temperature]`
/// - `thermal_stress`: `[stress / temperature]`
/// - `thermal_*_constant`: `strain` and `stress`, respectively
struct LocalThermalSampleKernel<F: Real, const NODES_PER_ELEMENT: usize> {
    thermal_strain: [[F; NODES_PER_ELEMENT]; 4],
    thermal_stress: [[F; NODES_PER_ELEMENT]; 4],
    thermal_strain_constant: [F; 4],
    thermal_stress_constant: [F; 4],
}

/// Build the dense thermal recovery blocks for one quadrature point.
///
/// This helper builds the local thermal operators and constant offsets associated with the
/// material reference temperature. Mechanical strain/stress recovery is assembled through the
/// shared query-coordinate mesh operators.
fn thermal_sample_kernel<F: Real, const NODES_PER_ELEMENT: usize>(
    sample: &VolumeSample<F, NODES_PER_ELEMENT>,
    thermal: &ThermalMaterial<F>,
    thermal_stress_unit: &[F; 4],
) -> LocalThermalSampleKernel<F, NODES_PER_ELEMENT> {
    let mut local = LocalThermalSampleKernel {
        thermal_strain: [[F::zero(); NODES_PER_ELEMENT]; 4],
        thermal_stress: [[F::zero(); NODES_PER_ELEMENT]; 4],
        thermal_strain_constant: [F::zero(); 4],
        thermal_stress_constant: [F::zero(); 4],
    };

    for component in 0..4 {
        for local_temp_node in 0..NODES_PER_ELEMENT {
            // These blocks map the nodal temperature field directly to thermal strain/stress
            // at this quadrature point.
            local.thermal_strain[component][local_temp_node] =
                thermal.alpha[component] * sample.n[local_temp_node];
            local.thermal_stress[component][local_temp_node] =
                thermal_stress_unit[component] * sample.n[local_temp_node];
        }
        local.thermal_strain_constant[component] =
            -thermal.alpha[component] * thermal.reference_temperature;
        local.thermal_stress_constant[component] =
            -thermal_stress_unit[component] * thermal.reference_temperature;
    }

    local
}

/// Return element-major quadrature-point references without doing a geometric point query.
///
/// Quadrature recovery already knows which element owns each point. These arrays have the same
/// shape expected by the query-coordinate strain/stress operators, but avoid the `O(nquery *
/// nelem)` nearest-element search that `query_quad_mesh` performs for arbitrary physical points.
fn element_major_reference_points<F: Real>(
    nelem: usize,
    quadrature: QuadratureRule,
) -> (Vec<usize>, Vec<[F; 2]>) {
    let references = gauss_volume::<F>(quadrature);
    let mut element_indices = Vec::with_capacity(nelem * references.len());
    let mut reference_points = Vec::with_capacity(nelem * references.len());
    for element_index in 0..nelem {
        for &(reference, _) in &references {
            element_indices.push(element_index);
            reference_points.push(reference);
        }
    }
    (element_indices, reference_points)
}

/// Assemble quadrature-point strain/stress recovery operators for one quadrilateral family.
///
/// Output shapes:
/// - `strain_*` and `stress_*`: `(4 * nq_per_element * mesh.num_elements(), 2 * mesh.num_nodes())`
/// - `thermal_*`: `(4 * nq_per_element * mesh.num_elements(), mesh.num_nodes())`
/// - `*_constant`: `(4 * nq_per_element * mesh.num_elements(),)`
///
/// Row meaning:
/// - rows `4*q .. 4*q + 3` correspond to quadrature point `q` in element-major order,
/// - within each quadrature point the row components are ordered `[rr, zz, tt, rz]`.
pub(crate) fn quadrature_field_operators_for_family<
    F: Real,
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    thermal_material_table: Option<&[ThermalMaterial<F>]>,
    material_orientation_angles: Option<&[F]>,
    formulation: Structural2dFormulation<F>,
    quadrature: QuadratureRule,
) -> Result<QuadratureFieldOperators<F>, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    validate_element_material_inputs(
        mesh.num_elements(),
        material_ids,
        material_orientation_angles,
    )?;

    let nq_per_element = quadrature.points_per_element();
    let nsamples = mesh.num_elements() * nq_per_element;
    let (element_indices, reference_points) =
        element_major_reference_points::<F>(mesh.num_elements(), quadrature);
    let strain_operator = quad_mesh_strain_operator::<
        Family::ReferenceElement,
        F,
        NODES_PER_ELEMENT,
        DOF_PER_ELEMENT,
    >(mesh, &element_indices, &reference_points, formulation)?;
    let stress_operator = quad_mesh_stress_operator::<
        Family::ReferenceElement,
        F,
        NODES_PER_ELEMENT,
        DOF_PER_ELEMENT,
    >(
        mesh,
        &element_indices,
        &reference_points,
        material_ids,
        material_table,
        material_orientation_angles,
        formulation,
    )?;

    let mut points = Vec::with_capacity(nsamples);
    let mut thermal_strain_rows = Vec::new();
    let mut thermal_strain_cols = Vec::new();
    let mut thermal_strain_vals = Vec::new();
    let mut thermal_stress_rows = Vec::new();
    let mut thermal_stress_cols = Vec::new();
    let mut thermal_stress_vals = Vec::new();
    let mut thermal_strain_constant = vec![F::zero(); nsamples * 4];
    let mut thermal_stress_constant = vec![F::zero(); nsamples * 4];

    for element_index in 0..mesh.num_elements() {
        let coords = mesh.element_coords(element_index)?;
        let nodes = mesh.element_nodes(element_index)?;
        let material_id = material_ids[element_index];
        let material = material_table.get(material_id).ok_or_else(|| {
            format!("material_id {material_id} on element {element_index} is out of range")
        })?;
        let thermal_material_base = thermal_material_table
            .map(|table| {
                table.get(material_id).ok_or_else(|| {
                    format!(
                        "thermal material_id {material_id} on element {element_index} is out of range"
                    )
                })
            })
            .transpose()?;
        let material_storage;
        let thermal_storage;
        let (material, thermal_material) = if let Some(angles) = material_orientation_angles {
            material_storage = rotate_material_in_plane(material, angles[element_index]);
            thermal_storage = thermal_material_base
                .map(|thermal| rotate_thermal_material_in_plane(thermal, angles[element_index]));
            (&material_storage, thermal_storage.as_ref())
        } else {
            (material, thermal_material_base)
        };
        let thermal_stress_unit =
            thermal_material.map(|thermal| constitutive_times_strain(material, &thermal.alpha));

        for (q_local, sample) in Family::volume_samples::<F>(&coords, quadrature)?
            .into_iter()
            .enumerate()
        {
            let row_base = 4 * (element_index * nq_per_element + q_local);
            let global_rows = [row_base, row_base + 1, row_base + 2, row_base + 3];
            points.push(sample.point);
            if let (Some(thermal_material), Some(thermal_stress_unit)) =
                (thermal_material, thermal_stress_unit.as_ref())
            {
                let local = thermal_sample_kernel(&sample, thermal_material, thermal_stress_unit);
                scatter_local_matrix(
                    &mut thermal_strain_rows,
                    &mut thermal_strain_cols,
                    &mut thermal_strain_vals,
                    &global_rows,
                    &nodes,
                    &local.thermal_strain,
                );
                scatter_local_matrix(
                    &mut thermal_stress_rows,
                    &mut thermal_stress_cols,
                    &mut thermal_stress_vals,
                    &global_rows,
                    &nodes,
                    &local.thermal_stress,
                );
                for component in 0..4 {
                    thermal_strain_constant[global_rows[component]] =
                        local.thermal_strain_constant[component];
                    thermal_stress_constant[global_rows[component]] =
                        local.thermal_stress_constant[component];
                }
            }
        }
    }

    Ok(QuadratureFieldOperators {
        points,
        strain_rows: strain_operator.rows,
        strain_cols: strain_operator.cols,
        strain_vals: strain_operator.vals,
        stress_rows: stress_operator.rows,
        stress_cols: stress_operator.cols,
        stress_vals: stress_operator.vals,
        thermal_strain_rows,
        thermal_strain_cols,
        thermal_strain_vals,
        thermal_stress_rows,
        thermal_stress_cols,
        thermal_stress_vals,
        thermal_strain_constant,
        thermal_stress_constant,
        nq_per_element,
        ntemp: thermal_material_table.map_or(0, |_| mesh.num_nodes()),
    })
}

#[cfg(test)]
mod tests {
    use super::quadrature_field_operators_for_family;
    use crate::mesh::{QuadratureRule, sampling};
    use crate::physics::solenoid_stress::axisym::{build_b_matrix, constitutive_times_b};
    use crate::physics::solenoid_stress::convenience::isotropic_axisymmetric_material;
    use crate::physics::solenoid_stress::family::Quad4Family;
    use crate::physics::solenoid_stress::test_utils::single_element_quad4_mesh;
    use crate::physics::solenoid_stress::types::{Structural2dFormulation, dof_per_element};

    /// Apply one triplet operator to a dense vector for direct-reference comparison in tests.
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
    /// Check that the sparse recovery operators reproduce direct `B` and `D B` evaluation.
    fn quadrature_field_operators_match_direct_b_and_db_application() {
        let mesh = single_element_quad4_mesh();
        let material_ids = [0usize];
        let material_table = [isotropic_axisymmetric_material(200.0e9, 0.27)];
        let operators =
            quadrature_field_operators_for_family::<f64, Quad4Family, 4, { dof_per_element(4) }>(
                mesh,
                &material_ids,
                &material_table,
                None,
                None,
                Structural2dFormulation::Axisymmetric,
                QuadratureRule::GaussLegendre3,
            )
            .expect("operator assembly should succeed");

        let u = [0.01, -0.02, 0.03, 0.01, 0.02, -0.01, -0.04, 0.02];
        let strain = apply_triplets(
            &operators.strain_rows,
            &operators.strain_cols,
            &operators.strain_vals,
            operators.points.len() * 4,
            &u,
        );
        let stress = apply_triplets(
            &operators.stress_rows,
            &operators.stress_cols,
            &operators.stress_vals,
            operators.points.len() * 4,
            &u,
        );

        let coords = mesh.element_coords(0).expect("element coords");
        let samples = sampling::volume_samples_quad4(&coords, QuadratureRule::GaussLegendre3)
            .expect("samples");
        for (q_local, sample) in samples.into_iter().enumerate() {
            let b = build_b_matrix::<f64, 4, 8>(
                Structural2dFormulation::Axisymmetric,
                &sample.n,
                &sample.grad_phys,
                sample.point,
            )
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
                // Stress is recovered from the displacement-gradient field and
                // picks up additional roundoff through the elastic stress-strain
                // law, so it is less sharp than the direct strain check.
                assert!((stress[row] - sig).abs() < 1.0e-3);
            }
        }
    }
}
