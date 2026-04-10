use crate::mesh::elements::quad2d::{quad4, quad9};
use crate::mesh::{MeshView, QuadratureRule};
use crate::physics::solenoid_stress::axisym::{
    accumulate_b_transpose_vector, build_b_matrix, constitutive_times_strain,
};
use crate::physics::solenoid_stress::geometry::{
    VolumeSample, validate_axisymmetric_nodes, volume_samples_quad4, volume_samples_quad9,
};
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, Real, ThermalMaterial, dof_per_element, two_pi,
};

use super::{SparseOperator, ThermalLoadOperator};

pub(crate) fn accumulate_thermal_load<
    F: Real,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    fe: &mut [F; DOF_PER_ELEMENT],
    material: &[[F; 4]; 4],
    thermal: &ThermalMaterial<F>,
    element_temperature: &[F; NODES_PER_ELEMENT],
    shape: &[F; NODES_PER_ELEMENT],
    b: &[[F; DOF_PER_ELEMENT]; 4],
    scale: F,
) {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    let mut temperature = F::zero();
    for local_node in 0..NODES_PER_ELEMENT {
        temperature = temperature + shape[local_node] * element_temperature[local_node];
    }
    let delta_temperature = temperature - thermal.reference_temperature;
    let thermal_strain = [
        thermal.alpha[0] * delta_temperature,
        thermal.alpha[1] * delta_temperature,
        thermal.alpha[2] * delta_temperature,
        thermal.alpha[3] * delta_temperature,
    ];
    let thermal_stress = constitutive_times_strain(material, &thermal_strain);
    accumulate_b_transpose_vector(fe, b, &thermal_stress, scale);
}

fn temperature_operator_impl<
    F: Real,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: MeshView<'_, F, NODES_PER_ELEMENT>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    thermal_material_table: &[ThermalMaterial<F>],
    quadrature: QuadratureRule,
    volume_samples_fn: fn(
        &[[F; 2]; NODES_PER_ELEMENT],
        QuadratureRule,
    ) -> Result<Vec<VolumeSample<F, NODES_PER_ELEMENT>>, String>,
) -> Result<ThermalLoadOperator<F>, String> {
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
    let ncol = mesh.num_nodes();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut reference_rhs = vec![F::zero(); ndof];
    let two_pi = two_pi::<F>();

    for element_index in 0..mesh.num_elements() {
        let coords = mesh.element_coords(element_index)?;
        let nodes = mesh.element_nodes(element_index)?;
        let material_id = material_ids[element_index];
        let material = material_table.get(material_id).ok_or_else(|| {
            format!("material_id {material_id} on element {element_index} is out of range")
        })?;
        let thermal = thermal_material_table.get(material_id).ok_or_else(|| {
            format!("thermal material_id {material_id} on element {element_index} is out of range")
        })?;
        let mut local_operator = [[F::zero(); NODES_PER_ELEMENT]; DOF_PER_ELEMENT];
        let mut local_reference_rhs = [F::zero(); DOF_PER_ELEMENT];

        for sample in volume_samples_fn(&coords, quadrature)? {
            let b = build_b_matrix::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                &sample.n,
                &sample.grad_phys,
                sample.point[0],
            )?;
            let scale = two_pi * sample.point[0] * sample.det_j * sample.weight;
            let thermal_strain_unit = thermal.alpha;
            let thermal_stress_unit = constitutive_times_strain(material, &thermal_strain_unit);
            let mut local_unit_rhs = [F::zero(); DOF_PER_ELEMENT];
            accumulate_b_transpose_vector(&mut local_unit_rhs, &b, &thermal_stress_unit, scale);

            for local_temp_node in 0..NODES_PER_ELEMENT {
                let scale_node = sample.n[local_temp_node];
                for dof in 0..DOF_PER_ELEMENT {
                    local_operator[dof][local_temp_node] =
                        local_operator[dof][local_temp_node] + local_unit_rhs[dof] * scale_node;
                }
            }
            for dof in 0..DOF_PER_ELEMENT {
                local_reference_rhs[dof] =
                    local_reference_rhs[dof] - local_unit_rhs[dof] * thermal.reference_temperature;
            }
        }

        for (local_node, global_node) in nodes.iter().copied().enumerate() {
            let dof_r = 2 * global_node;
            let dof_z = dof_r + 1;
            reference_rhs[dof_r] = reference_rhs[dof_r] + local_reference_rhs[2 * local_node];
            reference_rhs[dof_z] = reference_rhs[dof_z] + local_reference_rhs[2 * local_node + 1];
            for local_temp_node in 0..NODES_PER_ELEMENT {
                let global_temp_node = nodes[local_temp_node];
                let radial_value = local_operator[2 * local_node][local_temp_node];
                let axial_value = local_operator[2 * local_node + 1][local_temp_node];
                if radial_value != F::zero() {
                    rows.push(dof_r);
                    cols.push(global_temp_node);
                    vals.push(radial_value);
                }
                if axial_value != F::zero() {
                    rows.push(dof_z);
                    cols.push(global_temp_node);
                    vals.push(axial_value);
                }
            }
        }
    }

    Ok(ThermalLoadOperator {
        temperature_to_rhs: SparseOperator {
            rows,
            cols,
            vals,
            nrow: ndof,
            ncol,
        },
        reference_rhs,
    })
}

pub fn temperature_operator_quad4<F: Real>(
    mesh: MeshView<'_, F, { quad4::NODES_PER_ELEMENT }>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    thermal_material_table: &[ThermalMaterial<F>],
    quadrature: QuadratureRule,
) -> Result<ThermalLoadOperator<F>, String> {
    temperature_operator_impl::<
        F,
        { quad4::NODES_PER_ELEMENT },
        { dof_per_element(quad4::NODES_PER_ELEMENT) },
    >(
        mesh,
        material_ids,
        material_table,
        thermal_material_table,
        quadrature,
        volume_samples_quad4::<F>,
    )
}

pub fn temperature_operator_quad9<F: Real>(
    mesh: MeshView<'_, F, { quad9::NODES_PER_ELEMENT }>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    thermal_material_table: &[ThermalMaterial<F>],
    quadrature: QuadratureRule,
) -> Result<ThermalLoadOperator<F>, String> {
    temperature_operator_impl::<
        F,
        { quad9::NODES_PER_ELEMENT },
        { dof_per_element(quad9::NODES_PER_ELEMENT) },
    >(
        mesh,
        material_ids,
        material_table,
        thermal_material_table,
        quadrature,
        volume_samples_quad9::<F>,
    )
}
