use crate::mesh::elements::quad2d::{quad4, quad9};
use crate::mesh::{MeshView, QuadratureRule};
use crate::physics::solenoid_stress::axisym::{
    accumulate_b_transpose_vector, build_b_matrix, constitutive_times_strain,
};
use crate::physics::solenoid_stress::geometry::{
    VolumeSample, validate_axisymmetric_nodes, volume_samples_quad4, volume_samples_quad9,
};
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, Real, ThermalMaterial, dof_per_element, local_dofs, two_pi,
};

use super::{SparseOperator, ThermalLoadOperator, scatter_local_matrix};

struct LocalThermalKernel<F: Real, const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize> {
    temperature_to_rhs: [[F; NODES_PER_ELEMENT]; DOF_PER_ELEMENT],
    reference_rhs: [F; DOF_PER_ELEMENT],
}

fn thermal_element_kernel<F: Real, const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    samples: &[VolumeSample<F, NODES_PER_ELEMENT>],
    material: &[[F; 4]; 4],
    thermal: &ThermalMaterial<F>,
) -> Result<LocalThermalKernel<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>, String> {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    let mut local = LocalThermalKernel {
        temperature_to_rhs: [[F::zero(); NODES_PER_ELEMENT]; DOF_PER_ELEMENT],
        reference_rhs: [F::zero(); DOF_PER_ELEMENT],
    };
    let two_pi = two_pi::<F>();
    let thermal_stress_unit = constitutive_times_strain(material, &thermal.alpha);

    for sample in samples {
        let b = build_b_matrix::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            &sample.n,
            &sample.grad_phys,
            sample.point[0],
        )?;
        let scale = two_pi * sample.point[0] * sample.det_j * sample.weight;
        let mut local_unit_rhs = [F::zero(); DOF_PER_ELEMENT];
        accumulate_b_transpose_vector(&mut local_unit_rhs, &b, &thermal_stress_unit, scale);

        for local_temp_node in 0..NODES_PER_ELEMENT {
            let scale_node = sample.n[local_temp_node];
            for dof in 0..DOF_PER_ELEMENT {
                local.temperature_to_rhs[dof][local_temp_node] = local.temperature_to_rhs[dof]
                    [local_temp_node]
                    + local_unit_rhs[dof] * scale_node;
            }
        }
        for dof in 0..DOF_PER_ELEMENT {
            local.reference_rhs[dof] =
                local.reference_rhs[dof] - local_unit_rhs[dof] * thermal.reference_temperature;
        }
    }

    Ok(local)
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
        let samples = volume_samples_fn(&coords, quadrature)?;
        let local = thermal_element_kernel::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            &samples, material, thermal,
        )?;
        let global_rows = local_dofs::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(&nodes);
        scatter_local_matrix(
            &mut rows,
            &mut cols,
            &mut vals,
            &global_rows,
            &nodes,
            &local.temperature_to_rhs,
        );
        for dof in 0..DOF_PER_ELEMENT {
            reference_rhs[global_rows[dof]] =
                reference_rhs[global_rows[dof]] + local.reference_rhs[dof];
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
