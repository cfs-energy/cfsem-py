//! Sparse load operators for repeated-load solves.
//!
//! These operators map load amplitudes to the global right-hand side:
//! - body-force operator: `[f_r(0), f_z(0), ...] -> rhs`
//! - pressure operator: `[p_0, p_1, ...] -> rhs`
//! - traction operator: `[t_r(0), t_z(0), ...] -> rhs`
//! - temperature operator: `[T(node_0), T(node_1), ...] -> rhs`

use crate::physics::solenoid_stress::axisym::{
    accumulate_b_transpose_vector, build_b_matrix, constitutive_times_strain,
};
use crate::physics::solenoid_stress::geometry::{
    FaceSample, VolumeSample, face_samples_quad4, face_samples_quad9, volume_samples_quad4,
    volume_samples_quad9,
};
use crate::physics::solenoid_stress::mesh::{
    MeshView, PressureLoad, ThermalMaterial, TractionLoad,
};
use crate::physics::solenoid_stress::quad4;
use crate::physics::solenoid_stress::quad9;
use crate::physics::solenoid_stress::quadrature::QuadratureRule;
use crate::physics::solenoid_stress::types::{Real, two_pi};

#[derive(Debug, Clone)]
pub struct SparseOperator<F: Real> {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<F>,
    pub nrow: usize,
    pub ncol: usize,
}

#[derive(Debug, Clone)]
pub struct ThermalLoadOperator<F: Real> {
    pub temperature_to_rhs: SparseOperator<F>,
    pub reference_rhs: Vec<F>,
}

fn body_force_operator_impl<
    F: Real,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: MeshView<'_, F, NODES_PER_ELEMENT>,
    quadrature: QuadratureRule,
    volume_samples_fn: fn(
        &[[F; 2]; NODES_PER_ELEMENT],
        QuadratureRule,
    ) -> Result<Vec<VolumeSample<F, NODES_PER_ELEMENT>>, String>,
) -> Result<SparseOperator<F>, String> {
    debug_assert_eq!(DOF_PER_ELEMENT, 2 * NODES_PER_ELEMENT);
    mesh.validate_nodes()?;
    mesh.validate_connectivity()?;
    let ndof = mesh.num_nodes() * 2;
    let ncol = 2 * mesh.num_elements();
    let mut rows = Vec::with_capacity(mesh.num_elements() * DOF_PER_ELEMENT * 2);
    let mut cols = Vec::with_capacity(mesh.num_elements() * DOF_PER_ELEMENT * 2);
    let mut vals = Vec::with_capacity(mesh.num_elements() * DOF_PER_ELEMENT * 2);
    let two_pi = two_pi::<F>();

    for element_index in 0..mesh.num_elements() {
        let coords = mesh.element_coords(element_index)?;
        let nodes = mesh.element_nodes(element_index)?;
        let mut local_r = [F::zero(); DOF_PER_ELEMENT];
        let mut local_z = [F::zero(); DOF_PER_ELEMENT];

        for sample in volume_samples_fn(&coords, quadrature)? {
            let scale = two_pi * sample.point[0] * sample.det_j * sample.weight;
            for local_node in 0..NODES_PER_ELEMENT {
                local_r[2 * local_node] = local_r[2 * local_node] + scale * sample.n[local_node];
                local_z[2 * local_node + 1] =
                    local_z[2 * local_node + 1] + scale * sample.n[local_node];
            }
        }

        for (local_node, global_node) in nodes.iter().copied().enumerate() {
            let dof_r = 2 * global_node;
            let dof_z = dof_r + 1;
            let local_r_index = 2 * local_node;
            let local_z_index = local_r_index + 1;
            if local_r[local_r_index] != F::zero() {
                rows.push(dof_r);
                cols.push(2 * element_index);
                vals.push(local_r[local_r_index]);
            }
            if local_z[local_z_index] != F::zero() {
                rows.push(dof_z);
                cols.push(2 * element_index + 1);
                vals.push(local_z[local_z_index]);
            }
        }
    }

    Ok(SparseOperator {
        rows,
        cols,
        vals,
        nrow: ndof,
        ncol,
    })
}

fn pressure_operator_impl<F: Real, const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    mesh: MeshView<'_, F, NODES_PER_ELEMENT>,
    pressure_faces: &[PressureLoad<F>],
    quadrature: QuadratureRule,
    face_samples_fn: fn(
        &[[F; 2]; NODES_PER_ELEMENT],
        u8,
        QuadratureRule,
    ) -> Result<Vec<FaceSample<F, NODES_PER_ELEMENT>>, String>,
) -> Result<SparseOperator<F>, String> {
    debug_assert_eq!(DOF_PER_ELEMENT, 2 * NODES_PER_ELEMENT);
    mesh.validate_nodes()?;
    mesh.validate_connectivity()?;
    let ndof = mesh.num_nodes() * 2;
    let ncol = pressure_faces.len();
    let mut rows = Vec::with_capacity(pressure_faces.len() * DOF_PER_ELEMENT);
    let mut cols = Vec::with_capacity(pressure_faces.len() * DOF_PER_ELEMENT);
    let mut vals = Vec::with_capacity(pressure_faces.len() * DOF_PER_ELEMENT);
    let two_pi = two_pi::<F>();

    for (load_index, load) in pressure_faces.iter().enumerate() {
        if load.element >= mesh.num_elements() {
            return Err(format!(
                "pressure load references element {}, but mesh has only {} elements",
                load.element,
                mesh.num_elements()
            ));
        }
        let coords = mesh.element_coords(load.element)?;
        let nodes = mesh.element_nodes(load.element)?;
        let mut local = [F::zero(); DOF_PER_ELEMENT];

        for sample in face_samples_fn(&coords, load.local_face, quadrature)? {
            let normal_area = [sample.tangent[1], -sample.tangent[0]];
            let scale = -two_pi * sample.point[0] * sample.weight;
            for local_node in 0..NODES_PER_ELEMENT {
                local[2 * local_node] =
                    local[2 * local_node] + scale * sample.n[local_node] * normal_area[0];
                local[2 * local_node + 1] =
                    local[2 * local_node + 1] + scale * sample.n[local_node] * normal_area[1];
            }
        }

        for (local_node, global_node) in nodes.iter().copied().enumerate() {
            let dof_r = 2 * global_node;
            let dof_z = dof_r + 1;
            let local_r_index = 2 * local_node;
            let local_z_index = local_r_index + 1;
            if local[local_r_index] != F::zero() {
                rows.push(dof_r);
                cols.push(load_index);
                vals.push(local[local_r_index]);
            }
            if local[local_z_index] != F::zero() {
                rows.push(dof_z);
                cols.push(load_index);
                vals.push(local[local_z_index]);
            }
        }
    }

    Ok(SparseOperator {
        rows,
        cols,
        vals,
        nrow: ndof,
        ncol,
    })
}

fn traction_operator_impl<F: Real, const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    mesh: MeshView<'_, F, NODES_PER_ELEMENT>,
    traction_faces: &[TractionLoad<F>],
    quadrature: QuadratureRule,
    face_samples_fn: fn(
        &[[F; 2]; NODES_PER_ELEMENT],
        u8,
        QuadratureRule,
    ) -> Result<Vec<FaceSample<F, NODES_PER_ELEMENT>>, String>,
) -> Result<SparseOperator<F>, String> {
    debug_assert_eq!(DOF_PER_ELEMENT, 2 * NODES_PER_ELEMENT);
    mesh.validate_nodes()?;
    mesh.validate_connectivity()?;
    let ndof = mesh.num_nodes() * 2;
    let ncol = 2 * traction_faces.len();
    let mut rows = Vec::with_capacity(traction_faces.len() * DOF_PER_ELEMENT * 2);
    let mut cols = Vec::with_capacity(traction_faces.len() * DOF_PER_ELEMENT * 2);
    let mut vals = Vec::with_capacity(traction_faces.len() * DOF_PER_ELEMENT * 2);
    let two_pi = two_pi::<F>();

    for (load_index, load) in traction_faces.iter().enumerate() {
        if load.element >= mesh.num_elements() {
            return Err(format!(
                "traction load references element {}, but mesh has only {} elements",
                load.element,
                mesh.num_elements()
            ));
        }
        let coords = mesh.element_coords(load.element)?;
        let nodes = mesh.element_nodes(load.element)?;
        let mut local_r = [F::zero(); DOF_PER_ELEMENT];
        let mut local_z = [F::zero(); DOF_PER_ELEMENT];

        for sample in face_samples_fn(&coords, load.local_face, quadrature)? {
            let tangent_norm = (sample.tangent[0] * sample.tangent[0]
                + sample.tangent[1] * sample.tangent[1])
                .sqrt();
            let scale = two_pi * sample.point[0] * tangent_norm * sample.weight;
            for local_node in 0..NODES_PER_ELEMENT {
                local_r[2 * local_node] = local_r[2 * local_node] + scale * sample.n[local_node];
                local_z[2 * local_node + 1] =
                    local_z[2 * local_node + 1] + scale * sample.n[local_node];
            }
        }

        for (local_node, global_node) in nodes.iter().copied().enumerate() {
            let dof_r = 2 * global_node;
            let dof_z = dof_r + 1;
            let local_r_index = 2 * local_node;
            let local_z_index = local_r_index + 1;
            if local_r[local_r_index] != F::zero() {
                rows.push(dof_r);
                cols.push(2 * load_index);
                vals.push(local_r[local_r_index]);
            }
            if local_z[local_z_index] != F::zero() {
                rows.push(dof_z);
                cols.push(2 * load_index + 1);
                vals.push(local_z[local_z_index]);
            }
        }
    }

    Ok(SparseOperator {
        rows,
        cols,
        vals,
        nrow: ndof,
        ncol,
    })
}

pub fn body_force_operator_quad4<F: Real>(
    mesh: MeshView<'_, F, { quad4::NODES_PER_ELEMENT }>,
    quadrature: QuadratureRule,
) -> Result<SparseOperator<F>, String> {
    body_force_operator_impl::<F, { quad4::NODES_PER_ELEMENT }, { quad4::DOF_PER_ELEMENT }>(
        mesh,
        quadrature,
        volume_samples_quad4::<F>,
    )
}

pub fn body_force_operator_quad9<F: Real>(
    mesh: MeshView<'_, F, { quad9::NODES_PER_ELEMENT }>,
    quadrature: QuadratureRule,
) -> Result<SparseOperator<F>, String> {
    body_force_operator_impl::<F, { quad9::NODES_PER_ELEMENT }, { quad9::DOF_PER_ELEMENT }>(
        mesh,
        quadrature,
        volume_samples_quad9::<F>,
    )
}

pub fn pressure_operator_quad4<F: Real>(
    mesh: MeshView<'_, F, { quad4::NODES_PER_ELEMENT }>,
    pressure_faces: &[PressureLoad<F>],
    quadrature: QuadratureRule,
) -> Result<SparseOperator<F>, String> {
    pressure_operator_impl::<F, { quad4::NODES_PER_ELEMENT }, { quad4::DOF_PER_ELEMENT }>(
        mesh,
        pressure_faces,
        quadrature,
        face_samples_quad4::<F>,
    )
}

pub fn pressure_operator_quad9<F: Real>(
    mesh: MeshView<'_, F, { quad9::NODES_PER_ELEMENT }>,
    pressure_faces: &[PressureLoad<F>],
    quadrature: QuadratureRule,
) -> Result<SparseOperator<F>, String> {
    pressure_operator_impl::<F, { quad9::NODES_PER_ELEMENT }, { quad9::DOF_PER_ELEMENT }>(
        mesh,
        pressure_faces,
        quadrature,
        face_samples_quad9::<F>,
    )
}

pub fn traction_operator_quad4<F: Real>(
    mesh: MeshView<'_, F, { quad4::NODES_PER_ELEMENT }>,
    traction_faces: &[TractionLoad<F>],
    quadrature: QuadratureRule,
) -> Result<SparseOperator<F>, String> {
    traction_operator_impl::<F, { quad4::NODES_PER_ELEMENT }, { quad4::DOF_PER_ELEMENT }>(
        mesh,
        traction_faces,
        quadrature,
        face_samples_quad4::<F>,
    )
}

pub fn traction_operator_quad9<F: Real>(
    mesh: MeshView<'_, F, { quad9::NODES_PER_ELEMENT }>,
    traction_faces: &[TractionLoad<F>],
    quadrature: QuadratureRule,
) -> Result<SparseOperator<F>, String> {
    traction_operator_impl::<F, { quad9::NODES_PER_ELEMENT }, { quad9::DOF_PER_ELEMENT }>(
        mesh,
        traction_faces,
        quadrature,
        face_samples_quad9::<F>,
    )
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
    temperature_operator_impl::<F, { quad4::NODES_PER_ELEMENT }, { quad4::DOF_PER_ELEMENT }>(
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
    temperature_operator_impl::<F, { quad9::NODES_PER_ELEMENT }, { quad9::DOF_PER_ELEMENT }>(
        mesh,
        material_ids,
        material_table,
        thermal_material_table,
        quadrature,
        volume_samples_quad9::<F>,
    )
}
