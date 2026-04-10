//! Axisymmetric element assembly for small-strain elasticity.
//!
//! The element system matrix is assembled in Galerkin form
//! `K_e = integral(B^T D B 2*pi*r dA)` and the consistent load vectors are assembled from the
//! same weak form.

use crate::mesh::elements::quad2d::{quad4, quad9};
use crate::mesh::{MeshView, QuadratureRule};
use crate::physics::solenoid_stress::axisym::{accumulate_stiffness, build_b_matrix};
use crate::physics::solenoid_stress::geometry::{
    FaceSample, VolumeSample, face_samples_quad4, face_samples_quad9, validate_axisymmetric_nodes,
    volume_samples_quad4, volume_samples_quad9,
};
use crate::physics::solenoid_stress::loads::{
    accumulate_body_force, accumulate_thermal_load, pressure_element_load, traction_element_load,
};
use crate::physics::solenoid_stress::types::{
    AssemblyResult, DOF_PER_NODE, PressureLoad, Real, ThermalMaterial, TractionLoad,
    dof_per_element, two_pi,
};

fn assemble_axisymmetric_impl<
    F: Real,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: MeshView<'_, F, NODES_PER_ELEMENT>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    body_force: &[[F; 2]],
    pressure_loads: &[PressureLoad<F>],
    traction_loads: &[TractionLoad<F>],
    thermal_material_table: Option<&[ThermalMaterial<F>]>,
    nodal_temperature: Option<&[F]>,
    quadrature: QuadratureRule,
    volume_samples_fn: fn(
        &[[F; 2]; NODES_PER_ELEMENT],
        QuadratureRule,
    ) -> Result<Vec<VolumeSample<F, NODES_PER_ELEMENT>>, String>,
    face_samples_fn: fn(
        &[[F; 2]; NODES_PER_ELEMENT],
        u8,
        QuadratureRule,
    ) -> Result<Vec<FaceSample<F, NODES_PER_ELEMENT>>, String>,
) -> Result<AssemblyResult<F>, String> {
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
    if body_force.len() != mesh.num_elements() {
        return Err(format!(
            "body_force has length {}, but mesh has {} elements",
            body_force.len(),
            mesh.num_elements()
        ));
    }
    if thermal_material_table.is_some() != nodal_temperature.is_some() {
        return Err(
            "thermal_material_table and nodal_temperature must either both be provided or both be omitted"
                .to_string(),
        );
    }
    if let Some(temperature) = nodal_temperature
        && temperature.len() != mesh.num_nodes()
    {
        return Err(format!(
            "nodal_temperature has length {}, but mesh has {} nodes",
            temperature.len(),
            mesh.num_nodes()
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
        let thermal_material = thermal_material_table
            .map(|table| {
                table.get(material_id).ok_or_else(|| {
                    format!(
                        "thermal material_id {material_id} on element {element_index} is out of range"
                    )
                })
            })
            .transpose()?;
        let element_body_force = body_force[element_index];
        let mut ke = [[F::zero(); DOF_PER_ELEMENT]; DOF_PER_ELEMENT];
        let mut fe = [F::zero(); DOF_PER_ELEMENT];
        let mut element_temperature = [F::zero(); NODES_PER_ELEMENT];
        if let Some(temperature) = nodal_temperature {
            for (local_node, global_node) in nodes.iter().copied().enumerate() {
                element_temperature[local_node] = temperature[global_node];
            }
        }

        for sample in volume_samples_fn(&coords, quadrature)? {
            let b = build_b_matrix::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                &sample.n,
                &sample.grad_phys,
                sample.point[0],
            )?;
            let scale = two_pi * sample.point[0] * sample.det_j * sample.weight;
            accumulate_stiffness(&mut ke, material, &b, scale);
            accumulate_body_force(&mut fe, element_body_force, &sample.n, scale);
            if let Some(thermal) = thermal_material {
                accumulate_thermal_load(
                    &mut fe,
                    material,
                    thermal,
                    &element_temperature,
                    &sample.n,
                    &b,
                    scale,
                );
            }
        }

        let mut local_dofs = [0usize; DOF_PER_ELEMENT];
        for (local_node, global_node) in nodes.iter().copied().enumerate() {
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
        let fe = pressure_element_load::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            &coords,
            load.local_face,
            load.value,
            quadrature,
            face_samples_fn,
        )?;
        for (local_node, global_node) in nodes.iter().copied().enumerate() {
            rhs[2 * global_node] = rhs[2 * global_node] + fe[2 * local_node];
            rhs[2 * global_node + 1] = rhs[2 * global_node + 1] + fe[2 * local_node + 1];
        }
    }

    for load in traction_loads {
        if load.element >= mesh.num_elements() {
            return Err(format!(
                "traction load references element {}, but mesh has only {} elements",
                load.element,
                mesh.num_elements()
            ));
        }
        let coords = mesh.element_coords(load.element)?;
        let nodes = mesh.element_nodes(load.element)?;
        let fe = traction_element_load::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            &coords,
            load.local_face,
            load.value,
            quadrature,
            face_samples_fn,
        )?;
        for (local_node, global_node) in nodes.iter().copied().enumerate() {
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

/// Assemble the global axisymmetric Quad4 stiffness matrix and right-hand side.
pub fn assemble_axisymmetric_quad4<F: Real>(
    mesh: MeshView<'_, F, { quad4::NODES_PER_ELEMENT }>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    body_force: &[[F; 2]],
    pressure_loads: &[PressureLoad<F>],
    traction_loads: &[TractionLoad<F>],
    thermal_material_table: Option<&[ThermalMaterial<F>]>,
    nodal_temperature: Option<&[F]>,
    quadrature: QuadratureRule,
) -> Result<AssemblyResult<F>, String> {
    assemble_axisymmetric_impl::<
        F,
        { quad4::NODES_PER_ELEMENT },
        { dof_per_element(quad4::NODES_PER_ELEMENT) },
    >(
        mesh,
        material_ids,
        material_table,
        body_force,
        pressure_loads,
        traction_loads,
        thermal_material_table,
        nodal_temperature,
        quadrature,
        volume_samples_quad4::<F>,
        face_samples_quad4::<F>,
    )
}

/// Assemble the global axisymmetric Quad9 stiffness matrix and right-hand side.
pub fn assemble_axisymmetric_quad9<F: Real>(
    mesh: MeshView<'_, F, { quad9::NODES_PER_ELEMENT }>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    body_force: &[[F; 2]],
    pressure_loads: &[PressureLoad<F>],
    traction_loads: &[TractionLoad<F>],
    thermal_material_table: Option<&[ThermalMaterial<F>]>,
    nodal_temperature: Option<&[F]>,
    quadrature: QuadratureRule,
) -> Result<AssemblyResult<F>, String> {
    assemble_axisymmetric_impl::<
        F,
        { quad9::NODES_PER_ELEMENT },
        { dof_per_element(quad9::NODES_PER_ELEMENT) },
    >(
        mesh,
        material_ids,
        material_table,
        body_force,
        pressure_loads,
        traction_loads,
        thermal_material_table,
        nodal_temperature,
        quadrature,
        volume_samples_quad9::<F>,
        face_samples_quad9::<F>,
    )
}

#[cfg(test)]
mod tests {
    use super::assemble_axisymmetric_quad4;
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
            &[],
            None,
            None,
            QuadratureRule::GaussLegendre3,
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
