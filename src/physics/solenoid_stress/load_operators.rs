//! Sparse load operators for repeated-load solves.
//!
//! These operators map load amplitudes to the global right-hand side:
//! - body-force operator: `[f_r(0), f_z(0), ...] -> rhs`
//! - pressure operator: `[p_0, p_1, ...] -> rhs`

use crate::physics::solenoid_stress::geometry::{
    FaceSample, VolumeSample, face_samples_quad4, face_samples_quad9, volume_samples_quad4,
    volume_samples_quad9,
};
use crate::physics::solenoid_stress::mesh::{MeshView, PressureLoad};
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

        for (local_node, global_node) in nodes.into_iter().enumerate() {
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

        for (local_node, global_node) in nodes.into_iter().enumerate() {
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
