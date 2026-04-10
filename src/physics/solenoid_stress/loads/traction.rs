use crate::mesh::elements::quad2d::{quad4, quad9};
use crate::mesh::{MeshView, QuadratureRule};
use crate::physics::solenoid_stress::geometry::{
    FaceSample, face_samples_quad4, face_samples_quad9, validate_axisymmetric_nodes,
};
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, Real, TractionLoad, dof_per_element, two_pi,
};

use super::SparseOperator;

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
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_axisymmetric_nodes(mesh)?;
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

pub fn traction_operator_quad4<F: Real>(
    mesh: MeshView<'_, F, { quad4::NODES_PER_ELEMENT }>,
    traction_faces: &[TractionLoad<F>],
    quadrature: QuadratureRule,
) -> Result<SparseOperator<F>, String> {
    traction_operator_impl::<
        F,
        { quad4::NODES_PER_ELEMENT },
        { dof_per_element(quad4::NODES_PER_ELEMENT) },
    >(mesh, traction_faces, quadrature, face_samples_quad4::<F>)
}

pub fn traction_operator_quad9<F: Real>(
    mesh: MeshView<'_, F, { quad9::NODES_PER_ELEMENT }>,
    traction_faces: &[TractionLoad<F>],
    quadrature: QuadratureRule,
) -> Result<SparseOperator<F>, String> {
    traction_operator_impl::<
        F,
        { quad9::NODES_PER_ELEMENT },
        { dof_per_element(quad9::NODES_PER_ELEMENT) },
    >(mesh, traction_faces, quadrature, face_samples_quad9::<F>)
}
