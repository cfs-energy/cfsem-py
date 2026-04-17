use crate::mesh::elements::quad2d::{quad4, quad9};
use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::geometry::{
    FaceSample, face_samples_quad4, face_samples_quad9, validate_axisymmetric_mesh,
};
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, PressureLoad, Real, dof_per_element, local_dofs, two_pi,
};

use super::{SparseOperator, scatter_local_vector};

/// Build the local dense pressure-load vector for one loaded face.
///
/// The returned vector has length `DOF_PER_ELEMENT` and maps unit pressure
/// `[force / area]` on the face to the element's consistent nodal load vector
/// `[energy / distance]`.
///
/// Each vector entry therefore has units of area.
fn pressure_face_kernel<F: Real, const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    samples: &[FaceSample<F, NODES_PER_ELEMENT>],
) -> [F; DOF_PER_ELEMENT] {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    let mut local = [F::zero(); DOF_PER_ELEMENT];
    let two_pi = two_pi::<F>();

    for sample in samples {
        // Rotating the physical tangent gives `n * |dx/ds|`, so the line Jacobian is already
        // embedded in `normal_area`.
        let normal_area = [sample.tangent[1], -sample.tangent[0]];
        let scale = -two_pi * sample.point[0] * sample.weight;
        for local_node in 0..NODES_PER_ELEMENT {
            local[2 * local_node] =
                local[2 * local_node] + scale * sample.n[local_node] * normal_area[0];
            local[2 * local_node + 1] =
                local[2 * local_node + 1] + scale * sample.n[local_node] * normal_area[1];
        }
    }

    local
}

fn pressure_operator_impl<F: Real, const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
    pressure_faces: &[PressureLoad<F>],
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
    validate_axisymmetric_mesh(mesh)?;
    let ndof = mesh.num_nodes() * 2;
    let ncol = pressure_faces.len();
    let mut rows = Vec::with_capacity(pressure_faces.len() * DOF_PER_ELEMENT);
    let mut cols = Vec::with_capacity(pressure_faces.len() * DOF_PER_ELEMENT);
    let mut vals = Vec::with_capacity(pressure_faces.len() * DOF_PER_ELEMENT);

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
        let samples = face_samples_fn(&coords, load.local_face, quadrature)?;
        let local = pressure_face_kernel::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(&samples);
        let global_rows = local_dofs::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(&nodes);
        scatter_local_vector(
            &mut rows,
            &mut cols,
            &mut vals,
            &global_rows,
            load_index,
            &local,
        );
    }

    Ok(SparseOperator {
        rows,
        cols,
        vals,
        nrow: ndof,
        ncol,
    })
}

pub fn pressure_operator_quad4<F: Real>(
    mesh: QuadMeshView2d<'_, F, { quad4::NODES_PER_ELEMENT }>,
    pressure_faces: &[PressureLoad<F>],
    quadrature: QuadratureRule,
) -> Result<SparseOperator<F>, String> {
    pressure_operator_impl::<
        F,
        { quad4::NODES_PER_ELEMENT },
        { dof_per_element(quad4::NODES_PER_ELEMENT) },
    >(mesh, pressure_faces, quadrature, face_samples_quad4::<F>)
}

pub fn pressure_operator_quad9<F: Real>(
    mesh: QuadMeshView2d<'_, F, { quad9::NODES_PER_ELEMENT }>,
    pressure_faces: &[PressureLoad<F>],
    quadrature: QuadratureRule,
) -> Result<SparseOperator<F>, String> {
    pressure_operator_impl::<
        F,
        { quad9::NODES_PER_ELEMENT },
        { dof_per_element(quad9::NODES_PER_ELEMENT) },
    >(mesh, pressure_faces, quadrature, face_samples_quad9::<F>)
}
