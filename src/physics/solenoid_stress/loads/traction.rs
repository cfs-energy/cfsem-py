use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::family::QuadElementFamily;
use crate::physics::solenoid_stress::geometry::{FaceSample, validate_axisymmetric_mesh};
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, Real, TractionLoad, local_dofs, two_pi,
};

use super::{SparseOperator, scatter_local_matrix};

/// Build the local dense traction operator for one loaded face.
///
/// The returned block has shape `(DOF_PER_ELEMENT, 2)`. Its two columns correspond to:
/// - column `0`: unit radial traction `[force / area]`
/// - column `1`: unit axial traction `[force / area]`
///
/// Multiplying this block by `[t_r, t_z]^T` gives the element's consistent nodal load vector
/// `[energy / distance]`.
///
/// Each block entry therefore has units of area.
fn traction_face_kernel<F: Real, const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    samples: &[FaceSample<F, NODES_PER_ELEMENT>],
) -> [[F; 2]; DOF_PER_ELEMENT] {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    let mut local = [[F::zero(); 2]; DOF_PER_ELEMENT];
    let two_pi = two_pi::<F>();

    for sample in samples {
        // `|dx/ds|` is the physical line Jacobian for the face quadrature parameter.
        let tangent_norm =
            (sample.tangent[0] * sample.tangent[0] + sample.tangent[1] * sample.tangent[1]).sqrt();
        let scale = two_pi * sample.point[0] * tangent_norm * sample.weight;
        for local_node in 0..NODES_PER_ELEMENT {
            // The two columns encode independent unit tractions in the global radial and axial
            // directions, so the block is diagonal in those two traction components.
            local[2 * local_node][0] = local[2 * local_node][0] + scale * sample.n[local_node];
            local[2 * local_node + 1][1] =
                local[2 * local_node + 1][1] + scale * sample.n[local_node];
        }
    }

    local
}

pub(crate) fn traction_operator_for_family<
    F: Real,
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
    traction_faces: &[TractionLoad<F>],
    quadrature: QuadratureRule,
) -> Result<SparseOperator<F>, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_axisymmetric_mesh(mesh)?;
    let ndof = mesh.num_nodes() * 2;
    let ncol = 2 * traction_faces.len();
    let mut rows = Vec::with_capacity(traction_faces.len() * DOF_PER_ELEMENT * 2);
    let mut cols = Vec::with_capacity(traction_faces.len() * DOF_PER_ELEMENT * 2);
    let mut vals = Vec::with_capacity(traction_faces.len() * DOF_PER_ELEMENT * 2);

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
        let samples = Family::face_samples::<F>(&coords, load.local_face, quadrature)?;
        let local = traction_face_kernel::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(&samples);
        let global_rows = local_dofs::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(&nodes);
        let global_cols = [2 * load_index, 2 * load_index + 1];
        scatter_local_matrix(
            &mut rows,
            &mut cols,
            &mut vals,
            &global_rows,
            &global_cols,
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
