use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::family::QuadElementFamily;
use crate::physics::solenoid_stress::geometry::{FaceSample, validate_axisymmetric_mesh};
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, PressureLoad, Real, local_dofs, two_pi,
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

/// Assemble the global pressure-to-RHS operator for one quadrilateral family.
///
/// Output shape: `(2 * mesh.num_nodes(), pressure_faces.len())`.
///
/// Row meaning:
/// - row `2*a` is the radial generalized-force equation for node `a`,
/// - row `2*a + 1` is the axial generalized-force equation for node `a`.
///
/// Column meaning:
/// - column `j` is unit pressure on `pressure_faces[j]`.
///
/// Entry units: `[area]`.
pub(crate) fn pressure_operator_for_family<
    F: Real,
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
    pressure_faces: &[PressureLoad<F>],
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
        let samples = Family::face_samples::<F>(&coords, load.local_face, quadrature)?;
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
