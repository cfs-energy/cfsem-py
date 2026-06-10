use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::family::QuadElementFamily;
use crate::physics::solenoid_stress::geometry::{FaceSample, validate_structural_2d_mesh};
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, PressureLoad, Structural2dFormulation, local_dofs, scatter_local_vector,
};

use super::{SparseOperator, collect_sparse_operator_chunks, concat_sparse_operators};

/// Build the local dense pressure-load vector for one loaded face.
///
/// The returned vector has length `DOF_PER_ELEMENT` and maps unit pressure
/// `[force / area]` on the face to the element's consistent nodal load vector
/// `[energy / distance]`.
///
/// Each vector entry therefore has units of area.
fn pressure_face_kernel<const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    samples: &[FaceSample<f64, NODES_PER_ELEMENT>],
    formulation: Structural2dFormulation,
) -> Result<[f64; DOF_PER_ELEMENT], String> {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    let mut local = [0.0; DOF_PER_ELEMENT];

    for sample in samples {
        // Rotating the physical tangent gives `n * |dx/ds|`, so the line Jacobian is already
        // embedded in `normal_area`.
        let normal_area = [sample.tangent[1], -sample.tangent[0]];
        let scale = -formulation.face_scale(sample.point, 1.0, sample.weight)?;
        for local_node in 0..NODES_PER_ELEMENT {
            local[2 * local_node] =
                local[2 * local_node] + scale * sample.n[local_node] * normal_area[0];
            local[2 * local_node + 1] =
                local[2 * local_node + 1] + scale * sample.n[local_node] * normal_area[1];
        }
    }

    Ok(local)
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
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    pressure_faces: &[PressureLoad],
    formulation: Structural2dFormulation,
    quadrature: QuadratureRule,
) -> Result<SparseOperator, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    pressure_operator_range_for_family::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
        mesh,
        pressure_faces,
        formulation,
        quadrature,
        0,
        pressure_faces.len(),
    )
}

/// Assemble pressure loads using pressure-face ranges split across Rayon workers.
pub(crate) fn pressure_operator_for_family_par<
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    pressure_faces: &[PressureLoad],
    formulation: Structural2dFormulation,
    quadrature: QuadratureRule,
) -> Result<SparseOperator, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    let chunks = collect_sparse_operator_chunks(pressure_faces.len(), |start, end| {
        pressure_operator_range_for_family::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            mesh,
            pressure_faces,
            formulation,
            quadrature,
            start,
            end,
        )
    })?;

    Ok(concat_sparse_operators(
        chunks,
        mesh.num_nodes() * 2,
        pressure_faces.len(),
        pressure_faces.len() * DOF_PER_ELEMENT,
    ))
}

/// Apply pressure amplitudes directly to a reduced RHS without building a sparse operator.
pub(crate) fn apply_pressure_rhs_for_family<
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    pressure_faces: &[PressureLoad],
    formulation: Structural2dFormulation,
    quadrature: QuadratureRule,
    values: &[f64],
    global_to_reduced: &[usize],
    rhs: &mut [f64],
) -> Result<(), String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    if values.len() != pressure_faces.len() {
        return Err(format!(
            "pressure_values has length {}, but operator expects {} values",
            values.len(),
            pressure_faces.len()
        ));
    }
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
        let samples = Family::face_samples(&coords, load.local_face, quadrature)?;
        let local =
            pressure_face_kernel::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(&samples, formulation)?;
        let global_rows = local_dofs::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(&nodes);
        let value = values[load_index];
        for local_dof in 0..DOF_PER_ELEMENT {
            let reduced_row = global_to_reduced[global_rows[local_dof]];
            if reduced_row != usize::MAX {
                rhs[reduced_row] += local[local_dof] * value;
            }
        }
    }
    Ok(())
}

fn pressure_operator_range_for_family<
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    pressure_faces: &[PressureLoad],
    formulation: Structural2dFormulation,
    quadrature: QuadratureRule,
    load_start: usize,
    load_end: usize,
) -> Result<SparseOperator, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    let ndof = mesh.num_nodes() * 2;
    let ncol = pressure_faces.len();
    let nloads = load_end - load_start;
    let mut rows = Vec::with_capacity(nloads * DOF_PER_ELEMENT);
    let mut cols = Vec::with_capacity(nloads * DOF_PER_ELEMENT);
    let mut vals = Vec::with_capacity(nloads * DOF_PER_ELEMENT);

    for (load_index, load) in pressure_faces
        .iter()
        .enumerate()
        .skip(load_start)
        .take(nloads)
    {
        if load.element >= mesh.num_elements() {
            return Err(format!(
                "pressure load references element {}, but mesh has only {} elements",
                load.element,
                mesh.num_elements()
            ));
        }
        let coords = mesh.element_coords(load.element)?;
        let nodes = mesh.element_nodes(load.element)?;
        let samples = Family::face_samples(&coords, load.local_face, quadrature)?;
        let local =
            pressure_face_kernel::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(&samples, formulation)?;
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
