use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::family::QuadElementFamily;
use crate::physics::solenoid_stress::geometry::{FaceSample, validate_structural_2d_mesh};
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, Real, Structural2dFormulation, TractionLoad, local_dofs, scatter_local_matrix,
};

use super::{SparseOperator, collect_sparse_operator_chunks, concat_sparse_operators};

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
    formulation: Structural2dFormulation<F>,
) -> Result<[[F; 2]; DOF_PER_ELEMENT], String> {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    let mut local = [[F::zero(); 2]; DOF_PER_ELEMENT];

    for sample in samples {
        // `|dx/ds|` is the physical line Jacobian for the face quadrature parameter.
        let tangent_norm =
            (sample.tangent[0] * sample.tangent[0] + sample.tangent[1] * sample.tangent[1]).sqrt();
        let scale = formulation.face_scale(sample.point, tangent_norm, sample.weight)?;
        for local_node in 0..NODES_PER_ELEMENT {
            // The two columns encode independent unit tractions in the global radial and axial
            // directions, so the block is diagonal in those two traction components.
            local[2 * local_node][0] = local[2 * local_node][0] + scale * sample.n[local_node];
            local[2 * local_node + 1][1] =
                local[2 * local_node + 1][1] + scale * sample.n[local_node];
        }
    }

    Ok(local)
}

/// Assemble the global traction-to-RHS operator for one quadrilateral family.
///
/// Output shape: `(2 * mesh.num_nodes(), 2 * traction_faces.len())`.
///
/// Row meaning:
/// - row `2*a` is the radial generalized-force equation for node `a`,
/// - row `2*a + 1` is the axial generalized-force equation for node `a`.
///
/// Column meaning:
/// - column `2*j` is unit radial traction on `traction_faces[j]`,
/// - column `2*j + 1` is unit axial traction on `traction_faces[j]`.
///
/// Entry units: `[area]`.
pub(crate) fn traction_operator_for_family<
    F: Real,
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
    traction_faces: &[TractionLoad],
    formulation: Structural2dFormulation<F>,
    quadrature: QuadratureRule,
) -> Result<SparseOperator<F>, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    traction_operator_range_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
        mesh,
        traction_faces,
        formulation,
        quadrature,
        0,
        traction_faces.len(),
    )
}

/// Assemble traction loads using traction-face ranges split across Rayon workers.
pub(crate) fn traction_operator_for_family_par<
    F: Real,
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
    traction_faces: &[TractionLoad],
    formulation: Structural2dFormulation<F>,
    quadrature: QuadratureRule,
) -> Result<SparseOperator<F>, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    let chunks = collect_sparse_operator_chunks(traction_faces.len(), |start, end| {
        traction_operator_range_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            mesh,
            traction_faces,
            formulation,
            quadrature,
            start,
            end,
        )
    })?;

    Ok(concat_sparse_operators(
        chunks,
        mesh.num_nodes() * 2,
        2 * traction_faces.len(),
        traction_faces.len() * DOF_PER_ELEMENT * 2,
    ))
}

fn traction_operator_range_for_family<
    F: Real,
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, F, NODES_PER_ELEMENT>,
    traction_faces: &[TractionLoad],
    formulation: Structural2dFormulation<F>,
    quadrature: QuadratureRule,
    load_start: usize,
    load_end: usize,
) -> Result<SparseOperator<F>, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    let ndof = mesh.num_nodes() * 2;
    let ncol = 2 * traction_faces.len();
    let nloads = load_end - load_start;
    let mut rows = Vec::with_capacity(nloads * DOF_PER_ELEMENT * 2);
    let mut cols = Vec::with_capacity(nloads * DOF_PER_ELEMENT * 2);
    let mut vals = Vec::with_capacity(nloads * DOF_PER_ELEMENT * 2);

    for (load_index, load) in traction_faces
        .iter()
        .enumerate()
        .skip(load_start)
        .take(nloads)
    {
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
        let local =
            traction_face_kernel::<F, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(&samples, formulation)?;
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
