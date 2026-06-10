//! Sparse strain/stress recovery operators at element quadrature points.
//!
//! The displacement derivatives used in strain and stress recovery are implemented analytically
//! from the element shape functions, then mapped from reference-element coordinates into physical
//! meridian coordinates `(r, z)` with the element Jacobian. There is no finite-difference
//! approximation of the displacement field.
//!
//! Mechanical strain/stress recovery uses the same query-coordinate sparse operators that serve
//! arbitrary point probes. Quadrature recovery builds the known element-major
//! `(element_index, reference_point)` arrays directly, so it reuses that math without doing a
//! nearest-element search.
//!
//! The underlying chain is:
//! 1. Each element family defines closed-form shape functions `N_i(\xi,\eta)` and closed-form
//!    reference gradients.
//! 2. At each quadrature point, those reference gradients are mapped into physical-space
//!    gradients with the inverse Jacobian according to
//!    `partial N / partial (r, z) = J^{-T} partial N / partial (\xi, \eta)`.
//! 3. The strain-displacement matrix `B` is then built from those physical derivatives.
//!
//! In that `B` matrix:
//! - `e_rr = partial u_r / partial r`,
//! - `e_zz = partial u_z / partial z`,
//! - `g_rz = partial u_r / partial z + partial u_z / partial r`,
//! so those rows use the entries of `grad_phys` directly.
//!
//! The hoop term is slightly different:
//! - `e_tt = u_r / r`,
//! so that row uses `N_i / r`, not a spatial derivative.
//!
//! 4. Recovery then uses that same `B`:
//!    - strain recovery uses `B`,
//!    - stress recovery uses `D B`,
//!    where `D` is the local per-material `4 x 4` matrix for the elastic stress-strain law.
//!
//! Thermal recovery remains quadrature-specific because it maps nodal temperatures and material
//! reference temperatures to thermal strain/stress offsets.

use rayon::prelude::*;

#[cfg(test)]
use crate::mesh::elements::quad2d::quadrature::gauss_volume;
#[cfg(test)]
use crate::mesh::quad2d::{quad_mesh_strain_operator, quad_mesh_stress_operator};
use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::axisym::constitutive_times_strain;
use crate::physics::solenoid_stress::convenience::{
    rotate_material_in_plane, rotate_thermal_material_in_plane,
};
use crate::physics::solenoid_stress::family::QuadElementFamily;
use crate::physics::solenoid_stress::geometry::{VolumeSample, validate_structural_2d_mesh};
#[cfg(test)]
use crate::physics::solenoid_stress::types::scatter_local_matrix;
use crate::physics::solenoid_stress::types::{
    DOF_PER_NODE, Structural2dFormulation, ThermalMaterial, validate_element_material_inputs,
};
use crate::{chunksize, ranges_for_len};

/// Sparse quadrature-point recovery operators before reduction into the model-owned CSR form.
///
/// Rows are stored in quadrature-point-major order with axisymmetric component ordering
/// `[rr, zz, tt, rz]`, so rows `4*q .. 4*q + 3` correspond to one quadrature point.
#[cfg(test)]
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct QuadratureFieldOperators {
    /// Quadrature-point coordinates `(r, z)` in element-major order.
    ///
    /// Units: `[length]`.
    pub points: Vec<[f64; 2]>,
    /// Sparse row indices for the strain operator triplets.
    pub strain_rows: Vec<usize>,
    /// Sparse column indices for the strain operator triplets.
    ///
    /// Columns index full displacement DOFs `[u_r1, u_z1, ...]`.
    pub strain_cols: Vec<usize>,
    /// Sparse values for the strain operator triplets.
    ///
    /// Units: `[strain / displacement] = [1 / length]`.
    pub strain_vals: Vec<f64>,
    /// Sparse row indices for the stress operator triplets.
    pub stress_rows: Vec<usize>,
    /// Sparse column indices for the stress operator triplets.
    ///
    /// Columns index full displacement DOFs `[u_r1, u_z1, ...]`.
    pub stress_cols: Vec<usize>,
    /// Sparse values for the stress operator triplets.
    ///
    /// Units: `[stress / displacement] = [pressure / length]`.
    pub stress_vals: Vec<f64>,
    /// Sparse row indices for the thermal-strain operator triplets.
    pub thermal_strain_rows: Vec<usize>,
    /// Sparse column indices for the thermal-strain operator triplets.
    ///
    /// Columns index nodal temperatures `[temperature]`.
    pub thermal_strain_cols: Vec<usize>,
    /// Sparse values for the thermal-strain operator triplets.
    ///
    /// Units: `[strain / temperature]`.
    pub thermal_strain_vals: Vec<f64>,
    /// Sparse row indices for the thermal-stress operator triplets.
    pub thermal_stress_rows: Vec<usize>,
    /// Sparse column indices for the thermal-stress operator triplets.
    ///
    /// Columns index nodal temperatures `[temperature]`.
    pub thermal_stress_cols: Vec<usize>,
    /// Sparse values for the thermal-stress operator triplets.
    ///
    /// Units: `[stress / temperature]`.
    pub thermal_stress_vals: Vec<f64>,
    /// Constant quadrature-point thermal strain contribution from per-material reference temperature.
    ///
    /// Units: `[strain]`.
    pub thermal_strain_constant: Vec<f64>,
    /// Constant quadrature-point thermal stress contribution from per-material reference temperature.
    ///
    /// Units: `[stress]`.
    pub thermal_stress_constant: Vec<f64>,
    /// Number of quadrature points contributed by each element.
    pub nq_per_element: usize,
    /// Number of nodal temperatures the thermal operators act on.
    pub ntemp: usize,
}

/// Canonical CSR operator parts with strictly increasing column indices in each row.
#[derive(Debug, Clone)]
pub struct CsrOperatorParts {
    /// Number of matrix rows.
    pub nrow: usize,
    /// Number of matrix columns.
    pub ncol: usize,
    /// CSR row offsets with length `nrow + 1`.
    pub row_ptr: Vec<usize>,
    /// Column index for each stored value.
    pub col_idx: Vec<usize>,
    /// Nonzero values in row-major CSR order.
    pub vals: Vec<f64>,
}

struct CsrPartsBuilder {
    nrow: usize,
    ncol: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    vals: Vec<f64>,
}

impl CsrPartsBuilder {
    /// Start a CSR builder whose rows must be appended in increasing row order.
    fn new(nrow: usize, ncol: usize) -> Self {
        let mut row_ptr = Vec::with_capacity(nrow + 1);
        row_ptr.push(0);
        Self {
            nrow,
            ncol,
            row_ptr,
            col_idx: Vec::new(),
            vals: Vec::new(),
        }
    }

    /// Sort, coalesce, and append one row of `(column, value)` entries.
    ///
    /// Recovery assembly naturally emits a small unsorted row from local shape-function
    /// contributions.  Canonicalizing one row at a time avoids a global triplet sort and makes the
    /// finished parts directly suitable for faer's CSR constructor.
    fn push_canonical_row(&mut self, entries: &mut Vec<(usize, f64)>) {
        entries.sort_unstable_by_key(|(col, _)| *col);
        let mut pending: Option<(usize, f64)> = None;
        for (col, value) in entries.drain(..) {
            if value == 0.0 {
                continue;
            }
            match pending {
                Some((pending_col, pending_value)) if pending_col == col => {
                    pending = Some((pending_col, pending_value + value));
                }
                Some((pending_col, pending_value)) => {
                    if pending_value != 0.0 {
                        self.col_idx.push(pending_col);
                        self.vals.push(pending_value);
                    }
                    pending = Some((col, value));
                }
                None => pending = Some((col, value)),
            }
        }
        if let Some((col, value)) = pending
            && value != 0.0
        {
            self.col_idx.push(col);
            self.vals.push(value);
        }
        self.push_empty_row();
    }

    /// Append a row with no stored entries.
    fn push_empty_row(&mut self) {
        self.row_ptr.push(self.col_idx.len());
    }

    /// Finish the builder after exactly `nrow` rows have been appended.
    fn finish(self) -> CsrOperatorParts {
        debug_assert_eq!(self.row_ptr.len(), self.nrow + 1);
        CsrOperatorParts {
            nrow: self.nrow,
            ncol: self.ncol,
            row_ptr: self.row_ptr,
            col_idx: self.col_idx,
            vals: self.vals,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ReducedRecoverySelection {
    pub points: bool,
    pub strain_operator: bool,
    pub stress_operator: bool,
    pub thermal_strain_operator: bool,
    pub thermal_stress_operator: bool,
    pub strain_constant: bool,
    pub stress_constant: bool,
    pub thermal_strain_constant: bool,
    pub thermal_stress_constant: bool,
}

impl ReducedRecoverySelection {
    const fn needs_strain_rows(self) -> bool {
        self.strain_operator || self.strain_constant
    }

    const fn needs_stress_rows(self) -> bool {
        self.stress_operator || self.stress_constant
    }

    const fn needs_thermal_rows(self) -> bool {
        self.thermal_strain_operator
            || self.thermal_stress_operator
            || self.thermal_strain_constant
            || self.thermal_stress_constant
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SelectedReducedQuadratureFieldOperators {
    pub points: Option<Vec<[f64; 2]>>,
    pub strain_operator: Option<CsrOperatorParts>,
    pub stress_operator: Option<CsrOperatorParts>,
    pub thermal_strain_operator: Option<CsrOperatorParts>,
    pub thermal_stress_operator: Option<CsrOperatorParts>,
    pub strain_constant: Option<Vec<f64>>,
    pub stress_constant: Option<Vec<f64>>,
    pub thermal_strain_constant: Option<Vec<f64>>,
    pub thermal_stress_constant: Option<Vec<f64>>,
}

/// Dense thermal recovery operators for one quadrature point.
///
/// Units:
/// - `thermal_strain`: `[strain / temperature]`
/// - `thermal_stress`: `[stress / temperature]`
/// - `thermal_*_constant`: `strain` and `stress`, respectively
struct LocalThermalSampleKernel<const NODES_PER_ELEMENT: usize> {
    thermal_strain: [[f64; NODES_PER_ELEMENT]; 4],
    thermal_stress: [[f64; NODES_PER_ELEMENT]; 4],
    thermal_strain_constant: [f64; 4],
    thermal_stress_constant: [f64; 4],
}

/// Build the dense thermal recovery blocks for one quadrature point.
///
/// This helper builds the local thermal operators and constant offsets associated with the
/// material reference temperature. Mechanical strain/stress recovery is assembled through the
/// shared query-coordinate mesh operators.
fn thermal_sample_kernel<const NODES_PER_ELEMENT: usize>(
    sample: &VolumeSample<f64, NODES_PER_ELEMENT>,
    thermal: &ThermalMaterial,
    thermal_stress_unit: &[f64; 4],
) -> LocalThermalSampleKernel<NODES_PER_ELEMENT> {
    let mut local = LocalThermalSampleKernel {
        thermal_strain: [[0.0; NODES_PER_ELEMENT]; 4],
        thermal_stress: [[0.0; NODES_PER_ELEMENT]; 4],
        thermal_strain_constant: [0.0; 4],
        thermal_stress_constant: [0.0; 4],
    };

    for component in 0..4 {
        for local_temp_node in 0..NODES_PER_ELEMENT {
            // These blocks map the nodal temperature field directly to thermal strain/stress
            // at this quadrature point.
            local.thermal_strain[component][local_temp_node] =
                thermal.alpha[component] * sample.n[local_temp_node];
            local.thermal_stress[component][local_temp_node] =
                thermal_stress_unit[component] * sample.n[local_temp_node];
        }
        local.thermal_strain_constant[component] =
            -thermal.alpha[component] * thermal.reference_temperature;
        local.thermal_stress_constant[component] =
            -thermal_stress_unit[component] * thermal.reference_temperature;
    }

    local
}

/// Return element-major quadrature-point references without doing a geometric point query.
///
/// Quadrature recovery already knows which element owns each point. These arrays have the same
/// shape expected by the query-coordinate strain/stress operators, but avoid the `O(nquery *
/// nelem)` nearest-element search that `query_quad_mesh` performs for arbitrary physical points.
#[cfg(test)]
fn element_major_reference_points_range(
    element_start: usize,
    element_end: usize,
    quadrature: QuadratureRule,
) -> (Vec<usize>, Vec<[f64; 2]>) {
    let references = gauss_volume::<f64>(quadrature);
    let nelem = element_end - element_start;
    let mut element_indices = Vec::with_capacity(nelem * references.len());
    let mut reference_points = Vec::with_capacity(nelem * references.len());
    for element_index in element_start..element_end {
        for &(reference, _) in &references {
            element_indices.push(element_index);
            reference_points.push(reference);
        }
    }
    (element_indices, reference_points)
}

fn append_selected_reduced_displacement_entry(
    entries: &mut Vec<(usize, f64)>,
    collect_operator_entry: bool,
    constant: Option<&mut [f64]>,
    row: usize,
    full_col: usize,
    value: f64,
    global_to_reduced: &[usize],
    fixed_lookup: &[Option<f64>],
) {
    if value == 0.0 {
        return;
    }
    let reduced_col = global_to_reduced[full_col];
    if reduced_col != usize::MAX {
        if collect_operator_entry {
            entries.push((reduced_col, value));
        }
    } else if let (Some(fixed_value), Some(constant)) = (fixed_lookup[full_col], constant) {
        constant[row] = constant[row] + value * fixed_value;
    }
}

fn append_temperature_entry(entries: &mut Vec<(usize, f64)>, col: usize, value: f64) {
    // Temperature is not part of the structural Dirichlet reduction, so thermal recovery columns
    // remain indexed by the input temperature node.
    if value != 0.0 {
        entries.push((col, value));
    }
}

fn concat_csr_chunks(chunks: Vec<CsrOperatorParts>, nrow: usize, ncol: usize) -> CsrOperatorParts {
    // Each chunk covers a contiguous element range and therefore a contiguous row range.  The
    // per-row column order is already canonical, so concatenation only needs to offset row
    // pointers by the number of stored entries seen so far.
    let nnz = chunks.iter().map(|chunk| chunk.vals.len()).sum::<usize>();
    let mut row_ptr = Vec::with_capacity(nrow + 1);
    let mut col_idx = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);
    row_ptr.push(0);
    for chunk in chunks {
        debug_assert_eq!(chunk.ncol, ncol);
        let offset = col_idx.len();
        col_idx.extend(chunk.col_idx);
        vals.extend(chunk.vals);
        row_ptr.extend(chunk.row_ptr.into_iter().skip(1).map(|ptr| ptr + offset));
    }
    debug_assert_eq!(row_ptr.len(), nrow + 1);
    CsrOperatorParts {
        nrow,
        ncol,
        row_ptr,
        col_idx,
        vals,
    }
}

fn concat_selected_reduced_quadrature_chunks(
    chunks: Vec<SelectedReducedQuadratureFieldOperators>,
    selection: ReducedRecoverySelection,
    nelem: usize,
    nq_per_element: usize,
    ndof_reduced: usize,
    n_temperature_nodes: usize,
) -> SelectedReducedQuadratureFieldOperators {
    let npoints = nelem * nq_per_element;
    let nrow = npoints * 4;

    let mut points = selection.points.then(|| Vec::with_capacity(npoints));
    let mut strain_chunks = selection
        .strain_operator
        .then(|| Vec::with_capacity(chunks.len()));
    let mut stress_chunks = selection
        .stress_operator
        .then(|| Vec::with_capacity(chunks.len()));
    let mut thermal_strain_chunks = selection
        .thermal_strain_operator
        .then(|| Vec::with_capacity(chunks.len()));
    let mut thermal_stress_chunks = selection
        .thermal_stress_operator
        .then(|| Vec::with_capacity(chunks.len()));
    let mut strain_constant = selection.strain_constant.then(|| Vec::with_capacity(nrow));
    let mut stress_constant = selection.stress_constant.then(|| Vec::with_capacity(nrow));
    let mut thermal_strain_constant = selection
        .thermal_strain_constant
        .then(|| Vec::with_capacity(nrow));
    let mut thermal_stress_constant = selection
        .thermal_stress_constant
        .then(|| Vec::with_capacity(nrow));

    for chunk in chunks {
        if let Some(values) = points.as_mut() {
            values.extend(chunk.points.expect("selected points chunk"));
        }
        if let Some(values) = strain_chunks.as_mut() {
            values.push(chunk.strain_operator.expect("selected strain chunk"));
        }
        if let Some(values) = stress_chunks.as_mut() {
            values.push(chunk.stress_operator.expect("selected stress chunk"));
        }
        if let Some(values) = thermal_strain_chunks.as_mut() {
            values.push(
                chunk
                    .thermal_strain_operator
                    .expect("selected thermal strain chunk"),
            );
        }
        if let Some(values) = thermal_stress_chunks.as_mut() {
            values.push(
                chunk
                    .thermal_stress_operator
                    .expect("selected thermal stress chunk"),
            );
        }
        if let Some(values) = strain_constant.as_mut() {
            values.extend(
                chunk
                    .strain_constant
                    .expect("selected strain constant chunk"),
            );
        }
        if let Some(values) = stress_constant.as_mut() {
            values.extend(
                chunk
                    .stress_constant
                    .expect("selected stress constant chunk"),
            );
        }
        if let Some(values) = thermal_strain_constant.as_mut() {
            values.extend(
                chunk
                    .thermal_strain_constant
                    .expect("selected thermal strain constant chunk"),
            );
        }
        if let Some(values) = thermal_stress_constant.as_mut() {
            values.extend(
                chunk
                    .thermal_stress_constant
                    .expect("selected thermal stress constant chunk"),
            );
        }
    }

    SelectedReducedQuadratureFieldOperators {
        points,
        strain_operator: strain_chunks.map(|chunks| concat_csr_chunks(chunks, nrow, ndof_reduced)),
        stress_operator: stress_chunks.map(|chunks| concat_csr_chunks(chunks, nrow, ndof_reduced)),
        thermal_strain_operator: thermal_strain_chunks
            .map(|chunks| concat_csr_chunks(chunks, nrow, n_temperature_nodes)),
        thermal_stress_operator: thermal_stress_chunks
            .map(|chunks| concat_csr_chunks(chunks, nrow, n_temperature_nodes)),
        strain_constant,
        stress_constant,
        thermal_strain_constant,
        thermal_stress_constant,
    }
}

/// Assemble quadrature-point strain/stress recovery operators for one quadrilateral family.
///
/// Output shapes:
/// - `strain_*` and `stress_*`: `(4 * nq_per_element * mesh.num_elements(), 2 * mesh.num_nodes())`
/// - `thermal_*`: `(4 * nq_per_element * mesh.num_elements(), mesh.num_nodes())`
/// - `*_constant`: `(4 * nq_per_element * mesh.num_elements(),)`
///
/// Row meaning:
/// - rows `4*q .. 4*q + 3` correspond to quadrature point `q` in element-major order,
/// - within each quadrature point the row components are ordered `[rr, zz, tt, rz]`.
#[cfg(test)]
pub(crate) fn quadrature_field_operators_for_family<
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    material_ids: &[usize],
    material_table: &[[[f64; 4]; 4]],
    thermal_material_table: Option<&[ThermalMaterial]>,
    material_orientation_angles: Option<&[f64]>,
    formulation: Structural2dFormulation,
    quadrature: QuadratureRule,
) -> Result<QuadratureFieldOperators, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    validate_element_material_inputs(
        mesh.num_elements(),
        material_ids,
        material_orientation_angles,
    )?;

    quadrature_field_operators_range_for_family::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
        mesh,
        material_ids,
        material_table,
        thermal_material_table,
        material_orientation_angles,
        formulation,
        quadrature,
        0,
        mesh.num_elements(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn selected_reduced_quadrature_field_operators_for_family<
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    material_ids: &[usize],
    material_table: &[[[f64; 4]; 4]],
    thermal_material_table: Option<&[ThermalMaterial]>,
    material_orientation_angles: Option<&[f64]>,
    formulation: Structural2dFormulation,
    quadrature: QuadratureRule,
    global_to_reduced: &[usize],
    fixed_lookup: &[Option<f64>],
    ndof_reduced: usize,
    selection: ReducedRecoverySelection,
) -> Result<SelectedReducedQuadratureFieldOperators, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    validate_element_material_inputs(
        mesh.num_elements(),
        material_ids,
        material_orientation_angles,
    )?;
    selected_reduced_quadrature_field_operators_range_for_family::<
        Family,
        NODES_PER_ELEMENT,
        DOF_PER_ELEMENT,
    >(
        mesh,
        material_ids,
        material_table,
        thermal_material_table,
        material_orientation_angles,
        formulation,
        quadrature,
        global_to_reduced,
        fixed_lookup,
        ndof_reduced,
        selection,
        0,
        mesh.num_elements(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn selected_reduced_quadrature_field_operators_for_family_par<
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    material_ids: &[usize],
    material_table: &[[[f64; 4]; 4]],
    thermal_material_table: Option<&[ThermalMaterial]>,
    material_orientation_angles: Option<&[f64]>,
    formulation: Structural2dFormulation,
    quadrature: QuadratureRule,
    global_to_reduced: &[usize],
    fixed_lookup: &[Option<f64>],
    ndof_reduced: usize,
    selection: ReducedRecoverySelection,
) -> Result<SelectedReducedQuadratureFieldOperators, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    validate_structural_2d_mesh(mesh, formulation)?;
    validate_element_material_inputs(
        mesh.num_elements(),
        material_ids,
        material_orientation_angles,
    )?;
    let nelem = mesh.num_elements();
    let nq_per_element = quadrature.points_per_element();
    let n_temperature_nodes = if thermal_material_table.is_some() {
        mesh.num_nodes()
    } else {
        0
    };
    let chunks = ranges_for_len(nelem, chunksize(nelem))
        .into_par_iter()
        .map(|(start, end)| {
            selected_reduced_quadrature_field_operators_range_for_family::<
                Family,
                NODES_PER_ELEMENT,
                DOF_PER_ELEMENT,
            >(
                mesh,
                material_ids,
                material_table,
                thermal_material_table,
                material_orientation_angles,
                formulation,
                quadrature,
                global_to_reduced,
                fixed_lookup,
                ndof_reduced,
                selection,
                start,
                end,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(concat_selected_reduced_quadrature_chunks(
        chunks,
        selection,
        nelem,
        nq_per_element,
        ndof_reduced,
        n_temperature_nodes,
    ))
}

#[allow(clippy::too_many_arguments)]
fn selected_reduced_quadrature_field_operators_range_for_family<
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    material_ids: &[usize],
    material_table: &[[[f64; 4]; 4]],
    thermal_material_table: Option<&[ThermalMaterial]>,
    material_orientation_angles: Option<&[f64]>,
    formulation: Structural2dFormulation,
    quadrature: QuadratureRule,
    global_to_reduced: &[usize],
    fixed_lookup: &[Option<f64>],
    ndof_reduced: usize,
    selection: ReducedRecoverySelection,
    element_start: usize,
    element_end: usize,
) -> Result<SelectedReducedQuadratureFieldOperators, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    let nq_per_element = quadrature.points_per_element();
    let n_elements = element_end - element_start;
    let nrows = n_elements * nq_per_element * 4;
    let n_temperature_nodes = if thermal_material_table.is_some() {
        mesh.num_nodes()
    } else {
        0
    };
    let mut points = selection
        .points
        .then(|| Vec::with_capacity(n_elements * nq_per_element));
    let mut strain_operator = selection
        .strain_operator
        .then(|| CsrPartsBuilder::new(nrows, ndof_reduced));
    let mut stress_operator = selection
        .stress_operator
        .then(|| CsrPartsBuilder::new(nrows, ndof_reduced));
    let mut thermal_strain_operator = selection
        .thermal_strain_operator
        .then(|| CsrPartsBuilder::new(nrows, n_temperature_nodes));
    let mut thermal_stress_operator = selection
        .thermal_stress_operator
        .then(|| CsrPartsBuilder::new(nrows, n_temperature_nodes));
    let mut strain_constant = selection.strain_constant.then(|| vec![0.0; nrows]);
    let mut stress_constant = selection.stress_constant.then(|| vec![0.0; nrows]);
    let mut thermal_strain_constant = selection.thermal_strain_constant.then(|| vec![0.0; nrows]);
    let mut thermal_stress_constant = selection.thermal_stress_constant.then(|| vec![0.0; nrows]);
    let mut row_entries = Vec::with_capacity(DOF_PER_ELEMENT);

    for element_index in element_start..element_end {
        let coords = mesh.element_coords(element_index)?;
        let nodes = mesh.element_nodes(element_index)?;
        let material_id = material_ids[element_index];
        let material = material_table.get(material_id).ok_or_else(|| {
            format!("material_id {material_id} on element {element_index} is out of range")
        })?;
        let thermal_material_base = thermal_material_table
            .map(|table| {
                table.get(material_id).ok_or_else(|| {
                    format!(
                        "thermal material_id {material_id} on element {element_index} is out of range"
                    )
                })
            })
            .transpose()?;
        let material_storage;
        let thermal_storage;
        let (material, thermal_material) = if let Some(angles) = material_orientation_angles {
            material_storage = rotate_material_in_plane(material, angles[element_index]);
            thermal_storage = thermal_material_base
                .map(|thermal| rotate_thermal_material_in_plane(thermal, angles[element_index]));
            (&material_storage, thermal_storage.as_ref())
        } else {
            (material, thermal_material_base)
        };
        let thermal_stress_unit =
            thermal_material.map(|thermal| constitutive_times_strain(material, &thermal.alpha));

        for (q_local, sample) in Family::volume_samples(&coords, quadrature)?
            .into_iter()
            .enumerate()
        {
            let local_element_index = element_index - element_start;
            let row_base = 4 * (local_element_index * nq_per_element + q_local);
            if let Some(points) = points.as_mut() {
                points.push(sample.point);
            }
            let b = if selection.needs_strain_rows() || selection.needs_stress_rows() {
                Some(crate::physics::solenoid_stress::axisym::build_b_matrix::<
                    NODES_PER_ELEMENT,
                    DOF_PER_ELEMENT,
                >(
                    formulation, &sample.n, &sample.grad_phys, sample.point
                )?)
            } else {
                None
            };

            for component in 0..4 {
                let row = row_base + component;
                if selection.needs_strain_rows() {
                    let b = b
                        .as_ref()
                        .expect("B matrix should be built for strain rows");
                    row_entries.clear();
                    for local_node in 0..NODES_PER_ELEMENT {
                        for dof_component in 0..DOF_PER_NODE {
                            let local_dof = DOF_PER_NODE * local_node + dof_component;
                            let full_col = DOF_PER_NODE * nodes[local_node] + dof_component;
                            append_selected_reduced_displacement_entry(
                                &mut row_entries,
                                strain_operator.is_some(),
                                strain_constant.as_deref_mut(),
                                row,
                                full_col,
                                b[component][local_dof],
                                global_to_reduced,
                                fixed_lookup,
                            );
                        }
                    }
                    if let Some(operator) = strain_operator.as_mut() {
                        operator.push_canonical_row(&mut row_entries);
                    }
                }

                if selection.needs_stress_rows() {
                    let b = b
                        .as_ref()
                        .expect("B matrix should be built for stress rows");
                    row_entries.clear();
                    for local_node in 0..NODES_PER_ELEMENT {
                        for dof_component in 0..DOF_PER_NODE {
                            let local_dof = DOF_PER_NODE * local_node + dof_component;
                            let full_col = DOF_PER_NODE * nodes[local_node] + dof_component;
                            let mut value = 0.0;
                            for strain_component in 0..4 {
                                value = value
                                    + material[component][strain_component]
                                        * b[strain_component][local_dof];
                            }
                            append_selected_reduced_displacement_entry(
                                &mut row_entries,
                                stress_operator.is_some(),
                                stress_constant.as_deref_mut(),
                                row,
                                full_col,
                                value,
                                global_to_reduced,
                                fixed_lookup,
                            );
                        }
                    }
                    if let Some(operator) = stress_operator.as_mut() {
                        operator.push_canonical_row(&mut row_entries);
                    }
                }
            }

            if selection.needs_thermal_rows() {
                if let (Some(thermal_material), Some(thermal_stress_unit)) =
                    (thermal_material, thermal_stress_unit.as_ref())
                {
                    let local =
                        thermal_sample_kernel(&sample, thermal_material, thermal_stress_unit);
                    for component in 0..4 {
                        let row = row_base + component;
                        if let Some(constant) = thermal_strain_constant.as_mut() {
                            constant[row] = local.thermal_strain_constant[component];
                        }
                        if let Some(constant) = thermal_stress_constant.as_mut() {
                            constant[row] = local.thermal_stress_constant[component];
                        }

                        if let Some(operator) = thermal_strain_operator.as_mut() {
                            row_entries.clear();
                            for local_temp_node in 0..NODES_PER_ELEMENT {
                                append_temperature_entry(
                                    &mut row_entries,
                                    nodes[local_temp_node],
                                    local.thermal_strain[component][local_temp_node],
                                );
                            }
                            operator.push_canonical_row(&mut row_entries);
                        }

                        if let Some(operator) = thermal_stress_operator.as_mut() {
                            row_entries.clear();
                            for local_temp_node in 0..NODES_PER_ELEMENT {
                                append_temperature_entry(
                                    &mut row_entries,
                                    nodes[local_temp_node],
                                    local.thermal_stress[component][local_temp_node],
                                );
                            }
                            operator.push_canonical_row(&mut row_entries);
                        }
                    }
                } else {
                    for _ in 0..4 {
                        if let Some(operator) = thermal_strain_operator.as_mut() {
                            operator.push_empty_row();
                        }
                        if let Some(operator) = thermal_stress_operator.as_mut() {
                            operator.push_empty_row();
                        }
                    }
                }
            }
        }
    }

    Ok(SelectedReducedQuadratureFieldOperators {
        points,
        strain_operator: strain_operator.map(CsrPartsBuilder::finish),
        stress_operator: stress_operator.map(CsrPartsBuilder::finish),
        thermal_strain_operator: thermal_strain_operator.map(CsrPartsBuilder::finish),
        thermal_stress_operator: thermal_stress_operator.map(CsrPartsBuilder::finish),
        strain_constant,
        stress_constant,
        thermal_strain_constant,
        thermal_stress_constant,
    })
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn quadrature_field_operators_range_for_family<
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    mesh: QuadMeshView2d<'_, f64, NODES_PER_ELEMENT>,
    material_ids: &[usize],
    material_table: &[[[f64; 4]; 4]],
    thermal_material_table: Option<&[ThermalMaterial]>,
    material_orientation_angles: Option<&[f64]>,
    formulation: Structural2dFormulation,
    quadrature: QuadratureRule,
    element_start: usize,
    element_end: usize,
) -> Result<QuadratureFieldOperators, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    let nq_per_element = quadrature.points_per_element();
    let nsamples = (element_end - element_start) * nq_per_element;
    let (element_indices, reference_points) =
        element_major_reference_points_range(element_start, element_end, quadrature);
    let strain_operator = quad_mesh_strain_operator::<
        Family::ReferenceElement,
        NODES_PER_ELEMENT,
        DOF_PER_ELEMENT,
    >(mesh, &element_indices, &reference_points, formulation)?;
    let stress_operator =
        quad_mesh_stress_operator::<Family::ReferenceElement, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            mesh,
            &element_indices,
            &reference_points,
            material_ids,
            material_table,
            material_orientation_angles,
            formulation,
        )?;

    let mut points = Vec::with_capacity(nsamples);
    let mut thermal_strain_rows = Vec::new();
    let mut thermal_strain_cols = Vec::new();
    let mut thermal_strain_vals = Vec::new();
    let mut thermal_stress_rows = Vec::new();
    let mut thermal_stress_cols = Vec::new();
    let mut thermal_stress_vals = Vec::new();
    let mut thermal_strain_constant = vec![0.0; nsamples * 4];
    let mut thermal_stress_constant = vec![0.0; nsamples * 4];

    for element_index in element_start..element_end {
        let coords = mesh.element_coords(element_index)?;
        let nodes = mesh.element_nodes(element_index)?;
        let material_id = material_ids[element_index];
        let material = material_table.get(material_id).ok_or_else(|| {
            format!("material_id {material_id} on element {element_index} is out of range")
        })?;
        let thermal_material_base = thermal_material_table
            .map(|table| {
                table.get(material_id).ok_or_else(|| {
                    format!(
                        "thermal material_id {material_id} on element {element_index} is out of range"
                    )
                })
            })
            .transpose()?;
        let material_storage;
        let thermal_storage;
        let (material, thermal_material) = if let Some(angles) = material_orientation_angles {
            material_storage = rotate_material_in_plane(material, angles[element_index]);
            thermal_storage = thermal_material_base
                .map(|thermal| rotate_thermal_material_in_plane(thermal, angles[element_index]));
            (&material_storage, thermal_storage.as_ref())
        } else {
            (material, thermal_material_base)
        };
        let thermal_stress_unit =
            thermal_material.map(|thermal| constitutive_times_strain(material, &thermal.alpha));

        for (q_local, sample) in Family::volume_samples(&coords, quadrature)?
            .into_iter()
            .enumerate()
        {
            let local_element_index = element_index - element_start;
            let row_base = 4 * (local_element_index * nq_per_element + q_local);
            let global_rows = [row_base, row_base + 1, row_base + 2, row_base + 3];
            points.push(sample.point);
            if let (Some(thermal_material), Some(thermal_stress_unit)) =
                (thermal_material, thermal_stress_unit.as_ref())
            {
                let local = thermal_sample_kernel(&sample, thermal_material, thermal_stress_unit);
                scatter_local_matrix(
                    &mut thermal_strain_rows,
                    &mut thermal_strain_cols,
                    &mut thermal_strain_vals,
                    &global_rows,
                    &nodes,
                    &local.thermal_strain,
                );
                scatter_local_matrix(
                    &mut thermal_stress_rows,
                    &mut thermal_stress_cols,
                    &mut thermal_stress_vals,
                    &global_rows,
                    &nodes,
                    &local.thermal_stress,
                );
                for component in 0..4 {
                    thermal_strain_constant[global_rows[component]] =
                        local.thermal_strain_constant[component];
                    thermal_stress_constant[global_rows[component]] =
                        local.thermal_stress_constant[component];
                }
            }
        }
    }

    Ok(QuadratureFieldOperators {
        points,
        strain_rows: strain_operator.rows,
        strain_cols: strain_operator.cols,
        strain_vals: strain_operator.vals,
        stress_rows: stress_operator.rows,
        stress_cols: stress_operator.cols,
        stress_vals: stress_operator.vals,
        thermal_strain_rows,
        thermal_strain_cols,
        thermal_strain_vals,
        thermal_stress_rows,
        thermal_stress_cols,
        thermal_stress_vals,
        thermal_strain_constant,
        thermal_stress_constant,
        nq_per_element,
        ntemp: thermal_material_table.map_or(0, |_| mesh.num_nodes()),
    })
}

#[cfg(test)]
mod tests {
    use super::quadrature_field_operators_for_family;
    use crate::mesh::{QuadratureRule, sampling};
    use crate::physics::solenoid_stress::axisym::{build_b_matrix, constitutive_times_b};
    use crate::physics::solenoid_stress::convenience::isotropic_axisymmetric_material;
    use crate::physics::solenoid_stress::family::Quad4Family;
    use crate::physics::solenoid_stress::test_utils::single_element_quad4_mesh;
    use crate::physics::solenoid_stress::types::{Structural2dFormulation, dof_per_element};

    /// Apply one triplet operator to a dense vector for direct-reference comparison in tests.
    fn apply_triplets(
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        nrow: usize,
        x: &[f64],
    ) -> Vec<f64> {
        let mut out = vec![0.0; nrow];
        for ((row, col), val) in rows.iter().zip(cols).zip(vals) {
            out[*row] += *val * x[*col];
        }
        out
    }

    #[test]
    /// Check that the sparse recovery operators reproduce direct `B` and `D B` evaluation.
    fn quadrature_field_operators_match_direct_b_and_db_application() {
        let mesh = single_element_quad4_mesh();
        let material_ids = [0usize];
        let material_table = [isotropic_axisymmetric_material(200.0e9, 0.27)];
        let operators =
            quadrature_field_operators_for_family::<Quad4Family, 4, { dof_per_element(4) }>(
                mesh,
                &material_ids,
                &material_table,
                None,
                None,
                Structural2dFormulation::Axisymmetric,
                QuadratureRule::GaussLegendre3,
            )
            .expect("operator assembly should succeed");

        let u = [0.01, -0.02, 0.03, 0.01, 0.02, -0.01, -0.04, 0.02];
        let strain = apply_triplets(
            &operators.strain_rows,
            &operators.strain_cols,
            &operators.strain_vals,
            operators.points.len() * 4,
            &u,
        );
        let stress = apply_triplets(
            &operators.stress_rows,
            &operators.stress_cols,
            &operators.stress_vals,
            operators.points.len() * 4,
            &u,
        );

        let coords = mesh.element_coords(0).expect("element coords");
        let samples = sampling::volume_samples_quad4(&coords, QuadratureRule::GaussLegendre3)
            .expect("samples");
        for (q_local, sample) in samples.into_iter().enumerate() {
            let b = build_b_matrix::<4, 8>(
                Structural2dFormulation::Axisymmetric,
                &sample.n,
                &sample.grad_phys,
                sample.point,
            )
            .expect("B matrix");
            let db = constitutive_times_b(&material_table[0], &b);
            for component in 0..4 {
                let row = 4 * q_local + component;
                let mut eps = 0.0;
                let mut sig = 0.0;
                for dof in 0..8 {
                    eps += b[component][dof] * u[dof];
                    sig += db[component][dof] * u[dof];
                }
                assert!((strain[row] - eps).abs() < 1.0e-12);
                // Stress is recovered from the displacement-gradient field and
                // picks up additional roundoff through the elastic stress-strain
                // law, so it is less sharp than the direct strain check.
                assert!((stress[row] - sig).abs() < 1.0e-3);
            }
        }
    }
}
