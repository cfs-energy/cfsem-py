//! Reduced-model assembly and solve wrapper for the axisymmetric structural FEM.

use faer::Col;
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::Lu;
use faer::sparse::{SparseColMat, SparseRowMat, Triplet};

use crate::mesh::elements::quad2d::{quad4, quad9};
use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::assembly::assemble_stiffness_for_family;
use crate::physics::solenoid_stress::family::{Quad4Family, Quad9Family, QuadElementFamily};
use crate::physics::solenoid_stress::loads::{
    SparseOperator, body_force_operator_for_family, pressure_operator_for_family,
    temperature_operator_for_family, traction_operator_for_family,
};
use crate::physics::solenoid_stress::recovery::quadrature_field_operators_for_family;
use crate::physics::solenoid_stress::types::{
    PressureLoad, Real, ThermalMaterial, TractionLoad, dof_per_element,
};

/// Public element-family selector for the axisymmetric structural solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisymmetricElementType {
    /// Bilinear four-node quadrilateral in the meridian plane.
    Quad4,
    /// Quadratic nine-node quadrilateral in the meridian plane.
    Quad9,
}

impl AxisymmetricElementType {
    /// Return the canonical public string spelling used by the Python wrapper and docs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Quad4 => "quad4",
            Self::Quad9 => "quad9",
        }
    }

    /// Parse the compact numeric element code used by the low-level Python binding.
    pub fn from_code(code: u8) -> Result<Self, String> {
        match code {
            4 => Ok(Self::Quad4),
            9 => Ok(Self::Quad9),
            _ => Err(format!(
                "unsupported axisymmetric FEM element code {code}; use 4 or 9"
            )),
        }
    }
}

/// Borrowed element-connectivity input for model assembly.
///
/// Each variant stores element-node connectivity in the node ordering expected by the
/// corresponding quadrilateral family.
pub enum AxisymmetricElements<'a> {
    Quad4(&'a [[usize; 4]]),
    Quad9(&'a [[usize; 9]]),
}

/// Reduced-space recovery operators and constants stored on the assembled model.
///
/// All sparse operators in this struct act on the reduced displacement vector produced by the
/// constrained solve, except for the thermal operators, which act on the nodal temperature field.
#[derive(Debug, Clone)]
pub struct ReducedRecoveryOperators<F: Real> {
    /// Quadrature-point coordinates `(r, z)` in element-major order.
    ///
    /// Units: `[length]`.
    pub points_rz: Vec<[F; 2]>,
    /// CSR operator mapping reduced displacements `[length]` to quadrature-point strains
    /// `[dimensionless]`.
    ///
    /// Entry units: `[strain / displacement] = [1 / length]`.
    pub strain_operator: SparseRowMat<usize, F>,
    /// CSR operator mapping reduced displacements `[length]` to quadrature-point stresses.
    ///
    /// Entry units: `[stress / displacement] = [pressure / length]`.
    pub stress_operator: SparseRowMat<usize, F>,
    /// CSR operator mapping nodal temperatures `[temperature]` to quadrature-point thermal strain.
    ///
    /// Entry units: `[strain / temperature]`.
    pub thermal_strain_operator: SparseRowMat<usize, F>,
    /// CSR operator mapping nodal temperatures `[temperature]` to quadrature-point thermal stress.
    ///
    /// Entry units: `[stress / temperature]`.
    pub thermal_stress_operator: SparseRowMat<usize, F>,
    /// Constant strain offset induced by nonzero prescribed Dirichlet values.
    ///
    /// Units: `[strain]`.
    pub strain_constant: Vec<F>,
    /// Constant stress offset induced by nonzero prescribed Dirichlet values.
    ///
    /// Units: `[stress]`.
    pub stress_constant: Vec<F>,
    /// Constant thermal-strain offset induced by per-material reference temperature.
    ///
    /// Units: `[strain]`.
    pub thermal_strain_constant: Vec<F>,
    /// Constant thermal-stress offset induced by per-material reference temperature.
    ///
    /// Units: `[stress]`.
    pub thermal_stress_constant: Vec<F>,
    /// Number of quadrature points contributed by each element.
    pub nq_per_element: usize,
    /// Number of nodal temperatures expected by the thermal recovery operators.
    pub n_temperature_nodes: usize,
}

/// Fully assembled reduced axisymmetric structural model.
///
/// The public system stored here is the Dirichlet-reduced system.  `stiffness`, the load
/// operators, and `constant_rhs` all live in reduced displacement space, while the recovery
/// operators map reduced displacements back to quadrature-point strain and stress fields.
#[derive(Debug)]
pub struct AxisymmetricModel<F: Real> {
    /// Reduced structural stiffness matrix in CSC form.
    ///
    /// This matrix maps reduced displacements `[length]` to reduced generalized nodal forces
    /// `[energy / distance]`.
    ///
    /// Entry units: `[generalized force / displacement] = [energy / distance^2]`.
    pub stiffness: SparseColMat<usize, F>,
    /// Reduced RHS operator for per-element body-force density amplitudes.
    ///
    /// Columns are grouped by element as `[b_r, b_z]`.
    /// Entry units: `[volume]`.
    pub body_force_to_rhs: SparseRowMat<usize, F>,
    /// Reduced RHS operator for scalar pressure amplitudes on `pressure_faces`.
    ///
    /// One column per loaded face.
    /// Entry units: `[area]`.
    pub pressure_to_rhs: SparseRowMat<usize, F>,
    /// Reduced RHS operator for vector traction amplitudes on `traction_faces`.
    ///
    /// Columns are grouped by face as `[t_r, t_z]`.
    /// Entry units: `[area]`.
    pub traction_to_rhs: SparseRowMat<usize, F>,
    /// Reduced RHS operator for nodal temperatures.
    ///
    /// Entry units: `[generalized force / temperature] = [energy / (distance * temperature)]`.
    pub temperature_to_rhs: SparseRowMat<usize, F>,
    /// Constant reduced RHS contribution from prescribed displacements and thermal reference state.
    ///
    /// Units: `[energy / distance]`.
    pub constant_rhs: Vec<F>,
    /// Quadrature-point recovery operators and constants associated with this reduced model.
    pub recovery: ReducedRecoveryOperators<F>,
    /// Metadata listing the loaded pressure faces as `[element_index, local_face]`.
    pub pressure_faces: Vec<[usize; 2]>,
    /// Metadata listing the loaded traction faces as `[element_index, local_face]`.
    pub traction_faces: Vec<[usize; 2]>,
    /// Analysis mesh nodes `(r, z)` used by the backend.
    ///
    /// Units: `[length]`.
    pub analysis_nodes: Vec<[F; 2]>,
    /// Flattened analysis connectivity in element-major order.
    pub analysis_elements_flat: Vec<usize>,
    /// Number of nodes per analysis element.
    pub nodes_per_element: usize,
    /// Analysis element family used by the backend.
    pub element_type: AxisymmetricElementType,
    /// Number of displacement DOFs in the unreduced full system.
    pub ndof_full: usize,
    /// Number of displacement DOFs remaining after Dirichlet reduction.
    pub ndof_reduced: usize,
    /// Number of analysis elements.
    pub nelem: usize,
    /// Mapping from reduced displacement index to full-system DOF index.
    pub free_dofs: Vec<usize>,
    /// Full-system DOF indices removed by Dirichlet reduction.
    pub fixed_dofs: Vec<usize>,
    /// Prescribed displacement values for `fixed_dofs`.
    ///
    /// Units: `[length]`.
    pub fixed_values: Vec<F>,
    lu: Option<Lu<usize, F>>,
}

impl<F: Real> AxisymmetricModel<F> {
    /// Build the reduced right-hand side for one specific load state.
    ///
    /// Each optional input vector is interpreted in the column layout of the corresponding stored
    /// operator:
    /// - `body_force`: `[b_r, b_z]` per element with units `[force / volume]`,
    /// - `pressure_values`: one scalar pressure per `pressure_faces` entry with units
    ///   `[force / area]`,
    /// - `traction_values`: `[t_r, t_z]` per `traction_faces` entry with units `[force / area]`,
    /// - `nodal_temperature`: one temperature per analysis node.
    ///
    /// The returned vector has length `ndof_reduced` and units `[energy / distance]`.
    pub fn build_rhs(
        &self,
        body_force: Option<&[F]>,
        pressure_values: Option<&[F]>,
        traction_values: Option<&[F]>,
        nodal_temperature: Option<&[F]>,
    ) -> Result<Vec<F>, String> {
        let mut rhs = self.constant_rhs.clone();
        apply_csr_operator(&self.body_force_to_rhs, body_force, "body_force", &mut rhs)?;
        apply_csr_operator(
            &self.pressure_to_rhs,
            pressure_values,
            "pressure_values",
            &mut rhs,
        )?;
        apply_csr_operator(
            &self.traction_to_rhs,
            traction_values,
            "traction_values",
            &mut rhs,
        )?;
        if self.temperature_to_rhs.ncols() > 0 {
            let nodal_temperature = nodal_temperature.ok_or_else(|| {
                "nodal_temperature is required because this model includes thermal materials"
                    .to_string()
            })?;
            apply_csr_operator(
                &self.temperature_to_rhs,
                Some(nodal_temperature),
                "nodal_temperature",
                &mut rhs,
            )?;
        } else if let Some(nodal_temperature) = nodal_temperature {
            if !nodal_temperature.is_empty() {
                return Err(
                    "nodal_temperature was provided, but this model has no thermal operator"
                        .to_string(),
                );
            }
        }
        Ok(rhs)
    }

    /// Solve the reduced structural system and return the recovered full displacement vector.
    ///
    /// The model caches the sparse LU factorization of `stiffness` on first use, so repeated calls
    /// reuse the same factorization.
    ///
    /// Input units: `rhs` has units `[energy / distance]`.
    /// Output units: displacements `[length]`.
    pub fn solve(&mut self, rhs: &[F]) -> Result<Vec<F>, String> {
        if rhs.len() != self.ndof_reduced {
            return Err(format!(
                "rhs has length {}, but reduced system has {} rows",
                rhs.len(),
                self.ndof_reduced
            ));
        }
        if self.ndof_reduced == 0 {
            return Ok(self.recover_full(&[]));
        }
        if self.lu.is_none() {
            self.lu = Some(
                self.stiffness
                    .sp_lu()
                    .map_err(|err| format!("failed to factorize reduced stiffness: {err:?}"))?,
            );
        }
        let lu = self.lu.as_ref().expect("lu cache should be initialized");
        let mut reduced_solution = Col::<F>::zeros(self.ndof_reduced);
        for (index, value) in rhs.iter().copied().enumerate() {
            reduced_solution[index] = value;
        }
        lu.solve_in_place(reduced_solution.as_mut());
        let reduced_solution = (0..self.ndof_reduced)
            .map(|index| reduced_solution[index])
            .collect::<Vec<_>>();
        Ok(self.recover_full(&reduced_solution))
    }

    /// Reinsert prescribed Dirichlet values into a reduced solution vector.
    ///
    /// Input and output units: displacement `[length]`.
    pub fn recover_full(&self, reduced_solution: &[F]) -> Vec<F> {
        assert!(
            reduced_solution.len() == self.ndof_reduced,
            "reduced_solution has length {}, but reduced system has {} rows",
            reduced_solution.len(),
            self.ndof_reduced
        );
        let mut full = vec![F::zero(); self.ndof_full];
        for (&dof, &value) in self.fixed_dofs.iter().zip(&self.fixed_values) {
            full[dof] = value;
        }
        for (&dof, &value) in self.free_dofs.iter().zip(reduced_solution) {
            full[dof] = value;
        }
        full
    }
}

pub(crate) struct AxisymmetricModelBuilder<'a, F: Real> {
    nodes_rz: &'a [[F; 2]],
    elements: AxisymmetricElements<'a>,
    material_ids: &'a [usize],
    material_table: &'a [[[F; 4]; 4]],
    pressure_faces: Vec<PressureLoad<F>>,
    traction_faces: Vec<TractionLoad<F>>,
    thermal_material_table: Option<&'a [ThermalMaterial<F>]>,
    prescribed: Vec<(usize, F)>,
    quadrature: QuadratureRule,
}

impl<'a, F: Real> AxisymmetricModelBuilder<'a, F> {
    pub fn new(
        nodes_rz: &'a [[F; 2]],
        elements: AxisymmetricElements<'a>,
        material_ids: &'a [usize],
        material_table: &'a [[[F; 4]; 4]],
    ) -> Self {
        Self {
            nodes_rz,
            elements,
            material_ids,
            material_table,
            pressure_faces: Vec::new(),
            traction_faces: Vec::new(),
            thermal_material_table: None,
            prescribed: Vec::new(),
            quadrature: QuadratureRule::GaussLegendre3,
        }
    }

    pub fn quadrature(mut self, quadrature: QuadratureRule) -> Self {
        self.quadrature = quadrature;
        self
    }

    pub fn pressure_faces(mut self, pressure_faces: &'a [PressureLoad<F>]) -> Self {
        self.pressure_faces = pressure_faces.to_vec();
        self
    }

    pub fn traction_faces(mut self, traction_faces: &'a [TractionLoad<F>]) -> Self {
        self.traction_faces = traction_faces.to_vec();
        self
    }

    pub fn thermal_material_table(
        mut self,
        thermal_material_table: Option<&'a [ThermalMaterial<F>]>,
    ) -> Self {
        self.thermal_material_table = thermal_material_table;
        self
    }

    pub fn prescribed_dirichlet(mut self, prescribed: &'a [(usize, F)]) -> Self {
        self.prescribed = prescribed.to_vec();
        self
    }

    pub fn build(self) -> Result<AxisymmetricModel<F>, String> {
        match self.elements {
            AxisymmetricElements::Quad4(elements) => build_model_for_family::<
                F,
                Quad4Family,
                { quad4::NODES_PER_ELEMENT },
                { dof_per_element(quad4::NODES_PER_ELEMENT) },
            >(
                self.nodes_rz,
                elements,
                self.material_ids,
                self.material_table,
                &self.pressure_faces,
                &self.traction_faces,
                self.thermal_material_table,
                &self.prescribed,
                self.quadrature,
            ),
            AxisymmetricElements::Quad9(elements) => build_model_for_family::<
                F,
                Quad9Family,
                { quad9::NODES_PER_ELEMENT },
                { dof_per_element(quad9::NODES_PER_ELEMENT) },
            >(
                self.nodes_rz,
                elements,
                self.material_ids,
                self.material_table,
                &self.pressure_faces,
                &self.traction_faces,
                self.thermal_material_table,
                &self.prescribed,
                self.quadrature,
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Assemble the reduced axisymmetric structural model and all associated operators.
///
/// This is the single public Rust entry point for the axisymmetric FEM backend.  It accepts the
/// analysis mesh, material data, optional load topology, optional thermal material data, and
/// prescribed Dirichlet displacement values, then returns a reduced model containing:
/// - the reduced stiffness matrix,
/// - reduced RHS operators for all supported load types,
/// - cached metadata describing the analysis mesh and load faces, and
/// - reduced recovery operators for quadrature-point postprocessing.
pub fn assemble_axisymmetric<'a, F: Real>(
    nodes_rz: &'a [[F; 2]],
    elements: AxisymmetricElements<'a>,
    material_ids: &'a [usize],
    material_table: &'a [[[F; 4]; 4]],
    pressure_faces: &'a [PressureLoad<F>],
    traction_faces: &'a [TractionLoad<F>],
    thermal_material_table: Option<&'a [ThermalMaterial<F>]>,
    prescribed: &'a [(usize, F)],
    quadrature: QuadratureRule,
) -> Result<AxisymmetricModel<F>, String> {
    AxisymmetricModelBuilder::new(nodes_rz, elements, material_ids, material_table)
        .pressure_faces(pressure_faces)
        .traction_faces(traction_faces)
        .thermal_material_table(thermal_material_table)
        .prescribed_dirichlet(prescribed)
        .quadrature(quadrature)
        .build()
}

type ReducedLayout<F> = (Vec<usize>, Vec<usize>, Vec<F>, Vec<usize>, Vec<Option<F>>);

#[allow(clippy::too_many_arguments)]
fn build_model_for_family<
    F: Real,
    Family,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    nodes_rz: &[[F; 2]],
    elements: &[[usize; NODES_PER_ELEMENT]],
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    pressure_faces: &[PressureLoad<F>],
    traction_faces: &[TractionLoad<F>],
    thermal_material_table: Option<&[ThermalMaterial<F>]>,
    prescribed: &[(usize, F)],
    quadrature: QuadratureRule,
) -> Result<AxisymmetricModel<F>, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    let mesh = QuadMeshView2d { nodes_rz, elements };
    let ndof_full = nodes_rz.len() * 2;
    let nelem = elements.len();
    let (free_dofs, fixed_dofs, fixed_values, global_to_reduced, fixed_lookup) =
        reduce_layout(ndof_full, prescribed)?;
    let ndof_reduced = free_dofs.len();

    let stiffness_full = assemble_stiffness_for_family::<
        F,
        Family,
        NODES_PER_ELEMENT,
        DOF_PER_ELEMENT,
    >(mesh, material_ids, material_table, quadrature)?;
    let mut constant_rhs = vec![F::zero(); ndof_reduced];
    let stiffness_reduced = reduce_square_triplets(
        &stiffness_full.rows,
        &stiffness_full.cols,
        &stiffness_full.vals,
        &global_to_reduced,
        &fixed_lookup,
        &mut constant_rhs,
    );
    let stiffness = csc_from_triplets(ndof_reduced, ndof_reduced, stiffness_reduced)?;
    let reduce_operator =
        |operator| reduce_row_operator_to_csr(operator, &global_to_reduced, ndof_reduced);

    let (temperature_to_rhs, thermal_reference_rhs, n_temperature_nodes) =
        if let Some(thermal_material_table) = thermal_material_table {
            let thermal_full =
                temperature_operator_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                    mesh,
                    material_ids,
                    material_table,
                    thermal_material_table,
                    quadrature,
                )?;
            let reduced_reference_rhs = free_dofs
                .iter()
                .map(|&dof| thermal_full.reference_rhs[dof])
                .collect::<Vec<_>>();
            (
                reduce_operator(thermal_full.temperature_to_rhs)?,
                reduced_reference_rhs,
                nodes_rz.len(),
            )
        } else {
            (
                csr_from_parts(ndof_reduced, 0, Vec::new(), Vec::new(), Vec::new())?,
                vec![F::zero(); ndof_reduced],
                0,
            )
        };
    for (dst, src) in constant_rhs.iter_mut().zip(&thermal_reference_rhs) {
        *dst = *dst + *src;
    }

    let body_force_to_rhs = reduce_operator(body_force_operator_for_family::<
        F,
        Family,
        NODES_PER_ELEMENT,
        DOF_PER_ELEMENT,
    >(mesh, quadrature)?)?;
    let pressure_to_rhs = reduce_operator(pressure_operator_for_family::<
        F,
        Family,
        NODES_PER_ELEMENT,
        DOF_PER_ELEMENT,
    >(mesh, pressure_faces, quadrature)?)?;
    let traction_to_rhs = reduce_operator(traction_operator_for_family::<
        F,
        Family,
        NODES_PER_ELEMENT,
        DOF_PER_ELEMENT,
    >(mesh, traction_faces, quadrature)?)?;

    let recovery_full =
        quadrature_field_operators_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            mesh,
            material_ids,
            material_table,
            thermal_material_table,
            quadrature,
        )?;
    let nq_row_count = recovery_full.points_rz.len() * 4;
    let (strain_operator, strain_constant) = reduce_column_operator(
        recovery_full.strain_rows,
        recovery_full.strain_cols,
        recovery_full.strain_vals,
        nq_row_count,
        ndof_reduced,
        &global_to_reduced,
        &fixed_lookup,
    )?;
    let (stress_operator, stress_constant) = reduce_column_operator(
        recovery_full.stress_rows,
        recovery_full.stress_cols,
        recovery_full.stress_vals,
        nq_row_count,
        ndof_reduced,
        &global_to_reduced,
        &fixed_lookup,
    )?;
    let thermal_strain_operator = csr_from_parts(
        recovery_full.points_rz.len() * 4,
        recovery_full.ntemp,
        recovery_full.thermal_strain_rows,
        recovery_full.thermal_strain_cols,
        recovery_full.thermal_strain_vals,
    )?;
    let thermal_stress_operator = csr_from_parts(
        recovery_full.points_rz.len() * 4,
        recovery_full.ntemp,
        recovery_full.thermal_stress_rows,
        recovery_full.thermal_stress_cols,
        recovery_full.thermal_stress_vals,
    )?;

    Ok(AxisymmetricModel {
        stiffness,
        body_force_to_rhs,
        pressure_to_rhs,
        traction_to_rhs,
        temperature_to_rhs,
        constant_rhs,
        recovery: ReducedRecoveryOperators {
            points_rz: recovery_full.points_rz,
            strain_operator,
            stress_operator,
            thermal_strain_operator,
            thermal_stress_operator,
            strain_constant,
            stress_constant,
            thermal_strain_constant: recovery_full.thermal_strain_constant,
            thermal_stress_constant: recovery_full.thermal_stress_constant,
            nq_per_element: recovery_full.nq_per_element,
            n_temperature_nodes,
        },
        pressure_faces: pressure_faces
            .iter()
            .map(|face| [face.element, usize::from(face.local_face)])
            .collect(),
        traction_faces: traction_faces
            .iter()
            .map(|face| [face.element, usize::from(face.local_face)])
            .collect(),
        analysis_nodes: nodes_rz.to_vec(),
        analysis_elements_flat: Family::flatten_elements(elements),
        nodes_per_element: NODES_PER_ELEMENT,
        element_type: Family::element_type(),
        ndof_full,
        ndof_reduced,
        nelem,
        free_dofs,
        fixed_dofs,
        fixed_values,
        lu: None,
    })
}

fn reduce_layout<F: Real>(
    ndof_full: usize,
    prescribed: &[(usize, F)],
) -> Result<ReducedLayout<F>, String> {
    let mut prescribed_sorted = prescribed.to_vec();
    prescribed_sorted.sort_by_key(|&(dof, _)| dof);
    for window in prescribed_sorted.windows(2) {
        if window[0].0 == window[1].0 {
            return Err(format!(
                "prescribed DOF {} is specified more than once",
                window[0].0
            ));
        }
    }
    let mut fixed_dofs = Vec::with_capacity(prescribed_sorted.len());
    let mut fixed_values = Vec::with_capacity(prescribed_sorted.len());
    let mut fixed_lookup = vec![None; ndof_full];
    for &(dof, value) in &prescribed_sorted {
        if dof >= ndof_full {
            return Err(format!(
                "prescribed DOF {dof} is out of bounds for a system with {ndof_full} DOFs"
            ));
        }
        fixed_dofs.push(dof);
        fixed_values.push(value);
        fixed_lookup[dof] = Some(value);
    }
    let mut is_fixed = vec![false; ndof_full];
    for &dof in &fixed_dofs {
        is_fixed[dof] = true;
    }
    let mut free_dofs = Vec::with_capacity(ndof_full - fixed_dofs.len());
    let mut global_to_reduced = vec![usize::MAX; ndof_full];
    for dof in 0..ndof_full {
        if !is_fixed[dof] {
            global_to_reduced[dof] = free_dofs.len();
            free_dofs.push(dof);
        }
    }
    Ok((
        free_dofs,
        fixed_dofs,
        fixed_values,
        global_to_reduced,
        fixed_lookup,
    ))
}

fn reduce_square_triplets<F: Real>(
    rows: &[usize],
    cols: &[usize],
    vals: &[F],
    global_to_reduced: &[usize],
    fixed_lookup: &[Option<F>],
    constant_rhs: &mut [F],
) -> Vec<Triplet<usize, usize, F>> {
    let mut triplets = Vec::with_capacity(vals.len());
    for ((&row, &col), &value) in rows.iter().zip(cols).zip(vals) {
        let reduced_row = global_to_reduced[row];
        let reduced_col = global_to_reduced[col];
        if reduced_row != usize::MAX && reduced_col != usize::MAX {
            triplets.push(Triplet::new(reduced_row, reduced_col, value));
        } else if reduced_row != usize::MAX
            && let Some(fixed_value) = fixed_lookup[col]
        {
            constant_rhs[reduced_row] = constant_rhs[reduced_row] - value * fixed_value;
        }
    }
    triplets
}

fn reduce_row_operator<F: Real>(
    operator: SparseOperator<F>,
    global_to_reduced: &[usize],
    nrow_reduced: usize,
) -> SparseOperator<F> {
    let mut rows = Vec::with_capacity(operator.vals.len());
    let mut cols = Vec::with_capacity(operator.vals.len());
    let mut vals = Vec::with_capacity(operator.vals.len());
    for ((row, col), value) in operator
        .rows
        .into_iter()
        .zip(operator.cols.into_iter())
        .zip(operator.vals.into_iter())
    {
        let reduced_row = global_to_reduced[row];
        if reduced_row != usize::MAX {
            rows.push(reduced_row);
            cols.push(col);
            vals.push(value);
        }
    }
    SparseOperator {
        rows,
        cols,
        vals,
        nrow: nrow_reduced,
        ncol: operator.ncol,
    }
}

fn reduce_row_operator_to_csr<F: Real>(
    operator: SparseOperator<F>,
    global_to_reduced: &[usize],
    nrow_reduced: usize,
) -> Result<SparseRowMat<usize, F>, String> {
    let operator = reduce_row_operator(operator, global_to_reduced, nrow_reduced);
    csr_from_parts(
        operator.nrow,
        operator.ncol,
        operator.rows,
        operator.cols,
        operator.vals,
    )
}

fn reduce_column_operator<F: Real>(
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<F>,
    nrow: usize,
    ncol: usize,
    global_to_reduced: &[usize],
    fixed_lookup: &[Option<F>],
) -> Result<(SparseRowMat<usize, F>, Vec<F>), String> {
    let mut constant = vec![F::zero(); nrow];
    let mut reduced_rows = Vec::with_capacity(vals.len());
    let mut reduced_cols = Vec::with_capacity(vals.len());
    let mut reduced_vals = Vec::with_capacity(vals.len());
    for ((row, col), value) in rows.into_iter().zip(cols.into_iter()).zip(vals.into_iter()) {
        let reduced_col = global_to_reduced[col];
        if reduced_col != usize::MAX {
            reduced_rows.push(row);
            reduced_cols.push(reduced_col);
            reduced_vals.push(value);
        } else if let Some(fixed_value) = fixed_lookup[col] {
            constant[row] = constant[row] + value * fixed_value;
        }
    }
    Ok((
        csr_from_parts(nrow, ncol, reduced_rows, reduced_cols, reduced_vals)?,
        constant,
    ))
}

fn csr_from_parts<F: Real>(
    nrow: usize,
    ncol: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<F>,
) -> Result<SparseRowMat<usize, F>, String> {
    let triplets = rows
        .into_iter()
        .zip(cols)
        .zip(vals)
        .map(|((row, col), val)| Triplet::new(row, col, val))
        .collect::<Vec<_>>();
    SparseRowMat::try_new_from_triplets(nrow, ncol, &triplets)
        .map_err(|err| format!("failed to build CSR operator: {err:?}"))
}

fn csc_from_triplets<F: Real>(
    nrow: usize,
    ncol: usize,
    triplets: Vec<Triplet<usize, usize, F>>,
) -> Result<SparseColMat<usize, F>, String> {
    SparseColMat::try_new_from_triplets(nrow, ncol, &triplets)
        .map_err(|err| format!("failed to build CSC operator: {err:?}"))
}

fn apply_csr_operator<F: Real>(
    operator: &SparseRowMat<usize, F>,
    input: Option<&[F]>,
    name: &str,
    output: &mut [F],
) -> Result<(), String> {
    if operator.ncols() == 0 {
        if let Some(input) = input {
            if !input.is_empty() {
                return Err(format!(
                    "{name} was provided, but this model has no corresponding operator"
                ));
            }
        }
        return Ok(());
    }
    let input = input.unwrap_or(&[]);
    if input.len() != operator.ncols() {
        return Err(format!(
            "{name} has length {}, but operator expects {} values",
            input.len(),
            operator.ncols()
        ));
    }
    for row in 0..operator.nrows() {
        let start = operator.row_ptr()[row];
        let end = operator.row_ptr()[row + 1];
        let mut sum = F::zero();
        for index in start..end {
            sum = sum + operator.val()[index] * input[operator.col_idx()[index]];
        }
        output[row] = output[row] + sum;
    }
    Ok(())
}
