//! Reduced-model assembly and solve wrapper for the axisymmetric structural FEM.

use faer::Col;
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::Lu;
use faer::sparse::{SparseColMat, SparseRowMat, Triplet};

use crate::mesh::{MeshView, QuadratureRule};
use crate::physics::solenoid_stress::assembly::{
    assemble_stiffness_quad4, assemble_stiffness_quad9,
};
use crate::physics::solenoid_stress::loads::{
    SparseOperator, ThermalLoadOperator, body_force_operator_quad4, body_force_operator_quad9,
    pressure_operator_quad4, pressure_operator_quad9, temperature_operator_quad4,
    temperature_operator_quad9, traction_operator_quad4, traction_operator_quad9,
};
use crate::physics::solenoid_stress::recovery::{
    QuadratureFieldOperators, quadrature_field_operators_quad4, quadrature_field_operators_quad9,
};
use crate::physics::solenoid_stress::types::{PressureLoad, Real, ThermalMaterial, TractionLoad};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisymmetricElementType {
    Quad4,
    Quad9,
}

impl AxisymmetricElementType {
    pub const fn nodes_per_element(self) -> usize {
        match self {
            Self::Quad4 => 4,
            Self::Quad9 => 9,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Quad4 => "quad4",
            Self::Quad9 => "quad9",
        }
    }
}

pub enum AxisymmetricElements<'a> {
    Quad4(&'a [[usize; 4]]),
    Quad9(&'a [[usize; 9]]),
}

impl<'a> AxisymmetricElements<'a> {
    pub const fn element_type(&self) -> AxisymmetricElementType {
        match self {
            Self::Quad4(_) => AxisymmetricElementType::Quad4,
            Self::Quad9(_) => AxisymmetricElementType::Quad9,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Quad4(elements) => elements.len(),
            Self::Quad9(elements) => elements.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone)]
pub struct ReducedRecoveryOperators<F: Real> {
    pub points_rz: Vec<[F; 2]>,
    pub strain_operator: SparseRowMat<usize, F>,
    pub stress_operator: SparseRowMat<usize, F>,
    pub thermal_strain_operator: SparseRowMat<usize, F>,
    pub thermal_stress_operator: SparseRowMat<usize, F>,
    pub strain_constant: Vec<F>,
    pub stress_constant: Vec<F>,
    pub thermal_strain_constant: Vec<F>,
    pub thermal_stress_constant: Vec<F>,
    pub nq_per_element: usize,
    pub n_temperature_nodes: usize,
}

#[derive(Debug)]
pub struct AxisymmetricModel<F: Real> {
    stiffness: SparseColMat<usize, F>,
    body_force_to_rhs: SparseRowMat<usize, F>,
    pressure_to_rhs: SparseRowMat<usize, F>,
    traction_to_rhs: SparseRowMat<usize, F>,
    temperature_to_rhs: SparseRowMat<usize, F>,
    constant_rhs: Vec<F>,
    recovery: ReducedRecoveryOperators<F>,
    pressure_faces: Vec<[usize; 2]>,
    traction_faces: Vec<[usize; 2]>,
    analysis_nodes: Vec<[F; 2]>,
    analysis_elements_flat: Vec<usize>,
    nodes_per_element: usize,
    element_type: AxisymmetricElementType,
    ndof_full: usize,
    ndof_reduced: usize,
    nelem: usize,
    free_dofs: Vec<usize>,
    fixed_dofs: Vec<usize>,
    fixed_values: Vec<F>,
    lu: Option<Lu<usize, F>>,
}

impl<F: Real> AxisymmetricModel<F> {
    pub fn stiffness(&self) -> &SparseColMat<usize, F> {
        &self.stiffness
    }

    pub fn body_force_to_rhs(&self) -> &SparseRowMat<usize, F> {
        &self.body_force_to_rhs
    }

    pub fn pressure_to_rhs(&self) -> &SparseRowMat<usize, F> {
        &self.pressure_to_rhs
    }

    pub fn traction_to_rhs(&self) -> &SparseRowMat<usize, F> {
        &self.traction_to_rhs
    }

    pub fn temperature_to_rhs(&self) -> &SparseRowMat<usize, F> {
        &self.temperature_to_rhs
    }

    pub fn recovery(&self) -> &ReducedRecoveryOperators<F> {
        &self.recovery
    }

    pub fn constant_rhs(&self) -> &[F] {
        &self.constant_rhs
    }

    pub fn pressure_faces(&self) -> &[[usize; 2]] {
        &self.pressure_faces
    }

    pub fn traction_faces(&self) -> &[[usize; 2]] {
        &self.traction_faces
    }

    pub fn analysis_nodes(&self) -> &[[F; 2]] {
        &self.analysis_nodes
    }

    pub fn analysis_elements_flat(&self) -> &[usize] {
        &self.analysis_elements_flat
    }

    pub const fn nodes_per_element(&self) -> usize {
        self.nodes_per_element
    }

    pub const fn element_type(&self) -> AxisymmetricElementType {
        self.element_type
    }

    pub const fn ndof_full(&self) -> usize {
        self.ndof_full
    }

    pub const fn ndof_reduced(&self) -> usize {
        self.ndof_reduced
    }

    pub const fn nelem(&self) -> usize {
        self.nelem
    }

    pub fn free_dofs(&self) -> &[usize] {
        &self.free_dofs
    }

    pub fn fixed_dofs(&self) -> &[usize] {
        &self.fixed_dofs
    }

    pub fn fixed_values(&self) -> &[F] {
        &self.fixed_values
    }

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

pub struct AxisymmetricModelBuilder<'a, F: Real> {
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
            AxisymmetricElements::Quad4(elements) => build_model_for_mesh(
                self.nodes_rz,
                elements,
                AxisymmetricElementType::Quad4,
                self.material_ids,
                self.material_table,
                &self.pressure_faces,
                &self.traction_faces,
                self.thermal_material_table,
                &self.prescribed,
                self.quadrature,
                assemble_stiffness_quad4::<F>,
                body_force_operator_quad4::<F>,
                pressure_operator_quad4::<F>,
                traction_operator_quad4::<F>,
                temperature_operator_quad4::<F>,
                quadrature_field_operators_quad4::<F>,
            ),
            AxisymmetricElements::Quad9(elements) => build_model_for_mesh(
                self.nodes_rz,
                elements,
                AxisymmetricElementType::Quad9,
                self.material_ids,
                self.material_table,
                &self.pressure_faces,
                &self.traction_faces,
                self.thermal_material_table,
                &self.prescribed,
                self.quadrature,
                assemble_stiffness_quad9::<F>,
                body_force_operator_quad9::<F>,
                pressure_operator_quad9::<F>,
                traction_operator_quad9::<F>,
                temperature_operator_quad9::<F>,
                quadrature_field_operators_quad9::<F>,
            ),
        }
    }
}

type AssembleStiffnessFn<F, const NODES: usize> =
    fn(
        MeshView<'_, F, NODES>,
        &[usize],
        &[[[F; 4]; 4]],
        QuadratureRule,
    ) -> Result<crate::physics::solenoid_stress::types::StiffnessTriplets<F>, String>;

type BodyForceOperatorFn<F, const NODES: usize> =
    fn(MeshView<'_, F, NODES>, QuadratureRule) -> Result<SparseOperator<F>, String>;

type FaceOperatorFn<F, const NODES: usize, Load> =
    fn(MeshView<'_, F, NODES>, &[Load], QuadratureRule) -> Result<SparseOperator<F>, String>;

type TemperatureOperatorFn<F, const NODES: usize> = fn(
    MeshView<'_, F, NODES>,
    &[usize],
    &[[[F; 4]; 4]],
    &[ThermalMaterial<F>],
    QuadratureRule,
) -> Result<ThermalLoadOperator<F>, String>;

type RecoveryOperatorFn<F, const NODES: usize> = fn(
    MeshView<'_, F, NODES>,
    &[usize],
    &[[[F; 4]; 4]],
    Option<&[ThermalMaterial<F>]>,
    QuadratureRule,
) -> Result<QuadratureFieldOperators<F>, String>;

#[allow(clippy::too_many_arguments)]
fn build_model_for_mesh<F: Real, const NODES_PER_ELEMENT: usize>(
    nodes_rz: &[[F; 2]],
    elements: &[[usize; NODES_PER_ELEMENT]],
    element_type: AxisymmetricElementType,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    pressure_faces: &[PressureLoad<F>],
    traction_faces: &[TractionLoad<F>],
    thermal_material_table: Option<&[ThermalMaterial<F>]>,
    prescribed: &[(usize, F)],
    quadrature: QuadratureRule,
    assemble_stiffness_fn: AssembleStiffnessFn<F, NODES_PER_ELEMENT>,
    body_force_operator_fn: BodyForceOperatorFn<F, NODES_PER_ELEMENT>,
    pressure_operator_fn: FaceOperatorFn<F, NODES_PER_ELEMENT, PressureLoad<F>>,
    traction_operator_fn: FaceOperatorFn<F, NODES_PER_ELEMENT, TractionLoad<F>>,
    temperature_operator_fn: TemperatureOperatorFn<F, NODES_PER_ELEMENT>,
    recovery_operator_fn: RecoveryOperatorFn<F, NODES_PER_ELEMENT>,
) -> Result<AxisymmetricModel<F>, String> {
    let mesh = MeshView { nodes_rz, elements };
    let ndof_full = nodes_rz.len() * 2;
    let nelem = elements.len();
    let (free_dofs, fixed_dofs, fixed_values, global_to_reduced) =
        reduce_layout(ndof_full, prescribed)?;
    let ndof_reduced = free_dofs.len();

    let stiffness_full = assemble_stiffness_fn(mesh, material_ids, material_table, quadrature)?;
    let mut constant_rhs = vec![F::zero(); ndof_reduced];
    let stiffness_reduced = reduce_square_triplets(
        &stiffness_full.rows,
        &stiffness_full.cols,
        &stiffness_full.vals,
        &global_to_reduced,
        &fixed_values,
        &mut constant_rhs,
    );
    let stiffness = csc_from_triplets(ndof_reduced, ndof_reduced, stiffness_reduced)?;

    let body_force = reduce_row_operator(
        body_force_operator_fn(mesh, quadrature)?,
        &global_to_reduced,
        ndof_reduced,
    );
    let pressure = reduce_row_operator(
        pressure_operator_fn(mesh, pressure_faces, quadrature)?,
        &global_to_reduced,
        ndof_reduced,
    );
    let traction = reduce_row_operator(
        traction_operator_fn(mesh, traction_faces, quadrature)?,
        &global_to_reduced,
        ndof_reduced,
    );

    let (temperature_to_rhs, thermal_reference_rhs, n_temperature_nodes) =
        if let Some(thermal_material_table) = thermal_material_table {
            let thermal_full = temperature_operator_fn(
                mesh,
                material_ids,
                material_table,
                thermal_material_table,
                quadrature,
            )?;
            let reduced_temperature = reduce_row_operator(
                thermal_full.temperature_to_rhs,
                &global_to_reduced,
                ndof_reduced,
            );
            let reduced_reference_rhs = free_dofs
                .iter()
                .map(|&dof| thermal_full.reference_rhs[dof])
                .collect::<Vec<_>>();
            (
                csr_from_triplets(
                    reduced_temperature.nrow,
                    reduced_temperature.ncol,
                    reduced_temperature.rows,
                    reduced_temperature.cols,
                    reduced_temperature.vals,
                )?,
                reduced_reference_rhs,
                nodes_rz.len(),
            )
        } else {
            (
                csr_from_triplets(ndof_reduced, 0, Vec::new(), Vec::new(), Vec::new())?,
                vec![F::zero(); ndof_reduced],
                0,
            )
        };
    for (dst, src) in constant_rhs.iter_mut().zip(&thermal_reference_rhs) {
        *dst = *dst + *src;
    }

    let body_force_to_rhs = csr_from_triplets(
        body_force.nrow,
        body_force.ncol,
        body_force.rows,
        body_force.cols,
        body_force.vals,
    )?;
    let pressure_to_rhs = csr_from_triplets(
        pressure.nrow,
        pressure.ncol,
        pressure.rows,
        pressure.cols,
        pressure.vals,
    )?;
    let traction_to_rhs = csr_from_triplets(
        traction.nrow,
        traction.ncol,
        traction.rows,
        traction.cols,
        traction.vals,
    )?;

    let recovery_full = recovery_operator_fn(
        mesh,
        material_ids,
        material_table,
        thermal_material_table,
        quadrature,
    )?;
    let (strain_rows, strain_cols, strain_vals, strain_constant) = reduce_column_operator(
        recovery_full.strain_rows,
        recovery_full.strain_cols,
        recovery_full.strain_vals,
        &global_to_reduced,
        &fixed_values,
        vec![F::zero(); recovery_full.points_rz.len() * 4],
    );
    let (stress_rows, stress_cols, stress_vals, stress_constant) = reduce_column_operator(
        recovery_full.stress_rows,
        recovery_full.stress_cols,
        recovery_full.stress_vals,
        &global_to_reduced,
        &fixed_values,
        vec![F::zero(); recovery_full.points_rz.len() * 4],
    );
    let strain_operator = csr_from_triplets(
        recovery_full.points_rz.len() * 4,
        ndof_reduced,
        strain_rows,
        strain_cols,
        strain_vals,
    )?;
    let stress_operator = csr_from_triplets(
        recovery_full.points_rz.len() * 4,
        ndof_reduced,
        stress_rows,
        stress_cols,
        stress_vals,
    )?;
    let thermal_strain_operator = csr_from_triplets(
        recovery_full.points_rz.len() * 4,
        recovery_full.ntemp,
        recovery_full.thermal_strain_rows,
        recovery_full.thermal_strain_cols,
        recovery_full.thermal_strain_vals,
    )?;
    let thermal_stress_operator = csr_from_triplets(
        recovery_full.points_rz.len() * 4,
        recovery_full.ntemp,
        recovery_full.thermal_stress_rows,
        recovery_full.thermal_stress_cols,
        recovery_full.thermal_stress_vals,
    )?;

    let mut analysis_elements_flat = Vec::with_capacity(nelem * NODES_PER_ELEMENT);
    for conn in elements {
        analysis_elements_flat.extend_from_slice(conn);
    }

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
        analysis_elements_flat,
        nodes_per_element: NODES_PER_ELEMENT,
        element_type,
        ndof_full,
        ndof_reduced,
        nelem,
        free_dofs,
        fixed_dofs,
        fixed_values: fixed_values.into_iter().map(|(_, value)| value).collect(),
        lu: None,
    })
}

fn reduce_layout<F: Real>(
    ndof_full: usize,
    prescribed: &[(usize, F)],
) -> Result<(Vec<usize>, Vec<usize>, Vec<(usize, F)>, Vec<usize>), String> {
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
    for &(dof, value) in &prescribed_sorted {
        if dof >= ndof_full {
            return Err(format!(
                "prescribed DOF {dof} is out of bounds for a system with {ndof_full} DOFs"
            ));
        }
        fixed_dofs.push(dof);
        fixed_values.push((dof, value));
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
    Ok((free_dofs, fixed_dofs, fixed_values, global_to_reduced))
}

fn reduce_square_triplets<F: Real>(
    rows: &[usize],
    cols: &[usize],
    vals: &[F],
    global_to_reduced: &[usize],
    fixed_values: &[(usize, F)],
    constant_rhs: &mut [F],
) -> Vec<Triplet<usize, usize, F>> {
    let fixed_map = fixed_values
        .iter()
        .copied()
        .collect::<std::collections::BTreeMap<usize, F>>();
    let mut triplets = Vec::with_capacity(vals.len());
    for ((&row, &col), &value) in rows.iter().zip(cols).zip(vals) {
        let reduced_row = global_to_reduced[row];
        let reduced_col = global_to_reduced[col];
        if reduced_row != usize::MAX && reduced_col != usize::MAX {
            triplets.push(Triplet::new(reduced_row, reduced_col, value));
        } else if reduced_row != usize::MAX {
            if let Some(&fixed_value) = fixed_map.get(&col) {
                constant_rhs[reduced_row] = constant_rhs[reduced_row] - value * fixed_value;
            }
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

fn reduce_column_operator<F: Real>(
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<F>,
    global_to_reduced: &[usize],
    fixed_values: &[(usize, F)],
    mut constant: Vec<F>,
) -> (Vec<usize>, Vec<usize>, Vec<F>, Vec<F>) {
    let fixed_map = fixed_values
        .iter()
        .copied()
        .collect::<std::collections::BTreeMap<usize, F>>();
    let mut reduced_rows = Vec::with_capacity(vals.len());
    let mut reduced_cols = Vec::with_capacity(vals.len());
    let mut reduced_vals = Vec::with_capacity(vals.len());
    for ((row, col), value) in rows.into_iter().zip(cols.into_iter()).zip(vals.into_iter()) {
        let reduced_col = global_to_reduced[col];
        if reduced_col != usize::MAX {
            reduced_rows.push(row);
            reduced_cols.push(reduced_col);
            reduced_vals.push(value);
        } else if let Some(&fixed_value) = fixed_map.get(&col) {
            constant[row] = constant[row] + value * fixed_value;
        }
    }
    (reduced_rows, reduced_cols, reduced_vals, constant)
}

fn csr_from_triplets<F: Real>(
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
