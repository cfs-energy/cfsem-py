//! Reduced-model assembly and solve wrapper for the 2D structural FEM.

use deimos_numerics::sparse::{
    BiCGSTAB, BiCGSTABSolveError, DiagonalPrecond, Equilibration, EquilibrationParams, Precond,
    SparseMatVec,
};
use faer::Col;
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::Lu;
use faer::sparse::{SparseColMat, SparseColMatRef, SparseRowMat, Triplet};

use crate::mesh::elements::quad2d::{quad4, quad9};
use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::assembly::{
    assemble_stiffness_for_family, assemble_stiffness_for_family_par,
};
use crate::physics::solenoid_stress::convenience::{
    QuadratureFieldSamples, Structural2dElementMeasures, Structural2dElementQuadrature,
};
use crate::physics::solenoid_stress::family::{Quad4Family, Quad9Family, QuadElementFamily};
use crate::physics::solenoid_stress::loads::{
    SparseOperator, body_force_operator_for_family, pressure_operator_for_family,
    temperature_operator_for_family, traction_operator_for_family,
};
use crate::physics::solenoid_stress::recovery::quadrature_field_operators_for_family;
use crate::physics::solenoid_stress::types::{
    PressureLoad, Real, Structural2dFormulation, ThermalMaterial, TractionLoad, dof_per_element,
};

/// Public element-family selector for the 2D structural solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Structural2dElementType {
    /// Bilinear four-node quadrilateral in the analysis plane.
    Quad4,
    /// Quadratic nine-node quadrilateral in the analysis plane.
    Quad9,
}

/// Linear-solver method used for one reduced 2D structural FEM solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Structural2dSolveMethod<F: Real> {
    /// Sparse LU direct solve with lazy factorization caching.
    Direct,
    /// BiCGSTAB Krylov solve with optional diagonal preconditioning and equilibration.
    Bicgstab(BicgstabSolveOptions<F>),
}

/// Compact solver name used by diagnostics and Python bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Structural2dSolveMethodName {
    /// Sparse LU direct solve.
    Direct,
    /// BiCGSTAB Krylov solve.
    Bicgstab,
}

impl Structural2dSolveMethodName {
    /// Return the compact code used by the Python binding.
    pub const fn code(self) -> u8 {
        match self {
            Self::Direct => 0,
            Self::Bicgstab => 1,
        }
    }
}

/// Options for the BiCGSTAB reduced-system solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BicgstabSolveOptions<F: Real> {
    /// Absolute residual tolerance in RHS units. If `None`, the solve uses a RHS-scaled default.
    pub tolerance: Option<F>,
    /// Maximum number of BiCGSTAB iterations. If `None`, a size-based default is used.
    pub max_iterations: Option<usize>,
    /// Optional preconditioner used by BiCGSTAB.
    pub preconditioner: BicgstabPreconditioner,
    /// Optional matrix equilibration applied before BiCGSTAB.
    pub equilibration: BicgstabEquilibration<F>,
    /// Initial-guess policy for the reduced displacement vector.
    pub initial_guess: BicgstabInitialGuess,
}

impl<F: Real> Default for BicgstabSolveOptions<F> {
    fn default() -> Self {
        Self {
            tolerance: None,
            max_iterations: None,
            preconditioner: BicgstabPreconditioner::Diagonal,
            equilibration: BicgstabEquilibration::Ruiz(EquilibrationSolveOptions::default()),
            initial_guess: BicgstabInitialGuess::Zero,
        }
    }
}

/// Preconditioner selection for BiCGSTAB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BicgstabPreconditioner {
    /// No preconditioner.
    None,
    /// Diagonal/Jacobi-style preconditioner.
    Diagonal,
}

/// Equilibration selection for BiCGSTAB.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BicgstabEquilibration<F: Real> {
    /// Solve the original reduced system directly.
    None,
    /// Apply Ruiz-style row/column equilibration before solving.
    Ruiz(EquilibrationSolveOptions<F>),
}

/// Parameters controlling Ruiz-style sparse matrix equilibration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EquilibrationSolveOptions<F: Real> {
    /// Maximum number of equilibration iterations.
    pub max_iterations: usize,
    /// Dimensionless row/column balance tolerance.
    pub tolerance: F,
    /// Lower clamp applied to measured row/column norms.
    pub norm_floor: F,
    /// Upper clamp applied to measured row/column norms.
    pub norm_ceil: F,
}

impl<F: Real> Default for EquilibrationSolveOptions<F> {
    fn default() -> Self {
        let params = EquilibrationParams::<F>::default();
        Self {
            max_iterations: params.max_iters,
            tolerance: params.tol,
            norm_floor: params.norm_floor,
            norm_ceil: params.norm_ceil,
        }
    }
}

impl<F: Real> From<EquilibrationSolveOptions<F>> for EquilibrationParams<F> {
    fn from(options: EquilibrationSolveOptions<F>) -> Self {
        Self {
            max_iters: options.max_iterations,
            tol: options.tolerance,
            norm_floor: options.norm_floor,
            norm_ceil: options.norm_ceil,
        }
    }
}

/// Initial-guess policy for iterative solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BicgstabInitialGuess {
    /// Start from a zero reduced displacement vector.
    Zero,
    /// Reuse the previous converged BiCGSTAB reduced solution when available.
    Previous,
}

/// Full displacement solution plus solve diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct Structural2dSolveOutput<F: Real> {
    /// Full displacement vector with shape `(ndof_full,)`.
    pub displacement: Vec<F>,
    /// Solver diagnostics for the reduced solve.
    pub diagnostics: Structural2dSolveDiagnostics<F>,
}

/// Diagnostics from one reduced structural solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Structural2dSolveDiagnostics<F: Real> {
    /// Solver method used for this solve.
    pub method: Structural2dSolveMethodName,
    /// Whether the selected solver converged.
    pub converged: bool,
    /// Number of BiCGSTAB iterations, or zero for direct solves.
    pub iterations: usize,
    /// Latest residual norm in RHS units, or zero for direct solves.
    pub residual_norm: F,
    /// Whether row/column equilibration was applied.
    pub equilibrated: bool,
}

impl Structural2dElementType {
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
                "unsupported structural 2D FEM element code {code}; use 4 or 9"
            )),
        }
    }
}

/// Borrowed element-connectivity input for model assembly.
///
/// Each variant stores element-node connectivity in the node ordering expected by the
/// corresponding quadrilateral family. Element corner nodes must be ordered
/// counter-clockwise in the 2D analysis plane.
pub enum Structural2dElements<'a> {
    /// Four-node bilinear quadrilateral connectivity in local corner order.
    Quad4(&'a [[usize; 4]]),
    /// Nine-node quadratic quadrilateral connectivity in local corner, midside, center order.
    Quad9(&'a [[usize; 9]]),
}

/// Reduced-space recovery operators and constants stored on the assembled model.
///
/// All sparse operators in this struct act on the reduced displacement vector produced by the
/// constrained solve, except for the thermal operators, which act on the nodal temperature field.
/// `points` has flattened shape `(nelem * nq_per_element, 2)`. Each operator and constant
/// vector acts on or stores flattened quadrature-point blocks with row ordering
/// `[rr, zz, tt, rz]`, so those arrays have flattened shape `(4 * nelem * nq_per_element,)`.
#[derive(Debug, Clone)]
pub struct ReducedRecoveryOperators<F: Real> {
    /// Quadrature-point coordinates `(r, z)` in element-major order.
    ///
    /// Flattened shape: `(nelem * nq_per_element, 2)`.
    /// Units: `[length]`.
    pub points: Vec<[F; 2]>,
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
    /// Flattened shape: `(4 * nelem * nq_per_element,)`.
    /// Units: `[strain]`.
    pub strain_constant: Vec<F>,
    /// Constant stress offset induced by nonzero prescribed Dirichlet values.
    ///
    /// Flattened shape: `(4 * nelem * nq_per_element,)`.
    /// Units: `[stress]`.
    pub stress_constant: Vec<F>,
    /// Constant thermal-strain offset induced by per-material reference temperature.
    ///
    /// Flattened shape: `(4 * nelem * nq_per_element,)`.
    /// Units: `[strain]`.
    pub thermal_strain_constant: Vec<F>,
    /// Constant thermal-stress offset induced by per-material reference temperature.
    ///
    /// Flattened shape: `(4 * nelem * nq_per_element,)`.
    /// Units: `[stress]`.
    pub thermal_stress_constant: Vec<F>,
    /// Number of quadrature points contributed by each element.
    pub nq_per_element: usize,
    /// Number of nodal temperatures expected by the thermal recovery operators.
    pub n_temperature_nodes: usize,
}

/// Fully assembled reduced 2D structural model.
///
/// The public system stored here is the Dirichlet-reduced system.  `stiffness`, the load
/// operators, and `constant_rhs` all live in reduced displacement space, while the recovery
/// operators map reduced displacements back to quadrature-point strain and stress fields.
/// `analysis_nodes` has shape `(n_analysis_nodes, 2)`, `analysis_elements_flat` has shape
/// `(nelem * nodes_per_element,)`, `pressure_faces` has shape `(n_pressure_faces, 2)`, and
/// `traction_faces` has shape `(n_traction_faces, 2)`.
#[derive(Debug)]
pub struct Structural2dModel<F: Real> {
    /// Reduced structural stiffness matrix in CSC form.
    ///
    /// This matrix maps reduced displacements `[length]` to reduced generalized nodal forces
    /// `[energy / distance]`.
    ///
    /// Entry units: `[generalized force / displacement] = [energy / distance^2]`.
    pub stiffness: SparseColMat<usize, F>,
    /// Reduced RHS operator for per-element body-force density amplitudes.
    ///
    /// Shape: `(ndof_reduced, 2 * nelem)`. Columns are grouped by element as `[b_r, b_z]`.
    /// Entry units: `[volume]`.
    pub body_force_to_rhs: SparseRowMat<usize, F>,
    /// Reduced RHS operator for scalar pressure amplitudes on `pressure_faces`.
    ///
    /// Shape: `(ndof_reduced, n_pressure_faces)`. One column per loaded face.
    /// Entry units: `[area]`.
    pub pressure_to_rhs: SparseRowMat<usize, F>,
    /// Reduced RHS operator for vector traction amplitudes on `traction_faces`.
    ///
    /// Shape: `(ndof_reduced, 2 * n_traction_faces)`. Columns are grouped by face as `[t_r, t_z]`.
    /// Entry units: `[area]`.
    pub traction_to_rhs: SparseRowMat<usize, F>,
    /// Reduced RHS operator for nodal temperatures.
    ///
    /// Shape: `(ndof_reduced, n_temperature_nodes)`.
    /// Entry units: `[generalized force / temperature] = [energy / (distance * temperature)]`.
    pub temperature_to_rhs: SparseRowMat<usize, F>,
    /// Constant reduced RHS contribution from prescribed displacements and thermal reference state.
    ///
    /// Shape: `(ndof_reduced,)`.
    /// Units: `[energy / distance]`.
    pub constant_rhs: Vec<F>,
    /// Quadrature-point recovery operators and constants associated with this reduced model.
    pub recovery: ReducedRecoveryOperators<F>,
    /// Metadata listing the loaded pressure faces as `[element_index, local_face]`.
    ///
    /// Shape: `(n_pressure_faces, 2)`.
    pub pressure_faces: Vec<[usize; 2]>,
    /// Metadata listing the loaded traction faces as `[element_index, local_face]`.
    ///
    /// Shape: `(n_traction_faces, 2)`.
    pub traction_faces: Vec<[usize; 2]>,
    /// Analysis mesh nodes used by the backend.
    ///
    /// Shape: `(n_analysis_nodes, 2)`.
    /// Units: `[length]`.
    pub analysis_nodes: Vec<[F; 2]>,
    /// Flattened analysis connectivity in element-major order.
    ///
    /// Shape: `(nelem * nodes_per_element,)`.
    pub analysis_elements_flat: Vec<usize>,
    /// Number of nodes per analysis element.
    pub nodes_per_element: usize,
    /// Analysis element family used by the backend.
    pub element_type: Structural2dElementType,
    /// Volume and face quadrature rule used to assemble the stored operators.
    pub quadrature: QuadratureRule,
    /// Structural 2D formulation used by the backend.
    pub formulation: Structural2dFormulation<F>,
    /// Number of displacement DOFs in the unreduced full system.
    pub ndof_full: usize,
    /// Number of displacement DOFs remaining after Dirichlet reduction.
    pub ndof_reduced: usize,
    /// Number of analysis elements.
    pub nelem: usize,
    /// Mapping from reduced displacement index to full-system DOF index.
    ///
    /// Shape: `(ndof_reduced,)`.
    pub free_dofs: Vec<usize>,
    /// Full-system DOF indices removed by Dirichlet reduction.
    ///
    /// Shape: `(n_fixed,)`.
    pub fixed_dofs: Vec<usize>,
    /// Prescribed displacement values for `fixed_dofs`.
    ///
    /// Shape: `(n_fixed,)`.
    /// Units: `[length]`.
    pub fixed_values: Vec<F>,
    lu: Option<Lu<usize, F>>,
    diagonal_preconditioner: Option<DiagonalPrecond<F>>,
    equilibration: Option<Equilibration<F>>,
    equilibration_options: Option<EquilibrationSolveOptions<F>>,
    equilibrated_stiffness: Option<SparseColMat<usize, F>>,
    equilibrated_diagonal_preconditioner: Option<DiagonalPrecond<F>>,
    previous_bicgstab_solution: Option<Vec<F>>,
}

impl<F: Real> Structural2dModel<F> {
    /// Build one reduced structural right-hand side.
    ///
    /// Args:
    ///     body_force: Optional body-force amplitudes with shape `(2 * nelem,)`, grouped as
    ///         `[b_r, b_z]` per element. Units are `[force / volume]`.
    ///     pressure_values: Optional pressure amplitudes with shape `(n_pressure_faces,)`.
    ///         Units are `[force / area]`. Positive values act in the inward normal direction.
    ///     traction_values: Optional traction amplitudes with shape `(2 * n_traction_faces,)`,
    ///         grouped as `[t_r, t_z]` per traction face. Units are `[force / area]`.
    ///     nodal_temperature: Optional nodal temperatures with shape `(n_temperature_nodes,)`.
    ///         Units are `[temperature]`. Required only when the model includes thermal materials.
    ///
    /// Returns:
    ///     Reduced right-hand side with shape `(ndof_reduced,)` and units
    ///     `[generalized force] = [energy / distance]`.
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

    /// Solve the reduced structural system and recover the full displacement vector.
    ///
    /// The model caches the sparse LU factorization of `stiffness` on first use, so repeated
    /// calls reuse the same factorization.
    ///
    /// Args:
    ///     rhs: Reduced right-hand side with shape `(ndof_reduced,)` and units
    ///         `[generalized force] = [energy / distance]`.
    ///
    /// Returns:
    ///     Full displacement vector with shape `(ndof_full,)` and component ordering
    ///     `[u_r0, u_z0, u_r1, u_z1, ...]`. Units are `[length]`.
    pub fn solve(&mut self, rhs: &[F]) -> Result<Vec<F>, String> {
        let reduced_solution = self.solve_direct_reduced(rhs)?;
        Ok(self.recover_full(&reduced_solution))
    }

    /// Solve the reduced structural system with an explicit solver method.
    ///
    /// Args:
    ///     rhs: Reduced right-hand side with shape `(ndof_reduced,)` and units
    ///         `[generalized force] = [energy / distance]`.
    ///     method: Solver method and method-specific options.
    ///
    /// Returns:
    ///     Full displacement vector and solve diagnostics.
    pub fn solve_with_method(
        &mut self,
        rhs: &[F],
        method: Structural2dSolveMethod<F>,
    ) -> Result<Structural2dSolveOutput<F>, String> {
        let (reduced_solution, diagnostics) = match method {
            Structural2dSolveMethod::Direct => {
                (self.solve_direct_reduced(rhs)?, direct_diagnostics())
            }
            Structural2dSolveMethod::Bicgstab(options) => {
                self.solve_bicgstab_reduced(rhs, options)?
            }
        };
        Ok(Structural2dSolveOutput {
            displacement: self.recover_full(&reduced_solution),
            diagnostics,
        })
    }

    /// Solve the reduced structural system with the cached sparse LU factorization.
    fn solve_direct_reduced(&mut self, rhs: &[F]) -> Result<Vec<F>, String> {
        if rhs.len() != self.ndof_reduced {
            return Err(format!(
                "rhs has length {}, but reduced system has {} rows",
                rhs.len(),
                self.ndof_reduced
            ));
        }
        if self.ndof_reduced == 0 {
            return Ok(Vec::new());
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
        Ok(reduced_solution)
    }

    /// Solve the reduced structural system with BiCGSTAB.
    fn solve_bicgstab_reduced(
        &mut self,
        rhs: &[F],
        options: BicgstabSolveOptions<F>,
    ) -> Result<(Vec<F>, Structural2dSolveDiagnostics<F>), String> {
        validate_bicgstab_options(options)?;
        if rhs.len() != self.ndof_reduced {
            return Err(format!(
                "rhs has length {}, but reduced system has {} rows",
                rhs.len(),
                self.ndof_reduced
            ));
        }
        if self.ndof_reduced == 0 {
            return Ok((
                Vec::new(),
                Structural2dSolveDiagnostics {
                    method: Structural2dSolveMethodName::Bicgstab,
                    converged: true,
                    iterations: 0,
                    residual_norm: F::zero(),
                    equilibrated: !matches!(options.equilibration, BicgstabEquilibration::None),
                },
            ));
        }

        let tolerance = resolve_bicgstab_tolerance(rhs, options.tolerance);
        let max_iterations = options
            .max_iterations
            .unwrap_or_else(|| self.ndof_reduced.saturating_mul(2).max(100));
        let mut rhs_work = rhs.to_vec();
        let mut initial_guess = self.bicgstab_initial_guess(options.initial_guess);

        match options.equilibration {
            BicgstabEquilibration::None => {
                if options.preconditioner == BicgstabPreconditioner::Diagonal {
                    self.ensure_diagonal_preconditioner()?;
                }
                let (reduced_solution, diagnostics) = match options.preconditioner {
                    BicgstabPreconditioner::None => solve_bicgstab_no_precond(
                        self.stiffness.as_ref(),
                        &initial_guess,
                        &rhs_work,
                        tolerance,
                        max_iterations,
                        false,
                    )?,
                    BicgstabPreconditioner::Diagonal => solve_bicgstab_with_precond(
                        self.stiffness.as_ref(),
                        self.diagonal_preconditioner
                            .as_ref()
                            .expect("diagonal preconditioner should be initialized")
                            .clone(),
                        &initial_guess,
                        &rhs_work,
                        tolerance,
                        max_iterations,
                        false,
                    )?,
                };
                self.previous_bicgstab_solution = Some(reduced_solution.clone());
                Ok((reduced_solution, diagnostics))
            }
            BicgstabEquilibration::Ruiz(equilibration_options) => {
                self.ensure_equilibration(equilibration_options)?;
                if options.preconditioner == BicgstabPreconditioner::Diagonal {
                    self.ensure_equilibrated_diagonal_preconditioner()?;
                }
                let equilibration = self
                    .equilibration
                    .as_ref()
                    .expect("equilibration should be initialized");
                equilibration.scale_rhs_in_place(&mut rhs_work);
                equilibration.scale_initial_guess_in_place(&mut initial_guess);
                let (mut reduced_solution, diagnostics) = match options.preconditioner {
                    BicgstabPreconditioner::None => solve_bicgstab_no_precond(
                        self.equilibrated_stiffness
                            .as_ref()
                            .expect("equilibrated stiffness should be initialized")
                            .as_ref(),
                        &initial_guess,
                        &rhs_work,
                        tolerance,
                        max_iterations,
                        true,
                    )?,
                    BicgstabPreconditioner::Diagonal => solve_bicgstab_with_precond(
                        self.equilibrated_stiffness
                            .as_ref()
                            .expect("equilibrated stiffness should be initialized")
                            .as_ref(),
                        self.equilibrated_diagonal_preconditioner
                            .as_ref()
                            .expect("equilibrated diagonal preconditioner should be initialized")
                            .clone(),
                        &initial_guess,
                        &rhs_work,
                        tolerance,
                        max_iterations,
                        true,
                    )?,
                };
                equilibration.unscale_solution_in_place(&mut reduced_solution);
                self.previous_bicgstab_solution = Some(reduced_solution.clone());
                Ok((reduced_solution, diagnostics))
            }
        }
    }

    /// Return the reduced-space initial guess requested by the iterative solve options.
    fn bicgstab_initial_guess(&self, policy: BicgstabInitialGuess) -> Vec<F> {
        match policy {
            BicgstabInitialGuess::Zero => vec![F::zero(); self.ndof_reduced],
            BicgstabInitialGuess::Previous => self
                .previous_bicgstab_solution
                .as_ref()
                .filter(|solution| solution.len() == self.ndof_reduced)
                .cloned()
                .unwrap_or_else(|| vec![F::zero(); self.ndof_reduced]),
        }
    }

    /// Build and cache the diagonal preconditioner for the original reduced stiffness.
    fn ensure_diagonal_preconditioner(&mut self) -> Result<(), String> {
        if self.diagonal_preconditioner.is_none() {
            self.diagonal_preconditioner = Some(
                DiagonalPrecond::try_from(self.stiffness.as_ref())
                    .map_err(|err| format!("failed to build diagonal preconditioner: {err:?}"))?,
            );
        }
        Ok(())
    }

    /// Build and cache equilibration scales and the scaled reduced stiffness.
    fn ensure_equilibration(
        &mut self,
        options: EquilibrationSolveOptions<F>,
    ) -> Result<(), String> {
        if self.equilibration_options != Some(options) {
            self.equilibration = None;
            self.equilibrated_stiffness = None;
            self.equilibrated_diagonal_preconditioner = None;
            self.equilibration_options = Some(options);
        }
        if self.equilibration.is_none() || self.equilibrated_stiffness.is_none() {
            let equilibration = Equilibration::<F>::compute_from_csc(
                self.stiffness.as_ref(),
                EquilibrationParams::from(options),
            )
            .map_err(|err| format!("failed to equilibrate reduced stiffness: {err:?}"))?;
            let mut equilibrated_stiffness = self.stiffness.clone();
            equilibration.scale_csc_matrix_in_place(&mut equilibrated_stiffness);
            self.equilibration = Some(equilibration);
            self.equilibrated_stiffness = Some(equilibrated_stiffness);
            self.equilibrated_diagonal_preconditioner = None;
        }
        Ok(())
    }

    /// Build and cache the diagonal preconditioner for the equilibrated reduced stiffness.
    fn ensure_equilibrated_diagonal_preconditioner(&mut self) -> Result<(), String> {
        if self.equilibrated_diagonal_preconditioner.is_none() {
            let stiffness = self
                .equilibrated_stiffness
                .as_ref()
                .expect("equilibrated stiffness should be initialized");
            self.equilibrated_diagonal_preconditioner = Some(
                DiagonalPrecond::try_from(stiffness.as_ref()).map_err(|err| {
                    format!("failed to build equilibrated diagonal preconditioner: {err:?}")
                })?,
            );
        }
        Ok(())
    }

    /// Reinsert prescribed Dirichlet values into a reduced displacement vector.
    ///
    /// Args:
    ///     reduced_solution: Reduced displacement vector with shape `(ndof_reduced,)`.
    ///         Units are `[length]`.
    ///
    /// Returns:
    ///     Full displacement vector with shape `(ndof_full,)` and component ordering
    ///     `[u_r0, u_z0, u_r1, u_z1, ...]`. Units are `[length]`.
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

    /// Recompute the physical quadrature points and mapped weights for the stored analysis mesh.
    ///
    /// Returns:
    ///     Element-major quadrature data with:
    ///     - `points` length `nelem * nq_per_element`, each entry `(r, z)` with units `[length]`
    ///     - `weights_area` length `nelem * nq_per_element` with units `[area]`
    ///     - `weights_volume` length `nelem * nq_per_element` with units `[volume]`
    ///     - `nq_per_element` giving the number of consecutive quadrature entries per element
    pub fn element_quadrature(&self) -> Result<Structural2dElementQuadrature<F>, String> {
        match self.element_type {
            Structural2dElementType::Quad4 => {
                element_quadrature_for_family::<F, Quad4Family, { quad4::NODES_PER_ELEMENT }>(
                    &self.analysis_nodes,
                    &self.analysis_elements_flat,
                    self.nelem,
                    self.formulation,
                    self.quadrature,
                )
            }
            Structural2dElementType::Quad9 => {
                element_quadrature_for_family::<F, Quad9Family, { quad9::NODES_PER_ELEMENT }>(
                    &self.analysis_nodes,
                    &self.analysis_elements_flat,
                    self.nelem,
                    self.formulation,
                    self.quadrature,
                )
            }
        }
    }

    /// Return per-element analysis-plane area and represented volume from the model quadrature data.
    ///
    /// Returns:
    ///     Per-element measures with:
    ///     - `areas` shape `(nelem,)` and units `[area]`
    ///     - `volumes` shape `(nelem,)` and units `[volume]`
    pub fn element_measures(&self) -> Result<Structural2dElementMeasures<F>, String> {
        let quadrature = self.element_quadrature()?;
        let mut areas = vec![F::zero(); self.nelem];
        let mut volumes = vec![F::zero(); self.nelem];
        for element in 0..self.nelem {
            let start = element * quadrature.nq_per_element;
            let end = start + quadrature.nq_per_element;
            for &weight in &quadrature.weights_area[start..end] {
                areas[element] = areas[element] + weight;
            }
            for &weight in &quadrature.weights_volume[start..end] {
                volumes[element] = volumes[element] + weight;
            }
        }
        Ok(Structural2dElementMeasures { areas, volumes })
    }

    /// Recover quadrature-point strain and stress fields.
    ///
    /// Args:
    ///     displacements_full: Full displacement vector with shape `(ndof_full,)` and component
    ///         ordering `[u_r0, u_z0, u_r1, u_z1, ...]`. Units are `[length]`.
    ///     nodal_temperature: Optional nodal temperatures with shape `(n_temperature_nodes,)`.
    ///         Units are `[temperature]`. Required only when the model includes thermal materials.
    ///
    /// Returns:
    ///     Recovered quadrature fields where:
    ///     - `points` has length `nelem * nq_per_element` and units `[length]`
    ///     - `strain`, `thermal_strain`, and `elastic_strain` each have length
    ///       `nelem * nq_per_element`, with component order `[rr, zz, tt, rz]` and units `[strain]`
    ///     - `stress` has length `nelem * nq_per_element`, with component order
    ///       `[rr, zz, tt, rz]` and units `[stress]`
    ///     - `nq_per_element` gives the number of consecutive samples per element
    pub fn evaluate_quadrature(
        &self,
        displacements_full: &[F],
        nodal_temperature: Option<&[F]>,
    ) -> Result<QuadratureFieldSamples<F>, String> {
        if displacements_full.len() != self.ndof_full {
            return Err(format!(
                "displacements_full has length {}, but full system has {} DOFs",
                displacements_full.len(),
                self.ndof_full
            ));
        }

        let reduced = self
            .free_dofs
            .iter()
            .map(|&dof| displacements_full[dof])
            .collect::<Vec<_>>();
        let temperature = normalize_temperature(
            nodal_temperature,
            self.recovery.n_temperature_nodes,
            "nodal_temperature",
        )?;

        let strain = add_constant(
            csr_matvec(&self.recovery.strain_operator, &reduced),
            &self.recovery.strain_constant,
        );
        let stress_from_displacement = add_constant(
            csr_matvec(&self.recovery.stress_operator, &reduced),
            &self.recovery.stress_constant,
        );
        let thermal_strain = add_constant(
            csr_matvec(&self.recovery.thermal_strain_operator, temperature),
            &self.recovery.thermal_strain_constant,
        );
        let thermal_stress = add_constant(
            csr_matvec(&self.recovery.thermal_stress_operator, temperature),
            &self.recovery.thermal_stress_constant,
        );
        let stress = subtract_vectors(&stress_from_displacement, &thermal_stress)?;
        let strain = pack_rank4_field(strain)?;
        let thermal_strain = pack_rank4_field(thermal_strain)?;
        let stress = pack_rank4_field(stress)?;
        let elastic_strain = strain
            .iter()
            .zip(&thermal_strain)
            .map(|(total, thermal)| {
                [
                    total[0] - thermal[0],
                    total[1] - thermal[1],
                    total[2] - thermal[2],
                    total[3] - thermal[3],
                ]
            })
            .collect();

        Ok(QuadratureFieldSamples {
            points: self.recovery.points.clone(),
            strain,
            thermal_strain,
            elastic_strain,
            stress,
            nq_per_element: self.recovery.nq_per_element,
        })
    }
}

/// Diagnostics for the sparse LU path.
fn direct_diagnostics<F: Real>() -> Structural2dSolveDiagnostics<F> {
    Structural2dSolveDiagnostics {
        method: Structural2dSolveMethodName::Direct,
        converged: true,
        iterations: 0,
        residual_norm: F::zero(),
        equilibrated: false,
    }
}

/// Validate BiCGSTAB and equilibration options before allocating solver scratch space.
fn validate_bicgstab_options<F: Real>(options: BicgstabSolveOptions<F>) -> Result<(), String> {
    if let Some(tolerance) = options.tolerance
        && (!tolerance.is_finite() || tolerance <= F::zero())
    {
        return Err(format!(
            "BiCGSTAB tolerance must be finite and positive; got {tolerance:?}"
        ));
    }
    if let Some(max_iterations) = options.max_iterations
        && max_iterations == 0
    {
        return Err("BiCGSTAB max_iterations must be at least 1".to_string());
    }
    if let BicgstabEquilibration::Ruiz(equilibration) = options.equilibration {
        if equilibration.max_iterations == 0 {
            return Err("equilibration max_iterations must be at least 1".to_string());
        }
        if !equilibration.tolerance.is_finite() || equilibration.tolerance < F::zero() {
            return Err(format!(
                "equilibration tolerance must be finite and nonnegative; got {:?}",
                equilibration.tolerance
            ));
        }
        if !equilibration.norm_floor.is_finite()
            || equilibration.norm_floor <= F::zero()
            || !equilibration.norm_ceil.is_finite()
            || equilibration.norm_ceil <= equilibration.norm_floor
        {
            return Err(format!(
                "equilibration norm clamps must satisfy 0 < floor < ceil; got floor={:?}, ceil={:?}",
                equilibration.norm_floor, equilibration.norm_ceil
            ));
        }
    }
    Ok(())
}

/// Resolve a RHS-scaled absolute residual tolerance when the caller does not provide one.
fn resolve_bicgstab_tolerance<F: Real>(rhs: &[F], tolerance: Option<F>) -> F {
    if let Some(tolerance) = tolerance {
        return tolerance;
    }
    let mut norm_sq = F::zero();
    for &value in rhs {
        norm_sq = value.mul_add(value, norm_sq);
    }
    let rhs_norm = norm_sq.sqrt();
    let scale = if rhs_norm > F::zero() {
        rhs_norm
    } else {
        F::one()
    };
    scale * F::epsilon().sqrt()
}

/// Solve a reduced system with unpreconditioned BiCGSTAB.
fn solve_bicgstab_no_precond<F: Real>(
    matrix: SparseColMatRef<'_, usize, F>,
    initial_guess: &[F],
    rhs: &[F],
    tolerance: F,
    max_iterations: usize,
    equilibrated: bool,
) -> Result<(Vec<F>, Structural2dSolveDiagnostics<F>), String> {
    match BiCGSTAB::solve(matrix, initial_guess, rhs, tolerance, max_iterations) {
        Ok(solver) => Ok(bicgstab_success(solver, equilibrated)),
        Err(BiCGSTABSolveError::InvalidInput(err)) => Err(format!("invalid BiCGSTAB input: {err}")),
        Err(BiCGSTABSolveError::NoConvergence(solver)) => Err(format!(
            "BiCGSTAB did not converge within {} iterations; residual norm is {:?}",
            solver.iteration_count(),
            solver.err()
        )),
    }
}

/// Solve a reduced system with diagonally preconditioned BiCGSTAB.
fn solve_bicgstab_with_precond<F: Real>(
    matrix: SparseColMatRef<'_, usize, F>,
    preconditioner: DiagonalPrecond<F>,
    initial_guess: &[F],
    rhs: &[F],
    tolerance: F,
    max_iterations: usize,
    equilibrated: bool,
) -> Result<(Vec<F>, Structural2dSolveDiagnostics<F>), String> {
    match BiCGSTAB::solve_with_precond(
        matrix,
        preconditioner,
        initial_guess,
        rhs,
        tolerance,
        max_iterations,
    ) {
        Ok(solver) => Ok(bicgstab_success(solver, equilibrated)),
        Err(BiCGSTABSolveError::InvalidInput(err)) => Err(format!("invalid BiCGSTAB input: {err}")),
        Err(BiCGSTABSolveError::NoConvergence(solver)) => Err(format!(
            "BiCGSTAB did not converge within {} iterations; residual norm is {:?}",
            solver.iteration_count(),
            solver.err()
        )),
    }
}

/// Convert a converged BiCGSTAB solver state into a reduced solution and diagnostics.
fn bicgstab_success<F: Real, A, P>(
    solver: BiCGSTAB<F, A, P>,
    equilibrated: bool,
) -> (Vec<F>, Structural2dSolveDiagnostics<F>)
where
    A: SparseMatVec<F>,
    P: Precond<F>,
{
    let diagnostics = Structural2dSolveDiagnostics {
        method: Structural2dSolveMethodName::Bicgstab,
        converged: true,
        iterations: solver.iteration_count(),
        residual_norm: solver.err(),
        equilibrated,
    };
    let solution = solver.x().iter().copied().collect::<Vec<_>>();
    (solution, diagnostics)
}

#[allow(clippy::too_many_arguments)]
/// Assemble the reduced 2D structural model and all associated operators.
pub fn assemble_structural_2d<F: Real>(
    nodes_rz: &[[F; 2]],
    elements: Structural2dElements<'_>,
    material_ids: &[usize],
    material_table: &[[[F; 4]; 4]],
    pressure_faces: &[PressureLoad],
    traction_faces: &[TractionLoad],
    thermal_material_table: Option<&[ThermalMaterial<F>]>,
    material_orientation_angles: Option<&[F]>,
    prescribed: &[(usize, F)],
    formulation: Structural2dFormulation<F>,
    quadrature: QuadratureRule,
    par: bool,
) -> Result<Structural2dModel<F>, String> {
    match elements {
        Structural2dElements::Quad4(elements) => build_model_for_family::<
            F,
            Quad4Family,
            { quad4::NODES_PER_ELEMENT },
            { dof_per_element(quad4::NODES_PER_ELEMENT) },
        >(
            nodes_rz,
            elements,
            material_ids,
            material_table,
            pressure_faces,
            traction_faces,
            thermal_material_table,
            material_orientation_angles,
            prescribed,
            formulation,
            quadrature,
            par,
        ),
        Structural2dElements::Quad9(elements) => build_model_for_family::<
            F,
            Quad9Family,
            { quad9::NODES_PER_ELEMENT },
            { dof_per_element(quad9::NODES_PER_ELEMENT) },
        >(
            nodes_rz,
            elements,
            material_ids,
            material_table,
            pressure_faces,
            traction_faces,
            thermal_material_table,
            material_orientation_angles,
            prescribed,
            formulation,
            quadrature,
            par,
        ),
    }
}

/// Internal bundle returned by `reduce_layout`.
///
/// The tuple stores, in order:
/// - the reduced-to-full free DOF mapping,
/// - fixed DOF indices,
/// - fixed DOF values,
/// - the full-to-reduced lookup table,
/// - and a dense lookup of fixed values by full DOF index.
type ReducedLayout<F> = (Vec<usize>, Vec<usize>, Vec<F>, Vec<usize>, Vec<Option<F>>);

#[allow(clippy::too_many_arguments)]
/// Assemble the full set of reduced operators for one specific quadrilateral family.
///
/// This is the family-generic core of the public assembly path.  It builds the unreduced
/// operators, applies Dirichlet reduction, compresses the final sparse matrices, and packages the
/// result into the public `Structural2dModel`.
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
    pressure_faces: &[PressureLoad],
    traction_faces: &[TractionLoad],
    thermal_material_table: Option<&[ThermalMaterial<F>]>,
    material_orientation_angles: Option<&[F]>,
    prescribed: &[(usize, F)],
    formulation: Structural2dFormulation<F>,
    quadrature: QuadratureRule,
    par: bool,
) -> Result<Structural2dModel<F>, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    let mesh = QuadMeshView2d { nodes_rz, elements };
    let ndof_full = nodes_rz.len() * 2;
    let nelem = elements.len();
    // Build the full-space -> reduced-space maps once. Every stored operator after this point will
    // live in the constrained reduced system, while `fixed_lookup` carries the prescribed values
    // needed to fold eliminated DOFs back into constant RHS terms.
    let (free_dofs, fixed_dofs, fixed_values, global_to_reduced, fixed_lookup) =
        reduce_layout(ndof_full, prescribed)?;
    let ndof_reduced = free_dofs.len();

    // Assemble stiffness in the full displacement space first, then apply Dirichlet reduction.
    // This keeps the element kernels simple and pushes all constraint handling into the common
    // reduction helpers below.
    let stiffness_full = if par {
        assemble_stiffness_for_family_par::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            mesh,
            material_ids,
            material_table,
            material_orientation_angles,
            formulation,
            quadrature,
        )?
    } else {
        assemble_stiffness_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            mesh,
            material_ids,
            material_table,
            material_orientation_angles,
            formulation,
            quadrature,
        )?
    };
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
    // Most load operators are naturally assembled in full nodal space and then reduced by
    // dropping rows associated with prescribed displacement DOFs.
    let reduce_operator =
        |operator| reduce_row_operator_to_csr(operator, &global_to_reduced, ndof_reduced);

    let (temperature_to_rhs, thermal_reference_rhs, n_temperature_nodes) =
        if let Some(thermal_material_table) = thermal_material_table {
            // Thermal loading has both a temperature-dependent operator and a constant offset from
            // per-material reference temperature, so keep those two pieces separate until the end.
            let thermal_full =
                temperature_operator_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                    mesh,
                    material_ids,
                    material_table,
                    thermal_material_table,
                    material_orientation_angles,
                    formulation,
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
    // `constant_rhs` already contains the Dirichlet offset from stiffness reduction. Add the
    // reference-temperature contribution so all load-independent terms live in one vector.
    for (dst, src) in constant_rhs.iter_mut().zip(&thermal_reference_rhs) {
        *dst = *dst + *src;
    }

    // These operators are stored directly in reduced row space because `build_rhs(...)` and
    // `solve(...)` work only with the constrained system.
    let body_force_to_rhs = reduce_operator(body_force_operator_for_family::<
        F,
        Family,
        NODES_PER_ELEMENT,
        DOF_PER_ELEMENT,
    >(mesh, formulation, quadrature)?)?;
    let pressure_to_rhs = reduce_operator(pressure_operator_for_family::<
        F,
        Family,
        NODES_PER_ELEMENT,
        DOF_PER_ELEMENT,
    >(mesh, pressure_faces, formulation, quadrature)?)?;
    let traction_to_rhs = reduce_operator(traction_operator_for_family::<
        F,
        Family,
        NODES_PER_ELEMENT,
        DOF_PER_ELEMENT,
    >(mesh, traction_faces, formulation, quadrature)?)?;

    // Recovery is assembled in full displacement space, then its displacement columns are reduced.
    // Eliminated fixed-displacement columns become constant strain/stress offsets.
    let recovery_full =
        quadrature_field_operators_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            mesh,
            material_ids,
            material_table,
            thermal_material_table,
            material_orientation_angles,
            formulation,
            quadrature,
        )?;
    let nq_row_count = recovery_full.points.len() * 4;
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
        recovery_full.points.len() * 4,
        recovery_full.ntemp,
        recovery_full.thermal_strain_rows,
        recovery_full.thermal_strain_cols,
        recovery_full.thermal_strain_vals,
    )?;
    let thermal_stress_operator = csr_from_parts(
        recovery_full.points.len() * 4,
        recovery_full.ntemp,
        recovery_full.thermal_stress_rows,
        recovery_full.thermal_stress_cols,
        recovery_full.thermal_stress_vals,
    )?;

    Ok(Structural2dModel {
        stiffness,
        body_force_to_rhs,
        pressure_to_rhs,
        traction_to_rhs,
        temperature_to_rhs,
        constant_rhs,
        recovery: ReducedRecoveryOperators {
            points: recovery_full.points,
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
        quadrature,
        formulation,
        ndof_full,
        ndof_reduced,
        nelem,
        free_dofs,
        fixed_dofs,
        fixed_values,
        lu: None,
        diagonal_preconditioner: None,
        equilibration: None,
        equilibration_options: None,
        equilibrated_stiffness: None,
        equilibrated_diagonal_preconditioner: None,
        previous_bicgstab_solution: None,
    })
}

/// Partition full-system displacement DOFs into free and fixed sets for Dirichlet reduction.
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

/// Reduce full-system stiffness triplets to the free DOF subspace.
///
/// This helper also accumulates the Dirichlet offset `-K_fc u_c` into `constant_rhs`.
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

/// Drop rows belonging to fixed displacement DOFs from one full-system load operator.
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

/// Reduce one full-system load operator and compress it into CSR form.
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

/// Reduce a full-space recovery operator by eliminating fixed displacement columns.
///
/// The eliminated fixed-value contributions are accumulated into the returned constant vector.
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

/// Compress triplet data into a CSR matrix with checked shape and index validity.
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

/// Compress stiffness triplets into the CSC format used by the cached sparse LU factorization.
fn csc_from_triplets<F: Real>(
    nrow: usize,
    ncol: usize,
    triplets: Vec<Triplet<usize, usize, F>>,
) -> Result<SparseColMat<usize, F>, String> {
    SparseColMat::try_new_from_triplets(nrow, ncol, &triplets)
        .map_err(|err| format!("failed to build CSC operator: {err:?}"))
}

/// Apply one reduced CSR load operator to a dense load-amplitude vector and accumulate the result.
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

/// Gather one element's node coordinates from the flattened analysis connectivity.
fn element_coords_from_flat<F: Real, const NODES_PER_ELEMENT: usize>(
    analysis_nodes: &[[F; 2]],
    analysis_elements_flat: &[usize],
    element_index: usize,
) -> Result<[[F; 2]; NODES_PER_ELEMENT], String> {
    let start = element_index * NODES_PER_ELEMENT;
    let end = start + NODES_PER_ELEMENT;
    let conn = analysis_elements_flat
        .get(start..end)
        .ok_or_else(|| format!("element index {element_index} out of bounds"))?;
    let mut coords = [[F::zero(); 2]; NODES_PER_ELEMENT];
    for (local_node, &node) in conn.iter().enumerate() {
        coords[local_node] = *analysis_nodes.get(node).ok_or_else(|| {
            format!(
                "element {element_index} references node {node}, but analysis mesh has only {} nodes",
                analysis_nodes.len()
            )
        })?;
    }
    Ok(coords)
}

/// Recompute physical quadrature points and mapped weights for one stored element family.
fn element_quadrature_for_family<F: Real, Family, const NODES_PER_ELEMENT: usize>(
    analysis_nodes: &[[F; 2]],
    analysis_elements_flat: &[usize],
    nelem: usize,
    formulation: Structural2dFormulation<F>,
    quadrature: QuadratureRule,
) -> Result<Structural2dElementQuadrature<F>, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    let mut points = Vec::new();
    let mut weights_area = Vec::new();
    let mut weights_volume = Vec::new();
    let mut nq_per_element = None;
    for element_index in 0..nelem {
        let coords = element_coords_from_flat::<F, NODES_PER_ELEMENT>(
            analysis_nodes,
            analysis_elements_flat,
            element_index,
        )?;
        let samples = Family::volume_samples(&coords, quadrature)?;
        if let Some(nq) = nq_per_element {
            if samples.len() != nq {
                return Err(format!(
                    "element {element_index} produced {} quadrature points, expected {nq}",
                    samples.len()
                ));
            }
        } else {
            nq_per_element = Some(samples.len());
        }
        for sample in samples {
            let area_weight = sample.det_j * sample.weight;
            points.push(sample.point);
            weights_area.push(area_weight);
            weights_volume.push(formulation.volume_scale(
                sample.point,
                sample.det_j,
                sample.weight,
            )?);
        }
    }
    Ok(Structural2dElementQuadrature {
        points,
        weights_area,
        weights_volume,
        nq_per_element: nq_per_element.unwrap_or(0),
    })
}

/// Multiply one CSR operator by a dense input vector.
fn csr_matvec<F: Real>(operator: &SparseRowMat<usize, F>, input: &[F]) -> Vec<F> {
    let mut output = vec![F::zero(); operator.nrows()];
    for row in 0..operator.nrows() {
        let start = operator.row_ptr()[row];
        let end = operator.row_ptr()[row + 1];
        let mut sum = F::zero();
        for index in start..end {
            sum = sum + operator.val()[index] * input[operator.col_idx()[index]];
        }
        output[row] = sum;
    }
    output
}

/// Validate one optional temperature vector against the model's thermal operator width.
fn normalize_temperature<'a, F: Real>(
    temperature: Option<&'a [F]>,
    expected_len: usize,
    name: &str,
) -> Result<&'a [F], String> {
    match (expected_len, temperature) {
        (0, Some(values)) if !values.is_empty() => Err(format!(
            "{name} was provided, but this model has no thermal operator"
        )),
        (0, _) => Ok(&[]),
        (_, None) => Err(format!(
            "{name} is required because this model includes thermal materials"
        )),
        (expected, Some(values)) if values.len() != expected => Err(format!(
            "{name} has length {}, but thermal operators expect {expected} values",
            values.len()
        )),
        (_, Some(values)) => Ok(values),
    }
}

/// Add one constant vector to a dense field vector.
fn add_constant<F: Real>(mut values: Vec<F>, constant: &[F]) -> Vec<F> {
    assert!(
        values.len() == constant.len(),
        "cannot add vectors of lengths {} and {}",
        values.len(),
        constant.len()
    );
    for (dst, src) in values.iter_mut().zip(constant) {
        *dst = *dst + *src;
    }
    values
}

/// Subtract one dense vector from another.
fn subtract_vectors<F: Real>(lhs: &[F], rhs: &[F]) -> Result<Vec<F>, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "cannot subtract vectors of lengths {} and {}",
            lhs.len(),
            rhs.len()
        ));
    }
    Ok(lhs.iter().zip(rhs).map(|(&a, &b)| a - b).collect())
}

/// Pack a flattened quadrature field into one `[rr, zz, tt, rz]` sample per point.
fn pack_rank4_field<F: Real>(flat: Vec<F>) -> Result<Vec<[F; 4]>, String> {
    if flat.len() % 4 != 0 {
        return Err(format!(
            "quadrature field has {} entries, which is not divisible by 4",
            flat.len()
        ));
    }
    let mut packed = Vec::with_capacity(flat.len() / 4);
    for chunk in flat.chunks_exact(4) {
        packed.push([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::solenoid_stress::convenience::isotropic_axisymmetric_material;

    #[test]
    fn bicgstab_solve_matches_direct_solve() {
        let material = [isotropic_axisymmetric_material(200.0e9_f64, 0.27)];
        let mut model = assemble_structural_2d(
            &crate::physics::solenoid_stress::test_utils::SINGLE_QUAD4_NODES,
            Structural2dElements::Quad4(
                &crate::physics::solenoid_stress::test_utils::SINGLE_QUAD4_ELEMENTS,
            ),
            &[0],
            &material,
            &[],
            &[],
            None,
            None,
            &[(0, 0.0), (1, 0.0), (3, 0.0), (5, 0.0), (7, 0.0)],
            Structural2dFormulation::Axisymmetric,
            QuadratureRule::GaussLegendre3,
            false,
        )
        .unwrap();
        let rhs = model
            .build_rhs(Some(&[2.0e5, -1.0e5]), None, None, None)
            .unwrap();
        let direct = model.solve(&rhs).unwrap();
        let iterative = model
            .solve_with_method(
                &rhs,
                Structural2dSolveMethod::Bicgstab(BicgstabSolveOptions {
                    tolerance: Some(1.0e-9),
                    max_iterations: Some(1000),
                    ..BicgstabSolveOptions::default()
                }),
            )
            .unwrap();

        assert!(iterative.diagnostics.converged);
        assert!(iterative.diagnostics.iterations > 0);
        for (actual, expected) in iterative.displacement.iter().zip(&direct) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
    }
}
