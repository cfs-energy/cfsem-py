//! Reduced-model assembly and solve wrapper for the 2D structural FEM.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use faer::Col;
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::Lu;
use faer::sparse::{
    SparseColMat, SparseRowMat, SymbolicSparseColMat, SymbolicSparseRowMat, Triplet,
};
use rayon::prelude::*;

use crate::mesh::elements::quad2d::{quad4, quad9};
use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::assembly::{
    assemble_stiffness_chunks_for_family_par, assemble_stiffness_for_family,
};
use crate::physics::solenoid_stress::convenience::{
    QuadratureFieldSamples, Structural2dElementMeasures, Structural2dElementQuadrature,
};
use crate::physics::solenoid_stress::family::{Quad4Family, Quad9Family, QuadElementFamily};
use crate::physics::solenoid_stress::loads::{
    SparseOperator, body_force_operator_for_family, body_force_operator_for_family_par,
    pressure_operator_for_family, pressure_operator_for_family_par,
    temperature_operator_for_family, temperature_operator_for_family_par,
    traction_operator_for_family, traction_operator_for_family_par,
};
use crate::physics::solenoid_stress::recovery::{
    CsrOperatorParts, reduced_quadrature_field_operators_for_family,
    reduced_quadrature_field_operators_for_family_par,
};
use crate::physics::solenoid_stress::types::{
    PressureLoad, Real, StiffnessTriplets, Structural2dFormulation, ThermalMaterial, TractionLoad,
    dof_per_element,
};

type F = f64;

/// Public element-family selector for the 2D structural solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Structural2dElementType {
    /// Bilinear four-node quadrilateral in the analysis plane.
    Quad4,
    /// Quadratic nine-node quadrilateral in the analysis plane.
    Quad9,
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
pub struct ReducedRecoveryOperators {
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
pub struct Structural2dModel {
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
    pub recovery: ReducedRecoveryOperators,
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
}

impl Structural2dModel {
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
        let mut full = vec![0.0; self.ndof_full];
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
        let mut areas = vec![0.0; self.nelem];
        let mut volumes = vec![0.0; self.nelem];
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

#[allow(clippy::too_many_arguments)]
/// Assemble the reduced 2D structural model and all associated operators.
pub fn assemble_structural_2d(
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
) -> Result<Structural2dModel, String> {
    match elements {
        Structural2dElements::Quad4(elements) => build_model_for_family::<
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
fn build_model_for_family<Family, const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
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
) -> Result<Structural2dModel, String>
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

    let mut constant_rhs = vec![0.0; ndof_reduced];
    // Assemble stiffness in the full displacement space first, then apply Dirichlet reduction.
    // The parallel path keeps worker chunks separate so each chunk can be reduced and sorted
    // independently before a k-way merge builds the canonical CSC structure.
    let stiffness = if par {
        let stiffness_chunks = assemble_stiffness_chunks_for_family_par::<
            F,
            Family,
            NODES_PER_ELEMENT,
            DOF_PER_ELEMENT,
        >(
            mesh,
            material_ids,
            material_table,
            material_orientation_angles,
            formulation,
            quadrature,
        )?;
        let sorted_chunks = reduce_sort_stiffness_chunks(
            stiffness_chunks,
            &global_to_reduced,
            &fixed_lookup,
            &mut constant_rhs,
            ndof_reduced,
        );
        csc_from_sorted_triplet_chunks(ndof_reduced, ndof_reduced, sorted_chunks)
    } else {
        let stiffness_full =
            assemble_stiffness_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                material_ids,
                material_table,
                material_orientation_angles,
                formulation,
                quadrature,
            )?;
        let stiffness_reduced = reduce_square_triplets(
            &stiffness_full.rows,
            &stiffness_full.cols,
            &stiffness_full.vals,
            &global_to_reduced,
            &fixed_lookup,
            &mut constant_rhs,
        );
        csc_from_triplets(ndof_reduced, ndof_reduced, stiffness_reduced)
    };
    // Most load operators are naturally assembled in full nodal space and then reduced by
    // dropping rows associated with prescribed displacement DOFs.
    let reduce_operator =
        |operator| reduce_row_operator_to_csr(operator, &global_to_reduced, ndof_reduced);

    let (temperature_to_rhs, thermal_reference_rhs, n_temperature_nodes) = if let Some(
        thermal_material_table,
    ) =
        thermal_material_table
    {
        // Thermal loading has both a temperature-dependent operator and a constant offset from
        // per-material reference temperature, so keep those two pieces separate until the end.
        let thermal_full = if par {
            temperature_operator_for_family_par::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                material_ids,
                material_table,
                thermal_material_table,
                material_orientation_angles,
                formulation,
                quadrature,
            )
        } else {
            temperature_operator_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                material_ids,
                material_table,
                thermal_material_table,
                material_orientation_angles,
                formulation,
                quadrature,
            )
        }?;
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
            vec![0.0; ndof_reduced],
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
    let body_force_operator = if par {
        body_force_operator_for_family_par::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            mesh,
            formulation,
            quadrature,
        )?
    } else {
        body_force_operator_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            mesh,
            formulation,
            quadrature,
        )?
    };
    let body_force_to_rhs = reduce_operator(body_force_operator)?;
    let pressure_operator = if par {
        pressure_operator_for_family_par::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            mesh,
            pressure_faces,
            formulation,
            quadrature,
        )?
    } else {
        pressure_operator_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            mesh,
            pressure_faces,
            formulation,
            quadrature,
        )?
    };
    let pressure_to_rhs = reduce_operator(pressure_operator)?;
    let traction_operator = if par {
        traction_operator_for_family_par::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            mesh,
            traction_faces,
            formulation,
            quadrature,
        )?
    } else {
        traction_operator_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            mesh,
            traction_faces,
            formulation,
            quadrature,
        )?
    };
    let traction_to_rhs = reduce_operator(traction_operator)?;

    // Recovery rows are disjoint by quadrature point, so assemble directly in reduced CSR form
    // instead of building full-space triplets and reducing them afterwards.
    let recovery_reduced = if par {
        reduced_quadrature_field_operators_for_family_par::<
            F,
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
            &global_to_reduced,
            &fixed_lookup,
            ndof_reduced,
        )
    } else {
        reduced_quadrature_field_operators_for_family::<F, Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
            mesh,
            material_ids,
            material_table,
            thermal_material_table,
            material_orientation_angles,
            formulation,
            quadrature,
            &global_to_reduced,
            &fixed_lookup,
            ndof_reduced,
        )
    }?;
    let strain_operator = csr_from_canonical_parts(recovery_reduced.strain_operator);
    let stress_operator = csr_from_canonical_parts(recovery_reduced.stress_operator);
    let thermal_strain_operator =
        csr_from_canonical_parts(recovery_reduced.thermal_strain_operator);
    let thermal_stress_operator =
        csr_from_canonical_parts(recovery_reduced.thermal_stress_operator);

    Ok(Structural2dModel {
        stiffness,
        body_force_to_rhs,
        pressure_to_rhs,
        traction_to_rhs,
        temperature_to_rhs,
        constant_rhs,
        recovery: ReducedRecoveryOperators {
            points: recovery_reduced.points,
            strain_operator,
            stress_operator,
            thermal_strain_operator,
            thermal_stress_operator,
            strain_constant: recovery_reduced.strain_constant,
            stress_constant: recovery_reduced.stress_constant,
            thermal_strain_constant: recovery_reduced.thermal_strain_constant,
            thermal_stress_constant: recovery_reduced.thermal_stress_constant,
            nq_per_element: recovery_reduced.nq_per_element,
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

/// Reduce, sort, and coalesce each stiffness chunk independently.
///
/// Each Rayon worker returns full-system triplets for a disjoint element range.  Reduction can
/// still create duplicate `(row, col)` entries inside a chunk, so this function canonicalizes each
/// chunk before the final k-way merge.  The Dirichlet RHS offsets are accumulated per chunk to
/// avoid shared mutable state during the parallel pass.
fn reduce_sort_stiffness_chunks<F: Real>(
    chunks: Vec<StiffnessTriplets<F>>,
    global_to_reduced: &[usize],
    fixed_lookup: &[Option<F>],
    constant_rhs: &mut [F],
    ndof_reduced: usize,
) -> Vec<Vec<Triplet<usize, usize, F>>> {
    let reduced_chunks = chunks
        .into_par_iter()
        .map(|chunk| {
            let mut local_rhs = vec![F::zero(); ndof_reduced];
            let triplets = reduce_square_triplets(
                &chunk.rows,
                &chunk.cols,
                &chunk.vals,
                global_to_reduced,
                fixed_lookup,
                &mut local_rhs,
            );
            (sort_coalesce_triplets(triplets), local_rhs)
        })
        .collect::<Vec<_>>();

    let mut sorted_chunks = Vec::with_capacity(reduced_chunks.len());
    for (triplets, local_rhs) in reduced_chunks {
        for (dst, src) in constant_rhs.iter_mut().zip(local_rhs) {
            *dst = *dst + src;
        }
        sorted_chunks.push(triplets);
    }
    sorted_chunks
}

/// Return triplets in canonical CSC order with duplicate `(column, row)` entries coalesced.
fn sort_coalesce_triplets<F: Real>(
    mut triplets: Vec<Triplet<usize, usize, F>>,
) -> Vec<Triplet<usize, usize, F>> {
    triplets.sort_by(|a, b| a.col.cmp(&b.col).then_with(|| a.row.cmp(&b.row)));

    let mut coalesced = Vec::with_capacity(triplets.len());
    let mut pending: Option<Triplet<usize, usize, F>> = None;
    for triplet in triplets {
        if triplet.val == F::zero() {
            continue;
        }
        match pending {
            Some(mut current) if current.col == triplet.col && current.row == triplet.row => {
                current.val = current.val + triplet.val;
                pending = Some(current);
            }
            Some(current) => {
                if current.val != F::zero() {
                    coalesced.push(current);
                }
                pending = Some(triplet);
            }
            None => pending = Some(triplet),
        }
    }
    if let Some(current) = pending
        && current.val != F::zero()
    {
        coalesced.push(current);
    }
    coalesced
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

/// Build a CSR matrix from already canonical row pointers, column indices, and values.
fn csr_from_canonical_parts<F: Real>(parts: CsrOperatorParts<F>) -> SparseRowMat<usize, F> {
    let symbolic = SymbolicSparseRowMat::new_checked(
        parts.nrow,
        parts.ncol,
        parts.row_ptr,
        None,
        parts.col_idx,
    );
    SparseRowMat::new(symbolic, parts.vals)
}

/// Compress stiffness triplets into the CSC format used by the cached sparse LU factorization.
fn csc_from_triplets<F: Real>(
    nrow: usize,
    ncol: usize,
    mut triplets: Vec<Triplet<usize, usize, F>>,
) -> SparseColMat<usize, F> {
    triplets.sort_by(|a, b| a.col.cmp(&b.col).then_with(|| a.row.cmp(&b.row)));

    let (col_ptr, row_idx, vals) = pack_sorted_triplets_to_csc(nrow, ncol, triplets);
    let symbolic = SymbolicSparseColMat::new_checked(nrow, ncol, col_ptr, None, row_idx);
    SparseColMat::new(symbolic, vals)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct TripletCursor {
    /// Current triplet column in a sorted chunk.
    col: usize,
    /// Current triplet row in a sorted chunk.
    row: usize,
    /// Index of the source chunk in the outer chunk slice.
    chunk: usize,
    /// Index of the current triplet inside that source chunk.
    index: usize,
}

impl Ord for TripletCursor {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a max-heap, so reverse the ordering to pop the smallest `(col, row)`
        // cursor first during the k-way merge.
        other
            .col
            .cmp(&self.col)
            .then_with(|| other.row.cmp(&self.row))
            .then_with(|| other.chunk.cmp(&self.chunk))
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for TripletCursor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct CscPartsBuilder<F: Real> {
    /// Number of matrix rows.
    nrow: usize,
    /// Number of matrix columns.
    ncol: usize,
    /// CSC column offsets under construction.
    col_ptr: Vec<usize>,
    /// Row index for each stored value.
    row_idx: Vec<usize>,
    /// Nonzero values in column-major CSC order.
    vals: Vec<F>,
    /// First column whose pointer has not yet been finalized.
    next_col: usize,
    /// Most recent sorted entry, held back so duplicates can be coalesced before writing.
    pending: Option<Triplet<usize, usize, F>>,
}

impl<F: Real> CscPartsBuilder<F> {
    /// Start a CSC builder that accepts triplets sorted by `(column, row)`.
    fn new(nrow: usize, ncol: usize, capacity: usize) -> Self {
        let mut col_ptr = Vec::with_capacity(ncol + 1);
        col_ptr.push(0);
        Self {
            nrow,
            ncol,
            col_ptr,
            row_idx: Vec::with_capacity(capacity),
            vals: Vec::with_capacity(capacity),
            next_col: 0,
            pending: None,
        }
    }

    /// Append one sorted triplet, coalescing duplicates and skipping exact zeros.
    fn push_sorted(&mut self, triplet: Triplet<usize, usize, F>) {
        debug_assert!(triplet.row < self.nrow);
        debug_assert!(triplet.col < self.ncol);
        if triplet.val == F::zero() {
            return;
        }
        match self.pending {
            Some(mut current) if current.col == triplet.col && current.row == triplet.row => {
                current.val = current.val + triplet.val;
                self.pending = Some(current);
            }
            Some(current) => {
                self.push_entry(current);
                self.pending = Some(triplet);
            }
            None => self.pending = Some(triplet),
        }
    }

    /// Finalize pending entries and close all remaining column pointers.
    fn finish(mut self) -> (Vec<usize>, Vec<usize>, Vec<F>) {
        if let Some(current) = self.pending.take() {
            self.push_entry(current);
        }
        while self.next_col < self.ncol {
            self.col_ptr.push(self.row_idx.len());
            self.next_col += 1;
        }
        debug_assert_eq!(self.col_ptr.len(), self.ncol + 1);
        (self.col_ptr, self.row_idx, self.vals)
    }

    /// Write one already coalesced CSC entry and fill empty-column pointers before it.
    fn push_entry(&mut self, entry: Triplet<usize, usize, F>) {
        while self.next_col < entry.col {
            self.col_ptr.push(self.row_idx.len());
            self.next_col += 1;
        }
        if entry.val != F::zero() {
            self.row_idx.push(entry.row);
            self.vals.push(entry.val);
        }
    }
}

/// Merge already sorted/coalesced triplet chunks into the CSC format used by sparse LU.
fn csc_from_sorted_triplet_chunks<F: Real>(
    nrow: usize,
    ncol: usize,
    chunks: Vec<Vec<Triplet<usize, usize, F>>>,
) -> SparseColMat<usize, F> {
    let (col_ptr, row_idx, vals) = merge_sorted_triplet_chunks_to_csc(nrow, ncol, &chunks);
    let symbolic = SymbolicSparseColMat::new_checked(nrow, ncol, col_ptr, None, row_idx);
    SparseColMat::new(symbolic, vals)
}

fn merge_sorted_triplet_chunks_to_csc<F: Real>(
    nrow: usize,
    ncol: usize,
    chunks: &[Vec<Triplet<usize, usize, F>>],
) -> (Vec<usize>, Vec<usize>, Vec<F>) {
    let capacity = chunks.iter().map(Vec::len).sum::<usize>();
    let mut builder = CscPartsBuilder::new(nrow, ncol, capacity);
    let mut heap = BinaryHeap::with_capacity(chunks.len());
    // Seed the heap with the first triplet from each sorted chunk.  Every pop advances only that
    // source chunk, giving a k-way merge without flattening and sorting all triplets again.
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        if let Some(first) = chunk.first() {
            heap.push(TripletCursor {
                col: first.col,
                row: first.row,
                chunk: chunk_index,
                index: 0,
            });
        }
    }

    while let Some(cursor) = heap.pop() {
        let triplet = chunks[cursor.chunk][cursor.index];
        builder.push_sorted(triplet);

        let next_index = cursor.index + 1;
        if next_index < chunks[cursor.chunk].len() {
            let next = chunks[cursor.chunk][next_index];
            heap.push(TripletCursor {
                col: next.col,
                row: next.row,
                chunk: cursor.chunk,
                index: next_index,
            });
        }
    }

    builder.finish()
}

/// Pack triplets sorted by `(column, row)` into canonical CSC arrays.
fn pack_sorted_triplets_to_csc<F: Real>(
    nrow: usize,
    ncol: usize,
    triplets: Vec<Triplet<usize, usize, F>>,
) -> (Vec<usize>, Vec<usize>, Vec<F>) {
    let mut builder = CscPartsBuilder::new(nrow, ncol, triplets.len());
    for triplet in triplets {
        builder.push_sorted(triplet);
    }
    builder.finish()
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
