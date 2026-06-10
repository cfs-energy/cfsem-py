//! Reduced-model assembly and solve wrapper for the 2D structural FEM.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use faer::Col;
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::Lu;
use faer::sparse::{SparseColMat, SparseRowMat, SymbolicSparseColMat, Triplet};
use rayon::prelude::*;

use crate::mesh::elements::quad2d::quadrature::gauss_volume;
use crate::mesh::elements::quad2d::{mapping, quad4, quad9};
use crate::mesh::quad2d::{QuadReferenceElement, closest_reference_point};
use crate::mesh::{QuadMeshView2d, QuadratureRule};
use crate::physics::solenoid_stress::assembly::{
    assemble_stiffness_chunks_for_family_par, assemble_stiffness_for_family,
};
use crate::physics::solenoid_stress::axisym::{build_b_matrix, constitutive_times_strain};
use crate::physics::solenoid_stress::convenience::{
    Structural2dElementMeasures, rotate_material_in_plane, rotate_thermal_material_in_plane,
};
use crate::physics::solenoid_stress::family::{Quad4Family, Quad9Family, QuadElementFamily};
use crate::physics::solenoid_stress::loads::{
    SparseOperator, apply_body_force_rhs_for_family, apply_pressure_rhs_for_family,
    apply_temperature_rhs_for_family, apply_traction_rhs_for_family,
    body_force_operator_for_family, body_force_operator_for_family_par,
    pressure_operator_for_family, pressure_operator_for_family_par,
    temperature_operator_for_family, temperature_operator_for_family_par,
    thermal_reference_rhs_for_family, traction_operator_for_family,
    traction_operator_for_family_par,
};
use crate::physics::solenoid_stress::types::{
    PressureLoad, StiffnessTriplets, Structural2dFormulation, ThermalMaterial, TractionLoad,
    dof_per_element,
};

/// Element-owned point locations for structural 2D recovery.
#[derive(Debug, Clone)]
pub struct Structural2dPointLocations {
    /// Physical point coordinates in element-major or user-supplied order.
    pub points: Vec<[f64; 2]>,
    /// Owning element index for each point.
    pub element_indices: Vec<usize>,
    /// Reference coordinates for each point.
    pub reference_points: Vec<[f64; 2]>,
    /// Optional mapped analysis-plane quadrature weights.
    pub weights_area: Vec<f64>,
    /// Optional mapped represented-volume quadrature weights.
    pub weights_volume: Vec<f64>,
    /// Number of consecutive points contributed by each element for quadrature locations.
    pub points_per_element: usize,
}

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

/// Inputs retained so sparse exports and matrix-free operations can be built after assembly.
#[derive(Debug, Clone)]
struct Structural2dAssemblyData {
    material_ids: Vec<usize>,
    material_table: Vec<[[f64; 4]; 4]>,
    thermal_material_table: Option<Vec<ThermalMaterial>>,
    material_orientation_angles: Option<Vec<f64>>,
    global_to_reduced: Vec<usize>,
    par: bool,
}

/// Fully assembled reduced 2D structural model.
///
/// The public system stored here is the Dirichlet-reduced system. `stiffness` and `constant_rhs`
/// live in reduced displacement space. Sparse load and recovery operators are computed on demand
/// for explicit exports and are not retained by the model.
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
    pub stiffness: SparseColMat<usize, f64>,
    /// Constant reduced RHS contribution from prescribed displacements and thermal reference state.
    ///
    /// Shape: `(ndof_reduced,)`.
    /// Units: `[energy / distance]`.
    pub constant_rhs: Vec<f64>,
    /// Number of quadrature points contributed by each element.
    pub nq_per_element: usize,
    /// Number of nodal temperatures expected by thermal operators.
    pub n_temperature_nodes: usize,
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
    pub analysis_nodes: Vec<[f64; 2]>,
    /// Flattened analysis connectivity in element-major order.
    ///
    /// Shape: `(nelem * nodes_per_element,)`.
    pub analysis_elements_flat: Vec<usize>,
    /// Number of nodes per analysis element.
    pub nodes_per_element: usize,
    /// Analysis element family used by the backend.
    pub element_type: Structural2dElementType,
    /// Volume and face quadrature rule used for assembly, matrix-free loads, and sparse exports.
    pub quadrature: QuadratureRule,
    /// Structural 2D formulation used by the backend.
    pub formulation: Structural2dFormulation,
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
    pub fixed_values: Vec<f64>,
    /// Assembly inputs retained for matrix-free operations and sparse exports.
    assembly: Structural2dAssemblyData,
    lu: Option<Lu<usize, f64>>,
}

impl Structural2dModel {
    fn analysis_elements<const NODES_PER_ELEMENT: usize>(
        &self,
    ) -> Result<Vec<[usize; NODES_PER_ELEMENT]>, String> {
        elements_from_flat::<NODES_PER_ELEMENT>(
            &self.analysis_elements_flat,
            self.nelem,
            self.element_type,
        )
    }

    fn pressure_loads(&self) -> Vec<PressureLoad> {
        self.pressure_faces
            .iter()
            .map(|face| PressureLoad {
                element: face[0],
                local_face: face[1] as u8,
            })
            .collect()
    }

    fn traction_loads(&self) -> Vec<TractionLoad> {
        self.traction_faces
            .iter()
            .map(|face| TractionLoad {
                element: face[0],
                local_face: face[1] as u8,
            })
            .collect()
    }

    fn reduce_load_operator(
        &self,
        operator: SparseOperator,
    ) -> Result<SparseRowMat<usize, f64>, String> {
        reduce_row_operator_to_csr(
            operator,
            &self.assembly.global_to_reduced,
            self.ndof_reduced,
        )
    }

    fn build_body_force_to_rhs_for_family<
        Family,
        const NODES_PER_ELEMENT: usize,
        const DOF_PER_ELEMENT: usize,
    >(
        &self,
    ) -> Result<SparseRowMat<usize, f64>, String>
    where
        Family: QuadElementFamily<NODES_PER_ELEMENT>,
    {
        let elements = self.analysis_elements::<NODES_PER_ELEMENT>()?;
        let mesh = QuadMeshView2d {
            nodes_rz: &self.analysis_nodes,
            elements: &elements,
        };
        let operator = if self.assembly.par {
            body_force_operator_for_family_par::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                self.formulation,
                self.quadrature,
            )
        } else {
            body_force_operator_for_family::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                self.formulation,
                self.quadrature,
            )
        }?;
        self.reduce_load_operator(operator)
    }

    fn build_pressure_to_rhs_for_family<
        Family,
        const NODES_PER_ELEMENT: usize,
        const DOF_PER_ELEMENT: usize,
    >(
        &self,
    ) -> Result<SparseRowMat<usize, f64>, String>
    where
        Family: QuadElementFamily<NODES_PER_ELEMENT>,
    {
        let elements = self.analysis_elements::<NODES_PER_ELEMENT>()?;
        let pressure_faces = self.pressure_loads();
        let mesh = QuadMeshView2d {
            nodes_rz: &self.analysis_nodes,
            elements: &elements,
        };
        let operator = if self.assembly.par {
            pressure_operator_for_family_par::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                &pressure_faces,
                self.formulation,
                self.quadrature,
            )
        } else {
            pressure_operator_for_family::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                &pressure_faces,
                self.formulation,
                self.quadrature,
            )
        }?;
        self.reduce_load_operator(operator)
    }

    fn build_traction_to_rhs_for_family<
        Family,
        const NODES_PER_ELEMENT: usize,
        const DOF_PER_ELEMENT: usize,
    >(
        &self,
    ) -> Result<SparseRowMat<usize, f64>, String>
    where
        Family: QuadElementFamily<NODES_PER_ELEMENT>,
    {
        let elements = self.analysis_elements::<NODES_PER_ELEMENT>()?;
        let traction_faces = self.traction_loads();
        let mesh = QuadMeshView2d {
            nodes_rz: &self.analysis_nodes,
            elements: &elements,
        };
        let operator = if self.assembly.par {
            traction_operator_for_family_par::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                &traction_faces,
                self.formulation,
                self.quadrature,
            )
        } else {
            traction_operator_for_family::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                &traction_faces,
                self.formulation,
                self.quadrature,
            )
        }?;
        self.reduce_load_operator(operator)
    }

    fn build_temperature_to_rhs_for_family<
        Family,
        const NODES_PER_ELEMENT: usize,
        const DOF_PER_ELEMENT: usize,
    >(
        &self,
    ) -> Result<SparseRowMat<usize, f64>, String>
    where
        Family: QuadElementFamily<NODES_PER_ELEMENT>,
    {
        let Some(thermal_material_table) = self.assembly.thermal_material_table.as_ref() else {
            return csr_from_parts(self.ndof_reduced, 0, Vec::new(), Vec::new(), Vec::new());
        };
        let elements = self.analysis_elements::<NODES_PER_ELEMENT>()?;
        let mesh = QuadMeshView2d {
            nodes_rz: &self.analysis_nodes,
            elements: &elements,
        };
        let operator = if self.assembly.par {
            temperature_operator_for_family_par::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                &self.assembly.material_ids,
                &self.assembly.material_table,
                thermal_material_table,
                self.assembly.material_orientation_angles.as_deref(),
                self.formulation,
                self.quadrature,
            )
        } else {
            temperature_operator_for_family::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                &self.assembly.material_ids,
                &self.assembly.material_table,
                thermal_material_table,
                self.assembly.material_orientation_angles.as_deref(),
                self.formulation,
                self.quadrature,
            )
        }?;
        self.reduce_load_operator(operator.temperature_to_rhs)
    }

    pub fn body_force_to_rhs(&self) -> Result<SparseRowMat<usize, f64>, String> {
        match self.element_type {
            Structural2dElementType::Quad4 => self.build_body_force_to_rhs_for_family::<
                Quad4Family,
                { quad4::NODES_PER_ELEMENT },
                { dof_per_element(quad4::NODES_PER_ELEMENT) },
            >(),
            Structural2dElementType::Quad9 => self.build_body_force_to_rhs_for_family::<
                Quad9Family,
                { quad9::NODES_PER_ELEMENT },
                { dof_per_element(quad9::NODES_PER_ELEMENT) },
            >(),
        }
    }

    pub fn pressure_to_rhs(&self) -> Result<SparseRowMat<usize, f64>, String> {
        match self.element_type {
            Structural2dElementType::Quad4 => self.build_pressure_to_rhs_for_family::<
                Quad4Family,
                { quad4::NODES_PER_ELEMENT },
                { dof_per_element(quad4::NODES_PER_ELEMENT) },
            >(),
            Structural2dElementType::Quad9 => self.build_pressure_to_rhs_for_family::<
                Quad9Family,
                { quad9::NODES_PER_ELEMENT },
                { dof_per_element(quad9::NODES_PER_ELEMENT) },
            >(),
        }
    }

    pub fn traction_to_rhs(&self) -> Result<SparseRowMat<usize, f64>, String> {
        match self.element_type {
            Structural2dElementType::Quad4 => self.build_traction_to_rhs_for_family::<
                Quad4Family,
                { quad4::NODES_PER_ELEMENT },
                { dof_per_element(quad4::NODES_PER_ELEMENT) },
            >(),
            Structural2dElementType::Quad9 => self.build_traction_to_rhs_for_family::<
                Quad9Family,
                { quad9::NODES_PER_ELEMENT },
                { dof_per_element(quad9::NODES_PER_ELEMENT) },
            >(),
        }
    }

    pub fn temperature_to_rhs(&self) -> Result<SparseRowMat<usize, f64>, String> {
        match self.element_type {
            Structural2dElementType::Quad4 => self.build_temperature_to_rhs_for_family::<
                Quad4Family,
                { quad4::NODES_PER_ELEMENT },
                { dof_per_element(quad4::NODES_PER_ELEMENT) },
            >(),
            Structural2dElementType::Quad9 => self.build_temperature_to_rhs_for_family::<
                Quad9Family,
                { quad9::NODES_PER_ELEMENT },
                { dof_per_element(quad9::NODES_PER_ELEMENT) },
            >(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_matrix_free_loads_for_family<
        Family,
        const NODES_PER_ELEMENT: usize,
        const DOF_PER_ELEMENT: usize,
    >(
        &self,
        body_force: Option<&[f64]>,
        pressure_values: Option<&[f64]>,
        traction_values: Option<&[f64]>,
        nodal_temperature: Option<&[f64]>,
        rhs: &mut [f64],
    ) -> Result<(), String>
    where
        Family: QuadElementFamily<NODES_PER_ELEMENT>,
    {
        let elements = self.analysis_elements::<NODES_PER_ELEMENT>()?;
        let mesh = QuadMeshView2d {
            nodes_rz: &self.analysis_nodes,
            elements: &elements,
        };
        if let Some(body_force) = body_force {
            apply_body_force_rhs_for_family::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                self.formulation,
                self.quadrature,
                body_force,
                &self.assembly.global_to_reduced,
                rhs,
            )?;
        }
        if let Some(pressure_values) = pressure_values {
            let pressure_faces = self.pressure_loads();
            apply_pressure_rhs_for_family::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                &pressure_faces,
                self.formulation,
                self.quadrature,
                pressure_values,
                &self.assembly.global_to_reduced,
                rhs,
            )?;
        }
        if let Some(traction_values) = traction_values {
            let traction_faces = self.traction_loads();
            apply_traction_rhs_for_family::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                mesh,
                &traction_faces,
                self.formulation,
                self.quadrature,
                traction_values,
                &self.assembly.global_to_reduced,
                rhs,
            )?;
        }
        match (
            self.assembly.thermal_material_table.as_ref(),
            nodal_temperature,
        ) {
            (Some(thermal_material_table), Some(nodal_temperature)) => {
                apply_temperature_rhs_for_family::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                    mesh,
                    &self.assembly.material_ids,
                    &self.assembly.material_table,
                    thermal_material_table,
                    self.assembly.material_orientation_angles.as_deref(),
                    self.formulation,
                    self.quadrature,
                    nodal_temperature,
                    &self.assembly.global_to_reduced,
                    rhs,
                )?;
            }
            (Some(_), None) => {
                return Err(
                    "nodal_temperature is required because this model includes thermal materials"
                        .to_string(),
                );
            }
            (None, Some(nodal_temperature)) if !nodal_temperature.is_empty() => {
                return Err(
                    "nodal_temperature was provided, but this model has no thermal operator"
                        .to_string(),
                );
            }
            (None, _) => {}
        }
        Ok(())
    }

    fn apply_matrix_free_loads(
        &self,
        body_force: Option<&[f64]>,
        pressure_values: Option<&[f64]>,
        traction_values: Option<&[f64]>,
        nodal_temperature: Option<&[f64]>,
        rhs: &mut [f64],
    ) -> Result<(), String> {
        match self.element_type {
            Structural2dElementType::Quad4 => self.apply_matrix_free_loads_for_family::<
                Quad4Family,
                { quad4::NODES_PER_ELEMENT },
                { dof_per_element(quad4::NODES_PER_ELEMENT) },
            >(body_force, pressure_values, traction_values, nodal_temperature, rhs),
            Structural2dElementType::Quad9 => self.apply_matrix_free_loads_for_family::<
                Quad9Family,
                { quad9::NODES_PER_ELEMENT },
                { dof_per_element(quad9::NODES_PER_ELEMENT) },
            >(body_force, pressure_values, traction_values, nodal_temperature, rhs),
        }
    }

    fn evaluate_strain_for_locations_for_family<
        Family,
        const NODES_PER_ELEMENT: usize,
        const DOF_PER_ELEMENT: usize,
    >(
        &self,
        element_indices: &[usize],
        reference_points: &[[f64; 2]],
        displacements_full: &[f64],
    ) -> Result<Vec<[f64; 4]>, String>
    where
        Family: QuadElementFamily<NODES_PER_ELEMENT>,
    {
        validate_location_lengths(element_indices, reference_points)?;
        let elements = self.analysis_elements::<NODES_PER_ELEMENT>()?;
        let mesh = QuadMeshView2d {
            nodes_rz: &self.analysis_nodes,
            elements: &elements,
        };
        let mut strain = Vec::with_capacity(element_indices.len());
        for (&element_index, &reference) in element_indices.iter().zip(reference_points) {
            let coords = mesh.element_coords(element_index)?;
            let nodes = mesh.element_nodes(element_index)?;
            let mut local_u = [0.0; DOF_PER_ELEMENT];
            for local_node in 0..NODES_PER_ELEMENT {
                let global_node = nodes[local_node];
                local_u[2 * local_node] = displacements_full[2 * global_node];
                local_u[2 * local_node + 1] = displacements_full[2 * global_node + 1];
            }
            let shape = Family::ReferenceElement::shape(reference[0], reference[1]);
            let grad_ref = Family::ReferenceElement::grad_ref(reference[0], reference[1]);
            let jac = mapping::jacobian(&coords, &grad_ref);
            let inv_jac = mapping::inv_j(&jac)?;
            let grad_phys = mapping::grad_phys(&grad_ref, &inv_jac);
            let point = mapping::map_point(&coords, &shape);
            let b = build_b_matrix::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                self.formulation,
                &shape,
                &grad_phys,
                point,
            )?;
            let mut sample_strain = [0.0; 4];
            for component in 0..4 {
                let mut value = 0.0;
                for dof in 0..DOF_PER_ELEMENT {
                    value += b[component][dof] * local_u[dof];
                }
                sample_strain[component] = value;
            }
            strain.push(sample_strain);
        }
        Ok(strain)
    }

    fn evaluate_stress_for_locations_for_family<
        Family,
        const NODES_PER_ELEMENT: usize,
        const DOF_PER_ELEMENT: usize,
    >(
        &self,
        element_indices: &[usize],
        reference_points: &[[f64; 2]],
        displacements_full: &[f64],
    ) -> Result<Vec<[f64; 4]>, String>
    where
        Family: QuadElementFamily<NODES_PER_ELEMENT>,
    {
        validate_location_lengths(element_indices, reference_points)?;
        let elements = self.analysis_elements::<NODES_PER_ELEMENT>()?;
        let mesh = QuadMeshView2d {
            nodes_rz: &self.analysis_nodes,
            elements: &elements,
        };
        let mut stress = Vec::with_capacity(element_indices.len());
        for (&element_index, &reference) in element_indices.iter().zip(reference_points) {
            let coords = mesh.element_coords(element_index)?;
            let nodes = mesh.element_nodes(element_index)?;
            let material_id = self.assembly.material_ids[element_index];
            let material = self
                .assembly
                .material_table
                .get(material_id)
                .ok_or_else(|| {
                    format!("material_id {material_id} on element {element_index} is out of range")
                })?;
            let material_storage;
            let material = if let Some(angles) = self.assembly.material_orientation_angles.as_ref()
            {
                material_storage = rotate_material_in_plane(material, angles[element_index]);
                &material_storage
            } else {
                material
            };
            let mut local_u = [0.0; DOF_PER_ELEMENT];
            for local_node in 0..NODES_PER_ELEMENT {
                let global_node = nodes[local_node];
                local_u[2 * local_node] = displacements_full[2 * global_node];
                local_u[2 * local_node + 1] = displacements_full[2 * global_node + 1];
            }
            let shape = Family::ReferenceElement::shape(reference[0], reference[1]);
            let grad_ref = Family::ReferenceElement::grad_ref(reference[0], reference[1]);
            let jac = mapping::jacobian(&coords, &grad_ref);
            let inv_jac = mapping::inv_j(&jac)?;
            let grad_phys = mapping::grad_phys(&grad_ref, &inv_jac);
            let point = mapping::map_point(&coords, &shape);
            let b = build_b_matrix::<NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
                self.formulation,
                &shape,
                &grad_phys,
                point,
            )?;
            let mut sample_strain = [0.0; 4];
            for component in 0..4 {
                let mut value = 0.0;
                for dof in 0..DOF_PER_ELEMENT {
                    value += b[component][dof] * local_u[dof];
                }
                sample_strain[component] = value;
            }
            stress.push(constitutive_times_strain(material, &sample_strain));
        }
        Ok(stress)
    }

    fn evaluate_thermal_for_locations_for_family<
        Family,
        const NODES_PER_ELEMENT: usize,
        const STRESS: bool,
    >(
        &self,
        element_indices: &[usize],
        reference_points: &[[f64; 2]],
        nodal_temperature: &[f64],
    ) -> Result<Vec<[f64; 4]>, String>
    where
        Family: QuadElementFamily<NODES_PER_ELEMENT>,
    {
        validate_location_lengths(element_indices, reference_points)?;
        let Some(thermal_material_table) = self.assembly.thermal_material_table.as_ref() else {
            return Ok(vec![[0.0; 4]; element_indices.len()]);
        };
        if nodal_temperature.len() != self.n_temperature_nodes {
            return Err(format!(
                "nodal_temperature has length {}, but thermal recovery expects {} values",
                nodal_temperature.len(),
                self.n_temperature_nodes
            ));
        }

        let elements = self.analysis_elements::<NODES_PER_ELEMENT>()?;
        let mesh = QuadMeshView2d {
            nodes_rz: &self.analysis_nodes,
            elements: &elements,
        };
        let mut values = Vec::with_capacity(element_indices.len());
        for (&element_index, &reference) in element_indices.iter().zip(reference_points) {
            let nodes = mesh.element_nodes(element_index)?;
            let material_id = self.assembly.material_ids[element_index];
            let material = self
                .assembly
                .material_table
                .get(material_id)
                .ok_or_else(|| {
                    format!("material_id {material_id} on element {element_index} is out of range")
                })?;
            let thermal = thermal_material_table.get(material_id).ok_or_else(|| {
                format!(
                    "thermal material_id {material_id} on element {element_index} is out of range"
                )
            })?;
            let material_storage;
            let thermal_storage;
            let (material, thermal) = if let Some(angles) =
                self.assembly.material_orientation_angles.as_ref()
            {
                material_storage = rotate_material_in_plane(material, angles[element_index]);
                thermal_storage = rotate_thermal_material_in_plane(thermal, angles[element_index]);
                (&material_storage, &thermal_storage)
            } else {
                (material, thermal)
            };
            let thermal_stress_unit =
                STRESS.then(|| constitutive_times_strain(material, &thermal.alpha));

            let shape = Family::ReferenceElement::shape(reference[0], reference[1]);
            let mut temperature_delta = -thermal.reference_temperature;
            for local_node in 0..NODES_PER_ELEMENT {
                temperature_delta += shape[local_node] * nodal_temperature[nodes[local_node]];
            }
            let mut sample_value = [0.0; 4];
            if let Some(thermal_stress_unit) = thermal_stress_unit.as_ref() {
                for component in 0..4 {
                    sample_value[component] = thermal_stress_unit[component] * temperature_delta;
                }
            } else {
                for component in 0..4 {
                    sample_value[component] = thermal.alpha[component] * temperature_delta;
                }
            }
            values.push(sample_value);
        }
        Ok(values)
    }

    fn locate_points_in_elements_for_family<Family, const NODES_PER_ELEMENT: usize>(
        &self,
        points: &[[f64; 2]],
        element_indices: &[usize],
        max_iterations: usize,
    ) -> Result<Structural2dPointLocations, String>
    where
        Family: QuadElementFamily<NODES_PER_ELEMENT>,
    {
        if points.len() != element_indices.len() {
            return Err(format!(
                "points has length {}, but element_indices has length {}",
                points.len(),
                element_indices.len()
            ));
        }
        let elements = self.analysis_elements::<NODES_PER_ELEMENT>()?;
        let mesh = QuadMeshView2d {
            nodes_rz: &self.analysis_nodes,
            elements: &elements,
        };
        let mut projected_points = Vec::with_capacity(points.len());
        let mut reference_points = Vec::with_capacity(points.len());
        for (&point, &element_index) in points.iter().zip(element_indices) {
            let coords = mesh.element_coords(element_index)?;
            let (reference, projected, _distance) = closest_reference_point::<
                Family::ReferenceElement,
                f64,
                NODES_PER_ELEMENT,
            >(&coords, point, max_iterations);
            projected_points.push(projected);
            reference_points.push(reference);
        }
        Ok(Structural2dPointLocations {
            points: projected_points,
            element_indices: element_indices.to_vec(),
            reference_points,
            weights_area: Vec::new(),
            weights_volume: Vec::new(),
            points_per_element: 0,
        })
    }

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
        body_force: Option<&[f64]>,
        pressure_values: Option<&[f64]>,
        traction_values: Option<&[f64]>,
        nodal_temperature: Option<&[f64]>,
    ) -> Result<Vec<f64>, String> {
        let mut rhs = self.constant_rhs.clone();
        self.apply_matrix_free_loads(
            body_force,
            pressure_values,
            traction_values,
            nodal_temperature,
            &mut rhs,
        )?;
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
    pub fn solve(&mut self, rhs: &[f64]) -> Result<Vec<f64>, String> {
        let reduced_solution = self.solve_direct_reduced(rhs)?;
        Ok(self.recover_full(&reduced_solution))
    }

    /// Solve the reduced structural system with the cached sparse LU factorization.
    fn solve_direct_reduced(&mut self, rhs: &[f64]) -> Result<Vec<f64>, String> {
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
        let mut reduced_solution = Col::<f64>::zeros(self.ndof_reduced);
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
    pub fn recover_full(&self, reduced_solution: &[f64]) -> Vec<f64> {
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

    /// Return element-major quadrature locations and mapped weights.
    pub fn quadrature(&self) -> Result<Structural2dPointLocations, String> {
        match self.element_type {
            Structural2dElementType::Quad4 => {
                quadrature_locations_for_family::<Quad4Family, { quad4::NODES_PER_ELEMENT }>(
                    &self.analysis_nodes,
                    &self.analysis_elements_flat,
                    self.nelem,
                    self.formulation,
                    self.quadrature,
                )
            }
            Structural2dElementType::Quad9 => {
                quadrature_locations_for_family::<Quad9Family, { quad9::NODES_PER_ELEMENT }>(
                    &self.analysis_nodes,
                    &self.analysis_elements_flat,
                    self.nelem,
                    self.formulation,
                    self.quadrature,
                )
            }
        }
    }

    /// Locate physical points when their owning element indices are already known.
    pub fn locate_points_in_elements(
        &self,
        points: &[[f64; 2]],
        element_indices: &[usize],
        max_iterations: usize,
    ) -> Result<Structural2dPointLocations, String> {
        match self.element_type {
            Structural2dElementType::Quad4 => self
                .locate_points_in_elements_for_family::<Quad4Family, { quad4::NODES_PER_ELEMENT }>(
                    points,
                    element_indices,
                    max_iterations,
                ),
            Structural2dElementType::Quad9 => self
                .locate_points_in_elements_for_family::<Quad9Family, { quad9::NODES_PER_ELEMENT }>(
                    points,
                    element_indices,
                    max_iterations,
                ),
        }
    }

    /// Return per-element analysis-plane area and represented volume from the model quadrature data.
    ///
    /// Returns:
    ///     Per-element measures with:
    ///     - `areas` shape `(nelem,)` and units `[area]`
    ///     - `volumes` shape `(nelem,)` and units `[volume]`
    pub fn element_measures(&self) -> Result<Structural2dElementMeasures, String> {
        let quadrature = self.quadrature()?;
        let mut areas = vec![0.0; self.nelem];
        let mut volumes = vec![0.0; self.nelem];
        for element in 0..self.nelem {
            let start = element * quadrature.points_per_element;
            let end = start + quadrature.points_per_element;
            for &weight in &quadrature.weights_area[start..end] {
                areas[element] += weight;
            }
            for &weight in &quadrature.weights_volume[start..end] {
                volumes[element] += weight;
            }
        }
        Ok(Structural2dElementMeasures { areas, volumes })
    }

    /// Evaluate total strain at located element/reference points without sparse recovery matrices.
    pub fn strain(
        &self,
        element_indices: &[usize],
        reference_points: &[[f64; 2]],
        displacements_full: &[f64],
    ) -> Result<Vec<[f64; 4]>, String> {
        if displacements_full.len() != self.ndof_full {
            return Err(format!(
                "displacements_full has length {}, but full system has {} DOFs",
                displacements_full.len(),
                self.ndof_full
            ));
        }
        match self.element_type {
            Structural2dElementType::Quad4 => self.evaluate_strain_for_locations_for_family::<
                Quad4Family,
                { quad4::NODES_PER_ELEMENT },
                { dof_per_element(quad4::NODES_PER_ELEMENT) },
            >(element_indices, reference_points, displacements_full),
            Structural2dElementType::Quad9 => self.evaluate_strain_for_locations_for_family::<
                Quad9Family,
                { quad9::NODES_PER_ELEMENT },
                { dof_per_element(quad9::NODES_PER_ELEMENT) },
            >(element_indices, reference_points, displacements_full),
        }
    }

    /// Evaluate stress at located element/reference points without sparse recovery matrices.
    pub fn stress(
        &self,
        element_indices: &[usize],
        reference_points: &[[f64; 2]],
        displacements_full: &[f64],
    ) -> Result<Vec<[f64; 4]>, String> {
        if displacements_full.len() != self.ndof_full {
            return Err(format!(
                "displacements_full has length {}, but full system has {} DOFs",
                displacements_full.len(),
                self.ndof_full
            ));
        }
        match self.element_type {
            Structural2dElementType::Quad4 => self.evaluate_stress_for_locations_for_family::<
                Quad4Family,
                { quad4::NODES_PER_ELEMENT },
                { dof_per_element(quad4::NODES_PER_ELEMENT) },
            >(element_indices, reference_points, displacements_full),
            Structural2dElementType::Quad9 => self.evaluate_stress_for_locations_for_family::<
                Quad9Family,
                { quad9::NODES_PER_ELEMENT },
                { dof_per_element(quad9::NODES_PER_ELEMENT) },
            >(element_indices, reference_points, displacements_full),
        }
    }

    /// Evaluate thermal strain at located element/reference points without sparse recovery matrices.
    pub fn thermal_strain(
        &self,
        element_indices: &[usize],
        reference_points: &[[f64; 2]],
        nodal_temperature: &[f64],
    ) -> Result<Vec<[f64; 4]>, String> {
        match self.element_type {
            Structural2dElementType::Quad4 => self.evaluate_thermal_for_locations_for_family::<
                Quad4Family,
                { quad4::NODES_PER_ELEMENT },
                false,
            >(element_indices, reference_points, nodal_temperature),
            Structural2dElementType::Quad9 => self.evaluate_thermal_for_locations_for_family::<
                Quad9Family,
                { quad9::NODES_PER_ELEMENT },
                false,
            >(element_indices, reference_points, nodal_temperature),
        }
    }

    /// Evaluate thermal stress at located element/reference points without sparse recovery matrices.
    pub fn thermal_stress(
        &self,
        element_indices: &[usize],
        reference_points: &[[f64; 2]],
        nodal_temperature: &[f64],
    ) -> Result<Vec<[f64; 4]>, String> {
        match self.element_type {
            Structural2dElementType::Quad4 => self.evaluate_thermal_for_locations_for_family::<
                Quad4Family,
                { quad4::NODES_PER_ELEMENT },
                true,
            >(element_indices, reference_points, nodal_temperature),
            Structural2dElementType::Quad9 => self.evaluate_thermal_for_locations_for_family::<
                Quad9Family,
                { quad9::NODES_PER_ELEMENT },
                true,
            >(element_indices, reference_points, nodal_temperature),
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Assemble the reduced 2D structural model and all associated operators.
pub fn assemble_structural_2d(
    nodes_rz: &[[f64; 2]],
    elements: Structural2dElements<'_>,
    material_ids: &[usize],
    material_table: &[[[f64; 4]; 4]],
    pressure_faces: &[PressureLoad],
    traction_faces: &[TractionLoad],
    thermal_material_table: Option<&[ThermalMaterial]>,
    material_orientation_angles: Option<&[f64]>,
    prescribed: &[(usize, f64)],
    formulation: Structural2dFormulation,
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
type ReducedLayout = (
    Vec<usize>,
    Vec<usize>,
    Vec<f64>,
    Vec<usize>,
    Vec<Option<f64>>,
);

#[allow(clippy::too_many_arguments)]
/// Assemble the full set of reduced operators for one specific quadrilateral family.
///
/// This is the family-generic core of the public assembly path.  It builds the unreduced
/// operators, applies Dirichlet reduction, compresses the final sparse matrices, and packages the
/// result into the public `Structural2dModel`.
fn build_model_for_family<Family, const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    nodes_rz: &[[f64; 2]],
    elements: &[[usize; NODES_PER_ELEMENT]],
    material_ids: &[usize],
    material_table: &[[[f64; 4]; 4]],
    pressure_faces: &[PressureLoad],
    traction_faces: &[TractionLoad],
    thermal_material_table: Option<&[ThermalMaterial]>,
    material_orientation_angles: Option<&[f64]>,
    prescribed: &[(usize, f64)],
    formulation: Structural2dFormulation,
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
        let stiffness_chunks =
            assemble_stiffness_chunks_for_family_par::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
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
            assemble_stiffness_for_family::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
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
    let (thermal_reference_rhs, n_temperature_nodes) =
        if let Some(thermal_material_table) = thermal_material_table {
            let thermal_reference_full =
                thermal_reference_rhs_for_family::<Family, NODES_PER_ELEMENT, DOF_PER_ELEMENT>(
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
                .map(|&dof| thermal_reference_full[dof])
                .collect::<Vec<_>>();
            (reduced_reference_rhs, nodes_rz.len())
        } else {
            (vec![0.0; ndof_reduced], 0)
        };
    // `constant_rhs` already contains the Dirichlet offset from stiffness reduction. Add the
    // reference-temperature contribution so all load-independent terms live in one vector.
    for (dst, src) in constant_rhs.iter_mut().zip(&thermal_reference_rhs) {
        *dst = *dst + *src;
    }

    Ok(Structural2dModel {
        stiffness,
        constant_rhs,
        nq_per_element: quadrature.points_per_element(),
        n_temperature_nodes,
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
        assembly: Structural2dAssemblyData {
            material_ids: material_ids.to_vec(),
            material_table: material_table.to_vec(),
            thermal_material_table: thermal_material_table.map(|table| table.to_vec()),
            material_orientation_angles: material_orientation_angles.map(|angles| angles.to_vec()),
            global_to_reduced,
            par,
        },
        lu: None,
    })
}

/// Partition full-system displacement DOFs into free and fixed sets for Dirichlet reduction.
fn reduce_layout(ndof_full: usize, prescribed: &[(usize, f64)]) -> Result<ReducedLayout, String> {
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
fn reduce_square_triplets(
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
    global_to_reduced: &[usize],
    fixed_lookup: &[Option<f64>],
    constant_rhs: &mut [f64],
) -> Vec<Triplet<usize, usize, f64>> {
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
fn reduce_sort_stiffness_chunks(
    chunks: Vec<StiffnessTriplets>,
    global_to_reduced: &[usize],
    fixed_lookup: &[Option<f64>],
    constant_rhs: &mut [f64],
    ndof_reduced: usize,
) -> Vec<Vec<Triplet<usize, usize, f64>>> {
    let reduced_chunks = chunks
        .into_par_iter()
        .map(|chunk| {
            let mut local_rhs = vec![0.0; ndof_reduced];
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
fn sort_coalesce_triplets(
    mut triplets: Vec<Triplet<usize, usize, f64>>,
) -> Vec<Triplet<usize, usize, f64>> {
    triplets.sort_by(|a, b| a.col.cmp(&b.col).then_with(|| a.row.cmp(&b.row)));

    let mut coalesced = Vec::with_capacity(triplets.len());
    let mut pending: Option<Triplet<usize, usize, f64>> = None;
    for triplet in triplets {
        if triplet.val == 0.0 {
            continue;
        }
        match pending {
            Some(mut current) if current.col == triplet.col && current.row == triplet.row => {
                current.val = current.val + triplet.val;
                pending = Some(current);
            }
            Some(current) => {
                if current.val != 0.0 {
                    coalesced.push(current);
                }
                pending = Some(triplet);
            }
            None => pending = Some(triplet),
        }
    }
    if let Some(current) = pending
        && current.val != 0.0
    {
        coalesced.push(current);
    }
    coalesced
}

/// Drop rows belonging to fixed displacement DOFs from one full-system load operator.
fn reduce_row_operator(
    operator: SparseOperator,
    global_to_reduced: &[usize],
    nrow_reduced: usize,
) -> SparseOperator {
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
fn reduce_row_operator_to_csr(
    operator: SparseOperator,
    global_to_reduced: &[usize],
    nrow_reduced: usize,
) -> Result<SparseRowMat<usize, f64>, String> {
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
fn csr_from_parts(
    nrow: usize,
    ncol: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
) -> Result<SparseRowMat<usize, f64>, String> {
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
fn csc_from_triplets(
    nrow: usize,
    ncol: usize,
    mut triplets: Vec<Triplet<usize, usize, f64>>,
) -> SparseColMat<usize, f64> {
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

struct CscPartsBuilder {
    /// Number of matrix rows.
    nrow: usize,
    /// Number of matrix columns.
    ncol: usize,
    /// CSC column offsets under construction.
    col_ptr: Vec<usize>,
    /// Row index for each stored value.
    row_idx: Vec<usize>,
    /// Nonzero values in column-major CSC order.
    vals: Vec<f64>,
    /// First column whose pointer has not yet been finalized.
    next_col: usize,
    /// Most recent sorted entry, held back so duplicates can be coalesced before writing.
    pending: Option<Triplet<usize, usize, f64>>,
}

impl CscPartsBuilder {
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
    fn push_sorted(&mut self, triplet: Triplet<usize, usize, f64>) {
        debug_assert!(triplet.row < self.nrow);
        debug_assert!(triplet.col < self.ncol);
        if triplet.val == 0.0 {
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
    fn finish(mut self) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
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
    fn push_entry(&mut self, entry: Triplet<usize, usize, f64>) {
        while self.next_col < entry.col {
            self.col_ptr.push(self.row_idx.len());
            self.next_col += 1;
        }
        if entry.val != 0.0 {
            self.row_idx.push(entry.row);
            self.vals.push(entry.val);
        }
    }
}

/// Merge already sorted/coalesced triplet chunks into the CSC format used by sparse LU.
fn csc_from_sorted_triplet_chunks(
    nrow: usize,
    ncol: usize,
    chunks: Vec<Vec<Triplet<usize, usize, f64>>>,
) -> SparseColMat<usize, f64> {
    let (col_ptr, row_idx, vals) = merge_sorted_triplet_chunks_to_csc(nrow, ncol, &chunks);
    let symbolic = SymbolicSparseColMat::new_checked(nrow, ncol, col_ptr, None, row_idx);
    SparseColMat::new(symbolic, vals)
}

fn merge_sorted_triplet_chunks_to_csc(
    nrow: usize,
    ncol: usize,
    chunks: &[Vec<Triplet<usize, usize, f64>>],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
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
fn pack_sorted_triplets_to_csc(
    nrow: usize,
    ncol: usize,
    triplets: Vec<Triplet<usize, usize, f64>>,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut builder = CscPartsBuilder::new(nrow, ncol, triplets.len());
    for triplet in triplets {
        builder.push_sorted(triplet);
    }
    builder.finish()
}

/// Gather one element's node coordinates from the flattened analysis connectivity.
fn element_coords_from_flat<const NODES_PER_ELEMENT: usize>(
    analysis_nodes: &[[f64; 2]],
    analysis_elements_flat: &[usize],
    element_index: usize,
) -> Result<[[f64; 2]; NODES_PER_ELEMENT], String> {
    let start = element_index * NODES_PER_ELEMENT;
    let end = start + NODES_PER_ELEMENT;
    let conn = analysis_elements_flat
        .get(start..end)
        .ok_or_else(|| format!("element index {element_index} out of bounds"))?;
    let mut coords = [[0.0; 2]; NODES_PER_ELEMENT];
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

/// Reconstruct typed element connectivity from the stored flattened analysis connectivity.
fn elements_from_flat<const NODES_PER_ELEMENT: usize>(
    analysis_elements_flat: &[usize],
    nelem: usize,
    element_type: Structural2dElementType,
) -> Result<Vec<[usize; NODES_PER_ELEMENT]>, String> {
    let expected = nelem * NODES_PER_ELEMENT;
    if analysis_elements_flat.len() != expected {
        return Err(format!(
            "{} analysis elements with element type {} require {expected} flattened node entries, got {}",
            nelem,
            element_type.as_str(),
            analysis_elements_flat.len()
        ));
    }
    let mut elements = Vec::with_capacity(nelem);
    for chunk in analysis_elements_flat.chunks_exact(NODES_PER_ELEMENT) {
        let mut conn = [0usize; NODES_PER_ELEMENT];
        conn.copy_from_slice(chunk);
        elements.push(conn);
    }
    Ok(elements)
}

fn validate_location_lengths(
    element_indices: &[usize],
    reference_points: &[[f64; 2]],
) -> Result<(), String> {
    if element_indices.len() != reference_points.len() {
        return Err(format!(
            "element_indices has length {}, but reference_points has length {}",
            element_indices.len(),
            reference_points.len()
        ));
    }
    Ok(())
}

/// Recompute element-major quadrature locations and mapped weights for one stored element family.
fn quadrature_locations_for_family<Family, const NODES_PER_ELEMENT: usize>(
    analysis_nodes: &[[f64; 2]],
    analysis_elements_flat: &[usize],
    nelem: usize,
    formulation: Structural2dFormulation,
    quadrature: QuadratureRule,
) -> Result<Structural2dPointLocations, String>
where
    Family: QuadElementFamily<NODES_PER_ELEMENT>,
{
    let references = gauss_volume::<f64>(quadrature);
    let nq_per_element = references.len();
    let npoints = nelem * nq_per_element;
    let mut points = Vec::with_capacity(npoints);
    let mut element_indices = Vec::with_capacity(npoints);
    let mut reference_points = Vec::with_capacity(npoints);
    let mut weights_area = Vec::with_capacity(npoints);
    let mut weights_volume = Vec::with_capacity(npoints);
    for element_index in 0..nelem {
        let coords = element_coords_from_flat::<NODES_PER_ELEMENT>(
            analysis_nodes,
            analysis_elements_flat,
            element_index,
        )?;
        for &(reference, weight) in &references {
            let shape = Family::ReferenceElement::shape(reference[0], reference[1]);
            let grad_ref = Family::ReferenceElement::grad_ref(reference[0], reference[1]);
            let jac = mapping::jacobian(&coords, &grad_ref);
            let det_j = mapping::det_j(&jac);
            let point = mapping::map_point(&coords, &shape);
            let area_weight = det_j * weight;
            element_indices.push(element_index);
            reference_points.push(reference);
            points.push(point);
            weights_area.push(area_weight);
            weights_volume.push(formulation.volume_scale(point, det_j, weight)?);
        }
    }
    Ok(Structural2dPointLocations {
        points,
        element_indices,
        reference_points,
        weights_area,
        weights_volume,
        points_per_element: nq_per_element,
    })
}

#[cfg(test)]
mod tests {
    use super::{Structural2dElements, Structural2dModel, assemble_structural_2d, csr_from_parts};
    use crate::mesh::QuadMeshView2d;
    use crate::mesh::QuadratureRule;
    use crate::mesh::quad2d::{Quad4ReferenceElement, quad_mesh_strain_operator};
    use crate::physics::solenoid_stress::convenience::{
        isotropic_axisymmetric_material, isotropic_axisymmetric_thermal_material,
    };
    use crate::physics::solenoid_stress::test_utils::{SINGLE_QUAD4_ELEMENTS, SINGLE_QUAD4_NODES};
    use crate::physics::solenoid_stress::types::{
        PressureLoad, Structural2dFormulation, TractionLoad,
    };
    use faer::sparse::SparseRowMat;

    fn assert_allclose(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            let scale = actual.abs().max(expected.abs()).max(1.0);
            assert!(
                (actual - expected).abs() <= 1.0e-12 * scale,
                "actual {actual} != expected {expected}"
            );
        }
    }

    fn csr_matvec(operator: &SparseRowMat<usize, f64>, input: &[f64]) -> Vec<f64> {
        assert_eq!(operator.ncols(), input.len());
        let mut output = vec![0.0; operator.nrows()];
        for row in 0..operator.nrows() {
            for index in operator.row_ptr()[row]..operator.row_ptr()[row + 1] {
                output[row] += operator.val()[index] * input[operator.col_idx()[index]];
            }
        }
        output
    }

    fn add_assign(dst: &mut [f64], src: &[f64]) {
        assert_eq!(dst.len(), src.len());
        for (dst, src) in dst.iter_mut().zip(src) {
            *dst += *src;
        }
    }

    fn single_element_thermal_model() -> Structural2dModel {
        let material = isotropic_axisymmetric_material(200.0e9, 0.27);
        let thermal = isotropic_axisymmetric_thermal_material(1.2e-5, 293.15);
        assemble_structural_2d(
            &SINGLE_QUAD4_NODES,
            Structural2dElements::Quad4(&SINGLE_QUAD4_ELEMENTS),
            &[0],
            &[material],
            &[PressureLoad {
                element: 0,
                local_face: 1,
            }],
            &[TractionLoad {
                element: 0,
                local_face: 2,
            }],
            Some(&[thermal]),
            None,
            &[(0, 1.0e-6)],
            Structural2dFormulation::Axisymmetric,
            QuadratureRule::GaussLegendre3,
            false,
        )
        .expect("single-element model assembly should succeed")
    }

    #[test]
    fn build_rhs_matrix_free_matches_sparse_export_assembly() {
        let model = single_element_thermal_model();

        let body_force = [1.25e3, -2.5e3];
        let pressure = [3.0e5];
        let traction = [4.0e4, -1.5e4];
        let temperature = [296.0, 297.0, 298.0, 299.0];
        let matrix_free = model
            .build_rhs(
                Some(&body_force),
                Some(&pressure),
                Some(&traction),
                Some(&temperature),
            )
            .expect("matrix-free RHS should build");

        let mut sparse = model.constant_rhs.clone();
        add_assign(
            &mut sparse,
            &csr_matvec(
                &model.body_force_to_rhs().expect("body-force operator"),
                &body_force,
            ),
        );
        add_assign(
            &mut sparse,
            &csr_matvec(
                &model.pressure_to_rhs().expect("pressure operator"),
                &pressure,
            ),
        );
        add_assign(
            &mut sparse,
            &csr_matvec(
                &model.traction_to_rhs().expect("traction operator"),
                &traction,
            ),
        );
        add_assign(
            &mut sparse,
            &csr_matvec(
                &model.temperature_to_rhs().expect("temperature operator"),
                &temperature,
            ),
        );
        assert_allclose(&matrix_free, &sparse);
    }

    #[test]
    fn matrix_free_strain_matches_sparse_recovery_exports() {
        let model = single_element_thermal_model();
        let displacements_full = [
            1.0e-6, -2.0e-6, 2.0e-6, 1.0e-6, -1.5e-6, 2.5e-6, 3.0e-6, -3.5e-6,
        ];
        let locations = model.quadrature().expect("quadrature should evaluate");
        let matrix_free = model
            .strain(
                &locations.element_indices,
                &locations.reference_points,
                &displacements_full,
            )
            .expect("matrix-free strain should evaluate");

        let elements = model.analysis_elements::<4>().expect("analysis elements");
        let mesh = QuadMeshView2d {
            nodes_rz: &model.analysis_nodes,
            elements: &elements,
        };
        let strain_operator = quad_mesh_strain_operator::<Quad4ReferenceElement, 4, 8>(
            mesh,
            &locations.element_indices,
            &locations.reference_points,
            model.formulation,
        )
        .expect("location-based sparse strain operator should build");
        let strain_operator = csr_from_parts(
            strain_operator.nrow,
            strain_operator.ncol,
            strain_operator.rows,
            strain_operator.cols,
            strain_operator.vals,
        )
        .expect("strain operator should convert to CSR");
        let expected = csr_matvec(&strain_operator, &displacements_full);
        let actual = matrix_free
            .into_iter()
            .flat_map(|sample| sample.into_iter())
            .collect::<Vec<_>>();
        assert_allclose(&actual, &expected);
    }
}
