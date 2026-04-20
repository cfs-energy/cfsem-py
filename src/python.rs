use numpy::Element as NumpyElement;
use numpy::PyArray1;
use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray1};
use pyo3::create_exception;
use pyo3::exceptions;
use pyo3::prelude::*;
use std::fmt::Debug;

use crate::{math, mesh, physics};

create_exception!(cfsem, DimensionalityError, exceptions::PyValueError);

/// Errors from mismatch between python and rust
#[derive(Debug)]
#[allow(dead_code)]
enum PyInteropError {
    DimensionalityError { msg: String },
    ValueError { msg: String },
}

impl From<PyInteropError> for PyErr {
    fn from(val: PyInteropError) -> Self {
        match val {
            PyInteropError::DimensionalityError { msg } => DimensionalityError::new_err(msg),
            PyInteropError::ValueError { msg } => exceptions::PyValueError::new_err(msg),
        }
    }
}

fn split_xyz_array2(
    name: &str,
    arr: PyReadonlyArray2<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let arr = arr.as_array();
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyInteropError::DimensionalityError {
            msg: format!("{name} must have shape (n, 3)"),
        }
        .into());
    }

    let n = shape[0];
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        x.push(arr[[i, 0]]);
        y.push(arr[[i, 1]]);
        z.push(arr[[i, 2]]);
    }

    Ok((x, y, z))
}

fn split_triangle_index_array2(
    name: &str,
    arr: PyReadonlyArray2<i64>,
) -> PyResult<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let arr = arr.as_array();
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyInteropError::DimensionalityError {
            msg: format!("{name} must have shape (n, 3)"),
        }
        .into());
    }

    let n = shape[0];
    let mut i0 = Vec::with_capacity(n);
    let mut i1 = Vec::with_capacity(n);
    let mut i2 = Vec::with_capacity(n);
    for i in 0..n {
        let idx0 = arr[[i, 0]];
        let idx1 = arr[[i, 1]];
        let idx2 = arr[[i, 2]];
        if idx0 < 0 || idx1 < 0 || idx2 < 0 {
            return Err(PyInteropError::ValueError {
                msg: format!("{name} must contain nonnegative node indices"),
            }
            .into());
        }
        i0.push(idx0 as usize);
        i1.push(idx1 as usize);
        i2.push(idx2 as usize);
    }

    Ok((i0, i1, i2))
}

fn parse_triangle_quadrature(quad: &str) -> PyResult<physics::boundary_element::QuadratureKind> {
    match quad {
        "gl2" => Ok(physics::boundary_element::QuadratureKind::GaussLegendre2),
        "gl3" => Ok(physics::boundary_element::QuadratureKind::GaussLegendre3),
        "dunavant5" => Ok(physics::boundary_element::QuadratureKind::Dunavant5),
        _ => Err(PyInteropError::ValueError {
            msg: format!("Unsupported triangle quadrature rule: {quad}"),
        }
        .into()),
    }
}

fn read_axisym_nodes<F: physics::solenoid_stress::Real + NumpyElement>(
    name: &str,
    nodes: PyReadonlyArray2<'_, F>,
) -> PyResult<Vec<[F; 2]>> {
    let view = nodes.as_array();
    let shape = view.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(PyInteropError::DimensionalityError {
            msg: format!("{name} must have shape (nnode, 2)"),
        }
        .into());
    }
    let mut out = Vec::with_capacity(shape[0]);
    for row in view.rows() {
        out.push([row[0], row[1]]);
    }
    Ok(out)
}

fn read_axisym_elements<const NODES_PER_ELEMENT: usize>(
    name: &str,
    elements: PyReadonlyArray2<'_, u64>,
) -> PyResult<Vec<[usize; NODES_PER_ELEMENT]>> {
    let view = elements.as_array();
    let shape = view.shape();
    if shape.len() != 2 || shape[1] != NODES_PER_ELEMENT {
        return Err(PyInteropError::DimensionalityError {
            msg: format!("{name} must have shape (nelem, {NODES_PER_ELEMENT})"),
        }
        .into());
    }
    let mut out = Vec::with_capacity(shape[0]);
    for row in view.rows() {
        let mut entry = [0usize; NODES_PER_ELEMENT];
        for col in 0..NODES_PER_ELEMENT {
            entry[col] = usize::try_from(row[col]).map_err(|_| PyInteropError::ValueError {
                msg: format!("{name} contains an index that overflows usize"),
            })?;
        }
        out.push(entry);
    }
    Ok(out)
}

fn read_axisym_material_ids(
    name: &str,
    material_ids: PyReadonlyArray1<'_, u64>,
) -> PyResult<Vec<usize>> {
    let view = material_ids.as_array();
    let shape = view.shape();
    if shape.len() != 1 {
        return Err(PyInteropError::DimensionalityError {
            msg: format!("{name} must have shape (nelem,)"),
        }
        .into());
    }
    view.iter()
        .map(|&value| {
            usize::try_from(value).map_err(|_| PyInteropError::ValueError {
                msg: format!("{name} contains a value that overflows usize"),
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn read_axisym_material_table<F: physics::solenoid_stress::Real + NumpyElement>(
    name: &str,
    material_table: PyReadonlyArray3<'_, F>,
) -> PyResult<Vec<[[F; 4]; 4]>> {
    let view = material_table.as_array();
    let shape = view.shape();
    if shape.len() != 3 || shape[1] != 4 || shape[2] != 4 {
        return Err(PyInteropError::DimensionalityError {
            msg: format!("{name} must have shape (nmat, 4, 4)"),
        }
        .into());
    }
    let mut out = Vec::with_capacity(shape[0]);
    for matrix in view.outer_iter() {
        let mut entry = [[F::zero(); 4]; 4];
        for row in 0..4 {
            for col in 0..4 {
                entry[row][col] = matrix[[row, col]];
            }
        }
        out.push(entry);
    }
    Ok(out)
}

fn read_axisym_thermal_material_table<F: physics::solenoid_stress::Real + NumpyElement>(
    name: &str,
    thermal_material_table: PyReadonlyArray2<'_, F>,
) -> PyResult<Vec<physics::solenoid_stress::ThermalMaterial<F>>> {
    let view = thermal_material_table.as_array();
    let shape = view.shape();
    if shape.len() != 2 || shape[1] != 5 {
        return Err(PyInteropError::DimensionalityError {
            msg: format!("{name} must have shape (nmat, 5)"),
        }
        .into());
    }
    let mut out = Vec::with_capacity(shape[0]);
    for row in view.rows() {
        out.push(physics::solenoid_stress::ThermalMaterial {
            alpha: [row[0], row[1], row[2], row[3]],
            reference_temperature: row[4],
        });
    }
    Ok(out)
}

fn read_axisym_pressure_faces<F: physics::solenoid_stress::Real + NumpyElement>(
    pressure_faces: PyReadonlyArray2<'_, u64>,
) -> PyResult<Vec<physics::solenoid_stress::PressureLoad<F>>> {
    let faces_view = pressure_faces.as_array();
    let faces_shape = faces_view.shape();
    if faces_shape.len() != 2 || faces_shape[1] != 2 {
        return Err(PyInteropError::DimensionalityError {
            msg: "pressure_faces must have shape (nload, 2)".to_string(),
        }
        .into());
    }
    let mut out = Vec::with_capacity(faces_shape[0]);
    for row in faces_view.rows() {
        out.push(physics::solenoid_stress::PressureLoad {
            element: usize::try_from(row[0]).map_err(|_| PyInteropError::ValueError {
                msg: "pressure_faces element index overflowed usize".to_string(),
            })?,
            local_face: u8::try_from(row[1]).map_err(|_| PyInteropError::ValueError {
                msg: "pressure_faces local face overflowed u8".to_string(),
            })?,
            value: F::one(),
        });
    }
    Ok(out)
}

fn read_axisym_traction_faces<F: physics::solenoid_stress::Real + NumpyElement>(
    traction_faces: PyReadonlyArray2<'_, u64>,
) -> PyResult<Vec<physics::solenoid_stress::TractionLoad<F>>> {
    let faces_view = traction_faces.as_array();
    let faces_shape = faces_view.shape();
    if faces_shape.len() != 2 || faces_shape[1] != 2 {
        return Err(PyInteropError::DimensionalityError {
            msg: "traction_faces must have shape (nload, 2)".to_string(),
        }
        .into());
    }
    let mut out = Vec::with_capacity(faces_shape[0]);
    for row in faces_view.rows() {
        out.push(physics::solenoid_stress::TractionLoad {
            element: usize::try_from(row[0]).map_err(|_| PyInteropError::ValueError {
                msg: "traction_faces element index overflowed usize".to_string(),
            })?,
            local_face: u8::try_from(row[1]).map_err(|_| PyInteropError::ValueError {
                msg: "traction_faces local face overflowed u8".to_string(),
            })?,
            value: [F::zero(), F::zero()],
        });
    }
    Ok(out)
}

fn parse_solenoid_fem_quadrature(
    quadrature: u8,
) -> PyResult<physics::solenoid_stress::QuadratureRule> {
    physics::solenoid_stress::QuadratureRule::from_code(quadrature)
        .map_err(|msg| PyInteropError::ValueError { msg }.into())
}

fn flatten_axisym_points<F: physics::solenoid_stress::Real>(points: Vec<[F; 2]>) -> Vec<F> {
    let mut points_flat = Vec::with_capacity(points.len() * 2);
    for point in points {
        points_flat.push(point[0]);
        points_flat.push(point[1]);
    }
    points_flat
}

fn flatten_usize_pairs(pairs: &[[usize; 2]]) -> Vec<usize> {
    let mut out = Vec::with_capacity(pairs.len() * 2);
    for pair in pairs {
        out.push(pair[0]);
        out.push(pair[1]);
    }
    out
}

fn read_axisym_prescribed<F: physics::solenoid_stress::Real + NumpyElement>(
    prescribed_dofs: PyReadonlyArray1<'_, u64>,
    prescribed_values: PyReadonlyArray1<'_, F>,
) -> PyResult<Vec<(usize, F)>> {
    let dofs = prescribed_dofs.as_array();
    let values = prescribed_values.as_array();
    if dofs.ndim() != 1 {
        return Err(PyInteropError::DimensionalityError {
            msg: "prescribed_dofs must have shape (nfixed,)".to_string(),
        }
        .into());
    }
    if values.ndim() != 1 {
        return Err(PyInteropError::DimensionalityError {
            msg: "prescribed_values must have shape (nfixed,)".to_string(),
        }
        .into());
    }
    if dofs.len() != values.len() {
        return Err(PyInteropError::ValueError {
            msg: format!(
                "prescribed_dofs has length {}, but prescribed_values has length {}",
                dofs.len(),
                values.len()
            ),
        }
        .into());
    }
    dofs.iter()
        .zip(values.iter())
        .map(|(&dof, &value)| {
            usize::try_from(dof)
                .map(|dof| (dof, value))
                .map_err(|_| PyInteropError::ValueError {
                    msg: "prescribed_dofs contains a value that overflows usize".to_string(),
                })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

#[pyclass(module = "cfsem", unsendable)]
struct SolenoidStressAxisymmetricModelF64 {
    inner: physics::solenoid_stress::AxisymmetricModel<f64>,
}

#[pyclass(module = "cfsem", unsendable)]
struct SolenoidStressAxisymmetricModelF32 {
    inner: physics::solenoid_stress::AxisymmetricModel<f32>,
}

macro_rules! impl_solenoid_stress_model_pyclass {
    ($name:ident, $ty:ty) => {
        #[pymethods]
        impl $name {
            #[getter]
            fn ndof_full(&self) -> usize {
                self.inner.ndof_full
            }

            #[getter]
            fn ndof_reduced(&self) -> usize {
                self.inner.ndof_reduced
            }

            #[getter]
            fn nelem(&self) -> usize {
                self.inner.nelem
            }

            #[getter]
            fn n_temperature_nodes(&self) -> usize {
                self.inner.recovery.n_temperature_nodes
            }

            #[getter]
            fn nq_per_element(&self) -> usize {
                self.inner.recovery.nq_per_element
            }

            #[getter]
            fn nodes_per_element(&self) -> usize {
                self.inner.nodes_per_element
            }

            #[getter]
            fn element_type(&self) -> String {
                self.inner.element_type.as_str().to_string()
            }

            fn analysis_nodes_flat<'py>(&self, py: Python<'py>) -> Py<PyArray1<$ty>> {
                PyArray1::from_vec(py, flatten_axisym_points(self.inner.analysis_nodes.clone())).unbind()
            }

            fn analysis_elements_flat<'py>(&self, py: Python<'py>) -> Py<PyArray1<usize>> {
                PyArray1::from_vec(py, self.inner.analysis_elements_flat.clone()).unbind()
            }

            fn pressure_faces_flat<'py>(&self, py: Python<'py>) -> Py<PyArray1<usize>> {
                PyArray1::from_vec(py, flatten_usize_pairs(&self.inner.pressure_faces)).unbind()
            }

            fn traction_faces_flat<'py>(&self, py: Python<'py>) -> Py<PyArray1<usize>> {
                PyArray1::from_vec(py, flatten_usize_pairs(&self.inner.traction_faces)).unbind()
            }

            fn free_dofs<'py>(&self, py: Python<'py>) -> Py<PyArray1<usize>> {
                PyArray1::from_vec(py, self.inner.free_dofs.clone()).unbind()
            }

            fn fixed_dofs<'py>(&self, py: Python<'py>) -> Py<PyArray1<usize>> {
                PyArray1::from_vec(py, self.inner.fixed_dofs.clone()).unbind()
            }

            fn fixed_values<'py>(&self, py: Python<'py>) -> Py<PyArray1<$ty>> {
                PyArray1::from_vec(py, self.inner.fixed_values.clone()).unbind()
            }

            fn constant_rhs<'py>(&self, py: Python<'py>) -> Py<PyArray1<$ty>> {
                PyArray1::from_vec(py, self.inner.constant_rhs.clone()).unbind()
            }

            fn quadrature_points_flat<'py>(&self, py: Python<'py>) -> Py<PyArray1<$ty>> {
                PyArray1::from_vec(py, flatten_axisym_points(self.inner.recovery.points_rz.clone()))
                    .unbind()
            }

            fn strain_constant<'py>(&self, py: Python<'py>) -> Py<PyArray1<$ty>> {
                PyArray1::from_vec(py, self.inner.recovery.strain_constant.clone()).unbind()
            }

            fn stress_constant<'py>(&self, py: Python<'py>) -> Py<PyArray1<$ty>> {
                PyArray1::from_vec(py, self.inner.recovery.stress_constant.clone()).unbind()
            }

            fn thermal_strain_constant<'py>(&self, py: Python<'py>) -> Py<PyArray1<$ty>> {
                PyArray1::from_vec(py, self.inner.recovery.thermal_strain_constant.clone())
                    .unbind()
            }

            fn thermal_stress_constant<'py>(&self, py: Python<'py>) -> Py<PyArray1<$ty>> {
                PyArray1::from_vec(py, self.inner.recovery.thermal_stress_constant.clone())
                    .unbind()
            }

            fn stiffness_csc<'py>(
                &self,
                py: Python<'py>,
            ) -> (
                Py<PyArray1<$ty>>,
                Py<PyArray1<usize>>,
                Py<PyArray1<usize>>,
                usize,
                usize,
            ) {
                let stiffness = &self.inner.stiffness;
                (
                    PyArray1::from_vec(py, stiffness.val().to_vec()).unbind(),
                    PyArray1::from_vec(py, stiffness.row_idx().to_vec()).unbind(),
                    PyArray1::from_vec(py, stiffness.col_ptr().to_vec()).unbind(),
                    stiffness.nrows(),
                    stiffness.ncols(),
                )
            }

            fn body_force_to_rhs_csr<'py>(
                &self,
                py: Python<'py>,
            ) -> (
                Py<PyArray1<$ty>>,
                Py<PyArray1<usize>>,
                Py<PyArray1<usize>>,
                usize,
                usize,
            ) {
                let operator = &self.inner.body_force_to_rhs;
                (
                    PyArray1::from_vec(py, operator.val().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.col_idx().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.row_ptr().to_vec()).unbind(),
                    operator.nrows(),
                    operator.ncols(),
                )
            }

            fn pressure_to_rhs_csr<'py>(
                &self,
                py: Python<'py>,
            ) -> (
                Py<PyArray1<$ty>>,
                Py<PyArray1<usize>>,
                Py<PyArray1<usize>>,
                usize,
                usize,
            ) {
                let operator = &self.inner.pressure_to_rhs;
                (
                    PyArray1::from_vec(py, operator.val().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.col_idx().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.row_ptr().to_vec()).unbind(),
                    operator.nrows(),
                    operator.ncols(),
                )
            }

            fn traction_to_rhs_csr<'py>(
                &self,
                py: Python<'py>,
            ) -> (
                Py<PyArray1<$ty>>,
                Py<PyArray1<usize>>,
                Py<PyArray1<usize>>,
                usize,
                usize,
            ) {
                let operator = &self.inner.traction_to_rhs;
                (
                    PyArray1::from_vec(py, operator.val().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.col_idx().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.row_ptr().to_vec()).unbind(),
                    operator.nrows(),
                    operator.ncols(),
                )
            }

            fn temperature_to_rhs_csr<'py>(
                &self,
                py: Python<'py>,
            ) -> (
                Py<PyArray1<$ty>>,
                Py<PyArray1<usize>>,
                Py<PyArray1<usize>>,
                usize,
                usize,
            ) {
                let operator = &self.inner.temperature_to_rhs;
                (
                    PyArray1::from_vec(py, operator.val().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.col_idx().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.row_ptr().to_vec()).unbind(),
                    operator.nrows(),
                    operator.ncols(),
                )
            }

            fn strain_operator_csr<'py>(
                &self,
                py: Python<'py>,
            ) -> (
                Py<PyArray1<$ty>>,
                Py<PyArray1<usize>>,
                Py<PyArray1<usize>>,
                usize,
                usize,
            ) {
                let operator = &self.inner.recovery.strain_operator;
                (
                    PyArray1::from_vec(py, operator.val().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.col_idx().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.row_ptr().to_vec()).unbind(),
                    operator.nrows(),
                    operator.ncols(),
                )
            }

            fn stress_operator_csr<'py>(
                &self,
                py: Python<'py>,
            ) -> (
                Py<PyArray1<$ty>>,
                Py<PyArray1<usize>>,
                Py<PyArray1<usize>>,
                usize,
                usize,
            ) {
                let operator = &self.inner.recovery.stress_operator;
                (
                    PyArray1::from_vec(py, operator.val().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.col_idx().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.row_ptr().to_vec()).unbind(),
                    operator.nrows(),
                    operator.ncols(),
                )
            }

            fn thermal_strain_operator_csr<'py>(
                &self,
                py: Python<'py>,
            ) -> (
                Py<PyArray1<$ty>>,
                Py<PyArray1<usize>>,
                Py<PyArray1<usize>>,
                usize,
                usize,
            ) {
                let operator = &self.inner.recovery.thermal_strain_operator;
                (
                    PyArray1::from_vec(py, operator.val().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.col_idx().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.row_ptr().to_vec()).unbind(),
                    operator.nrows(),
                    operator.ncols(),
                )
            }

            fn thermal_stress_operator_csr<'py>(
                &self,
                py: Python<'py>,
            ) -> (
                Py<PyArray1<$ty>>,
                Py<PyArray1<usize>>,
                Py<PyArray1<usize>>,
                usize,
                usize,
            ) {
                let operator = &self.inner.recovery.thermal_stress_operator;
                (
                    PyArray1::from_vec(py, operator.val().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.col_idx().to_vec()).unbind(),
                    PyArray1::from_vec(py, operator.row_ptr().to_vec()).unbind(),
                    operator.nrows(),
                    operator.ncols(),
                )
            }

            #[pyo3(signature = (body_force=None, pressure_values=None, traction_values=None, nodal_temperature=None))]
            fn build_rhs<'py>(
                &self,
                py: Python<'py>,
                body_force: Option<PyReadonlyArray1<'_, $ty>>,
                pressure_values: Option<PyReadonlyArray1<'_, $ty>>,
                traction_values: Option<PyReadonlyArray1<'_, $ty>>,
                nodal_temperature: Option<PyReadonlyArray1<'_, $ty>>,
            ) -> PyResult<Py<PyArray1<$ty>>> {
                let body_force = match &body_force {
                    Some(arr) => Some(arr.as_slice()?),
                    None => None,
                };
                let pressure_values = match &pressure_values {
                    Some(arr) => Some(arr.as_slice()?),
                    None => None,
                };
                let traction_values = match &traction_values {
                    Some(arr) => Some(arr.as_slice()?),
                    None => None,
                };
                let nodal_temperature = match &nodal_temperature {
                    Some(arr) => Some(arr.as_slice()?),
                    None => None,
                };
                let rhs = self
                    .inner
                    .build_rhs(body_force, pressure_values, traction_values, nodal_temperature)
                    .map_err(|msg| PyInteropError::ValueError { msg })?;
                Ok(PyArray1::from_vec(py, rhs).unbind())
            }

            fn solve<'py>(
                &mut self,
                py: Python<'py>,
                rhs: PyReadonlyArray1<'_, $ty>,
            ) -> PyResult<Py<PyArray1<$ty>>> {
                let rhs = rhs.as_slice()?;
                let solution = self
                    .inner
                    .solve(rhs)
                    .map_err(|msg| PyInteropError::ValueError { msg })?;
                Ok(PyArray1::from_vec(py, solution).unbind())
            }
        }
    };
}

impl_solenoid_stress_model_pyclass!(SolenoidStressAxisymmetricModelF64, f64);
impl_solenoid_stress_model_pyclass!(SolenoidStressAxisymmetricModelF32, f32);

fn assemble_axisymmetric_model_low_level<F: physics::solenoid_stress::Real + NumpyElement>(
    nodes: PyReadonlyArray2<'_, F>,
    elements: PyReadonlyArray2<'_, u64>,
    material_ids: PyReadonlyArray1<'_, u64>,
    material_table: PyReadonlyArray3<'_, F>,
    pressure_faces: PyReadonlyArray2<'_, u64>,
    traction_faces: PyReadonlyArray2<'_, u64>,
    thermal_material_table: PyReadonlyArray2<'_, F>,
    prescribed_dofs: PyReadonlyArray1<'_, u64>,
    prescribed_values: PyReadonlyArray1<'_, F>,
    element_type: u8,
    quadrature: u8,
) -> PyResult<physics::solenoid_stress::AxisymmetricModel<F>> {
    let quadrature = parse_solenoid_fem_quadrature(quadrature)?;
    let element_type = physics::solenoid_stress::AxisymmetricElementType::from_code(element_type)
        .map_err(|msg| PyInteropError::ValueError { msg })?;
    let nodes = read_axisym_nodes("nodes", nodes)?;
    let material_ids = read_axisym_material_ids("material_ids", material_ids)?;
    let material_table = read_axisym_material_table("material_table", material_table)?;
    let pressure_faces = read_axisym_pressure_faces::<F>(pressure_faces)?;
    let traction_faces = read_axisym_traction_faces::<F>(traction_faces)?;
    let thermal_material_table =
        read_axisym_thermal_material_table("thermal_material_table", thermal_material_table)?;
    let prescribed = read_axisym_prescribed(prescribed_dofs, prescribed_values)?;
    let thermal_material_table =
        (!thermal_material_table.is_empty()).then_some(thermal_material_table.as_slice());
    match element_type {
        physics::solenoid_stress::AxisymmetricElementType::Quad4 => {
            let elements = read_axisym_elements::<4>("elements", elements)?;
            physics::solenoid_stress::assemble_axisymmetric(
                &nodes,
                physics::solenoid_stress::AxisymmetricElements::Quad4(&elements),
                &material_ids,
                &material_table,
                &pressure_faces,
                &traction_faces,
                thermal_material_table,
                &prescribed,
                quadrature,
            )
            .map_err(|msg| PyInteropError::ValueError { msg }.into())
        }
        physics::solenoid_stress::AxisymmetricElementType::Quad9 => {
            let elements = read_axisym_elements::<9>("elements", elements)?;
            physics::solenoid_stress::assemble_axisymmetric(
                &nodes,
                physics::solenoid_stress::AxisymmetricElements::Quad9(&elements),
                &material_ids,
                &material_table,
                &pressure_faces,
                &traction_faces,
                thermal_material_table,
                &prescribed,
                quadrature,
            )
            .map_err(|msg| PyInteropError::ValueError { msg }.into())
        }
    }
}

#[pyfunction]
fn solenoid_stress_fem_assemble_model_axisymmetric_f64(
    nodes: PyReadonlyArray2<'_, f64>,
    elements: PyReadonlyArray2<'_, u64>,
    material_ids: PyReadonlyArray1<'_, u64>,
    material_table: PyReadonlyArray3<'_, f64>,
    pressure_faces: PyReadonlyArray2<'_, u64>,
    traction_faces: PyReadonlyArray2<'_, u64>,
    thermal_material_table: PyReadonlyArray2<'_, f64>,
    prescribed_dofs: PyReadonlyArray1<'_, u64>,
    prescribed_values: PyReadonlyArray1<'_, f64>,
    element_type: u8,
    quadrature: u8,
) -> PyResult<SolenoidStressAxisymmetricModelF64> {
    Ok(SolenoidStressAxisymmetricModelF64 {
        inner: assemble_axisymmetric_model_low_level::<f64>(
            nodes,
            elements,
            material_ids,
            material_table,
            pressure_faces,
            traction_faces,
            thermal_material_table,
            prescribed_dofs,
            prescribed_values,
            element_type,
            quadrature,
        )?,
    })
}

#[pyfunction]
fn solenoid_stress_fem_assemble_model_axisymmetric_f32(
    nodes: PyReadonlyArray2<'_, f32>,
    elements: PyReadonlyArray2<'_, u64>,
    material_ids: PyReadonlyArray1<'_, u64>,
    material_table: PyReadonlyArray3<'_, f32>,
    pressure_faces: PyReadonlyArray2<'_, u64>,
    traction_faces: PyReadonlyArray2<'_, u64>,
    thermal_material_table: PyReadonlyArray2<'_, f32>,
    prescribed_dofs: PyReadonlyArray1<'_, u64>,
    prescribed_values: PyReadonlyArray1<'_, f32>,
    element_type: u8,
    quadrature: u8,
) -> PyResult<SolenoidStressAxisymmetricModelF32> {
    Ok(SolenoidStressAxisymmetricModelF32 {
        inner: assemble_axisymmetric_model_low_level::<f32>(
            nodes,
            elements,
            material_ids,
            material_table,
            pressure_faces,
            traction_faces,
            thermal_material_table,
            prescribed_dofs,
            prescribed_values,
            element_type,
            quadrature,
        )?,
    })
}

#[pyfunction]
fn filament_helix_path(
    path: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m]
    helix_start_offset: (f64, f64, f64),
    twist_pitch: f64,
    angle_offset: f64,
    mut out: (
        PyReadwriteArray1<f64>,
        PyReadwriteArray1<f64>,
        PyReadwriteArray1<f64>,
    ),
) -> PyResult<()> {
    // Unpack
    _3tup_slice_ro!(path);
    _3tup_slice_mut!(out);

    // Calculate
    match mesh::filament_helix_path(path, helix_start_offset, twist_pitch, angle_offset, out) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    Ok(())
}

#[pyfunction]
fn rotate_filaments_about_path(
    path: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m]
    angle_offset: f64,
    mut out: (
        PyReadwriteArray1<f64>,
        PyReadwriteArray1<f64>,
        PyReadwriteArray1<f64>,
    ),
) -> PyResult<()> {
    // Unpack
    _3tup_slice_ro!(path);
    _3tup_slice_mut!(out);

    // Calculate
    match mesh::rotate_filaments_about_path(path, angle_offset, out) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    Ok(())
}

/// Python bindings for cfsemrs::physics::flux_circular_filament
#[pyfunction]
fn flux_circular_filament(
    current: PyReadonlyArray1<f64>,
    rfil: PyReadonlyArray1<f64>,
    zfil: PyReadonlyArray1<f64>,
    rprime: PyReadonlyArray1<f64>,
    zprime: PyReadonlyArray1<f64>,
    par: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    let rzifil = (rfil, zfil, current);
    _3tup_slice_ro!(rzifil);
    let obs = (rprime, zprime);
    _2tup_slice_ro!(obs);

    // Initialize output
    let mut psi = vec![0.0; obs.0.len()];

    // Select variant
    let func = match par {
        true => physics::circular_filament::flux_circular_filament_par,
        false => physics::circular_filament::flux_circular_filament,
    };

    // Do calculations
    match func(rzifil, obs, &mut psi[..]) {
        Ok(_) => {}
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    // Acquire global interpreter lock, which will be released when it goes out of scope
    Python::attach(|py| {
        Ok(PyArray1::from_vec(py, psi).unbind()) // Make PyObject
    })
}

/// Python bindings for cfsemrs::physics::circular_filament::vector_potential_circular_filament
#[pyfunction]
fn vector_potential_circular_filament(
    current: PyReadonlyArray1<f64>,
    rfil: PyReadonlyArray1<f64>,
    zfil: PyReadonlyArray1<f64>,
    rprime: PyReadonlyArray1<f64>,
    zprime: PyReadonlyArray1<f64>,
    par: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    let rzifil = (rfil, zfil, current);
    _3tup_slice_ro!(rzifil);
    let obs = (rprime, zprime);
    _2tup_slice_ro!(obs);

    // Initialize output
    let mut out = vec![0.0; obs.0.len()];

    // Select variant
    let func = match par {
        true => physics::circular_filament::vector_potential_circular_filament_par,
        false => physics::circular_filament::vector_potential_circular_filament,
    };

    // Do calculations
    match func(rzifil, obs, &mut out[..]) {
        Ok(_) => {}
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    // Acquire global interpreter lock, which will be released when it goes out of scope
    Python::attach(|py| {
        Ok(PyArray1::from_vec(py, out).unbind()) // Make PyObject
    })
}

/// Python bindings for cfsemrs::physics::flux_density_circular_filament
#[pyfunction]
fn flux_density_circular_filament(
    current: PyReadonlyArray1<f64>,
    rfil: PyReadonlyArray1<f64>,
    zfil: PyReadonlyArray1<f64>,
    rprime: PyReadonlyArray1<f64>,
    zprime: PyReadonlyArray1<f64>,
    par: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    let rzifil = (rfil, zfil, current);
    _3tup_slice_ro!(rzifil);
    let obs = (rprime, zprime);
    _2tup_slice_ro!(obs);

    // Initialize output
    let n = obs.0.len();
    let (mut br, mut bz) = (vec![0.0; n], vec![0.0; n]);

    // Select variant
    let func = match par {
        true => physics::circular_filament::flux_density_circular_filament_par,
        false => physics::circular_filament::flux_density_circular_filament,
    };

    // Do calculations
    match func(rzifil, obs, (&mut br, &mut bz)) {
        Ok(_) => {}
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    // Acquire global interpreter lock, which will be released when it goes out of scope
    Python::attach(|py| {
        let br: Py<PyArray1<f64>> = PyArray1::from_slice(py, &br).unbind(); // Make PyObject
        let bz: Py<PyArray1<f64>> = PyArray1::from_slice(py, &bz).unbind(); // Make PyObject

        Ok((br, bz))
    })
}

/// Python bindings for cfsemrs::physics::linear_filament::flux_density_linear_filament
#[pyfunction(signature = (xyzp, xyzfil, dlxyzfil, ifil, wire_radius, par=true))]
fn flux_density_linear_filament(
    xyzp: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Test point coords
    xyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament origin coords (start of segment)
    dlxyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament length delta
    ifil: PyReadonlyArray1<f64>,        // [A] filament current
    wire_radius: PyReadonlyArray1<f64>, // [m] filament radius
    par: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    _3tup_slice_ro!(xyzp);
    _3tup_slice_ro!(xyzfil);
    _3tup_slice_ro!(dlxyzfil);
    let ifil = ifil.as_slice()?;
    let wire_radius = wire_radius.as_slice()?;

    // Do calculations
    let n = xyzp.0.len();
    let (mut bx, mut by, mut bz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);

    let func = match par {
        true => physics::linear_filament::flux_density_linear_filament_par,
        false => physics::linear_filament::flux_density_linear_filament,
    };
    match func(
        xyzp,
        xyzfil,
        dlxyzfil,
        ifil,
        wire_radius,
        (&mut bx, &mut by, &mut bz),
    ) {
        Ok(x) => x,
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    _3tup_ret!((bx, f64), (by, f64), (bz, f64))
}

/// Python bindings for cfsemrs::physics::linear_filament::flux_density_linear_filament_matrix
#[pyfunction(signature = (xyzp, xyzfil, dlxyzfil, ifil, wire_radius, par=true))]
fn flux_density_linear_filament_matrix(
    xyzp: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Test point coords
    xyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament origin coords (start of segment)
    dlxyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament length delta
    ifil: PyReadonlyArray1<f64>,        // [A] filament current
    wire_radius: PyReadonlyArray1<f64>, // [m] filament radius
    par: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    _3tup_slice_ro!(xyzp);
    _3tup_slice_ro!(xyzfil);
    _3tup_slice_ro!(dlxyzfil);
    let ifil = ifil.as_slice()?;
    let wire_radius = wire_radius.as_slice()?;

    let nout = xyzp.0.len().checked_mul(xyzfil.0.len()).ok_or_else(|| {
        PyInteropError::DimensionalityError {
            msg: "Output size overflow in flux_density_linear_filament_matrix".to_string(),
        }
    })?;
    let (mut bx, mut by, mut bz) = (vec![0.0; nout], vec![0.0; nout], vec![0.0; nout]);

    let func = match par {
        true => physics::linear_filament::flux_density_linear_filament_matrix_par,
        false => physics::linear_filament::flux_density_linear_filament_matrix,
    };
    match func(
        xyzp,
        xyzfil,
        dlxyzfil,
        ifil,
        wire_radius,
        (&mut bx, &mut by, &mut bz),
    ) {
        Ok(x) => x,
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    _3tup_ret!((bx, f64), (by, f64), (bz, f64))
}

/// Python bindings for cfsemrs::physics::point_source::segment::flux_density_point_segment
#[pyfunction]
fn flux_density_point_segment(
    xyzp: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Test point coords
    xyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament origin coords (start of segment)
    dlxyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament length delta
    ifil: PyReadonlyArray1<f64>, // [A] filament current
    par: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    _3tup_slice_ro!(xyzp);
    _3tup_slice_ro!(xyzfil);
    _3tup_slice_ro!(dlxyzfil);
    let ifil = ifil.as_slice()?;

    let n = xyzp.0.len();
    let (mut bx, mut by, mut bz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);

    let func = match par {
        true => physics::point_source::segment::flux_density_point_segment_par,
        false => physics::point_source::segment::flux_density_point_segment,
    };
    match func(xyzp, xyzfil, dlxyzfil, ifil, (&mut bx, &mut by, &mut bz)) {
        Ok(x) => x,
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    _3tup_ret!((bx, f64), (by, f64), (bz, f64))
}

/// Python bindings for cfsemrs::physics::linear_filament::vector_potential_linear_filament
#[pyfunction(signature = (xyzp, xyzfil, dlxyzfil, ifil, wire_radius, par=true))]
fn vector_potential_linear_filament(
    xyzp: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Test point coords
    xyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament origin coords (start of segment)
    dlxyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament length delta
    ifil: PyReadonlyArray1<f64>,        // [A] filament current
    wire_radius: PyReadonlyArray1<f64>, // [m] filament radius
    par: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    _3tup_slice_ro!(xyzp);
    _3tup_slice_ro!(xyzfil);
    _3tup_slice_ro!(dlxyzfil);
    let ifil = ifil.as_slice()?;
    let wire_radius = wire_radius.as_slice()?;

    // Do calculations
    let n = xyzp.0.len();
    let (mut outx, mut outy, mut outz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);

    let func = match par {
        true => physics::linear_filament::vector_potential_linear_filament_par,
        false => physics::linear_filament::vector_potential_linear_filament,
    };
    match func(
        xyzp,
        xyzfil,
        dlxyzfil,
        ifil,
        wire_radius,
        (&mut outx, &mut outy, &mut outz),
    ) {
        Ok(x) => x,
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    _3tup_ret!((outx, f64), (outy, f64), (outz, f64))
}

/// Python bindings for cfsemrs::physics::linear_filament::vector_potential_linear_filament_matrix
#[pyfunction(signature = (xyzp, xyzfil, dlxyzfil, ifil, wire_radius, par=true))]
fn vector_potential_linear_filament_matrix(
    xyzp: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Test point coords
    xyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament origin coords (start of segment)
    dlxyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament length delta
    ifil: PyReadonlyArray1<f64>,        // [A] filament current
    wire_radius: PyReadonlyArray1<f64>, // [m] filament radius
    par: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    _3tup_slice_ro!(xyzp);
    _3tup_slice_ro!(xyzfil);
    _3tup_slice_ro!(dlxyzfil);
    let ifil = ifil.as_slice()?;
    let wire_radius = wire_radius.as_slice()?;

    let nout = xyzp.0.len().checked_mul(xyzfil.0.len()).ok_or_else(|| {
        PyInteropError::DimensionalityError {
            msg: "Output size overflow in vector_potential_linear_filament_matrix".to_string(),
        }
    })?;
    let (mut outx, mut outy, mut outz) = (vec![0.0; nout], vec![0.0; nout], vec![0.0; nout]);

    let func = match par {
        true => physics::linear_filament::vector_potential_linear_filament_matrix_par,
        false => physics::linear_filament::vector_potential_linear_filament_matrix,
    };
    match func(
        xyzp,
        xyzfil,
        dlxyzfil,
        ifil,
        wire_radius,
        (&mut outx, &mut outy, &mut outz),
    ) {
        Ok(x) => x,
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    _3tup_ret!((outx, f64), (outy, f64), (outz, f64))
}

/// Python bindings for cfsemrs::physics::point_source::segment::vector_potential_point_segment
#[pyfunction]
fn vector_potential_point_segment(
    xyzp: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Test point coords
    xyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament origin coords (start of segment)
    dlxyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament length delta
    ifil: PyReadonlyArray1<f64>, // [A] filament current
    par: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    _3tup_slice_ro!(xyzp);
    _3tup_slice_ro!(xyzfil);
    _3tup_slice_ro!(dlxyzfil);
    let ifil = ifil.as_slice()?;

    let n = xyzp.0.len();
    let (mut outx, mut outy, mut outz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);

    let func = match par {
        true => physics::point_source::segment::vector_potential_point_segment_par,
        false => physics::point_source::segment::vector_potential_point_segment,
    };
    match func(
        xyzp,
        xyzfil,
        dlxyzfil,
        ifil,
        (&mut outx, &mut outy, &mut outz),
    ) {
        Ok(x) => x,
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    _3tup_ret!((outx, f64), (outy, f64), (outz, f64))
}

#[pyfunction(signature = (xyzfil0, dlxyzfil0, xyzfil1, dlxyzfil1, wire_radius))]
fn inductance_piecewise_linear_filaments(
    xyzfil0: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament origin coords (start of segment)
    dlxyzfil0: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament length delta
    xyzfil1: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament origin coords (start of segment)
    dlxyzfil1: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament length delta
    wire_radius: PyReadonlyArray1<f64>, // [m] Source filament radius
) -> PyResult<f64> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    _3tup_slice_ro!(xyzfil0);
    _3tup_slice_ro!(dlxyzfil0);
    _3tup_slice_ro!(xyzfil1);
    _3tup_slice_ro!(dlxyzfil1);
    let wire_radius = wire_radius.as_slice()?;

    // Do calculations
    let inductance = match physics::linear_filament::inductance_piecewise_linear_filaments(
        xyzfil0,
        dlxyzfil0,
        xyzfil1,
        dlxyzfil1,
        wire_radius,
    ) {
        Ok(x) => x,
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    Ok(inductance)
}

#[pyfunction(signature = (xyzfil_tgt, dlxyzfil_tgt, xyzfil_src, dlxyzfil_src, wire_radius_src))]
fn inductance_linear_filaments(
    py: Python<'_>,
    xyzfil_tgt: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] target filament origin coords
    dlxyzfil_tgt: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] target filament length delta
    xyzfil_src: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] source filament origin coords
    dlxyzfil_src: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] source filament length delta
    wire_radius_src: PyReadonlyArray1<f64>, // [m] source filament radius
) -> PyResult<Py<PyArray1<f64>>> {
    _3tup_slice_ro!(xyzfil_tgt);
    _3tup_slice_ro!(dlxyzfil_tgt);
    _3tup_slice_ro!(xyzfil_src);
    _3tup_slice_ro!(dlxyzfil_src);
    let wire_radius_src = wire_radius_src.as_slice()?;

    let ntgt = xyzfil_tgt.0.len();
    let mut out = vec![0.0; ntgt];
    match physics::linear_filament::inductance_linear_filaments(
        xyzfil_tgt,
        dlxyzfil_tgt,
        xyzfil_src,
        dlxyzfil_src,
        wire_radius_src,
        &mut out,
    ) {
        Ok(x) => x,
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    Ok(PyArray1::from_vec(py, out).unbind())
}

#[pyfunction(signature = (xyzfil_tgt, dlxyzfil_tgt, xyzfil_src, dlxyzfil_src, wire_radius_src, par=true))]
fn inductance_linear_filaments_matrix(
    py: Python<'_>,
    xyzfil_tgt: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] target filament origin coords
    dlxyzfil_tgt: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] target filament length delta
    xyzfil_src: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] source filament origin coords
    dlxyzfil_src: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] source filament length delta
    wire_radius_src: PyReadonlyArray1<f64>, // [m] source filament radius
    par: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    _3tup_slice_ro!(xyzfil_tgt);
    _3tup_slice_ro!(dlxyzfil_tgt);
    _3tup_slice_ro!(xyzfil_src);
    _3tup_slice_ro!(dlxyzfil_src);
    let wire_radius_src = wire_radius_src.as_slice()?;

    let nout = xyzfil_tgt
        .0
        .len()
        .checked_mul(xyzfil_src.0.len())
        .ok_or_else(|| PyInteropError::DimensionalityError {
            msg: "Output size overflow in inductance_linear_filaments_matrix".to_string(),
        })?;
    let mut out = vec![0.0; nout];
    let func = match par {
        true => physics::linear_filament::inductance_linear_filaments_matrix_par,
        false => physics::linear_filament::inductance_linear_filaments_matrix,
    };
    match func(
        xyzfil_tgt,
        dlxyzfil_tgt,
        xyzfil_src,
        dlxyzfil_src,
        wire_radius_src,
        &mut out,
    ) {
        Ok(x) => x,
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    Ok(PyArray1::from_vec(py, out).unbind())
}

/// Python bindings for cfsemrs::physics::gradshafranov::gs_operator_order2
#[pyfunction]
fn gs_operator_order2(
    rs: PyReadonlyArray1<f64>,
    zs: PyReadonlyArray1<f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<usize>>, Py<PyArray1<usize>>)> {
    // Process inputs
    let rs = rs.as_slice()?;
    let zs = zs.as_slice()?;

    // Do calculations
    let (vals, rows, cols) = physics::gradshafranov::gs_operator_order2(rs, zs);

    _3tup_ret!((vals, f64), (rows, usize), (cols, usize))
}

/// Python bindings for cfsemrs::physics::gradshafranov::gs_operator_order4
#[pyfunction]
fn gs_operator_order4(
    rs: PyReadonlyArray1<f64>,
    zs: PyReadonlyArray1<f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<usize>>, Py<PyArray1<usize>>)> {
    // Process inputs
    let rs = rs.as_slice()?;
    let zs = zs.as_slice()?;

    // Do calculations
    let (vals, rows, cols) = physics::gradshafranov::gs_operator_order4(rs, zs);

    _3tup_ret!((vals, f64), (rows, usize), (cols, usize))
}

/// Python bindings for cfsemrs::math::ellipe
#[pyfunction]
fn ellipe(x: f64) -> f64 {
    math::ellipe(x)
}

/// Python bindings for cfsemrs::math::ellipk
#[pyfunction]
fn ellipk(x: f64) -> f64 {
    math::ellipk(x)
}

/// Python bindings for cfsemrs::physics::flux_density_circular_filament_cartesian
#[pyfunction]
fn flux_density_circular_filament_cartesian(
    current: PyReadonlyArray1<f64>,
    rfil: PyReadonlyArray1<f64>,
    zfil: PyReadonlyArray1<f64>,
    xyzobs: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Observation point coords
    par: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    let rzifil = (rfil, zfil, current);
    _3tup_slice_ro!(rzifil);
    let (rfil, zfil, current) = rzifil;
    _3tup_slice_ro!(xyzobs);

    // Initialize output
    let n = xyzobs.0.len();
    let (mut bx, mut by, mut bz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);

    // Select variant
    let func = match par {
        true => physics::circular_filament::flux_density_circular_filament_cartesian_par,
        false => physics::circular_filament::flux_density_circular_filament_cartesian,
    };

    // Do calculations
    match func(
        (&rfil, &zfil, &current),
        xyzobs,
        (&mut bx, &mut by, &mut bz),
    ) {
        Ok(_) => {}
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    _3tup_ret!((bx, f64), (by, f64), (bz, f64))
}

/// Python bindings for cfsemrs::physics::mutual_inductance_circular_to_linear
#[pyfunction]
fn mutual_inductance_circular_to_linear(
    rfil: PyReadonlyArray1<f64>,
    zfil: PyReadonlyArray1<f64>,
    nfil: PyReadonlyArray1<f64>,
    xyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament origin coords (start of segment)
    dlxyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament length delta
    par: bool,
) -> PyResult<f64> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    let rznfil = (rfil, zfil, nfil);
    _3tup_slice_ro!(rznfil);
    _3tup_slice_ro!(xyzfil);
    _3tup_slice_ro!(dlxyzfil);

    // Select variant
    let func = match par {
        true => physics::circular_filament::mutual_inductance_circular_to_linear_par,
        false => physics::circular_filament::mutual_inductance_circular_to_linear,
    };

    // Do calculations
    let m = match func(rznfil, xyzfil, dlxyzfil) {
        Ok(x) => x,
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    Ok(m)
}

/// Python bindings for cfsemrs::physics::point_source::flux_density_dipole
#[pyfunction]
fn flux_density_dipole(
    loc: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] dipole locations in cartesian coordinates
    moment: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [A-m^2] dipole moment vector
    obs: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Observation point coords
    outer_radius: PyReadonlyArray1<f64>,
    par: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    _3tup_slice_ro!(loc);
    _3tup_slice_ro!(moment);
    _3tup_slice_ro!(obs);

    // Do calculations
    let n = obs.0.len();
    let (mut outx, mut outy, mut outz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);

    let func = match par {
        true => physics::point_source::flux_density_dipole_par,
        false => physics::point_source::flux_density_dipole,
    };
    match func(
        loc,
        moment,
        outer_radius.as_slice()?,
        obs,
        (&mut outx, &mut outy, &mut outz),
    ) {
        Ok(x) => x,
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    _3tup_ret!((outx, f64), (outy, f64), (outz, f64))
}

/// Python bindings for cfsemrs::physics::point_source::vector_potential_dipole
#[pyfunction]
fn vector_potential_dipole(
    loc: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] dipole locations in cartesian coordinates
    moment: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [A-m^2] dipole moment vector
    obs: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Observation point coords
    outer_radius: PyReadonlyArray1<f64>,
    par: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    _3tup_slice_ro!(loc);
    _3tup_slice_ro!(moment);
    _3tup_slice_ro!(obs);

    // Do calculations
    let n = obs.0.len();
    let (mut outx, mut outy, mut outz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);

    let func = match par {
        true => physics::point_source::vector_potential_dipole_par,
        false => physics::point_source::vector_potential_dipole,
    };
    match func(
        loc,
        moment,
        outer_radius.as_slice()?,
        obs,
        (&mut outx, &mut outy, &mut outz),
    ) {
        Ok(x) => x,
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    _3tup_ret!((outx, f64), (outy, f64), (outz, f64))
}

/// Python bindings for cfsemrs::physics::body_force_density_circular_filament_cartesian
#[pyfunction]
fn body_force_density_circular_filament_cartesian(
    current: PyReadonlyArray1<f64>,
    rfil: PyReadonlyArray1<f64>,
    zfil: PyReadonlyArray1<f64>,
    obs: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament origin coords (start of segment)
    j: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [A/m^2] current density at observation points
    par: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    let rzifil = (rfil, zfil, current);
    _3tup_slice_ro!(rzifil);
    let (rfil, zfil, current) = rzifil;
    _3tup_slice_ro!(obs);
    _3tup_slice_ro!(j);

    // Select variant
    let func = match par {
        true => physics::circular_filament::body_force_density_circular_filament_cartesian_par,
        false => physics::circular_filament::body_force_density_circular_filament_cartesian,
    };

    // Do calculations
    let n = obs.0.len();
    let (mut outx, mut outy, mut outz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let out = (&mut outx[..], &mut outy[..], &mut outz[..]);

    match func((&rfil, &zfil, &current), obs, j, out) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    _3tup_ret!((outx, f64), (outy, f64), (outz, f64))
}

/// Python bindings for cfsemrs::physics::body_force_density_linear_filament
#[pyfunction]
#[pyo3(signature = (xyzfil, dlxyzfil, ifil, obs, j, wire_radius, par=true))]
fn body_force_density_linear_filament(
    xyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament origin coords (start of segment)
    dlxyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament length delta
    ifil: PyReadonlyArray1<f64>, // [A] filament current
    obs: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [m] Filament origin coords (start of segment)
    j: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ), // [A/m^2] current density at observation points
    wire_radius: PyReadonlyArray1<f64>, // [m] filament radius
    par: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    _3tup_slice_ro!(xyzfil);
    _3tup_slice_ro!(dlxyzfil);
    let ifil = ifil.as_slice()?;
    _3tup_slice_ro!(obs);
    _3tup_slice_ro!(j);
    let wire_radius = wire_radius.as_slice()?;

    // Select variant
    let func = match par {
        true => physics::linear_filament::body_force_density_linear_filament_par,
        false => physics::linear_filament::body_force_density_linear_filament,
    };

    // Do calculations
    let n = obs.0.len();
    let (mut outx, mut outy, mut outz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let out = (&mut outx[..], &mut outy[..], &mut outz[..]);

    match func(xyzfil, dlxyzfil, ifil, wire_radius, obs, j, out) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    _3tup_ret!((outx, f64), (outy, f64), (outz, f64))
}

fn triangle_mesh_view<'a>(
    nodes: &'a (Vec<f64>, Vec<f64>, Vec<f64>),
    triangles: &'a (Vec<usize>, Vec<usize>, Vec<usize>),
) -> PyResult<mesh::TriangleMeshView<'a>> {
    mesh::TriangleMeshView::new(
        (&nodes.0, &nodes.1, &nodes.2),
        (&triangles.0, &triangles.1, &triangles.2),
    )
    .map_err(|msg| {
        PyInteropError::ValueError {
            msg: msg.to_string(),
        }
        .into()
    })
}

#[pyfunction(signature = (obs, nodes, triangles, s, par=true, quad="gl3"))]
fn flux_density_triangle_mesh(
    obs: PyReadonlyArray2<f64>,
    nodes: PyReadonlyArray2<f64>,
    triangles: PyReadonlyArray2<i64>,
    s: PyReadonlyArray1<f64>,
    par: bool,
    quad: &str,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let obs = split_xyz_array2("obs", obs)?;
    let nodes = split_xyz_array2("nodes", nodes)?;
    let triangles = split_triangle_index_array2("triangles", triangles)?;
    let mesh = triangle_mesh_view(&nodes, &triangles)?;
    let s = s.as_slice()?;
    let quad = parse_triangle_quadrature(&quad)?;

    let n = obs.0.len();
    let (mut bx, mut by, mut bz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);

    let func = match par {
        true => physics::boundary_element::flux_density_triangle_mesh_par,
        false => physics::boundary_element::flux_density_triangle_mesh,
    };
    match func(
        (&obs.0, &obs.1, &obs.2),
        &mesh,
        s,
        quad,
        (&mut bx, &mut by, &mut bz),
    ) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    _3tup_ret!((bx, f64), (by, f64), (bz, f64))
}

#[pyfunction(signature = (obs, nodes, triangles, s, par=true, quad="gl3"))]
fn vector_potential_triangle_mesh(
    obs: PyReadonlyArray2<f64>,
    nodes: PyReadonlyArray2<f64>,
    triangles: PyReadonlyArray2<i64>,
    s: PyReadonlyArray1<f64>,
    par: bool,
    quad: &str,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let obs = split_xyz_array2("obs", obs)?;
    let nodes = split_xyz_array2("nodes", nodes)?;
    let triangles = split_triangle_index_array2("triangles", triangles)?;
    let mesh = triangle_mesh_view(&nodes, &triangles)?;
    let s = s.as_slice()?;
    let quad = parse_triangle_quadrature(&quad)?;

    let n = obs.0.len();
    let (mut ax, mut ay, mut az) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);

    let func = match par {
        true => physics::boundary_element::vector_potential_triangle_mesh_par,
        false => physics::boundary_element::vector_potential_triangle_mesh,
    };
    match func(
        (&obs.0, &obs.1, &obs.2),
        &mesh,
        s,
        quad,
        (&mut ax, &mut ay, &mut az),
    ) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    _3tup_ret!((ax, f64), (ay, f64), (az, f64))
}

#[pyfunction(signature = (obs, nodes, triangles, par=true, quad="gl3"))]
fn flux_density_triangle_mesh_mapping(
    obs: PyReadonlyArray2<f64>,
    nodes: PyReadonlyArray2<f64>,
    triangles: PyReadonlyArray2<i64>,
    par: bool,
    quad: &str,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let obs = split_xyz_array2("obs", obs)?;
    let nodes = split_xyz_array2("nodes", nodes)?;
    let triangles = split_triangle_index_array2("triangles", triangles)?;
    let mesh = triangle_mesh_view(&nodes, &triangles)?;
    let quad = parse_triangle_quadrature(&quad)?;

    let nout =
        obs.0
            .len()
            .checked_mul(mesh.nnode())
            .ok_or(PyInteropError::DimensionalityError {
                msg: "Flux-density mapping size overflow".to_string(),
            })?;
    let (mut bx, mut by, mut bz) = (vec![0.0; nout], vec![0.0; nout], vec![0.0; nout]);

    let func = match par {
        true => physics::boundary_element::flux_density_triangle_mesh_mapping_par,
        false => physics::boundary_element::flux_density_triangle_mesh_mapping,
    };
    match func(
        (&obs.0, &obs.1, &obs.2),
        &mesh,
        quad,
        (&mut bx, &mut by, &mut bz),
    ) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    _3tup_ret!((bx, f64), (by, f64), (bz, f64))
}

#[pyfunction(signature = (obs, nodes, triangles, par=true, quad="gl3"))]
fn vector_potential_triangle_mesh_mapping(
    obs: PyReadonlyArray2<f64>,
    nodes: PyReadonlyArray2<f64>,
    triangles: PyReadonlyArray2<i64>,
    par: bool,
    quad: &str,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let obs = split_xyz_array2("obs", obs)?;
    let nodes = split_xyz_array2("nodes", nodes)?;
    let triangles = split_triangle_index_array2("triangles", triangles)?;
    let mesh = triangle_mesh_view(&nodes, &triangles)?;
    let quad = parse_triangle_quadrature(&quad)?;

    let nout =
        obs.0
            .len()
            .checked_mul(mesh.nnode())
            .ok_or(PyInteropError::DimensionalityError {
                msg: "Vector-potential mapping size overflow".to_string(),
            })?;
    let (mut ax, mut ay, mut az) = (vec![0.0; nout], vec![0.0; nout], vec![0.0; nout]);

    let func = match par {
        true => physics::boundary_element::vector_potential_triangle_mesh_mapping_par,
        false => physics::boundary_element::vector_potential_triangle_mesh_mapping,
    };
    match func(
        (&obs.0, &obs.1, &obs.2),
        &mesh,
        quad,
        (&mut ax, &mut ay, &mut az),
    ) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    _3tup_ret!((ax, f64), (ay, f64), (az, f64))
}

#[pyfunction]
fn triangle_mesh_current_density(
    nodes: PyReadonlyArray2<f64>,
    triangles: PyReadonlyArray2<i64>,
    s: PyReadonlyArray1<f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let nodes = split_xyz_array2("nodes", nodes)?;
    let triangles = split_triangle_index_array2("triangles", triangles)?;
    let mesh = triangle_mesh_view(&nodes, &triangles)?;
    let s = s.as_slice()?;

    let ntri = mesh.len();
    let (mut jx, mut jy, mut jz) = (vec![0.0; ntri], vec![0.0; ntri], vec![0.0; ntri]);

    match physics::boundary_element::triangle_mesh_current_density(
        &mesh,
        s,
        (&mut jx, &mut jy, &mut jz),
    ) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    _3tup_ret!((jx, f64), (jy, f64), (jz, f64))
}

#[pyfunction(signature = (nodes, triangles, quad="gl3"))]
fn triangle_mesh_quadrature_points(
    nodes: PyReadonlyArray2<f64>,
    triangles: PyReadonlyArray2<i64>,
    quad: &str,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    usize,
)> {
    let nodes = split_xyz_array2("nodes", nodes)?;
    let triangles = split_triangle_index_array2("triangles", triangles)?;
    let mesh = triangle_mesh_view(&nodes, &triangles)?;
    let quad = parse_triangle_quadrature(&quad)?;

    let ntri = mesh.len();
    let nqp = physics::boundary_element::triangle_quadrature_count(quad);
    let nout = ntri * nqp;
    let (mut xq, mut yq, mut zq, mut wq) = (
        vec![0.0; nout],
        vec![0.0; nout],
        vec![0.0; nout],
        vec![0.0; nout],
    );

    match physics::boundary_element::triangle_mesh_quadrature_points(
        &mesh,
        quad,
        (&mut xq, &mut yq, &mut zq),
        &mut wq,
    ) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    Python::attach(|py| {
        Ok((
            PyArray1::from_vec(py, xq).unbind(),
            PyArray1::from_vec(py, yq).unbind(),
            PyArray1::from_vec(py, zq).unbind(),
            PyArray1::from_vec(py, wq).unbind(),
            nqp,
        ))
    })
}

#[pyfunction(signature = (nodes, triangles, par=true, quad="gl3"))]
fn triangle_mesh_inductance_matrix(
    nodes: PyReadonlyArray2<f64>,
    triangles: PyReadonlyArray2<i64>,
    par: bool,
    quad: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let nodes = split_xyz_array2("nodes", nodes)?;
    let triangles = split_triangle_index_array2("triangles", triangles)?;
    let mesh = triangle_mesh_view(&nodes, &triangles)?;
    let quad = parse_triangle_quadrature(&quad)?;

    let nnode = mesh.nnode();
    let nout = nnode
        .checked_mul(nnode)
        .ok_or(PyInteropError::DimensionalityError {
            msg: "Inductance matrix size overflow".to_string(),
        })?;
    let mut out = vec![0.0; nout];

    let func = match par {
        true => physics::boundary_element::triangle_mesh_inductance_matrix_par,
        false => physics::boundary_element::triangle_mesh_inductance_matrix,
    };
    match func(&mesh, quad, &mut out) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    Python::attach(|py| Ok(PyArray1::from_vec(py, out).unbind()))
}

#[pyfunction(signature = (xyzfil, dlxyzfil, wire_radius, nodes_tgt, triangles_tgt, par=true, quad="gl3"))]
fn triangle_mesh_inductance_mapping_from_linear_filaments(
    xyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ),
    dlxyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ),
    wire_radius: PyReadonlyArray1<f64>,
    nodes_tgt: PyReadonlyArray2<f64>,
    triangles_tgt: PyReadonlyArray2<i64>,
    par: bool,
    quad: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    _3tup_slice_ro!(xyzfil);
    _3tup_slice_ro!(dlxyzfil);
    let wire_radius = wire_radius.as_slice()?;
    let nodes_tgt = split_xyz_array2("nodes_tgt", nodes_tgt)?;
    let triangles_tgt = split_triangle_index_array2("triangles_tgt", triangles_tgt)?;
    let mesh_tgt = triangle_mesh_view(&nodes_tgt, &triangles_tgt)?;
    let quad = parse_triangle_quadrature(&quad)?;

    let nout = mesh_tgt.nnode().checked_mul(xyzfil.0.len()).ok_or(
        PyInteropError::DimensionalityError {
            msg: "Inductance mapping size overflow".to_string(),
        },
    )?;
    let mut out = vec![0.0; nout];

    let func = match par {
        true => {
            physics::boundary_element::triangle_mesh_inductance_mapping_from_linear_filaments_par
        }
        false => physics::boundary_element::triangle_mesh_inductance_mapping_from_linear_filaments,
    };
    match func(xyzfil, dlxyzfil, wire_radius, &mesh_tgt, quad, &mut out) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    Python::attach(|py| Ok(PyArray1::from_vec(py, out).unbind()))
}

#[pyfunction(signature = (rfil, zfil, nodes_tgt, triangles_tgt, par=true, quad="gl3"))]
fn triangle_mesh_inductance_mapping_from_circular_filaments(
    rfil: PyReadonlyArray1<f64>,
    zfil: PyReadonlyArray1<f64>,
    nodes_tgt: PyReadonlyArray2<f64>,
    triangles_tgt: PyReadonlyArray2<i64>,
    par: bool,
    quad: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let rfil = rfil.as_slice()?;
    let zfil = zfil.as_slice()?;
    let nodes_tgt = split_xyz_array2("nodes_tgt", nodes_tgt)?;
    let triangles_tgt = split_triangle_index_array2("triangles_tgt", triangles_tgt)?;
    let mesh_tgt = triangle_mesh_view(&nodes_tgt, &triangles_tgt)?;
    let quad = parse_triangle_quadrature(&quad)?;

    let nout =
        mesh_tgt
            .nnode()
            .checked_mul(rfil.len())
            .ok_or(PyInteropError::DimensionalityError {
                msg: "Inductance mapping size overflow".to_string(),
            })?;
    let mut out = vec![0.0; nout];

    let func = match par {
        true => {
            physics::boundary_element::triangle_mesh_inductance_mapping_from_circular_filaments_par
        }
        false => {
            physics::boundary_element::triangle_mesh_inductance_mapping_from_circular_filaments
        }
    };
    match func(rfil, zfil, &mesh_tgt, quad, &mut out) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    Python::attach(|py| Ok(PyArray1::from_vec(py, out).unbind()))
}

#[pyfunction(signature = (loc, moment_dir, outer_radius, nodes_tgt, triangles_tgt, par=true, quad="gl3"))]
fn triangle_mesh_flux_linkage_mapping_from_dipoles(
    loc: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ),
    moment_dir: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ),
    outer_radius: PyReadonlyArray1<f64>,
    nodes_tgt: PyReadonlyArray2<f64>,
    triangles_tgt: PyReadonlyArray2<i64>,
    par: bool,
    quad: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    _3tup_slice_ro!(loc);
    _3tup_slice_ro!(moment_dir);
    let outer_radius = outer_radius.as_slice()?;
    let nodes_tgt = split_xyz_array2("nodes_tgt", nodes_tgt)?;
    let triangles_tgt = split_triangle_index_array2("triangles_tgt", triangles_tgt)?;
    let mesh_tgt = triangle_mesh_view(&nodes_tgt, &triangles_tgt)?;
    let quad = parse_triangle_quadrature(&quad)?;

    let nout =
        mesh_tgt
            .nnode()
            .checked_mul(loc.0.len())
            .ok_or(PyInteropError::DimensionalityError {
                msg: "Flux-linkage mapping size overflow".to_string(),
            })?;
    let mut out = vec![0.0; nout];

    let func = match par {
        true => physics::boundary_element::triangle_mesh_flux_linkage_mapping_from_dipoles_par,
        false => physics::boundary_element::triangle_mesh_flux_linkage_mapping_from_dipoles,
    };
    match func(loc, moment_dir, outer_radius, &mesh_tgt, quad, &mut out) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    Python::attach(|py| Ok(PyArray1::from_vec(py, out).unbind()))
}

#[pyfunction(signature = (nodes_src, triangles_src, nodes_tgt, triangles_tgt, s_tgt, par=true, quad="gl3"))]
fn triangle_mesh_force_mapping(
    nodes_src: PyReadonlyArray2<f64>,
    triangles_src: PyReadonlyArray2<i64>,
    nodes_tgt: PyReadonlyArray2<f64>,
    triangles_tgt: PyReadonlyArray2<i64>,
    s_tgt: PyReadonlyArray1<f64>,
    par: bool,
    quad: &str,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let nodes_src = split_xyz_array2("nodes_src", nodes_src)?;
    let triangles_src = split_triangle_index_array2("triangles_src", triangles_src)?;
    let mesh_src = triangle_mesh_view(&nodes_src, &triangles_src)?;
    let nodes_tgt = split_xyz_array2("nodes_tgt", nodes_tgt)?;
    let triangles_tgt = split_triangle_index_array2("triangles_tgt", triangles_tgt)?;
    let mesh_tgt = triangle_mesh_view(&nodes_tgt, &triangles_tgt)?;
    let s_tgt = s_tgt.as_slice()?;
    let quad = parse_triangle_quadrature(quad)?;

    let nnode_src = mesh_src.nnode();
    let ntri_tgt = mesh_tgt.len();
    let nout = nnode_src
        .checked_mul(ntri_tgt)
        .ok_or(PyInteropError::DimensionalityError {
            msg: "Force mapping size overflow".to_string(),
        })?;
    let (mut fx, mut fy, mut fz) = (vec![0.0; nout], vec![0.0; nout], vec![0.0; nout]);

    let func = match par {
        true => physics::boundary_element::triangle_mesh_force_mapping_par,
        false => physics::boundary_element::triangle_mesh_force_mapping,
    };
    match func(
        &mesh_src,
        &mesh_tgt,
        s_tgt,
        quad,
        (&mut fx, &mut fy, &mut fz),
    ) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    _3tup_ret!((fx, f64), (fy, f64), (fz, f64))
}

#[pyfunction(signature = (nodes, triangles, s, par=true, quad="gl3"))]
fn triangle_mesh_self_force_mapping(
    nodes: PyReadonlyArray2<f64>,
    triangles: PyReadonlyArray2<i64>,
    s: PyReadonlyArray1<f64>,
    par: bool,
    quad: &str,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let nodes = split_xyz_array2("nodes", nodes)?;
    let triangles = split_triangle_index_array2("triangles", triangles)?;
    let mesh = triangle_mesh_view(&nodes, &triangles)?;
    let s = s.as_slice()?;
    let quad = parse_triangle_quadrature(quad)?;

    let nnode = mesh.nnode();
    let ntri = mesh.len();
    let nout = nnode
        .checked_mul(ntri)
        .ok_or(PyInteropError::DimensionalityError {
            msg: "Force mapping size overflow".to_string(),
        })?;
    let (mut fx, mut fy, mut fz) = (vec![0.0; nout], vec![0.0; nout], vec![0.0; nout]);

    let func = match par {
        true => physics::boundary_element::triangle_mesh_self_force_mapping_par,
        false => physics::boundary_element::triangle_mesh_self_force_mapping,
    };
    match func(&mesh, s, quad, (&mut fx, &mut fy, &mut fz)) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    _3tup_ret!((fx, f64), (fy, f64), (fz, f64))
}

#[pyfunction(signature = (xyzfil, dlxyzfil, wire_radius, nodes_tgt, triangles_tgt, s_tgt, par=true, quad="gl3"))]
fn triangle_mesh_force_mapping_from_linear_filaments(
    xyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ),
    dlxyzfil: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ),
    wire_radius: PyReadonlyArray1<f64>,
    nodes_tgt: PyReadonlyArray2<f64>,
    triangles_tgt: PyReadonlyArray2<i64>,
    s_tgt: PyReadonlyArray1<f64>,
    par: bool,
    quad: &str,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    _3tup_slice_ro!(xyzfil);
    _3tup_slice_ro!(dlxyzfil);
    let wire_radius = wire_radius.as_slice()?;
    let nodes_tgt = split_xyz_array2("nodes_tgt", nodes_tgt)?;
    let triangles_tgt = split_triangle_index_array2("triangles_tgt", triangles_tgt)?;
    let mesh_tgt = triangle_mesh_view(&nodes_tgt, &triangles_tgt)?;
    let s_tgt = s_tgt.as_slice()?;
    let quad = parse_triangle_quadrature(quad)?;

    let nfil = xyzfil.0.len();
    let ntri_tgt = mesh_tgt.len();
    let nout = nfil
        .checked_mul(ntri_tgt)
        .ok_or(PyInteropError::DimensionalityError {
            msg: "Force mapping size overflow".to_string(),
        })?;
    let (mut fx, mut fy, mut fz) = (vec![0.0; nout], vec![0.0; nout], vec![0.0; nout]);

    let func = match par {
        true => physics::boundary_element::triangle_mesh_force_mapping_from_linear_filaments_par,
        false => physics::boundary_element::triangle_mesh_force_mapping_from_linear_filaments,
    };
    match func(
        xyzfil,
        dlxyzfil,
        wire_radius,
        &mesh_tgt,
        s_tgt,
        quad,
        (&mut fx, &mut fy, &mut fz),
    ) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    _3tup_ret!((fx, f64), (fy, f64), (fz, f64))
}

#[pyfunction(signature = (rfil, zfil, nodes_tgt, triangles_tgt, s_tgt, par=true, quad="gl3"))]
fn triangle_mesh_force_mapping_from_circular_filaments(
    rfil: PyReadonlyArray1<f64>,
    zfil: PyReadonlyArray1<f64>,
    nodes_tgt: PyReadonlyArray2<f64>,
    triangles_tgt: PyReadonlyArray2<i64>,
    s_tgt: PyReadonlyArray1<f64>,
    par: bool,
    quad: &str,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let rfil = rfil.as_slice()?;
    let zfil = zfil.as_slice()?;
    let nodes_tgt = split_xyz_array2("nodes_tgt", nodes_tgt)?;
    let triangles_tgt = split_triangle_index_array2("triangles_tgt", triangles_tgt)?;
    let mesh_tgt = triangle_mesh_view(&nodes_tgt, &triangles_tgt)?;
    let s_tgt = s_tgt.as_slice()?;
    let quad = parse_triangle_quadrature(quad)?;

    let nfil = rfil.len();
    let ntri_tgt = mesh_tgt.len();
    let nout = nfil
        .checked_mul(ntri_tgt)
        .ok_or(PyInteropError::DimensionalityError {
            msg: "Force mapping size overflow".to_string(),
        })?;
    let (mut fx, mut fy, mut fz) = (vec![0.0; nout], vec![0.0; nout], vec![0.0; nout]);

    let func = match par {
        true => physics::boundary_element::triangle_mesh_force_mapping_from_circular_filaments_par,
        false => physics::boundary_element::triangle_mesh_force_mapping_from_circular_filaments,
    };
    match func(
        rfil,
        zfil,
        &mesh_tgt,
        s_tgt,
        quad,
        (&mut fx, &mut fy, &mut fz),
    ) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    _3tup_ret!((fx, f64), (fy, f64), (fz, f64))
}

#[pyfunction(signature = (loc, moment_dir, outer_radius, nodes_tgt, triangles_tgt, s_tgt, par=true, quad="gl3"))]
fn triangle_mesh_force_mapping_from_dipoles(
    loc: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ),
    moment_dir: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ),
    outer_radius: PyReadonlyArray1<f64>,
    nodes_tgt: PyReadonlyArray2<f64>,
    triangles_tgt: PyReadonlyArray2<i64>,
    s_tgt: PyReadonlyArray1<f64>,
    par: bool,
    quad: &str,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    _3tup_slice_ro!(loc);
    _3tup_slice_ro!(moment_dir);
    let outer_radius = outer_radius.as_slice()?;
    let nodes_tgt = split_xyz_array2("nodes_tgt", nodes_tgt)?;
    let triangles_tgt = split_triangle_index_array2("triangles_tgt", triangles_tgt)?;
    let mesh_tgt = triangle_mesh_view(&nodes_tgt, &triangles_tgt)?;
    let s_tgt = s_tgt.as_slice()?;
    let quad = parse_triangle_quadrature(quad)?;

    let ndip = loc.0.len();
    let ntri_tgt = mesh_tgt.len();
    let nout = ndip
        .checked_mul(ntri_tgt)
        .ok_or(PyInteropError::DimensionalityError {
            msg: "Force mapping size overflow".to_string(),
        })?;
    let (mut fx, mut fy, mut fz) = (vec![0.0; nout], vec![0.0; nout], vec![0.0; nout]);

    let func = match par {
        true => physics::boundary_element::triangle_mesh_force_mapping_from_dipoles_par,
        false => physics::boundary_element::triangle_mesh_force_mapping_from_dipoles,
    };
    match func(
        loc,
        moment_dir,
        outer_radius,
        &mesh_tgt,
        s_tgt,
        quad,
        (&mut fx, &mut fy, &mut fz),
    ) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    _3tup_ret!((fx, f64), (fy, f64), (fz, f64))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
#[pyo3(name = "cfsem")]
fn _cfsem<'py>(_py: Python, m: Bound<'py, PyModule>) -> PyResult<()> {
    m.add("DimensionalityError", _py.get_type::<DimensionalityError>())?;

    // Circular filaments
    m.add_function(wrap_pyfunction!(flux_circular_filament, m.clone())?)?;
    m.add_function(wrap_pyfunction!(flux_density_circular_filament, m.clone())?)?;
    m.add_function(wrap_pyfunction!(
        flux_density_circular_filament_cartesian,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        vector_potential_circular_filament,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        mutual_inductance_circular_to_linear,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        body_force_density_circular_filament_cartesian,
        m.clone()
    )?)?;

    // Linear filaments
    m.add_function(wrap_pyfunction!(flux_density_linear_filament, m.clone())?)?;
    m.add_function(wrap_pyfunction!(flux_density_triangle_mesh, m.clone())?)?;
    m.add_function(wrap_pyfunction!(flux_density_point_segment, m.clone())?)?;
    m.add_function(wrap_pyfunction!(triangle_mesh_current_density, m.clone())?)?;
    m.add_function(wrap_pyfunction!(
        triangle_mesh_quadrature_points,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(triangle_mesh_force_mapping, m.clone())?)?;
    m.add_function(wrap_pyfunction!(
        triangle_mesh_self_force_mapping,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        triangle_mesh_force_mapping_from_linear_filaments,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        triangle_mesh_force_mapping_from_circular_filaments,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        triangle_mesh_force_mapping_from_dipoles,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        triangle_mesh_inductance_matrix,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        triangle_mesh_inductance_mapping_from_linear_filaments,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        triangle_mesh_inductance_mapping_from_circular_filaments,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        triangle_mesh_flux_linkage_mapping_from_dipoles,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        flux_density_linear_filament_matrix,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        vector_potential_linear_filament,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(vector_potential_triangle_mesh, m.clone())?)?;
    m.add_function(wrap_pyfunction!(
        flux_density_triangle_mesh_mapping,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        vector_potential_triangle_mesh_mapping,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(vector_potential_point_segment, m.clone())?)?;
    m.add_function(wrap_pyfunction!(
        vector_potential_linear_filament_matrix,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        inductance_piecewise_linear_filaments,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(inductance_linear_filaments, m.clone())?)?;
    m.add_function(wrap_pyfunction!(
        inductance_linear_filaments_matrix,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        body_force_density_linear_filament,
        m.clone()
    )?)?;

    // Differential operators
    m.add_function(wrap_pyfunction!(gs_operator_order2, m.clone())?)?;
    m.add_function(wrap_pyfunction!(gs_operator_order4, m.clone())?)?;

    // Pure math
    m.add_function(wrap_pyfunction!(ellipe, m.clone())?)?;
    m.add_function(wrap_pyfunction!(ellipk, m.clone())?)?;

    // Filamentization and meshing
    m.add_function(wrap_pyfunction!(filament_helix_path, m.clone())?)?;
    m.add_function(wrap_pyfunction!(rotate_filaments_about_path, m.clone())?)?;

    // Point sources
    m.add_function(wrap_pyfunction!(flux_density_dipole, m.clone())?)?;
    m.add_function(wrap_pyfunction!(vector_potential_dipole, m.clone())?)?;

    // Solenoid stress FEM
    m.add_class::<SolenoidStressAxisymmetricModelF64>()?;
    m.add_class::<SolenoidStressAxisymmetricModelF32>()?;
    m.add_function(wrap_pyfunction!(
        solenoid_stress_fem_assemble_model_axisymmetric_f64,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        solenoid_stress_fem_assemble_model_axisymmetric_f32,
        m.clone()
    )?)?;

    Ok(())
}

/// Convert a 3-tuple of PyReadonlyArray to read-only slices, shadowing the original name
macro_rules! _3tup_slice_ro {
    ($x:ident) => {
        let $x = ($x.0.as_slice()?, $x.1.as_slice()?, $x.2.as_slice()?);
    };
}

/// Convert a 2-tuple of PyReadonlyArray to read-only slices, shadowing the original name
macro_rules! _2tup_slice_ro {
    ($x:ident) => {
        let $x = ($x.0.as_slice()?, $x.1.as_slice()?);
    };
}

/// Convert a 3-tuple of PyReadwriteArray to read-write slices, shadowing the original name
macro_rules! _3tup_slice_mut {
    ($x:ident) => {
        let $x = (
            $x.0.as_slice_mut()?,
            $x.1.as_slice_mut()?,
            $x.2.as_slice_mut()?,
        );
    };
}

/// Assemble an Ok((x, y, z)) for 3 output arrays of potentially different types
macro_rules! _3tup_ret {
    (($x:ident, $xt:ty), ($y:ident, $yt:ty), ($z:ident, $zt:ty)) => {
        // Acquire global interpreter lock, which will be released when it goes out of scope
        Python::attach(|py| {
            let $x: Py<PyArray1<$xt>> = PyArray1::from_vec(py, $x).unbind();
            let $y: Py<PyArray1<$yt>> = PyArray1::from_vec(py, $y).unbind();
            let $z: Py<PyArray1<$zt>> = PyArray1::from_vec(py, $z).unbind();

            Ok(($x, $y, $z))
        })
    };
}

pub(crate) use _2tup_slice_ro;
pub(crate) use _3tup_ret;
pub(crate) use _3tup_slice_mut;
pub(crate) use _3tup_slice_ro;
