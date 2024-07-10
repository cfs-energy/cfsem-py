//! Python bindings for cfsemrs
#![allow(non_snake_case)]

use numpy::PyArray1;
use numpy::PyArrayMethods;
use pyo3::exceptions;
use pyo3::prelude::*;
use std::fmt::Debug;

use cfsem::{math, mesh, physics};

/// Errors from mismatch between python and rust
#[derive(Debug)]
#[allow(dead_code)]
enum PyInteropError {
    DimensionalityError { msg: String },
}

impl From<PyInteropError> for PyErr {
    fn from(val: PyInteropError) -> Self {
        exceptions::PyValueError::new_err(format!("{:#?}", &val))
    }
}

#[pyfunction]
fn filament_helix_path<'py>(
    path: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ), // [m]
    helix_start_offset: (f64, f64, f64),
    twist_pitch: f64,
    angle_offset: f64,
    out: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ),
) -> PyResult<()> {
    // Unpack
    let xp_readonly = path.0.readonly();
    let xp = xp_readonly.as_slice()?;

    let yp_readonly = path.1.readonly();
    let yp = yp_readonly.as_slice()?;

    let zp_readonly = path.2.readonly();
    let zp = zp_readonly.as_slice()?;

    let mut xfil_readwrite = out.0.readwrite();
    let xfil = xfil_readwrite.as_slice_mut()?;

    let mut yfil_readwrite = out.1.readwrite();
    let yfil = yfil_readwrite.as_slice_mut()?;

    let mut zfil_readwrite = out.2.readwrite();
    let zfil = zfil_readwrite.as_slice_mut()?;

    // Calculate
    match mesh::filament_helix_path(
        (xp, yp, zp),
        helix_start_offset,
        twist_pitch,
        angle_offset,
        (xfil, yfil, zfil),
    ) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    Ok(())
}

#[pyfunction]
fn rotate_filaments_about_path<'py>(
    path: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ), // [m]
    angle_offset: f64,
    out: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ),
) -> PyResult<()> {
    // Unpack
    let xp_readonly = path.0.readonly();
    let xp = xp_readonly.as_slice()?;

    let yp_readonly = path.1.readonly();
    let yp = yp_readonly.as_slice()?;

    let zp_readonly = path.2.readonly();
    let zp = zp_readonly.as_slice()?;

    let mut xfil_readwrite = out.0.readwrite();
    let xfil = xfil_readwrite.as_slice_mut()?;

    let mut yfil_readwrite = out.1.readwrite();
    let yfil = yfil_readwrite.as_slice_mut()?;

    let mut zfil_readwrite = out.2.readwrite();
    let zfil = zfil_readwrite.as_slice_mut()?;

    // Calculate
    match mesh::rotate_filaments_about_path((xp, yp, zp), angle_offset, (xfil, yfil, zfil)) {
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
fn flux_circular_filament<'py>(
    current: Bound<'py, PyArray1<f64>>,
    r: Bound<'py, PyArray1<f64>>,
    z: Bound<'py, PyArray1<f64>>,
    rprime: Bound<'py, PyArray1<f64>>,
    zprime: Bound<'py, PyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    let current_readonly = current.readonly();
    let current = current_readonly.as_slice()?;
    let r_readonly = r.readonly();
    let r = r_readonly.as_slice()?;
    let z_readonly = z.readonly();
    let z = z_readonly.as_slice()?;
    let rprime_readonly = rprime.readonly();
    let rprime = rprime_readonly.as_slice()?;
    let zprime_readonly = zprime.readonly();
    let zprime = zprime_readonly.as_slice()?;

    // Get array shapes, make sure they make sense
    let m = rprime.len();

    // Initialize output
    let mut psi = vec![0.0; m];

    // Do calculations
    match physics::flux_circular_filament(current, r, z, rprime, zprime, &mut psi[..]) {
        Ok(_) => {}
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    // Acquire global interpreter lock, which will be released when it goes out of scope
    Python::with_gil(|py| {
        Ok(PyArray1::from_vec_bound(py, psi).unbind()) // Make PyObject
    })
}

/// Python bindings for cfsemrs::physics::flux_density_circular_filament
#[pyfunction]
fn flux_density_circular_filament<'py>(
    current: Bound<'py, PyArray1<f64>>,
    rfil: Bound<'py, PyArray1<f64>>,
    zfil: Bound<'py, PyArray1<f64>>,
    rprime: Bound<'py, PyArray1<f64>>,
    zprime: Bound<'py, PyArray1<f64>>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    let current_readonly = current.readonly();
    let current = current_readonly.as_slice()?;
    let rfil_readonly = rfil.readonly();
    let rfil = rfil_readonly.as_slice()?;
    let zfil_readonly = zfil.readonly();
    let zfil = zfil_readonly.as_slice()?;
    let rprime_readonly = rprime.readonly();
    let rprime = rprime_readonly.as_slice()?;
    let zprime_readonly = zprime.readonly();
    let zprime = zprime_readonly.as_slice()?;

    // Initialize output
    let n = rprime.len();
    let mut br = vec![0.0; n];
    let mut bz = vec![0.0; n];

    // Do calculations
    match physics::flux_density_circular_filament(
        &current, &rfil, &zfil, &rprime, &zprime, &mut br, &mut bz,
    ) {
        Ok(_) => {}
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    }

    // Acquire global interpreter lock, which will be released when it goes out of scope
    Python::with_gil(|py| {
        let br: Py<PyArray1<f64>> = PyArray1::from_slice_bound(py, &br).unbind(); // Make PyObject
        let bz: Py<PyArray1<f64>> = PyArray1::from_slice_bound(py, &bz).unbind(); // Make PyObject

        Ok((br, bz))
    })
}

/// Python bindings for cfsemrs::physics::biotsavart::flux_density_biot_savart
#[pyfunction]
fn flux_density_biot_savart<'py>(
    xyzp: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ), // [m] Test point coords
    xyzfil: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ), // [m] Filament origin coords (start of segment)
    dlxyzfil: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ), // [m] Filament length delta
    ifil: Bound<'py, PyArray1<f64>>, // [A] filament current
    par: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    let xpro = xyzp.0.readonly();
    let ypro = xyzp.1.readonly();
    let zpro = xyzp.2.readonly();
    let xyzp = (xpro.as_slice()?, ypro.as_slice()?, zpro.as_slice()?);

    let xfilro = xyzfil.0.readonly();
    let yfilro = xyzfil.1.readonly();
    let zfilro = xyzfil.2.readonly();
    let xyzfil = (xfilro.as_slice()?, yfilro.as_slice()?, zfilro.as_slice()?);

    let dlxfilro = dlxyzfil.0.readonly();
    let dlyfilro = dlxyzfil.1.readonly();
    let dlzfilro = dlxyzfil.2.readonly();
    let dlxyzfil = (
        dlxfilro.as_slice()?,
        dlyfilro.as_slice()?,
        dlzfilro.as_slice()?,
    );
    let ifilro = ifil.readonly();
    let ifil = ifilro.as_slice()?;

    // Do calculations
    let n = xyzp.0.len();
    let (mut bx, mut by, mut bz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);

    let func = match par {
        true => physics::biotsavart::flux_density_biot_savart_par,
        false => physics::biotsavart::flux_density_biot_savart,
    };
    match func(xyzp, xyzfil, dlxyzfil, ifil, (&mut bx, &mut by, &mut bz)) {
        Ok(x) => x,
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    // Acquire global interpreter lock, which will be released when it goes out of scope
    Python::with_gil(|py| {
        let bx: Py<PyArray1<f64>> = PyArray1::from_vec_bound(py, bx).unbind();
        let by: Py<PyArray1<f64>> = PyArray1::from_vec_bound(py, by).unbind();
        let bz: Py<PyArray1<f64>> = PyArray1::from_vec_bound(py, bz).unbind();

        Ok((bx, by, bz))
    })
}

#[pyfunction]
fn inductance_piecewise_linear_filaments<'py>(
    xyzfil0: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ), // [m] Filament origin coords (start of segment)
    dlxyzfil0: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ), // [m] Filament length delta
    xyzfil1: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ), // [m] Filament origin coords (start of segment)
    dlxyzfil1: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ), // [m] Filament length delta
    self_inductance: bool, // Whether this is being used as a self-inductance calc
) -> PyResult<f64> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
    let xfilro0 = xyzfil0.0.readonly();
    let yfilro0 = xyzfil0.1.readonly();
    let zfilro0 = xyzfil0.2.readonly();
    let xyzfil0 = (
        xfilro0.as_slice()?,
        yfilro0.as_slice()?,
        zfilro0.as_slice()?,
    );

    let dlxfilro0 = dlxyzfil0.0.readonly();
    let dlyfilro0 = dlxyzfil0.1.readonly();
    let dlzfilro0 = dlxyzfil0.2.readonly();
    let dlxyzfil0 = (
        dlxfilro0.as_slice()?,
        dlyfilro0.as_slice()?,
        dlzfilro0.as_slice()?,
    );

    let xfilro1 = xyzfil1.0.readonly();
    let yfilro1 = xyzfil1.1.readonly();
    let zfilro1 = xyzfil1.2.readonly();
    let xyzfil1 = (
        xfilro1.as_slice()?,
        yfilro1.as_slice()?,
        zfilro1.as_slice()?,
    );

    let dlxfilro1 = dlxyzfil1.0.readonly();
    let dlyfilro1 = dlxyzfil1.1.readonly();
    let dlzfilro1 = dlxyzfil1.2.readonly();
    let dlxyzfil1 = (
        dlxfilro1.as_slice()?,
        dlyfilro1.as_slice()?,
        dlzfilro1.as_slice()?,
    );

    // Do calculations
    let inductance = match physics::linear_filament::inductance_piecewise_linear_filaments(
        xyzfil0,
        dlxyzfil0,
        xyzfil1,
        dlxyzfil1,
        self_inductance,
    ) {
        Ok(x) => x,
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    Ok(inductance)
}

/// Python bindings for cfsemrs::physics::gradshafranov::gs_operator_order2
#[pyfunction]
fn gs_operator_order2<'py>(
    rs: Bound<'py, PyArray1<f64>>,
    zs: Bound<'py, PyArray1<f64>>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<usize>>, Py<PyArray1<usize>>)> {
    // Process inputs
    let rsro = rs.readonly();
    let rs = rsro.as_slice()?;
    let zsro = zs.readonly();
    let zs = zsro.as_slice()?;

    // Do calculations
    let (vals, rows, cols) = physics::gradshafranov::gs_operator_order2(rs, zs);

    // Acquire global interpreter lock, which will be released when it goes out of scope
    Python::with_gil(|py| {
        let vals: Py<PyArray1<f64>> = PyArray1::from_vec_bound(py, vals).unbind();
        let rows: Py<PyArray1<usize>> = PyArray1::from_vec_bound(py, rows).unbind();
        let cols: Py<PyArray1<usize>> = PyArray1::from_vec_bound(py, cols).unbind();

        Ok((vals, rows, cols))
    })
}

/// Python bindings for cfsemrs::physics::gradshafranov::gs_operator_order4
#[pyfunction]
fn gs_operator_order4<'py>(
    rs: Bound<'py, PyArray1<f64>>,
    zs: Bound<'py, PyArray1<f64>>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<usize>>, Py<PyArray1<usize>>)> {
    // Process inputs
    let rsro = rs.readonly();
    let rs = rsro.as_slice()?;
    let zsro = zs.readonly();
    let zs = zsro.as_slice()?;

    // Do calculations
    let (vals, rows, cols) = physics::gradshafranov::gs_operator_order4(rs, zs);

    // Acquire global interpreter lock, which will be released when it goes out of scope
    Python::with_gil(|py| {
        let vals: Py<PyArray1<f64>> = PyArray1::from_vec_bound(py, vals).unbind();
        let rows: Py<PyArray1<usize>> = PyArray1::from_vec_bound(py, rows).unbind();
        let cols: Py<PyArray1<usize>> = PyArray1::from_vec_bound(py, cols).unbind();

        Ok((vals, rows, cols))
    })
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

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
#[pyo3(name = "_cfsem")]
fn _cfsem<'py>(_py: Python, m: Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(flux_circular_filament, m.clone())?)?;
    m.add_function(wrap_pyfunction!(flux_density_circular_filament, m.clone())?)?;
    m.add_function(wrap_pyfunction!(flux_density_biot_savart, m.clone())?)?;
    m.add_function(wrap_pyfunction!(
        inductance_piecewise_linear_filaments,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(gs_operator_order2, m.clone())?)?;
    m.add_function(wrap_pyfunction!(gs_operator_order4, m.clone())?)?;
    m.add_function(wrap_pyfunction!(ellipe, m.clone())?)?;
    m.add_function(wrap_pyfunction!(ellipk, m.clone())?)?;
    m.add_function(wrap_pyfunction!(filament_helix_path, m.clone())?)?;
    m.add_function(wrap_pyfunction!(rotate_filaments_about_path, m.clone())?)?;
    Ok(())
}
