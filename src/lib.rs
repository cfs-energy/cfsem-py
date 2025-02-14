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
fn flux_circular_filament<'py>(
    current: Bound<'py, PyArray1<f64>>,
    rfil: Bound<'py, PyArray1<f64>>,
    zfil: Bound<'py, PyArray1<f64>>,
    rprime: Bound<'py, PyArray1<f64>>,
    zprime: Bound<'py, PyArray1<f64>>,
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
    Python::with_gil(|py| {
        Ok(PyArray1::from_vec(py, psi).unbind()) // Make PyObject
    })
}

/// Python bindings for cfsemrs::physics::circular_filament::vector_potential_circular_filament
#[pyfunction]
fn vector_potential_circular_filament<'py>(
    current: Bound<'py, PyArray1<f64>>,
    rfil: Bound<'py, PyArray1<f64>>,
    zfil: Bound<'py, PyArray1<f64>>,
    rprime: Bound<'py, PyArray1<f64>>,
    zprime: Bound<'py, PyArray1<f64>>,
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
    Python::with_gil(|py| {
        Ok(PyArray1::from_vec(py, out).unbind()) // Make PyObject
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
    Python::with_gil(|py| {
        let br: Py<PyArray1<f64>> = PyArray1::from_slice(py, &br).unbind(); // Make PyObject
        let bz: Py<PyArray1<f64>> = PyArray1::from_slice(py, &bz).unbind(); // Make PyObject

        Ok((br, bz))
    })
}

/// Python bindings for cfsemrs::physics::linear_filament::flux_density_linear_filament
#[pyfunction]
fn flux_density_linear_filament<'py>(
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
    _3tup_slice_ro!(xyzp);
    _3tup_slice_ro!(xyzfil);
    _3tup_slice_ro!(dlxyzfil);
    let ifilro = ifil.readonly();
    let ifil = ifilro.as_slice()?;

    // Do calculations
    let n = xyzp.0.len();
    let (mut bx, mut by, mut bz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);

    let func = match par {
        true => physics::linear_filament::flux_density_linear_filament_par,
        false => physics::linear_filament::flux_density_linear_filament,
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
#[pyfunction]
fn vector_potential_linear_filament<'py>(
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
    _3tup_slice_ro!(xyzp);
    _3tup_slice_ro!(xyzfil);
    _3tup_slice_ro!(dlxyzfil);

    let ifilro = ifil.readonly();
    let ifil = ifilro.as_slice()?;

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
    _3tup_slice_ro!(xyzfil0);
    _3tup_slice_ro!(dlxyzfil0);
    _3tup_slice_ro!(xyzfil1);
    _3tup_slice_ro!(dlxyzfil1);

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

    _3tup_ret!((vals, f64), (rows, usize), (cols, usize))
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
fn flux_density_circular_filament_cartesian<'py>(
    current: Bound<'py, PyArray1<f64>>,
    rfil: Bound<'py, PyArray1<f64>>,
    zfil: Bound<'py, PyArray1<f64>>,
    xyzobs: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
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
fn mutual_inductance_circular_to_linear<'py>(
    rfil: Bound<'py, PyArray1<f64>>,
    zfil: Bound<'py, PyArray1<f64>>,
    nfil: Bound<'py, PyArray1<f64>>,
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
fn flux_density_dipole<'py>(
    loc: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ), // [m] dipole locations in cartesian coordinates
    moment: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ), // [A-m^2] dipole moment vector
    obs: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ), // [m] Observation point coords
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
    match func(loc, moment, obs, (&mut outx, &mut outy, &mut outz)) {
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
fn body_force_density_circular_filament_cartesian<'py>(
    current: Bound<'py, PyArray1<f64>>,
    rfil: Bound<'py, PyArray1<f64>>,
    zfil: Bound<'py, PyArray1<f64>>,
    obs: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ), // [m] Filament origin coords (start of segment)
    j: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
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
fn body_force_density_linear_filament<'py>(
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
    obs: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ), // [m] Filament origin coords (start of segment)
    j: (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ), // [A/m^2] current density at observation points
    par: bool,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    // Get references to contiguous data as slice
    // or error if data is not contiguous
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

    let obsxro = obs.0.readonly();
    let obsyro = obs.1.readonly();
    let obszro = obs.2.readonly();
    let obs = (obsxro.as_slice()?, obsyro.as_slice()?, obszro.as_slice()?);

    let jxro = j.0.readonly();
    let jyro = j.1.readonly();
    let jzro = j.2.readonly();
    let j = (jxro.as_slice()?, jyro.as_slice()?, jzro.as_slice()?);

    // Select variant
    let func = match par {
        true => physics::linear_filament::body_force_density_linear_filament_par,
        false => physics::linear_filament::body_force_density_linear_filament,
    };

    // Do calculations
    let n = obs.0.len();
    let (mut outx, mut outy, mut outz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let out = (&mut outx[..], &mut outy[..], &mut outz[..]);

    match func(xyzfil, dlxyzfil, ifil, obs, j, out) {
        Ok(_) => (),
        Err(x) => {
            let err: PyErr = PyInteropError::DimensionalityError { msg: x.to_string() }.into();
            return Err(err);
        }
    };

    _3tup_ret!((outx, f64), (outy, f64), (outz, f64))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
#[pyo3(name = "_cfsem")]
fn _cfsem<'py>(_py: Python, m: Bound<'py, PyModule>) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(
        vector_potential_linear_filament,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        inductance_piecewise_linear_filaments,
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

    Ok(())
}

/// Convert a 3-tuple of PyArray to read-only slices, shadowing the original name
macro_rules! _3tup_slice_ro {
    ($x:ident) => {
        let _ro = ($x.0.readonly(), $x.1.readonly(), $x.2.readonly());
        let $x = (_ro.0.as_slice()?, _ro.1.as_slice()?, _ro.2.as_slice()?);
    };
}

/// Convert a 2-tuple of PyArray to read-only slices, shadowing the original name
macro_rules! _2tup_slice_ro {
    ($x:ident) => {
        let _ro = ($x.0.readonly(), $x.1.readonly());
        let $x = (_ro.0.as_slice()?, _ro.1.as_slice()?);
    };
}

/// Convert a 3-tuple of PyArray to read-write slices, shadowing the original name
macro_rules! _3tup_slice_mut {
    ($x:ident) => {
        let mut _rw = ($x.0.readwrite(), $x.1.readwrite(), $x.2.readwrite());
        let $x = (
            _rw.0.as_slice_mut()?,
            _rw.1.as_slice_mut()?,
            _rw.2.as_slice_mut()?,
        );
    };
}

/// Assemble an Ok((x, y, z)) for 3 output arrays of potentially different types
macro_rules! _3tup_ret {
    (($x:ident, $xt:ty), ($y:ident, $yt:ty), ($z:ident, $zt:ty)) => {
        // Acquire global interpreter lock, which will be released when it goes out of scope
        Python::with_gil(|py| {
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
