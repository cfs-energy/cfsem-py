//! Sparse load operators for repeated-load and one-shot solves.
//!
//! Each load type lives in its own submodule.  One-shot assembly in Python builds the global
//! right-hand side by applying these operators, and repeated-load solves reuse the same operators
//! directly.

use crate::physics::solenoid_stress::types::Real;

mod body_force;
mod pressure;
mod thermal;
mod traction;

#[derive(Debug, Clone)]
pub struct SparseOperator<F: Real> {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<F>,
    pub nrow: usize,
    pub ncol: usize,
}

#[derive(Debug, Clone)]
pub struct ThermalLoadOperator<F: Real> {
    pub temperature_to_rhs: SparseOperator<F>,
    pub reference_rhs: Vec<F>,
}

pub(super) fn scatter_local_vector<F: Real, const NROW: usize>(
    rows: &mut Vec<usize>,
    cols: &mut Vec<usize>,
    vals: &mut Vec<F>,
    global_rows: &[usize; NROW],
    global_col: usize,
    local: &[F; NROW],
) {
    for row in 0..NROW {
        let value = local[row];
        if value != F::zero() {
            rows.push(global_rows[row]);
            cols.push(global_col);
            vals.push(value);
        }
    }
}

pub(super) fn scatter_local_matrix<F: Real, const NROW: usize, const NCOL: usize>(
    rows: &mut Vec<usize>,
    cols: &mut Vec<usize>,
    vals: &mut Vec<F>,
    global_rows: &[usize; NROW],
    global_cols: &[usize; NCOL],
    local: &[[F; NCOL]; NROW],
) {
    for row in 0..NROW {
        for col in 0..NCOL {
            let value = local[row][col];
            if value != F::zero() {
                rows.push(global_rows[row]);
                cols.push(global_cols[col]);
                vals.push(value);
            }
        }
    }
}

pub use body_force::{body_force_operator_quad4, body_force_operator_quad9};
pub use pressure::{pressure_operator_quad4, pressure_operator_quad9};
pub use thermal::{temperature_operator_quad4, temperature_operator_quad9};
pub use traction::{traction_operator_quad4, traction_operator_quad9};
