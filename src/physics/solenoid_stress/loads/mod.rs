//! Sparse load operators for repeated-load and one-shot solves.
//!
//! Each load type lives in its own submodule.  One-shot assembly in Python builds the global
//! right-hand side by applying these operators, and repeated-load solves reuse the same operators
//! directly.
//!
//! The triplet containers in this module describe linear maps before compression into CSR form.
//! For any such operator, rows index output equations, columns index load-amplitude inputs, and
//! each stored coefficient has units
//! `[output units / input units]`.

use crate::physics::solenoid_stress::types::Real;

mod body_force;
mod pressure;
mod thermal;
mod traction;

/// Sparse triplet representation of one load operator before CSR compression.
///
/// This container is used during assembly because finite-element contributions are naturally
/// accumulated element by element into COO-style triplets.  The semantic meaning of the rows,
/// columns, and coefficient units depends on the specific operator:
/// - body-force operators map per-element `[force / volume]` inputs to reduced RHS entries
///   `[energy / distance]`, so their coefficients have units `[volume]`,
/// - pressure and traction operators map boundary loads `[force / area]` to reduced RHS entries,
///   so their coefficients have units `[area]`,
/// - temperature operators map nodal temperatures `[temperature]` to reduced RHS entries, so their
///   coefficients have units `[energy / (distance * temperature)]`.
#[derive(Debug, Clone)]
pub struct SparseOperator<F: Real> {
    /// Output row index for each stored triplet coefficient.
    pub rows: Vec<usize>,
    /// Input column index for each stored triplet coefficient.
    pub cols: Vec<usize>,
    /// Nonzero operator coefficients in triplet order.
    pub vals: Vec<F>,
    /// Number of operator rows.
    pub nrow: usize,
    /// Number of operator columns.
    pub ncol: usize,
}

/// Thermal load operator split into a temperature-dependent matrix part and a constant offset.
///
/// The thermoelastic load is
/// `f_thermal = temperature_to_rhs @ T + reference_rhs`,
/// where `T` is the nodal temperature vector `[temperature]`.
#[derive(Debug, Clone)]
pub struct ThermalLoadOperator<F: Real> {
    /// Sparse operator mapping nodal temperatures `[temperature]` to reduced RHS entries
    /// `[energy / distance]`.
    pub temperature_to_rhs: SparseOperator<F>,
    /// Constant reduced RHS contribution from the per-material reference temperatures.
    ///
    /// Units: `[energy / distance]`.
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

pub(crate) use body_force::body_force_operator_for_family;
pub(crate) use pressure::pressure_operator_for_family;
pub(crate) use thermal::temperature_operator_for_family;
pub(crate) use traction::traction_operator_for_family;
