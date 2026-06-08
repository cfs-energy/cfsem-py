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

fn concat_sparse_operators<F: Real>(
    chunks: Vec<SparseOperator<F>>,
    nrow: usize,
    ncol: usize,
    capacity: usize,
) -> SparseOperator<F> {
    let mut rows = Vec::with_capacity(capacity);
    let mut cols = Vec::with_capacity(capacity);
    let mut vals = Vec::with_capacity(capacity);

    for chunk in chunks {
        debug_assert_eq!(chunk.nrow, nrow);
        debug_assert_eq!(chunk.ncol, ncol);
        rows.extend(chunk.rows);
        cols.extend(chunk.cols);
        vals.extend(chunk.vals);
    }

    SparseOperator {
        rows,
        cols,
        vals,
        nrow,
        ncol,
    }
}

fn concat_thermal_load_operators<F: Real>(
    chunks: Vec<ThermalLoadOperator<F>>,
    ndof: usize,
    ncol: usize,
    capacity: usize,
) -> ThermalLoadOperator<F> {
    let mut reference_rhs = vec![F::zero(); ndof];
    let sparse_chunks = chunks
        .into_iter()
        .map(|chunk| {
            for (dst, src) in reference_rhs.iter_mut().zip(chunk.reference_rhs) {
                *dst = *dst + src;
            }
            chunk.temperature_to_rhs
        })
        .collect::<Vec<_>>();

    ThermalLoadOperator {
        temperature_to_rhs: concat_sparse_operators(sparse_chunks, ndof, ncol, capacity),
        reference_rhs,
    }
}

fn ranges_for_len(len: usize, chunk: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(len.div_ceil(chunk));
    let mut start = 0;
    while start < len {
        let end = (start + chunk).min(len);
        ranges.push((start, end));
        start = end;
    }
    ranges
}

pub(crate) use body_force::{body_force_operator_for_family, body_force_operator_for_family_par};
pub(crate) use pressure::{pressure_operator_for_family, pressure_operator_for_family_par};
pub(crate) use thermal::{temperature_operator_for_family, temperature_operator_for_family_par};
pub(crate) use traction::{traction_operator_for_family, traction_operator_for_family_par};
