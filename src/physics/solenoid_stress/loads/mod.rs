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

use rayon::prelude::*;

use crate::{chunksize, ranges_for_len};

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
pub struct SparseOperator {
    /// Output row index for each stored triplet coefficient.
    pub rows: Vec<usize>,
    /// Input column index for each stored triplet coefficient.
    pub cols: Vec<usize>,
    /// Nonzero operator coefficients in triplet order.
    pub vals: Vec<f64>,
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
pub struct ThermalLoadOperator {
    /// Sparse operator mapping nodal temperatures `[temperature]` to reduced RHS entries
    /// `[energy / distance]`.
    pub temperature_to_rhs: SparseOperator,
    /// Constant reduced RHS contribution from the per-material reference temperatures.
    ///
    /// Units: `[energy / distance]`.
    pub reference_rhs: Vec<f64>,
}

fn concat_sparse_operators(
    chunks: Vec<SparseOperator>,
    nrow: usize,
    ncol: usize,
    capacity: usize,
) -> SparseOperator {
    // Worker chunks are emitted in the same order as `ranges_for_len`, so appending their triplet
    // buffers preserves the serial load ordering.  Any duplicate structural rows are handled later
    // by the CSR reduction/compression code.
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

fn concat_thermal_load_operators(
    chunks: Vec<ThermalLoadOperator>,
    ndof: usize,
    ncol: usize,
    capacity: usize,
) -> ThermalLoadOperator {
    // Thermal loads have one sparse temperature operator plus a dense reference-temperature RHS.
    // The sparse part can be concatenated like other loads; the dense offsets must be summed over
    // all worker chunks because every element contributes to the same global RHS vector.
    let mut reference_rhs = vec![0.0; ndof];
    let sparse_chunks = chunks
        .into_iter()
        .map(|chunk| {
            for (dst, src) in reference_rhs.iter_mut().zip(chunk.reference_rhs) {
                *dst += src;
            }
            chunk.temperature_to_rhs
        })
        .collect::<Vec<_>>();

    ThermalLoadOperator {
        temperature_to_rhs: concat_sparse_operators(sparse_chunks, ndof, ncol, capacity),
        reference_rhs,
    }
}

fn collect_sparse_operator_chunks(
    len: usize,
    build: impl Fn(usize, usize) -> Result<SparseOperator, String> + Sync,
) -> Result<Vec<SparseOperator>, String> {
    // Parallel wrappers differ only in the unit of work they split over: elements for body force
    // and thermal loads, boundary load records for pressure and traction.
    ranges_for_len(len, chunksize(len))
        .into_par_iter()
        .map(|(start, end)| build(start, end))
        .collect()
}

fn collect_thermal_load_operator_chunks(
    len: usize,
    build: impl Fn(usize, usize) -> Result<ThermalLoadOperator, String> + Sync,
) -> Result<Vec<ThermalLoadOperator>, String> {
    // Keep thermal chunks typed separately so the reference-temperature RHS cannot be accidentally
    // discarded by a generic sparse-only collector.
    ranges_for_len(len, chunksize(len))
        .into_par_iter()
        .map(|(start, end)| build(start, end))
        .collect()
}

pub(crate) use body_force::{
    apply_body_force_rhs_for_family, body_force_operator_for_family,
    body_force_operator_for_family_par,
};
pub(crate) use pressure::{
    apply_pressure_rhs_for_family, pressure_operator_for_family, pressure_operator_for_family_par,
};
pub(crate) use thermal::{
    apply_temperature_rhs_for_family, temperature_operator_for_family,
    temperature_operator_for_family_par, thermal_reference_rhs_for_family,
};
pub(crate) use traction::{
    apply_traction_rhs_for_family, traction_operator_for_family, traction_operator_for_family_par,
};
