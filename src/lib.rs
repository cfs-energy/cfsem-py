#![doc=include_str!("../README.md")]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_late_init)]
#![allow(non_snake_case)]

#[cfg(feature = "python")]
pub mod python;

#[allow(unused_imports)] // Doesn't build without this
#[cfg(feature = "python")]
use python::*;

use std::sync::LazyLock;

pub mod math;
pub mod mesh;
pub mod physics;

#[cfg(test)]
pub(crate) mod testing;

/// (H/m) vacuum magnetic permeability.
/// Value from 2022 CODATA recommended values, [NIST SPI 961](https://physics.nist.gov/cuu/pdf/wall_2022.pdf).
pub const MU_0: f64 = 0.999_999_999_87 * core::f64::consts::PI * 4e-7; // [H/m]

/// (H/m) Recurring constant multiple of `mu_0`
pub const MU0_OVER_4PI: f64 = MU_0 / (4.0 * core::f64::consts::PI);

/// Number of physical CPU cores available to Rayon-backed parallel loops.
///
/// This is populated once on first access and then reused so chunk-size
/// selection does not repeatedly query the OS.
static PHYSICAL_CORES: LazyLock<usize> = LazyLock::new(num_cpus::get_physical);

/// Chunk size for parallelism.
///
/// The heuristic aims for one chunk per physical core.  It intentionally returns at least one so
/// small meshes keep the same code path without special-case empty chunk handling.
pub(crate) fn chunksize(nelem: usize) -> usize {
    (nelem / (*PHYSICAL_CORES).max(1)).max(1)
}

/// Contiguous half-open ranges covering `0..len`.
///
/// The returned ranges preserve source ordering, which lets callers concatenate independently
/// assembled chunks without reindexing rows or columns.
pub(crate) fn ranges_for_len(len: usize, chunk: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(len.div_ceil(chunk));
    let mut start = 0;
    while start < len {
        let end = (start + chunk).min(len);
        ranges.push((start, end));
        start = end;
    }
    ranges
}

#[macro_use]
pub(crate) mod macros {

    /// Make sure the length of any number of vec/array/slice are the same.
    macro_rules! check_length {
        ($n:expr, $($y:expr),+) => {
            $(  // Repeat for all y
                if $y.len() != $n {
                    return Err("Length mismatch");
                }
            )+
        };
    }

    /// Make sure the lengths of 3 vec/array/slice in a tuple are the same
    macro_rules! check_length_3tup {
        ($n:expr, $x:expr) => {
            if $x.0.len() != $n || $x.1.len() != $n || $x.2.len() != $n {
                return Err("Length mismatch");
            }
        };
    }

    macro_rules! par_chunks_3tup {
        ($x:expr, $n:expr) => {
            (
                $x.0.par_chunks($n),
                $x.1.par_chunks($n),
                $x.2.par_chunks($n),
            )
        };
    }

    macro_rules! mut_par_chunks_3tup {
        ($x:expr, $n:expr) => {
            (
                $x.0.par_chunks_mut($n),
                $x.1.par_chunks_mut($n),
                $x.2.par_chunks_mut($n),
            )
        };
    }

    pub(crate) use mut_par_chunks_3tup;
    pub(crate) use par_chunks_3tup;

    pub(crate) use check_length;
    pub(crate) use check_length_3tup;
}
