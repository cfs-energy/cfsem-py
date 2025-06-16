#![doc=include_str!("../README.md")]
#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_late_init)]
#![allow(non_snake_case)]

#[cfg(feature = "python")]
pub mod python;

#[allow(unused_imports)] // Doesn't build without this
#[cfg(feature = "python")]
use python::*;

use std::num::NonZeroUsize;

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

/// Chunk size for parallelism
pub(crate) fn chunksize(nelem: usize) -> usize {
    let ncores = std::thread::available_parallelism()
        .unwrap_or(NonZeroUsize::MIN)
        .get();

    (nelem / ncores).max(1)
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
