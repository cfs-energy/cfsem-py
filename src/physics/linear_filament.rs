//! Magnetics calculations for piecewise-linear current filaments.

use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

use crate::mesh::quadrature::{GaussLegendreRule, gauss_legendre_unit_interval_table};
use crate::{
    chunksize,
    math::{PointLineDistance, cross3, point_line_distance_with_endpoints},
};

use crate::math::Scalar;
use crate::{MU0_OVER_4PI, macros::*};

/// (m) minimum representable nonzero wire thickness.
const MIN_WIRE_THICKNESS: f64 = 1e-10;

/// Three-point target-segment integral for one unit-current source segment.
#[inline]
fn inductance_linear_filament_pair(
    src_start: (f64, f64, f64),
    src_delta: (f64, f64, f64),
    wire_radius_src: f64,
    tgt_start: (f64, f64, f64),
    tgt_delta: (f64, f64, f64),
) -> f64 {
    let src_end = (
        src_start.0 + src_delta.0,
        src_start.1 + src_delta.1,
        src_start.2 + src_delta.2,
    );
    let gl3_unit = gauss_legendre_unit_interval_table(GaussLegendreRule::Gauss3);
    let mut inductance = 0.0;
    for &[tq, wq] in gl3_unit {
        let obs = (
            tgt_delta.0.mul_add(tq, tgt_start.0),
            tgt_delta.1.mul_add(tq, tgt_start.1),
            tgt_delta.2.mul_add(tq, tgt_start.2),
        );
        let (ax, ay, az) = vector_potential_linear_filament_scalar(
            (src_start, src_end, 1.0),
            wire_radius_src,
            obs,
        );
        inductance += wq * (ax * tgt_delta.0 + ay * tgt_delta.1 + az * tgt_delta.2);
    }
    inductance
}

/// Estimate the inductive coupling between two piecewise-linear current filaments.
///
/// This uses the vector-potential line-integral form
/// `M = ∮ A_source · dl_target` with a 1 A source current on each source segment,
/// evaluated with 3-point Gauss-Legendre quadrature on each target segment.
///
/// # Arguments
///
/// * `xyzfil0`:         (m) filament origin coordinates for first path, length `n`
/// * `dlxyzfil0`:       (m) filament segment lengths for first path, length `n`
/// * `xyzfil1`:         (m) filament origin coordinates for second path, length `n`
/// * `dlxyzfil1`:       (m) filament segment lengths for second path, length `n`
/// * `wire_radius`:     (m) source filament radius for first path, length `n`
///
/// # Assumptions
///
/// * Thin, well-behaved filaments
/// * Uniform current distribution within segments
///     * Low frequency operation; no skin effect
/// * Vacuum permeability everywhere
/// * Each filament has a constant current in all segments
///   (otherwise the result is an interaction matrix rather than a scalar)
pub fn inductance_piecewise_linear_filaments(
    xyzfil0: (&[f64], &[f64], &[f64]),
    dlxyzfil0: (&[f64], &[f64], &[f64]),
    xyzfil1: (&[f64], &[f64], &[f64]),
    dlxyzfil1: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
) -> Result<f64, &'static str> {
    // Unpack
    let (xfil1, yfil1, zfil1) = xyzfil1;
    let (dlxfil1, dlyfil1, dlzfil1) = dlxyzfil1;

    // Check lengths; Error if they do not match
    let n = xyzfil0.0.len();
    check_length!(
        n,
        xyzfil0.0,
        xyzfil0.1,
        xyzfil0.2,
        dlxyzfil0.0,
        dlxyzfil0.1,
        dlxyzfil0.2,
        wire_radius
    );

    let m = xfil1.len();
    check_length!(m, xfil1, yfil1, zfil1, dlxfil1, dlyfil1, dlzfil1);

    let (xfil0, yfil0, zfil0) = xyzfil0;
    let (dlxfil0, dlyfil0, dlzfil0) = dlxyzfil0;
    let mut inductance = 0.0; // [H]
    for j in 0..m {
        let tgt_start = (xfil1[j], yfil1[j], zfil1[j]);
        let tgt_delta = (dlxfil1[j], dlyfil1[j], dlzfil1[j]);
        for i in 0..n {
            inductance += inductance_linear_filament_pair(
                (xfil0[i], yfil0[i], zfil0[i]),
                (dlxfil0[i], dlyfil0[i], dlzfil0[i]),
                wire_radius[i],
                tgt_start,
                tgt_delta,
            );
        }
    }

    Ok(inductance)
}

/// Estimate the inductive coupling from many source filament segments to many target filament segments.
///
/// This uses the same `A·dl` kernel as [`inductance_piecewise_linear_filaments`], but with a
/// source/target segment API analogous to the linear-filament field evaluators. Each output entry
/// is the inductive coupling between the full source collection and one target segment.
///
/// # Arguments
///
/// * `xyzfil_tgt`:      (m) target filament origin coordinates, each length `ntgt`
/// * `dlxyzfil_tgt`:    (m) target filament segment lengths, each length `ntgt`
/// * `xyzfil_src`:      (m) source filament origin coordinates, each length `nsrc`
/// * `dlxyzfil_src`:    (m) source filament segment lengths, each length `nsrc`
/// * `wire_radius_src`: (m) source filament radius, length `nsrc`
/// * `out`:             (H) inductive coupling to each target filament, length `ntgt`
pub fn inductance_linear_filaments(
    xyzfil_tgt: (&[f64], &[f64], &[f64]),
    dlxyzfil_tgt: (&[f64], &[f64], &[f64]),
    xyzfil_src: (&[f64], &[f64], &[f64]),
    dlxyzfil_src: (&[f64], &[f64], &[f64]),
    wire_radius_src: &[f64],
    out: &mut [f64],
) -> Result<(), &'static str> {
    let ntgt = xyzfil_tgt.0.len();
    check_length!(
        ntgt,
        xyzfil_tgt.0,
        xyzfil_tgt.1,
        xyzfil_tgt.2,
        dlxyzfil_tgt.0,
        dlxyzfil_tgt.1,
        dlxyzfil_tgt.2,
        out
    );

    let nsrc = xyzfil_src.0.len();
    check_length!(
        nsrc,
        xyzfil_src.0,
        xyzfil_src.1,
        xyzfil_src.2,
        dlxyzfil_src.0,
        dlxyzfil_src.1,
        dlxyzfil_src.2,
        wire_radius_src
    );

    out.fill(0.0);
    for j in 0..ntgt {
        out[j] = inductance_piecewise_linear_filaments(
            xyzfil_src,
            dlxyzfil_src,
            (
                &xyzfil_tgt.0[j..j + 1],
                &xyzfil_tgt.1[j..j + 1],
                &xyzfil_tgt.2[j..j + 1],
            ),
            (
                &dlxyzfil_tgt.0[j..j + 1],
                &dlxyzfil_tgt.1[j..j + 1],
                &dlxyzfil_tgt.2[j..j + 1],
            ),
            wire_radius_src,
        )?; // [H]
    }

    Ok(())
}

/// Inductive coupling matrix between disjoint source and target filament segments.
///
/// The output is row-major with shape `(nsrc, ntgt)`, where each row corresponds to one source
/// segment and each column corresponds to one target segment. Summing down the source rows for a
/// fixed target column recovers the contracted output from [`inductance_linear_filaments`].
///
/// # Arguments
///
/// * `xyzfil_tgt`:      (m) target filament origin coordinates, each length `ntgt`
/// * `dlxyzfil_tgt`:    (m) target filament segment lengths, each length `ntgt`
/// * `xyzfil_src`:      (m) source filament origin coordinates, each length `nsrc`
/// * `dlxyzfil_src`:    (m) source filament segment lengths, each length `nsrc`
/// * `wire_radius_src`: (m) source filament radius, length `nsrc`
/// * `out`:             (H) row-major `(nsrc, ntgt)` inductance matrix
pub fn inductance_linear_filaments_matrix(
    xyzfil_tgt: (&[f64], &[f64], &[f64]),
    dlxyzfil_tgt: (&[f64], &[f64], &[f64]),
    xyzfil_src: (&[f64], &[f64], &[f64]),
    dlxyzfil_src: (&[f64], &[f64], &[f64]),
    wire_radius_src: &[f64],
    out: &mut [f64],
) -> Result<(), &'static str> {
    let ntgt = xyzfil_tgt.0.len();
    check_length!(
        ntgt,
        xyzfil_tgt.0,
        xyzfil_tgt.1,
        xyzfil_tgt.2,
        dlxyzfil_tgt.0,
        dlxyzfil_tgt.1,
        dlxyzfil_tgt.2
    );

    let nsrc = xyzfil_src.0.len();
    check_length!(
        nsrc,
        xyzfil_src.0,
        xyzfil_src.1,
        xyzfil_src.2,
        dlxyzfil_src.0,
        dlxyzfil_src.1,
        dlxyzfil_src.2,
        wire_radius_src
    );

    let expected_len = nsrc
        .checked_mul(ntgt)
        .ok_or("Output size overflow in inductance_linear_filaments_matrix")?;
    check_length!(expected_len, out);

    out.fill(0.0);
    for i in 0..nsrc {
        let row = &mut out[i * ntgt..(i + 1) * ntgt];
        inductance_linear_filaments(
            xyzfil_tgt,
            dlxyzfil_tgt,
            (
                &xyzfil_src.0[i..i + 1],
                &xyzfil_src.1[i..i + 1],
                &xyzfil_src.2[i..i + 1],
            ),
            (
                &dlxyzfil_src.0[i..i + 1],
                &dlxyzfil_src.1[i..i + 1],
                &dlxyzfil_src.2[i..i + 1],
            ),
            &wire_radius_src[i..i + 1],
            row,
        )?;
    }

    Ok(())
}

/// Parallel inductive coupling matrix between disjoint source and target filament segments.
///
/// The output layout matches [`inductance_linear_filaments_matrix`].
///
/// # Arguments
///
/// * `xyzfil_tgt`:      (m) target filament origin coordinates, each length `ntgt`
/// * `dlxyzfil_tgt`:    (m) target filament segment lengths, each length `ntgt`
/// * `xyzfil_src`:      (m) source filament origin coordinates, each length `nsrc`
/// * `dlxyzfil_src`:    (m) source filament segment lengths, each length `nsrc`
/// * `wire_radius_src`: (m) source filament radius, length `nsrc`
/// * `out`:             (H) row-major `(nsrc, ntgt)` inductance matrix
pub fn inductance_linear_filaments_matrix_par(
    xyzfil_tgt: (&[f64], &[f64], &[f64]),
    dlxyzfil_tgt: (&[f64], &[f64], &[f64]),
    xyzfil_src: (&[f64], &[f64], &[f64]),
    dlxyzfil_src: (&[f64], &[f64], &[f64]),
    wire_radius_src: &[f64],
    out: &mut [f64],
) -> Result<(), &'static str> {
    let ntgt = xyzfil_tgt.0.len();
    check_length!(
        ntgt,
        xyzfil_tgt.0,
        xyzfil_tgt.1,
        xyzfil_tgt.2,
        dlxyzfil_tgt.0,
        dlxyzfil_tgt.1,
        dlxyzfil_tgt.2
    );

    let nsrc = xyzfil_src.0.len();
    check_length!(
        nsrc,
        xyzfil_src.0,
        xyzfil_src.1,
        xyzfil_src.2,
        dlxyzfil_src.0,
        dlxyzfil_src.1,
        dlxyzfil_src.2,
        wire_radius_src
    );

    let expected_len = nsrc
        .checked_mul(ntgt)
        .ok_or("Output size overflow in inductance_linear_filaments_matrix_par")?;
    check_length!(expected_len, out);

    out.fill(0.0);
    out.par_chunks_mut(ntgt)
        .enumerate()
        .try_for_each(|(i, row)| {
            inductance_linear_filaments(
                xyzfil_tgt,
                dlxyzfil_tgt,
                (
                    &xyzfil_src.0[i..i + 1],
                    &xyzfil_src.1[i..i + 1],
                    &xyzfil_src.2[i..i + 1],
                ),
                (
                    &dlxyzfil_src.0[i..i + 1],
                    &dlxyzfil_src.1[i..i + 1],
                    &dlxyzfil_src.2[i..i + 1],
                ),
                &wire_radius_src[i..i + 1],
                row,
            )
        })?;

    Ok(())
}

/// Validate filament geometry and a canonical CSC interaction pattern.
fn validate_sparse_inductance_inputs(
    xyzfil_tgt: (&[f64], &[f64], &[f64]),
    dlxyzfil_tgt: (&[f64], &[f64], &[f64]),
    xyzfil_src: (&[f64], &[f64], &[f64]),
    dlxyzfil_src: (&[f64], &[f64], &[f64]),
    wire_radius_src: &[f64],
    row_indices: &[usize],
    column_pointers: &[usize],
    out: &[f64],
) -> Result<(usize, usize), &'static str> {
    let ntgt = xyzfil_tgt.0.len();
    check_length!(
        ntgt,
        xyzfil_tgt.0,
        xyzfil_tgt.1,
        xyzfil_tgt.2,
        dlxyzfil_tgt.0,
        dlxyzfil_tgt.1,
        dlxyzfil_tgt.2
    );

    let nsrc = xyzfil_src.0.len();
    check_length!(
        nsrc,
        xyzfil_src.0,
        xyzfil_src.1,
        xyzfil_src.2,
        dlxyzfil_src.0,
        dlxyzfil_src.1,
        dlxyzfil_src.2,
        wire_radius_src
    );
    check_length!(row_indices.len(), out);

    if column_pointers.len() != ntgt + 1 {
        return Err("CSC column pointer length must equal target count plus one");
    }
    if column_pointers.first() != Some(&0) {
        return Err("CSC column pointers must start at zero");
    }
    if column_pointers.last() != Some(&row_indices.len()) {
        return Err("CSC final column pointer must equal the stored-entry count");
    }
    if column_pointers
        .windows(2)
        .any(|pointers| pointers[0] > pointers[1])
    {
        return Err("CSC column pointers must be nondecreasing");
    }

    for column in 0..ntgt {
        let rows = &row_indices[column_pointers[column]..column_pointers[column + 1]];
        if rows.iter().any(|&row| row >= nsrc) {
            return Err("CSC row index exceeds the source count");
        }
        if rows.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("CSC row indices must be sorted and unique within each column");
        }
    }

    Ok((nsrc, ntgt))
}

/// Evaluate one stored CSC source-target interaction.
#[inline]
fn inductance_linear_filaments_sparse_entry(
    xyzfil_tgt: (&[f64], &[f64], &[f64]),
    dlxyzfil_tgt: (&[f64], &[f64], &[f64]),
    xyzfil_src: (&[f64], &[f64], &[f64]),
    dlxyzfil_src: (&[f64], &[f64], &[f64]),
    wire_radius_src: &[f64],
    target: usize,
    source: usize,
) -> f64 {
    inductance_linear_filament_pair(
        (
            xyzfil_src.0[source],
            xyzfil_src.1[source],
            xyzfil_src.2[source],
        ),
        (
            dlxyzfil_src.0[source],
            dlxyzfil_src.1[source],
            dlxyzfil_src.2[source],
        ),
        wire_radius_src[source],
        (
            xyzfil_tgt.0[target],
            xyzfil_tgt.1[target],
            xyzfil_tgt.2[target],
        ),
        (
            dlxyzfil_tgt.0[target],
            dlxyzfil_tgt.1[target],
            dlxyzfil_tgt.2[target],
        ),
    )
}

/// Evaluate selected source-target filament inductances into CSC value storage.
///
/// `row_indices` and `column_pointers` describe a canonical CSC matrix with shape
/// `(nsrc, ntgt)`, where source segments are rows and target segments are columns. Each stored
/// coordinate is evaluated with the finite-radius source kernel and three-point Gauss--Legendre
/// integration over the complete target segment. `out` has one value per stored coordinate and
/// retains explicit numerical zeros.
pub fn inductance_linear_filaments_sparse_csc(
    xyzfil_tgt: (&[f64], &[f64], &[f64]),
    dlxyzfil_tgt: (&[f64], &[f64], &[f64]),
    xyzfil_src: (&[f64], &[f64], &[f64]),
    dlxyzfil_src: (&[f64], &[f64], &[f64]),
    wire_radius_src: &[f64],
    row_indices: &[usize],
    column_pointers: &[usize],
    out: &mut [f64],
) -> Result<(), &'static str> {
    let (_, ntgt) = validate_sparse_inductance_inputs(
        xyzfil_tgt,
        dlxyzfil_tgt,
        xyzfil_src,
        dlxyzfil_src,
        wire_radius_src,
        row_indices,
        column_pointers,
        out,
    )?;

    for target in 0..ntgt {
        for entry in column_pointers[target]..column_pointers[target + 1] {
            out[entry] = inductance_linear_filaments_sparse_entry(
                xyzfil_tgt,
                dlxyzfil_tgt,
                xyzfil_src,
                dlxyzfil_src,
                wire_radius_src,
                target,
                row_indices[entry],
            );
        }
    }

    Ok(())
}

/// Parallel variant of [`inductance_linear_filaments_sparse_csc`].
pub fn inductance_linear_filaments_sparse_csc_par(
    xyzfil_tgt: (&[f64], &[f64], &[f64]),
    dlxyzfil_tgt: (&[f64], &[f64], &[f64]),
    xyzfil_src: (&[f64], &[f64], &[f64]),
    dlxyzfil_src: (&[f64], &[f64], &[f64]),
    wire_radius_src: &[f64],
    row_indices: &[usize],
    column_pointers: &[usize],
    out: &mut [f64],
) -> Result<(), &'static str> {
    validate_sparse_inductance_inputs(
        xyzfil_tgt,
        dlxyzfil_tgt,
        xyzfil_src,
        dlxyzfil_src,
        wire_radius_src,
        row_indices,
        column_pointers,
        out,
    )?;

    // Partition by stored entries instead of columns so a single dense column still uses all
    // workers. Each chunk locates its first target once, then follows the CSC boundaries linearly.
    let entry_chunk_size = chunksize(out.len());
    out.par_chunks_mut(entry_chunk_size)
        .enumerate()
        .for_each(|(chunk_id, values)| {
            let entry_start = chunk_id * entry_chunk_size;
            let mut target = column_pointers.partition_point(|&pointer| pointer <= entry_start) - 1;
            for (offset, value) in values.iter_mut().enumerate() {
                let entry = entry_start + offset;
                while column_pointers[target + 1] <= entry {
                    target += 1;
                }
                *value = inductance_linear_filaments_sparse_entry(
                    xyzfil_tgt,
                    dlxyzfil_tgt,
                    xyzfil_src,
                    dlxyzfil_src,
                    wire_radius_src,
                    target,
                    row_indices[entry],
                );
            }
        });

    Ok(())
}

/// Biot-Savart calculation for B-field contribution from many current filament
/// segments to many observation points.
///
/// Uses filament midpoint as field source.
///
/// This variant of the function is parallelized over chunks of observation points.
///
/// # Arguments
///
/// * `xyzp`:     (m) Observation point coords, each length `n`
/// * `xyzfil`:   (m) Filament origin coords (start of segment), each length `m`
/// * `dlxyzfil`: (m) Filament segment length deltas, each length `m`
/// * `ifil`:     (A) Filament current, length `m`
/// * `wire_radius`: (m) (Half-) thickness of conductor, length `m`
/// * `out`:      (T) bx, by, bz at observation points, each length `n`
pub fn flux_density_linear_filament_par(
    xyzp: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
    wire_radius: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    // Chunk inputs
    let n = chunksize(xyzp.0.len());
    let (xpc, ypc, zpc) = par_chunks_3tup!(xyzp, n);
    let (bxc, byc, bzc) = mut_par_chunks_3tup!(out, n);

    // Run calcs
    (bxc, byc, bzc, xpc, ypc, zpc)
        .into_par_iter()
        .try_for_each(|(bx, by, bz, xp, yp, zp)| {
            flux_density_linear_filament(
                (xp, yp, zp),
                xyzfil,
                dlxyzfil,
                ifil,
                wire_radius,
                (bx, by, bz),
            )
        })?;

    Ok(())
}

/// Biot-Savart calculation for B-field contribution from many current filament
/// segments to many observation points, returned as explicit target-source matrices.
///
/// Each output array is row-major with shape `(nobs, nfil)`, where each row corresponds
/// to one observation point and each column corresponds to one source segment. The source
/// currents are applied before writing the matrix entries, so summing each row recovers
/// the contracted output from [`flux_density_linear_filament`].
///
/// # Arguments
///
/// * `xyzp`:     (m) Observation point coords, each length `nobs`
/// * `xyzfil`:   (m) Filament origin coords (start of segment), each length `nfil`
/// * `dlxyzfil`: (m) Filament segment length deltas, each length `nfil`
/// * `ifil`:     (A) Filament current, length `nfil`
/// * `wire_radius`: (m) conductor radius, length `nfil`
/// * `out`:      (T) row-major `(nobs, nfil)` Bx, By, Bz matrices
pub fn flux_density_linear_filament_matrix_par(
    xyzp: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
    wire_radius: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let nfil = xyzfil.0.len();
    let nobs = xyzp.0.len();
    if nobs == 0 || nfil == 0 {
        return flux_density_linear_filament_matrix(xyzp, xyzfil, dlxyzfil, ifil, wire_radius, out);
    }

    let n = chunksize(nobs);
    (
        xyzp.0.par_chunks(n),
        xyzp.1.par_chunks(n),
        xyzp.2.par_chunks(n),
        out.0.par_chunks_mut(n * nfil),
        out.1.par_chunks_mut(n * nfil),
        out.2.par_chunks_mut(n * nfil),
    )
        .into_par_iter()
        .try_for_each(|(xp, yp, zp, bx, by, bz)| {
            flux_density_linear_filament_matrix(
                (xp, yp, zp),
                xyzfil,
                dlxyzfil,
                ifil,
                wire_radius,
                (bx, by, bz),
            )
        })?;

    Ok(())
}

/// Biot-Savart calculation for B-field contribution from many current filament
/// segments to many observation points.
///
/// Uses filament midpoint as field source.
///
/// # Arguments
///
/// * `xyzp`:     (m) Observation point coords, each length `n`
/// * `xyzfil`:   (m) Filament origin coords (start of segment), each length `m`
/// * `dlxyzfil`: (m) Filament segment length deltas, each length `m`
/// * `ifil`:     (A) Filament current, length `m`
/// * `wire_radius`: (m) (Half-) thickness of conductor, length `m`
/// * `out`:      (T) bx, by, bz at observation points, each length `n`
pub fn flux_density_linear_filament(
    xyzp: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
    wire_radius: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    // Unpack
    let (xp, yp, zp) = xyzp;
    let (xfil, yfil, zfil) = xyzfil;
    let (dlxfil, dlyfil, dlzfil) = dlxyzfil;

    let (bx, by, bz) = out;

    // Check lengths; if there is any possibility of mismatch,
    // the compiler will bypass vectorization
    let n = xfil.len();
    let m = xp.len();
    check_length!(m, xp, yp, zp, bx, by, bz);
    check_length!(
        n,
        xfil,
        yfil,
        zfil,
        dlxfil,
        dlyfil,
        dlzfil,
        ifil,
        wire_radius
    );

    // Zero output
    bx.fill(0.0);
    by.fill(0.0);
    bz.fill(0.0);

    // For each filament, evaluate the contribution to each observation point.
    //
    // The inner function is inlined, so values that are reused between iterations
    // can be pulled to the outer scope by the compiler and do not affect performance.
    for i in 0..n {
        for j in 0..m {
            // Filament
            let fil0 = (xfil[i], yfil[i], zfil[i]); // [m] start point
            let fil1 = (fil0.0 + dlxfil[i], fil0.1 + dlyfil[i], fil0.2 + dlzfil[i]); // [m] end point
            let current = ifil[i];

            // Observation point
            let obs = (xp[j], yp[j], zp[j]); // [m]

            // Field contributions
            let (bxc, byc, bzc) =
                flux_density_linear_filament_scalar((fil0, fil1, current), wire_radius[i], obs);
            bx[j] += bxc;
            by[j] += byc;
            bz[j] += bzc;
        }
    }

    Ok(())
}

/// Biot-Savart calculation for B-field contribution from many current filament
/// segments to many observation points, returned as explicit target-source matrices.
///
/// Each output array is row-major with shape `(nobs, nfil)`, where each row corresponds
/// to one observation point and each column corresponds to one source segment. The source
/// currents are applied before writing the matrix entries, so summing each row recovers
/// the contracted output from [`flux_density_linear_filament`].
///
/// # Arguments
///
/// * `xyzp`:     (m) Observation point coords, each length `nobs`
/// * `xyzfil`:   (m) Filament origin coords (start of segment), each length `nfil`
/// * `dlxyzfil`: (m) Filament segment length deltas, each length `nfil`
/// * `ifil`:     (A) Filament current, length `nfil`
/// * `wire_radius`: (m) conductor radius, length `nfil`
/// * `out`:      (T) row-major `(nobs, nfil)` Bx, By, Bz matrices
pub fn flux_density_linear_filament_matrix(
    xyzp: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
    wire_radius: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let (xp, yp, zp) = xyzp;
    let (xfil, yfil, zfil) = xyzfil;
    let (dlxfil, dlyfil, dlzfil) = dlxyzfil;
    let (bx, by, bz) = out;

    let nfil = xfil.len();
    let nobs = xp.len();
    check_length!(nobs, xp, yp, zp);
    check_length!(
        nfil,
        xfil,
        yfil,
        zfil,
        dlxfil,
        dlyfil,
        dlzfil,
        ifil,
        wire_radius
    );

    let expected_len = nobs
        .checked_mul(nfil)
        .ok_or("Output size overflow in flux_density_linear_filament_matrix")?;
    check_length!(expected_len, bx, by, bz);

    bx.fill(0.0);
    by.fill(0.0);
    bz.fill(0.0);

    for j in 0..nobs {
        let obs = (xp[j], yp[j], zp[j]); // [m]
        let row = j * nfil;
        for i in 0..nfil {
            let fil0 = (xfil[i], yfil[i], zfil[i]); // [m]
            let fil1 = (fil0.0 + dlxfil[i], fil0.1 + dlyfil[i], fil0.2 + dlzfil[i]); // [m]
            let current = ifil[i]; // [A]
            let (bxc, byc, bzc) =
                flux_density_linear_filament_scalar((fil0, fil1, current), wire_radius[i], obs);
            bx[row + i] = bxc; // [T]
            by[row + i] = byc; // [T]
            bz[row + i] = bzc; // [T]
        }
    }

    Ok(())
}

/// Biot-Savart calculation for B-field contribution from one filament
/// to one observation point.
///
/// Uses the formula for continuous current distribution and finite wire thickness.
/// Inside the wire radius, the field blends linearly to zero at the center.
///
/// Draws from Griffiths eq'n 5.37 and Zahn eq'n 5.4.17 with inspiration from
/// rat-mlfmm's van Lanen kernel to replace the expensive sine functions with geometric
/// equivalents and to provide handling of the singularity near the filament axis.
///
/// ```text
///        p (target)
///        *
///       /|\
///      / | \
///   ap/  |  \bp
///    / ∠a|∠b \
///   /    |    \
///  a-----m-----b  -> I  
///```
///
/// The base formula is
///
/// $ |B| = \frac{\mu_0 I}{4 \pi r_\perp}  (sin(\theta_b) - sin(\theta_a)) $
///
/// with the direction determined by $ \hat{dL} \times \hat r_\perp $ with $r_\perp$ defined perpendicular
/// from the axis of the filament to the target point.
///
/// The otherwise-expensive sine functions are evaluated directly using distance magnitudes.
///
/// Inside the wire radius, the formula is modified to blend linearly to zero at the wire center,
/// which is physically consistent with the field of a cylinder of uniform current density.
///
/// ## References
///
/// * \[1\] D. J. Griffiths, Introduction to electrodynamics, Fourth edition. Boston: Pearson, 2014.
/// * \[2\] J. van Nugteren and N. Deelen, “rat-mlfmm,” GitLab repository. Accessed: Jan. 16, 2026. \[Online\].
///         Available: <https://gitlab.com/Project-Rat/rat-mlfmm/-/tree/1e1d387522fafac50c0540af1ebb15d1d506d33d>
/// * \[3\] M. Zahn, “5.4: The Vector Potential,” Engineering LibreTexts. Accessed: Jan. 20, 2026. \[Online\].
///         Available: <https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Electro-Optics/Electromagnetic_Field_Theory%3A_A_Problem_Solving_Approach_(Zahn)/05%3A_The_Magnetic_Field/5.04%3A_The_Vector_Potential>
///
/// # Arguments
///
/// * `xyzifil`:   (m, m, A) Filament start and end coords and current
/// * `wire_radius`: (m) (Half-) thickness of conductor.
/// * `xyzp`:     (m) Observation point coords
///
/// # Returns
///
/// * `b`:        (T) Magnetic flux density (B-field)
#[inline]
pub fn flux_density_linear_filament_scalar<T: Scalar>(
    xyzifil: ((T, T, T), (T, T, T), T),
    wire_radius: T,
    xyzobs: (T, T, T),
) -> (T, T, T) {
    // Unpack
    let (start, end, ifil) = xyzifil;

    // Get perpendicular distance and distance from each endpoint to the target,
    // and a fraction between 0 and 1 representing finite-thickness blending:
    // 0 at the wire axis, 1 at/above wire radius.
    let PointLineDistance {
        perp,
        perp_hat,
        dist_a,
        dist_b,
        frac,
        para_a,
        para_b,
        ab_norm: dlhat,
    } = point_line_distance_with_endpoints(
        [start.0, start.1, start.2],
        [end.0, end.1, end.2],
        [xyzobs.0, xyzobs.1, xyzobs.2],
        wire_radius,
    );

    // Sine of the angle formed by the lines from the target to each endpoint
    // and the line of the filament.
    let sin_theta_a = para_a / dist_a; // (dimensionless)
    let sin_theta_b = para_b / dist_b; // (dimensionless)

    // Geometric component of B-field magnitude,
    // including linear falloff inside finite-thickness wire.
    let kappa =
        (T::ZERO - crate::math::cast::<T>(MU0_OVER_4PI)) * ifil * (sin_theta_b - sin_theta_a); // (V-s/m)

    // This factor is constant across all x, y, and z components.
    let c = frac * kappa / perp; // (A/m)

    // Direction of cross(dL, r), the direction of the field.
    let cxyz = cross3(dlhat, perp_hat); // (dimensionless)

    // Assemble final B-field components.
    let bx = c * cxyz[0]; // [T]
    let by = c * cxyz[1]; // [T]
    let bz = c * cxyz[2]; // [T]

    // Finally, determine whether we are clipping to zero.
    if frac > crate::math::cast::<T>(1e6 * f64::EPSILON)
        && perp > crate::math::cast::<T>(MIN_WIRE_THICKNESS)
    {
        (bx, by, bz)
    } else {
        (T::ZERO, T::ZERO, T::ZERO)
    }
}

/// Vector potential calculation for A-field contribution from many current filament
/// segments to many observation points.
///
/// Uses filament midpoint as field source.
///
/// This variant of the function is parallelized over chunks of observation points.
///
/// # Arguments
///
/// * `xyzp`:     (m) Observation point coords, each length `n`
/// * `xyzfil`:   (m) Filament origin coords (start of segment), each length `m`
/// * `dlxyzfil`: (m) Filament segment length deltas, each length `m`
/// * `ifil`:     (A) Filament current, length `m`
/// * `wire_radius`: (m) (Half-) thickness of conductor, length `m`
/// * `out`:      (V-s/m) ax, ay, az at observation points, each length `n`
pub fn vector_potential_linear_filament_par(
    xyzp: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
    wire_radius: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    // Chunk inputs
    let n = chunksize(xyzp.0.len());
    let (xpc, ypc, zpc) = par_chunks_3tup!(xyzp, n);
    let (bxc, byc, bzc) = mut_par_chunks_3tup!(out, n);

    // Run calcs
    (bxc, byc, bzc, xpc, ypc, zpc)
        .into_par_iter()
        .try_for_each(|(bx, by, bz, xp, yp, zp)| {
            vector_potential_linear_filament(
                (xp, yp, zp),
                xyzfil,
                dlxyzfil,
                ifil,
                wire_radius,
                (bx, by, bz),
            )
        })?;

    Ok(())
}

/// Vector potential calculation for A-field contribution from many current filament
/// segments to many observation points, returned as explicit target-source matrices.
///
/// Each output array is row-major with shape `(nobs, nfil)`, where each row corresponds
/// to one observation point and each column corresponds to one source segment. The source
/// currents are applied before writing the matrix entries, so summing each row recovers
/// the contracted output from [`vector_potential_linear_filament`].
///
/// # Arguments
///
/// * `xyzp`:     (m) Observation point coords, each length `nobs`
/// * `xyzfil`:   (m) Filament origin coords (start of segment), each length `nfil`
/// * `dlxyzfil`: (m) Filament segment length deltas, each length `nfil`
/// * `ifil`:     (A) Filament current, length `nfil`
/// * `wire_radius`: (m) conductor radius, length `nfil`
/// * `out`:      (V-s/m) row-major `(nobs, nfil)` Ax, Ay, Az matrices
pub fn vector_potential_linear_filament_matrix_par(
    xyzp: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
    wire_radius: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let nfil = xyzfil.0.len();
    let nobs = xyzp.0.len();
    if nobs == 0 || nfil == 0 {
        return vector_potential_linear_filament_matrix(
            xyzp,
            xyzfil,
            dlxyzfil,
            ifil,
            wire_radius,
            out,
        );
    }

    let n = chunksize(nobs);
    (
        xyzp.0.par_chunks(n),
        xyzp.1.par_chunks(n),
        xyzp.2.par_chunks(n),
        out.0.par_chunks_mut(n * nfil),
        out.1.par_chunks_mut(n * nfil),
        out.2.par_chunks_mut(n * nfil),
    )
        .into_par_iter()
        .try_for_each(|(xp, yp, zp, ax, ay, az)| {
            vector_potential_linear_filament_matrix(
                (xp, yp, zp),
                xyzfil,
                dlxyzfil,
                ifil,
                wire_radius,
                (ax, ay, az),
            )
        })?;

    Ok(())
}

/// Vector potential calculation for A-field contribution from many current filament
/// segments to many observation points.
///
/// Uses filament midpoint as field source.
///
/// # Arguments
///
/// * `xyzp`:     (m) Observation point coords, each length `n`
/// * `xyzfil`:   (m) Filament origin coords (start of segment), each length `m`
/// * `dlxyzfil`: (m) Filament segment length deltas, each length `m`
/// * `ifil`:     (A) Filament current, length `m`
/// * `wire_radius`: (m) (Half-) thickness of conductor, length `m`
/// * `out`:      (V-s/m) ax, ay, az at observation points, each length `n`
pub fn vector_potential_linear_filament(
    xyzp: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
    wire_radius: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    // Unpack
    let (xp, yp, zp) = xyzp;
    let (xfil, yfil, zfil) = xyzfil;
    let (dlxfil, dlyfil, dlzfil) = dlxyzfil;

    let (ax, ay, az) = out;

    // Check lengths; if there is any possibility of mismatch,
    // the compiler will bypass vectorization
    let n = xfil.len();
    let m = xp.len();
    check_length!(m, xp, yp, zp, ax, ay, az);
    check_length!(
        n,
        xfil,
        yfil,
        zfil,
        dlxfil,
        dlyfil,
        dlzfil,
        ifil,
        wire_radius
    );

    // Zero output
    ax.fill(0.0);
    ay.fill(0.0);
    az.fill(0.0);

    // For each filament, evaluate the contribution to each observation point.
    //
    // The inner function is inlined, so values that are reused between iterations
    // can be pulled to the outer scope by the compiler and do not affect performance.
    for i in 0..n {
        for j in 0..m {
            // Filament
            let fil0 = (xfil[i], yfil[i], zfil[i]); // [m] start point
            let fil1 = (fil0.0 + dlxfil[i], fil0.1 + dlyfil[i], fil0.2 + dlzfil[i]); // [m] end point
            let current = ifil[i];

            // Observation point
            let obs = (xp[j], yp[j], zp[j]); // [m]

            // Field contributions
            let (axc, ayc, azc) =
                vector_potential_linear_filament_scalar((fil0, fil1, current), wire_radius[i], obs);
            ax[j] += axc;
            ay[j] += ayc;
            az[j] += azc;
        }
    }

    Ok(())
}

/// Vector potential calculation for A-field contribution from many current filament
/// segments to many observation points, returned as explicit target-source matrices.
///
/// Each output array is row-major with shape `(nobs, nfil)`, where each row corresponds
/// to one observation point and each column corresponds to one source segment. The source
/// currents are applied before writing the matrix entries, so summing each row recovers
/// the contracted output from [`vector_potential_linear_filament`].
///
/// # Arguments
///
/// * `xyzp`:     (m) Observation point coords, each length `nobs`
/// * `xyzfil`:   (m) Filament origin coords (start of segment), each length `nfil`
/// * `dlxyzfil`: (m) Filament segment length deltas, each length `nfil`
/// * `ifil`:     (A) Filament current, length `nfil`
/// * `wire_radius`: (m) conductor radius, length `nfil`
/// * `out`:      (V-s/m) row-major `(nobs, nfil)` Ax, Ay, Az matrices
pub fn vector_potential_linear_filament_matrix(
    xyzp: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
    wire_radius: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let (xp, yp, zp) = xyzp;
    let (xfil, yfil, zfil) = xyzfil;
    let (dlxfil, dlyfil, dlzfil) = dlxyzfil;
    let (ax, ay, az) = out;

    let nfil = xfil.len();
    let nobs = xp.len();
    check_length!(nobs, xp, yp, zp);
    check_length!(
        nfil,
        xfil,
        yfil,
        zfil,
        dlxfil,
        dlyfil,
        dlzfil,
        ifil,
        wire_radius
    );

    let expected_len = nobs
        .checked_mul(nfil)
        .ok_or("Output size overflow in vector_potential_linear_filament_matrix")?;
    check_length!(expected_len, ax, ay, az);

    ax.fill(0.0);
    ay.fill(0.0);
    az.fill(0.0);

    for j in 0..nobs {
        let obs = (xp[j], yp[j], zp[j]); // [m]
        let row = j * nfil;
        for i in 0..nfil {
            let fil0 = (xfil[i], yfil[i], zfil[i]); // [m]
            let fil1 = (fil0.0 + dlxfil[i], fil0.1 + dlyfil[i], fil0.2 + dlzfil[i]); // [m]
            let current = ifil[i]; // [A]
            let (axc, ayc, azc) =
                vector_potential_linear_filament_scalar((fil0, fil1, current), wire_radius[i], obs);
            ax[row + i] = axc; // [V-s/m]
            ay[row + i] = ayc; // [V-s/m]
            az[row + i] = azc; // [V-s/m]
        }
    }

    Ok(())
}

/// Vector potential (A-field) from a linear current filament segment to an observation point.
///
/// Uses the formula for finite segment length and finite wire thickness.
///
/// Because an infinitesimally-thick wire produces a nonphysical singularity
/// at the axis, a minimum wire radius of `MIN_WIRE_THICKNESS` is imposed.
///
/// The base formula implemented here is:
///
/// $$
/// A_z
/// = \frac{\mu_0 I}{4\pi}\int_{-L/2}^{L/2}
/// \frac{dz'}{\sqrt{(z-z')^2+r^2}}
/// = \frac{\mu_0 I}{4\pi}\ln(
/// \frac{
/// -z + \frac{L}{2} + \sqrt{(z-\frac{L}{2})^2 + r^2}
/// }{
/// -(z+\frac{L}{2}) + \sqrt{(z+\frac{L}{2})^2 + r^2}
/// }
/// )
/// $$
///
/// This has been manipulated to formulate in terms of components of the distance from the
/// filament endpoints and filament axis to the target point:
///
/// $$ k1 = -||bp_\parallel|| + ||bp|| $$
/// $$ k2 = -||ap_\parallel|| + ||ap|| $$
/// $$ A_\parallel = \frac{\mu_0 I}{4 \pi} \ln (\frac{k1}{k2}) $$
///
/// # Arguments
///
/// * `xyzifil`:   (m, m, A) Filament start and end coords and current
/// * `xyzobs`:     (m) Observation point coords
///
/// # Returns
///
/// * `a`:        (V-s/m) Vector potential x, y, z components
#[inline]
pub fn vector_potential_linear_filament_scalar<T: Scalar>(
    xyzifil: ((T, T, T), (T, T, T), T),
    wire_radius: T,
    xyzobs: (T, T, T),
) -> (T, T, T) {
    use crate::math::{PointLineDistance, point_line_distance_with_endpoints};

    // Unpack
    let (start, end, ifil) = xyzifil;

    // Regularize the line-filament singularity with a minimum core radius.
    let core_radius = wire_radius.max(crate::math::cast::<T>(MIN_WIRE_THICKNESS));

    // Get perpendicular distance and distance from each endpoint to the target,
    // and a fraction between 0 and 1 representing finite-thickness blending:
    // 0 at the wire axis, 1 at/above wire radius.
    let PointLineDistance {
        perp,
        dist_a,
        dist_b,
        frac,
        para_a,
        para_b,
        ab_norm: dlhat,
        perp_hat: _,
    } = point_line_distance_with_endpoints(
        [start.0, start.1, start.2],
        [end.0, end.1, end.2],
        [xyzobs.0, xyzobs.1, xyzobs.2],
        core_radius,
    );

    // Sine of the angle formed by the lines from the target to each endpoint
    // and the line of the filament.
    let sin_theta_a = para_a / dist_a; // (dimensionless)
    let sin_theta_b = para_b / dist_b; // (dimensionless)

    // Geometric component of B-field magnitude,
    // including linear falloff inside finite-thickness wire.
    let kappa =
        (T::ZERO - crate::math::cast::<T>(MU0_OVER_4PI)) * ifil * (sin_theta_b - sin_theta_a); // (V-s/m)

    // NOTE: up to this point, this has been the same as the B-field calculation.

    // Finite segment length log-form for thin filament.
    // If the raw observation point location is inside the conductor,
    // then it is clipped to the nearest point on the surface of the conductor.
    //
    // The field shape inside the conductor is handled later; this separation
    // is necessary due to the discontinuity in the current density vector field.
    let perp2 = perp * perp;
    let k1 = if para_b >= T::ZERO {
        // Each branch is equal, but numerically stable in different regimes.
        //
        // To keep the argument of the log term from going to zero near the
        // axis where the normed distance is almost the same as the parallel distance,
        // we need to come up with a way of phrasing the difference between them that
        // puts a value in both the numerator and denominator that is clamped to wire radius.
        //
        // To do this, we can do the rationalization trick to convert an equation of the form
        // `sqrt(x^2 + y^2) - x`, which goes to zero when `y` goes to zero because both `x` and `y` are clamped,
        // to
        // `sqrt(x^2 + y^2) - x = (sqrt(x^2 + y^2) - x) * (sqrt(x^2 + y^2) + x) / (sqrt(x^2 + y^2) + x)`
        // then simplifying the numerator and leaving the denominator as-is to produce
        // `sqrt(x^2 + y^2) - x = y^2 / (sqrt(x^2 + y^2) + x)`, which becomes useful because
        // `y` is the perpendicular distance, here, which is clamped. So, after this transformation,
        // we have the algebraic equivalent of the original expression, but it never goes to either
        // zero or infinity as long as the parallel distance is nonnegative.
        //
        // So, for our formula, this reads as:
        // `dist - para = (dist - para)(dist + para) / (dist + para)` (first rationalization)
        // `dist^2 = para^2 + perp^2` (just expanding the existing formula)
        // `(dist - para)(dist + para) = dist^2 - para^2 = perp^2` (substitute into numerator of rationalized expression)
        // `dist - para = perp^2 / (dist + para)` (final rationalized expression)
        perp2 / (dist_b + para_b)
    } else {
        // The rationalized expression works as long as the parallel distance is always positive
        // so that it can't cancel out `dist`; because `dist` is always positive, when
        // `para` is negative, we can switch back to the original formula, which subtracts a
        // negative value from a positive value and always produces a positive value.
        //
        // This way, both regimes produce a strictly positive value for `k` that guarantees
        // `log(k1/k2)` is nonsingular everywhere.
        dist_b - para_b
    }; // (m)
    let k2 = if para_a >= T::ZERO {
        // Each branch is equal, but numerically stable in different regimes
        perp2 / (dist_a + para_a)
    } else {
        dist_a - para_a
    }; // (m)
    let a_edge = crate::math::cast::<T>(MU0_OVER_4PI)
        * ifil
        * crate::math::cast::<T>(libm::log((k1 / k2).max(T::ZERO).to_f64())); // (V-s/m)

    // Finite-thickness effect for points inside the conductor or near the endpoints.
    //
    // Because the B-field is the curl of the A-field, where the B-field falls off
    // linearly inside the conductor, the A-field falls off quadratically.
    // This constrains the _form_ of the variation of A inside the conductor, but leaves
    // a free integration constant known as the gauge.
    //
    // Outside the conductor, the gauge is already chosen consistent with both curl(A)=B
    // and with the evaluation of mutual inductances and voltages due to dA/dt.
    // This is easy because it is a constant outside the conductor, where the enclosed
    // current is not changing.
    //
    // However, inside the conductor, some extra attention to the gauge is needed
    // in order to ensure that the field is both continuous at the edge and
    // consistent with curl(A)=B under changing enclosed current.
    //
    // Here, we use a gauge shift (kappa, the factor shared with the B-field calc) to
    // ensure the A-field inside the conductor is consistent with curl(A)=B
    // both inside and outside the conductor.

    // (dimensionless) Quadratic fall-off (vs. linear for B-field)
    let blend = crate::math::cast::<T>(0.5) * frac.mul_add(T::ZERO - frac, T::ONE); //  1/2 (1 - frac^2)
    // (V-s/m) a_mag = a_edge + 0.5 * kappa * (1 - frac^2), reworked for mul_add
    let a_mag = kappa.mul_add(blend, a_edge); // (V-s/m) Gauge-shifted magnitude

    // Direction is always aligned with the segment.
    // (V-s) final vector potential
    let (ax, ay, az) = (a_mag * dlhat[0], a_mag * dlhat[1], a_mag * dlhat[2]);

    // Return continuous vector potential; avoid hard clipping at small radius.
    (ax, ay, az)
}

/// JxB (Lorentz) body force density (per volume) due to a linear current
/// filament segment at an observation point with some current density (per area).
///
/// Uses filament midpoint as field source.
///
/// # Arguments
///
/// * `xyzifil`:   (m, m, A) Filament start and end coords and current
/// * `wire_radius`: (m) (Half-) thickness of conductor.
/// * `xyzobs`:    (m) Observation point coords
/// * `jobs`:      (A/m^2) Current density vector at observation point
///
/// # Returns
///
/// * `jxb`:        (N/m^3) Body force density
pub fn body_force_density_linear_filament_scalar(
    xyzifil: ((f64, f64, f64), (f64, f64, f64), f64),
    wire_radius: f64,
    xyzobs: (f64, f64, f64),
    jobs: (f64, f64, f64),
) -> (f64, f64, f64) {
    // Get magnetic flux density at target point
    let (bx, by, bz) = flux_density_linear_filament_scalar(xyzifil, wire_radius, xyzobs); // [T]

    // Take JxB Lorentz force
    let out = cross3([jobs.0, jobs.1, jobs.2], [bx, by, bz]); // [N/m^3]
    (out[0], out[1], out[2])
}

/// JxB (Lorentz) body force density (per volume) due to a linear current
/// filament segment at an observation point with some current density (per area).
///
/// Uses filament midpoint as field source.
///
/// # Arguments
///
/// * `xyzifil`:   (m, m, A) Filament start and end coords and current
/// * `dlxyzfil`:  (m) Filament segment length deltas, each length `m`
/// * `ifil`:      (A) Filament current, length `m`
/// * `wire_radius`: (m) (Half-) thickness of conductor, length `m`
/// * `xyzobs`:    (m) Observation point coords
/// * `jobs`:      (A/m^2) Current density vector at observation point
/// * `out`:       (N/m^3) Body force density x, y, z components
pub fn body_force_density_linear_filament(
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
    wire_radius: &[f64],
    xyzobs: (&[f64], &[f64], &[f64]),
    jobs: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    // Unpack
    let (xp, yp, zp) = xyzobs;
    let (jx, jy, jz) = jobs;
    let (xfil, yfil, zfil) = xyzfil;
    let (dlxfil, dlyfil, dlzfil) = dlxyzfil;

    let (outx, outy, outz) = out;

    // Check lengths; if there is any possibility of mismatch,
    // the compiler will bypass vectorization
    let n = xfil.len();
    let m = xp.len();

    check_length!(
        n,
        xfil,
        yfil,
        zfil,
        dlxfil,
        dlyfil,
        dlzfil,
        ifil,
        wire_radius
    );
    check_length!(m, xp, yp, zp, jx, jy, jz, outx, outy, outz);

    // Zero output
    outx.fill(0.0);
    outy.fill(0.0);
    outz.fill(0.0);

    // For each filament, evaluate the contribution to each observation point.
    //
    // The inner function is inlined, so values that are reused between iterations
    // can be pulled to the outer scope by the compiler and do not affect performance.
    for i in 0..n {
        for j in 0..m {
            // Geometry
            let fil0 = (xfil[i], yfil[i], zfil[i]); // [m] this filament start
            let fil1 = (fil0.0 + dlxfil[i], fil0.1 + dlyfil[i], fil0.2 + dlzfil[i]); // [m] this filament end
            let obs = (xp[j], yp[j], zp[j]); // [m] this observation point
            let jj = (jx[j], jy[j], jz[j]); // [A/m^2] current density vector at obs point

            // [V-s/m] vector potential contribution of this filament to this observation point
            let (jxbx, jxby, jxbz) = body_force_density_linear_filament_scalar(
                (fil0, fil1, ifil[i]),
                wire_radius[i],
                obs,
                jj,
            );
            outx[j] += jxbx;
            outy[j] += jxby;
            outz[j] += jxbz;
        }
    }

    Ok(())
}

/// JxB (Lorentz) body force density (per volume) due to a linear current
/// filament segment at an observation point with some current density (per area).
///
/// Uses filament midpoint as field source.
///
/// This variant is parallelized over chunks of observation points.
///
/// # Arguments
///
/// * `xyzifil`:   (m, m, A) Filament start and end coords and current
/// * `dlxyzfil`:  (m) Filament segment length deltas, each length `m`
/// * `ifil`:      (A) Filament current, length `m`
/// * `wire_radius`: (m) (Half-) thickness of conductor, length `m`
/// * `xyzobs`:    (m) Observation point coords
/// * `jobs`:      (A/m^2) Current density vector at observation point
/// * `out`:       (N/m^3) Body force density x, y, z components
pub fn body_force_density_linear_filament_par(
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
    wire_radius: &[f64],
    xyzobs: (&[f64], &[f64], &[f64]),
    jobs: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    // Chunk inputs
    let n = chunksize(xyzobs.0.len());
    let (xpc, ypc, zpc) = par_chunks_3tup!(xyzobs, n);
    let (jxc, jyc, jzc) = par_chunks_3tup!(jobs, n);
    let (outxc, outyc, outzc) = mut_par_chunks_3tup!(out, n);

    // Run calcs
    (outxc, outyc, outzc, xpc, ypc, zpc, jxc, jyc, jzc)
        .into_par_iter()
        .try_for_each(|(outx, outy, outz, xp, yp, zp, jx, jy, jz)| {
            body_force_density_linear_filament(
                xyzfil,
                dlxyzfil,
                ifil,
                wire_radius,
                (xp, yp, zp),
                (jx, jy, jz),
                (outx, outy, outz),
            )
        })?;

    Ok(())
}

#[cfg(test)]
mod test {
    use std::f64::consts::{E, PI};

    use super::*;
    use crate::math::norm3;
    use crate::physics::point_source::segment::{
        flux_density_point_segment, vector_potential_point_segment,
    };
    use crate::testing::*;

    fn distributed_filaments_rect_grid(
        start: (f64, f64, f64),
        end: (f64, f64, f64),
        wire_radius: f64,
        total_current: f64,
        n_side: usize,
    ) -> (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        assert!(n_side > 0);
        // This helper is only for the test geometry used below (axis-aligned along z).
        assert!(approx(start.0, end.0, 0.0, 1e-15));
        assert!(approx(start.1, end.1, 0.0, 1e-15));

        let mut xfil_ref = Vec::new();
        let mut yfil_ref = Vec::new();
        let mut zfil_ref = Vec::new();

        let cell = 2.0 * wire_radius / n_side as f64;
        for i in 0..n_side {
            let x = start.0 - wire_radius + (i as f64 + 0.5) * cell;
            for j in 0..n_side {
                let y = start.1 - wire_radius + (j as f64 + 0.5) * cell;
                let dx = x - start.0;
                let dy = y - start.1;
                if dx * dx + dy * dy <= wire_radius * wire_radius {
                    xfil_ref.push(x);
                    yfil_ref.push(y);
                    zfil_ref.push(start.2);
                }
            }
        }

        let nsub = xfil_ref.len();
        assert!(
            nsub > 10000,
            "expected more than 10000 distributed filaments, got {}",
            nsub
        );

        let dlx_ref = vec![end.0 - start.0; nsub];
        let dly_ref = vec![end.1 - start.1; nsub];
        let dlz_ref = vec![end.2 - start.2; nsub];
        let ifil_ref = vec![total_current / nsub as f64; nsub];
        let wire_radius_ref = vec![0.0; nsub];

        (
            xfil_ref,
            yfil_ref,
            zfil_ref,
            dlx_ref,
            dly_ref,
            dlz_ref,
            ifil_ref,
            wire_radius_ref,
        )
    }

    /// Make sure the forces have the right sign
    /// and self-forces sum to zero within discretization error
    #[test]
    fn test_body_force_density() {
        let (rtol, atol) = (1e-9, 1e-10);

        let ndiscr = 100; // Discretizations of circular filament into linear filaments

        // Make some circular filaments
        let (rfil, zfil, nfil) = example_circular_filaments();

        // For filament self-field, use the filament roots as the observation points
        for i in 0..rfil.len() {
            let (ri, zi, ni) = (rfil[i], zfil[i], nfil[i]);

            let (x, y, z) = discretize_circular_filament(ri, zi, ndiscr);
            let dl = (&diff(&x)[..], &diff(&y)[..], &diff(&z)[..]);
            let (jxbx, jxby, jxbz) = (
                &mut vec![0.0; ndiscr - 1],
                &mut vec![0.0; ndiscr - 1],
                &mut vec![0.0; ndiscr - 1],
            );
            body_force_density_linear_filament(
                (&x[..ndiscr - 1], &y[..ndiscr - 1], &z[..ndiscr - 1]),
                dl,
                &vec![ni; ndiscr - 1][..],
                &vec![0.0; ndiscr - 1][..],
                (&x[..ndiscr - 1], &y[..ndiscr - 1], &z[..ndiscr - 1]),
                dl,
                (jxbx, jxby, jxbz),
            )
            .unwrap();

            // Make sure the totals sum to zero - a magnet can't put a net force on itself
            let jxbx_sum: f64 = jxbx.iter().sum();
            let jxby_sum: f64 = jxby.iter().sum();
            let jxbz_sum: f64 = jxbz.iter().sum();
            assert!(approx(0.0, jxbx_sum, rtol, atol));
            assert!(approx(0.0, jxby_sum, rtol, atol));
            assert!(approx(0.0, jxbz_sum, rtol, atol));

            // Make sure jxb points outward everywhere
            for j in 0..ndiscr - 1 {
                let r: (f64, f64, f64) = (x[j], y[j], 0.0);
                let rxjxb = cross3([r.0, r.1, r.2], [jxbx[j], jxby[j], jxbz[j]]);
                // Linear filaments aren't perfectly aligned, so we need a slighter wider tolerance here
                assert!(approx(0.0, norm3(rxjxb), rtol, 1e-8));
            }
        }

        // For filament pairs, make sure axial force is pulling them together
        for i in 0..rfil.len() {
            let (ri, zif, ni) = (rfil[i], zfil[i], nfil[i]);
            let (xi, yi, zi) = discretize_circular_filament(ri, zif, ndiscr);
            let dli = (&diff(&xi)[..], &diff(&yi)[..], &diff(&zi)[..]);

            for j in 0..rfil.len() {
                // Self-field examined separately
                if j == i {
                    continue;
                }

                let (rj, zjf, nj) = (rfil[j], zfil[j], nfil[j]);
                let (xj, yj, zj) = discretize_circular_filament(rj, zjf, ndiscr);
                let dlj = (&diff(&xj)[..], &diff(&yj)[..], &diff(&zj)[..]);
                let mid = (
                    &midpoints(&xj)[..],
                    &midpoints(&yj)[..],
                    &midpoints(&zj)[..],
                );

                let (jxbx, jxby, jxbz) = (
                    &mut vec![0.0; ndiscr - 1],
                    &mut vec![0.0; ndiscr - 1],
                    &mut vec![0.0; ndiscr - 1],
                );
                body_force_density_linear_filament(
                    (&xi[..ndiscr - 1], &yi[..ndiscr - 1], &zi[..ndiscr - 1]),
                    dli,
                    &vec![ni * nj; xi.len() - 1][..],
                    &vec![0.0; xi.len() - 1][..],
                    mid,
                    dlj,
                    (jxbx, jxby, jxbz),
                )
                .unwrap();

                // Expect attracting force from j toward i,
                // and no centering force because the loops are coaxial
                let jxbx_sum: f64 = jxbx.iter().sum();
                let jxby_sum: f64 = jxby.iter().sum();
                let jxbz_sum: f64 = jxbz.iter().sum();
                assert!(approx(0.0, jxbx_sum, rtol, atol));
                assert!(approx(0.0, jxby_sum, rtol, atol));
                assert!(jxbz_sum.signum() == (zif - zjf).signum());
            }
        }
    }

    /// Compare single-segment Biot-Savart against discretized point-source segments.
    #[test]
    fn test_flux_density_against_point_segment_discretization() {
        let (rtol, atol) = (1e-6, 1e-12);

        let start = (0.0, 0.0, -0.5);
        let end = (0.0, 0.0, 0.5);

        let xfil = [start.0];
        let yfil = [start.1];
        let zfil = [start.2];
        let dlx = [end.0 - start.0];
        let dly = [end.1 - start.1];
        let dlz = [end.2 - start.2];
        let xyzfil = (&xfil[..], &yfil[..], &zfil[..]);
        let dlxyz = (&dlx[..], &dly[..], &dlz[..]);

        let ngrid = 100;
        let span = 10.0;
        let xvals: Vec<f64> = (0..ngrid)
            .map(|i| -span + (2.0 * span) * (i as f64) / (ngrid as f64 - 1.0))
            .collect();
        let yvals = xvals.clone();
        let zvals = xvals.clone();

        let total = ngrid * ngrid * ngrid;
        let mut xp = Vec::with_capacity(total);
        let mut yp = Vec::with_capacity(total);
        let mut zp = Vec::with_capacity(total);
        for &x in &xvals {
            for &y in &yvals {
                for &z in &zvals {
                    xp.push(x);
                    yp.push(y);
                    zp.push(z);
                }
            }
        }
        let xyzp = (&xp[..], &yp[..], &zp[..]);

        let nseg = 1000;
        let dz = (end.2 - start.2) / nseg as f64;
        let mut xfil_ps = Vec::with_capacity(nseg);
        let mut yfil_ps = Vec::with_capacity(nseg);
        let mut zfil_ps = Vec::with_capacity(nseg);
        for i in 0..nseg {
            xfil_ps.push(start.0);
            yfil_ps.push(start.1);
            zfil_ps.push(start.2 + dz * i as f64);
        }
        let dlx = vec![0.0; nseg];
        let dly = vec![0.0; nseg];
        let dlz = vec![dz; nseg];

        for &current in &[1.0, E] {
            let ifil = [current];
            let ifil_ps = vec![current; nseg];

            let mut bx = vec![0.0; total];
            let mut by = vec![0.0; total];
            let mut bz = vec![0.0; total];
            flux_density_linear_filament(
                xyzp,
                xyzfil,
                dlxyz,
                &ifil,
                &[0.0],
                (&mut bx, &mut by, &mut bz),
            )
            .unwrap();

            let mut bx_ps = vec![0.0; total];
            let mut by_ps = vec![0.0; total];
            let mut bz_ps = vec![0.0; total];
            flux_density_point_segment(
                xyzp,
                (&xfil_ps, &yfil_ps, &zfil_ps),
                (&dlx, &dly, &dlz),
                &ifil_ps,
                (&mut bx_ps, &mut by_ps, &mut bz_ps),
            )
            .unwrap();

            for i in 0..xp.len() {
                assert!(
                    approx(bx[i], bx_ps[i], rtol, atol),
                    "current={} A: bx is {}, should be {}",
                    current,
                    bx[i],
                    bx_ps[i]
                );
                assert!(
                    approx(by[i], by_ps[i], rtol, atol),
                    "current={} A: by is {}, should be {}",
                    current,
                    by[i],
                    by_ps[i]
                );
                assert!(
                    approx(bz[i], bz_ps[i], rtol, atol),
                    "current={} A: bz is {}, should be {}",
                    current,
                    bz[i],
                    bz_ps[i]
                );
            }
        }
    }

    /// Check linear falloff inside finite wire radius.
    #[test]
    fn test_flux_density_linear_falloff_inside_wire() {
        let (rtol, atol) = (1e-10, 1e-14);
        let wire_radius = 0.1;

        let start = (0.0, 0.0, -0.5);
        let end = (0.0, 0.0, 0.5);
        let ifil = [1.0];
        let xfil = [start.0];
        let yfil = [start.1];
        let zfil = [start.2];
        let dlx = [end.0 - start.0];
        let dly = [end.1 - start.1];
        let dlz = [end.2 - start.2];

        let ratios = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0];
        let xp: Vec<f64> = ratios.iter().map(|r| r * wire_radius).collect();
        let yp: Vec<f64> = vec![0.0; ratios.len()];
        let zp: Vec<f64> = vec![0.0; ratios.len()];

        let mut bx = vec![0.0; ratios.len()];
        let mut by = vec![0.0; ratios.len()];
        let mut bz = vec![0.0; ratios.len()];
        flux_density_linear_filament(
            (&xp, &yp, &zp),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            &[wire_radius],
            (&mut bx, &mut by, &mut bz),
        )
        .unwrap();

        let bmag: Vec<f64> = bx
            .iter()
            .zip(by.iter())
            .zip(bz.iter())
            .map(|((bx, by), bz)| (bx * bx + by * by + bz * bz).sqrt())
            .collect();

        let b_edge = bmag[0];
        for (i, &ratio) in ratios.iter().enumerate() {
            let expected = b_edge * ratio;
            assert!(
                approx(expected, bmag[i], rtol, atol),
                "ratio {}: |B| = {:.6e}, expected {:.6e}",
                ratio,
                bmag[i],
                expected
            );
        }
    }

    #[test]
    fn test_flux_density_against_distributed_filaments_equivalent_area() {
        let wire_radius = 0.1;
        let start = (0.0, 0.0, -1.0);
        let end = (0.0, 0.0, 1.0);
        let ifil = [1.0];
        let xfil = [start.0];
        let yfil = [start.1];
        let zfil = [start.2];
        let dlx = [end.0 - start.0];
        let dly = [end.1 - start.1];
        let dlz = [end.2 - start.2];

        let (xfil_ref, yfil_ref, zfil_ref, dlx_ref, dly_ref, dlz_ref, ifil_ref, wire_radius_ref) =
            distributed_filaments_rect_grid(start, end, wire_radius, ifil[0], 120);

        // Sample in two regions:
        // 1) along filament centerline (inside segment),
        // 2) outside the conductor near the midpoint.
        let axis_z = [-10.0, -0.4, 0.0, 0.4, 10.0];
        let outside_x = [0.12, 0.16, 0.2, 0.3, 0.5, 10.0];
        let mut xp = Vec::with_capacity(axis_z.len() + outside_x.len());
        let mut yp = Vec::with_capacity(axis_z.len() + outside_x.len());
        let mut zp = Vec::with_capacity(axis_z.len() + outside_x.len());

        for &z in &axis_z {
            xp.push(0.0);
            yp.push(0.0);
            zp.push(z);
        }
        for &x in &outside_x {
            xp.push(x);
            yp.push(0.0);
            zp.push(0.0);
        }

        let mut bx = vec![0.0; xp.len()];
        let mut by = vec![0.0; xp.len()];
        let mut bz = vec![0.0; xp.len()];
        flux_density_linear_filament(
            (&xp[..], &yp[..], &zp[..]),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            &[wire_radius],
            (&mut bx[..], &mut by[..], &mut bz[..]),
        )
        .unwrap();

        let mut bx_ref = vec![0.0; xp.len()];
        let mut by_ref = vec![0.0; xp.len()];
        let mut bz_ref = vec![0.0; xp.len()];
        flux_density_linear_filament(
            (&xp[..], &yp[..], &zp[..]),
            (&xfil_ref, &yfil_ref, &zfil_ref),
            (&dlx_ref, &dly_ref, &dlz_ref),
            &ifil_ref,
            &wire_radius_ref,
            (&mut bx_ref[..], &mut by_ref[..], &mut bz_ref[..]),
        )
        .unwrap();

        let mut abs_err_axis = Vec::with_capacity(axis_z.len());
        let mut rel_err_outside = Vec::with_capacity(outside_x.len());
        for i in 0..xp.len() {
            let b = norm3([bx[i], by[i], bz[i]]);
            let b_ref = norm3([bx_ref[i], by_ref[i], bz_ref[i]]);
            if i < axis_z.len() {
                let err = (b - b_ref).abs();
                assert!(err.is_finite());
                abs_err_axis.push(err);
            } else {
                let denom = b_ref.abs().max(1e-30);
                let err = (b - b_ref).abs() / denom;
                assert!(err.is_finite());
                rel_err_outside.push(err);
            }
        }

        let max_abs_axis = abs_err_axis.iter().copied().fold(0.0, f64::max);
        let max_rel_outside = rel_err_outside.iter().copied().fold(0.0, f64::max);
        let mean_rel_outside = rel_err_outside.iter().sum::<f64>() / rel_err_outside.len() as f64;

        assert!(
            max_abs_axis < 1e-13,
            "max axis absolute error too large: {:.6e}, abs_err_axis={:?}",
            max_abs_axis,
            abs_err_axis
        );
        assert!(
            max_rel_outside < 1e-3,
            "max outside relative error too large: {:.6e}, rel_err_outside={:?}",
            max_rel_outside,
            rel_err_outside
        );
        assert!(
            mean_rel_outside < 1e-3,
            "mean outside relative error too large: {:.6e}, rel_err_outside={:?}",
            mean_rel_outside,
            rel_err_outside
        );
    }

    #[test]
    fn test_point_line_distance_endpoint_clamp_values() {
        use crate::math::point_line_distance_with_endpoints;

        let (rtol, atol) = (1e-12, 1e-15);
        let start = (0.0, 0.0, -0.5);
        let end = (0.0, 0.0, 0.5);
        let wire_radius = 0.1;
        let x = 0.02;

        // At the endpoint plane, behavior should match in-segment clamping.
        let at_endpoint = point_line_distance_with_endpoints(
            [start.0, start.1, start.2],
            [end.0, end.1, end.2],
            [x, 0.0, end.2],
            wire_radius,
        );
        assert!(approx(0.1, at_endpoint.perp, rtol, atol));
        assert!(approx(0.2, at_endpoint.frac, rtol, atol));

        // Outside endpoint projection, clamping behavior is unchanged without endpoint blending.
        let halfway = point_line_distance_with_endpoints(
            [start.0, start.1, start.2],
            [end.0, end.1, end.2],
            [x, 0.0, end.2 + 0.05],
            wire_radius,
        );
        assert!(approx(0.1, halfway.perp, rtol, atol));
        assert!(approx(0.2, halfway.frac, rtol, atol));

        // One wire radius beyond the endpoint projection remains clamped the same way.
        let outside = point_line_distance_with_endpoints(
            [start.0, start.1, start.2],
            [end.0, end.1, end.2],
            [x, 0.0, end.2 + 0.1],
            wire_radius,
        );
        assert!(approx(0.1, outside.perp, rtol, atol));
        assert!(approx(0.2, outside.frac, rtol, atol));
    }

    #[test]
    fn test_flux_density_no_endpoint_blend_differs_from_thin_outside_projection() {
        let wire_radius = 0.1;
        let start = (0.0, 0.0, -0.5);
        let end = (0.0, 0.0, 0.5);
        let ifil = 1.0;
        let x = 0.02;
        let overhangs = [0.0, 0.02, 0.05, 0.1, 0.2];

        let mut diff: Vec<f64> = Vec::with_capacity(overhangs.len());

        for &overhang in &overhangs {
            let obs = (x, 0.0, end.2 + overhang);
            let b_finite =
                flux_density_linear_filament_scalar((start, end, ifil), wire_radius, obs);
            let b_thin = flux_density_linear_filament_scalar((start, end, ifil), 0.0, obs);
            diff.push(norm3([
                b_finite.0 - b_thin.0,
                b_finite.1 - b_thin.1,
                b_finite.2 - b_thin.2,
            ]));
        }

        assert!(diff[0] > 0.0);
        assert!(diff[diff.len() - 1] > 0.0);
        assert!(diff.iter().all(|d| d.is_finite()));
    }

    /// Centerline values should be finite and not NaN.
    #[test]
    fn test_flux_density_centerline_finite() {
        let start = (0.0, 0.0, -0.5);
        let end = (0.0, 0.0, 0.5);
        let ifil = [1.0];
        let xfil = [start.0];
        let yfil = [start.1];
        let zfil = [start.2];
        let dlx = [end.0 - start.0];
        let dly = [end.1 - start.1];
        let dlz = [end.2 - start.2];

        let z_points = [-0.5, -0.25, 0.0, 0.25, 0.5];
        let xp: Vec<f64> = vec![0.0; z_points.len()];
        let yp: Vec<f64> = vec![0.0; z_points.len()];
        let zp: Vec<f64> = z_points.to_vec();

        for &wire_radius in &[0.0, 0.01, 0.1] {
            let mut bx = vec![0.0; z_points.len()];
            let mut by = vec![0.0; z_points.len()];
            let mut bz = vec![0.0; z_points.len()];
            flux_density_linear_filament(
                (&xp, &yp, &zp),
                (&xfil, &yfil, &zfil),
                (&dlx, &dly, &dlz),
                &ifil,
                &[wire_radius],
                (&mut bx, &mut by, &mut bz),
            )
            .unwrap();

            for i in 0..z_points.len() {
                assert!(bx[i].is_finite(), "bx[{}] is {}", i, bx[i]);
                assert!(by[i].is_finite(), "by[{}] is {}", i, by[i]);
                assert!(bz[i].is_finite(), "bz[{}] is {}", i, bz[i]);
            }
        }
    }

    /// Explicitly check axis evaluations at both endpoints and midpoint are non-singular.
    #[test]
    fn test_flux_density_axis_endpoint_midpoint_nonsingular_scalar() {
        let start = (0.0_f64, 0.0, -0.5);
        let end = (0.0_f64, 0.0, 0.5);
        let midpoint = (0.0, 0.0, 0.5 * (start.2 + end.2));
        let ifil = 1.0_f64;
        let axis_points = [("start", start), ("midpoint", midpoint), ("end", end)];

        for &wire_radius in &[0.0, 0.01, 0.1] {
            for &(label, p) in &axis_points {
                let b = flux_density_linear_filament_scalar((start, end, ifil), wire_radius, p);
                assert!(
                    b.0.is_finite(),
                    "Bx is non-finite at {} for wire_radius={}",
                    label,
                    wire_radius
                );
                assert!(
                    b.1.is_finite(),
                    "By is non-finite at {} for wire_radius={}",
                    label,
                    wire_radius
                );
                assert!(
                    b.2.is_finite(),
                    "Bz is non-finite at {} for wire_radius={}",
                    label,
                    wire_radius
                );
            }
        }
    }

    /// Compare single-segment vector potential against discretized point-source segments.
    #[test]
    fn test_vector_potential_against_point_segment_discretization() {
        let (rtol, atol) = (1e-6, 1e-15);

        let start = (0.0, 0.0, -0.5);
        let end = (0.0, 0.0, 0.5);

        let xfil = [start.0];
        let yfil = [start.1];
        let zfil = [start.2];
        let dlx = [end.0 - start.0];
        let dly = [end.1 - start.1];
        let dlz = [end.2 - start.2];
        let xyzfil = (&xfil[..], &yfil[..], &zfil[..]);
        let dlxyz = (&dlx[..], &dly[..], &dlz[..]);

        let ngrid = 100;
        let span = 10.0;
        let xvals: Vec<f64> = (0..ngrid)
            .map(|i| -span + (2.0 * span) * (i as f64) / (ngrid as f64 - 1.0))
            .collect();
        let yvals = xvals.clone();
        let zvals = xvals.clone();

        let total = ngrid * ngrid * ngrid;
        let mut xp = Vec::with_capacity(total);
        let mut yp = Vec::with_capacity(total);
        let mut zp = Vec::with_capacity(total);
        for &x in &xvals {
            for &y in &yvals {
                for &z in &zvals {
                    xp.push(x);
                    yp.push(y);
                    zp.push(z);
                }
            }
        }
        let xyzp = (&xp[..], &yp[..], &zp[..]);

        let nseg = 1000;
        let dz = (end.2 - start.2) / nseg as f64;
        let mut xfil_ps = Vec::with_capacity(nseg);
        let mut yfil_ps = Vec::with_capacity(nseg);
        let mut zfil_ps = Vec::with_capacity(nseg);
        for i in 0..nseg {
            xfil_ps.push(start.0);
            yfil_ps.push(start.1);
            zfil_ps.push(start.2 + dz * i as f64);
        }
        let dlx = vec![0.0; nseg];
        let dly = vec![0.0; nseg];
        let dlz = vec![dz; nseg];

        for &current in &[1.0, E] {
            let ifil = [current];
            let ifil_ps = vec![current; nseg];

            let mut ax = vec![0.0; total];
            let mut ay = vec![0.0; total];
            let mut az = vec![0.0; total];
            vector_potential_linear_filament(
                xyzp,
                xyzfil,
                dlxyz,
                &ifil,
                &[0.0],
                (&mut ax, &mut ay, &mut az),
            )
            .unwrap();

            let mut ax_ps = vec![0.0; total];
            let mut ay_ps = vec![0.0; total];
            let mut az_ps = vec![0.0; total];
            vector_potential_point_segment(
                xyzp,
                (&xfil_ps, &yfil_ps, &zfil_ps),
                (&dlx, &dly, &dlz),
                &ifil_ps,
                (&mut ax_ps, &mut ay_ps, &mut az_ps),
            )
            .unwrap();

            for i in 0..xp.len() {
                assert!(
                    approx(ax[i], ax_ps[i], rtol, atol),
                    "current={} A: ax is {}, should be {}",
                    current,
                    ax[i],
                    ax_ps[i]
                );
                assert!(
                    approx(ay[i], ay_ps[i], rtol, atol),
                    "current={} A: ay is {}, should be {}",
                    current,
                    ay[i],
                    ay_ps[i]
                );
                assert!(
                    approx(az[i], az_ps[i], rtol, atol),
                    "current={} A: az is {}, should be {}",
                    current,
                    az[i],
                    az_ps[i]
                );
            }
        }
    }

    /// Check quadratic gauge shift inside finite wire radius.
    #[test]
    fn test_vector_potential_quadratic_falloff_inside_wire() {
        let (rtol, atol) = (1e-10, 1e-14);
        let wire_radius = 0.1;

        let start = (0.0, 0.0, -0.5);
        let end = (0.0, 0.0, 0.5);
        let ifil = [1.0];
        let xfil = [start.0];
        let yfil = [start.1];
        let zfil = [start.2];
        let dlx = [end.0 - start.0];
        let dly = [end.1 - start.1];
        let dlz = [end.2 - start.2];

        let ratios = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0];
        let xp: Vec<f64> = ratios.iter().map(|r| r * wire_radius).collect();
        let yp: Vec<f64> = vec![0.0; ratios.len()];
        let zp: Vec<f64> = vec![0.0; ratios.len()];

        let mut ax = vec![0.0; ratios.len()];
        let mut ay = vec![0.0; ratios.len()];
        let mut az = vec![0.0; ratios.len()];
        vector_potential_linear_filament(
            (&xp, &yp, &zp),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            &[wire_radius],
            (&mut ax, &mut ay, &mut az),
        )
        .unwrap();

        let amag: Vec<f64> = ax
            .iter()
            .zip(ay.iter())
            .zip(az.iter())
            .map(|((ax, ay), az)| (ax * ax + ay * ay + az * az).sqrt())
            .collect();

        let a_edge = amag[0];
        let pld = crate::math::point_line_distance_with_endpoints(
            [start.0, start.1, start.2],
            [end.0, end.1, end.2],
            [wire_radius, 0.0, 0.0],
            wire_radius,
        );
        let sin_theta_a = pld.para_a / pld.dist_a;
        let sin_theta_b = pld.para_b / pld.dist_b;
        let kappa = -MU0_OVER_4PI * ifil[0] * (sin_theta_b - sin_theta_a);
        for (i, &ratio) in ratios.iter().enumerate() {
            let expected = a_edge + 0.5 * kappa * (1.0 - ratio * ratio);
            assert!(
                approx(expected, amag[i], rtol, atol),
                "ratio {}: |A| = {:.6e}, expected {:.6e}",
                ratio,
                amag[i],
                expected
            );
        }
    }

    #[test]
    fn test_vector_potential_against_distributed_filaments_equivalent_area() {
        let wire_radius = 0.1;
        let start = (0.0, 0.0, -1.0);
        let end = (0.0, 0.0, 1.0);
        let ifil = [1.0];
        let xfil = [start.0];
        let yfil = [start.1];
        let zfil = [start.2];
        let dlx = [end.0 - start.0];
        let dly = [end.1 - start.1];
        let dlz = [end.2 - start.2];

        let (xfil_ref, yfil_ref, zfil_ref, dlx_ref, dly_ref, dlz_ref, ifil_ref, wire_radius_ref) =
            distributed_filaments_rect_grid(start, end, wire_radius, ifil[0], 120);

        // Sample in two regions:
        // 1) along filament centerline (inside segment),
        // 2) outside the conductor near the midpoint.
        let axis_z = [-10.0, -0.4, 0.0, 0.4, 10.0];
        let outside_x = [0.12, 0.16, 0.2, 0.3, 0.5, 10.0];
        let mut xp = Vec::with_capacity(axis_z.len() + outside_x.len());
        let mut yp = Vec::with_capacity(axis_z.len() + outside_x.len());
        let mut zp = Vec::with_capacity(axis_z.len() + outside_x.len());

        for &z in &axis_z {
            xp.push(0.0);
            yp.push(0.0);
            zp.push(z);
        }
        for &x in &outside_x {
            xp.push(x);
            yp.push(0.0);
            zp.push(0.0);
        }

        let mut ax = vec![0.0; xp.len()];
        let mut ay = vec![0.0; xp.len()];
        let mut az = vec![0.0; xp.len()];
        vector_potential_linear_filament(
            (&xp[..], &yp[..], &zp[..]),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            &[wire_radius],
            (&mut ax[..], &mut ay[..], &mut az[..]),
        )
        .unwrap();

        let mut ax_ref = vec![0.0; xp.len()];
        let mut ay_ref = vec![0.0; xp.len()];
        let mut az_ref = vec![0.0; xp.len()];
        vector_potential_linear_filament(
            (&xp[..], &yp[..], &zp[..]),
            (&xfil_ref, &yfil_ref, &zfil_ref),
            (&dlx_ref, &dly_ref, &dlz_ref),
            &ifil_ref,
            &wire_radius_ref,
            (&mut ax_ref[..], &mut ay_ref[..], &mut az_ref[..]),
        )
        .unwrap();

        let mut rel_err_axis = Vec::with_capacity(axis_z.len());
        let mut rel_err_outside = Vec::with_capacity(outside_x.len());
        for i in 0..xp.len() {
            let a = norm3([ax[i], ay[i], az[i]]);
            let a_ref = norm3([ax_ref[i], ay_ref[i], az_ref[i]]);
            let denom = a_ref.abs().max(1e-30);
            let err = (a - a_ref).abs() / denom;
            assert!(err.is_finite());
            if i < axis_z.len() {
                rel_err_axis.push(err);
            } else {
                rel_err_outside.push(err);
            }
        }

        let max_rel_axis = rel_err_axis.iter().copied().fold(0.0, f64::max);
        let mean_rel_axis = rel_err_axis.iter().sum::<f64>() / rel_err_axis.len() as f64;
        let max_rel_outside = rel_err_outside.iter().copied().fold(0.0, f64::max);
        let mean_rel_outside = rel_err_outside.iter().sum::<f64>() / rel_err_outside.len() as f64;

        assert!(
            max_rel_axis < 1e-3,
            "max axis relative error too large: {:.6e}, rel_err_axis={:?}",
            max_rel_axis,
            rel_err_axis
        );
        assert!(
            mean_rel_axis < 1e-3,
            "mean axis relative error too large: {:.6e}, rel_err_axis={:?}",
            mean_rel_axis,
            rel_err_axis
        );
        assert!(
            max_rel_outside < 1e-3,
            "max outside relative error too large: {:.6e}, rel_err_outside={:?}",
            max_rel_outside,
            rel_err_outside
        );
        assert!(
            mean_rel_outside < 1e-3,
            "mean outside relative error too large: {:.6e}, rel_err_outside={:?}",
            mean_rel_outside,
            rel_err_outside
        );
    }

    #[test]
    fn test_vector_potential_no_endpoint_blend_differs_from_thin_outside_projection() {
        let wire_radius = 0.1_f64;
        let start = (0.0_f64, 0.0, -0.5);
        let end = (0.0_f64, 0.0, 0.5);
        let ifil = 1.0_f64;
        let x = 0.02_f64;
        let overhangs = [0.0_f64, 0.02, 0.05, 0.1, 0.2];

        let mut diff = Vec::with_capacity(overhangs.len());

        for &overhang in &overhangs {
            let obs = (x, 0.0, end.2 + overhang);
            let a_finite =
                vector_potential_linear_filament_scalar((start, end, ifil), wire_radius, obs);
            let a_thin = vector_potential_linear_filament_scalar((start, end, ifil), 0.0, obs);
            diff.push(norm3([
                a_finite.0 - a_thin.0,
                a_finite.1 - a_thin.1,
                a_finite.2 - a_thin.2,
            ]));
        }

        assert!(diff[0] > 0.0);
        assert!(diff[diff.len() - 1] > 0.0);
        assert!(diff.iter().all(|d| d.is_finite()));
    }

    /// Centerline values should be finite and not NaN.
    #[test]
    fn test_vector_potential_centerline_finite() {
        let start = (0.0, 0.0, -0.5);
        let end = (0.0, 0.0, 0.5);
        let ifil = [1.0];
        let xfil = [start.0];
        let yfil = [start.1];
        let zfil = [start.2];
        let dlx = [end.0 - start.0];
        let dly = [end.1 - start.1];
        let dlz = [end.2 - start.2];

        let z_points = [-0.5, -0.25, 0.0, 0.25, 0.5];
        let xp: Vec<f64> = vec![0.0; z_points.len()];
        let yp: Vec<f64> = vec![0.0; z_points.len()];
        let zp: Vec<f64> = z_points.to_vec();

        for &wire_radius in &[0.0, 0.01, 0.1] {
            let mut ax = vec![0.0; z_points.len()];
            let mut ay = vec![0.0; z_points.len()];
            let mut az = vec![0.0; z_points.len()];
            vector_potential_linear_filament(
                (&xp, &yp, &zp),
                (&xfil, &yfil, &zfil),
                (&dlx, &dly, &dlz),
                &ifil,
                &[wire_radius],
                (&mut ax, &mut ay, &mut az),
            )
            .unwrap();

            for i in 0..z_points.len() {
                assert!(ax[i].is_finite(), "ax[{}] is {}", i, ax[i]);
                assert!(ay[i].is_finite(), "ay[{}] is {}", i, ay[i]);
                assert!(az[i].is_finite(), "az[{}] is {}", i, az[i]);
            }
        }
    }

    /// Explicitly check axis evaluations at both endpoints and midpoint are non-singular.
    #[test]
    fn test_vector_potential_axis_endpoint_midpoint_nonsingular_scalar() {
        let start = (0.0_f64, 0.0, -0.5);
        let end = (0.0_f64, 0.0, 0.5);
        let midpoint = (0.0, 0.0, 0.5 * (start.2 + end.2));
        let ifil = 1.0_f64;
        let axis_points = [("start", start), ("midpoint", midpoint), ("end", end)];

        for &wire_radius in &[0.0_f64, 0.01, 0.1] {
            for &(label, p) in &axis_points {
                let a = vector_potential_linear_filament_scalar((start, end, ifil), wire_radius, p);
                assert!(
                    a.0.is_finite(),
                    "Ax is non-finite at {} for wire_radius={}",
                    label,
                    wire_radius
                );
                assert!(
                    a.1.is_finite(),
                    "Ay is non-finite at {} for wire_radius={}",
                    label,
                    wire_radius
                );
                assert!(
                    a.2.is_finite(),
                    "Az is non-finite at {} for wire_radius={}",
                    label,
                    wire_radius
                );
            }
        }
    }

    #[test]
    fn test_vector_potential_curl_matches_b_inside_wire() {
        let (rtol, atol) = (5e-3, 1e-12);
        let start = (0.0, 0.0, -0.5);
        let end = (0.0, 0.0, 0.5);
        let ifil = 1.0;
        let wire_radius = 0.1;
        let p = (0.03, 0.04, 0.0); // Inside wire, away from endpoints.
        let eps = 1e-6;

        let a_at = |x: f64, y: f64, z: f64| {
            vector_potential_linear_filament_scalar((start, end, ifil), wire_radius, (x, y, z))
        };

        // da/dx
        let (_ax0, ay0, az0) = a_at(p.0 - eps, p.1, p.2);
        let (_ax1, ay1, az1) = a_at(p.0 + eps, p.1, p.2);
        let daz_dx = (az1 - az0) / (2.0 * eps);
        let day_dx = (ay1 - ay0) / (2.0 * eps);

        // da/dy
        let (ax0, _ay0, az0) = a_at(p.0, p.1 - eps, p.2);
        let (ax1, _ay1, az1) = a_at(p.0, p.1 + eps, p.2);
        let daz_dy = (az1 - az0) / (2.0 * eps);
        let dax_dy = (ax1 - ax0) / (2.0 * eps);

        // da/dz
        let (ax0, ay0, _az0) = a_at(p.0, p.1, p.2 - eps);
        let (ax1, ay1, _az1) = a_at(p.0, p.1, p.2 + eps);
        let day_dz = (ay1 - ay0) / (2.0 * eps);
        let dax_dz = (ax1 - ax0) / (2.0 * eps);

        let curl_a = (daz_dy - day_dz, dax_dz - daz_dx, day_dx - dax_dy);
        let b = flux_density_linear_filament_scalar((start, end, ifil), wire_radius, p);

        assert!(
            approx(b.0, curl_a.0, rtol, atol),
            "bx = {:.6e}, curl_a.x = {:.6e}",
            b.0,
            curl_a.0
        );
        assert!(
            approx(b.1, curl_a.1, rtol, atol),
            "by = {:.6e}, curl_a.y = {:.6e}",
            b.1,
            curl_a.1
        );
        assert!(
            approx(b.2, curl_a.2, rtol, atol),
            "bz = {:.6e}, curl_a.z = {:.6e}",
            b.2,
            curl_a.2
        );
    }

    /// Check that B = curl(A)
    #[test]
    fn test_vector_potential() {
        // One super basic filament as the source
        let xyz = [0.0];
        let dlxyz = [1.0];

        // Build a second scattering of filament locations as the target
        const NFIL: usize = 10;
        let xfil2: Vec<f64> = (0..NFIL).map(|i| (i as f64).sin() + PI).collect();
        let yfil2: Vec<f64> = (0..NFIL).map(|i| (i as f64).cos() - PI).collect();
        let zfil2: Vec<f64> = (0..NFIL)
            .map(|i| (i as f64) - (NFIL as f64) / 2.0 + PI)
            .collect();
        let xyzfil2 = (
            &xfil2[..=NFIL - 2],
            &yfil2[..=NFIL - 2],
            &zfil2[..=NFIL - 2],
        );

        let dlxfil2: Vec<f64> = (0..=NFIL - 2).map(|i| xfil2[i + 1] - xfil2[i]).collect();
        let dlyfil2: Vec<f64> = (0..=NFIL - 2).map(|i| yfil2[i + 1] - yfil2[i]).collect();
        let dlzfil2: Vec<f64> = (0..=NFIL - 2).map(|i| zfil2[i + 1] - zfil2[i]).collect();
        let dlxyzfil2 = (&dlxfil2[..], &dlyfil2[..], &dlzfil2[..]);

        let mut xquad2 = vec![0.0; 3 * (NFIL - 1)];
        let mut yquad2 = vec![0.0; 3 * (NFIL - 1)];
        let mut zquad2 = vec![0.0; 3 * (NFIL - 1)];
        let gl3_unit = gauss_legendre_unit_interval_table(GaussLegendreRule::Gauss3);
        for i in 0..NFIL - 1 {
            let row = 3 * i;
            for (iq, &[tq, _]) in gl3_unit.iter().enumerate() {
                xquad2[row + iq] = dlxfil2[i].mul_add(tq, xfil2[i]);
                yquad2[row + iq] = dlyfil2[i].mul_add(tq, yfil2[i]);
                zquad2[row + iq] = dlzfil2[i].mul_add(tq, zfil2[i]);
            }
        }

        let outx = &mut [0.0; 3 * (NFIL - 1)];
        let outy = &mut [0.0; 3 * (NFIL - 1)];
        let outz = &mut [0.0; 3 * (NFIL - 1)];
        vector_potential_linear_filament(
            (&xquad2, &yquad2, &zquad2),
            (&xyz, &xyz, &xyz),
            (&dlxyz, &dlxyz, &dlxyz),
            &[1.0],
            &[0.0],
            (outx, outy, outz),
        )
        .unwrap();
        // Here the mutual inductance of the two filaments is calculated from the
        // vector potential at filament 2 due to 1 ampere of current flowing in filament 1.
        // By Stokes' theorem, the line integral of A over filament 2 is equal to the
        // magnetic flux through a surface bounded by filament 2. The flux through
        // filament 2 due to 1 ampere of current in filament 1 is the mutual inductance.
        // (We are stretching the applicability of Stokes' therorem because the filaments
        // are not closed loops).
        //
        // This should match exactly because the inductance helper now uses the same
        // 3-point Gauss-Legendre A·dl construction with a 1 A source current.
        let m_from_a = (0..NFIL - 1)
            .map(|i| {
                let row = 3 * i;
                (0..3)
                    .map(|iq| {
                        let idx = row + iq;
                        gl3_unit[iq][1]
                            * (outx[idx] * dlxfil2[i]
                                + outy[idx] * dlyfil2[i]
                                + outz[idx] * dlzfil2[i])
                    })
                    .sum::<f64>()
            })
            .sum::<f64>();
        let wire_radius = [0.0];
        let m = inductance_piecewise_linear_filaments(
            (&xyz, &xyz, &xyz),
            (&dlxyz, &dlxyz, &dlxyz),
            xyzfil2,
            dlxyzfil2,
            &wire_radius,
        )
        .unwrap();
        assert!(
            approx(m, m_from_a, 1e-12, 1e-15),
            "m = {:.3e}, m_from_a = {:.3e}",
            m,
            m_from_a
        );

        let vp = |x: f64, y: f64, z: f64| {
            let mut outx = [0.0];
            let mut outy = [0.0];
            let mut outz = [0.0];

            vector_potential_linear_filament(
                (&[x], &[y], &[z]),
                (&xyz, &xyz, &xyz),
                (&dlxyz, &dlxyz, &dlxyz),
                &[1.0],
                &[0.0],
                (&mut outx, &mut outy, &mut outz),
            )
            .unwrap();

            (outx[0], outy[0], outz[0])
        };

        let vals = [
            0.25, 0.5, 2.1, 10.0, 100.0, 1000.0, -1000.0, -100.0, -10.0, -2.0, -0.5, -0.25,
        ];
        // finite diff delta needs to be small enough to be accurate
        // but large enough that we can tell the difference between adjacent points
        // that are very far from the origin
        for x in vals.iter() {
            for y in vals.iter() {
                for z in vals.iter() {
                    // Skip the diagonal, which will land exactly on a filament
                    // several times, and the field is non-smooth on-axis.
                    if x == y && x == z {
                        continue;
                    }

                    let x = &(x + 1e-2); // Slightly adjust to avoid nans
                    let y = &(y + 1e-2);
                    let z = &(z - 1e-2);

                    // Scale tolerance and step size based on distance
                    let r: f64 = norm3([*x, *y, *z]);
                    let atol = 1e-12 / r.max(1.0); // Smaller absolute tolerance as field falls off
                    let eps = 1e-8 * r; // Larger finite difference delta in far-field for resolution

                    // Brute-force jac because we're only using it once
                    let mut da = [[0.0; 3]; 3];
                    // da/dx
                    let (ax0, ay0, az0) = vp(*x - eps, *y, *z);
                    let (ax1, ay1, az1) = vp(*x + eps, *y, *z);
                    da[0][0] = (ax1 - ax0) / (2.0 * eps);
                    da[0][1] = (ay1 - ay0) / (2.0 * eps);
                    da[0][2] = (az1 - az0) / (2.0 * eps);

                    // da/dy
                    let (ax0, ay0, az0) = vp(*x, *y - eps, *z);
                    let (ax1, ay1, az1) = vp(*x, *y + eps, *z);
                    da[1][0] = (ax1 - ax0) / (2.0 * eps);
                    da[1][1] = (ay1 - ay0) / (2.0 * eps);
                    da[1][2] = (az1 - az0) / (2.0 * eps);

                    // da/dz
                    let (ax0, ay0, az0) = vp(*x, *y, *z - eps);
                    let (ax1, ay1, az1) = vp(*x, *y, *z + eps);
                    da[2][0] = (ax1 - ax0) / (2.0 * eps);
                    da[2][1] = (ay1 - ay0) / (2.0 * eps);
                    da[2][2] = (az1 - az0) / (2.0 * eps);

                    // B = curl(A)
                    let daz_dy = da[1][2];
                    let day_dz = da[2][1];

                    let daz_dx = da[0][2];
                    let dax_dz = da[2][0];

                    let day_dx = da[0][1];
                    let dax_dy = da[1][0];

                    let ca = [daz_dy - day_dz, dax_dz - daz_dx, day_dx - dax_dy];

                    // B via biot-savart
                    let mut bx = [0.0];
                    let mut by = [0.0];
                    let mut bz = [0.0];
                    flux_density_linear_filament(
                        (&[*x][..], &[*y][..], &[*z][..]),
                        (&xyz[..], &xyz[..], &xyz[..]),
                        (&dlxyz[..], &dlxyz[..], &dlxyz[..]),
                        &[1.0],
                        &[0.0],
                        (&mut bx, &mut by, &mut bz),
                    )
                    .unwrap();

                    println!("x,y,z = {:.2},{:.2},{:.2}", x, y, z);
                    assert!(
                        approx(bx[0], ca[0], 1e-6, atol),
                        "bx = {:.6e}, ca[0] = {:.6e}",
                        bx[0],
                        ca[0]
                    );
                    assert!(
                        approx(by[0], ca[1], 1e-6, atol),
                        "by = {:.6e}, ca[1] = {:.6e}",
                        by[0],
                        ca[1]
                    );
                    assert!(
                        approx(bz[0], ca[2], 1e-6, atol),
                        "bz = {:.6e}, ca[2] = {:.6e}",
                        bz[0],
                        ca[2],
                    );
                }
            }
        }
    }

    /// Check that parallel variants of functions produce the same result as serial.
    /// This also incidentally tests defensive zeroing of input slices.
    #[test]
    fn test_serial_vs_parallel() {
        const NFIL: usize = 10;
        const NOBS: usize = 100;

        // Build a scattering of filament locations
        let xfil: Vec<f64> = (0..NFIL).map(|i| (i as f64).sin()).collect();
        let yfil: Vec<f64> = (0..NFIL).map(|i| (i as f64).cos()).collect();
        let zfil: Vec<f64> = (0..NFIL)
            .map(|i| (i as f64) - (NFIL as f64) / 2.0)
            .collect();
        let xyzfil = (&xfil[..=NFIL - 2], &yfil[..=NFIL - 2], &zfil[..=NFIL - 2]);

        let dlxfil: Vec<f64> = (0..=NFIL - 2).map(|i| xfil[i + 1] - xfil[i]).collect();
        let dlyfil: Vec<f64> = (0..=NFIL - 2).map(|i| yfil[i + 1] - yfil[i]).collect();
        let dlzfil: Vec<f64> = (0..=NFIL - 2).map(|i| zfil[i + 1] - zfil[i]).collect();
        let dlxyzfil = (&dlxfil[..], &dlyfil[..], &dlzfil[..]);

        let ifil: &[f64] = &(0..NFIL - 1).map(|i| i as f64).collect::<Vec<f64>>()[..];

        // Build a scattering of observation locations
        let xp: Vec<f64> = (0..NOBS).map(|i| 2.0 * (i as f64).sin() + 2.1).collect();
        let yp: Vec<f64> = (0..NOBS).map(|i| 4.0 * (2.0 * i as f64).cos()).collect();
        let zp: Vec<f64> = (0..NOBS).map(|i| (0.1 * i as f64).exp()).collect();
        let xyzp = (&xp[..], &yp[..], &zp[..]);

        // Some output storage
        // Initialize with different values for each buffer to test zeroing
        let out0 = &mut [0.0; NOBS];
        let out1 = &mut [1.0; NOBS];
        let out2 = &mut [2.0; NOBS];
        let out3 = &mut [3.0; NOBS];
        let out4 = &mut [4.0; NOBS];
        let out5 = &mut [5.0; NOBS];

        // Flux density
        let wire_radius = vec![0.0; ifil.len()];
        flux_density_linear_filament(
            xyzp,
            xyzfil,
            dlxyzfil,
            ifil,
            &wire_radius,
            (out0, out1, out2),
        )
        .unwrap();
        flux_density_linear_filament_par(
            xyzp,
            xyzfil,
            dlxyzfil,
            ifil,
            &wire_radius,
            (out3, out4, out5),
        )
        .unwrap();
        for i in 0..NOBS {
            assert_eq!(out0[i], out3[i]);
            assert_eq!(out1[i], out4[i]);
            assert_eq!(out2[i], out5[i]);
        }

        // Reinit to test zeroing
        let out0 = &mut [0.0; NOBS];
        let out1 = &mut [1.0; NOBS];
        let out2 = &mut [2.0; NOBS];
        let out3 = &mut [3.0; NOBS];
        let out4 = &mut [4.0; NOBS];
        let out5 = &mut [5.0; NOBS];

        // Vector potential
        let wire_radius = vec![0.0; ifil.len()];
        vector_potential_linear_filament(
            xyzp,
            xyzfil,
            dlxyzfil,
            ifil,
            &wire_radius,
            (out0, out1, out2),
        )
        .unwrap();
        vector_potential_linear_filament_par(
            xyzp,
            xyzfil,
            dlxyzfil,
            ifil,
            &wire_radius,
            (out3, out4, out5),
        )
        .unwrap();
        for i in 0..NOBS {
            assert_eq!(out0[i], out3[i]);
            assert_eq!(out1[i], out4[i]);
            assert_eq!(out2[i], out5[i]);
        }
    }

    #[test]
    fn test_vector_potential_matrix_contracts_to_vector() {
        const NFIL: usize = 6;
        const NOBS: usize = 5;

        let xfil: Vec<f64> = (0..=NFIL).map(|i| 0.2 * i as f64).collect();
        let yfil: Vec<f64> = (0..=NFIL).map(|i| (0.3 * i as f64).sin()).collect();
        let zfil: Vec<f64> = (0..=NFIL).map(|i| (0.2 * i as f64).cos()).collect();
        let xyzfil = (&xfil[..NFIL], &yfil[..NFIL], &zfil[..NFIL]);
        let dlxfil: Vec<f64> = (0..NFIL).map(|i| xfil[i + 1] - xfil[i]).collect();
        let dlyfil: Vec<f64> = (0..NFIL).map(|i| yfil[i + 1] - yfil[i]).collect();
        let dlzfil: Vec<f64> = (0..NFIL).map(|i| zfil[i + 1] - zfil[i]).collect();
        let dlxyzfil = (&dlxfil[..], &dlyfil[..], &dlzfil[..]);
        let ifil: Vec<f64> = (0..NFIL).map(|i| 0.5 + i as f64).collect();
        let wire_radius: Vec<f64> = (0..NFIL).map(|i| 1e-3 * (1.0 + i as f64)).collect();

        let xp: Vec<f64> = (0..NOBS).map(|i| 0.1 + 0.4 * i as f64).collect();
        let yp: Vec<f64> = (0..NOBS).map(|i| -0.3 + 0.2 * i as f64).collect();
        let zp: Vec<f64> = (0..NOBS).map(|i| 0.2 - 0.1 * i as f64).collect();
        let xyzp = (&xp[..], &yp[..], &zp[..]);

        let mut ax = vec![0.0; NOBS];
        let mut ay = vec![0.0; NOBS];
        let mut az = vec![0.0; NOBS];
        vector_potential_linear_filament(
            xyzp,
            xyzfil,
            dlxyzfil,
            &ifil,
            &wire_radius,
            (&mut ax, &mut ay, &mut az),
        )
        .unwrap();

        let mut axm = vec![0.0; NOBS * NFIL];
        let mut aym = vec![0.0; NOBS * NFIL];
        let mut azm = vec![0.0; NOBS * NFIL];
        let mut axmp = vec![0.0; NOBS * NFIL];
        let mut aymp = vec![0.0; NOBS * NFIL];
        let mut azmp = vec![0.0; NOBS * NFIL];
        vector_potential_linear_filament_matrix(
            xyzp,
            xyzfil,
            dlxyzfil,
            &ifil,
            &wire_radius,
            (&mut axm, &mut aym, &mut azm),
        )
        .unwrap();
        vector_potential_linear_filament_matrix_par(
            xyzp,
            xyzfil,
            dlxyzfil,
            &ifil,
            &wire_radius,
            (&mut axmp, &mut aymp, &mut azmp),
        )
        .unwrap();

        assert_eq!(axm, axmp);
        assert_eq!(aym, aymp);
        assert_eq!(azm, azmp);
        for j in 0..NOBS {
            let row = j * NFIL;
            assert!(approx(
                ax[j],
                axm[row..row + NFIL].iter().sum(),
                1e-12,
                1e-15
            ));
            assert!(approx(
                ay[j],
                aym[row..row + NFIL].iter().sum(),
                1e-12,
                1e-15
            ));
            assert!(approx(
                az[j],
                azm[row..row + NFIL].iter().sum(),
                1e-12,
                1e-15
            ));
        }
    }

    #[test]
    fn test_flux_density_matrix_contracts_to_vector() {
        const NFIL: usize = 6;
        const NOBS: usize = 5;

        let xfil: Vec<f64> = (0..=NFIL).map(|i| 0.2 * i as f64).collect();
        let yfil: Vec<f64> = (0..=NFIL).map(|i| (0.3 * i as f64).sin()).collect();
        let zfil: Vec<f64> = (0..=NFIL).map(|i| (0.2 * i as f64).cos()).collect();
        let xyzfil = (&xfil[..NFIL], &yfil[..NFIL], &zfil[..NFIL]);
        let dlxfil: Vec<f64> = (0..NFIL).map(|i| xfil[i + 1] - xfil[i]).collect();
        let dlyfil: Vec<f64> = (0..NFIL).map(|i| yfil[i + 1] - yfil[i]).collect();
        let dlzfil: Vec<f64> = (0..NFIL).map(|i| zfil[i + 1] - zfil[i]).collect();
        let dlxyzfil = (&dlxfil[..], &dlyfil[..], &dlzfil[..]);
        let ifil: Vec<f64> = (0..NFIL).map(|i| 0.5 + i as f64).collect();
        let wire_radius: Vec<f64> = (0..NFIL).map(|i| 1e-3 * (1.0 + i as f64)).collect();

        let xp: Vec<f64> = (0..NOBS).map(|i| 0.1 + 0.4 * i as f64).collect();
        let yp: Vec<f64> = (0..NOBS).map(|i| -0.3 + 0.2 * i as f64).collect();
        let zp: Vec<f64> = (0..NOBS).map(|i| 0.2 - 0.1 * i as f64).collect();
        let xyzp = (&xp[..], &yp[..], &zp[..]);

        let mut bx = vec![0.0; NOBS];
        let mut by = vec![0.0; NOBS];
        let mut bz = vec![0.0; NOBS];
        flux_density_linear_filament(
            xyzp,
            xyzfil,
            dlxyzfil,
            &ifil,
            &wire_radius,
            (&mut bx, &mut by, &mut bz),
        )
        .unwrap();

        let mut bxm = vec![0.0; NOBS * NFIL];
        let mut bym = vec![0.0; NOBS * NFIL];
        let mut bzm = vec![0.0; NOBS * NFIL];
        let mut bxmp = vec![0.0; NOBS * NFIL];
        let mut bymp = vec![0.0; NOBS * NFIL];
        let mut bzmp = vec![0.0; NOBS * NFIL];
        flux_density_linear_filament_matrix(
            xyzp,
            xyzfil,
            dlxyzfil,
            &ifil,
            &wire_radius,
            (&mut bxm, &mut bym, &mut bzm),
        )
        .unwrap();
        flux_density_linear_filament_matrix_par(
            xyzp,
            xyzfil,
            dlxyzfil,
            &ifil,
            &wire_radius,
            (&mut bxmp, &mut bymp, &mut bzmp),
        )
        .unwrap();

        assert_eq!(bxm, bxmp);
        assert_eq!(bym, bymp);
        assert_eq!(bzm, bzmp);
        for j in 0..NOBS {
            let row = j * NFIL;
            assert!(approx(
                bx[j],
                bxm[row..row + NFIL].iter().sum(),
                1e-12,
                1e-15
            ));
            assert!(approx(
                by[j],
                bym[row..row + NFIL].iter().sum(),
                1e-12,
                1e-15
            ));
            assert!(approx(
                bz[j],
                bzm[row..row + NFIL].iter().sum(),
                1e-12,
                1e-15
            ));
        }
    }

    #[test]
    fn test_inductance_linear_filaments_contracts_piecewise() {
        const NSRC: usize = 4;
        const NTGT: usize = 5;

        let xsrc: Vec<f64> = (0..NSRC).map(|i| -0.4 + 0.3 * i as f64).collect();
        let ysrc: Vec<f64> = (0..NSRC).map(|i| 0.2 * (i as f64).sin()).collect();
        let zsrc: Vec<f64> = (0..NSRC).map(|i| -0.1 + 0.15 * i as f64).collect();
        let xyzsrc = (&xsrc[..], &ysrc[..], &zsrc[..]);
        let dlxsrc: Vec<f64> = (0..NSRC).map(|i| 0.05 + 0.01 * i as f64).collect();
        let dlysrc: Vec<f64> = (0..NSRC).map(|i| -0.03 + 0.02 * i as f64).collect();
        let dlzsrc: Vec<f64> = (0..NSRC).map(|i| 0.04 - 0.01 * i as f64).collect();
        let dlxyzsrc = (&dlxsrc[..], &dlysrc[..], &dlzsrc[..]);
        let wire_radius: Vec<f64> = (0..NSRC).map(|i| 2e-3 * (1.0 + i as f64)).collect();

        let xtgt: Vec<f64> = (0..NTGT).map(|i| 0.6 + 0.25 * i as f64).collect();
        let ytgt: Vec<f64> = (0..NTGT).map(|i| -0.15 + 0.05 * i as f64).collect();
        let ztgt: Vec<f64> = (0..NTGT).map(|i| 0.3 - 0.07 * i as f64).collect();
        let xyztgt = (&xtgt[..], &ytgt[..], &ztgt[..]);
        let dlxtgt: Vec<f64> = (0..NTGT).map(|i| -0.04 + 0.01 * i as f64).collect();
        let dlytgt: Vec<f64> = (0..NTGT).map(|i| 0.03 + 0.005 * i as f64).collect();
        let dlztgt: Vec<f64> = (0..NTGT).map(|i| 0.02 - 0.004 * i as f64).collect();
        let dlxyztgt = (&dlxtgt[..], &dlytgt[..], &dlztgt[..]);

        let mut out = vec![0.0; NTGT];
        inductance_linear_filaments(xyztgt, dlxyztgt, xyzsrc, dlxyzsrc, &wire_radius, &mut out)
            .unwrap();

        let total =
            inductance_piecewise_linear_filaments(xyzsrc, dlxyzsrc, xyztgt, dlxyztgt, &wire_radius)
                .unwrap();
        assert!(approx(total, out.iter().sum(), 1e-12, 1e-15));

        for j in 0..NTGT {
            let mj = inductance_piecewise_linear_filaments(
                xyzsrc,
                dlxyzsrc,
                (&xtgt[j..j + 1], &ytgt[j..j + 1], &ztgt[j..j + 1]),
                (&dlxtgt[j..j + 1], &dlytgt[j..j + 1], &dlztgt[j..j + 1]),
                &wire_radius,
            )
            .unwrap();
            assert!(approx(mj, out[j], 1e-12, 1e-15));
        }
    }

    #[test]
    fn test_inductance_linear_filaments_matrix_contracts_to_vector() {
        const NSRC: usize = 4;
        const NTGT: usize = 5;

        let xsrc: Vec<f64> = (0..NSRC).map(|i| -0.4 + 0.3 * i as f64).collect();
        let ysrc: Vec<f64> = (0..NSRC).map(|i| 0.2 * (i as f64).sin()).collect();
        let zsrc: Vec<f64> = (0..NSRC).map(|i| -0.1 + 0.15 * i as f64).collect();
        let xyzsrc = (&xsrc[..], &ysrc[..], &zsrc[..]);
        let dlxsrc: Vec<f64> = (0..NSRC).map(|i| 0.05 + 0.01 * i as f64).collect();
        let dlysrc: Vec<f64> = (0..NSRC).map(|i| -0.03 + 0.02 * i as f64).collect();
        let dlzsrc: Vec<f64> = (0..NSRC).map(|i| 0.04 - 0.01 * i as f64).collect();
        let dlxyzsrc = (&dlxsrc[..], &dlysrc[..], &dlzsrc[..]);
        let wire_radius: Vec<f64> = (0..NSRC).map(|i| 2e-3 * (1.0 + i as f64)).collect();

        let xtgt: Vec<f64> = (0..NTGT).map(|i| 0.6 + 0.25 * i as f64).collect();
        let ytgt: Vec<f64> = (0..NTGT).map(|i| -0.15 + 0.05 * i as f64).collect();
        let ztgt: Vec<f64> = (0..NTGT).map(|i| 0.3 - 0.07 * i as f64).collect();
        let xyztgt = (&xtgt[..], &ytgt[..], &ztgt[..]);
        let dlxtgt: Vec<f64> = (0..NTGT).map(|i| -0.04 + 0.01 * i as f64).collect();
        let dlytgt: Vec<f64> = (0..NTGT).map(|i| 0.03 + 0.005 * i as f64).collect();
        let dlztgt: Vec<f64> = (0..NTGT).map(|i| 0.02 - 0.004 * i as f64).collect();
        let dlxyztgt = (&dlxtgt[..], &dlytgt[..], &dlztgt[..]);

        let mut out = vec![0.0; NTGT];
        inductance_linear_filaments(xyztgt, dlxyztgt, xyzsrc, dlxyzsrc, &wire_radius, &mut out)
            .unwrap();

        let mut mm = vec![0.0; NSRC * NTGT];
        let mut mmp = vec![0.0; NSRC * NTGT];
        inductance_linear_filaments_matrix(
            xyztgt,
            dlxyztgt,
            xyzsrc,
            dlxyzsrc,
            &wire_radius,
            &mut mm,
        )
        .unwrap();
        inductance_linear_filaments_matrix_par(
            xyztgt,
            dlxyztgt,
            xyzsrc,
            dlxyzsrc,
            &wire_radius,
            &mut mmp,
        )
        .unwrap();

        assert_eq!(mm, mmp);
        for j in 0..NTGT {
            let contracted = (0..NSRC).map(|i| mm[i * NTGT + j]).sum::<f64>();
            assert!(approx(out[j], contracted, 1e-12, 1e-15));
        }
    }

    #[test]
    fn test_sparse_csc_inductance_matches_selected_dense_entries() {
        let xyzsrc = (&[0.0, 1.0, 2.0][..], &[0.0, 0.1, -0.1][..], &[0.0; 3][..]);
        let dlxyzsrc = (&[0.0; 3][..], &[0.0; 3][..], &[0.8; 3][..]);
        let wire_radius = [0.01, 0.02, 0.03];
        let xyztgt = (
            &[0.2, 1.2, 2.2, 3.2][..],
            &[0.3, -0.2, 0.1, 0.0][..],
            &[0.1, 0.2, -0.1, 0.3][..],
        );
        let dlxyztgt = (&[0.1; 4][..], &[0.05; 4][..], &[0.4, 0.4, 0.0, 0.4][..]);
        let row_indices = [0, 2, 1, 0, 1, 2];
        let column_pointers = [0, 2, 2, 3, 6];

        let mut dense = vec![0.0; 3 * 4];
        inductance_linear_filaments_matrix(
            xyztgt,
            dlxyztgt,
            xyzsrc,
            dlxyzsrc,
            &wire_radius,
            &mut dense,
        )
        .unwrap();

        let mut sparse = vec![f64::NAN; row_indices.len()];
        let mut sparse_par = vec![f64::NAN; row_indices.len()];
        inductance_linear_filaments_sparse_csc(
            xyztgt,
            dlxyztgt,
            xyzsrc,
            dlxyzsrc,
            &wire_radius,
            &row_indices,
            &column_pointers,
            &mut sparse,
        )
        .unwrap();
        inductance_linear_filaments_sparse_csc_par(
            xyztgt,
            dlxyztgt,
            xyzsrc,
            dlxyzsrc,
            &wire_radius,
            &row_indices,
            &column_pointers,
            &mut sparse_par,
        )
        .unwrap();

        assert_eq!(sparse, sparse_par);
        for target in 0..4 {
            for entry in column_pointers[target]..column_pointers[target + 1] {
                let source = row_indices[entry];
                assert!(approx(
                    sparse[entry],
                    dense[source * 4 + target],
                    1e-14,
                    1e-18
                ));
            }
        }
        assert_eq!(sparse[2], 0.0);
    }

    #[test]
    fn test_sparse_csc_inductance_rejects_noncanonical_pattern() {
        let xyz = (&[0.0, 1.0][..], &[0.0; 2][..], &[0.0; 2][..]);
        let dlxyz = (&[0.0; 2][..], &[0.0; 2][..], &[1.0; 2][..]);
        let mut out = [0.0; 2];
        let err = inductance_linear_filaments_sparse_csc(
            xyz,
            dlxyz,
            xyz,
            dlxyz,
            &[0.01; 2],
            &[1, 1],
            &[0, 2, 2],
            &mut out,
        )
        .unwrap_err();
        assert_eq!(
            err,
            "CSC row indices must be sorted and unique within each column"
        );
    }
}
