//! Magnetics calculations for piecewise-linear current filaments.

use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

use crate::{
    chunksize,
    math::{cross3, dot3, rss3},
};

use crate::{MU0_OVER_4PI, macros::*};

/// (m) minimum representable nonzero wire thickness.
const MIN_WIRE_THICKNESS: f64 = 1e-10;

/// Estimate the mutual inductance between two piecewise-linear current filaments.
///
/// Uses filament midpoints as field source and target.
///
/// # Arguments
///
/// * `xyzfil0`:         (m) filament origin coordinates for first path, length `n`
/// * `dlxyzfil0`:       (m) filament segment lengths for first path, length `n`
/// * `xyzfil1`:         (m) filament origin coordinates for second path, length `n`
/// * `dlxyzfil1`:       (m) filament segment lengths for second path, length `n`
/// * `self_inductance`: Flag for whether this calc is being used for self-inductance,
///                      in which case segment self-field terms are replaced with a hand-calc
///
/// # Commentary
///
/// Uses Neumann's Formula for the mutual inductance of arbitrary loops, which is
/// originally from \[2\] and can be found in a more friendly format on wikipedia.
///
/// When `self_inductance` flag is set, zeroes-out the contributions from self-pairings
/// to resolve the thin-filament self-inductance singularity and replaces the
/// segment self-inductance term with an analytic value from equation 4 (with Y=1/2) of \[3\],
/// which is a scalar-per-length value for low-frequency operation (uniform section current).
///
/// # Assumptions
///
/// * Thin, well-behaved filaments
/// * Uniform current distribution within segments
///     * Low frequency operation; no skin effect
///       (which would reduce the segment self-field term)
/// * Vacuum permeability everywhere
/// * Each filament has a constant current in all segments
///   (otherwise we need an inductance matrix)
///
/// # References
///
///   \[1\] “Inductance,” Wikipedia. Dec. 12, 2022. Accessed: Jan. 23, 2023. \[Online\].
///         Available: <https://en.wikipedia.org/w/index.php?title=Inductance>
///
///   \[2\] F. E. Neumann, “Allgemeine Gesetze der inducirten elektrischen Ströme,”
///         Jan. 1846, doi: [10.1002/andp.18461430103](https://doi.org/10.1002/andp.18461430103).
///
///   \[3\] R. Dengler, “Self inductance of a wire loop as a curve integral,”
///         AEM, vol. 5, no. 1, p. 1, Jan. 2016, doi: [10.7716/aem.v5i1.331](https://doi.org/10.7716/aem.v5i1.331).
pub fn inductance_piecewise_linear_filaments(
    xyzfil0: (&[f64], &[f64], &[f64]),
    dlxyzfil0: (&[f64], &[f64], &[f64]),
    xyzfil1: (&[f64], &[f64], &[f64]),
    dlxyzfil1: (&[f64], &[f64], &[f64]),
    self_inductance: bool,
) -> Result<f64, &'static str> {
    // Unpack
    let (xfil0, yfil0, zfil0) = xyzfil0;
    let (dlxfil0, dlyfil0, dlzfil0) = dlxyzfil0;
    let (xfil1, yfil1, zfil1) = xyzfil1;
    let (dlxfil1, dlyfil1, dlzfil1) = dlxyzfil1;

    // Check lengths; Error if they do not match
    let n = xfil0.len();
    check_length!(n, xfil0, yfil0, zfil0, dlxfil0, dlyfil0, dlzfil0);

    let m = xfil1.len();
    check_length!(m, xfil1, yfil1, zfil1, dlxfil1, dlyfil1, dlzfil1);

    if self_inductance && m != n {
        return Err(
            "For self-inductance runs, the two paths must be the same length and should be identical",
        );
    }

    let mut inductance: f64 = 0.0; // [H], although it is in [m] until the final calc
    let mut total_length: f64 = 0.0; // [m]
    for i in 0..n {
        // Filament i midpoint
        let dlxi = dlxfil0[i]; // [m]
        let dlyi = dlyfil0[i]; // [m]
        let dlzi = dlzfil0[i]; // [m]
        let xmidi = dlxi.mul_add(0.5, xfil0[i]); // [m]
        let ymidi = dlyi.mul_add(0.5, yfil0[i]); // [m]
        let zmidi = dlzi.mul_add(0.5, zfil0[i]); // [m]

        // Accumulate total length if we need it
        if self_inductance {
            total_length += rss3(dlxi, dlyi, dlzi);
        }

        for j in 0..m {
            // Skip self-interaction terms which are handled separately
            if self_inductance && i == j {
                continue;
            }

            // Filament j midpoint
            let dlxj = dlxfil1[j]; // [m]
            let dlyj = dlyfil1[j]; // [m]
            let dlzj = dlzfil1[j]; // [m]
            let xmidj = dlxj.mul_add(0.5, xfil1[j]); // [m]
            let ymidj = dlyj.mul_add(0.5, yfil1[j]); // [m]
            let zmidj = dlzj.mul_add(0.5, zfil1[j]); // [m]

            // Distance between midpoints
            let rx = xmidi - xmidj;
            let ry = ymidi - ymidj;
            let rz = zmidi - zmidj;
            let dist = rss3(rx, ry, rz);

            // Dot product of segment vectors
            let dxdot = dot3(dlxi, dlyi, dlzi, dlxj, dlyj, dlzj);

            inductance += dxdot / dist;
        }
    }

    // Add self-inductance of individual filament segments
    // if this is a self-inductance calc
    if self_inductance {
        inductance += 0.5 * total_length;
    }

    // Finally, do the shared constant factor
    inductance *= MU0_OVER_4PI;

    Ok(inductance)
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

/// Biot-Savart calculation for B-field contribution one filament
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
///        |
///        |  d_perp (from line to p)
///        |
///        q (closest point on line)
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
/// Inside the wire radius, the formula is modified to blend linearly to zero at the wire center.
///
/// ## References
///
/// * \[1\] D. J. Griffiths, Introduction to electrodynamics, Fourth edition. Boston: Pearson, 2014.
/// * \[2\] J. van Nugteren and N. Deelen, “rat-mlfmm,” GitLab repository. Accessed: Jan. 16, 2026. [Online].
///         Available: https://gitlab.com/Project-Rat/rat-mlfmm/-/tree/1e1d387522fafac50c0540af1ebb15d1d506d33d
/// * \[3\] M. Zahn, “5.4: The Vector Potential,” Engineering LibreTexts. Accessed: Jan. 20, 2026. [Online].
///         Available: https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Electro-Optics/Electromagnetic_Field_Theory%3A_A_Problem_Solving_Approach_(Zahn)/05%3A_The_Magnetic_Field/5.04%3A_The_Vector_Potential
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
pub fn flux_density_linear_filament_scalar(
    xyzifil: ((f64, f64, f64), (f64, f64, f64), f64),
    wire_radius: f64,
    xyzobs: (f64, f64, f64),
) -> (f64, f64, f64) {
    use crate::math::{PointLineDistance, point_line_distance_with_endpoints};

    // Unpack
    let (start, end, ifil) = xyzifil;

    // Get perpendicular distance and distance from each endpoint to the target,
    // and a fraction between 0 and 1 representing how far the point is from the center of the wire
    // to the edge of the wire.
    // All 3 distances are clamped to at least the wire radius.
    let PointLineDistance {
        perp,
        perp_hat,
        dist_a,
        dist_b,
        frac,
        para_a,
        para_b,
        ab_norm: dlhat,
    } = point_line_distance_with_endpoints(start, end, xyzobs, wire_radius);

    // Sine of the angle formed by the lines from the target to each endpoint
    // and the line of the filament.
    let sin_theta_a = para_a / dist_a; // (dimensionless)
    let sin_theta_b = para_b / dist_b; // (dimensionless)

    // Geometric component of B-field magnitude,
    // including linear falloff inside finite-thickness wire.
    let geometric_factor = -frac * (sin_theta_b - sin_theta_a); // (dimensionless)

    // This factor is constant across all x, y, and z components.
    let c = geometric_factor * MU0_OVER_4PI * ifil / perp; // (A/m)

    // Direction of cross(dL, r), the direction of the field.
    let (cx, cy, cz) = cross3(
        dlhat.0, dlhat.1, dlhat.2, perp_hat.0, perp_hat.1, perp_hat.2,
    ); // (dimensionless)

    // Assemble final B-field components.
    let bx = c * cx; // [T]
    let by = c * cy; // [T]
    let bz = c * cz; // [T]

    // Finally, determine whether we are clipping to zero.
    if frac > 1e6 * f64::EPSILON && perp > MIN_WIRE_THICKNESS {
        return (bx, by, bz);
    } else {
        return (0.0, 0.0, 0.0);
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

/// Vector potential (A-field) from a linear current filament segment to an observation point.
///
/// Uses the formula for finite segment length and finite wire thickness.
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
pub fn vector_potential_linear_filament_scalar(
    xyzifil: ((f64, f64, f64), (f64, f64, f64), f64),
    wire_radius: f64,
    xyzobs: (f64, f64, f64),
) -> (f64, f64, f64) {
    use crate::math::{PointLineDistance, point_line_distance_with_endpoints};

    // Unpack
    let (start, end, ifil) = xyzifil;

    // Get perpendicular distance and distance from each endpoint to the target,
    // and a fraction between 0 and 1 representing how far the point is from the center of the wire
    // to the edge of the wire.
    // All 3 distances are clamped to at least the wire radius.
    let PointLineDistance {
        perp,
        dist_a,
        dist_b,
        frac,
        para_a,
        para_b,
        ab_norm: dlhat,
        perp_hat: _,
    } = point_line_distance_with_endpoints(start, end, xyzobs, wire_radius);

    // Finite segment length log-form with quadratic blend to zero at axis.
    let k1 = -para_b + dist_b;
    let k2 = -para_a + dist_a;
    let frac2 = frac * frac; // Quadratic fall-off (as opposed to linear for B-field)
    let a_mag = frac2 * MU0_OVER_4PI * ifil * libm::log((k1 / k2).max(0.0));

    // Direction is always aligned with the segment.
    let (ax, ay, az) = (a_mag * dlhat.0, a_mag * dlhat.1, a_mag * dlhat.2);

    // Finally, determine whether we are clipping to zero.
    if frac > 1e6 * f64::EPSILON && perp > MIN_WIRE_THICKNESS {
        return (ax, ay, az);
    } else {
        return (0.0, 0.0, 0.0);
    }
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
    cross3(jobs.0, jobs.1, jobs.2, bx, by, bz) // [N/m^3]
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
    use std::f64::consts::PI;

    use super::*;
    use crate::physics::point_source::segment::{
        flux_density_point_segment, vector_potential_point_segment,
    };
    use crate::testing::*;

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
                let rxjxb = cross3(r.0, r.1, r.2, jxbx[j], jxby[j], jxbz[j]);
                // Linear filaments aren't perfectly aligned, so we need a slighter wider tolerance here
                assert!(approx(0.0, rss3(rxjxb.0, rxjxb.1, rxjxb.2), rtol, 1e-8));
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
        let ifil = [1.0];

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
        let ifil_ps = vec![1.0; nseg];

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
                "bx is {}, should be {}",
                bx[i],
                bx_ps[i]
            );
            assert!(
                approx(by[i], by_ps[i], rtol, atol),
                "by is {}, should be {}",
                by[i],
                by_ps[i]
            );
            assert!(
                approx(bz[i], bz_ps[i], rtol, atol),
                "bz is {}, should be {}",
                bz[i],
                bz_ps[i]
            );
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

    /// Centerline values should be finite and not NaN.
    #[test]
    fn test_flux_density_centerline_finite() {
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

        let xp = [0.0, 0.0, 0.0];
        let yp = [0.0, 0.0, 0.0];
        let zp = [-0.25, 0.0, 0.25];

        let mut bx = [0.0; 3];
        let mut by = [0.0; 3];
        let mut bz = [0.0; 3];
        flux_density_linear_filament(
            (&xp, &yp, &zp),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            &[wire_radius],
            (&mut bx, &mut by, &mut bz),
        )
        .unwrap();

        for i in 0..xp.len() {
            assert!(bx[i].is_finite(), "bx[{}] is {}", i, bx[i]);
            assert!(by[i].is_finite(), "by[{}] is {}", i, by[i]);
            assert!(bz[i].is_finite(), "bz[{}] is {}", i, bz[i]);
        }
    }

    /// Compare single-segment vector potential against discretized point-source segments.
    #[test]
    fn test_vector_potential_against_point_segment_discretization() {
        let (rtol, atol) = (1e-6, 1e-15);

        let start = (0.0, 0.0, -0.5);
        let end = (0.0, 0.0, 0.5);
        let ifil = [1.0];

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
        let ifil_ps = vec![1.0; nseg];

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
                "ax is {}, should be {}",
                ax[i],
                ax_ps[i]
            );
            assert!(
                approx(ay[i], ay_ps[i], rtol, atol),
                "ay is {}, should be {}",
                ay[i],
                ay_ps[i]
            );
            assert!(
                approx(az[i], az_ps[i], rtol, atol),
                "az is {}, should be {}",
                az[i],
                az_ps[i]
            );
        }
    }

    /// Check quadratic falloff inside finite wire radius.
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
        for (i, &ratio) in ratios.iter().enumerate() {
            let expected = a_edge * ratio * ratio;
            assert!(
                approx(expected, amag[i], rtol, atol),
                "ratio {}: |A| = {:.6e}, expected {:.6e}",
                ratio,
                amag[i],
                expected
            );
        }
    }

    /// Centerline values should be finite and not NaN.
    #[test]
    fn test_vector_potential_centerline_finite() {
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

        let xp = [0.0, 0.0, 0.0];
        let yp = [0.0, 0.0, 0.0];
        let zp = [-0.25, 0.0, 0.25];

        let mut ax = [0.0; 3];
        let mut ay = [0.0; 3];
        let mut az = [0.0; 3];
        vector_potential_linear_filament(
            (&xp, &yp, &zp),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            &[wire_radius],
            (&mut ax, &mut ay, &mut az),
        )
        .unwrap();

        for i in 0..xp.len() {
            assert!(ax[i].is_finite(), "ax[{}] is {}", i, ax[i]);
            assert!(ay[i].is_finite(), "ay[{}] is {}", i, ay[i]);
            assert!(az[i].is_finite(), "az[{}] is {}", i, az[i]);
        }
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

        let xmid2: Vec<f64> = xfil2
            .iter()
            .zip(dlxfil2.iter())
            .map(|(x, dx)| x + dx / 2.0)
            .collect();
        let ymid2: Vec<f64> = yfil2
            .iter()
            .zip(dlyfil2.iter())
            .map(|(x, dx)| x + dx / 2.0)
            .collect();
        let zmid2: Vec<f64> = zfil2
            .iter()
            .zip(dlzfil2.iter())
            .map(|(x, dx)| x + dx / 2.0)
            .collect();

        // Check against Neumann's formula for mutual inductance
        let outx = &mut [0.0; NFIL - 1];
        let outy = &mut [0.0; NFIL - 1];
        let outz = &mut [0.0; NFIL - 1];
        vector_potential_linear_filament(
            (&xmid2, &ymid2, &zmid2),
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
        // Because inductance_piecewise_linear_filaments uses Neumann's formula, which is
        // exactly equivalent to the point-source formulation of the vector potential,
        // we expect a small amount of error to the finite-length segment formula here.
        let a_dot_dl: Vec<f64> = (0..NFIL - 1)
            .map(|i| outx[i] * dlxfil2[i] + outy[i] * dlyfil2[i] + outz[i] * dlzfil2[i])
            .collect();
        let m_from_a = a_dot_dl.iter().sum();
        let m = inductance_piecewise_linear_filaments(
            (&xyz, &xyz, &xyz),
            (&dlxyz, &dlxyz, &dlxyz),
            xyzfil2,
            dlxyzfil2,
            false,
        )
        .unwrap();
        assert!(
            approx(m, m_from_a, 1e-2, 1e-15),
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
                    let r = rss3(*x, *y, *z);
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
}
