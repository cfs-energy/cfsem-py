//! Magnetics calculations for circular current filaments.

use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

use crate::{
    chunksize,
    macros::{check_length, check_length_3tup, mut_par_chunks_3tup, par_chunks_3tup},
    math::{cross3, dot3, ellipe, ellipk, rss3},
};

use crate::{MU_0, MU0_OVER_4PI};

/// Flux contributions from some circular filaments to some observation points, which happens to be
/// the Green's function for the Grad-Shafranov elliptic operator, $\Delta^{\*}$.
/// This variant of the function is parallelized over chunks of observation points.
///
/// # Arguments
///
/// * `rzifil`:  (m, m, A-turns) r-coord, z-coord, and current of each filament, length `m`
/// * `rzobs`:   (m, m) r-coord, and z-coord of each observation point, length `n`
/// * `out`:     (Wb), poloidal flux at observation location, length `n`
///
/// # Commentary
///
/// Represents contribution from a current at (R, Z) to an observation point at (Rprime, Zprime)
///
/// Note Jardin's 4.61-4.66 presents it with a different definition of
/// the elliptic integrals from what is used here and in scipy.
///
/// # References
///
///   \[1\] D. Kaltsas, A. Kuiroukidis, and G. Throumoulopoulos, “A tokamak pertinent analytic equilibrium with plasma flow of arbitrary direction,”
///         Physics of Plasmas, vol. 26, p. 124501, Dec. 2019,
///         doi: [10.1063/1.5120341](https://doi.org/10.1063/1.5120341).
///
///   \[2\] S. Jardin, *Computational Methods in Plasma Physics*, 1st ed. USA: CRC Press, Inc., 2010.
///
///   \[3\] J. Huang and J. Menard, “Development of an Auto-Convergent Free-Boundary Axisymmetric Equilibrium Solver,”
///         Journal of Undergraduate Research, vol. 6, Jan. 2006, Accessed: May 05, 2021. \[Online\].
///         Available: <https://www.osti.gov/biblio/1051805-development-auto-convergent-free-boundary-axisymmetric-equilibrium-solver>
///
///   \[4\] J. C. Simpson, J. E. Lane, C. D. Immer, R. C. Youngquist, and T. Steinrock,
///         “Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop,”
///         Jan. 01, 2001. Accessed: Sep. 06, 2022. [Online]. Available: <https://ntrs.nasa.gov/citations/20010038494>
pub fn flux_circular_filament_par(
    rzifil: (&[f64], &[f64], &[f64]),
    rzobs: (&[f64], &[f64]),
    out: &mut [f64],
) -> Result<(), &'static str> {
    // Unpack
    let (rprime, zprime) = rzobs;

    // Chunk inputs
    let n = chunksize(rprime.len());
    let rprimec = rprime.par_chunks(n);
    let zprimec = zprime.par_chunks(n);
    let outc = out.par_chunks_mut(n);

    // Run calcs
    (outc, rprimec, zprimec)
        .into_par_iter()
        .try_for_each(|(outc, rc, zc)| flux_circular_filament(rzifil, (rc, zc), outc))?;

    Ok(())
}

/// Flux contributions from some circular filaments to some observation points, which happens to be
/// the Green's function for the Grad-Shafranov elliptic operator, $\Delta^{\*}$.
///
/// # Arguments
///
/// * `rzifil`:  (m, m, A-turns) r-coord, z-coord, and current of each filament, length `m`
/// * `rzobs`:   (m, m) r-coord, and z-coord of each observation point, length `n`
/// * `out`:     (Wb), poloidal flux at observation location, length `n`
///
/// # Commentary
///
/// Represents contribution from a current at (R, Z) to an observation point at (Rprime, Zprime)
///
/// Note Jardin's 4.61-4.66 presents it with a different definition of
/// the elliptic integrals from what is used here and in scipy.
///
/// # References
///
///   \[1\] D. Kaltsas, A. Kuiroukidis, and G. Throumoulopoulos, “A tokamak pertinent analytic equilibrium with plasma flow of arbitrary direction,”
///         Physics of Plasmas, vol. 26, p. 124501, Dec. 2019,
///         doi: [10.1063/1.5120341](https://doi.org/10.1063/1.5120341).
///
///   \[2\] S. Jardin, *Computational Methods in Plasma Physics*, 1st ed. USA: CRC Press, Inc., 2010.
///
///   \[3\] J. Huang and J. Menard, “Development of an Auto-Convergent Free-Boundary Axisymmetric Equilibrium Solver,”
///         Journal of Undergraduate Research, vol. 6, Jan. 2006, Accessed: May 05, 2021. \[Online\].
///         Available: <https://www.osti.gov/biblio/1051805-development-auto-convergent-free-boundary-axisymmetric-equilibrium-solver>
///
///   \[4\] J. C. Simpson, J. E. Lane, C. D. Immer, R. C. Youngquist, and T. Steinrock,
///         “Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop,”
///         Jan. 01, 2001. Accessed: Sep. 06, 2022. [Online]. Available: <https://ntrs.nasa.gov/citations/20010038494>
pub fn flux_circular_filament(
    rzifil: (&[f64], &[f64], &[f64]),
    rzobs: (&[f64], &[f64]),
    out: &mut [f64],
) -> Result<(), &'static str> {
    // Unpack
    let (rfil, zfil, ifil) = rzifil;
    let (rprime, zprime) = rzobs;

    // Check lengths; Error if they do not match
    let m: usize = ifil.len();
    let n: usize = rprime.len();
    check_length_3tup!(m, &rzifil);
    check_length!(n, rprime, zprime);
    check_length!(n, out);

    // Zero output
    out.fill(0.0);

    for i in 0..n {
        for j in 0..m {
            // The inner function is inlined, so values that are reused between iterations
            // can be pulled to the outer scope by the compiler and do not affect performance
            out[i] +=
                flux_circular_filament_scalar((rfil[j], zfil[j], ifil[j]), (rprime[i], zprime[i]));
        }
    }

    Ok(())
}

/// Flux contributions from some circular filaments to some observation points, which happens to be
/// the Green's function for the Grad-Shafranov elliptic operator, $\Delta^{\*}$.
///
/// # Arguments
///
/// * `rzifil`:  (m, m, A-turns) r-coord, z-coord, and current of filament
/// * `rzobs`:   (m, m) r-coord, and z-coord of observation point
///
/// # Returns
///
/// * `psi`: (Wb) or (H-A) or (T-m^2) or (V-s), poloidal flux at observation location
///
/// # Commentary
///
/// Represents contribution from a current at (R, Z) to an observation point at (Rprime, Zprime)
///
/// Note Jardin's 4.61-4.66 presents it with a different definition of
/// the elliptic integrals from what is used here and in scipy.
///
/// # References
///
///   \[1\] D. Kaltsas, A. Kuiroukidis, and G. Throumoulopoulos, “A tokamak pertinent analytic equilibrium with plasma flow of arbitrary direction,”
///         Physics of Plasmas, vol. 26, p. 124501, Dec. 2019,
///         doi: [10.1063/1.5120341](https://doi.org/10.1063/1.5120341).
///
///   \[2\] S. Jardin, *Computational Methods in Plasma Physics*, 1st ed. USA: CRC Press, Inc., 2010.
///
///   \[3\] J. Huang and J. Menard, “Development of an Auto-Convergent Free-Boundary Axisymmetric Equilibrium Solver,”
///         Journal of Undergraduate Research, vol. 6, Jan. 2006, Accessed: May 05, 2021. \[Online\].
///         Available: <https://www.osti.gov/biblio/1051805-development-auto-convergent-free-boundary-axisymmetric-equilibrium-solver>
///
///   \[4\] J. C. Simpson, J. E. Lane, C. D. Immer, R. C. Youngquist, and T. Steinrock,
///         “Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop,”
///         Jan. 01, 2001. Accessed: Sep. 06, 2022. [Online]. Available: <https://ntrs.nasa.gov/citations/20010038494>
#[inline]
pub fn flux_circular_filament_scalar(rzifil: (f64, f64, f64), rzobs: (f64, f64)) -> f64 {
    // Unpack
    let (rfil, zfil, ifil) = rzifil;
    let (rprime, zprime) = rzobs;
    // Evaluate
    let rrprime = rfil * rprime;
    let r_plus_rprime = rfil + rprime;
    let z_minus_zprime = zfil - zprime;
    let k2 = 4.0 * rrprime / (r_plus_rprime.powi(2) + z_minus_zprime.powi(2));
    // [V-s]
    MU_0 * ifil * (rrprime / k2).sqrt() * ((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2))
}

/// Off-axis Br,Bz components for a circular current filament in vacuum.
/// This variant of the function is parallelized over chunks of observation points.
///
/// # Arguments
///
/// * `rzifil`:  (m, m, A-turns) r-coord, z-coord, and current of each filament, length `m`
/// * `rzobs`:   (m, m) r-coord, and z-coord of each observation point, length `n`
/// * `out`:     (T, T), r- and z-components of magnetic flux density at observation location, length `n`
///
/// # Commentary
///
/// Near-exact formula (except numerically-evaluated elliptic integrals).
/// See eqns. 12,13 pg. 34 in \[1\], eqn 9.8.7 in \[2\], and all of \[3\].
///
/// Note the formula for Br as given by \[1\] is incorrect and does not satisfy the
/// constraints of the calculation without correcting by a factor of (z / r).
///
/// # References
///
///   \[1\] D. B. Montgomery and J. Terrell,
///         “Some Useful Information For The Design Of Aircore Solenoids,
///         Part I. Relationships Between Magnetic Field, Power, Ampere-Turns
///         And Current Density. Part II. Homogeneous Magnetic Fields,”
///         Massachusetts Inst. Of Tech. Francis Bitter National Magnet Lab, Cambridge, MA,
///         Nov. 1961. Accessed: May 18, 2021. \[Online\].
///         Available: <https://apps.dtic.mil/sti/citations/tr/AD0269073>
///
///   \[2\] 8.02 Course Notes. Available: <https://web.mit.edu/8.02t/www/802TEAL3D/visualizations/coursenotes/modules/guide09.pdf>
///
///   \[3\] Eric Dennyson, "Magnet Formulas". Available: <https://tiggerntatie.github.io/emagnet-py/offaxis/off_axis_loop.html>
///
///   \[4\] J. C. Simpson, J. E. Lane, C. D. Immer, R. C. Youngquist, and T. Steinrock,
///         “Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop,”
///         Jan. 01, 2001. Accessed: Sep. 06, 2022. [Online]. Available: <https://ntrs.nasa.gov/citations/20010038494>
pub fn flux_density_circular_filament_par(
    rzifil: (&[f64], &[f64], &[f64]),
    rzobs: (&[f64], &[f64]),
    out: (&mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    // Unpack
    let (rprime, zprime) = rzobs;
    let (out_r, out_z) = out;

    // Chunk inputs
    let n = chunksize(rprime.len());

    let rprimec = rprime.par_chunks(n);
    let zprimec = zprime.par_chunks(n);

    let outrc = out_r.par_chunks_mut(n);
    let outzc = out_z.par_chunks_mut(n);

    // Run calcs
    (outrc, outzc, rprimec, zprimec)
        .into_par_iter()
        .try_for_each(|(orc, ozc, rc, zc)| {
            flux_density_circular_filament(rzifil, (rc, zc), (orc, ozc))
        })?;

    Ok(())
}

/// Off-axis Br,Bz components for a circular current filament in vacuum.
///
/// # Arguments
///
/// * `rzifil`:  (m, m, A-turns) r-coord, z-coord, and current of each filament, length `m`
/// * `rzobs`:   (m, m) r-coord, and z-coord of each observation point, length `n`
/// * `out`:     (T, T), r- and z-components of magnetic flux density at observation location, length `n`
///
/// # Commentary
///
/// Near-exact formula (except numerically-evaluated elliptic integrals).
/// See eqns. 12,13 pg. 34 in \[1\], eqn 9.8.7 in \[2\], and all of \[3\].
///
/// Note the formula for Br as given by \[1\] is incorrect and does not satisfy the
/// constraints of the calculation without correcting by a factor of (z / r).
///
/// # References
///
///   \[1\] D. B. Montgomery and J. Terrell,
///         “Some Useful Information For The Design Of Aircore Solenoids,
///         Part I. Relationships Between Magnetic Field, Power, Ampere-Turns
///         And Current Density. Part II. Homogeneous Magnetic Fields,”
///         Massachusetts Inst. Of Tech. Francis Bitter National Magnet Lab, Cambridge, MA,
///         Nov. 1961. Accessed: May 18, 2021. \[Online\].
///         Available: <https://apps.dtic.mil/sti/citations/tr/AD0269073>
///
///   \[2\] 8.02 Course Notes. Available: <https://web.mit.edu/8.02t/www/802TEAL3D/visualizations/coursenotes/modules/guide09.pdf>
///
///   \[3\] Eric Dennyson, "Magnet Formulas". Available: <https://tiggerntatie.github.io/emagnet-py/offaxis/off_axis_loop.html>
///
///   \[4\] J. C. Simpson, J. E. Lane, C. D. Immer, R. C. Youngquist, and T. Steinrock,
///         “Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop,”
///         Jan. 01, 2001. Accessed: Sep. 06, 2022. [Online]. Available: <https://ntrs.nasa.gov/citations/20010038494>
pub fn flux_density_circular_filament(
    rzifil: (&[f64], &[f64], &[f64]),
    rzobs: (&[f64], &[f64]),
    out: (&mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    // Unpack
    let (rfil, zfil, ifil) = rzifil;
    let (rprime, zprime) = rzobs;
    let (out_r, out_z) = out;

    // Check lengths
    let n = ifil.len();
    let m = rprime.len();
    check_length_3tup!(n, &rzifil);
    check_length!(m, &out_r, &out_z);

    // Zero output
    out_r.fill(0.0);
    out_z.fill(0.0);

    // There aren't necessarily more observation points or filaments, depending on the use case.
    // The more common extreme is to see a very large number of filaments evaluated at a smaller
    // number of observation points. However, this particular calc suffers badly when iterating
    // over observation points first, so to capture a 50% speedup for cases with >=10 observation
    // points at the expense of a 30% slowdown for evaluating single observation points, we
    // iterate over filaments first here.
    for i in 0..n {
        for j in 0..m {
            // The inner function is inlined, so values that are reused between iterations
            // can be pulled to the outer scope by the compiler and do not affect performance
            let (br, bz) = flux_density_circular_filament_scalar(
                (rfil[i], zfil[i], ifil[i]),
                (rprime[j], zprime[j]),
            );
            out_r[j] += br;
            out_z[j] += bz;
        }
    }

    Ok(())
}

/// Off-axis Br,Bz components for a circular current filament in vacuum.
///
/// # Arguments
///
/// * `rzifil`:  (m, m, A-turns) r-coord, z-coord, and current of filament, length `m`
/// * `rzobs`:   (m, m) r-coord, and z-coord of observation point, length `n`
///
/// # Returns
///
/// * `(br, bz)`:   (T, T), r- and z-component of magnetic flux density at observation location
///
/// # Commentary
///
/// Near-exact formula (except numerically-evaluated elliptic integrals).
/// See eqns. 12,13 pg. 34 in \[1\], eqn 9.8.7 in \[2\], and all of \[3\].
///
/// Note the formula for Br as given by \[1\] is incorrect and does not satisfy the
/// constraints of the calculation without correcting by a factor of (z / r).
///
/// # References
///
///   \[1\] D. B. Montgomery and J. Terrell,
///         “Some Useful Information For The Design Of Aircore Solenoids,
///         Part I. Relationships Between Magnetic Field, Power, Ampere-Turns
///         And Current Density. Part II. Homogeneous Magnetic Fields,”
///         Massachusetts Inst. Of Tech. Francis Bitter National Magnet Lab, Cambridge, MA,
///         Nov. 1961. Accessed: May 18, 2021. \[Online\].
///         Available: <https://apps.dtic.mil/sti/citations/tr/AD0269073>
///
///   \[2\] 8.02 Course Notes. Available: <https://web.mit.edu/8.02t/www/802TEAL3D/visualizations/coursenotes/modules/guide09.pdf>
///
///   \[3\] Eric Dennyson, "Magnet Formulas". Available: <https://tiggerntatie.github.io/emagnet-py/offaxis/off_axis_loop.html>
///
///   \[4\] J. C. Simpson, J. E. Lane, C. D. Immer, R. C. Youngquist, and T. Steinrock,
///         “Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop,”
///         Jan. 01, 2001. Accessed: Sep. 06, 2022. [Online]. Available: <https://ntrs.nasa.gov/citations/20010038494>
#[inline]
pub fn flux_density_circular_filament_scalar(
    rzifil: (f64, f64, f64),
    rzobs: (f64, f64),
) -> (f64, f64) {
    // Unpack
    let (rfil, zfil, ifil) = rzifil;
    let (rprime, zprime) = rzobs;

    // Evaluate
    let z = zprime - zfil; // [m]

    let z2 = z * z; // [m^2]
    let r2 = rprime * rprime; // [m^2]

    let rpr = rfil + rprime;

    let q = rpr.mul_add(rpr, z2); // [m^2]
    let k2 = 4.0 * rfil * rprime / q; // [nondim]

    let a0 = 2.0 * ifil / q.sqrt(); // [A/m]

    let f = ellipk(k2); // [nondim]
    let s = ellipe(k2) / (1.0 - k2); // [nondim]

    // Bake some reusable values
    let s_over_q = s / q; // [m^-2]
    let rfil2 = rfil * rfil; // [m^2]

    // Magnetic field intensity, less the factor of 4pi that we have adjusted out of mu_0
    let hr = (z / rprime) * a0 * s_over_q.mul_add(rfil2 + r2 + z2, -f);
    let hz = a0 * s_over_q.mul_add(rfil2 - r2 - z2, f);

    // Magnetic flux density assuming vacuum permeability
    let br = MU0_OVER_4PI * hr;
    let bz = MU0_OVER_4PI * hz;

    (br, bz)
}

/// Flux density of a circular filament in cartesian form
/// at a location given in cartesian coordinates.
///
/// For additional documentation and commentary, see [flux_density_circular_filament_scalar].
#[inline]
pub fn flux_density_circular_filament_cartesian_scalar(
    rzifil: (f64, f64, f64),
    xyzobs: (f64, f64, f64),
) -> (f64, f64, f64) {
    // Unpack
    let (x, y, z) = xyzobs;
    // Convert cartesian point to cylindrical
    let (robs, phiobs, zobs) = crate::math::cartesian_to_cylindrical(x, y, z);
    // Get axisymmetric B-field
    let (br, bz) = flux_density_circular_filament_scalar(rzifil, (robs, zobs));
    // Convert axisymmetric B-field to cartesian
    let (bx, by, bz) = (br * libm::cos(phiobs), br * libm::sin(phiobs), bz);
    (bx, by, bz)
}

/// Flux density of a circular filament in cartesian form
/// at a set of locations given in cartesian coordinates.
///
/// For additional documentation and commentary, see [flux_density_circular_filament_scalar].
pub fn flux_density_circular_filament_cartesian(
    rzifil: (&[f64], &[f64], &[f64]),
    xyzobs: (&[f64], &[f64], &[f64]),
    bxyz_out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    // Unpack
    let (rfil, zfil, ifil) = rzifil;
    let (x, y, z) = xyzobs;
    let (bx, by, bz) = bxyz_out;

    // Check lengths
    let n = ifil.len();
    check_length_3tup!(n, &rzifil);

    let m = x.len();
    check_length_3tup!(m, &xyzobs);
    check_length!(m, bx, by, bz);

    // Zero output
    bx.fill(0.0);
    by.fill(0.0);
    bz.fill(0.0);

    // Do calcs
    // Because we will parallelize over chunks of output points to avoid mutexes,
    // the inner loop is over the circular filaments s.t. performance remains viable
    // when examining the contribution of a large number of filaments to a small
    // number of observation points.
    for j in 0..m {
        for i in 0..n {
            // The inner function is inlined, so values that are reused between iterations
            // can be pulled to the outer scope by the compiler and do not affect performance
            let rzifil_i = (rfil[i], zfil[i], ifil[i]);
            let xyzobs_j = (x[j], y[j], z[j]);
            let (bxo, byo, bzo) =
                flux_density_circular_filament_cartesian_scalar(rzifil_i, xyzobs_j);
            bx[j] += bxo;
            by[j] += byo;
            bz[j] += bzo;
        }
    }

    Ok(())
}

/// Flux density of a circular filament in cartesian form
/// at a set of locations given in cartesian coordinates.
/// Parallelized over chunks of observation points.
///
/// For additional documentation and commentary, see [flux_density_circular_filament_scalar].
pub fn flux_density_circular_filament_cartesian_par(
    rzifil: (&[f64], &[f64], &[f64]),
    xyzobs: (&[f64], &[f64], &[f64]),
    bxyz_out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    // Chunk
    let n = chunksize(xyzobs.0.len());
    let (xc, yc, zc) = par_chunks_3tup!(xyzobs, n);
    let (outbxc, outbyc, outbzc) = mut_par_chunks_3tup!(bxyz_out, n);

    // Evaluate
    (xc, yc, zc, outbxc, outbyc, outbzc)
        .into_par_iter()
        .try_for_each(|(xci, yci, zci, bxci, byci, bzci)| {
            let xyzobs_i = (xci, yci, zci);
            let bxyz_out_i = (bxci, byci, bzci);
            flux_density_circular_filament_cartesian(rzifil, xyzobs_i, bxyz_out_i)
        })?;

    Ok(())
}

/// Off-axis A_phi component for a circular current filament in vacuum.
/// This variant of the function is parallelized over chunks of observation points.
///
/// # Arguments
///
/// * `rzifil`:  (m, m, A-turns) r-coord, z-coord, and current of each filament, length `m`
/// * `rzobs`:   (m, m) r-coord, and z-coord of observation points, length `n`
/// * `out`: (V-s/m), phi-component of magnetic vector potential at observation locations, length `n`
///
/// # Commentary
///
/// Near-exact formula (except numerically-evaluated elliptic integrals).
/// The vector potential of a loop has zero r- and z- components due to symmetry,
/// and does not vary in the phi-direction.
///
/// # References
///
///   \[1\] J. C. Simpson, J. E. Lane, C. D. Immer, R. C. Youngquist, and T. Steinrock,
///         “Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop,”
///         Jan. 01, 2001. Accessed: Sep. 06, 2022. [Online]. Available: <https://ntrs.nasa.gov/citations/20010038494>
pub fn vector_potential_circular_filament_par(
    rzifil: (&[f64], &[f64], &[f64]),
    rzobs: (&[f64], &[f64]),
    out: &mut [f64],
) -> Result<(), &'static str> {
    // Unpack
    let (rprime, zprime) = rzobs;

    // Chunk inputs
    let n = chunksize(rprime.len());
    let rprimec = rprime.par_chunks(n);
    let zprimec = zprime.par_chunks(n);
    let outc = out.par_chunks_mut(n);

    // Run calcs
    (outc, rprimec, zprimec)
        .into_par_iter()
        .try_for_each(|(outc, rc, zc)| {
            vector_potential_circular_filament(rzifil, (rc, zc), outc)
        })?;

    Ok(())
}

/// Off-axis A_phi component for a circular current filament in vacuum.
///
/// # Arguments
///
/// * `rzifil`:  (m, m, A-turns) r-coord, z-coord, and current of each filament, length `m`
/// * `rzobs`:   (m, m) r-coord, and z-coord of observation points, length `n`
/// * `out`: (V-s/m), phi-component of magnetic vector potential at observation locations, length `n`
///
/// # Commentary
///
/// Near-exact formula (except numerically-evaluated elliptic integrals).
/// The vector potential of a loop has zero r- and z- components due to symmetry,
/// and does not vary in the phi-direction.
///
/// # References
///
///   \[1\] J. C. Simpson, J. E. Lane, C. D. Immer, R. C. Youngquist, and T. Steinrock,
///         “Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop,”
///         Jan. 01, 2001. Accessed: Sep. 06, 2022. [Online]. Available: <https://ntrs.nasa.gov/citations/20010038494>
pub fn vector_potential_circular_filament(
    rzifil: (&[f64], &[f64], &[f64]),
    rzobs: (&[f64], &[f64]),
    out: &mut [f64],
) -> Result<(), &'static str> {
    // Unpack
    let (rfil, zfil, ifil) = rzifil;
    let (rprime, zprime) = rzobs;

    // Check lengths
    let n = ifil.len();
    check_length_3tup!(n, &rzifil);
    let m = rprime.len();
    check_length!(m, rprime, zprime, out);

    // Zero output
    out.fill(0.0);

    for i in 0..n {
        for j in 0..m {
            // The inner function is inlined, so values that are reused between iterations
            // can be pulled to the outer scope by the compiler and do not affect performance
            out[j] += vector_potential_circular_filament_scalar(
                (rfil[i], zfil[i], ifil[i]),
                (rprime[j], zprime[j]),
            );
        }
    }

    Ok(())
}

/// Off-axis A_phi component for a circular current filament in vacuum.
///
/// # Arguments
///
/// * `rzifil`:  (m, m, A-turns) r-coord, z-coord, and current of filament, length `m`
/// * `rzobs`:   (m, m) r-coord, and z-coord of observation point, length `n`
///
/// # Returns
/// * `a_phi`: (V-s/m), phi-component of magnetic vector potential at observation location
///
/// # Commentary
///
/// Near-exact formula (except numerically-evaluated elliptic integrals).
/// The vector potential of a loop has zero r- and z- components due to symmetry,
/// and does not vary in the phi-direction.
///
/// # References
///
///   \[1\] J. C. Simpson, J. E. Lane, C. D. Immer, R. C. Youngquist, and T. Steinrock,
///         “Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop,”
///         Jan. 01, 2001. Accessed: Sep. 06, 2022. [Online]. Available: <https://ntrs.nasa.gov/citations/20010038494>
#[inline]
pub fn vector_potential_circular_filament_scalar(
    rzifil: (f64, f64, f64),
    rzobs: (f64, f64),
) -> f64 {
    // Unpack
    let (rfil, zfil, ifil) = rzifil;
    let (rprime, zprime) = rzobs;

    // Eq. 1 and 2 of Simpson2001 give a formula for the vector potential of a loop in spherical coordinates.
    // Here, we use that formula adjusted to cylindrical coordinates.
    // r_spherical*sin(theta) = r_cylindrical
    // r_spherical^2 = r_cylindrical^2 + z^2
    let z = zprime - zfil; // [m]

    // Assemble argument to elliptic integrals
    let rpr2 = (rfil + rprime).powf(2.0);
    let denom = z.mul_add(z, rpr2);
    let numer = 4.0 * rfil * rprime;
    let k2 = numer / denom;

    // Elliptic integral terms
    let c0 = ((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2)) / k2;

    // Factor multiplied into elliptic integral terms
    let c1 = MU0_OVER_4PI * ifil * 4.0 * rfil / denom.sqrt();

    // [V-s/m] phi-component of vector potential
    c0 * c1 // Other components are zero
}

/// Mutual inductance between a circular filament and a linear filament.
/// This method is much faster (~100x typically) than discretizing the circular loop
/// into linear segments and using Neumann's formula.
///
/// This formula is accurate only if the magnetic vector potential due to
/// the circular filament varies negligibly over the length of the linear filament.
/// The linear filament should be much shorter than the radius of the circular filament.
///
/// Discussion of the equivalence of the line integral of vector potential and the flux through a
/// surface (the mutual inductance) can be found in \[1\] eqn. 7.52 .
///
/// # References
///
/// \[1\] E. M. Purcell and D. J. Morin, “Electricity and Magnetism,”
///     Higher Education from Cambridge University Press. Accessed: Feb. 10, 2025. [Online].
///     Available: https://www.cambridge.org/highereducation/books/electricity-and-magnetism/0F97BB6C5D3A56F19B9835EDBEAB087C
///
/// \[2\] “Magnetic vector potential,” Wikipedia. Nov. 26, 2024. Accessed: Feb. 10, 2025. [Online].
///     Available: https://en.wikipedia.org/w/index.php?title=Magnetic_vector_potential&oldid=1259654939#Magnetic_vector_potential
///
/// # Arguments
///
/// * `rznfil`:    (m, m, nondim) r,z-coord and number of turns of circular filament
/// * `xyzfil0`:   (m) (x, y, z) coordinates of start of linear segment
/// * `xyzfil1`:   (m) (x, y, z) coordinates of end of linear segment
///
/// # Returns
///
/// * `m`: (H), mutual inductance
#[inline]
pub fn mutual_inductance_circular_to_linear_scalar(
    rznfil: (f64, f64, f64),
    xyzfil0: (f64, f64, f64),
    xyzfil1: (f64, f64, f64),
) -> f64 {
    // First, get the filament vector
    let dlxfil = xyzfil1.0 - xyzfil0.0; // [m]
    let dlyfil = xyzfil1.1 - xyzfil0.1;
    let dlzfil = xyzfil1.2 - xyzfil0.2;
    // Next, we need to map the linear filament into cylindrical coordinates
    //    r = (x^2 + y^2)^0.5 in cylindrical
    let path_r = rss3(xyzfil0.0, xyzfil0.1, 0.0); // [m]
    let path_dr = rss3(dlxfil, dlyfil, 0.0); // [m]

    //    phi = tan^-1(y/x)
    let path_phi0 = libm::atan2(xyzfil0.1, xyzfil0.0);
    let path_phi1 = libm::atan2(xyzfil1.1, xyzfil1.0);

    //    midpoint is best for capturing curvature in piecewise-linear paths properly
    let path_r_mid = path_r + path_dr / 2.0; // [m]
    let path_z_mid = xyzfil0.2 + dlzfil / 2.0; // [m]
    let path_phi_mid = (path_phi0 + path_phi1) / 2.0;

    // Get cylindrical vector potential at linear segment midpoint
    // for a unit current (1.0A * number of turns), which is equivalent
    // to mutual inductance per unit length of the target filament
    // [H/m]
    let a_phi_per_A = vector_potential_circular_filament_scalar(rznfil, (path_r_mid, path_z_mid));

    // Convert cylindrical vector potential to cartesian
    // Note that the conversion of a _point_ in cylindrical to cartesian
    // is different from the conversion of a _vector_ in cylindrical to cartesian.
    let a_x_per_A = -a_phi_per_A * libm::sin(path_phi_mid);
    let a_y_per_A = a_phi_per_A * libm::cos(path_phi_mid);
    let a_z_per_A = 0.0;

    // Recover mutual inductance as dot(A, dL)/I

    dot3(a_x_per_A, a_y_per_A, a_z_per_A, dlxfil, dlyfil, dlzfil)
}

/// Mutual inductance between a collection of circular filaments and a piecewise-linear filament.
/// This method is much faster (~100x typically) than discretizing the circular loop
/// into linear segments and using Neumann's formula.
/// Assumes all target filaments are connected electrically in series.
///
/// Discussion of the equivalence of the line integral of vector potential and the flux through a
/// surface (the mutual inductance) can be found in \[1\] eqn. 7.52 .
///
/// # References
///
/// \[1\] E. M. Purcell and D. J. Morin, “Electricity and Magnetism,”
///     Higher Education from Cambridge University Press. Accessed: Feb. 10, 2025. [Online].
///     Available: https://www.cambridge.org/highereducation/books/electricity-and-magnetism/0F97BB6C5D3A56F19B9835EDBEAB087C
///
/// \[2\] “Magnetic vector potential,” Wikipedia. Nov. 26, 2024. Accessed: Feb. 10, 2025. [Online].
///     Available: https://en.wikipedia.org/w/index.php?title=Magnetic_vector_potential&oldid=1259654939#Magnetic_vector_potential
///
/// # Arguments
///
/// * `rznfil`:  (m, m, nondim) r,z-coord and number of turns of each circular filament, length `n`
/// * `xyzfil`:   (m) Filament origin coords (start of segment), each length `m`
/// * `dlxyzfil`: (m) Filament segment length deltas, each length `m`
///
/// # Returns
///
/// * `mutual_inductance`: (H), mutual inductance
pub fn mutual_inductance_circular_to_linear(
    rznfil: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
) -> Result<f64, &'static str> {
    // Check lengths; Error if they do not match
    let m = xyzfil.0.len();
    check_length_3tup!(m, &xyzfil);
    check_length_3tup!(m, &dlxyzfil);
    if m < 2 {
        // Need at least 2 points to form a piecewise linear path
        return Err("Input length mismatch");
    }

    // Check lengths; Error if they do not match
    let n = rznfil.0.len();
    check_length_3tup!(n, &rznfil);

    let mut mutual_inductance = 0.0; // [H]

    for i in 0..m {
        for j in 0..n {
            // The inner function is inlined, so values that are reused between iterations
            // can be pulled to the outer scope by the compiler and do not affect performance
            let xyzfil0 = (xyzfil.0[i], xyzfil.1[i], xyzfil.2[i]);
            let xyzfil1 = (
                xyzfil.0[i] + dlxyzfil.0[i],
                xyzfil.1[i] + dlxyzfil.1[i],
                xyzfil.2[i] + dlxyzfil.2[i],
            );
            mutual_inductance += mutual_inductance_circular_to_linear_scalar(
                (rznfil.0[j], rznfil.1[j], rznfil.2[j]),
                xyzfil0,
                xyzfil1,
            );
        }
    }

    Ok(mutual_inductance)
}

/// Mutual inductance between a collection of circular filaments and a piecewise-linear filament.
/// This method is much faster (~100x typically) than discretizing the circular loop
/// into linear segments and using Neumann's formula.
/// Assumes all target filaments are connected electrically in series.
///
/// Discussion of the equivalence of the line integral of vector potential and the flux through a
/// surface (the mutual inductance) can be found in \[1\] eqn. 7.52 .
///
/// # References
///
/// \[1\] E. M. Purcell and D. J. Morin, “Electricity and Magnetism,”
///     Higher Education from Cambridge University Press. Accessed: Feb. 10, 2025. [Online].
///     Available: https://www.cambridge.org/highereducation/books/electricity-and-magnetism/0F97BB6C5D3A56F19B9835EDBEAB087C
///
/// \[2\] “Magnetic vector potential,” Wikipedia. Nov. 26, 2024. Accessed: Feb. 10, 2025. [Online].
///     Available: https://en.wikipedia.org/w/index.php?title=Magnetic_vector_potential&oldid=1259654939#Magnetic_vector_potential
///
/// # Arguments
///
/// * `rznfil`:  (m, m, nondim) r,z-coord and number of turns of each circular filament, length `n`
/// * `xyzfil`:   (m) Filament origin coords (start of segment), each length `m`
/// * `dlxyzfil`: (m) Filament segment length deltas, each length `m`
///
/// # Returns
///
/// * `m`: (V-s/m), phi-component of magnetic vector potential at observation locations
pub fn mutual_inductance_circular_to_linear_par(
    rznfil: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
) -> Result<f64, &'static str> {
    // Chunk inputs
    let n = chunksize(rznfil.0.len());
    let (rfilc, zfilc, nfilc) = par_chunks_3tup!(rznfil, n);

    // Run calcs
    // We have to sum over contributions that are each individually fallible,
    // which results in a bit of clutter with the fold-reduce pattern
    let mutual_inductance = (nfilc, rfilc, zfilc)
        .into_par_iter()
        .try_fold(
            || 0.0,
            |acc, (nc, rc, zc)| {
                let m_contrib =
                    mutual_inductance_circular_to_linear((rc, zc, nc), xyzfil, dlxyzfil)?;
                Ok::<f64, &'static str>(acc + m_contrib)
            },
        )
        .try_reduce(|| 0.0, |acc, v| Ok(acc + v))?;

    Ok(mutual_inductance)
}

/// JxB (Lorentz) body force density (per volume) in cartesian form due to a circular current
/// filament segment at an observation point in cartesian form with some current density (per area).
///
/// # Arguments
///
/// * `rzifil`:    (m, m, A-turns) r-coord, z-coord, and current of filament
/// * `xyzobs`:    (m) Observation point coords
/// * `jobs`:      (A/m^2) Current density vector at observation point
///
/// # Returns
///
/// * `jxb`:        (N/m^3) Body force density in cartesian form
pub fn body_force_density_circular_filament_cartesian_scalar(
    rzifil: (f64, f64, f64),
    xyzobs: (f64, f64, f64),
    jobs: (f64, f64, f64),
) -> (f64, f64, f64) {
    // Get flux density in cartesian coordinates
    let (bx, by, bz) = flux_density_circular_filament_cartesian_scalar(rzifil, xyzobs);

    // Take JxB Lorentz force
    cross3(jobs.0, jobs.1, jobs.2, bx, by, bz) // [N/m^3]
}

/// JxB (Lorentz) body force density (per volume) in cartesian form due to a circular current
/// filament segment at an observation point in cartesian form with some current density (per area).
///
/// # Arguments
///
/// * `rzifil`:    (m, m, A-turns) r-coord, z-coord, and current of filament, each length `m`
/// * `xyzobs`:    (m) Observation point coords, each length `n`
/// * `jobs`:      (A/m^2) Current density vector at observation point, each length `n`
///
/// # Returns
///
/// * `jxb`:        (N/m^3) Body force density in cartesian form
pub fn body_force_density_circular_filament_cartesian(
    rzifil: (&[f64], &[f64], &[f64]),
    xyzobs: (&[f64], &[f64], &[f64]),
    jobs: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    // Unpack
    let (rfil, zfil, ifil) = rzifil;
    let (x, y, z) = xyzobs;
    let (jx, jy, jz) = jobs;
    let (outx, outy, outz) = out;

    // Check lengths
    let n = ifil.len();
    check_length!(n, rfil, zfil);
    let m = x.len();
    check_length!(m, x, y, z, jx, jy, jz, outx, outy, outz);

    // Zero output
    outx.fill(0.0);
    outy.fill(0.0);
    outz.fill(0.0);

    // Do calcs
    for j in 0..m {
        for i in 0..n {
            // The inner function is inlined, so values that are reused between iterations
            // can be pulled to the outer scope by the compiler and do not affect performance
            let rzifil_i = (rfil[i], zfil[i], ifil[i]);
            let xyzobs_j = (x[j], y[j], z[j]);
            let jj = (jx[j], jy[j], jz[j]);
            let (jxbx, jxby, jxbz) =
                body_force_density_circular_filament_cartesian_scalar(rzifil_i, xyzobs_j, jj);
            outx[j] += jxbx;
            outy[j] += jxby;
            outz[j] += jxbz;
        }
    }

    Ok(())
}

/// JxB (Lorentz) body force density (per volume) in cartesian form due to a circular current
/// filament segment at an observation point in cartesian form with some current density (per area).
/// This variant is parallelized over chunks of observation points.
///
/// # Arguments
///
/// * `rzifil`:    (m, m, A-turns) r-coord, z-coord, and current of filament, each length `m`
/// * `xyzobs`:    (m) Observation point coords, each length `n`
/// * `jobs`:      (A/m^2) Current density vector at observation point, each length `n`
/// * `out`:        (N/m^3) Body force density in cartesian form, each length `n`
pub fn body_force_density_circular_filament_cartesian_par(
    rzifil: (&[f64], &[f64], &[f64]),
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
            body_force_density_circular_filament_cartesian(
                rzifil,
                (xp, yp, zp),
                (jx, jy, jz),
                (outx, outy, outz),
            )
        })?;

    Ok(())
}

#[cfg(test)]
mod test {
    use core::f64::consts::PI;

    use super::*;
    use crate::{physics::linear_filament::body_force_density_linear_filament, testing::*};

    /// Make sure that force between a circular filament and a piecewise linear filament
    /// is equal and opposite
    #[test]
    fn test_body_force_density() {
        // Because the force from the circular filament to the lienar filament is calculated
        // with closed-form circular filament B-field while the force from the linear filament
        // to the circular filament is calculated by discretizing the circular filament,
        // tightening tolerances requires excessive discretization of the circular loop,
        // and some deviation is expected.
        let (rtol, atol) = (5e-2, 1e-9);

        // Make some circular filaments
        let (rfil, zfil, nfil) = example_circular_filaments();
        // Use number-of-turns as the filament current
        // so that the result is in per-amp units
        let rzifil = (&rfil[..], &zfil[..], &nfil[..]);

        // Make a slightly tilted helical piecewise-linear filament
        let xyzfil1 = example_helix();
        let n = xyzfil1.0.len();
        let xyzobs = (
            &xyzfil1.0[..n - 1],
            &xyzfil1.1[..n - 1],
            &xyzfil1.2[..n - 1],
        );
        let (x, y, z) = &xyzfil1;
        let dl = (&diff(x)[..], &diff(y)[..], &diff(z)[..]);

        // We need a current density vector for testing that is aligned with the direction
        // of the filament. A natural choice is to use the filament direction vector dL
        // directly, capitalizing on the conversion between the biot-savart volume integral
        // over cross(J, r)dV and the line integral over cross(I*dL, r) and using unit
        // volume and area.
        let j_vec = dl;

        // Calculate force from circular filaments to helix,
        // using filament direction vector as the current density vector
        // to represent unit current on the linear filaments
        let (outx, outy, outz) = (
            &mut x.clone()[..n - 1],
            &mut x.clone()[..n - 1],
            &mut x.clone()[..n - 1],
        );
        body_force_density_circular_filament_cartesian(rzifil, xyzobs, j_vec, (outx, outy, outz))
            .unwrap();
        let out_sum: (f64, f64, f64) = (outx.iter().sum(), outy.iter().sum(), outz.iter().sum());

        // Calculate force from helix to circular filaments
        // by discretizing circular filaments
        let mut out2_sum = (0.0, 0.0, 0.0);
        let ndiscr = 100;
        let dl1 = (&diff(x)[..], &diff(y)[..], &diff(z)[..]);
        for i in 0..rfil.len() {
            let (xi, yi, zi) = discretize_circular_filament(rfil[i], zfil[i], ndiscr);
            let (outxi, outyi, outzi) = (
                &mut xi.clone()[..ndiscr - 1],
                &mut yi.clone()[..ndiscr - 1],
                &mut zi.clone()[..ndiscr - 1],
            );
            let dl2 = (&diff(&xi)[..], &diff(&yi)[..], &diff(&zi)[..]);
            // Using the target filament direction as the current density vector again for convenience,
            let j2 = dl2;

            // Each filament is the same length, so we can broadcast one current value here.
            // For second-order accuracy, target filament midpoints are used.
            body_force_density_linear_filament(
                (
                    &xyzfil1.0[..n - 1],
                    &xyzfil1.1[..n - 1],
                    &xyzfil1.2[..n - 1],
                ),
                dl1,
                &vec![1.0; x.len()][..],
                (&midpoints(&xi), &midpoints(&yi), &midpoints(&zi)),
                j2,
                (outxi, outyi, outzi),
            )
            .unwrap();

            out2_sum.0 += nfil[i] * outxi.iter().sum::<f64>();
            out2_sum.1 += nfil[i] * outyi.iter().sum::<f64>();
            out2_sum.2 += nfil[i] * outzi.iter().sum::<f64>();
        }

        // Equal and opposite reaction
        assert!(approx(out_sum.0, -out2_sum.0, rtol, atol));
        assert!(approx(out_sum.1, -out2_sum.1, rtol, atol));
        assert!(approx(out_sum.2, -out2_sum.2, rtol, atol));
    }

    /// Make sure that the cylindrical-to-cartesian conversion produces the
    /// same result achieved by discretizing the circular filament into linear
    /// segments, and that the serial and parallel variants produce the same result
    #[test]
    fn test_flux_density_circular_filament_cartesian() {
        // It takes a massive amount of discretization to achieve
        // better relative tolerance in the linear discretized calc,
        // and that discretization ultimately causes the accumulated
        // float roundoff error from the extra addition operations
        // to outcompete the improvement from increasing geometric
        // fidelity.
        let rtol = 2e-2;
        let atol = 1e-10;

        // Make some circular filaments
        let (rfil, zfil, nfil) = example_circular_filaments();
        // Use number-of-turns as the filament current
        // so that the result is in per-amp units
        let rzifil = (&rfil[..], &zfil[..], &nfil[..]);

        // Make a slightly tilted helical piecewise-linear filament
        let xyzfil1 = example_helix();
        let xyzobs = (&xyzfil1.0[..], &xyzfil1.1[..], &xyzfil1.2[..]);
        let x = &xyzfil1.0;

        // Do calcs
        let (bx0, by0, bz0) = (&mut x.clone()[..], &mut x.clone()[..], &mut x.clone()[..]);
        flux_density_circular_filament_cartesian(rzifil, xyzobs, (bx0, by0, bz0)).unwrap();

        let (bx1, by1, bz1) = (&mut x.clone()[..], &mut x.clone()[..], &mut x.clone()[..]);
        flux_density_circular_filament_cartesian_par(rzifil, xyzobs, (bx1, by1, bz1)).unwrap();

        let (bx2, by2, bz2) = (&mut x.clone()[..], &mut x.clone()[..], &mut x.clone()[..]);
        bx2.fill(0.0);
        by2.fill(0.0);
        bz2.fill(0.0);
        for i in 0..rfil.len() {
            // Set up inputs
            let (r, z, nturns) = (rfil[i], zfil[i], nfil[i]);
            let ndiscr = 400;
            let xyzfil0 = discretize_circular_filament(r, z, ndiscr);
            let xyzfil0 = (&xyzfil0.0[..], &xyzfil0.1[..], &xyzfil0.2[..]);
            let dlxyzfil = (
                &diff(xyzfil0.0)[..],
                &diff(xyzfil0.1)[..],
                &diff(xyzfil0.2)[..],
            );

            let ifil = vec![nturns; ndiscr - 1];
            let (xcontrib, ycontrib, zcontrib) =
                (&mut x.clone()[..], &mut x.clone()[..], &mut x.clone()[..]);

            // Do calc
            crate::physics::linear_filament::flux_density_linear_filament_par(
                xyzobs,
                (
                    &xyzfil0.0[..ndiscr - 1],
                    &xyzfil0.1[..ndiscr - 1],
                    &xyzfil0.2[..ndiscr - 1],
                ),
                dlxyzfil,
                &ifil[..],
                (xcontrib, ycontrib, zcontrib),
            )
            .unwrap();

            // Sum contributions
            for j in 0..x.len() {
                bx2[j] += xcontrib[j];
                by2[j] += ycontrib[j];
                bz2[j] += zcontrib[j];
            }
        }

        // Compare
        for j in 0..x.len() {
            assert!(approx(bx0[j], bx1[j], 1e-12, 1e-12)); // Serial vs parallel
            assert!(approx(bx0[j], bx2[j], rtol, atol)); // Serial vs discretized

            assert!(approx(by0[j], by1[j], 1e-12, 1e-12)); // Serial vs parallel
            assert!(approx(by0[j], by2[j], rtol, atol)); // Serial vs discretized

            assert!(approx(bz0[j], bz1[j], 1e-12, 1e-12)); // Serial vs parallel
            assert!(approx(bz0[j], bz2[j], rtol, atol)); // Serial vs discretized
        }
    }

    /// Make sure the circular-to-linear mutual inductance calc matches
    /// the result achieved by discretizing the circular filament
    /// into linear segments, and matches between serial and parallel variants
    #[test]
    fn test_mutual_inductance_to_linear() {
        // It takes excessive discretization to achieve improved tolerance
        // in the linear filament equivalent calc
        let rtol = 1e-2;
        let atol = 1e-12;

        // Make some circular filaments
        let (rfil, zfil, nfil) = example_circular_filaments();

        // Make a slightly tilted helical piecewise-linear filament
        let xyzfil1 = example_helix();
        let (x, y, z) = (&xyzfil1.0[..], &xyzfil1.1[..], &xyzfil1.2[..]);
        let dlxyzfil1 = (&diff(x)[..], &diff(y)[..], &diff(z)[..]);
        let n = x.len();

        // Get mutual inductance by purpose-made calc
        // [H]
        let mutual_inductance = mutual_inductance_circular_to_linear(
            (&rfil, &zfil, &nfil),
            (&x[..n - 1], &y[..n - 1], &z[..n - 1]),
            dlxyzfil1,
        )
        .unwrap();
        let mutual_inductance_par = mutual_inductance_circular_to_linear_par(
            (&rfil, &zfil, &nfil),
            (&x[..n - 1], &y[..n - 1], &z[..n - 1]),
            dlxyzfil1,
        )
        .unwrap();

        // Get mutual inductance by brute-force calc
        let mut mutual_inductance_2 = 0.0;
        for i in 0..rfil.len() {
            let ndiscr = 100;
            let (xfil, yfil, zfil) = discretize_circular_filament(rfil[i], zfil[i], ndiscr);
            let dlxfil0 = diff(&xfil);
            let dlyfil0 = diff(&yfil);
            let dlzfil0 = diff(&zfil);
            let dlxyzfil0 = (&dlxfil0[..], &dlyfil0[..], &dlzfil0[..]);
            mutual_inductance_2 += nfil[i]
                * crate::physics::linear_filament::inductance_piecewise_linear_filaments(
                    (
                        &xfil[0..ndiscr - 1],
                        &yfil[0..ndiscr - 1],
                        &zfil[0..ndiscr - 1],
                    ),
                    dlxyzfil0,
                    (&x[0..n - 1], &y[0..n - 1], &z[0..n - 1]),
                    dlxyzfil1,
                    false,
                )
                .unwrap();
        }

        // Parallel and serial should match exactly, although changing the sum order
        // produce slight differences due to float roundoff
        assert!(approx(
            mutual_inductance,
            mutual_inductance_par,
            1e-10,
            1e-12
        ));
        // The brute force discretization calc takes an excessive
        // amount of discretization to reach accuracy <1e-3, but converges rapidly to
        // about 1e-2 relative accuracy
        assert!(approx(mutual_inductance_2, mutual_inductance, rtol, atol));
    }

    /// Check that B = curl(A)
    /// and that psi = integral(dot(A, dL)) =  2pi * r * a
    #[test]
    fn test_vector_potential() {
        let rfil = 1.0 / core::f64::consts::PI; // [m] some number
        let zfil = 1.0 / core::f64::consts::E; // [m] some number

        let vp = |r: f64, z: f64| {
            let mut out = [0.0];

            vector_potential_circular_filament((&[rfil], &[zfil], &[1.0]), (&[r], &[z]), &mut out)
                .unwrap();

            out[0]
        };

        let zvals = [0.25, 0.5, 2.5, 10.0, 0.0, -10.0, -2.5, -0.5, -0.25];
        let rvals = [0.25, 0.5, 2.5, 10.0];
        // finite diff delta needs to be small enough to be accurate
        // but large enough that we can tell the difference between adjacent points
        // that are very far from the origin
        let eps = 1e-7;
        for r in rvals.iter() {
            for z in zvals.iter() {
                // Finite-difference curl of the vector potential in cylindrical coordinates.
                // The radial and z components of the vector potential are zero.
                let mut ca = [0.0; 3];
                // curl(A)[0] = - d(A_phi) / dz
                let a0 = vp(*r, *z - eps);
                let a1 = vp(*r, *z + eps);
                ca[0] = -(a1 - a0) / (2.0 * eps);
                // curl(A)[2] = (1 / rho ) d(rho A_phi) / d(rho)
                let ra0 = (*r - eps) * vp(*r - eps, *z);
                let ra1 = (*r + eps) * vp(*r + eps, *z);
                ca[2] = (ra1 - ra0) / (2.0 * eps) / *r;

                // B via biot-savart
                let mut br = [0.0];
                let mut bz = [0.0];
                flux_density_circular_filament(
                    (&[rfil], &[zfil], &[1.0]),
                    (&[*r], &[*z]),
                    (&mut br, &mut bz),
                )
                .unwrap();

                assert!(approx(br[0], ca[0], 1e-7, 1e-13));
                assert!(approx(bz[0], ca[2], 1e-7, 1e-13));

                // Flux via analytic formula
                // psi = integral(dot(A, dL)) =  2pi * r * a
                let psi_from_a = 2.0 * PI * *r * vp(*r, *z);
                let mut psi = [0.0];
                flux_circular_filament((&[rfil], &[zfil], &[1.0]), (&[*r], &[*z]), &mut psi)
                    .unwrap();
                println!("{psi:?}, {psi_from_a}");
                assert!(approx(psi_from_a, psi[0], 1e-10, 0.0)); // Should be very close to float roundoff
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
        let rfil: Vec<f64> = (0..NFIL).map(|i| (i as f64).sin() + 1.2).collect();
        let zfil: Vec<f64> = (0..NFIL)
            .map(|i| (i as f64) - (NFIL as f64) / 2.0)
            .collect();
        let ifil: Vec<f64> = (0..NFIL).map(|i| (i as f64)).collect();

        // Build a scattering of observation locations
        let rprime: Vec<f64> = (0..NOBS).map(|i| 2.0 * (i as f64).sin() + 2.1).collect();
        let zprime: Vec<f64> = (0..NOBS).map(|i| 4.0 * (2.0 * i as f64).cos()).collect();

        // Some output storage
        // Initialize with different values for each buffer to test zeroing
        let out0 = &mut [0.0; NOBS];
        let out1 = &mut [1.0; NOBS];
        let out2 = &mut [2.0; NOBS];
        let out3 = &mut [3.0; NOBS];

        // Flux
        flux_circular_filament((&rfil, &zfil, &ifil), (&rprime, &zprime), out0).unwrap();
        flux_circular_filament_par((&rfil, &zfil, &ifil), (&rprime, &zprime), out1).unwrap();
        for i in 0..NOBS {
            assert_eq!(out0[i], out1[i]);
        }

        // Flux density
        flux_density_circular_filament((&rfil, &zfil, &ifil), (&rprime, &zprime), (out0, out1))
            .unwrap();
        flux_density_circular_filament_par((&rfil, &zfil, &ifil), (&rprime, &zprime), (out2, out3))
            .unwrap();
        for i in 0..NOBS {
            assert_eq!(out0[i], out2[i]);
            assert_eq!(out1[i], out3[i]);
        }

        // Vector potential
        let out0 = &mut [0.0; NOBS]; // Reinit with different values to test zeroing
        let out1 = &mut [1.0; NOBS];
        vector_potential_circular_filament((&rfil, &zfil, &ifil), (&rprime, &zprime), out0)
            .unwrap();
        vector_potential_circular_filament_par((&rfil, &zfil, &ifil), (&rprime, &zprime), out1)
            .unwrap();
        for i in 0..NOBS {
            assert_eq!(out0[i], out1[i]);
        }
    }
}
