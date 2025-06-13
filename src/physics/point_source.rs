//! Calculations for 0D field sources such as dipoles.

use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

use crate::{
    MU0_OVER_4PI, chunksize,
    macros::{check_length_3tup, mut_par_chunks_3tup, par_chunks_3tup},
    math::{dot3, rss3},
};

/// Magnetic flux density of a dipole in cartesian coordiantes.
///
/// Arguments
///
/// * loc: (m) location of the point source
/// * moment: (A-m^2) magnetic moment vector of the point source
/// * obs: (m) observation point to examine
///
/// Returns
///
/// * (bx, by, bz) [T] magnetic field components at observation point
#[inline]
pub fn flux_density_dipole_scalar(
    loc: (f64, f64, f64),
    moment: (f64, f64, f64),
    obs: (f64, f64, f64),
) -> (f64, f64, f64) {
    // Radius vector decomposed into direction and magnitude
    let r = (obs.0 - loc.0, obs.1 - loc.1, obs.2 - loc.2); // [m]
    let rmag = rss3(r.0, r.1, r.2); // [m]
    let rhat = (r.0 / rmag, r.1 / rmag, r.2 / rmag); // [dimensionless]
    let rinv3 = rmag.powf(-3.0);

    // r(dot(m, r))/|r|^5 reordered to avoid computing the 5th power for improved float resolution
    let m_dot_r = dot3(moment.0, moment.1, moment.2, rhat.0, rhat.1, rhat.2);
    let rmr = (rhat.0 * m_dot_r, rhat.1 * m_dot_r, rhat.2 * m_dot_r);

    // Assemble components
    let c = 3.0 * rinv3;
    let term1 = (rmr.0 * c, rmr.1 * c, rmr.2 * c);
    let term2 = (-moment.0 * rinv3, -moment.1 * rinv3, -moment.2 * rinv3);
    let tsum = (term1.0 + term2.0, term1.1 + term2.1, term1.2 + term2.2);

    let (bx, by, bz) = (
        MU0_OVER_4PI * tsum.0,
        MU0_OVER_4PI * tsum.1,
        MU0_OVER_4PI * tsum.2,
    );

    (bx, by, bz) // [T]
}

/// Magnetic flux density of a dipole in cartesian coordiantes.
///
/// Arguments
///
/// * loc: (m) location of the point source
/// * moment: (A-m^2) magnetic moment vector of the point source
/// * obs: (m) observation point to examine
/// * out: (T) storage for B-field
///
/// Returns
///
/// * (bx, by, bz) [T] magnetic field components at observation point
#[inline]
pub fn flux_density_dipole(
    loc: (&[f64], &[f64], &[f64]),
    moment: (&[f64], &[f64], &[f64]),
    obs: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    // Check lengths
    let m = loc.0.len();
    let n = obs.0.len();

    check_length_3tup!(m, &loc);
    check_length_3tup!(m, &moment);
    check_length_3tup!(n, &obs);
    check_length_3tup!(n, &out);

    // Do calcs
    for i in 0..n {
        for j in 0..m {
            let obsi = (obs.0[i], obs.1[i], obs.2[i]);
            let locj = (loc.0[j], loc.1[j], loc.2[j]);
            let momentj = (moment.0[j], moment.1[j], moment.2[j]);
            let (bx, by, bz) = flux_density_dipole_scalar(locj, momentj, obsi);
            out.0[i] += bx;
            out.1[i] += by;
            out.2[i] += bz;
        }
    }

    Ok(())
}

/// Magnetic flux density of a dipole in cartesian coordiantes.
/// Parallelized over chunks of observation points and vectorized over source points.
///
/// Arguments
///
/// * loc: (m) location of the point source
/// * moment: (A-m^2) magnetic moment vector of the point source
/// * obs: (m) observation point to examine
/// * out: (T) storage for B-field
///
/// Returns
///
/// * (bx, by, bz) [T] magnetic field components at observation point
#[inline]
pub fn flux_density_dipole_par(
    loc: (&[f64], &[f64], &[f64]),
    moment: (&[f64], &[f64], &[f64]),
    obs: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    // Chunk inputs
    let n = chunksize(obs.0.len());
    let (obsxc, obsyc, obszc) = par_chunks_3tup!(obs, n);
    let (outxc, outyc, outzc) = mut_par_chunks_3tup!(out, n);

    // Run calcs
    (outxc, outyc, outzc, obsxc, obsyc, obszc)
        .into_par_iter()
        .try_for_each(|(outx, outy, outz, obsx, obsy, obsz)| {
            flux_density_dipole(loc, moment, (obsx, obsy, obsz), (outx, outy, outz))
        })?;

    Ok(())
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    use crate::testing::*;

    // Make sure that far from a dipole, the field
    // is consistent with a very small loop placed with the same center.
    #[test]
    fn test_flux_density() {
        // At a distance much greater than the loop radius,
        // the error between a small loop and a dipole should be
        // almost entirely due to numerics
        let (rtol, atol) = (1e-12, 1e-14);

        // Make a small filament with unit current
        let rfil = PI / 1000.0; // [m]
        let zfil = 0.07; // [m]
        let ifil = 1.0; // [A]

        // Make an equivalent dipole
        let loc = (0.0, 0.0, zfil);
        let s = PI * rfil.powf(2.0); // [m^2] poloidal area of circular filament
        let m = s * ifil; // [A-m^2], magnetic moment of circular filament
        let moment = (0.0, 0.0, m); // [A-m^2], magnetic moment of dipole

        // Make a mesh of evaluation points
        let ngrid = 6;
        let xgrid = linspace(-1.0, 1.0, ngrid);
        let ygrid = linspace(-1.0, 1.0, ngrid);
        let zgrid = linspace(-1.0, 1.0, ngrid);
        let mesh = meshgrid(&[&xgrid[..], &ygrid[..], &zgrid[..]]);
        let (xmesh, ymesh, zmesh) = (&mesh[0], &mesh[1], &mesh[2]);

        // Run both circular filament and dipole calcs at each point
        let nobs = xmesh.len(); // number of observation points
        let outx_circ = &mut vec![0.0; nobs][..];
        let outy_circ = &mut vec![0.0; nobs][..];
        let outz_circ = &mut vec![0.0; nobs][..];
        crate::physics::circular_filament::flux_density_circular_filament_cartesian_par(
            (&[rfil], &[zfil], &[ifil]),
            (&xmesh[..], &ymesh[..], &zmesh[..]),
            (outx_circ, outy_circ, outz_circ),
        )
        .unwrap();

        let outx_dipole = &mut vec![0.0; nobs][..];
        let outy_dipole = &mut vec![0.0; nobs][..];
        let outz_dipole = &mut vec![0.0; nobs][..];
        super::flux_density_dipole_par(
            (&[loc.0], &[loc.1], &[loc.2]),
            (&[moment.0], &[moment.1], &[moment.2]),
            (xmesh, ymesh, zmesh),
            (outx_dipole, outy_dipole, outz_dipole),
        )
        .unwrap();

        // Check for match between dipole and small loop
        for i in 0..nobs {
            assert!(approx(outx_circ[i], outx_dipole[i], rtol, atol));
            assert!(approx(outy_circ[i], outy_dipole[i], rtol, atol));
            assert!(approx(outz_circ[i], outz_dipole[i], rtol, atol));
        }

        println!(
            "{:e}, {:e}",
            outx_circ.iter().fold(0.0, |acc, v| v.abs().max(acc)),
            outx_dipole.iter().fold(0.0, |acc, v| v.abs().max(acc))
        );
    }
}
