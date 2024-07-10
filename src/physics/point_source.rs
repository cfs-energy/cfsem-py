//! Calculations for 0D field sources such as dipoles.

use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

use crate::{
    MU0_OVER_4PI, chunksize,
    macros::{check_length_3tup, mut_par_chunks_3tup, par_chunks_3tup},
    math::{cross3, dot3, rss3, switch_float},
    physics::volumetric::{
        flux_density_inside_magnetized_sphere, vector_potential_inside_magnetized_sphere,
    },
};

use libm::pow;

/// Magnetic flux density of a dipole in cartesian coordinates.
///
/// Arguments
///
/// * loc: (m) location of the point source
/// * moment: (A-m^2) magnetic moment vector of the point source
/// * obs: (m) observation point to examine
/// * outer_radius: (m) radius inside which to defer to magnetized sphere calc
///
/// Returns
///
/// * (bx, by, bz) [T] magnetic field components at observation point
#[inline]
pub fn flux_density_dipole_scalar(
    loc: (f64, f64, f64),
    moment: (f64, f64, f64),
    outer_radius: f64,
    obs: (f64, f64, f64),
) -> (f64, f64, f64) {
    // Radius vector decomposed into direction and magnitude
    let r = (obs.0 - loc.0, obs.1 - loc.1, obs.2 - loc.2); // [m]
    let rmag = rss3(r.0, r.1, r.2); // [m]
    let rhat = (r.0 / rmag, r.1 / rmag, r.2 / rmag); // [dimensionless]
    let rinv3 = pow(rmag, -3.0);

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

    // Defer to magnetized sphere if necessary
    let inside = rmag < outer_radius;
    let (bx_inside, by_inside, bz_inside) =
        flux_density_inside_magnetized_sphere(moment, outer_radius);
    let bx = switch_float(bx_inside, bx, inside);
    let by = switch_float(by_inside, by, inside);
    let bz = switch_float(bz_inside, bz, inside);

    (bx, by, bz) // [T]
}

/// Magnetic flux density of a dipole in cartesian coordinates.
/// For more details, see [flux_density_dipole_scalar].
#[inline]
pub fn flux_density_dipole(
    loc: (&[f64], &[f64], &[f64]),
    moment: (&[f64], &[f64], &[f64]),
    outer_radius: &[f64],
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
            let roj = outer_radius[j];
            let (bx, by, bz) = flux_density_dipole_scalar(locj, momentj, roj, obsi);
            out.0[i] += bx;
            out.1[i] += by;
            out.2[i] += bz;
        }
    }

    Ok(())
}

/// Magnetic flux density of a dipole in cartesian coordinates.
/// Parallelized over chunks of observation points and vectorized over source points.
/// For more details, see [flux_density_dipole_scalar].
#[inline]
pub fn flux_density_dipole_par(
    loc: (&[f64], &[f64], &[f64]),
    moment: (&[f64], &[f64], &[f64]),
    outer_radius: &[f64],
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
            flux_density_dipole(
                loc,
                moment,
                outer_radius,
                (obsx, obsy, obsz),
                (outx, outy, outz),
            )
        })?;

    Ok(())
}

/// Vector potential of a dipole in cartesian coordinates.
///
/// Arguments
///
/// * loc: (m) location of the point source
/// * moment: (A-m^2) magnetic moment vector of the point source
/// * obs: (m) observation point to examine
/// * outer_radius: (m) radius inside which to defer to magnetized sphere calc
///
/// Returns
///
/// * (ax, ay, az) [V⋅s⋅m-1] vector potential components at observation point
#[inline]
pub fn vector_potential_dipole_scalar(
    loc: (f64, f64, f64),
    moment: (f64, f64, f64),
    outer_radius: f64,
    obs: (f64, f64, f64),
) -> (f64, f64, f64) {
    // Radius and moment vectors decomposed into direction and magnitude
    let r = (obs.0 - loc.0, obs.1 - loc.1, obs.2 - loc.2); // [m]
    let rmag = rss3(r.0, r.1, r.2); // [m]
    let rhat = (r.0 / rmag, r.1 / rmag, r.2 / rmag); // [dimensionless]
    let m = moment;
    let mmag = rss3(m.0, m.1, m.2);
    let mhat = (m.0 / mmag, m.1 / mmag, m.2 / mmag);
    let rinv2 = pow(rmag, -2.0);

    // mhat x rhat
    // Use normalized vectors for cross product to improve float roundoff
    let mhat_cross_rhat = cross3(mhat.0, mhat.1, mhat.2, rhat.0, rhat.1, rhat.2);

    // Assemble components
    let c = MU0_OVER_4PI * mmag * rinv2; // Shared factor
    let tsum = (
        mhat_cross_rhat.0 * c,
        mhat_cross_rhat.1 * c,
        mhat_cross_rhat.2 * c,
    );
    let (ax, ay, az) = (tsum.0, tsum.1, tsum.2);

    // Defer to magnetized sphere if necessary
    let inside = rmag < outer_radius;
    let (ax_inside, ay_inside, az_inside) =
        vector_potential_inside_magnetized_sphere(mhat_cross_rhat, mmag, rmag, outer_radius);
    let ax = switch_float(ax_inside, ax, inside);
    let ay = switch_float(ay_inside, ay, inside);
    let az = switch_float(az_inside, az, inside);

    (ax, ay, az) // [V⋅s⋅m-1]
}

/// Magnetic vector potential of a dipole in cartesian coordinates.
/// For more details, see [vector_potential_dipole_scalar].
#[inline]
pub fn vector_potential_dipole(
    loc: (&[f64], &[f64], &[f64]),
    moment: (&[f64], &[f64], &[f64]),
    outer_radius: &[f64],
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
            let roj = outer_radius[j];
            let (ax, ay, az) = vector_potential_dipole_scalar(locj, momentj, roj, obsi);
            out.0[i] += ax;
            out.1[i] += ay;
            out.2[i] += az;
        }
    }

    Ok(())
}

/// Magnetic flux density of a dipole in cartesian coordinates.
/// Parallelized over chunks of observation points and vectorized over source points.
/// For more details, see [vector_potential_dipole_scalar].
#[inline]
pub fn vector_potential_dipole_par(
    loc: (&[f64], &[f64], &[f64]),
    moment: (&[f64], &[f64], &[f64]),
    outer_radius: &[f64],
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
            vector_potential_dipole(
                loc,
                moment,
                outer_radius,
                (obsx, obsy, obsz),
                (outx, outy, outz),
            )
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
            &[0.0],
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
    }

    // Make sure that far from a dipole, the vector potential
    // is consistent with a very small loop placed with the same center.
    #[test]
    fn test_vector_potential() {
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
        let ygrid: Vec<f64> = vec![0.0];
        let zgrid = linspace(-1.0, 1.0, ngrid);
        let mesh = meshgrid(&[&xgrid[..], &ygrid[..], &zgrid[..]]);
        let (xmesh, ymesh, zmesh) = (&mesh[0], &mesh[1], &mesh[2]);

        // Run both circular filament and dipole calcs at each point
        let nobs = xmesh.len(); // number of observation points
        let outx_circ = &mut vec![0.0; nobs][..];
        let outy_circ = &mut vec![0.0; nobs][..];
        let outz_circ = &mut vec![0.0; nobs][..];
        crate::physics::circular_filament::vector_potential_circular_filament_par(
            (&[rfil], &[zfil], &[ifil]),
            (&xmesh[..], &zmesh[..]),
            outy_circ,
        )
        .unwrap();

        let outx_dipole = &mut vec![0.0; nobs][..];
        let outy_dipole = &mut vec![0.0; nobs][..];
        let outz_dipole = &mut vec![0.0; nobs][..];
        super::vector_potential_dipole_par(
            (&[loc.0], &[loc.1], &[loc.2]),
            (&[moment.0], &[moment.1], &[moment.2]),
            &[0.0],
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
    }

    #[test]
    fn test_vector_potential_curl_inside_outer_radius() {
        let (rtol, atol) = (1e-10, 1e-13);

        let loc = (0.0, 0.0, 0.0);
        let moment = (0.0, 0.0, 0.25);
        let outer_radius = 0.3;

        let points = [
            (0.05, 0.02, -0.04),
            (-0.07, 0.03, 0.06),
            (0.09, -0.05, 0.01),
            (-0.04, -0.03, -0.08),
        ];

        let eps = 1e-6;

        let vp = |x: f64, y: f64, z: f64| {
            super::vector_potential_dipole_scalar(loc, moment, outer_radius, (x, y, z))
        };

        for &(x, y, z) in &points {
            let b = super::flux_density_dipole_scalar(loc, moment, outer_radius, (x, y, z));

            let mut da = [[0.0f64; 3]; 3];

            let (ax0, ay0, az0) = vp(x - eps, y, z);
            let (ax1, ay1, az1) = vp(x + eps, y, z);
            da[0][0] = (ax1 - ax0) / (2.0 * eps);
            da[0][1] = (ay1 - ay0) / (2.0 * eps);
            da[0][2] = (az1 - az0) / (2.0 * eps);

            let (ax0, ay0, az0) = vp(x, y - eps, z);
            let (ax1, ay1, az1) = vp(x, y + eps, z);
            da[1][0] = (ax1 - ax0) / (2.0 * eps);
            da[1][1] = (ay1 - ay0) / (2.0 * eps);
            da[1][2] = (az1 - az0) / (2.0 * eps);

            let (ax0, ay0, az0) = vp(x, y, z - eps);
            let (ax1, ay1, az1) = vp(x, y, z + eps);
            da[2][0] = (ax1 - ax0) / (2.0 * eps);
            da[2][1] = (ay1 - ay0) / (2.0 * eps);
            da[2][2] = (az1 - az0) / (2.0 * eps);

            let curl = (
                da[1][2] - da[2][1],
                da[2][0] - da[0][2],
                da[0][1] - da[1][0],
            );

            let bmag = (b.0.powi(2) + b.1.powi(2) + b.2.powi(2)).sqrt();
            assert!(bmag > 1e-12);

            assert!(approx(curl.0, b.0, rtol, atol));
            assert!(approx(curl.1, b.1, rtol, atol));
            assert!(approx(curl.2, b.2, rtol, atol));
        }
    }
}
