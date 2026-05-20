//! Magnetics calculations for piecewise-linear current filaments
//! in point-source form.

use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

use crate::physics::point_source::current_element::{
    flux_density_current_element_scalar, vector_potential_current_element_scalar,
};
use crate::{chunksize, math::cross3, physics::hierarchical::DualTreeScalar};

use crate::macros::*;

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
/// * `out`:      (T) bx, by, bz at observation points, each length `n`
pub fn flux_density_point_segment_par(
    xyzp: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
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
            flux_density_point_segment((xp, yp, zp), xyzfil, dlxyzfil, ifil, (bx, by, bz))
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
/// * `out`:      (T) bx, by, bz at observation points, each length `n`
pub fn flux_density_point_segment(
    xyzp: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
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
    check_length!(n, xfil, yfil, zfil, dlxfil, dlyfil, dlzfil, ifil);

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
            let (bxc, byc, bzc) = flux_density_point_segment_scalar((fil0, fil1, current), obs);
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
/// Uses filament midpoint as field source.
///
/// # Arguments
///
/// * `xyzifil`:   (m, m, A) Filament start and end coords and current
/// * `xyzp`:     (m) Observation point coords
///
/// # Returns
///
/// * `b`:        (T) Magnetic flux density (B-field)
#[inline]
pub fn flux_density_point_segment_scalar<T: DualTreeScalar>(
    xyzifil: ((T, T, T), (T, T, T), T),
    xyzobs: (T, T, T),
) -> (T, T, T) {
    // Unpack
    let (xyz0, xyz1, ifil) = xyzifil;
    let (xp, yp, zp) = xyzobs;

    // Get filament midpoint and length vector
    let half = T::from_f64(0.5);
    let xmid = half.mul_add(xyz1.0 - xyz0.0, xyz0.0);
    let ymid = half.mul_add(xyz1.1 - xyz0.1, xyz0.1);
    let zmid = half.mul_add(xyz1.2 - xyz0.2, xyz0.2);
    let dl = (xyz1.0 - xyz0.0, xyz1.1 - xyz0.1, xyz1.2 - xyz0.2);
    let moment = [ifil * dl.0, ifil * dl.1, ifil * dl.2];
    let b = flux_density_current_element_scalar([xmid, ymid, zmid], moment, [xp, yp, zp]);
    (b[0], b[1], b[2])
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
/// * `out`:      (V-s/m) ax, ay, az at observation points, each length `n`
pub fn vector_potential_point_segment_par(
    xyzp: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
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
            vector_potential_point_segment((xp, yp, zp), xyzfil, dlxyzfil, ifil, (bx, by, bz))
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
/// * `out`:      (V-s/m) ax, ay, az at observation points, each length `n`
pub fn vector_potential_point_segment(
    xyzp: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
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
    check_length!(n, xfil, yfil, zfil, dlxfil, dlyfil, dlzfil, ifil);

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
            let (axc, ayc, azc) = vector_potential_point_segment_scalar((fil0, fil1, current), obs);
            ax[j] += axc;
            ay[j] += ayc;
            az[j] += azc;
        }
    }

    Ok(())
}

/// Vector potential (A-field) from a linear current
/// filament segment to an observation point.
///
/// Uses filament midpoint as field source.
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
pub fn vector_potential_point_segment_scalar<T: DualTreeScalar>(
    xyzifil: ((T, T, T), (T, T, T), T),
    xyzobs: (T, T, T),
) -> (T, T, T) {
    // Unpack
    let (xyz0, xyz1, ifil) = xyzifil;

    // Get filament midpoint and length vector
    let half = T::from_f64(0.5);
    let xmid = half.mul_add(xyz1.0 - xyz0.0, xyz0.0);
    let ymid = half.mul_add(xyz1.1 - xyz0.1, xyz0.1);
    let zmid = half.mul_add(xyz1.2 - xyz0.2, xyz0.2);
    let dl = (xyz1.0 - xyz0.0, xyz1.1 - xyz0.1, xyz1.2 - xyz0.2);
    let moment = [ifil * dl.0, ifil * dl.1, ifil * dl.2];
    let a = vector_potential_current_element_scalar(
        [xmid, ymid, zmid],
        moment,
        [xyzobs.0, xyzobs.1, xyzobs.2],
    );
    (a[0], a[1], a[2])
}

/// JxB (Lorentz) body force density (per volume) due to a linear current
/// filament segment at an observation point with some current density (per area).
///
/// Uses filament midpoint as field source.
///
/// # Arguments
///
/// * `xyzifil`:   (m, m, A) Filament start and end coords and current
/// * `xyzobs`:    (m) Observation point coords
/// * `jobs`:      (A/m^2) Current density vector at observation point
///
/// # Returns
///
/// * `jxb`:        (N/m^3) Body force density
pub fn body_force_density_point_segment_scalar(
    xyzifil: ((f64, f64, f64), (f64, f64, f64), f64),
    xyzobs: (f64, f64, f64),
    jobs: (f64, f64, f64),
) -> (f64, f64, f64) {
    // Get magnetic flux density at target point
    let (bx, by, bz) = flux_density_point_segment_scalar(xyzifil, xyzobs); // [T]

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
/// * `xyzobs`:    (m) Observation point coords
/// * `jobs`:      (A/m^2) Current density vector at observation point
/// * `out`:       (N/m^3) Body force density x, y, z components
pub fn body_force_density_point_segment(
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
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

    check_length!(n, xfil, yfil, zfil, dlxfil, dlyfil, dlzfil);
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
            let (jxbx, jxby, jxbz) =
                body_force_density_point_segment_scalar((fil0, fil1, ifil[i]), obs, jj);
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
/// * `xyzobs`:    (m) Observation point coords
/// * `jobs`:      (A/m^2) Current density vector at observation point
/// * `out`:       (N/m^3) Body force density x, y, z components
pub fn body_force_density_point_segment_par(
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
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
            body_force_density_point_segment(
                xyzfil,
                dlxyzfil,
                ifil,
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
    use crate::math::norm3;
    use crate::mesh::quadrature::{GaussLegendreRule, gauss_legendre_unit_interval_table};
    use crate::physics::linear_filament::{
        inductance_piecewise_linear_filaments, vector_potential_linear_filament,
    };
    use crate::physics::point_source::current_element::{
        flux_density_current_element_scalar, vector_potential_current_element_scalar,
    };
    use crate::testing::*;

    #[test]
    fn test_point_segment_scalars_match_current_element_midpoint_mapping() {
        let xyz0 = (-0.4, 0.2, 0.7);
        let xyz1 = (0.8, -0.3, 1.1);
        let ifil = -2.3;
        let obs = (1.4, -0.9, 0.6);

        let dl = (xyz1.0 - xyz0.0, xyz1.1 - xyz0.1, xyz1.2 - xyz0.2);
        let xmid = dl.0.mul_add(0.5, xyz0.0);
        let ymid = dl.1.mul_add(0.5, xyz0.1);
        let zmid = dl.2.mul_add(0.5, xyz0.2);
        let moment = [ifil * dl.0, ifil * dl.1, ifil * dl.2];

        let b_segment = flux_density_point_segment_scalar((xyz0, xyz1, ifil), obs);
        let a_segment = vector_potential_point_segment_scalar((xyz0, xyz1, ifil), obs);
        let b_element =
            flux_density_current_element_scalar([xmid, ymid, zmid], moment, [obs.0, obs.1, obs.2]);
        let a_element = vector_potential_current_element_scalar(
            [xmid, ymid, zmid],
            moment,
            [obs.0, obs.1, obs.2],
        );

        for axis in 0..3 {
            let b_segment_axis = [b_segment.0, b_segment.1, b_segment.2][axis];
            let a_segment_axis = [a_segment.0, a_segment.1, a_segment.2][axis];
            assert!(
                approx(b_segment_axis, b_element[axis], 0.0, 1e-15),
                "point-segment B/current-element mismatch at axis {axis}: segment={:.16e}, element={:.16e}",
                b_segment_axis,
                b_element[axis],
            );
            assert!(
                approx(a_segment_axis, a_element[axis], 0.0, 1e-15),
                "point-segment A/current-element mismatch at axis {axis}: segment={:.16e}, element={:.16e}",
                a_segment_axis,
                a_element[axis],
            );
        }
    }

    #[test]
    fn test_current_element_scalars_zero_inside_distance_tolerance() {
        let src = [0.1_f64, -0.2, 0.3];
        let moment = [0.4_f64, -0.7, 1.1];

        let obs_near = [src[0] + 0.5e-14, src[1], src[2]];
        let b_near = flux_density_current_element_scalar(src, moment, obs_near);
        let a_near = vector_potential_current_element_scalar(src, moment, obs_near);
        assert_eq!(b_near, [0.0, 0.0, 0.0]);
        assert_eq!(a_near, [0.0, 0.0, 0.0]);

        let obs_far = [src[0] + 2.0e-14, src[1], src[2]];
        let b_far = flux_density_current_element_scalar(src, moment, obs_far);
        let a_far = vector_potential_current_element_scalar(src, moment, obs_far);
        assert!(b_far.iter().all(|v| v.is_finite()));
        assert!(a_far.iter().all(|v| v.is_finite()));
        assert!(
            b_far.iter().any(|&v| v != 0.0),
            "flux-density current-element kernel stayed zero outside distance tolerance"
        );
        assert!(
            a_far.iter().any(|&v| v != 0.0),
            "vector-potential current-element kernel stayed zero outside distance tolerance"
        );
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
            body_force_density_point_segment(
                (&x[..ndiscr - 1], &y[..ndiscr - 1], &z[..ndiscr - 1]),
                dl,
                &vec![ni; x.len()][..],
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
                body_force_density_point_segment(
                    (&xi[..ndiscr - 1], &yi[..ndiscr - 1], &zi[..ndiscr - 1]),
                    dli,
                    &vec![ni * nj; xi.len() - 1][..],
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

        let gl3_unit = gauss_legendre_unit_interval_table(GaussLegendreRule::Gauss3);
        let mut xquad2 = vec![0.0; 3 * (NFIL - 1)];
        let mut yquad2 = vec![0.0; 3 * (NFIL - 1)];
        let mut zquad2 = vec![0.0; 3 * (NFIL - 1)];
        for i in 0..NFIL - 1 {
            let row = 3 * i;
            for (iq, &[tq, _]) in gl3_unit.iter().enumerate() {
                xquad2[row + iq] = dlxfil2[i].mul_add(tq, xfil2[i]);
                yquad2[row + iq] = dlyfil2[i].mul_add(tq, yfil2[i]);
                zquad2[row + iq] = dlzfil2[i].mul_add(tq, zfil2[i]);
            }
        }

        // Get the point-segment vector potential on the target quadrature points.
        let outx = &mut [0.0; 3 * (NFIL - 1)];
        let outy = &mut [0.0; 3 * (NFIL - 1)];
        let outz = &mut [0.0; 3 * (NFIL - 1)];
        vector_potential_point_segment(
            (&xquad2, &yquad2, &zquad2),
            (&xyz, &xyz, &xyz),
            (&dlxyz, &dlxyz, &dlxyz),
            &[1.0],
            (outx, outy, outz),
        )
        .unwrap();
        // Here the mutual inductance of the two filaments is calculated from the
        // vector potential at filament 2 due to 1 ampere of current flowing in filament 1.
        // By Stokes' theorem, the line integral of A over filament 2 is equal to the
        // magnetic flux through a surface bounded by filament 2. The flux through
        // filament 2 due to 1 ampere of current in filament 1 is the mutual inductance.
        // (We are stretching the applicability of Stokes' therorem because the filaments
        // are not closed loops)
        let m_from_point_segment_a = (0..NFIL - 1)
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

        // Use the finite-segment vector potential as the reference for inductance.
        let outx_ref = &mut [0.0; 3 * (NFIL - 1)];
        let outy_ref = &mut [0.0; 3 * (NFIL - 1)];
        let outz_ref = &mut [0.0; 3 * (NFIL - 1)];
        vector_potential_linear_filament(
            (&xquad2, &yquad2, &zquad2),
            (&xyz, &xyz, &xyz),
            (&dlxyz, &dlxyz, &dlxyz),
            &[1.0],
            &[0.0],
            (outx_ref, outy_ref, outz_ref),
        )
        .unwrap();
        let m_from_line_a = (0..NFIL - 1)
            .map(|i| {
                let row = 3 * i;
                (0..3)
                    .map(|iq| {
                        let idx = row + iq;
                        gl3_unit[iq][1]
                            * (outx_ref[idx] * dlxfil2[i]
                                + outy_ref[idx] * dlyfil2[i]
                                + outz_ref[idx] * dlzfil2[i])
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
        assert!(approx(m, m_from_line_a, 1e-12, 1e-15));
        assert!(m_from_point_segment_a.is_finite());

        let vp = |x: f64, y: f64, z: f64| {
            let mut outx = [0.0];
            let mut outy = [0.0];
            let mut outz = [0.0];

            vector_potential_point_segment(
                (&[x], &[y], &[z]),
                (&xyz, &xyz, &xyz),
                (&dlxyz, &dlxyz, &dlxyz),
                &[1.0],
                (&mut outx, &mut outy, &mut outz),
            )
            .unwrap();

            (outx[0], outy[0], outz[0])
        };

        let vals = [
            0.25, 0.5, 2.5, 10.0, 100.0, 1000.0, -1000.0, -100.0, -10.0, -2.5, -0.5, -0.25,
        ];
        // finite diff delta needs to be small enough to be accurate
        // but large enough that we can tell the difference between adjacent points
        // that are very far from the origin
        let eps = 1e-7;
        for x in vals.iter() {
            for y in vals.iter() {
                for z in vals.iter() {
                    let x = &(x + 1e-2); // Slightly adjust to avoid nans
                    let y = &(y + 1e-2);
                    let z = &(z - 1e-2);

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
                    flux_density_point_segment(
                        (&[*x], &[*y], &[*z]),
                        (&xyz, &xyz, &xyz),
                        (&dlxyz, &dlxyz, &dlxyz),
                        &[1.0],
                        (&mut bx, &mut by, &mut bz),
                    )
                    .unwrap();

                    assert!(approx(bx[0], ca[0], 1e-6, 1e-15));
                    assert!(approx(by[0], ca[1], 1e-6, 1e-15));
                    assert!(approx(bz[0], ca[2], 1e-6, 1e-15));
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
        flux_density_point_segment(xyzp, xyzfil, dlxyzfil, ifil, (out0, out1, out2)).unwrap();
        flux_density_point_segment_par(xyzp, xyzfil, dlxyzfil, ifil, (out3, out4, out5)).unwrap();
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
        vector_potential_point_segment(xyzp, xyzfil, dlxyzfil, ifil, (out0, out1, out2)).unwrap();
        vector_potential_point_segment_par(xyzp, xyzfil, dlxyzfil, ifil, (out3, out4, out5))
            .unwrap();
        for i in 0..NOBS {
            assert_eq!(out0[i], out3[i]);
            assert_eq!(out1[i], out4[i]);
            assert_eq!(out2[i], out5[i]);
        }
    }
}
