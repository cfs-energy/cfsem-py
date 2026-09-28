//! Magnetics calculations for circular current filaments.

use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector3};

use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

use crate::{
    chunksize,
    macros::{check_length, check_length_3tup, mut_par_chunks_3tup, par_chunks_3tup},
    math::{dot3, ellipe, ellipe_complement, ellipk, ellipk_complement, norm3},
};

use crate::{MU_0, MU0_OVER_4PI};

/// Observation-radius / filament-radius cutoff for the on-axis approximation.
const ON_AXIS_RADIUS_RATIO: f64 = 1e-4;

/// Poloidal flux from circular conductors with per-source circular cross-section radii.
/// Parallelized over chunks of observation points.
///
/// # Arguments
///
/// * `rzifil`: (m, m, A-turns) major radius, z-coord, and current per source, length `m`
/// * `wire_radius`: (m) circular cross-section radius per source, length `m`; zero for thin filaments
/// * `rzobs`: (m, m) cylindrical observation coordinates, length `n`
/// * `out`: (Wb) poloidal flux at each observation, length `n`
///
/// See [vector_potential_circular_filament_finite_thickness_scalar] for the
/// positive-radius interior and near-exterior approximation, and
/// [vector_potential_circular_filament_scalar] for the ideal-filament formula.
/// Flux is $2\pi R_\mathrm{obs} A_\phi$.
pub fn flux_circular_filament_par(
    rzifil: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    rzobs: (&[f64], &[f64]),
    out: &mut [f64],
) -> Result<(), &'static str> {
    vector_potential_circular_filament_par(rzifil, wire_radius, rzobs, out)?;
    for (flux, r) in out.iter_mut().zip(rzobs.0) {
        *flux *= 2.0 * core::f64::consts::PI * r;
    }
    Ok(())
}

/// Poloidal flux from circular conductors with per-source circular cross-section radii.
///
/// # Arguments
///
/// * `rzifil`: (m, m, A-turns) major radius, z-coord, and current per source, length `m`
/// * `wire_radius`: (m) circular cross-section radius per source, length `m`; zero for thin filaments
/// * `rzobs`: (m, m) cylindrical observation coordinates, length `n`
/// * `out`: (Wb) poloidal flux at each observation, length `n`
///
/// See [vector_potential_circular_filament_finite_thickness_scalar] for the
/// positive-radius interior and near-exterior approximation, and
/// [vector_potential_circular_filament_scalar] for the ideal-filament formula.
/// Flux is $2\pi R_\mathrm{obs} A_\phi$.
pub fn flux_circular_filament(
    rzifil: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    rzobs: (&[f64], &[f64]),
    out: &mut [f64],
) -> Result<(), &'static str> {
    vector_potential_circular_filament(rzifil, wire_radius, rzobs, out)?;
    // Radius depends only on the observation, so scale once after summing sources.
    for (flux, r) in out.iter_mut().zip(rzobs.0) {
        *flux *= 2.0 * core::f64::consts::PI * r;
    }
    Ok(())
}

/// Poloidal flux (Wb) from one circular conductor at one observation point.
///
/// Arguments are `(major radius, z, current)` in (m, m, A-turns), the circular
/// cross-section `wire_radius` in m, and cylindrical `(R, Z)` observation coordinates in m.
/// Uses $\Psi=2\pi R_\mathrm{obs} A_\phi$, with the model and validity limits of
/// [vector_potential_circular_filament_finite_thickness_scalar]. Zero wire radius
/// selects the ideal-filament formula in [vector_potential_circular_filament_scalar].
#[inline]
pub fn flux_circular_filament_scalar(
    rzifil: (f64, f64, f64),
    wire_radius: f64,
    rzobs: (f64, f64),
) -> f64 {
    2.0 * core::f64::consts::PI
        * rzobs.0
        * vector_potential_circular_filament_finite_thickness_scalar(rzifil, wire_radius, rzobs)
}

// Original thin-filament kernels retained independently for regression tests.
#[cfg(test)]
fn flux_circular_filament_thin(
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
            out[i] += flux_circular_filament_thin_scalar(
                (rfil[j], zfil[j], ifil[j]),
                (rprime[i], zprime[i]),
            );
        }
    }

    Ok(())
}

#[cfg(test)]
fn flux_circular_filament_thin_scalar(rzifil: (f64, f64, f64), rzobs: (f64, f64)) -> f64 {
    // Unpack
    let (rfil, zfil, ifil) = rzifil;
    let (rprime, zprime) = rzobs;
    // Evaluate
    let rrprime = rfil * rprime;
    let rpr = rfil + rprime;
    let zmz = zfil - zprime;
    let k2 = 4.0 * rrprime / (rpr.mul_add(rpr, zmz * zmz));
    // [V-s]
    MU_0 * ifil * (rrprime / k2).sqrt() * ((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2))
}

/// Br,Bz from circular conductors with per-source circular cross-section radii.
/// This variant of the function is parallelized over chunks of observation points.
///
/// # Arguments
///
/// * `rzifil`:  (m, m, A-turns) r-coord, z-coord, and current of each filament, length `m`
/// * `wire_radius`: (m) circular conductor-section radius per source, length `m`; zero for thin filaments
/// * `rzobs`:   (m, m) r-coord, and z-coord of each observation point, length `n`
/// * `out`:     (T, T), r- and z-components of magnetic flux density at observation location, length `n`
///
/// For field formulas, numerical treatment, and references, see
/// [flux_density_circular_filament_finite_radius_scalar] for positive-radius
/// interior and near-exterior fields and [flux_density_circular_filament_scalar]
/// for the zero-radius ideal-filament field.
pub fn flux_density_circular_filament_par(
    rzifil: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    rzobs: (&[f64], &[f64]),
    out: (&mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    // Unpack
    let (rprime, zprime) = rzobs;
    let (out_r, out_z) = out;

    check_length_3tup!(rzifil.2.len(), &rzifil);
    check_length!(rzifil.2.len(), wire_radius);
    check_length!(rprime.len(), zprime, out_r, out_z);

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
            flux_density_circular_filament(rzifil, wire_radius, (rc, zc), (orc, ozc))
        })?;

    Ok(())
}

/// Br,Bz from circular conductors with per-source circular cross-section radii.
///
/// # Arguments
///
/// * `rzifil`:  (m, m, A-turns) r-coord, z-coord, and current of each filament, length `m`
/// * `wire_radius`: (m) circular conductor-section radius per source, length `m`; zero for thin filaments
/// * `rzobs`:   (m, m) r-coord, and z-coord of each observation point, length `n`
/// * `out`:     (T, T), r- and z-components of magnetic flux density at observation location, length `n`
///
/// For field formulas, numerical treatment, and references, see
/// [flux_density_circular_filament_finite_radius_scalar] for positive-radius
/// interior and near-exterior fields and [flux_density_circular_filament_scalar]
/// for the zero-radius ideal-filament field.
pub fn flux_density_circular_filament(
    rzifil: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    rzobs: (&[f64], &[f64]),
    out: (&mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let (rfil, zfil, ifil) = rzifil;
    let (rprime, zprime) = rzobs;
    let (out_r, out_z) = out;

    let n = ifil.len();
    let m = rprime.len();
    check_length_3tup!(n, &rzifil);
    check_length!(n, wire_radius);
    check_length!(m, zprime, &out_r, &out_z);

    out_r.fill(0.0);
    out_z.fill(0.0);

    // With one observation point there is no observation loop to vectorize.
    // Sum scalar contributions directly to avoid per-filament run dispatch.
    if m == 1 {
        for i in 0..n {
            let source = (rfil[i], zfil[i], ifil[i]);
            let obs = (rprime[0], zprime[0]);
            let (br, bz) = if wire_radius[i] == 0.0 {
                flux_density_circular_filament_scalar(source, obs)
            } else {
                flux_density_circular_filament_finite_radius_scalar(source, wire_radius[i], obs)
            };
            out_r[0] += br;
            out_z[0] += bz;
        }
        return Ok(());
    }

    // Outside the largest filament's cutoff, every contribution is off-axis.
    // Group observation points once to avoid per-pair axis checks for thin sources.
    let max_filament_radius = rfil.iter().map(|r| r.abs()).fold(0.0, f64::max);
    let max_cutoff = ON_AXIS_RADIUS_RATIO * max_filament_radius;
    let mut start = 0;
    for radii in rprime.chunk_by(|a, b| (a.abs() <= max_cutoff) == (b.abs() <= max_cutoff)) {
        let end = start + radii.len();
        let obs = (radii, &zprime[start..end]);
        let out = (&mut out_r[start..end], &mut out_z[start..end]);
        if radii[0].abs() <= max_cutoff {
            // The scalar kernel checks each filament's own cutoff.
            accumulate_flux_density(
                rzifil,
                wire_radius,
                obs,
                out,
                flux_density_circular_filament_scalar,
                flux_density_circular_filament_finite_radius_scalar,
            );
        } else {
            accumulate_flux_density(
                rzifil,
                wire_radius,
                obs,
                out,
                flux_density_circular_filament_off_axis,
                flux_density_circular_filament_finite_radius_off_axis,
            );
        }
        start = end;
    }

    Ok(())
}

#[inline]
fn accumulate_flux_density(
    rzifil: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    rzobs: (&[f64], &[f64]),
    out: (&mut [f64], &mut [f64]),
    thin_field: impl Fn((f64, f64, f64), (f64, f64)) -> (f64, f64),
    field: impl Fn((f64, f64, f64), f64, (f64, f64)) -> (f64, f64),
) {
    let (rfil, zfil, ifil) = rzifil;
    let (rprime, zprime) = rzobs;
    let (out_r, out_z) = out;
    for i in 0..ifil.len() {
        let source = (rfil[i], zfil[i], ifil[i]);
        // Dispatch once per source to retain vectorization for zero-radius
        // sources, including when mixed with finite-radius sources.
        if wire_radius[i] == 0.0 {
            for j in 0..rprime.len() {
                let (br, bz) = thin_field(source, (rprime[j], zprime[j]));
                out_r[j] += br;
                out_z[j] += bz;
            }
        } else {
            for j in 0..rprime.len() {
                let (br, bz) = field(source, wire_radius[i], (rprime[j], zprime[j]));
                out_r[j] += br;
                out_z[j] += bz;
            }
        }
    }
}

// Preserved thin-filament vector kernel for independent zero-radius regression tests.
#[cfg(test)]
fn flux_density_circular_filament_thin(
    rzifil: (&[f64], &[f64], &[f64]),
    rzobs: (&[f64], &[f64]),
    out: (&mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let (rfil, zfil, ifil) = rzifil;
    let (rprime, zprime) = rzobs;
    let (out_r, out_z) = out;

    let n = ifil.len();
    let m = rprime.len();
    check_length_3tup!(n, &rzifil);
    check_length!(m, &out_r, &out_z);

    out_r.fill(0.0);
    out_z.fill(0.0);

    // With one observation point there is no observation loop to vectorize.
    // Sum scalar contributions directly to avoid per-filament run dispatch.
    if m == 1 {
        for i in 0..n {
            let (br, bz) = flux_density_circular_filament_thin_scalar(
                (rfil[i], zfil[i], ifil[i]),
                (rprime[0], zprime[0]),
            );
            out_r[0] += br;
            out_z[0] += bz;
        }
        return Ok(());
    }

    // Outside the largest filament's cutoff, every contribution is off-axis.
    // Group observation points once to preserve the branch-free off-axis loop.
    let max_filament_radius = rfil.iter().map(|r| r.abs()).fold(0.0, f64::max);
    let max_cutoff = ON_AXIS_RADIUS_RATIO * max_filament_radius;
    let mut start = 0;
    for radii in rprime.chunk_by(|a, b| (a.abs() <= max_cutoff) == (b.abs() <= max_cutoff)) {
        let end = start + radii.len();
        let obs = (radii, &zprime[start..end]);
        let out = (&mut out_r[start..end], &mut out_z[start..end]);
        if radii[0].abs() <= max_cutoff {
            // The scalar kernel checks each filament's own cutoff.
            accumulate_flux_density_thin(
                rzifil,
                obs,
                out,
                flux_density_circular_filament_thin_scalar,
            );
        } else {
            accumulate_flux_density_thin(rzifil, obs, out, |filament, observation| {
                flux_density_circular_filament_off_axis(filament, observation)
            });
        }
        start = end;
    }

    Ok(())
}

#[cfg(test)]
#[inline]
fn accumulate_flux_density_thin(
    rzifil: (&[f64], &[f64], &[f64]),
    rzobs: (&[f64], &[f64]),
    out: (&mut [f64], &mut [f64]),
    field: impl Fn((f64, f64, f64), (f64, f64)) -> (f64, f64),
) {
    let (rfil, zfil, ifil) = rzifil;
    let (rprime, zprime) = rzobs;
    let (out_r, out_z) = out;
    for i in 0..ifil.len() {
        for j in 0..rprime.len() {
            let (br, bz) = field((rfil[i], zfil[i], ifil[i]), (rprime[j], zprime[j]));
            out_r[j] += br;
            out_z[j] += bz;
        }
    }
}

#[cfg(test)]
fn flux_density_circular_filament_thin_scalar(
    rzifil: (f64, f64, f64),
    rzobs: (f64, f64),
) -> (f64, f64) {
    if rzobs.0.abs() <= ON_AXIS_RADIUS_RATIO * rzifil.0.abs() {
        return flux_density_circular_filament_on_axis(rzifil, rzobs.1);
    }
    flux_density_circular_filament_off_axis(rzifil, rzobs)
}

/// Br,Bz components for a circular current filament in vacuum, including on the axis.
///
/// The ideal filament remains singular at the source location.
///
/// # Arguments
///
/// * `rzifil`:  (m, m, A-turns) r-coord, z-coord, and current of filament
/// * `rzobs`:   (m, m) r-coord and z-coord of observation point
///
/// # Returns
///
/// * `(br, bz)`:   (T, T), r- and z-component of magnetic flux density at observation location
///
/// # On-axis field
///
/// For `R/a <= 1e-4`, where `a` is the filament radius, uses
/// [flux_density_circular_filament_on_axis] at the same axial position.
/// This clips the small radial field to zero and approximates the axial field
/// to avoid cancellation in the elliptic-integral expression near the axis.
/// The cutoff is relative to each filament's radius and includes its boundary.
/// Outside the cutoff, the off-axis expression is used without modification.
///
/// # Off-axis field
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
///   \[2\] MIT, *8.02 Course Notes*, Chapter 9, “Sources of Magnetic Fields,”
///         Example 9.2, eqs. 9.1.13–9.1.15 (on-axis), and Appendix 1, eq. 9.8.7 (off-axis).
///         Available: <https://web.mit.edu/8.02t/www/802TEAL3D/visualizations/coursenotes/modules/guide09.pdf>
///
///   \[3\] Eric Dennyson, "Magnet Formulas". Available: <https://tiggerntatie.github.io/emagnet-py/offaxis/off_axis_loop.html>
///
///   \[4\] J. C. Simpson, J. E. Lane, C. D. Immer, R. C. Youngquist, and T. Steinrock,
///         “Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop,”
///         Jan. 01, 2001. Accessed: Sep. 06, 2022. \[Online\]. Available: <https://ntrs.nasa.gov/citations/20010038494>
#[inline]
pub fn flux_density_circular_filament_scalar(
    rzifil: (f64, f64, f64),
    rzobs: (f64, f64),
) -> (f64, f64) {
    flux_density_circular_filament_finite_radius_scalar(rzifil, 0.0, rzobs)
}

/// Br,Bz inside and near a circular loop with a finite circular conductor section.
///
/// Implements the thin-conductor model of Hurwitz et al. \[1\], equations 16–19,
/// for uniform azimuthal current density `I / (pi * wire_radius^2)` in vacuum.
/// For positive wire radius, requires `wire_radius < rfil`; accuracy requires
/// `wire_radius / rfil << 1` and distance from the conductor centerline comparable
/// to the wire radius.
/// This is a local approximation, including the just-outside field, not a
/// far-field calculation or an exact solution for a thick torus.
/// A zero wire radius instead evaluates the ideal-filament formula, including
/// the on-axis treatment documented in [flux_density_circular_filament_scalar].
/// The zero-radius field is valid throughout space except at the filament itself.
///
/// # Arguments
///
/// * `rzifil`: (m, m, A-turns) loop major radius, z-coord, and total current
/// * `wire_radius`: (m) radius of the circular conductor cross-section; zero for an ideal filament
/// * `rzobs`: (m, m) cylindrical observation coordinates
///
/// # Returns
///
/// * `(br, bz)`: (T, T) radial and axial magnetic flux density
///
/// # Formula
///
/// The field is the sum of a regularized centerline field, the local straight
/// cylinder field, and a curvature correction (equations 17, 18, and 19).
/// With major radius $a$, wire radius $b$, and $Q = 4a^2 + b^2/\sqrt{e}$,
/// the centerline integral in equation 17 reduces to
///
/// $$B_{\mathrm{reg},Z} = \frac{\mu_0 I}{2\pi\sqrt{Q}}[K(m)-E(m)],
/// \qquad m = \frac{4a^2}{Q}.$$
///
/// The complementary parameter $1-m = b^2/(\sqrt{e}Q)$ is computed directly
/// to avoid rounding $m$ to one for very thin conductors. The elliptic integrals
/// use the same approximations as [ellipk] and [ellipe].
///
/// In the local frame of the paper, $\mathbf{e}_2=-\mathbf{e}_R$,
/// $\mathbf{e}_3=\mathbf{e}_Z$, and curvature is $1/a$. Writing
/// $u=(R-a)/b$ and $v=(Z-Z_\mathrm{fil})/b$, the interior correction is
///
/// $$\mathbf{B}^{<} = \frac{\mu_0 I}{8\pi a}
/// \left[-uv\,\mathbf{e}_R + \left(\frac32-\frac{u^2+3v^2}{2}\right)
/// \mathbf{e}_Z\right].$$
///
/// The interior branch includes the surface; equation 19b supplies the
/// continuous near-exterior correction. At the conductor centerline, the
/// radial field vanishes and the axial field is finite:
/// $B_Z=B_{\mathrm{reg},Z}+3\mu_0 I/(16\pi a)$, approaching
/// $\mu_0 I\ln(8a/b)/(4\pi a)$ for $b/a\to0$.
///
/// # References
///
/// \[1\] S. Hurwitz, M. Landreman, and T. M. Antonsen Jr.,
/// “Efficient calculation of the self magnetic field, self-force, and
/// self-inductance for electromagnetic coils,” 2023, equations 16–19.
/// Available: <https://arxiv.org/abs/2310.09313>.
#[inline]
pub fn flux_density_circular_filament_finite_radius_scalar(
    rzifil: (f64, f64, f64),
    wire_radius: f64,
    rzobs: (f64, f64),
) -> (f64, f64) {
    if wire_radius == 0.0 && rzobs.0.abs() <= ON_AXIS_RADIUS_RATIO * rzifil.0.abs() {
        return flux_density_circular_filament_on_axis(rzifil, rzobs.1);
    }
    flux_density_circular_filament_finite_radius_off_axis(rzifil, wire_radius, rzobs)
}

/// Finite-radius dispatch after the ideal filament's on-axis case is handled.
/// Bulk callers with known off-axis observations can skip the per-pair axis check.
#[inline]
fn flux_density_circular_filament_finite_radius_off_axis(
    rzifil: (f64, f64, f64),
    wire_radius: f64,
    rzobs: (f64, f64),
) -> (f64, f64) {
    if wire_radius == 0.0 {
        return flux_density_circular_filament_off_axis(rzifil, rzobs);
    }

    let (rfil, zfil, ifil) = rzifil;
    let u = (rzobs.0 - rfil) / wire_radius; // [nondim], outward from centerline
    let v = (rzobs.1 - zfil) / wire_radius; // [nondim], axial offset
    let s2 = u.mul_add(u, v * v); // [nondim], squared distance / wire_radius^2

    let aspect = wire_radius / rfil; // [nondim]
    let core2 = aspect * aspect / core::f64::consts::E.sqrt(); // [nondim]
    let q = 4.0 + core2; // [nondim], Q / rfil^2
    let complement = core2 / q; // [nondim], 1 - m without cancellation
    let loop_scale = MU0_OVER_4PI * ifil / rfil; // [T]
    let bz_reg = 2.0 * loop_scale / q.sqrt()
        * (ellipk_complement(complement) - ellipe_complement(complement)); // [T]
    let cylinder_scale = 2.0 * MU0_OVER_4PI * ifil / wire_radius; // [T]
    let curvature_scale = 0.5 * loop_scale; // [T]

    if s2 <= 1.0 {
        // Equations 18 and 19a, with no divisions by distance at the centerline.
        let br = cylinder_scale * v - curvature_scale * u * v;
        let bz =
            bz_reg - cylinder_scale * u + curvature_scale * (1.5 - 0.5 * u.mul_add(u, 3.0 * v * v));
        (br, bz)
    } else {
        // Equations 18 and 19b; cos(2 theta) = (u^2 - v^2) / s2.
        let inv_s2 = s2.recip();
        let cos_2theta = (u * u - v * v) * inv_s2;
        let br = cylinder_scale * v * inv_s2 + curvature_scale * u * v * inv_s2 * (inv_s2 - 2.0);
        let bz = bz_reg - cylinder_scale * u * inv_s2
            + curvature_scale * (0.5 - s2.ln() + cos_2theta * (1.0 - 0.5 * inv_s2));
        (br, bz)
    }
}

/// Br,Bz components on the symmetry axis of a circular current filament in vacuum.
///
/// # Arguments
///
/// * `rzifil`: (m, m, A-turns) positive radius, z-coord, and current of the filament
/// * `zobs`: (m) z-coord of the observation point on the axis
///
/// # Returns
///
/// * `(br, bz)`: (T, T) magnetic flux density, with `br = 0`
///
/// # Formula
///
/// For a loop of radius $a = r_\mathrm{fil} > 0$, current $I = i_\mathrm{fil}$,
/// and axial separation $\Delta z = z_\mathrm{obs} - z_\mathrm{fil}$, symmetry gives
/// $B_R = 0$ on the axis. Integrating the Biot-Savart law gives
///
/// $$B_Z = \frac{\mu_0 I a^2}{2(a^2 + \Delta z^2)^{3/2}}.$$
///
/// See \[1\], Example 9.2, eqs. 9.1.13–9.1.15. At the loop center this reduces
/// to $B_Z = \mu_0 I/(2a)$; the sign follows the current's right-hand rule.
///
/// This analytic expression avoids the removable division by `R` in the
/// off-axis radial field and cancellation
/// in the axial field far from the loop. It requires no elliptic integrals.
/// With $d^2 = a^2 + \Delta z^2$, the calculation uses
/// $(\mu_0 I/2)(a^2/d^2)/\sqrt{d^2}$ to avoid forming a cubed distance.
///
/// # References
///
/// \[1\] MIT, *8.02 Course Notes*, Chapter 9, “Sources of Magnetic Fields,”
/// Example 9.2, eqs. 9.1.13–9.1.15.
/// Available: <https://web.mit.edu/8.02t/www/802TEAL3D/visualizations/coursenotes/modules/guide09.pdf>
#[inline]
pub fn flux_density_circular_filament_on_axis(rzifil: (f64, f64, f64), zobs: f64) -> (f64, f64) {
    let (rfil, zfil, ifil) = rzifil;
    let z = zobs - zfil; // [m]
    let d2 = rfil.mul_add(rfil, z * z); // [m^2]
    let bz = 0.5 * MU_0 * ifil * (rfil * rfil / d2) / d2.sqrt(); // [T]
    (0.0, bz)
}

/// Off-axis Br,Bz components for one circular current filament at one observation point.
///
/// Requires nonzero observation radius. See [flux_density_circular_filament_scalar]
/// for the elliptic-integral formula references, axis limit, and singularity behavior.
#[inline]
fn flux_density_circular_filament_off_axis(
    rzifil: (f64, f64, f64),
    rzobs: (f64, f64),
) -> (f64, f64) {
    let (rfil, zfil, ifil) = rzifil;
    let (rprime, zprime) = rzobs;
    let z = zprime - zfil; // [m]
    let z2 = z * z; // [m^2]
    let r2 = rprime * rprime; // [m^2]

    let rpr = rfil + rprime; // [m]

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

/// Cartesian field of one translated and oriented circular conductor.
///
/// `rifil` is (major radius in m, current in A-turns), `loc` is the loop
/// center in m, and `normal` is a finite nonzero Cartesian normal (normalized
/// internally). Positive current follows the right-hand rule about this normal.
/// `wire_radius` is the circular conductor-section radius in m; zero selects
/// an ideal filament. `xyzobs` and the returned field (T) use world coordinates.
///
/// A nalgebra isometry maps the observation into the loop's local xy plane;
/// the local field is rotated back without translating the vector. See
/// [flux_density_circular_filament_finite_radius_scalar] for the positive-radius
/// near-conductor approximation and [flux_density_circular_filament_scalar]
/// for the ideal-filament formula and on-axis treatment.
///
/// Invalid geometry propagates through floating-point arithmetic as NaNs.
pub fn flux_density_circular_filament_cartesian_scalar(
    rifil: (f64, f64),
    loc: (f64, f64, f64),
    normal: (f64, f64, f64),
    wire_radius: f64,
    xyzobs: (f64, f64, f64),
) -> (f64, f64, f64) {
    let source = CartesianCircularFilament::new(rifil, loc, normal, wire_radius);
    let b = source.field(Point3::new(xyzobs.0, xyzobs.1, xyzobs.2));
    (b.x, b.y, b.z)
}

/// Cartesian fields from independently located and oriented circular conductors.
///
/// `rifil` contains (major radii, currents); `loc` contains loop centers and
/// `normal` contains loop normals as (x, y, z) component slices. All source
/// slices, including `wire_radius`, have the same length. `xyzobs` and
/// `bxyz_out` have one entry per observation, in world coordinates (m and T).
/// See [flux_density_circular_filament_cartesian_scalar] for conventions,
/// field formulas, validity limits, and NaN propagation for invalid geometry.
pub fn flux_density_circular_filament_cartesian(
    rifil: (&[f64], &[f64]),
    loc: (&[f64], &[f64], &[f64]),
    normal: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    xyzobs: (&[f64], &[f64], &[f64]),
    bxyz_out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    check_length_3tup!(xyzobs.0.len(), &xyzobs);
    check_length_3tup!(xyzobs.0.len(), &bxyz_out);
    let sources = cartesian_circular_sources(rifil, loc, normal, wire_radius)?;
    cartesian_circular_field(&sources, xyzobs, bxyz_out);
    Ok(())
}

/// Parallel version of [flux_density_circular_filament_cartesian].
/// Uses the same per-source centers, normals, radii, and world-coordinate outputs.
pub fn flux_density_circular_filament_cartesian_par(
    rifil: (&[f64], &[f64]),
    loc: (&[f64], &[f64], &[f64]),
    normal: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    xyzobs: (&[f64], &[f64], &[f64]),
    bxyz_out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    check_length_3tup!(xyzobs.0.len(), &xyzobs);
    check_length_3tup!(xyzobs.0.len(), &bxyz_out);
    // Build each transform once, then share it across chunks.
    let sources = cartesian_circular_sources(rifil, loc, normal, wire_radius)?;
    let n = chunksize(xyzobs.0.len());
    let (xc, yc, zc) = par_chunks_3tup!(xyzobs, n);
    let (bx, by, bz) = mut_par_chunks_3tup!(bxyz_out, n);
    (xc, yc, zc, bx, by, bz)
        .into_par_iter()
        .for_each(|(x, y, z, bx, by, bz)| {
            cartesian_circular_field(&sources, (x, y, z), (bx, by, bz));
        });
    Ok(())
}

// A loop's fixed geometry is constructed once per source, not per observation.
struct CartesianCircularFilament {
    local_to_world: Isometry3<f64>,
    radius: f64,
    current: f64,
    wire_radius: f64,
}

impl CartesianCircularFilament {
    fn new(
        rifil: (f64, f64),
        center: (f64, f64, f64),
        normal: (f64, f64, f64),
        wire_radius: f64,
    ) -> Self {
        let center = Vector3::new(center.0, center.1, center.2);
        let normal = Vector3::new(normal.0, normal.1, normal.2);
        // Scale before normalizing to avoid overflow/underflow for non-unit inputs.
        let normal = (normal / normal.amax()).normalize();
        // Roll about the normal is immaterial. Choose a nonparallel up vector
        // so the frame remains well conditioned, including nearly reversed axes.
        let up = if normal.y.abs() < 0.9 {
            Vector3::y()
        } else {
            Vector3::x()
        };
        let rotation = UnitQuaternion::face_towards(&normal, &up);
        Self {
            local_to_world: Isometry3::from_parts(Translation3::from(center), rotation),
            radius: rifil.0,
            current: rifil.1,
            wire_radius,
        }
    }

    #[inline]
    fn field(&self, observation: Point3<f64>) -> Vector3<f64> {
        let local = self.local_to_world.inverse_transform_point(&observation);
        let radial = Vector3::new(local.x, local.y, 0.0);
        let r = radial.norm();
        let (br, bz) = flux_density_circular_filament_finite_radius_scalar(
            (self.radius, 0.0, self.current),
            self.wire_radius,
            (r, local.z),
        );
        let mut b = Vector3::new(0.0, 0.0, bz);
        if r != 0.0 {
            b += (radial / r) * br;
        }
        self.local_to_world.transform_vector(&b)
    }
}

fn cartesian_circular_sources(
    rifil: (&[f64], &[f64]),
    loc: (&[f64], &[f64], &[f64]),
    normal: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
) -> Result<Vec<CartesianCircularFilament>, &'static str> {
    let n = rifil.0.len();
    check_length!(n, rifil.1, wire_radius);
    check_length_3tup!(n, &loc);
    check_length_3tup!(n, &normal);
    Ok((0..n)
        .map(|i| {
            CartesianCircularFilament::new(
                (rifil.0[i], rifil.1[i]),
                (loc.0[i], loc.1[i], loc.2[i]),
                (normal.0[i], normal.1[i], normal.2[i]),
                wire_radius[i],
            )
        })
        .collect())
}

fn cartesian_circular_field(
    sources: &[CartesianCircularFilament],
    xyzobs: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) {
    out.0.fill(0.0);
    out.1.fill(0.0);
    out.2.fill(0.0);
    for source in sources {
        for j in 0..xyzobs.0.len() {
            let b = source.field(Point3::new(xyzobs.0[j], xyzobs.1[j], xyzobs.2[j]));
            out.0[j] += b.x;
            out.1[j] += b.y;
            out.2[j] += b.z;
        }
    }
}

/// A_phi from circular conductors with per-source circular cross-section radii.
/// Parallelized over chunks of observation points.
///
/// # Arguments
///
/// * `rzifil`: (m, m, A-turns) major radius, z-coord, and current per source, length `m`
/// * `wire_radius`: (m) circular cross-section radius per source, length `m`; zero for thin filaments
/// * `rzobs`: (m, m) cylindrical observation coordinates, length `n`
/// * `out`: (V-s/m) azimuthal vector potential at each observation, length `n`
///
/// See [vector_potential_circular_filament_finite_thickness_scalar] for the
/// positive-radius interior and near-exterior approximation, and
/// [vector_potential_circular_filament_scalar] for the ideal-filament formula.
pub fn vector_potential_circular_filament_par(
    rzifil: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    rzobs: (&[f64], &[f64]),
    out: &mut [f64],
) -> Result<(), &'static str> {
    let (rprime, zprime) = rzobs;
    check_length_3tup!(rzifil.2.len(), &rzifil);
    check_length!(rzifil.2.len(), wire_radius);
    check_length!(rprime.len(), zprime, out);

    let n = chunksize(rprime.len());
    (
        out.par_chunks_mut(n),
        rprime.par_chunks(n),
        zprime.par_chunks(n),
    )
        .into_par_iter()
        .try_for_each(|(outc, rc, zc)| {
            vector_potential_circular_filament(rzifil, wire_radius, (rc, zc), outc)
        })
}

/// A_phi from circular conductors with per-source circular cross-section radii.
///
/// # Arguments
///
/// * `rzifil`: (m, m, A-turns) major radius, z-coord, and current per source, length `m`
/// * `wire_radius`: (m) circular cross-section radius per source, length `m`; zero for thin filaments
/// * `rzobs`: (m, m) cylindrical observation coordinates, length `n`
/// * `out`: (V-s/m) azimuthal vector potential at each observation, length `n`
///
/// See [vector_potential_circular_filament_finite_thickness_scalar] for the
/// positive-radius interior and near-exterior approximation, and
/// [vector_potential_circular_filament_scalar] for the ideal-filament formula.
pub fn vector_potential_circular_filament(
    rzifil: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    rzobs: (&[f64], &[f64]),
    out: &mut [f64],
) -> Result<(), &'static str> {
    let (rfil, zfil, ifil) = rzifil;
    let (rprime, zprime) = rzobs;
    check_length_3tup!(ifil.len(), &rzifil);
    check_length!(ifil.len(), wire_radius);
    check_length!(rprime.len(), zprime, out);
    out.fill(0.0);

    for i in 0..ifil.len() {
        let source = (rfil[i], zfil[i], ifil[i]);
        // Dispatch once per source to preserve the thin kernel's vectorization.
        if wire_radius[i] == 0.0 {
            for j in 0..rprime.len() {
                out[j] += vector_potential_circular_filament_scalar(source, (rprime[j], zprime[j]));
            }
        } else {
            for j in 0..rprime.len() {
                out[j] += vector_potential_circular_filament_finite_thickness_scalar(
                    source,
                    wire_radius[i],
                    (rprime[j], zprime[j]),
                );
            }
        }
    }
    Ok(())
}

// Original thin-filament vector kernel retained for regression tests.
#[cfg(test)]
fn vector_potential_circular_filament_thin(
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
            out[j] += vector_potential_circular_filament_thin_scalar(
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
///         Jan. 01, 2001. Accessed: Sep. 06, 2022. \[Online\]. Available: <https://ntrs.nasa.gov/citations/20010038494>
#[inline]
pub fn vector_potential_circular_filament_scalar(
    rzifil: (f64, f64, f64),
    rzobs: (f64, f64),
) -> f64 {
    vector_potential_circular_filament_thin_scalar(rzifil, rzobs)
}

// Original ideal-filament formula, also used by the private vector test reference.
#[inline]
fn vector_potential_circular_filament_thin_scalar(
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
    let rpr = rfil + rprime;
    let rpr2 = rpr * rpr;
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

/// A_phi inside and near a circular loop with a finite circular conductor section.
///
/// Assumes uniform azimuthal current density in vacuum, `0 < wire_radius < rfil`,
/// `wire_radius / rfil << 1`, and observation distance comparable to the wire radius.
/// Positive radii use the Hurwitz near-conductor approximation throughout; this
/// is not a global thick-torus solution. Zero radius delegates to
/// [vector_potential_circular_filament_scalar], preserving its singularities.
///
/// # Arguments
///
/// * `rzifil`: (m, m, A-turns) loop major radius, z-coord, and total current
/// * `wire_radius`: (m) circular cross-section radius; zero for an ideal filament
/// * `rzobs`: (m, m) cylindrical observation coordinates
///
/// # Returns
///
/// * `a_phi`: (V-s/m) azimuthal magnetic vector potential
///
/// # Formula
///
/// With major radius $a$, wire radius $b$, $x=R-a$, $s^2=x^2+(Z-Z_\mathrm{fil})^2$,
/// $q=x/a$, and $t=s^2/b^2$, specializing equations 35 and 53 of \[1\] gives
///
/// $$A_\phi=\frac{\mu_0 I}{4\pi}\begin{cases}
/// (2-q)[\ln(8a/b)-2]+1+q-t(1-q/4), & s\le b,\\
/// (2-q)[\ln(8a/s)-2]+q+q/(4t), & s>b.
/// \end{cases}$$
///
/// The far integral contributes $(2-q)[\ln(4/\phi_0)-2]$; its cutoff cancels
/// the $\ln(2\phi_0)$ in equation 53. The exterior uses $\ln(a/s)$ from
/// equation 53b (the summary equation 30 instead prints $\ln(a/b)$).
/// Both branches and their first derivatives agree at the surface. At the
/// centerline, $A_\phi=\mu_0 I[2\ln(8a/b)-3]/(4\pi)$ is finite.
///
/// # References
///
/// \[1\] S. Hurwitz, M. Landreman, and T. M. Antonsen Jr.,
/// “Efficient calculation of the self magnetic field, self-force, and
/// self-inductance for electromagnetic coils,” 2023, Appendix A, equations 35 and 53.
/// Available: <https://arxiv.org/abs/2310.09313>.
#[inline]
pub fn vector_potential_circular_filament_finite_thickness_scalar(
    rzifil: (f64, f64, f64),
    wire_radius: f64,
    rzobs: (f64, f64),
) -> f64 {
    if wire_radius == 0.0 {
        return vector_potential_circular_filament_scalar(rzifil, rzobs);
    }

    let (rfil, zfil, ifil) = rzifil;
    let x = rzobs.0 - rfil; // [m], outward from centerline
    let u = x / wire_radius;
    let v = (rzobs.1 - zfil) / wire_radius;
    let t = u.mul_add(u, v * v); // [nondim], s^2 / wire_radius^2
    let q = x / rfil; // [nondim], negative of kappa*s*cos(theta) in the paper
    let log_radius = (8.0 * (rfil / wire_radius)).ln();

    let potential = if t <= 1.0 {
        (2.0 - q) * (log_radius - 2.0) + 1.0 + q - t * (1.0 - 0.25 * q)
    } else {
        // ln(8a/s) = ln(8a/b) - ln(t)/2, retaining the exterior thickness term.
        (2.0 - q) * (log_radius - 0.5 * t.ln() - 2.0) + q + 0.25 * q / t
    };
    MU0_OVER_4PI * ifil * potential // [V-s/m]
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
///     Higher Education from Cambridge University Press. Accessed: Feb. 10, 2025. \[Online\].
///     Available: <https://www.cambridge.org/highereducation/books/electricity-and-magnetism/0F97BB6C5D3A56F19B9835EDBEAB087C>
///
/// \[2\] “Magnetic vector potential,” Wikipedia. Nov. 26, 2024. Accessed: Feb. 10, 2025. \[Online\].
///     Available: <https://en.wikipedia.org/w/index.php?title=Magnetic_vector_potential&oldid=1259654939#Magnetic_vector_potential>
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
    let path_r = norm3([xyzfil0.0, xyzfil0.1, 0.0]); // [m]
    let path_dr = norm3([dlxfil, dlyfil, 0.0]); // [m]

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

    dot3([a_x_per_A, a_y_per_A, a_z_per_A], [dlxfil, dlyfil, dlzfil])
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
///     Higher Education from Cambridge University Press. Accessed: Feb. 10, 2025. \[Online\].
///     Available: <https://www.cambridge.org/highereducation/books/electricity-and-magnetism/0F97BB6C5D3A56F19B9835EDBEAB087C>
///
/// \[2\] “Magnetic vector potential,” Wikipedia. Nov. 26, 2024. Accessed: Feb. 10, 2025. \[Online\].
///     Available: <https://en.wikipedia.org/w/index.php?title=Magnetic_vector_potential&oldid=1259654939#Magnetic_vector_potential>
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
///     Higher Education from Cambridge University Press. Accessed: Feb. 10, 2025. \[Online\].
///     Available: <https://www.cambridge.org/highereducation/books/electricity-and-magnetism/0F97BB6C5D3A56F19B9835EDBEAB087C>
///
/// \[2\] “Magnetic vector potential,” Wikipedia. Nov. 26, 2024. Accessed: Feb. 10, 2025. \[Online\].
///     Available: <https://en.wikipedia.org/w/index.php?title=Magnetic_vector_potential&oldid=1259654939#Magnetic_vector_potential>
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

/// Lorentz body force density (N/m^3) from one oriented circular conductor.
///
/// Source arguments and NaN propagation follow [flux_density_circular_filament_cartesian_scalar].
/// `xyzobs` (m), `jobs` (A/m^2), and the returned JxB vector use world coordinates.
pub fn body_force_density_circular_filament_cartesian_scalar(
    rifil: (f64, f64),
    loc: (f64, f64, f64),
    normal: (f64, f64, f64),
    wire_radius: f64,
    xyzobs: (f64, f64, f64),
    jobs: (f64, f64, f64),
) -> (f64, f64, f64) {
    let b =
        flux_density_circular_filament_cartesian_scalar(rifil, loc, normal, wire_radius, xyzobs);
    let force = Vector3::new(jobs.0, jobs.1, jobs.2).cross(&Vector3::new(b.0, b.1, b.2));
    (force.x, force.y, force.z)
}

/// Lorentz body force densities from independently oriented circular conductors.
///
/// Source arguments follow [flux_density_circular_filament_cartesian]. Observation
/// coordinates (m), current densities `jobs` (A/m^2), and outputs (N/m^3) are
/// component slices in world coordinates, with one entry per observation.
pub fn body_force_density_circular_filament_cartesian(
    rifil: (&[f64], &[f64]),
    loc: (&[f64], &[f64], &[f64]),
    normal: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    xyzobs: (&[f64], &[f64], &[f64]),
    jobs: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    check_length_3tup!(xyzobs.0.len(), &jobs);
    flux_density_circular_filament_cartesian(
        rifil,
        loc,
        normal,
        wire_radius,
        xyzobs,
        (&mut *out.0, &mut *out.1, &mut *out.2),
    )?;
    cartesian_lorentz_force(jobs, out);
    Ok(())
}

/// Parallel version of [body_force_density_circular_filament_cartesian].
pub fn body_force_density_circular_filament_cartesian_par(
    rifil: (&[f64], &[f64]),
    loc: (&[f64], &[f64], &[f64]),
    normal: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    xyzobs: (&[f64], &[f64], &[f64]),
    jobs: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    check_length_3tup!(xyzobs.0.len(), &jobs);
    flux_density_circular_filament_cartesian_par(
        rifil,
        loc,
        normal,
        wire_radius,
        xyzobs,
        (&mut *out.0, &mut *out.1, &mut *out.2),
    )?;
    cartesian_lorentz_force(jobs, out);
    Ok(())
}

fn cartesian_lorentz_force(
    jobs: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) {
    for j in 0..jobs.0.len() {
        let force = Vector3::new(jobs.0[j], jobs.1[j], jobs.2[j])
            .cross(&Vector3::new(out.0[j], out.1[j], out.2[j]));
        out.0[j] = force.x;
        out.1[j] = force.y;
        out.2[j] = force.z;
    }
}

#[cfg(test)]
mod test {
    use core::f64::consts::PI;

    use super::*;
    use crate::{physics::linear_filament::body_force_density_linear_filament, testing::*};

    fn vector_from_tuple(v: (f64, f64, f64)) -> Vector3<f64> {
        Vector3::new(v.0, v.1, v.2)
    }

    #[test]
    fn test_cartesian_pose_covariance() {
        use nalgebra::Rotation3;
        let center = Vector3::new(0.7, -0.4, 1.3);
        for rotation in [
            Rotation3::identity(),
            Rotation3::from_axis_angle(&Vector3::x_axis(), PI),
            Rotation3::from_axis_angle(&Vector3::x_axis(), PI - 1e-9),
            Rotation3::from_axis_angle(&Vector3::x_axis(), 1e-9),
            Rotation3::from_axis_angle(&Vector3::y_axis(), PI / 2.0),
            Rotation3::from_euler_angles(0.4, -0.7, 1.2),
        ] {
            let normal = rotation * Vector3::z();
            for wire_radius in [0.0, 0.01] {
                for local in [
                    Vector3::new(1.003, 0.0, 0.004),
                    Vector3::new(0.998, 0.004, -0.003),
                ] {
                    let world = center + rotation * local;
                    let r = local.x.hypot(local.y);
                    let (br, bz) = flux_density_circular_filament_finite_radius_scalar(
                        (1.0, 0.0, 2.0),
                        wire_radius,
                        (r, local.z),
                    );
                    let local_b = Vector3::new(br * local.x / r, br * local.y / r, bz);
                    let expected = rotation * local_b;
                    let local_j = Vector3::new(1.0, -2.0, 0.3);
                    let world_j = rotation * local_j;
                    let expected_force = rotation * local_j.cross(&local_b);
                    let actual = flux_density_circular_filament_cartesian_scalar(
                        (1.0, 2.0),
                        (center.x, center.y, center.z),
                        (normal.x, normal.y, normal.z),
                        wire_radius,
                        (world.x, world.y, world.z),
                    );
                    let force = body_force_density_circular_filament_cartesian_scalar(
                        (1.0, 2.0),
                        (center.x, center.y, center.z),
                        (normal.x, normal.y, normal.z),
                        wire_radius,
                        (world.x, world.y, world.z),
                        (world_j.x, world_j.y, world_j.z),
                    );
                    assert!(
                        (vector_from_tuple(actual) - expected).norm() < 2e-10 * expected.norm()
                    );
                    assert!(
                        (vector_from_tuple(force) - expected_force).norm()
                            < 2e-10 * expected_force.norm()
                    );
                }
            }
            // Loop-center field independently fixes the orientation and current sign.
            for scale in [1e-300, 1.0, 1e300] {
                let n = scale * normal;
                let actual = flux_density_circular_filament_cartesian_scalar(
                    (1.0, 2.0),
                    (center.x, center.y, center.z),
                    (n.x, n.y, n.z),
                    0.0,
                    (center.x, center.y, center.z),
                );
                assert!((vector_from_tuple(actual) - MU_0 * normal).norm() < 1e-14 * MU_0);
            }
            let world = center + rotation * Vector3::new(1.0, 0.0, 0.0);
            let b = flux_density_circular_filament_cartesian_scalar(
                (1.0, 2.0),
                (center.x, center.y, center.z),
                (normal.x, normal.y, normal.z),
                0.01,
                (world.x, world.y, world.z),
            );
            assert!([b.0, b.1, b.2].iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_cartesian_pose_against_biot_savart() {
        let rotation = nalgebra::Rotation3::from_euler_angles(0.4, -0.7, 1.2);
        let center = Vector3::new(0.7, -0.4, 1.3);
        let normal = rotation * Vector3::z();
        let obs = Vector3::new(1.2, 0.3, 2.0);
        let mut expected = Vector3::zeros();
        let dphi = 2.0 * PI / 8192.0;
        for i in 0..8192 {
            let phi = (i as f64 + 0.5) * dphi;
            let source = center + rotation * Vector3::new(phi.cos(), phi.sin(), 0.0);
            let dl = rotation * Vector3::new(-phi.sin(), phi.cos(), 0.0) * dphi;
            let offset = obs - source;
            expected += MU0_OVER_4PI * 2.0 * dl.cross(&offset) / offset.norm().powi(3);
        }
        let actual = flux_density_circular_filament_cartesian_scalar(
            (1.0, 2.0),
            (center.x, center.y, center.z),
            (normal.x, normal.y, normal.z),
            0.0,
            (obs.x, obs.y, obs.z),
        );
        // Elliptic-integral approximation accuracy dominates the quadrature error.
        assert!((vector_from_tuple(actual) - expected).norm() < 2e-7 * expected.norm());
        let reversed = flux_density_circular_filament_cartesian_scalar(
            (1.0, 2.0),
            (center.x, center.y, center.z),
            (-normal.x, -normal.y, -normal.z),
            0.0,
            (obs.x, obs.y, obs.z),
        );
        assert!(
            (vector_from_tuple(reversed) + vector_from_tuple(actual)).norm()
                < 1e-12 * expected.norm()
        );
    }

    #[test]
    fn test_cartesian_pose_sources_and_force() {
        let radii = [1.0, 0.8, 0.5];
        let current = [1.0, -2.0, 0.5];
        let centers = (
            &[0.0, 1.0, 1.0][..],
            &[0.0, -0.8, 0.0][..],
            &[0.0, 0.0, -0.6][..],
        );
        let normals = (
            &[0.0, 2.0, 0.0][..],
            &[0.0, 0.0, -3.0][..],
            &[1.0, 0.0, 0.0][..],
        );
        let wire = [0.01, 0.02, 0.0];
        let x = [1.0, 1.003, 0.998];
        let y = [0.0, 0.002, -0.003];
        let z = [0.0, 0.004, -0.002];
        for n in [0, 1, 3] {
            let obs = (&x[..n], &y[..n], &z[..n]);
            let jobs = (
                &[1.0, -2.0, 0.3][..n],
                &[0.7, 0.0, -0.4][..n],
                &[-0.3, 0.5, 0.0][..n],
            );
            let expected: Vec<_> = (0..n)
                .map(|j| {
                    (0..3)
                        .map(|i| {
                            vector_from_tuple(flux_density_circular_filament_cartesian_scalar(
                                (radii[i], current[i]),
                                (centers.0[i], centers.1[i], centers.2[i]),
                                (normals.0[i], normals.1[i], normals.2[i]),
                                wire[i],
                                (x[j], y[j], z[j]),
                            ))
                        })
                        .fold(Vector3::zeros(), |a, b| a + b)
                })
                .collect();
            for calc in [
                flux_density_circular_filament_cartesian,
                flux_density_circular_filament_cartesian_par,
            ] {
                let (mut bx, mut by, mut bz) = (vec![99.0; n], vec![99.0; n], vec![99.0; n]);
                calc(
                    (&radii, &current),
                    centers,
                    normals,
                    &wire,
                    obs,
                    (&mut bx, &mut by, &mut bz),
                )
                .unwrap();
                for j in 0..n {
                    assert_eq!(Vector3::new(bx[j], by[j], bz[j]), expected[j]);
                    assert!(expected[j].iter().all(|v| v.is_finite()));
                }
            }
            for calc in [
                body_force_density_circular_filament_cartesian,
                body_force_density_circular_filament_cartesian_par,
            ] {
                let (mut fx, mut fy, mut fz) = (vec![99.0; n], vec![99.0; n], vec![99.0; n]);
                calc(
                    (&radii, &current),
                    centers,
                    normals,
                    &wire,
                    obs,
                    jobs,
                    (&mut fx, &mut fy, &mut fz),
                )
                .unwrap();
                for j in 0..n {
                    let force = Vector3::new(jobs.0[j], jobs.1[j], jobs.2[j]).cross(&expected[j]);
                    assert_eq!(Vector3::new(fx[j], fy[j], fz[j]), force);
                }
            }
        }
    }

    #[test]
    fn test_cartesian_pose_shape_validation() {
        let good = (&[0.0, 0.0][..], &[0.0, 0.0][..], &[1.0, 1.0][..]);
        let bad_length = (&[0.0][..], good.1, good.2);
        for n in [0, 2] {
            let obs = (&[0.0, 0.0][..n], &[0.0, 0.0][..n], &[0.0, 0.0][..n]);
            for (centers, normals, wire) in [
                (bad_length, good, &[0.0, 0.01][..]),
                (good, bad_length, &[0.0, 0.01][..]),
                (good, good, &[0.0][..]),
            ] {
                for calc in [
                    flux_density_circular_filament_cartesian,
                    flux_density_circular_filament_cartesian_par,
                ] {
                    let (mut x, mut y, mut z) = (vec![99.0; n], vec![99.0; n], vec![99.0; n]);
                    assert!(
                        calc(
                            (&[1.0; 2], &[1.0; 2]),
                            centers,
                            normals,
                            wire,
                            obs,
                            (&mut x, &mut y, &mut z)
                        )
                        .is_err()
                    );
                    assert!(x.iter().chain(&y).chain(&z).all(|v| *v == 99.0));
                }
                for calc in [
                    body_force_density_circular_filament_cartesian,
                    body_force_density_circular_filament_cartesian_par,
                ] {
                    let (mut x, mut y, mut z) = (vec![99.0; n], vec![99.0; n], vec![99.0; n]);
                    assert!(
                        calc(
                            (&[1.0; 2], &[1.0; 2]),
                            centers,
                            normals,
                            wire,
                            obs,
                            obs,
                            (&mut x, &mut y, &mut z)
                        )
                        .is_err()
                    );
                    assert!(x.iter().chain(&y).chain(&z).all(|v| *v == 99.0));
                }
            }
        }
    }

    #[test]
    fn test_cartesian_pose_nan_propagation() {
        let zero = (0.0, 0.0, 0.0);
        let axis = (0.0, 0.0, 1.0);
        for wire_radius in [0.0, 0.01] {
            for (loc, normal, obs) in [
                (zero, zero, axis),
                (zero, (f64::NAN, 0.0, 1.0), axis),
                (zero, (0.0, f64::INFINITY, 1.0), axis),
                ((0.0, 0.0, f64::NAN), axis, axis),
                ((f64::INFINITY, 0.0, 0.0), axis, axis),
                (zero, axis, (0.0, f64::NAN, 1.0)),
            ] {
                let b = flux_density_circular_filament_cartesian_scalar(
                    (1.0, 1.0),
                    loc,
                    normal,
                    wire_radius,
                    obs,
                );
                let force = body_force_density_circular_filament_cartesian_scalar(
                    (1.0, 1.0),
                    loc,
                    normal,
                    wire_radius,
                    obs,
                    axis,
                );
                assert!(
                    [b.0, b.1, b.2, force.0, force.1, force.2]
                        .iter()
                        .all(|v| v.is_nan())
                );
            }
        }
    }

    #[test]
    fn test_potential_and_flux_zero_radius_against_preserved_kernels() {
        for scale in [1e-6, 1.0, 1e6] {
            let rfil = [0.5 * scale, scale, 2.0 * scale];
            let zfil = [-0.2 * scale, 0.3 * scale, 0.8 * scale];
            let current = [2.0, -3.0, 0.7];
            let robs = [0.0, 0.2 * scale, 1.003 * scale, 2.5 * scale];
            let zobs = [0.5 * scale, -0.7 * scale, 0.305 * scale, 4.0 * scale];
            for nfils in [0, 1, 3] {
                let source = (&rfil[..nfils], &zfil[..nfils], &current[..nfils]);
                for nobs in [0, 1, 4] {
                    let obs = (&robs[..nobs], &zobs[..nobs]);
                    let mut old_a = vec![99.0; nobs];
                    let mut old_flux = vec![99.0; nobs];
                    vector_potential_circular_filament_thin(source, obs, &mut old_a).unwrap();
                    flux_circular_filament_thin(source, obs, &mut old_flux).unwrap();
                    for radius in [0.0, -0.0] {
                        let radii = vec![radius; nfils];
                        for calc in [
                            vector_potential_circular_filament,
                            vector_potential_circular_filament_par,
                        ] {
                            let mut actual = vec![99.0; nobs];
                            calc(source, &radii, obs, &mut actual).unwrap();
                            for (a, old) in actual.iter().zip(&old_a) {
                                assert_eq!(a.to_bits(), old.to_bits());
                            }
                        }
                        for calc in [flux_circular_filament, flux_circular_filament_par] {
                            let mut actual = vec![99.0; nobs];
                            calc(source, &radii, obs, &mut actual).unwrap();
                            for (flux, old) in actual.iter().zip(&old_flux) {
                                // Scaling after summation changes rounding; preserve
                                // the old formula's accuracy and axis singularity.
                                assert!(
                                    (flux.is_nan() && old.is_nan())
                                        || approx(*flux, *old, 5e-12, 1e-18 * scale)
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_potential_and_flux_mixed_radius_sources() {
        let source = (
            &[1.0, 1.0, 0.4][..],
            &[0.0, 0.0, -0.3][..],
            &[1.0, -2.0, 0.5][..],
        );
        let radii = [0.01, 0.02, 0.0];
        let robs = [1.0, 1.003, 1.015];
        let zobs = [0.0, 0.004, 0.02];
        for nobs in [0, 1, 3] {
            let obs = (&robs[..nobs], &zobs[..nobs]);
            let expected: Vec<f64> = (0..nobs)
                .map(|j| {
                    (0..3)
                        .map(|i| {
                            vector_potential_circular_filament_finite_thickness_scalar(
                                (source.0[i], source.1[i], source.2[i]),
                                radii[i],
                                (robs[j], zobs[j]),
                            )
                        })
                        .sum()
                })
                .collect();
            for calc in [
                vector_potential_circular_filament,
                vector_potential_circular_filament_par,
            ] {
                let mut actual = vec![99.0; nobs];
                calc(source, &radii, obs, &mut actual).unwrap();
                assert_eq!(actual, expected);
                assert!(actual.iter().all(|a| a.is_finite()));
            }
            for calc in [flux_circular_filament, flux_circular_filament_par] {
                let mut actual = vec![99.0; nobs];
                calc(source, &radii, obs, &mut actual).unwrap();
                for j in 0..nobs {
                    assert_eq!(actual[j], 2.0 * PI * robs[j] * expected[j]);
                    let scalar_sum: f64 = (0..3)
                        .map(|i| {
                            flux_circular_filament_scalar(
                                (source.0[i], source.1[i], source.2[i]),
                                radii[i],
                                (robs[j], zobs[j]),
                            )
                        })
                        .sum();
                    assert!(approx(actual[j], scalar_sum, 1e-14, 0.0));
                }
            }
        }
    }

    #[test]
    fn test_potential_and_flux_length_validation() {
        for calc in [
            vector_potential_circular_filament,
            vector_potential_circular_filament_par,
            flux_circular_filament,
            flux_circular_filament_par,
        ] {
            for nobs in [0, 2] {
                for radii in [&[][..], &[0.01][..], &[0.01, 0.02, 0.03][..]] {
                    let mut out = vec![99.0; nobs];
                    assert!(
                        calc(
                            (&[1.0; 2], &[0.0; 2], &[1.0; 2]),
                            radii,
                            (&vec![1.0; nobs], &vec![0.0; nobs]),
                            &mut out
                        )
                        .is_err()
                    );
                    assert_eq!(out, vec![99.0; nobs]);
                }
                let mut out = vec![99.0; nobs];
                assert!(
                    calc(
                        (&[1.0], &[0.0; 2], &[1.0; 2]),
                        &[0.0; 2],
                        (&vec![1.0; nobs], &vec![0.0; nobs]),
                        &mut out
                    )
                    .is_err()
                );
                assert_eq!(out, vec![99.0; nobs]);
            }
            for (robs, zobs, nout) in [
                (&[1.0][..], &[][..], 1),
                (&[][..], &[0.0][..], 1),
                (&[1.0][..], &[0.0][..], 2),
            ] {
                let mut out = vec![99.0; nout];
                assert!(calc((&[1.0], &[0.0], &[1.0]), &[0.01], (robs, zobs), &mut out).is_err());
                assert_eq!(out, vec![99.0; nout]);
            }
        }
    }

    #[test]
    fn test_finite_thickness_potential_zero_matches_ideal() {
        for scale in [1e-6, 1.0, 1e6] {
            for current in [-3.0, 0.0, 2.0] {
                let filament = (scale, -0.4 * scale, current);
                for (r, z) in [
                    (0.0, 0.3),
                    (1e-5, -0.4),
                    (0.5, 0.7),
                    (1.0, -0.4),
                    (1.001, -0.398),
                    (10.0, 20.0),
                ] {
                    let obs = (r * scale, z * scale);
                    let expected = vector_potential_circular_filament_scalar(filament, obs);
                    for radius in [0.0, -0.0] {
                        let actual = vector_potential_circular_filament_finite_thickness_scalar(
                            filament, radius, obs,
                        );
                        // Preserve all existing behavior, including NaN on the
                        // symmetry axis and the ideal source singularity.
                        assert_eq!(actual.to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn test_finite_thickness_potential_against_cross_section_integral() {
        // Uniform-J disk average of exact unit-current loop potentials, a=I=1,
        // normalized by mu0*I/(4*pi). SciPy ellipkm1(d2/Q), ellipe(1-d2/Q),
        // 256-point Gauss-Legendre radial quadrature, 1024 midpoint angles.
        // Interior quadrature uses observation-centered polar coordinates to
        // integrate the logarithmic singularity; exterior uses disk-centered
        // coordinates. Doubling orders from 128/512 changes results by <1e-7.
        let cases = [
            (0.01, 0.0, 0.0, 10.3693163884398),
            (0.01, 0.3, 0.4, 10.1085274488821),
            (0.01, -0.5, 0.5, 9.88728781974263),
            (0.01, 0.0, 1.0, 9.36953373854208),
            (0.01, 0.9, 1.2, 8.53051493483393),
            (0.01, 1.2, 0.0, 8.96535750309147),
            (0.1, 0.0, 0.0, 5.77046965840007),
            (0.1, 0.3, 0.4, 5.48459527448686),
            (0.1, -0.5, 0.5, 5.34282513259251),
            (0.1, 0.0, 1.0, 4.78355172239662),
            (0.1, 0.9, 1.2, 3.90920314136678),
            (0.1, 1.2, 0.0, 4.30512798795341),
        ];
        for (b, u, v, expected) in cases {
            let actual = vector_potential_circular_filament_finite_thickness_scalar(
                (1.0, 0.0, 1.0),
                b,
                (1.0 + b * u, b * v),
            ) / MU0_OVER_4PI;
            // O((b/a)^2) model error: 0.01% at b/a=.01, 1% at b/a=.1.
            assert!(
                (actual - expected).abs() < b * b * expected,
                "b={b}, u={u}, v={v}, actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_finite_thickness_potential_surface_continuity() {
        let b = 0.01;
        let h = b * 1e-5;
        for i in 0..16 {
            let theta = 2.0 * PI * i as f64 / 16.0;
            let potential = |s: f64| {
                vector_potential_circular_filament_finite_thickness_scalar(
                    (1.0, 0.0, 1.0),
                    b,
                    (1.0 + s * theta.cos(), s * theta.sin()),
                )
            };
            let (inside, surface, outside) = (potential(b - h), potential(b), potential(b + h));
            assert!((inside - surface).abs() < 3e-5 * MU0_OVER_4PI);
            assert!((outside - surface).abs() < 3e-5 * MU0_OVER_4PI);
            let derivative_inside = (surface - inside) / h;
            let derivative_outside = (outside - surface) / h;
            assert!((derivative_inside - derivative_outside).abs() < 1e-4 * MU0_OVER_4PI / b);
        }
    }

    #[test]
    fn test_finite_thickness_potential_curl_matches_field_to_retained_order() {
        for b in [0.01_f64, 0.001] {
            let h = b * 1e-4;
            for (u, v) in [(0.0, 0.0), (0.3, 0.4), (-0.5, 0.5), (0.9, 1.2)] {
                let (r, z) = (1.0 + b * u, b * v);
                let potential = |r, z| {
                    vector_potential_circular_filament_finite_thickness_scalar(
                        (1.0, 0.0, 1.0),
                        b,
                        (r, z),
                    )
                };
                let br = -(potential(r, z + h) - potential(r, z - h)) / (2.0 * h);
                let bz =
                    ((r + h) * potential(r + h, z) - (r - h) * potential(r - h, z)) / (2.0 * h * r);
                let expected =
                    flux_density_circular_filament_finite_radius_scalar((1.0, 0.0, 1.0), b, (r, z));
                // The two asymptotic models retain different higher-order
                // terms. Their curl agreement is O(b*ln(8/b)) in these units,
                // or O(b^2*ln(8/b)) relative to the local cylinder field.
                let tol = 2.0 * MU0_OVER_4PI * b * (8.0 / b).ln();
                assert!((br - expected.0).hypot(bz - expected.1) < tol);
            }
        }
    }

    #[test]
    fn test_finite_thickness_potential_centerline_and_scaling() {
        for b in [0.01, 1e-8, 1e-12] {
            let centerline = vector_potential_circular_filament_finite_thickness_scalar(
                (1.0, 0.0, 1.0),
                b,
                (1.0, 0.0),
            );
            assert!(centerline.is_finite());
            assert!((centerline / MU0_OVER_4PI - (2.0 * (8.0 / b).ln() - 3.0)).abs() < 1e-13);
        }
        for (u, v) in [(0.0, 0.0), (0.3, 0.4), (-0.9, 1.2)] {
            let base = vector_potential_circular_filament_finite_thickness_scalar(
                (1.0, 0.0, 1.0),
                0.01,
                (1.0 + 0.01 * u, 0.01 * v),
            );
            for scale in [1e-6, 1.0, 1e6] {
                for current in [-3.0, 0.0, 2.0] {
                    // A is invariant under uniform geometric scaling and
                    // even under reflection about the loop plane.
                    let actual = vector_potential_circular_filament_finite_thickness_scalar(
                        (scale, 0.7 * scale, current),
                        0.01 * scale,
                        ((1.0 + 0.01 * u) * scale, (0.7 - 0.01 * v) * scale),
                    );
                    assert!((actual - current * base).abs() < 1e-12 * base.abs());
                }
            }
        }
    }

    #[test]
    fn test_zero_radius_vectors_against_preserved_thin_kernel() {
        for scale in [1e-6, 1.0, 1e6] {
            let rfil = [0.5 * scale, scale, 2.0 * scale];
            let zfil = [0.0, scale, -0.5 * scale];
            let current = [2.0, -3.0, 0.7];
            let radii = [0.0; 3];
            let robs = [0.0, 1e-6, 0.1, 1e-4, 2e-4, 1.0, 0.0].map(|r| r * scale);
            let zobs = [0.25 * scale; 7];
            for nsrc in [0, 1, 3] {
                for nobs in [0, 1, 7] {
                    let source = (&rfil[..nsrc], &zfil[..nsrc], &current[..nsrc]);
                    let obs = (&robs[..nobs], &zobs[..nobs]);
                    let (mut expected_r, mut expected_z) = (vec![1.0; nobs], vec![2.0; nobs]);
                    flux_density_circular_filament_thin(
                        source,
                        obs,
                        (&mut expected_r, &mut expected_z),
                    )
                    .unwrap();
                    for calc in [
                        flux_density_circular_filament,
                        flux_density_circular_filament_par,
                    ] {
                        let (mut br, mut bz) = (vec![3.0; nobs], vec![4.0; nobs]);
                        calc(source, &radii[..nsrc], obs, (&mut br, &mut bz)).unwrap();
                        assert_eq!(br, expected_r);
                        assert_eq!(bz, expected_z);
                    }
                }
            }
        }
    }

    #[test]
    fn test_finite_radius_vectors_sum_per_source_scalars() {
        let rfil = [1.0, 1.003, 0.998];
        let zfil = [0.0, 0.002, -0.001];
        let current = [1.0, -2.0, 0.5];
        let radii = [0.01, 0.02, 0.0];
        let robs = [1.0, 1.006, 0.995, 1.012];
        let zobs = [0.0, 0.004, 0.003, -0.001];
        for nobs in [1, 4] {
            for calc in [
                flux_density_circular_filament,
                flux_density_circular_filament_par,
            ] {
                let (mut br, mut bz) = (vec![1.0; nobs], vec![2.0; nobs]);
                calc(
                    (&rfil, &zfil, &current),
                    &radii,
                    (&robs[..nobs], &zobs[..nobs]),
                    (&mut br, &mut bz),
                )
                .unwrap();
                for j in 0..nobs {
                    let mut expected = (0.0, 0.0);
                    for i in 0..rfil.len() {
                        let field = flux_density_circular_filament_finite_radius_scalar(
                            (rfil[i], zfil[i], current[i]),
                            radii[i],
                            (robs[j], zobs[j]),
                        );
                        expected.0 += field.0;
                        expected.1 += field.1;
                    }
                    assert_eq!((br[j], bz[j]), expected);
                }
            }
        }
    }

    #[test]
    fn test_finite_radius_vector_length_validation() {
        for calc in [
            flux_density_circular_filament,
            flux_density_circular_filament_par,
        ] {
            let source = (&[1.0][..], &[0.0][..], &[2.0][..]);
            let (mut br, mut bz) = ([3.0; 2], [4.0; 2]);
            // Validate before modifying outputs, including in the parallel wrapper.
            assert!(calc(source, &[], (&[0.0; 2], &[0.0; 2]), (&mut br, &mut bz)).is_err());
            assert!(calc(source, &[0.01], (&[0.0; 2], &[0.0]), (&mut br, &mut bz)).is_err());
            assert!(calc(source, &[0.01], (&[0.0], &[0.0]), (&mut br, &mut bz)).is_err());
            assert!(
                calc(
                    (&[], &[0.0], &[2.0]),
                    &[0.01],
                    (&[0.0; 2], &[0.0; 2]),
                    (&mut br, &mut bz)
                )
                .is_err()
            );
            assert_eq!(br, [3.0; 2]);
            assert_eq!(bz, [4.0; 2]);
            // Empty observation arrays must not bypass source/radius validation.
            assert!(calc(source, &[], (&[], &[]), (&mut [], &mut [])).is_err());
        }
    }

    #[test]
    fn test_finite_radius_zero_matches_ideal_field() {
        let filament = (2.0, -0.4, -3.0);
        for wire_radius in [0.0, -0.0] {
            for obs in [(0.0, 0.3), (1e-5, -0.4), (1.0, 0.7), (2.1, -0.39)] {
                let expected = if obs.0 <= ON_AXIS_RADIUS_RATIO * filament.0 {
                    flux_density_circular_filament_on_axis(filament, obs.1)
                } else {
                    flux_density_circular_filament_off_axis(filament, obs)
                };
                assert_eq!(
                    flux_density_circular_filament_finite_radius_scalar(filament, wire_radius, obs),
                    expected
                );
            }
            let source = flux_density_circular_filament_finite_radius_scalar(
                filament,
                wire_radius,
                (filament.0, filament.1),
            );
            assert!(!source.0.is_finite() || !source.1.is_finite());
        }
    }

    #[test]
    fn test_finite_radius_against_hurwitz_equations() {
        let (a, zfil, current) = (2.0, -0.4, 3.0);
        for aspect in [0.02, 0.1] {
            let b = aspect * a;
            // Direct periodic quadrature of equation 17, independent of the
            // closed elliptic-integral reduction used by the scalar kernel.
            let n = 65536;
            let dphi = 2.0 * PI / n as f64;
            let bz_reg = MU0_OVER_4PI
                * current
                * dphi
                * (0..n)
                    .map(|i| {
                        let phi = (i as f64 + 0.5) * dphi;
                        let numerator = a * a * (1.0 - phi.cos());
                        numerator
                            / (2.0 * numerator + b * b / core::f64::consts::E.sqrt()).powf(1.5)
                    })
                    .sum::<f64>();
            for distance in [0.0, 0.3, 0.999, 1.0, 1.2, 2.0] {
                let s = distance * b;
                for theta in [0.0_f64, 0.7, 2.3, PI, 4.5] {
                    // Paper's normal points inward: e2 = -eR, e3 = eZ.
                    let obs = (a - s * theta.cos(), zfil + s * theta.sin());
                    let cylinder =
                        2.0 * MU0_OVER_4PI * current * if s <= b { s / (b * b) } else { 1.0 / s };
                    let curvature = MU0_OVER_4PI * current / (2.0 * a);
                    let (b2, b3) = if s <= b {
                        (
                            -s * s * (2.0 * theta).sin() / (2.0 * b * b),
                            1.5 + s * s / (b * b) * ((2.0 * theta).cos() / 2.0 - 1.0),
                        )
                    } else {
                        (
                            (b * b / (2.0 * s * s) - 1.0) * (2.0 * theta).sin(),
                            0.5 - 2.0 * (s / b).ln()
                                + (1.0 - b * b / (2.0 * s * s)) * (2.0 * theta).cos(),
                        )
                    };
                    let expected = (
                        cylinder * theta.sin() - curvature * b2,
                        bz_reg + cylinder * theta.cos() + curvature * b3,
                    );
                    let actual = flux_density_circular_filament_finite_radius_scalar(
                        (a, zfil, current),
                        b,
                        obs,
                    );
                    // Absolute tolerance accounts for the elliptic-integral fits.
                    let tol = 8e-8 * MU0_OVER_4PI * current / a;
                    assert!((actual.0 - expected.0).abs() < tol);
                    assert!((actual.1 - expected.1).abs() < tol);
                }
            }
        }
    }

    #[test]
    fn test_finite_radius_against_cross_section_integral() {
        // Independent uniform-J disk integration of unit-current circular loops
        // (a = I = 1), normalized by mu0*I/(4*pi*a). Observation-centered polar
        // coordinates remove the 1/distance singularity via the area Jacobian.
        // Reference: SciPy ellipkm1(d2/q), ellipe(1-d2/q), 256-point Gauss–Legendre
        // radial quadrature and 1024 midpoint angles. Doubling both orders from
        // 128/512 changed these vectors by less than 4e-10 relatively.
        let cases = [
            (0.01, 0.0, 0.0, 0.0, 6.68459083826242),
            (0.01, 0.5, 0.0, 0.0, -93.3998252276275),
            (0.01, 0.0, 0.5, 99.9780058466979, 6.49706713001394),
            (0.01, 0.5, 0.5, 99.853938832051, -93.586570635344),
            (0.01, -0.5, 0.5, 100.103633652062, 106.455855446888),
            (0.1, 0.0, 0.0, 0.0, 4.38065660604968),
            (0.1, 0.5, 0.0, 0.0, -5.81090160130197),
            (0.1, 0.0, 0.5, 9.86653941687754, 4.19186834520168),
            (0.1, 0.5, 0.5, 9.7576610149033, -5.99207737754346),
            (0.1, -0.5, 0.5, 9.99007518217192, 14.2593819318384),
        ];
        for (b, u, v, br, bz) in cases {
            let actual = flux_density_circular_filament_finite_radius_scalar(
                (1.0, 0.0, 1.0),
                b,
                (1.0 + b * u, b * v),
            );
            let error = (actual.0 / MU0_OVER_4PI - br).hypot(actual.1 / MU0_OVER_4PI - bz);
            // Model truncation, rather than quadrature or floating-point error,
            // sets these tolerances: 0.03% at b/a=.01 and 3% at b/a=.1.
            let rtol = if b == 0.01 { 3e-4 } else { 0.03 };
            assert!(error < rtol * br.hypot(bz), "b={b}, u={u}, v={v}");
        }
    }

    #[test]
    fn test_finite_radius_surface_continuity() {
        let b = 0.01;
        for i in 0..16 {
            let theta = 2.0 * PI * i as f64 / 16.0;
            let field = |s: f64| {
                flux_density_circular_filament_finite_radius_scalar(
                    (1.0, 0.0, 1.0),
                    b,
                    (1.0 + s * theta.cos(), s * theta.sin()),
                )
            };
            let surface = field(b);
            for s in [b * (1.0 - 1e-8), b * (1.0 + 1e-8)] {
                let nearby = field(s);
                assert!(
                    (nearby.0 - surface.0).hypot(nearby.1 - surface.1)
                        < 1e-7 * surface.0.hypot(surface.1)
                );
            }
        }
    }

    #[test]
    fn test_finite_radius_centerline_and_straight_wire_limit() {
        // These aspect ratios round m to 1 if the complement is not retained.
        for b in [1e-8, 1e-12] {
            for current in [-3.0, 0.0, 2.0] {
                let (br, bz) = flux_density_circular_filament_finite_radius_scalar(
                    (1.0, 0.7, current),
                    b,
                    (1.0, 0.7),
                );
                let expected = MU0_OVER_4PI * current * (8.0 / b).ln();
                assert_eq!(br, 0.0);
                assert!((bz - expected).abs() <= 1e-12 * expected.abs());
            }
        }
        let (br, bz) = flux_density_circular_filament_finite_radius_scalar(
            (1e10, 0.0, 1.0),
            0.1,
            (1e10, 0.03),
        );
        let cylinder = 2.0 * MU0_OVER_4PI * 0.03 / 0.1_f64.powi(2);
        assert!((br - cylinder).abs() < 1e-14 * cylinder);
        assert!(bz.abs() < 1e-9 * cylinder);
    }

    #[test]
    fn test_finite_radius_symmetry_and_scaling() {
        for (u, v) in [(0.3, 0.4), (-0.8, 0.9)] {
            let base = flux_density_circular_filament_finite_radius_scalar(
                (1.0, 0.0, 1.0),
                0.01,
                (1.0 + 0.01 * u, 0.01 * v),
            );
            for scale in [1e-6, 1.0, 1e6] {
                for current in [-3.0, 0.0, 2.0] {
                    // Include a z-translation and reflection about the loop plane.
                    let actual = flux_density_circular_filament_finite_radius_scalar(
                        (scale, 0.7 * scale, current),
                        0.01 * scale,
                        ((1.0 + 0.01 * u) * scale, (0.7 - 0.01 * v) * scale),
                    );
                    let expected = (-base.0 * current / scale, base.1 * current / scale);
                    let tol = 1e-12 * expected.0.hypot(expected.1);
                    assert!((actual.0 - expected.0).hypot(actual.1 - expected.1) <= tol);
                }
            }
        }
    }

    #[test]
    fn test_flux_density_on_axis() {
        for radius in [1e-6, 0.3, 2.0, 1e6] {
            let zfil = 0.7 * radius;
            for current in [-3.0, 0.0, 2.0] {
                for offset in [-1e4, -2.0, 0.0, 2.0, 1e4] {
                    let zobs = zfil + offset * radius;
                    let dz = zobs - zfil;
                    let expected = MU_0 * current * radius * radius
                        / (2.0 * (radius * radius + dz * dz).powf(1.5));
                    let on_axis =
                        flux_density_circular_filament_on_axis((radius, zfil, current), zobs);
                    for robs in [0.0, -0.0] {
                        let (br, bz) = flux_density_circular_filament_scalar(
                            (radius, zfil, current),
                            (robs, zobs),
                        );
                        assert_eq!((br, bz), on_axis);
                        assert_eq!(br, 0.0);
                        assert!(
                            (bz - expected).abs() <= 2e-15 * expected.abs(),
                            "{bz} != {expected}"
                        );
                        let (bx, by, bz) = flux_density_circular_filament_cartesian_scalar(
                            (radius, current),
                            (0.0, 0.0, zfil),
                            (0.0, 0.0, 1.0),
                            0.0,
                            (robs, robs, zobs),
                        );
                        assert_eq!(bx, 0.0);
                        assert_eq!(by, 0.0);
                        assert!((bz - expected).abs() <= 2e-15 * expected.abs());
                    }
                }
            }
        }
    }

    #[test]
    fn test_flux_density_axis_cutoff() {
        for radius in [1e-6, 0.3, 2.0, 1e6] {
            let cutoff = 1e-4 * radius;
            let filament = (radius, 0.7 * radius, -3.0);
            for dz in [-radius, 0.0, radius] {
                let zobs = filament.1 + dz;
                let on_axis = flux_density_circular_filament_on_axis(filament, zobs);
                for robs in [0.0, 100.0 * f64::EPSILON * radius, 0.5 * cutoff, cutoff] {
                    let field = flux_density_circular_filament_scalar(filament, (robs, zobs));
                    assert_eq!(field, on_axis);
                    let field_xyz = {
                        let (r, z, current) = filament;
                        flux_density_circular_filament_cartesian_scalar(
                            (r, current),
                            (0.0, 0.0, z),
                            (0.0, 0.0, 1.0),
                            0.0,
                            (robs, 0.0, zobs),
                        )
                    };
                    assert_eq!(field_xyz, (0.0, 0.0, on_axis.1));
                }
                let robs = cutoff.next_up();
                let field = flux_density_circular_filament_scalar(filament, (robs, zobs));
                assert_eq!(
                    field,
                    flux_density_circular_filament_off_axis(filament, (robs, zobs))
                );
                if dz != 0.0 {
                    assert_ne!(field.0, 0.0);
                }
            }

            // A relative cutoff must not regularize the source, even for tiny loops.
            let field = flux_density_circular_filament_scalar(filament, (radius, filament.1));
            assert!(!field.0.is_finite() || !field.1.is_finite());
        }
    }

    #[test]
    fn test_flux_density_mixed_axis_cutoffs() {
        let rfil = [0.25, 2.0];
        let zfil = [-0.5, 0.75];
        let ifil = [2.0, -3.0];
        let robs = [0.0, 1e-6, 2.5e-5, 1e-4, 2e-4, 1e-3, 0.1, 1e-5];
        let zobs = [1.0; 8];
        let mut br = [1.0; 8];
        let mut bz = [1.0; 8];
        let mut br_par = [2.0; 8];
        let mut bz_par = [2.0; 8];
        let sources = (&rfil[..], &zfil[..], &ifil[..]);
        flux_density_circular_filament(sources, &[0.0; 2], (&robs, &zobs), (&mut br, &mut bz))
            .unwrap();
        flux_density_circular_filament_par(
            sources,
            &[0.0; 2],
            (&robs, &zobs),
            (&mut br_par, &mut bz_par),
        )
        .unwrap();
        for j in 0..robs.len() {
            let mut expected = (0.0, 0.0);
            for i in 0..rfil.len() {
                let field = flux_density_circular_filament_scalar(
                    (rfil[i], zfil[i], ifil[i]),
                    (robs[j], zobs[j]),
                );
                expected.0 += field.0;
                expected.1 += field.1;
            }
            assert_eq!((br[j], bz[j]), expected);
            assert_eq!((br_par[j], bz_par[j]), expected);
        }
    }

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
        {
            let (r, z, current) = rzifil;
            let zero = vec![0.0; r.len()];
            let one = vec![1.0; r.len()];
            body_force_density_circular_filament_cartesian(
                (r, current),
                (&zero, &zero, z),
                (&zero, &zero, &one),
                &vec![0.0; rzifil.2.len()],
                xyzobs,
                j_vec,
                (outx, outy, outz),
            )
        }
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
                &vec![1.0; n - 1][..],
                &vec![0.0; n - 1][..],
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
        {
            let (r, z, current) = rzifil;
            let zero = vec![0.0; r.len()];
            let one = vec![1.0; r.len()];
            flux_density_circular_filament_cartesian(
                (r, current),
                (&zero, &zero, z),
                (&zero, &zero, &one),
                &vec![0.0; rzifil.2.len()],
                xyzobs,
                (bx0, by0, bz0),
            )
        }
        .unwrap();

        let (bx1, by1, bz1) = (&mut x.clone()[..], &mut x.clone()[..], &mut x.clone()[..]);
        {
            let (r, z, current) = rzifil;
            let zero = vec![0.0; r.len()];
            let one = vec![1.0; r.len()];
            flux_density_circular_filament_cartesian_par(
                (r, current),
                (&zero, &zero, z),
                (&zero, &zero, &one),
                &vec![0.0; rzifil.2.len()],
                xyzobs,
                (bx1, by1, bz1),
            )
        }
        .unwrap();

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
                &vec![0.0; ifil.len()],
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
            let wire_radius = vec![0.0; ndiscr - 1];
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
                    &wire_radius,
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

            vector_potential_circular_filament(
                (&[rfil], &[zfil], &[1.0]),
                &[0.0],
                (&[r], &[z]),
                &mut out,
            )
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
                    &[0.0],
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
                flux_circular_filament(
                    (&[rfil], &[zfil], &[1.0]),
                    &[0.0],
                    (&[*r], &[*z]),
                    &mut psi,
                )
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
        let ifil: Vec<f64> = (0..NFIL).map(|i| i as f64).collect();

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
        flux_circular_filament(
            (&rfil, &zfil, &ifil),
            &[0.0; NFIL],
            (&rprime, &zprime),
            out0,
        )
        .unwrap();
        flux_circular_filament_par(
            (&rfil, &zfil, &ifil),
            &[0.0; NFIL],
            (&rprime, &zprime),
            out1,
        )
        .unwrap();
        for i in 0..NOBS {
            assert_eq!(out0[i], out1[i]);
        }

        // Flux density
        flux_density_circular_filament(
            (&rfil, &zfil, &ifil),
            &[0.0; NFIL],
            (&rprime, &zprime),
            (out0, out1),
        )
        .unwrap();
        flux_density_circular_filament_par(
            (&rfil, &zfil, &ifil),
            &[0.0; NFIL],
            (&rprime, &zprime),
            (out2, out3),
        )
        .unwrap();
        for i in 0..NOBS {
            assert_eq!(out0[i], out2[i]);
            assert_eq!(out1[i], out3[i]);
        }

        // Vector potential
        let out0 = &mut [0.0; NOBS]; // Reinit with different values to test zeroing
        let out1 = &mut [1.0; NOBS];
        vector_potential_circular_filament(
            (&rfil, &zfil, &ifil),
            &[0.0; NFIL],
            (&rprime, &zprime),
            out0,
        )
        .unwrap();
        vector_potential_circular_filament_par(
            (&rfil, &zfil, &ifil),
            &[0.0; NFIL],
            (&rprime, &zprime),
            out1,
        )
        .unwrap();
        for i in 0..NOBS {
            assert_eq!(out0[i], out1[i]);
        }
    }
}
