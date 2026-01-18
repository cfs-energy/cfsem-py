//! Limited bindings to Project Rat's rat-mlfmm C++ library.

mod ffi;

use crate::macros::check_length_3tup;
use std::ffi::CStr;

struct Context {
    raw: *mut ffi::RatMlfmmContext,
}

impl Context {
    pub fn new() -> Result<Self, String> {
        let raw = unsafe { ffi::rat_mlfmm_context_create() };
        if raw.is_null() {
            Err(last_error())
        } else {
            Ok(Self { raw })
        }
    }

    pub fn from_options(opts: Option<&MlfmmOptions>) -> Result<Self, String> {
        let mut ctx = Self::new()?;
        ctx.apply_options(opts)?;
        Ok(ctx)
    }

    pub fn set_sources_linear(
        &mut self,
        rs_xyz: (&[f64], &[f64], &[f64]),
        drs_xyz: (&[f64], &[f64], &[f64]),
        currents: &[f64],
        eps: &[f64],
    ) -> Result<(), String> {
        let num_sources = rs_xyz.0.len();
        check_length_3tup_result(num_sources, rs_xyz).map_err(|err| err.to_string())?;
        check_length_3tup_result(num_sources, drs_xyz).map_err(|err| err.to_string())?;
        if currents.len() != num_sources || eps.len() != num_sources {
            return Err("currents and eps must match sources length".to_string());
        }
        let ok = unsafe {
            ffi::rat_mlfmm_context_set_sources_linear(
                self.raw,
                rs_xyz.0.as_ptr(),
                rs_xyz.1.as_ptr(),
                rs_xyz.2.as_ptr(),
                drs_xyz.0.as_ptr(),
                drs_xyz.1.as_ptr(),
                drs_xyz.2.as_ptr(),
                currents.as_ptr(),
                eps.as_ptr(),
                num_sources,
            )
        };
        if ok == 0 { Err(last_error()) } else { Ok(()) }
    }

    pub fn set_targets(&mut self, rt_xyz: (&[f64], &[f64], &[f64])) -> Result<(), String> {
        let num_targets = rt_xyz.0.len();
        check_length_3tup_result(num_targets, rt_xyz).map_err(|err| err.to_string())?;
        let ok = unsafe {
            ffi::rat_mlfmm_context_set_targets(
                self.raw,
                rt_xyz.0.as_ptr(),
                rt_xyz.1.as_ptr(),
                rt_xyz.2.as_ptr(),
                num_targets,
            )
        };
        if ok == 0 { Err(last_error()) } else { Ok(()) }
    }

    pub fn set_van_lanen(&mut self, use_van_lanen: bool) -> Result<(), String> {
        let ok = unsafe { ffi::rat_mlfmm_context_set_van_lanen(self.raw, use_van_lanen as i32) };
        if ok == 0 { Err(last_error()) } else { Ok(()) }
    }

    pub fn set_num_exp(&mut self, num_exp: i32) -> Result<(), String> {
        let ok = unsafe { ffi::rat_mlfmm_context_set_num_exp(self.raw, num_exp) };
        if ok == 0 { Err(last_error()) } else { Ok(()) }
    }

    pub fn set_direct_mode(&mut self, mode: ffi::RatMlfmmDirectMode) -> Result<(), String> {
        let ok = unsafe { ffi::rat_mlfmm_context_set_direct_mode(self.raw, mode) };
        if ok == 0 { Err(last_error()) } else { Ok(()) }
    }

    pub fn set_direct_threshold(&mut self, threshold: f64) -> Result<(), String> {
        let ok = unsafe { ffi::rat_mlfmm_context_set_direct_threshold(self.raw, threshold) };
        if ok == 0 { Err(last_error()) } else { Ok(()) }
    }

    pub fn set_direct_threshold_count(&mut self, threshold: u64) -> Result<(), String> {
        self.set_direct_threshold(threshold as f64)
    }

    pub fn compute_ba(
        &mut self,
        out_b_xyz: (&mut [f64], &mut [f64], &mut [f64]),
        out_a_xyz: (&mut [f64], &mut [f64], &mut [f64]),
    ) -> Result<(), String> {
        let n = out_b_xyz.0.len();
        check_length_3tup_result(n, (&out_b_xyz.0, &out_b_xyz.1, &out_b_xyz.2))
            .map_err(|err| err.to_string())?;
        check_length_3tup_result(n, (&out_a_xyz.0, &out_a_xyz.1, &out_a_xyz.2))
            .map_err(|err| err.to_string())?;
        let ok = unsafe {
            ffi::rat_mlfmm_context_compute_ba(
                self.raw,
                out_b_xyz.0.as_mut_ptr(),
                out_b_xyz.1.as_mut_ptr(),
                out_b_xyz.2.as_mut_ptr(),
                n,
                out_a_xyz.0.as_mut_ptr(),
                out_a_xyz.1.as_mut_ptr(),
                out_a_xyz.2.as_mut_ptr(),
                n,
            )
        };
        if ok == 0 { Err(last_error()) } else { Ok(()) }
    }

    fn apply_options(&mut self, opts: Option<&MlfmmOptions>) -> Result<(), String> {
        let default_opts;
        let opts = match opts {
            Some(value) => value,
            None => {
                default_opts = MlfmmOptions::default();
                &default_opts
            }
        };

        self.set_van_lanen(opts.use_van_lanen)?;
        self.set_direct_mode(ffi::RatMlfmmDirectMode::Threshold)?;
        let threshold = opts
            .direct_threshold
            .unwrap_or(DEFAULT_DIRECT_THRESHOLD)
            .max(1);
        self.set_direct_threshold_count(threshold)?;
        if let Some(order) = opts.order {
            self.set_num_exp(order)?;
        }
        Ok(())
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { ffi::rat_mlfmm_context_destroy(self.raw) };
    }
}

#[derive(Clone, Debug)]
pub struct MlfmmOptions {
    pub use_van_lanen: bool,
    pub direct_threshold: Option<u64>,
    pub order: Option<i32>,
}

impl Default for MlfmmOptions {
    fn default() -> Self {
        Self {
            use_van_lanen: true,
            direct_threshold: None,
            order: None,
        }
    }
}

const DEFAULT_DIRECT_THRESHOLD: u64 = 10_000_000;

/// MLFMM calculation of B-field and A-field from linear filament segments.
///
/// Args:
///     xyzp: observation points (x, y, z), shape (3, N).
///     xyzfil: filament segment start points (x, y, z), shape (3, M).
///     dlxyzfil: segment deltas from start to end (x, y, z), shape (3, M).
///     ifil: filament segment currents, length M.
///     eps: Van Lanen softening parameter, length M.
///     opts: MLFMM options (direct threshold, Van Lanen, etc.).
///     out_b_xyz: output B-field components (x, y, z), length N each.
///     out_a_xyz: output A-field components (x, y, z), length N each.
///
/// Returns:
///     Ok(()) on success, Err(String) on failure.
pub fn fields_linear_filament_mlfmm(
    xyzp: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
    eps: &[f64],
    opts: Option<&MlfmmOptions>,
    out_b_xyz: (&mut [f64], &mut [f64], &mut [f64]),
    out_a_xyz: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), String> {
    let mut ctx = Context::from_options(opts)?;
    ctx.set_sources_linear(xyzfil, dlxyzfil, ifil, eps)?;
    ctx.set_targets(xyzp)?;
    ctx.compute_ba(out_b_xyz, out_a_xyz)
}

fn last_error() -> String {
    unsafe {
        let err = ffi::rat_mlfmm_last_error();
        if err.is_null() {
            return "unknown error".to_string();
        }
        CStr::from_ptr(err).to_string_lossy().into_owned()
    }
}

fn check_length_3tup_result(n: usize, tuple: (&[f64], &[f64], &[f64])) -> Result<(), &'static str> {
    check_length_3tup!(n, tuple);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MlfmmOptions, fields_linear_filament_mlfmm};
    use crate::physics::linear_filament::flux_density_linear_filament;
    use crate::physics::linear_filament::vector_potential_linear_filament;

    #[test]
    fn compares_mlfmm_with_linear_filament() {
        let xfil = [0.0, 0.5];
        let yfil = [0.0, 0.0];
        let zfil = [0.0, 0.0];
        let dlx = [0.5, 0.5];
        let dly = [0.0, 0.0];
        let dlz = [0.0, 0.0];
        let ifil = [10.0, 10.0];

        let eps = [1e-6, 1e-6];

        let targets_x = [0.25, 0.75, 0.5];
        let targets_y = [0.1, 0.2, 0.3];
        let targets_z = [0.0, 0.1, -0.2];

        let xp = targets_x;
        let yp = targets_y;
        let zp = targets_z;

        let mut bx = vec![0.0; xp.len()];
        let mut by = vec![0.0; xp.len()];
        let mut bz = vec![0.0; xp.len()];
        let wire_radius = vec![0.0; ifil.len()];
        flux_density_linear_filament(
            (&xp, &yp, &zp),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            &wire_radius,
            (&mut bx, &mut by, &mut bz),
        )
        .expect("linear filament calc failed");

        let opts = MlfmmOptions {
            use_van_lanen: false,
            direct_threshold: Some(1_000_000),
            order: None,
        };
        let mut bx_mlfmm = vec![0.0; xp.len()];
        let mut by_mlfmm = vec![0.0; xp.len()];
        let mut bz_mlfmm = vec![0.0; xp.len()];
        let mut ax_mlfmm = vec![0.0; xp.len()];
        let mut ay_mlfmm = vec![0.0; xp.len()];
        let mut az_mlfmm = vec![0.0; xp.len()];
        fields_linear_filament_mlfmm(
            (&targets_x, &targets_y, &targets_z),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            &eps,
            Some(&opts),
            (&mut bx_mlfmm, &mut by_mlfmm, &mut bz_mlfmm),
            (&mut ax_mlfmm, &mut ay_mlfmm, &mut az_mlfmm),
        )
        .expect("mlfmm compute failed");

        let tol = 1e-5_f64;
        for i in 0..3 {
            let expect = [bx[i], by[i], bz[i]];
            let got = [bx_mlfmm[i], by_mlfmm[i], bz_mlfmm[i]];
            for j in 0..3 {
                let denom = expect[j].abs().max(1.0);
                let err = (got[j] - expect[j]).abs() / denom;
                assert!(
                    err <= tol,
                    "component mismatch at target {i} axis {j}: got {}, expected {}, rel err {}",
                    got[j],
                    expect[j],
                    err
                );
            }
        }
    }

    #[test]
    fn compares_mlfmm_with_linear_filament_van_lanen_fmm() {
        let xfil = [0.0, 0.5];
        let yfil = [0.0, 0.0];
        let zfil = [0.0, 0.0];
        let dlx = [0.5, 0.5];
        let dly = [0.0, 0.0];
        let dlz = [0.0, 0.0];
        let ifil = [10.0, 10.0];

        let eps = [1e-3, 1e-3];

        let targets_x = [0.25, 0.75, 0.5];
        let targets_y = [2.0, -2.5, 3.0];
        let targets_z = [0.5, 1.0, -1.5];

        let xp = targets_x;
        let yp = targets_y;
        let zp = targets_z;

        let mut bx = vec![0.0; xp.len()];
        let mut by = vec![0.0; xp.len()];
        let mut bz = vec![0.0; xp.len()];
        let wire_radius = vec![0.0; ifil.len()];
        flux_density_linear_filament(
            (&xp, &yp, &zp),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            &wire_radius,
            (&mut bx, &mut by, &mut bz),
        )
        .expect("linear filament calc failed");

        let opts = MlfmmOptions {
            use_van_lanen: true,
            direct_threshold: Some(1),
            order: None,
        };
        let mut bx_mlfmm = vec![0.0; xp.len()];
        let mut by_mlfmm = vec![0.0; xp.len()];
        let mut bz_mlfmm = vec![0.0; xp.len()];
        let mut ax_mlfmm = vec![0.0; xp.len()];
        let mut ay_mlfmm = vec![0.0; xp.len()];
        let mut az_mlfmm = vec![0.0; xp.len()];
        fields_linear_filament_mlfmm(
            (&targets_x, &targets_y, &targets_z),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            &eps,
            Some(&opts),
            (&mut bx_mlfmm, &mut by_mlfmm, &mut bz_mlfmm),
            (&mut ax_mlfmm, &mut ay_mlfmm, &mut az_mlfmm),
        )
        .expect("mlfmm compute failed");

        let tol = 5e-3_f64;
        for i in 0..3 {
            let expect = [bx[i], by[i], bz[i]];
            let got = [bx_mlfmm[i], by_mlfmm[i], bz_mlfmm[i]];
            for j in 0..3 {
                let denom = expect[j].abs().max(1.0);
                let err = (got[j] - expect[j]).abs() / denom;
                assert!(
                    err <= tol,
                    "component mismatch at target {i} axis {j}: got {}, expected {}, rel err {}",
                    got[j],
                    expect[j],
                    err
                );
            }
        }
    }

    #[test]
    fn compares_mlfmm_with_vector_potential() {
        let xfil = [0.0, 0.5];
        let yfil = [0.0, 0.0];
        let zfil = [0.0, 0.0];
        let dlx = [0.5, 0.5];
        let dly = [0.0, 0.0];
        let dlz = [0.0, 0.0];
        let ifil = [10.0, 10.0];

        let eps = [1e-6, 1e-6];

        let targets_x = [0.25, 0.75, 0.5];
        let targets_y = [0.2, -0.4, 0.3];
        let targets_z = [0.0, 0.1, -0.2];

        let xp = targets_x;
        let yp = targets_y;
        let zp = targets_z;

        let mut ax = vec![0.0; xp.len()];
        let mut ay = vec![0.0; xp.len()];
        let mut az = vec![0.0; xp.len()];
        vector_potential_linear_filament(
            (&xp, &yp, &zp),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            (&mut ax, &mut ay, &mut az),
        )
        .expect("vector potential calc failed");

        let opts = MlfmmOptions {
            use_van_lanen: false,
            direct_threshold: Some(1_000_000),
            order: None,
        };
        let mut bx_mlfmm = vec![0.0; xp.len()];
        let mut by_mlfmm = vec![0.0; xp.len()];
        let mut bz_mlfmm = vec![0.0; xp.len()];
        let mut ax_mlfmm = vec![0.0; xp.len()];
        let mut ay_mlfmm = vec![0.0; xp.len()];
        let mut az_mlfmm = vec![0.0; xp.len()];
        fields_linear_filament_mlfmm(
            (&targets_x, &targets_y, &targets_z),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            &eps,
            Some(&opts),
            (&mut bx_mlfmm, &mut by_mlfmm, &mut bz_mlfmm),
            (&mut ax_mlfmm, &mut ay_mlfmm, &mut az_mlfmm),
        )
        .expect("mlfmm compute failed");

        let tol = 1e-5_f64;
        for i in 0..3 {
            let expect = [ax[i], ay[i], az[i]];
            let got = [ax_mlfmm[i], ay_mlfmm[i], az_mlfmm[i]];
            for j in 0..3 {
                let denom = expect[j].abs().max(1.0);
                let err = (got[j] - expect[j]).abs() / denom;
                assert!(
                    err <= tol,
                    "component mismatch at target {i} axis {j}: got {}, expected {}, rel err {}",
                    got[j],
                    expect[j],
                    err
                );
            }
        }
    }

    #[test]
    fn compares_mlfmm_with_vector_potential_van_lanen_fmm() {
        let xfil = [0.0, 0.5];
        let yfil = [0.0, 0.0];
        let zfil = [0.0, 0.0];
        let dlx = [0.5, 0.5];
        let dly = [0.0, 0.0];
        let dlz = [0.0, 0.0];
        let ifil = [10.0, 10.0];

        let eps = [1e-3, 1e-3];

        let targets_x = [0.25, 0.75, 0.5];
        let targets_y = [2.0, -2.5, 3.0];
        let targets_z = [0.5, 1.0, -1.5];

        let xp = targets_x;
        let yp = targets_y;
        let zp = targets_z;

        let mut ax = vec![0.0; xp.len()];
        let mut ay = vec![0.0; xp.len()];
        let mut az = vec![0.0; xp.len()];
        vector_potential_linear_filament(
            (&xp, &yp, &zp),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            (&mut ax, &mut ay, &mut az),
        )
        .expect("vector potential calc failed");

        let opts = MlfmmOptions {
            use_van_lanen: true,
            direct_threshold: Some(1),
            order: None,
        };
        let mut bx_mlfmm = vec![0.0; xp.len()];
        let mut by_mlfmm = vec![0.0; xp.len()];
        let mut bz_mlfmm = vec![0.0; xp.len()];
        let mut ax_mlfmm = vec![0.0; xp.len()];
        let mut ay_mlfmm = vec![0.0; xp.len()];
        let mut az_mlfmm = vec![0.0; xp.len()];
        fields_linear_filament_mlfmm(
            (&targets_x, &targets_y, &targets_z),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            &eps,
            Some(&opts),
            (&mut bx_mlfmm, &mut by_mlfmm, &mut bz_mlfmm),
            (&mut ax_mlfmm, &mut ay_mlfmm, &mut az_mlfmm),
        )
        .expect("mlfmm compute failed");

        let tol = 5e-3_f64;
        for i in 0..3 {
            let expect = [ax[i], ay[i], az[i]];
            let got = [ax_mlfmm[i], ay_mlfmm[i], az_mlfmm[i]];
            for j in 0..3 {
                let denom = expect[j].abs().max(1.0);
                let err = (got[j] - expect[j]).abs() / denom;
                assert!(
                    err <= tol,
                    "component mismatch at target {i} axis {j}: got {}, expected {}, rel err {}",
                    got[j],
                    expect[j],
                    err
                );
            }
        }
    }
}
