mod ffi;

use std::ffi::CStr;

pub struct Context {
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

    pub fn set_sources_linear(
        &mut self,
        rs_xyz: &[f64],
        drs_xyz: &[f64],
        currents: &[f64],
        eps: &[f64],
    ) -> Result<(), String> {
        if rs_xyz.len() != drs_xyz.len() {
            return Err("rs_xyz and drs_xyz must have the same length".to_string());
        }
        if rs_xyz.len() % 3 != 0 {
            return Err("rs_xyz length must be a multiple of 3".to_string());
        }
        let num_sources = rs_xyz.len() / 3;
        if currents.len() != num_sources || eps.len() != num_sources {
            return Err("currents and eps must have length num_sources".to_string());
        }
        let ok = unsafe {
            ffi::rat_mlfmm_context_set_sources_linear(
                self.raw,
                rs_xyz.as_ptr(),
                drs_xyz.as_ptr(),
                currents.as_ptr(),
                eps.as_ptr(),
                num_sources,
            )
        };
        if ok == 0 {
            Err(last_error())
        } else {
            Ok(())
        }
    }

    pub fn set_targets(&mut self, rt_xyz: &[f64]) -> Result<(), String> {
        if rt_xyz.len() % 3 != 0 {
            return Err("rt_xyz length must be a multiple of 3".to_string());
        }
        let num_targets = rt_xyz.len() / 3;
        let ok = unsafe { ffi::rat_mlfmm_context_set_targets(self.raw, rt_xyz.as_ptr(), num_targets) };
        if ok == 0 {
            Err(last_error())
        } else {
            Ok(())
        }
    }

    pub fn set_van_lanen(&mut self, use_van_lanen: bool) -> Result<(), String> {
        let ok = unsafe { ffi::rat_mlfmm_context_set_van_lanen(self.raw, use_van_lanen as i32) };
        if ok == 0 {
            Err(last_error())
        } else {
            Ok(())
        }
    }

    pub fn set_num_exp(&mut self, num_exp: i32) -> Result<(), String> {
        let ok = unsafe { ffi::rat_mlfmm_context_set_num_exp(self.raw, num_exp) };
        if ok == 0 {
            Err(last_error())
        } else {
            Ok(())
        }
    }

    pub fn set_direct_mode(&mut self, mode: ffi::RatMlfmmDirectMode) -> Result<(), String> {
        let ok = unsafe { ffi::rat_mlfmm_context_set_direct_mode(self.raw, mode) };
        if ok == 0 {
            Err(last_error())
        } else {
            Ok(())
        }
    }

    pub fn set_direct_threshold(&mut self, threshold: f64) -> Result<(), String> {
        let ok = unsafe { ffi::rat_mlfmm_context_set_direct_threshold(self.raw, threshold) };
        if ok == 0 {
            Err(last_error())
        } else {
            Ok(())
        }
    }

    pub fn compute_b(&mut self, out_b_xyz: &mut [f64]) -> Result<(), String> {
        let ok = unsafe {
            ffi::rat_mlfmm_context_compute_b(self.raw, out_b_xyz.as_mut_ptr(), out_b_xyz.len())
        };
        if ok == 0 {
            Err(last_error())
        } else {
            Ok(())
        }
    }

    pub fn compute_a(&mut self, out_a_xyz: &mut [f64]) -> Result<(), String> {
        let ok = unsafe {
            ffi::rat_mlfmm_context_compute_a(self.raw, out_a_xyz.as_mut_ptr(), out_a_xyz.len())
        };
        if ok == 0 {
            Err(last_error())
        } else {
            Ok(())
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { ffi::rat_mlfmm_context_destroy(self.raw) };
    }
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

#[cfg(test)]
mod tests {
    use super::{ffi, Context};
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

        let rs = [
            0.25, 0.0, 0.0, //
            0.75, 0.0, 0.0,
        ];
        let drs = [
            0.5, 0.0, 0.0, //
            0.5, 0.0, 0.0,
        ];
        let eps = [1e-6, 1e-6];

        let targets = [
            0.25, 0.1, 0.0, //
            0.75, 0.2, 0.1, //
            0.5, 0.3, -0.2,
        ];

        let xp = [targets[0], targets[3], targets[6]];
        let yp = [targets[1], targets[4], targets[7]];
        let zp = [targets[2], targets[5], targets[8]];

        let mut bx = vec![0.0; xp.len()];
        let mut by = vec![0.0; xp.len()];
        let mut bz = vec![0.0; xp.len()];
        flux_density_linear_filament(
            (&xp, &yp, &zp),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            (&mut bx, &mut by, &mut bz),
        )
        .expect("linear filament calc failed");

        let mut ctx = Context::new().expect("mlfmm context create failed");
        ctx.set_sources_linear(&rs, &drs, &ifil, &eps)
            .expect("mlfmm set sources failed");
        ctx.set_targets(&targets)
            .expect("mlfmm set targets failed");
        ctx.set_van_lanen(false)
            .expect("mlfmm set van lanen failed");
        ctx.set_direct_mode(ffi::RatMlfmmDirectMode::Always)
            .expect("mlfmm set direct mode failed");

        let mut b_mlfmm = vec![0.0; 9];
        ctx.compute_b(&mut b_mlfmm)
            .expect("mlfmm compute failed");

        let tol = 1e-5_f64;
        for i in 0..3 {
            let base = i * 3;
            let expect = [bx[i], by[i], bz[i]];
            let got = [b_mlfmm[base], b_mlfmm[base + 1], b_mlfmm[base + 2]];
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

        let rs = [
            0.25, 0.0, 0.0, //
            0.75, 0.0, 0.0,
        ];
        let drs = [
            0.5, 0.0, 0.0, //
            0.5, 0.0, 0.0,
        ];
        let eps = [1e-3, 1e-3];

        let targets = [
            0.25, 2.0, 0.5, //
            0.75, -2.5, 1.0, //
            0.5, 3.0, -1.5,
        ];

        let xp = [targets[0], targets[3], targets[6]];
        let yp = [targets[1], targets[4], targets[7]];
        let zp = [targets[2], targets[5], targets[8]];

        let mut bx = vec![0.0; xp.len()];
        let mut by = vec![0.0; xp.len()];
        let mut bz = vec![0.0; xp.len()];
        flux_density_linear_filament(
            (&xp, &yp, &zp),
            (&xfil, &yfil, &zfil),
            (&dlx, &dly, &dlz),
            &ifil,
            (&mut bx, &mut by, &mut bz),
        )
        .expect("linear filament calc failed");

        let mut ctx = Context::new().expect("mlfmm context create failed");
        ctx.set_sources_linear(&rs, &drs, &ifil, &eps)
            .expect("mlfmm set sources failed");
        ctx.set_targets(&targets)
            .expect("mlfmm set targets failed");
        ctx.set_van_lanen(true)
            .expect("mlfmm set van lanen failed");
        ctx.set_direct_mode(ffi::RatMlfmmDirectMode::Never)
            .expect("mlfmm set direct mode failed");

        let mut b_mlfmm = vec![0.0; 9];
        ctx.compute_b(&mut b_mlfmm)
            .expect("mlfmm compute failed");

        let tol = 5e-3_f64;
        for i in 0..3 {
            let base = i * 3;
            let expect = [bx[i], by[i], bz[i]];
            let got = [b_mlfmm[base], b_mlfmm[base + 1], b_mlfmm[base + 2]];
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

        let rs = [
            0.25, 0.0, 0.0, //
            0.75, 0.0, 0.0,
        ];
        let drs = [
            0.5, 0.0, 0.0, //
            0.5, 0.0, 0.0,
        ];
        let eps = [1e-6, 1e-6];

        let targets = [
            0.25, 0.2, 0.0, //
            0.75, -0.4, 0.1, //
            0.5, 0.3, -0.2,
        ];

        let xp = [targets[0], targets[3], targets[6]];
        let yp = [targets[1], targets[4], targets[7]];
        let zp = [targets[2], targets[5], targets[8]];

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

        let mut ctx = Context::new().expect("mlfmm context create failed");
        ctx.set_sources_linear(&rs, &drs, &ifil, &eps)
            .expect("mlfmm set sources failed");
        ctx.set_targets(&targets)
            .expect("mlfmm set targets failed");
        ctx.set_van_lanen(false)
            .expect("mlfmm set van lanen failed");
        ctx.set_direct_mode(ffi::RatMlfmmDirectMode::Always)
            .expect("mlfmm set direct mode failed");

        let mut a_mlfmm = vec![0.0; 9];
        ctx.compute_a(&mut a_mlfmm)
            .expect("mlfmm compute failed");

        let tol = 1e-5_f64;
        for i in 0..3 {
            let base = i * 3;
            let expect = [ax[i], ay[i], az[i]];
            let got = [a_mlfmm[base], a_mlfmm[base + 1], a_mlfmm[base + 2]];
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

        let rs = [
            0.25, 0.0, 0.0, //
            0.75, 0.0, 0.0,
        ];
        let drs = [
            0.5, 0.0, 0.0, //
            0.5, 0.0, 0.0,
        ];
        let eps = [1e-3, 1e-3];

        let targets = [
            0.25, 2.0, 0.5, //
            0.75, -2.5, 1.0, //
            0.5, 3.0, -1.5,
        ];

        let xp = [targets[0], targets[3], targets[6]];
        let yp = [targets[1], targets[4], targets[7]];
        let zp = [targets[2], targets[5], targets[8]];

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

        let mut ctx = Context::new().expect("mlfmm context create failed");
        ctx.set_sources_linear(&rs, &drs, &ifil, &eps)
            .expect("mlfmm set sources failed");
        ctx.set_targets(&targets)
            .expect("mlfmm set targets failed");
        ctx.set_van_lanen(true)
            .expect("mlfmm set van lanen failed");
        ctx.set_direct_mode(ffi::RatMlfmmDirectMode::Never)
            .expect("mlfmm set direct mode failed");

        let mut a_mlfmm = vec![0.0; 9];
        ctx.compute_a(&mut a_mlfmm)
            .expect("mlfmm compute failed");

        let tol = 5e-3_f64;
        for i in 0..3 {
            let base = i * 3;
            let expect = [ax[i], ay[i], az[i]];
            let got = [a_mlfmm[base], a_mlfmm[base + 1], a_mlfmm[base + 2]];
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
