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
