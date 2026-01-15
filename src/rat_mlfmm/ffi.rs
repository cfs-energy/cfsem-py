use std::os::raw::{c_char, c_int};

#[repr(C)]
pub struct RatMlfmmContext {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum RatMlfmmDirectMode {
    Always = 0,
    Threshold = 1,
    Never = 2,
}

unsafe extern "C" {
    pub fn rat_mlfmm_context_create() -> *mut RatMlfmmContext;
    pub fn rat_mlfmm_context_destroy(ctx: *mut RatMlfmmContext);

    pub fn rat_mlfmm_context_set_sources_linear(
        ctx: *mut RatMlfmmContext,
        rs_xyz: *const f64,
        drs_xyz: *const f64,
        currents: *const f64,
        eps: *const f64,
        num_sources: usize,
    ) -> c_int;

    pub fn rat_mlfmm_context_set_targets(
        ctx: *mut RatMlfmmContext,
        rt_xyz: *const f64,
        num_targets: usize,
    ) -> c_int;

    pub fn rat_mlfmm_context_set_van_lanen(ctx: *mut RatMlfmmContext, use_van_lanen: c_int)
        -> c_int;

    pub fn rat_mlfmm_context_set_num_exp(ctx: *mut RatMlfmmContext, num_exp: c_int) -> c_int;

    pub fn rat_mlfmm_context_set_direct_mode(
        ctx: *mut RatMlfmmContext,
        mode: RatMlfmmDirectMode,
    ) -> c_int;

    pub fn rat_mlfmm_context_set_direct_threshold(
        ctx: *mut RatMlfmmContext,
        threshold: f64,
    ) -> c_int;

    pub fn rat_mlfmm_context_compute_b(
        ctx: *mut RatMlfmmContext,
        out_b_xyz: *mut f64,
        out_len: usize,
    ) -> c_int;

    pub fn rat_mlfmm_last_error() -> *const c_char;
}
