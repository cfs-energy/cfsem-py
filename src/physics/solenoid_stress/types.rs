use num_traits::{Float, FromPrimitive};

pub trait Real: Float + FromPrimitive + Copy + std::fmt::Debug + Send + Sync + 'static {}

impl<T> Real for T where T: Float + FromPrimitive + Copy + std::fmt::Debug + Send + Sync + 'static {}

pub fn cast<F: Real>(value: f64) -> F {
    F::from_f64(value).expect("finite f64 literal should cast to target float")
}

pub fn two_pi<F: Real>() -> F {
    cast(2.0 * core::f64::consts::PI)
}
