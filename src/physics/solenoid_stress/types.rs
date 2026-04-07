//! Shared numeric traits and constants for the solenoid-stress backend.

use num_traits::{Float, FromPrimitive};

/// Floating-point trait bound used throughout the solenoid-stress backend.
///
/// Keeping the bound in one place makes it easier to support both `f32` and `f64` entry points
/// without duplicating generic constraints everywhere else.
pub trait Real: Float + FromPrimitive + Copy + std::fmt::Debug + Send + Sync + 'static {}

impl<T> Real for T where T: Float + FromPrimitive + Copy + std::fmt::Debug + Send + Sync + 'static {}

/// Cast a literal `f64` constant into the active floating-point type.
pub fn cast<F: Real>(value: f64) -> F {
    F::from_f64(value).expect("finite f64 literal should cast to target float")
}

/// Return the constant `2*pi` in the active floating-point type.
pub fn two_pi<F: Real>() -> F {
    cast(2.0 * core::f64::consts::PI)
}
