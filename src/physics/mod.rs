//! Electromagnetics calculations.
pub mod boundary_element;
pub mod circular_filament;
pub mod gradshafranov;
pub mod linear_filament;
pub mod point_source;
pub mod solenoid_stress;
pub(crate) mod volumetric;

pub use circular_filament::{flux_circular_filament, flux_density_circular_filament};
