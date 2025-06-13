//! Electromagnetics calculations.
pub mod biotsavart;
pub mod circular_filament;
pub mod gradshafranov;
pub mod linear_filament;
pub mod point_source;

#[doc(hidden)] // Might make breaking changes soon
pub mod mesh_filament;

pub use circular_filament::{flux_circular_filament, flux_density_circular_filament};
