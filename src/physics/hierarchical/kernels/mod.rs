//! Concrete kernels for the generic dual-tree infrastructure.

mod boundary_element;
mod boundary_element_flux_density;
mod boundary_element_vector_potential;
mod dipole;
mod dipole_flux_density;
mod dipole_vector_potential;
mod linear_filament_flux_density;
mod linear_filament_vector_potential;

pub use boundary_element::{BoundaryElementSummary, BoundaryElementTriangle};
pub use boundary_element_flux_density::BoundaryElementFluxDensityKernel;
pub use boundary_element_vector_potential::BoundaryElementVectorPotentialKernel;
pub use dipole::{DipoleSource, DipoleTarget, DipoleTargetSummary};
pub use dipole_flux_density::{DipoleFluxDensityKernel, DipoleFluxDensitySummary};
pub use dipole_vector_potential::{DipoleVectorPotentialKernel, DipoleVectorPotentialSummary};
pub use linear_filament_flux_density::{
    LinearFilamentFluxDensityKernel, LinearFilamentFluxDensitySummary, LinearFilamentSource,
};
pub use linear_filament_vector_potential::{
    LinearFilamentVectorPotentialKernel, LinearFilamentVectorPotentialSummary,
};
