//! Concrete kernels for the generic single-source-tree infrastructure.

mod boundary_element;
mod boundary_element_flux_density;
mod boundary_element_vector_potential;
mod dipole;
mod dipole_flux_density;
mod dipole_vector_potential;
mod linear_filament;
mod linear_filament_flux_density;
mod linear_filament_vector_potential;

pub use boundary_element::{
    BoundaryElementNodalValues, BoundaryElementSummary, BoundaryElementTriangle,
    BoundaryElementTriangles,
};
pub use boundary_element_flux_density::BoundaryElementFluxDensityKernel;
pub use boundary_element_vector_potential::BoundaryElementVectorPotentialKernel;
pub use dipole::{
    DipoleMoments, DipoleSource, DipoleSources, DipoleSummary, DipoleTarget, DipoleTargetSummary,
    DipoleTargets,
};
pub use dipole_flux_density::{DipoleFluxDensityKernel, DipoleFluxDensitySummary};
pub use dipole_vector_potential::{DipoleVectorPotentialKernel, DipoleVectorPotentialSummary};
pub use linear_filament::{LinearFilamentSource, LinearFilamentSources, LinearFilamentSummary};
pub use linear_filament_flux_density::{
    LinearFilamentFluxDensityKernel, LinearFilamentFluxDensitySummary,
};
pub use linear_filament_vector_potential::{
    LinearFilamentVectorPotentialKernel, LinearFilamentVectorPotentialSummary,
};
