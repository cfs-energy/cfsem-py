//! Concrete kernels for the generic dual-tree infrastructure.

mod dipole;
mod dipole_flux_density;
mod dipole_vector_potential;

pub use dipole::{DipoleSource, DipoleTarget, DipoleTargetSummary};
pub use dipole_flux_density::{DipoleFluxDensityKernel, DipoleFluxDensitySummary};
pub use dipole_vector_potential::{DipoleVectorPotentialKernel, DipoleVectorPotentialSummary};
