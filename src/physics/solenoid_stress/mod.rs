//! Axisymmetric finite-element elasticity helpers for solenoid stress problems.
//!
//! The 2D-axisymmetric small-strain formulation implemented here uses the standard
//! displacement-based Galerkin finite-element construction
//! `K_e = integral(B^T D B 2*pi*r dA)`.
//!
//! References:
//! - Thomas J. R. Hughes, *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*, 1987.
//! - Klaus-Juergen Bathe, *Finite Element Procedures*, 1996.
//! - J. N. Reddy, *An Introduction to the Finite Element Method*, 3rd ed., 2005.

mod assembly;
mod axisym;
mod geometry;
mod loads;
mod mesh;
mod quad4;
mod quadrature;
mod types;

pub use assembly::assemble_axisymmetric_quad4;
pub use geometry::{ElementMeasures, ElementQuadrature, element_measures, element_quadrature};
pub use mesh::{AssemblyResult, MeshView, PressureLoad};
pub use quadrature::QuadratureRule;
pub use types::Real;
