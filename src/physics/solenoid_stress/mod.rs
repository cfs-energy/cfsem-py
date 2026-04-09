//! Axisymmetric finite-element elasticity helpers for solenoid stress problems.
//!
//! The 2D-axisymmetric small-strain formulation implemented here uses the standard
//! displacement-based Galerkin finite-element construction
//! `K_e = integral(B^T D B 2*pi*r dA)`.
//!
//! At a high level, each bilinear quadrilateral element is treated as a mapping from a
//! reference square `(\xi, \eta) in [-1, 1]^2` into physical `(r, z)` space.  At each
//! quadrature point the solver:
//! - evaluates the shape functions `N_i(\xi, \eta)`,
//! - maps their reference gradients into physical gradients with the element Jacobian,
//! - builds the axisymmetric strain-displacement matrix `B`,
//! - forms `B^T D B` for the local stiffness contribution, and
//! - scales the contribution by the usual area weight `det(J) w` and by the additional
//!   axisymmetric revolution factor `2*pi*r`.
//!
//! The code is organized so that each module owns one step of that pipeline:
//! - [`quad4`] defines the bilinear shape functions and geometric mapping.
//! - [`quadrature`] provides the Gauss rules on the reference square and its edges.
//! - [`geometry`] evaluates quadrature-point locations, weights, and gradients in physical space.
//! - [`axisym`] constructs the axisymmetric strain operator and local stiffness kernel.
//! - [`loads`] assembles consistent nodal loads from body forces and face tractions.
//! - [`assembly`] ties the pieces together into sparse triplets plus the global right-hand side.
//!
//! References:
//! - Thomas J. R. Hughes, *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*, 1987.
//! - Klaus-Juergen Bathe, *Finite Element Procedures*, 1996.
//! - J. N. Reddy, *An Introduction to the Finite Element Method*, 3rd ed., 2005.

mod assembly;
mod axisym;
mod geometry;
mod load_operators;
mod loads;
mod mesh;
mod quad4;
mod quad9;
mod quadrature;
mod recovery;
mod types;

pub use assembly::{assemble_axisymmetric_quad4, assemble_axisymmetric_quad9};
pub use geometry::{
    ElementMeasures, ElementQuadrature, element_measures_quad4, element_measures_quad9,
    element_quadrature_quad4, element_quadrature_quad9,
};
pub use load_operators::{
    SparseOperator, ThermalLoadOperator, body_force_operator_quad4, body_force_operator_quad9,
    pressure_operator_quad4, pressure_operator_quad9, temperature_operator_quad4,
    temperature_operator_quad9, traction_operator_quad4, traction_operator_quad9,
};
pub use mesh::{AssemblyResult, MeshView, PressureLoad, ThermalMaterial, TractionLoad};
pub use quadrature::QuadratureRule;
pub use recovery::{
    QuadratureFieldOperators, quadrature_field_operators_quad4, quadrature_field_operators_quad9,
};
pub use types::Real;
