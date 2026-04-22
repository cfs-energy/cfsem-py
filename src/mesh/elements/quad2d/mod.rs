//! Quadrilateral reference elements and shared quad-family helpers.
//!
//! The interpolation functions and face numbering used by the quadrilateral reference elements
//! follow the standard isoparametric finite-element setup summarized by Bower, especially
//! Section 8.1 and Table 8.3 of *Applied Mechanics of Solids*.

pub mod mapping;
pub mod quad4;
pub mod quad9;
pub mod quadrature;

pub use quadrature::QuadratureRule;
