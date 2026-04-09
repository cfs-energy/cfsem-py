//! Axisymmetric finite-element elasticity helpers for solenoid stress problems.
//!
//! The 2D-axisymmetric small-strain formulation implemented here uses the standard
//! displacement-based Galerkin finite-element construction
//! `K_e = integral(B^T D B 2*pi*r dA)`.
//!
//! Each node carries two displacement unknowns: radial `u_r` and axial `u_z`.  The global
//! linear system therefore has the form `K u = f`, where `u = [u_r(0), u_z(0), u_r(1), u_z(1), ...]^T`.
//! The corresponding load-vector entries are generalized nodal forces, not usually literal point
//! forces.  Entry `f[2a]` is the force-like quantity work-conjugate to the radial displacement
//! degree of freedom at node `a`, and `f[2a + 1]` is the corresponding axial quantity.
//!
//! Each equation in the assembled system is a weak equilibrium statement for one nodal test
//! displacement pattern: the internal virtual work from the elastic stress field balances the
//! external virtual work from body forces, surface loads, and thermal strain.  In that sense, row
//! `i` of `K u = f` should be read as "the restoring force associated with test degree of freedom
//! `i` equals the applied generalized force associated with that same test degree of freedom,"
//! rather than as a pointwise force balance written directly at one node.
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
//! The right-hand side is assembled from the same weak form, so each load type is converted into a
//! consistent nodal load vector:
//! - body force density `b = [b_r, b_z]` contributes `f_body = integral(N^T b 2*pi*r dA)`,
//! - scalar pressure `p` on a face contributes `f_pressure = integral(N^T (-p n) 2*pi*r ds)`,
//! - vector traction `t = [t_r, t_z]` on a face contributes `f_traction = integral(N^T t 2*pi*r ds)`,
//! - thermal strain contributes an equivalent load
//!   `f_thermal = integral(B^T D epsilon_th 2*pi*r dA)`, where
//!   `epsilon_th = alpha * (T - T_ref)`.
//!
//! The thermal term is an eigenstrain load, not an externally applied traction or body force.  It
//! appears on the right-hand side because the constitutive law is evaluated as
//! `sigma = D (epsilon - epsilon_th)`, so the `D epsilon_th` contribution is moved to the load
//! vector as an equivalent nodal force.
//!
//! The code is organized so that each module owns one step of that pipeline:
//! - [`quad4`] defines the bilinear shape functions and geometric mapping.
//! - [`quad9`] defines the quadratic shape functions and geometric mapping.
//! - [`quadrature`] provides the Gauss rules on the reference square and its edges.
//! - [`geometry`] evaluates quadrature-point locations, weights, and gradients in physical space.
//! - [`axisym`] constructs the axisymmetric strain operator and local stiffness kernel.
//! - [`loads`] assembles consistent nodal loads from body forces, pressures, and tractions.
//! - [`load_operators`] builds sparse linear maps from load amplitudes or nodal temperatures to the
//!   global right-hand side for repeated-load solves.
//! - [`assembly`] ties the pieces together into sparse triplets plus the global right-hand side.
//! - [`recovery`] builds sparse operators for quadrature-point strain and stress recovery.
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
