//! Axisymmetric finite-element elasticity helpers for solenoid stress problems.
//!
//! The 2D-axisymmetric small-strain formulation implemented here uses the standard
//! displacement-based Galerkin finite-element construction
//! `K_e = integral(B^T D B 2*pi*r dA)`.
//!
//! This formula is best read from right to left.  For one element with nodal displacement vector
//! `u_e = [u_r1, u_z1, u_r2, u_z2, ...]^T`, the strain at a quadrature point is
//! `epsilon = B u_e`, and the constitutive law gives `sigma = D epsilon = D B u_e`.  Converting
//! that pointwise stress field back into equivalent nodal forces by virtual work gives
//! `f_int,e = integral(B^T sigma 2*pi*r dA) = integral(B^T D B u_e 2*pi*r dA) = K_e u_e`.
//! So `K_e` is the element stiffness matrix: it maps one element displacement pattern to the
//! internal restoring forces associated with that same element's nodal degrees of freedom.
//!
//! Each symbol has a distinct role:
//! - `u_e` is the vector of element nodal displacement degrees of freedom
//!   `[u_r1, u_z1, u_r2, u_z2, ...]^T`,
//! - `u` is the interpolated displacement field inside the element at one point,
//! - `N` is the shape-function interpolation matrix that maps nodal displacements to pointwise
//!   displacement through `u = N u_e`,
//! - `J` is the element Jacobian of the mapping from reference coordinates `(\xi, \eta)` to
//!   physical coordinates `(r, z)`, so it converts reference-space gradients and differential area
//!   into physical-space gradients and area through `dA = det(J) d\xi d\eta`,
//! - `B` maps nodal displacements to the axisymmetric strain vector,
//! - `D` maps strain to stress through the material law,
//! - `epsilon` is the strain vector at a point,
//! - `sigma` is the stress vector at a point,
//! - `B^T` maps stress back to equivalent nodal forces,
//! - `dA` is the differential area in the `(r, z)` cross-section, and
//! - `2*pi*r` is the axisymmetric revolution factor that converts cross-sectional area into the
//!   volume of the corresponding ring in 3D.
//!
//! The interpolation statement `u = N u_e` means that the displacement field inside one element is
//! reconstructed from the element nodal displacement values.  At any point in the element,
//! `u(r, z) = [u_r(r, z), u_z(r, z)]^T`, while `u_e` stores the nodal radial and axial
//! displacements.  If the element has nodes `1..n`, then the interpolation matrix has the block
//! form
//! - `[ N_1  0    N_2  0   ...  N_n  0 ]`,
//! - `[ 0    N_1  0    N_2 ...  0    N_n ]`,
//! where the scalar shape functions `N_i(r, z)` are evaluated at the point of interest.
//! Multiplying by `u_e` gives
//! - `u_r(r, z) = N_1 u_r1 + N_2 u_r2 + ... + N_n u_rn`,
//! - `u_z(r, z) = N_1 u_z1 + N_2 u_z2 + ... + N_n u_zn`.
//! In other words, the element displacement field is an interpolation of the nodal displacements.
//! The shape functions are chosen so that `N_i = 1` at node `i` and `N_i = 0` at the other element
//! nodes, which guarantees that the interpolated field reproduces the nodal values exactly at the
//! nodes.  The strain-displacement matrix `B` is obtained by differentiating this interpolation, so
//! strains are computed from the spatial gradients of the same shape functions.
//!
//! The Jacobian `J` describes how the element mapping stretches, skews, and scales the reference
//! square when it is carried into physical `(r, z)` space.  Its determinant `det(J)` is the local
//! area-scaling factor between the reference element and the physical element, which is why the
//! quadrature weights later appear as `det(J) w`.  The inverse Jacobian is also what converts
//! reference-coordinate shape-function gradients into physical gradients, which are then used to
//! build the strain-displacement matrix `B`.
//!
//! One useful interpretation of a single matrix entry `K_e[i, j]` is: apply a unit displacement in
//! local degree of freedom `j`, hold all other local degrees of freedom fixed, and `K_e[i, j]`
//! gives the internal generalized force induced in local degree of freedom `i`.  That is why the
//! matrix has stiffness units and why material farther from the axis contributes more strongly
//! through the `2*pi*r` weight.
//!
//! The variational statement behind all of this is the principle of virtual work.  Rather than
//! enforcing equilibrium pointwise in strong form, the finite-element method enforces
//! `delta W_int = delta W_ext` for every admissible virtual displacement field `delta u`.  A
//! virtual displacement is not an actual motion in time; it is an imagined infinitesimal kinematic
//! perturbation used to probe whether the current stress state is in equilibrium.  If the body is
//! in equilibrium, then the internal stresses and the applied loads must do equal virtual work
//! against every such perturbation.
//!
//! In axisymmetric small-strain elasticity, the virtual-work statement can be written schematically
//! as
//! - `delta W_int = integral((delta epsilon)^T sigma 2*pi*r dA)`,
//! - `delta W_ext = integral((delta u)^T b 2*pi*r dA) + integral((delta u)^T t 2*pi*r ds)`,
//! where `b` is body-force density and `t` is an applied surface traction.  Thermal strain enters
//! through the constitutive law `sigma = D (epsilon - epsilon_th)` and can therefore be moved to
//! the right-hand side as an equivalent load.
//!
//! After approximating the displacement field with element shape functions, one writes
//! - `u = N u_e`,
//! - `delta u = N delta u_e`,
//! - `epsilon = B u_e`,
//! - `delta epsilon = B delta u_e`,
//! where `u_e` collects the element nodal displacement degrees of freedom.  Substituting these
//! into the virtual-work statement gives
//! - `delta W_int = delta u_e^T [integral(B^T D B 2*pi*r dA)] u_e`,
//! - `delta W_ext = delta u_e^T f_e`.
//! Since `delta u_e` is arbitrary, the bracketed quantity defines the element equations
//! `K_e u_e = f_e`.  After assembling the element contributions over the whole mesh, this becomes
//! the global linear system `K u = f`.
//!
//! This is the origin of the generalized-force interpretation used throughout the implementation.
//! A generalized force is simply the quantity that is work-conjugate to a generalized displacement
//! coordinate.  Here the generalized coordinates are the nodal radial and axial displacements, so
//! the load-vector entries are the corresponding radial and axial generalized nodal forces.
//! Distributed loads are therefore converted into equivalent nodal loads by asking: which nodal
//! force vector would produce the same virtual work as the original distributed loading for every
//! virtual displacement field representable by the element basis?
//!
//! This is why the right-hand side is assembled with `N^T` for direct force-like loads and `B^T`
//! for stress-like loads:
//! - body forces, pressures, and tractions act through virtual displacements and contribute
//!   `integral(N^T (...) 2*pi*r dA)` or `integral(N^T (...) 2*pi*r ds)`,
//! - thermal strain first produces stress through `D epsilon_th`, then contributes through
//!   `integral(B^T D epsilon_th 2*pi*r dA)`.
//!
//! From this viewpoint, each row of the global system corresponds to one test displacement pattern,
//! typically "activate one nodal degree of freedom and set all other virtual degrees of freedom to
//! zero."  The row equation states that the internal restoring force associated with that test
//! pattern balances the applied generalized force associated with the same test pattern.  It is
//! therefore better interpreted as a weak equilibrium statement than as a literal free-body-diagram
//! force balance at a node.
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
//! - maps their reference gradients into physical gradients with the element Jacobian `J`,
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
//! Face integrals for pressure and traction are evaluated by parameterizing each loaded element
//! edge with a 1D reference coordinate `s in [-1, 1]` and applying a 1D Gauss rule along that
//! edge.  At each face quadrature point the solver:
//! - maps `s` to a face point `(\xi, \eta)` on the reference element,
//! - evaluates the shape functions there,
//! - uses the element Jacobian `J` to map the reference edge direction `d[\xi,\eta]/ds` into the
//!   physical tangent `dx/ds`,
//! - evaluates the physical face point `(r, z)`, and
//! - multiplies by the axisymmetric surface measure `2*pi*r`.
//! The same quadrature setting that selects the tensor-product volume rule also selects the 1D face
//! rule: `GaussLegendre3`/`gl3` and `GaussLegendre4`/`gl4` correspond to 3-point and 4-point
//! Gauss-Legendre quadrature along each
//! loaded face, respectively.
//!
//! The physical line element is `dS = 2*pi*r |dx/ds| ds`, so traction loads contribute
//! `integral(N^T t 2*pi*r |dx/ds| ds)`.  Pressure uses the face normal rather than a prescribed
//! global direction.  In the implementation the tangent is rotated to
//! `normal_area = [t_z, -t_r]`, which bundles the outward normal direction together with the line
//! Jacobian `|dx/ds|`.  The pressure integral is therefore evaluated as
//! `integral(N^T (-p normal_area) 2*pi*r ds)` without separately normalizing the face normal.
//! This is why face orientation and consistent element node ordering matter for pressure loads.
//!
//! The code is organized so that each module owns one step of that pipeline:
//! - [`crate::mesh::elements::quad4`] defines the bilinear shape functions and reference-element
//!   geometry.
//! - [`crate::mesh::elements::quad9`] defines the quadratic shape functions and reference-element
//!   geometry.
//! - [`crate::mesh::quadrature`] provides the shared 1D Gauss-Legendre rules on an interval.
//! - [`crate::mesh::elements::quad2d::quadrature`] builds the quadrilateral tensor-product square
//!   and face rules from that 1D basis.
//! - [`geometry`] adds axisymmetric validation and evaluates the `2*pi*r`-weighted element
//!   summaries needed by the structural solver.
//! - [`axisym`] constructs the axisymmetric strain operator and local stiffness kernel.
//! - [`loads`] assembles consistent nodal loads from body forces, pressures, and tractions.
//! - [`load_operators`] builds sparse linear maps from load amplitudes or nodal temperatures to the
//!   global right-hand side for repeated-load solves.
//! - [`assembly`] ties the pieces together into sparse triplets plus the global right-hand side.
//! - [`recovery`] builds sparse operators for quadrature-point strain and stress recovery.
//!
//! References:
//! - E. L. Wilson, "Structural Analysis of Axisymmetric Solids," *AIAA Journal*, 3(12), pp. 2269-2274, December 1965. doi:10.2514/3.3356.
//! - R. A. Mitchell, R. M. Woolley, and C. R. Fisher, "Formulation and experimental verification of an axisymmetric finite-element structural analysis," *Journal of Research of the National Bureau of Standards Section C*, 75C, 1971.
//! - I. Fried, "Notes on the finite element analysis of the axisymmetric elastic solid," *International Journal of Solids and Structures*, 10(3), 1974.
//! - Thomas J. R. Hughes, *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*, 1987.
//! - Klaus-Juergen Bathe, *Finite Element Procedures*, 1996.
//! - J. N. Reddy, *An Introduction to the Finite Element Method*, 3rd ed., 2005.

mod assembly;
mod axisym;
mod geometry;
mod load_operators;
mod loads;
mod recovery;
mod types;

pub use crate::mesh::elements::quad2d::{quad4, quad9};
pub use crate::mesh::{MeshView, QuadratureRule};
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
pub use recovery::{
    QuadratureFieldOperators, quadrature_field_operators_quad4, quadrature_field_operators_quad9,
};
pub use types::{
    AssemblyResult, DOF_PER_NODE, PressureLoad, Real, ThermalMaterial, TractionLoad,
    dof_per_element,
};
