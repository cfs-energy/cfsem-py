//! Axisymmetric finite-element elasticity helpers for solenoid stress problems.
//!
//! The 2D-axisymmetric small-strain formulation implemented here uses the
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
//! A generalized force is the quantity that is work-conjugate to a generalized displacement
//! coordinate.  Its "force-like" character comes from units: because virtual work has units of
//! energy and displacement has units of distance, the conjugate quantity has units of energy per
//! distance.  Here the generalized coordinates are the nodal radial and axial displacements, so
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
//! The corresponding load-vector entries are generalized nodal forces rather than literal point
//! forces.  Entry `f[2a]` is the energy-per-distance quantity work-conjugate to the radial
//! displacement degree of freedom at node `a`, and `f[2a + 1]` is the corresponding axial
//! quantity.
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
//! - scales the contribution by the area weight `det(J) w` and by the additional
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
//! ## Load-specific virtual-work interpretation
//!
//! It is useful to read each load contribution as the answer to the question: what generalized
//! nodal force vector would do the same virtual work as the underlying distributed load, for every
//! displacement field representable by this element basis?  The element right-hand side is the sum
//! of those consistent nodal representations, so after assembly the global vector `f` is the total
//! external virtual-work functional written in nodal coordinates.
//!
//! ### Body force
//!
//! A body-force density `b = [b_r, b_z]` acts throughout the revolved element volume.  Its virtual
//! work is
//! `delta W_body = integral((delta u)^T b 2*pi*r dA)`,
//! which becomes
//! `delta u_e^T integral(N^T b 2*pi*r dA)`
//! after interpolation.  The corresponding consistent nodal load is therefore
//! `f_body = integral(N^T b 2*pi*r dA)`.
//!
//! In the code this is accumulated point-by-point inside the same volume quadrature loop that
//! forms the stiffness matrix.  Each quadrature point contributes a small force increment to the
//! radial and axial nodal degrees of freedom in proportion to the local shape values `N_i`.  This
//! means the nodal loads are not arbitrary lumped forces; they are the projection of the true
//! distributed volume loading onto the element basis.  If the body force is, for example, a
//! `J x B` Lorentz force density, then the assembled `f_body` is the mechanical forcing needed so
//! that the elastic response `K u` can balance the magnetic load in weak form.
//!
//! From an energy viewpoint, body force contributes a linear potential term
//! `- integral(u^T b 2*pi*r dA)`, so its nodal representation is the gradient of that potential
//! with respect to the nodal displacements.
//!
//! ### Pressure
//!
//! Pressure is a scalar normal load on an element face.  Its physical traction is `-p n`, so the
//! virtual work is
//! `delta W_pressure = integral((delta u)^T (-p n) 2*pi*r ds)`.
//! After interpolation this becomes
//! `delta u_e^T integral(N^T (-p n) 2*pi*r ds)`,
//! which defines
//! `f_pressure = integral(N^T (-p n) 2*pi*r ds)`.
//!
//! Pressure is different from a general traction because the load direction is determined by the
//! face geometry itself rather than being prescribed independently.  In the implementation, the
//! outward normal and the face Jacobian are bundled together as `normal_area = [t_z, -t_r]`, so
//! the integral is evaluated without separately normalizing the normal vector.  This preserves the
//! correct resultant force from a uniform pressure and ensures that the nodal load vector does the
//! same virtual work as the true distributed surface stress.
//!
//! In force-balance terms, pressure loads are how the formulation applies boundary tractions that
//! compress or separate the body through the face normal.  In energy terms, they contribute the
//! negative potential of the applied boundary traction against admissible displacements.
//!
//! ### Traction
//!
//! A traction load is a prescribed vector `t = [t_r, t_z]` on a face, expressed directly in the
//! global `(r, z)` directions.  Its virtual work is
//! `delta W_traction = integral((delta u)^T t 2*pi*r ds)`,
//! so the corresponding nodal load is
//! `f_traction = integral(N^T t 2*pi*r ds)`.
//!
//! Compared to pressure, traction does not derive its direction from the face normal.  It is the
//! right abstraction when the boundary load is known in global components, for example an imposed
//! axial pull, a radial support reaction represented as a load, or a prescribed tangential/shear
//! surface stress.  The line element enters through `|dx/ds|`, but the direction remains the user-
//! supplied global traction vector.
//!
//! Because the consistent load vector is obtained from `N^T t`, the resulting nodal forces
//! preserve the correct total force and moment for all displacement fields representable by the
//! element basis.  That is the key benefit of the consistent-load construction over ad hoc nodal
//! lumping.
//!
//! ### Thermal strain
//!
//! Thermal strain is fundamentally different from the previous three load types because it is not
//! an externally applied force density.  Instead, it is an eigenstrain
//! `epsilon_th = alpha * (T - T_ref)` representing the strain the material would adopt if it were
//! free to expand or contract without mechanical constraint.  The constitutive law is written as
//! `sigma = D (epsilon - epsilon_th)`, so the thermal part enters the weak form as
//! `delta W_thermal = - integral((delta epsilon)^T D epsilon_th 2*pi*r dA)`.
//! After interpolation this becomes
//! `- delta u_e^T integral(B^T D epsilon_th 2*pi*r dA)`,
//! which is why the equivalent nodal contribution is
//! `f_thermal = integral(B^T D epsilon_th 2*pi*r dA)`.
//!
//! This is best understood as an incompatibility load.  If the body were completely free, the
//! displacement field could match the thermal strain and the elastic stress would vanish.  When
//! boundary conditions or neighboring material prevent that free expansion, the difference
//! `epsilon - epsilon_th` produces stress, and the equivalent nodal thermal load is the term that
//! drives the structural solve toward the constrained thermoelastic equilibrium state.
//!
//! In this implementation `epsilon_th` is known before the structural solve because the
//! temperature field is treated as prescribed input data rather than as an additional structural
//! unknown.  The sequence is therefore:
//! - provide nodal temperatures on the mesh,
//! - interpolate those nodal values to each quadrature point with the same shape functions `N`,
//! - form `DeltaT = T - T_ref` using the per-material reference temperature,
//! - compute `epsilon_th = alpha * DeltaT`,
//! - assemble `f_thermal = integral(B^T D epsilon_th 2*pi*r dA)`,
//! - then solve the structural system for `u`.
//! In other words, the temperature field is an input to the structural problem, not one of its
//! unknowns.  The solver does not need to guess `epsilon_th`; it computes `epsilon_th` directly
//! from the supplied thermal state before it begins solving for displacement.
//! This makes the current formulation one-way coupled: temperature drives mechanics, but the
//! structural solve does not solve for temperature itself.
//!
//! The clearest energy interpretation comes from the elastic strain-energy density
//! `1/2 (epsilon - epsilon_th)^T D (epsilon - epsilon_th)`.  Expanding this expression gives the
//! quadratic mechanical term plus a linear coupling term in the nodal displacements.  That
//! linear term is exactly what appears on the right-hand side as the thermal load vector.  So
//! `f_thermal` is not an external "push" in the same sense as pressure or body force; it is the
//! nodal representation of the stress-free strain state that the structure would prefer to realize.
//!
//! ### How the actual load-vector entries are assembled
//!
//! The integral formulas above explain the continuum meaning of each load.  In the code, those
//! integrals are evaluated quadrature point by quadrature point to build a local element load
//! vector `f_e`, and that local vector is then scattered into the global right-hand side.
//!
//! For body force, one assumes an elementwise-constant load `b = [b_r, b_z]`.  At one volume
//! quadrature point `q`, the code forms the axisymmetric volume scale
//! `scale_q = 2*pi*r_q det(J_q) w_q`.
//! If node `i` has local radial degree of freedom `2i` and local axial degree of freedom
//! `2i + 1`, then the contribution from that quadrature point is
//! - `f_e[2i]     += scale_q N_i(q) b_r`,
//! - `f_e[2i + 1] += scale_q N_i(q) b_z`.
//! This is just the discrete form of `integral(N^T b 2*pi*r dA)`: each shape value `N_i(q)` tells
//! how much of the local distributed force should be assigned to node `i`.
//!
//! For pressure, each loaded face carries one scalar value `p`.  At one face quadrature point the
//! code computes the physical face tangent `dx/ds`, rotates it into
//! `normal_area = [t_z, -t_r]`, and uses the scale `scale_q = 2*pi*r_q w_q`.  The local load
//! entries then receive
//! - `f_e[2i]     += scale_q N_i(q) (-p) normal_area_r`,
//! - `f_e[2i + 1] += scale_q N_i(q) (-p) normal_area_z`.
//! The line-Jacobian is already embedded in `normal_area`, so pressure is assembled as a normal
//! traction without separately dividing by or multiplying by `|dx/ds|`.
//!
//! For traction, the supplied load is already a global vector `t = [t_r, t_z]`, so the code uses
//! the physical line-element scale directly:
//! `scale_q = 2*pi*r_q |dx/ds|_q w_q`.
//! The local entries receive
//! - `f_e[2i]     += scale_q N_i(q) t_r`,
//! - `f_e[2i + 1] += scale_q N_i(q) t_z`.
//! This is the face analogue of the body-force assembly: the shape functions distribute the
//! continuous boundary load into equivalent nodal generalized forces.
//!
//! Thermal strain is slightly different because it enters through stress rather than directly
//! through a force density.  At one volume quadrature point, the code first interpolates the
//! prescribed nodal temperatures,
//! `T_q = sum_j N_j(q) T_j`,
//! then forms
//! `DeltaT_q = T_q - T_ref`,
//! `epsilon_th,q = alpha DeltaT_q`,
//! and
//! `sigma_th,q = D epsilon_th,q`.
//! The local right-hand side then receives
//! `f_e[a] += scale_q sum_m B[m, a] sigma_th,q[m]`,
//! where again `scale_q = 2*pi*r_q det(J_q) w_q`.  In compact form this is exactly the quadrature
//! expansion of
//! `integral(B^T D epsilon_th 2*pi*r dA)`.
//!
//! After the local vector `f_e` has been accumulated, assembly is the scatter operation:
//! if local node `i` corresponds to global node `g`, then `f_e[2i]` adds into global row `2g`
//! and `f_e[2i + 1]` adds into global row `2g + 1`.  Contributions from neighboring elements sum
//! into the same global rows, which is why the final right-hand side represents the combined
//! generalized force seen by each global displacement degree of freedom.
//!
//! The reusable-load API in [`load_operators`] constructs the same objects in column form rather
//! than summing them immediately.  A body-force operator column is the global RHS produced by unit
//! radial or axial body force on one element.  A pressure operator column is the global RHS
//! produced by unit pressure on one loaded face.  A traction operator contributes two columns per
//! loaded face, for unit radial and unit axial traction.  A temperature operator column is the
//! global RHS produced by unit temperature at one node, while the per-material reference
//! temperature contributes a separate constant offset vector.  Multiplying those sparse operators
//! by the current load amplitudes reproduces the same assembled right-hand side that direct
//! quadrature assembly would have produced.
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
//! - [`crate::mesh::elements::quad2d::quad4`] defines the bilinear shape functions and
//!   reference-element geometry.
//! - [`crate::mesh::elements::quad2d::quad9`] defines the quadratic shape functions and
//!   reference-element geometry.
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
