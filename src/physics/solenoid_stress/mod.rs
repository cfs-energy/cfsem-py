//! Axisymmetric finite-element elasticity with anisotropic materials, surface traction/pressure,
//! body force loads, and thermal strain.
//!
//! ## Origin
//!
//! The equations implemented here were first published in 1965 by Wilson at Rocketdyne,
//! just two years after the first computerized finite-element program was run
//! on an early punchcard machine, because the space race had drawn attention
//! and resources toward the analysis of axisymmetric pressure vessels, heat shields,
//! nozzles, and turbomachinery which combine pressure, thermal, and centrifugal body force loads.
//!
//! Here, we reuse those early results in an equivalent form, but for a different purpose:
//! analysis of axisymmetric electromagnets under pressure, thermal, and Lorentz body force loads.
//!
//! ## Form of the Stiffness Matrix
//!
//! The 2D-axisymmetric small-strain formulation implemented here uses the
//! displacement-based Galerkin finite-element stiffness construction
//! `K_e = integral(B^T D B 2*pi*r dA)`.
//!
//! Starting with this relatively general framing, the axisymmetry is essentially entirely contained in
//! the factor of `2*pi*r` and the relation `epsilon_tt = u_r/r` which ties hoop strain to radial displacement
//! in order to eliminate the independence of the 3rd dimension.
//!
//! The stiffness formula is best read from right to left.  For one element with nodal displacement vector
//! `u_e = [u_r1, u_z1, u_r2, u_z2, ...]^T`, the strain at a quadrature point is
//! `epsilon = B u_e`, and the elastic stress-strain law gives `sigma = D epsilon = D B u_e`.  Converting
//! that pointwise stress field back into equivalent nodal forces by virtual work gives
//! `f_int,e = integral(B^T sigma 2*pi*r dA) = integral(B^T D B u_e 2*pi*r dA) = K_e u_e`.
//! So `K_e` is the element stiffness matrix: it maps one element displacement pattern to the
//! internal restoring forces associated with that same element's nodal degrees of freedom.
//!
//! ## Glossary
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
//! - `D` is the constitutive matrix, the per-material `4 x 4` matrix for the elastic
//!   stress-strain law in axisymmetric stress/strain ordering,
//! - `epsilon` is the strain vector at a point,
//! - `sigma` is the stress vector at a point,
//! - `B^T` maps stress back to equivalent nodal forces,
//! - `dA` is the differential area in the `(r, z)` cross-section, and
//! - `2*pi*r` is the axisymmetric revolution factor that converts cross-sectional area into the
//!   volume of the corresponding ring in 3D.
//!
//! ## Governing Equations & Solution Representation
//!
//! The interpolation statement `u = N u_e` means that the displacement field inside one element is
//! reconstructed from the element nodal displacement values.  At any point in the element,
//! `u(r, z) = [u_r(r, z), u_z(r, z)]^T`, while `u_e` stores the nodal radial and axial
//! displacements.  If the element has nodes `1..n`, then the interpolation matrix has the block
//! form
//! - `[ N_1  0    N_2  0   ...  N_n  0 ]`,
//! - `[ 0    N_1  0    N_2 ...  0    N_n ]`,
//!
//! where the scalar shape functions `N_i(r, z)` are evaluated at the point of interest.
//! Multiplying by `u_e` gives
//! - `u_r(r, z) = N_1 u_r1 + N_2 u_r2 + ... + N_n u_rn`,
//! - `u_z(r, z) = N_1 u_z1 + N_2 u_z2 + ... + N_n u_zn`.
//!
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
//! The governing equations are the strain-displacement equation, the
//! elastic stress-strain law, the equation of static equilibrium for stresses, and the boundary
//! conditions on displacement and stress.
//!
//! ## Weak Form & Loads Representation
//!
//! In axisymmetric small-strain elasticity, the weak form is: find a displacement field `u`
//! satisfying the prescribed displacement boundary conditions such that, for every admissible
//! virtual displacement field `delta u`,
//! `integral((delta epsilon)^T sigma 2*pi*r dA)
//!  - integral((delta u)^T b 2*pi*r dA)
//!  - integral((delta u)^T t 2*pi*r ds) = 0`.
//!
//! Here `b` is body-force density, `t` is an applied traction on the traction boundary, and
//! `sigma = D (epsilon - epsilon_th)` if thermal strain is active.  When no traction or pressure
//! load is prescribed on a boundary segment, that boundary is traction-free in this weak sense.
//!
//! After approximating the displacement field with element shape functions, one writes
//! - `u = N u_e`,
//! - `delta u = N delta u_e`,
//! - `epsilon = B u_e`,
//! - `delta epsilon = B delta u_e`,
//!
//! where `u_e` collects the element nodal displacement degrees of freedom.  Substituting these
//! into the virtual-work statement gives
//! - `delta W_int = delta u_e^T [integral(B^T D B 2*pi*r dA)] u_e`,
//! - `delta W_ext = delta u_e^T f_e`.
//!
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
//! - applies the per-material `4 x 4` elastic stress-strain matrix `D` locally to form
//!   `B^T D B` for the local stiffness contribution, and
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
//! appears on the right-hand side because the elastic stress-strain law is evaluated as
//! `sigma = D (epsilon - epsilon_th)`, so the `D epsilon_th` contribution is moved to the load
//! vector as an equivalent nodal force.
//!
//! ## Stress-Strain Relation
//!
//! At each quadrature point the axisymmetric displacement field produces the strain vector
//! `epsilon = [e_rr, e_zz, e_tt, g_rz]^T`,
//! where
//! - `e_rr = du_r/dr`,
//! - `e_zz = du_z/dz`,
//! - `e_tt = u_r / r`,
//! - `g_rz = du_r/dz + du_z/dr`.
//!
//! In the implementation this kinematic map is written as
//! `epsilon = B u_e`,
//! where `u_e` is the vector of element nodal displacements and `B` is the strain-displacement
//! matrix assembled from the shape-function gradients and the hoop term `N_i / r`.
//!
//! The stress vector is stored in the matching ordering
//! `sigma = [sigma_rr, sigma_zz, sigma_tt, tau_rz]^T`.
//! For a purely mechanical solve the elastic stress-strain law is
//! `sigma = D epsilon`.
//! When thermal strain is active, the elastic strain is
//! `epsilon_elastic = epsilon - epsilon_th`,
//! and the elastic stress-strain law becomes
//! `sigma = D (epsilon - epsilon_th)`.
//! The matrix `D` therefore maps strain-like quantities in that axisymmetric ordering into the
//! corresponding stress-like quantities in the same ordering.  In the implementation `D` is not
//! assembled into any global stress-strain operator.  Instead, each quadrature point applies the
//! relevant per-material `4 x 4` matrix directly, so stiffness assembly and stress recovery use
//! the same elastic stress-strain law in a matrix-free local form.
//!
//! From an energy viewpoint, the strain-energy density is
//! `1/2 epsilon^T D epsilon`
//! in the purely mechanical case, or
//! `1/2 (epsilon - epsilon_th)^T D (epsilon - epsilon_th)`
//! in the thermoelastic case.  The stiffness matrix comes from the quadratic part of that energy,
//! while the thermal load vector comes from the linear coupling term generated by the thermal
//! offset.
//!
//! ## Stiffness-Matrix Assembly
//!
//! The left-hand side is assembled from the bilinear form
//! `K_e = integral(B^T D B 2*pi*r dA)`.
//! In the code this integral is evaluated quadrature point by quadrature point to build a local
//! dense element matrix `K_e`, and that local matrix is then scattered into the global sparse
//! stiffness matrix `K`.
//!
//! At one volume quadrature point `q`, the solver has:
//! - shape values `N_i(q)`,
//! - physical shape gradients,
//! - the local radius `r_q`,
//! - the elastic stress-strain matrix `D`,
//! - and the axisymmetric scale `scale_q = 2*pi*r_q det(J_q) w_q`.
//!
//! From these it builds the strain-displacement matrix `B_q` and then forms the dense
//! quadrature-point kernel
//! `scale_q B_q^T D B_q`.
//!
//! The important implementation detail is that `D` is applied only as this local `4 x 4`
//! contribution.  There is no assembled global stress-strain matrix; the backend computes `D B_q`
//! directly for each quadrature point and accumulates the resulting `B_q^T (D B_q)` contribution
//! into the element stiffness matrix.
//!
//! Entry by entry, this means that each local matrix coefficient receives
//! `K_e[a, b] += scale_q sum_m B_q[m, a] (D B_q)[m, b]`.
//! So one stiffness entry measures how displacement degree of freedom `b` contributes to stress at
//! the quadrature point, and how that stress contributes to the internal generalized force
//! associated with degree of freedom `a`.
//!
//! This is the matrix form of the internal virtual-work contribution
//! `delta u_e^T K_e u_e`.
//! Because the same bilinear form is used in both argument slots, the assembled element stiffness
//! is symmetric when `D` is symmetric.  The factor `2*pi*r_q` makes material at larger radius
//! contribute more strongly because the same cross-sectional patch represents a larger 3D ring.
//!
//! After the local dense matrix `K_e` has been accumulated, the code scatters it into global
//! sparse triplets.  If local node `i` maps to global node `g_i`, then local degree of freedom
//! `2i` maps to global row or column `2g_i`, and local degree of freedom `2i + 1` maps to
//! `2g_i + 1`.  Every local entry `K_e[a, b]` is therefore added into the global location
//! `K[A, B]`, where `A` and `B` are the corresponding global displacement indices.  Contributions
//! from neighboring elements sum into the same global matrix entries, which is how the assembled
//! matrix enforces compatibility and equilibrium across the mesh.
//!
//! Stress and strain recovery use the same element-owned point locations.  Matrix-free recovery on
//! [`Structural2dModel`] evaluates `B_q u_e` or `D B_q u_e` directly at each location; sparse
//! recovery operators store the same row actions for workflows that want reusable matrices.  As in
//! stiffness assembly, stress recovery uses the individual per-material `4 x 4` elastic
//! stress-strain matrix at each quadrature point rather than through any assembled global `D`
//! operator.  Applying those evaluations to the global displacement vector therefore recovers
//! `epsilon_q = B_q u_e` and `sigma_q = D epsilon_q`, or `sigma_q = D (epsilon_q - epsilon_th,q)`
//! when thermal strain is present.
//!
//! ## Load-Specific Virtual-Work Interpretation
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
//! free to expand or contract without mechanical constraint.  The elastic stress-strain law is written as
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
//!
//! In other words, the temperature field is an input to the structural problem, not one of its
//! unknowns.  The solver does not need to guess `epsilon_th`; it computes `epsilon_th` directly
//! from the supplied thermal state before it begins solving for displacement.
//!
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
//! integrals are evaluated quadrature point by quadrature point to build sparse load operators.
//! Applying one of those operators to a vector of load amplitudes produces the global right-hand
//! side contribution for that load type.
//!
//! For body force, one assumes an elementwise-constant load `b = [b_r, b_z]`.  The operator has
//! two columns per element: one for unit radial body force on that element and one for unit axial
//! body force.  At one volume quadrature point `q`, the code forms the axisymmetric volume scale
//! `scale_q = 2*pi*r_q det(J_q) w_q`.
//! If node `i` has local radial degree of freedom `2i` and local axial degree of freedom
//! `2i + 1`, then the corresponding operator column entries receive
//! - `scale_q N_i(q)` in the radial row for the unit-radial-body-force column,
//! - `scale_q N_i(q)` in the axial row for the unit-axial-body-force column.
//!
//! Multiplying those columns by the actual per-element `b_r` and `b_z` values reproduces
//! `integral(N^T b 2*pi*r dA)`.
//!
//! For pressure, each loaded face contributes one column corresponding to unit pressure on that
//! face.  At one face quadrature point the code computes the physical face tangent `dx/ds`,
//! rotates it into `normal_area = [t_z, -t_r]`, and uses the scale `scale_q = 2*pi*r_q w_q`.
//! The relevant operator column entries then receive
//! - `scale_q N_i(q) (-normal_area_r)` in the radial row,
//! - `scale_q N_i(q) (-normal_area_z)` in the axial row.
//!
//! The line-Jacobian is already embedded in `normal_area`, so pressure is represented as a normal
//! traction without separately dividing by or multiplying by `|dx/ds|`.
//!
//! For traction, the operator has two columns per loaded face: one for unit radial traction and
//! one for unit axial traction.  The code uses the physical line-element scale
//! `scale_q = 2*pi*r_q |dx/ds|_q w_q`.
//! The corresponding column entries receive
//! - `scale_q N_i(q)` in the radial row for the unit-radial-traction column,
//! - `scale_q N_i(q)` in the axial row for the unit-axial-traction column.
//!
//! This is the face analogue of the body-force operator: the shape functions distribute the
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
//! The thermal operator has one column per temperature degree of freedom.  Column `j` represents
//! unit temperature at node `j`, so the quadrature contribution to that column is
//! `scale_q N_j(q) B^T sigma_th,unit`,
//! where `sigma_th,unit = D alpha`.  The per-material reference temperature is an offset rather
//! than a variable degree of freedom, so it contributes a separate constant right-hand-side vector
//! instead of additional operator columns.
//!
//! For every load type, neighboring elements and faces add their quadrature contributions into the
//! same global rows, so each operator column is already a fully assembled global load pattern.
//! One-shot assembly in Python forms the final right-hand side by multiplying those operators by
//! the requested load amplitudes and summing the resulting vectors.
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
//!
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
//! - `geometry` adds axisymmetric validation and evaluates the `2*pi*r`-weighted element
//!   summaries needed by the structural solver.
//! - `axisym` constructs the axisymmetric strain operator and local stiffness kernel.
//! - `loads` applies distributed loads matrix-free and builds sparse linear maps from load
//!   amplitudes or nodal temperatures to the global right-hand side.
//! - `assembly` assembles the stiffness matrix into sparse triplets.
//! - `model` owns structural solve assembly and matrix-free recovery at located element points.
//! - [`crate::mesh::quad2d`] builds sparse interpolation, strain, and stress recovery operators for
//!   quadrilateral meshes.
//!
//! References:
//! - Allan F. Bower, *Applied Mechanics of Solids*, CRC Press, 2009.  See especially Section 8.1
//!   for the displacement-based finite-element formulation and Table 8.3 for 2D interpolation
//!   functions.
//! - E. L. Wilson, "Structural Analysis of Axisymmetric Solids," *AIAA Journal*, 3(12), pp. 2269-2274, December 1965. doi:10.2514/3.3356.
//! - R. A. Mitchell, R. M. Woolley, and C. R. Fisher, "Formulation and experimental verification of an axisymmetric finite-element structural analysis," *Journal of Research of the National Bureau of Standards Section C*, 75C, 1971.
//! - I. Fried, "Notes on the finite element analysis of the axisymmetric elastic solid," *International Journal of Solids and Structures*, 10(3), 1974.

mod assembly;
mod axisym;
mod convenience;
mod family;
mod geometry;
mod loads;
mod model;
#[cfg(test)]
mod test_utils;
mod types;

pub use crate::mesh::QuadratureRule;
pub(crate) use axisym::build_b_matrix;
pub use convenience::{
    ElevatedQuad9Mesh, Structural2dElementMeasures, cfsem_radial_material, infer_quad9_mesh,
    isotropic_axisymmetric_material, isotropic_axisymmetric_thermal_material,
    orthotropic_axisymmetric_thermal_material, rotate_material_in_plane,
    rotate_thermal_expansion_in_plane, rotate_thermal_material_in_plane,
};
pub use model::{
    Structural2dElementType, Structural2dElements, Structural2dModel, assemble_structural_2d,
};
pub(crate) use types::validate_element_material_inputs;
pub use types::{
    DOF_PER_NODE, PressureLoad, Structural2dFormulation, ThermalMaterial, TractionLoad,
    dof_per_element,
};
