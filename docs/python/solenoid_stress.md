# Solenoid Stress & Strain

This package includes three complementary layers:

- a 1D finite-difference radial stress solver for winding-pack models with zero `rz` shear,
- a 2D quadrilateral FEM solver with axisymmetric and plane-strain formulations, reusable sparse load operators, and cached Rust-side sparse-LU solves,
- analytic reference formulas used for validation and convergence studies.

## 1D Finite-Difference Solver

::: cfsem.solenoid_stress.SolenoidStress1D

::: cfsem.solenoid_stress.SolenoidStress1DOperators

::: cfsem.solenoid_stress.solenoid_1d_structural_factor

::: cfsem.solenoid_stress.solenoid_1d_structural_rhs

## 2D FEM

The FEM path supports:

- axisymmetric and plane-strain structural formulations,
- `quad4`, inferred `quad9`, and explicit `quad9` elements,
- `gl3` and `gl4` quadrature,
- optional per-element in-plane material orientation angles,
- optional threaded stiffness assembly with `par=True`,
- reusable reduced-space operators for body force, pressure, traction, and nodal-temperature thermal strain,
- reduced quadrature-point recovery operators for strain and stress,
- direct sparse-LU reduced-system solves,
- `float64` numeric storage; `float32` inputs are accepted and normalized to `float64`,
- model-owned Dirichlet constraints applied during assembly.

The intended workflow is:

1. call `assemble_structural_2d(...)` once with mesh, materials, load topology, and prescribed Dirichlet values,
2. build each reduced load vector with `model.build_rhs(...)` or the exposed sparse operators,
3. solve with `model.solve(rhs)`, using the cached sparse-LU factorization,
4. recover quadrature strain and stress with `model.evaluate_quadrature(...)`.

By default, `model.solve(rhs)` uses the direct sparse-LU path and returns the full displacement
array. The sparse-LU factorization is built lazily on the first solve and then cached on the model
for repeated right-hand sides.

### Formulation Notes

The axisymmetric and plane-strain solvers share the same 2D quadrilateral mesh, two displacement
unknowns per node, and four-component strain/stress storage. The difference is how that 2D mesh
represents a 3D body.

For `formulation="axisymmetric"`, the coordinates are interpreted as `(r, z)`. Each quadrature
area sample represents a full ring, so stiffness and load integrals use the volume scale
`2*pi*r*dA`. The strain vector is `[rr, zz, tt, rz]`; the out-of-plane hoop strain is not an
independent displacement derivative, but is recovered from the radial displacement as
`epsilon_tt = u_r / r`.

For `formulation="plane_strain"`, the coordinates are interpreted as `(x, y)`. Each area sample
represents a prismatic slice with user-supplied `thickness`, so integrals use the volume scale
`thickness*dA`. The strain vector is `[xx, yy, zz, xy]`; the out-of-plane strain is constrained to
`epsilon_zz = 0`, while `sigma_zz` can still be nonzero through the constitutive matrix.

Plane strain and plane stress are different 2D reductions. Plane strain models a body that is long,
periodic, or otherwise constrained in the out-of-plane direction, with zero out-of-plane strain and
generally nonzero out-of-plane stress. Plane stress models a thin sheet or plate with traction-free
faces through the thickness, with zero out-of-plane stress and generally nonzero out-of-plane
strain. This FEM path currently implements axisymmetric and plane-strain reductions; it does not
implement a plane-stress constitutive reduction.

::: cfsem.solenoid_stress.fem2d

## Analytic Reference Formulas

::: cfsem.solenoid_stress.s_long_solenoid

::: cfsem.solenoid_stress.s_radial_thick_wall_cylinder

::: cfsem.solenoid_stress.s_hoop_thick_wall_cylinder
