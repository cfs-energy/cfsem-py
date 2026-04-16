# Solenoid Stress & Strain

This package includes three complementary layers:

- a 1D finite-difference radial stress solver for winding-pack models with zero `rz` shear,
- a 2D axisymmetric quadrilateral FEM solver with a constrained reduced model, reusable sparse load operators, and a cached Rust-side LU solve,
- analytic reference formulas used for validation and convergence studies.

## 1D Finite-Difference Solver

::: cfsem.solenoid_stress.SolenoidStress1D

::: cfsem.solenoid_stress.SolenoidStress1DOperators

::: cfsem.solenoid_stress.solenoid_1d_structural_factor

::: cfsem.solenoid_stress.solenoid_1d_structural_rhs

## Axisymmetric FEM

The FEM path supports:

- `quad4` and inferred `quad9` elements,
- `gl3` and `gl4` quadrature,
- reusable reduced-space operators for body force, pressure, traction, and nodal-temperature thermal strain,
- reduced quadrature-point recovery operators for strain and stress,
- model-owned Dirichlet constraints applied during assembly.

The intended workflow is:

1. call `assemble_axisymmetric(...)` once with mesh, materials, load topology, and prescribed Dirichlet values,
2. build each reduced load vector with `model.build_rhs(...)` or the exposed sparse operators,
3. solve with `model.solve(rhs)`, which reuses a cached LU factorization,
4. recover quadrature strain and stress with `model.evaluate_quadrature(...)`.

::: cfsem.solenoid_stress.axisymmetric_fem

## Analytic Reference Formulas

::: cfsem.solenoid_stress.s_long_solenoid

::: cfsem.solenoid_stress.s_radial_thick_wall_cylinder

::: cfsem.solenoid_stress.s_hoop_thick_wall_cylinder
