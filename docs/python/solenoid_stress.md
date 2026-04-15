# Solenoid Stress & Strain

This package includes three complementary layers:

- a 1D finite-difference radial stress solver for winding-pack models with zero `rz` shear,
- a 2D axisymmetric quadrilateral FEM solver with reusable sparse load operators,
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
- reusable sparse load operators for body force, pressure, traction, and nodal-temperature thermal strain,
- quadrature-point recovery operators for strain and stress.

::: cfsem.solenoid_stress.axisymmetric_fem

## Analytic Reference Formulas

::: cfsem.solenoid_stress.s_long_solenoid

::: cfsem.solenoid_stress.s_radial_thick_wall_cylinder

::: cfsem.solenoid_stress.s_hoop_thick_wall_cylinder
