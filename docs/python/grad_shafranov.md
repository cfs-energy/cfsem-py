# Grad-Shafranov

These helpers provide the axisymmetric finite-difference flux solve path.

Key preconditions:

- `flux_solver()` and `solve_flux_axisymmetric()` require strictly increasing, regular
  `r` and `z` grids with `r > 0` and at least 7 points on each axis.
- `solve_flux_axisymmetric()` expects `meshes = np.meshgrid(*grids, indexing="ij")`
  and `current_density` on the same `(nr, nz)` layout, with zero current on the
  finite-difference boundary.
- `gradient_order4()` and `calc_flux_density_from_flux()` require at least 6 points
  on each axis, and `calc_flux_density_from_flux()` also requires `rmesh > 0`.

::: cfsem.gs_operator_order2

::: cfsem.gs_operator_order4

::: cfsem.gradient_order4

::: cfsem.calc_flux_density_from_flux

::: cfsem.flux_solver

::: cfsem.solve_flux_axisymmetric
