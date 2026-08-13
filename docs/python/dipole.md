# Dipole

Hierarchical dipole solvers accept an optional interaction filter. `skip="near"` returns the
far-only result, `skip="far"` returns the direct near-only result, and the default `skip=None`
evaluates both. With `extra_diagnostics=True`, the diagnostics include the direct-interaction
pattern as a canonical `(nsrc, ntgt)` SciPy CSC matrix.

## Fields

::: cfsem.flux_density_dipole

::: cfsem.flux_density_dipole_hierarchical

::: cfsem.vector_potential_dipole

::: cfsem.vector_potential_dipole_hierarchical
