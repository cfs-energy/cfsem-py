# Linear Filament

Hierarchical field solvers accept `skip=None`, `skip="near"`, or `skip="far"`. The value names the
interaction class to omit: `skip="near"` evaluates only accepted far-field summaries, while
`skip="far"` evaluates only direct near-field interactions. A full solve is therefore the sum of
those two filtered solves, up to floating-point roundoff. Filtering happens in the common evaluator,
so skipped kernel calculations are not performed.

## Fields

::: cfsem.flux_density_linear_filament

::: cfsem.flux_density_linear_filament_hierarchical

::: cfsem.vector_potential_linear_filament

::: cfsem.vector_potential_linear_filament_hierarchical

When `extra_diagnostics=True`, the result includes
`diagnostics.near_field_interaction_map`, a canonical SciPy CSC matrix with shape `(nsrc, ntgt)`.
Source indices are rows and target indices are columns. The map records the direct-interaction
classification for the target points supplied to the hierarchical solve; its stored data values are
structural markers only.

## Inductance

::: cfsem.inductance_linear_filaments

::: cfsem.inductance_linear_filaments_sparse

The sparse method evaluates every stored coordinate with three-point Gauss--Legendre integration
over the complete target segment. Thus, midpoint targets can classify near interactions cheaply,
while the resulting inductance entries still integrate along the full target segments:

```python
import cfsem

midpoints = tuple(start + 0.5 * delta for start, delta in zip(xyzfil, dlxyzfil))
far = cfsem.vector_potential_linear_filament_hierarchical(
    midpoints,
    xyzfil,
    dlxyzfil,
    current,
    wire_radius,
    theta=0.05,
    skip="near",  # far-only field
    extra_diagnostics=True,
)
near_pattern = far.diagnostics.near_field_interaction_map
near_inductance = cfsem.inductance_linear_filaments_sparse(
    xyzfil,
    dlxyzfil,
    xyzfil,
    dlxyzfil,
    near_pattern,
    wire_radius_src=wire_radius,
)
near_coupling = near_inductance.T @ current
```

The pointwise near vector potential and segment-integrated near inductance share a classification
pattern, but they are not numerically interchangeable. The map must be rebuilt if geometry,
`theta`, construction method, kernel acceptance logic, or acceptance-relevant source moments
change. For a self-coupled workflow that needs symmetric structural support, make that modeling
choice explicitly before evaluation:

```python
symmetric_pattern = (near_pattern + near_pattern.T).astype(bool).tocsc()
symmetric_pattern.sum_duplicates()
symmetric_pattern.sort_indices()
near_inductance = cfsem.inductance_linear_filaments_sparse(
    xyzfil, dlxyzfil, xyzfil, dlxyzfil, symmetric_pattern, wire_radius
)
```

## Force

::: cfsem.body_force_density_linear_filament

## Paths

::: cfsem.filament_coil

::: cfsem.filament_helix_path

::: cfsem.rotate_filaments_about_path
