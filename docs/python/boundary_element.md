# Boundary Element

Hierarchical boundary-element solvers accept `skip="near"` for a far-only result and `skip="far"`
for a direct near-only result; `skip=None` evaluates both. Extra diagnostics include the resulting
direct-interaction pattern as a canonical `(ntri, ntgt)` SciPy CSC matrix.

## Fields

Triangle B-field evaluation is analytic away from each finite source triangle. At a
point directly on a source triangle, that triangle's contribution is defined as zero
because the ideal current sheet has distinct one-sided limits. Use a small signed
normal offset when a particular side is required. The B-field APIs therefore have no
triangle-quadrature argument. Triangle vector-potential evaluation is also analytic,
and remains finite and continuous on triangle interiors, edges, and vertices. Its APIs
therefore no longer accept a triangle-quadrature argument either. Quadrature remains
available where it still controls target integration, including inductance and force.

::: cfsem.flux_density_triangle_mesh

::: cfsem.flux_density_triangle_mesh_hierarchical

::: cfsem.vector_potential_triangle_mesh

::: cfsem.vector_potential_triangle_mesh_hierarchical

## Field Mappings

::: cfsem.flux_density_triangle_mesh_mapping

::: cfsem.vector_potential_triangle_mesh_mapping

## Mesh Utilities

::: cfsem.triangle_mesh_current_density

::: cfsem.triangle_mesh_quadrature_points

## Inductance

::: cfsem.triangle_mesh_inductance_matrix

::: cfsem.triangle_mesh_inductance_mapping_from_linear_filaments

::: cfsem.triangle_mesh_inductance_mapping_from_circular_filaments

::: cfsem.triangle_mesh_flux_linkage_mapping_from_dipoles

## Force

::: cfsem.triangle_mesh_force_mapping

::: cfsem.triangle_mesh_self_force_mapping

::: cfsem.triangle_mesh_force_mapping_from_linear_filaments

::: cfsem.triangle_mesh_force_mapping_from_circular_filaments

::: cfsem.triangle_mesh_force_mapping_from_dipoles
