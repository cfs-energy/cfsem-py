# Boundary Element

## Fields

Triangle B-field evaluation is analytic away from each finite source triangle. At a
point directly on a source triangle, that triangle's contribution is defined as zero
because the ideal current sheet has distinct one-sided limits. Use a small signed
normal offset when a particular side is required. The B-field APIs therefore have no
triangle-quadrature argument; vector-potential and inductance APIs retain one where it
still controls target integration.

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
