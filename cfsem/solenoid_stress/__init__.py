"""Structural solenoid stress utilities.

This subpackage includes three complementary toolsets:

- a 1D finite-difference radial stress solver for deck-of-cards winding-pack models,
- a 2D quadrilateral FEM implementation centered on reusable sparse load and
  recovery operators, with convenience methods for `build_rhs(...)`, `solve(...)`, and
  quadrature-field recovery,
- analytic reference formulas used for validation and convergence studies.
"""

from .solenoid_1d import (
    SolenoidStress1D,
    SolenoidStress1DOperators,
    solenoid_1d_structural_factor,
    solenoid_1d_structural_rhs,
)
from .solenoid_handcalc import s_long_solenoid
from .thermal_handcalc import s_thermal_long_cylinder_linear_temperature
from .thick_wall_cylinder_handcalc import (
    s_hoop_thick_wall_cylinder,
    s_radial_thick_wall_cylinder,
)
from .fem2d import (
    ElementMeasures,
    ElementQuadrature,
    ElevatedQuad9Mesh,
    QuadMeshInterpolation,
    QuadMeshQuery,
    QuadratureFieldSamples,
    Structural2DFEMModel,
    assemble_structural_2d,
    cfsem_radial_material,
    interpolate_quad_mesh_values,
    infer_quad9_mesh,
    isotropic_axisymmetric_material,
    isotropic_axisymmetric_thermal_material,
    isotropic_plane_strain_material,
    isotropic_plane_strain_thermal_material,
    orthotropic_axisymmetric_thermal_material,
    orthotropic_plane_strain_thermal_material,
    pack_material_tables_from_tags,
    quad_mesh_interpolation_operator,
    quad_mesh_strain_operator,
    query_quad_mesh,
)

__all__ = [
    "ElementMeasures",
    "ElementQuadrature",
    "ElevatedQuad9Mesh",
    "QuadMeshInterpolation",
    "QuadMeshQuery",
    "QuadratureFieldSamples",
    "SolenoidStress1D",
    "SolenoidStress1DOperators",
    "Structural2DFEMModel",
    "assemble_structural_2d",
    "cfsem_radial_material",
    "interpolate_quad_mesh_values",
    "infer_quad9_mesh",
    "isotropic_axisymmetric_material",
    "isotropic_axisymmetric_thermal_material",
    "isotropic_plane_strain_material",
    "isotropic_plane_strain_thermal_material",
    "orthotropic_axisymmetric_thermal_material",
    "orthotropic_plane_strain_thermal_material",
    "pack_material_tables_from_tags",
    "quad_mesh_interpolation_operator",
    "quad_mesh_strain_operator",
    "query_quad_mesh",
    "s_hoop_thick_wall_cylinder",
    "s_long_solenoid",
    "s_radial_thick_wall_cylinder",
    "s_thermal_long_cylinder_linear_temperature",
    "solenoid_1d_structural_factor",
    "solenoid_1d_structural_rhs",
]
