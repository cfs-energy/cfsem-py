from typing import Literal

from numpy import float64, int64
from numpy.typing import NDArray

ArrayTriple = tuple[NDArray[float64], NDArray[float64], NDArray[float64]]
ArrayPair = tuple[NDArray[float64], NDArray[float64]]

class DimensionalityError(Exception): ...

def body_force_density_circular_filament_cartesian(*args: object, **kwargs: object) -> ArrayTriple: ...
def body_force_density_linear_filament(*args: object, **kwargs: object) -> ArrayTriple: ...
def ellipe(*args: object, **kwargs: object) -> NDArray[float64]: ...
def ellipk(*args: object, **kwargs: object) -> NDArray[float64]: ...
def filament_helix_path(*args: object, **kwargs: object) -> ArrayTriple: ...
def flux_circular_filament(*args: object, **kwargs: object) -> NDArray[float64]: ...
def flux_density_circular_filament(*args: object, **kwargs: object) -> ArrayPair: ...
def flux_density_circular_filament_cartesian(*args: object, **kwargs: object) -> ArrayTriple: ...
def flux_density_dipole(*args: object, **kwargs: object) -> ArrayTriple: ...
def flux_density_dipole_hierarchical(*args: object, **kwargs: object) -> ArrayTriple: ...
def flux_density_linear_filament(*args: object, **kwargs: object) -> ArrayTriple: ...
def flux_density_linear_filament_hierarchical(*args: object, **kwargs: object) -> ArrayTriple: ...
def flux_density_linear_filament_matrix(*args: object, **kwargs: object) -> NDArray[float64]: ...
def flux_density_point_segment(*args: object, **kwargs: object) -> ArrayTriple: ...
def flux_density_triangle_mesh(*args: object, **kwargs: object) -> ArrayTriple: ...
def flux_density_triangle_mesh_hierarchical(*args: object, **kwargs: object) -> ArrayTriple: ...
def flux_density_triangle_mesh_mapping(*args: object, **kwargs: object) -> ArrayTriple: ...
def gs_operator_order2(*args: object, **kwargs: object) -> ArrayTriple: ...
def gs_operator_order4(*args: object, **kwargs: object) -> ArrayTriple: ...
def inductance_linear_filaments(*args: object, **kwargs: object) -> NDArray[float64]: ...
def inductance_linear_filaments_matrix(*args: object, **kwargs: object) -> NDArray[float64]: ...
def inductance_piecewise_linear_filaments(*args: object, **kwargs: object) -> float: ...
def mutual_inductance_circular_to_linear(*args: object, **kwargs: object) -> NDArray[float64]: ...
def rotate_filaments_about_path(*args: object, **kwargs: object) -> ArrayTriple: ...
def triangle_mesh_current_density(*args: object, **kwargs: object) -> NDArray[float64]: ...
def triangle_mesh_flux_linkage_mapping_from_dipoles(*args: object, **kwargs: object) -> NDArray[float64]: ...
def triangle_mesh_force_mapping(*args: object, **kwargs: object) -> ArrayTriple: ...
def triangle_mesh_force_mapping_from_circular_filaments(*args: object, **kwargs: object) -> ArrayTriple: ...
def triangle_mesh_force_mapping_from_dipoles(*args: object, **kwargs: object) -> ArrayTriple: ...
def triangle_mesh_force_mapping_from_linear_filaments(*args: object, **kwargs: object) -> ArrayTriple: ...
def triangle_mesh_force_mapping_from_linear_filaments_matrix(
    *args: object, **kwargs: object
) -> NDArray[float64]: ...
def triangle_mesh_force_mapping_from_dipoles_matrix(*args: object, **kwargs: object) -> NDArray[float64]: ...
def triangle_mesh_force_mapping_from_circular_filaments_matrix(
    *args: object, **kwargs: object
) -> NDArray[float64]: ...
def triangle_mesh_inductance_mapping_from_circular_filaments(
    *args: object, **kwargs: object
) -> NDArray[float64]: ...
def triangle_mesh_inductance_mapping_from_linear_filaments(
    *args: object, **kwargs: object
) -> NDArray[float64]: ...
def triangle_mesh_inductance_matrix(*args: object, **kwargs: object) -> NDArray[float64]: ...
def triangle_mesh_quadrature_points(
    *args: object, **kwargs: object
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], int]: ...
def triangle_mesh_self_force_mapping(*args: object, **kwargs: object) -> ArrayTriple: ...
def vector_potential_circular_filament(*args: object, **kwargs: object) -> NDArray[float64]: ...
def vector_potential_dipole(*args: object, **kwargs: object) -> ArrayTriple: ...
def vector_potential_dipole_hierarchical(*args: object, **kwargs: object) -> ArrayTriple: ...
def vector_potential_linear_filament(*args: object, **kwargs: object) -> ArrayTriple: ...
def vector_potential_linear_filament_hierarchical(*args: object, **kwargs: object) -> ArrayTriple: ...
def vector_potential_linear_filament_matrix(*args: object, **kwargs: object) -> NDArray[float64]: ...
def vector_potential_point_segment(*args: object, **kwargs: object) -> ArrayTriple: ...
def vector_potential_triangle_mesh(*args: object, **kwargs: object) -> ArrayTriple: ...
def vector_potential_triangle_mesh_hierarchical(*args: object, **kwargs: object) -> ArrayTriple: ...
def vector_potential_triangle_mesh_mapping(*args: object, **kwargs: object) -> ArrayTriple: ...
def solenoid_stress_fem_assemble_model_2d_f32(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_assemble_model_2d_f64(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_cfsem_radial_material_f32(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_cfsem_radial_material_f64(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_infer_quad9_mesh_f32(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_infer_quad9_mesh_f64(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_quad_mesh_interpolation_operator_f32(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_quad_mesh_interpolation_operator_f64(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_quad_mesh_query_f32(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_quad_mesh_query_f64(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_quad_mesh_strain_operator_f32(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_quad_mesh_strain_operator_f64(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_quad_mesh_stress_operator_f32(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_quad_mesh_stress_operator_f64(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_isotropic_axisymmetric_material_f32(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_isotropic_axisymmetric_material_f64(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_isotropic_plane_strain_material_f32(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_isotropic_plane_strain_material_f64(*args: object, **kwargs: object) -> object: ...
def solenoid_stress_fem_isotropic_axisymmetric_thermal_material_f32(
    *args: object, **kwargs: object
) -> object: ...
def solenoid_stress_fem_isotropic_axisymmetric_thermal_material_f64(
    *args: object, **kwargs: object
) -> object: ...
def solenoid_stress_fem_isotropic_plane_strain_thermal_material_f32(
    *args: object, **kwargs: object
) -> object: ...
def solenoid_stress_fem_isotropic_plane_strain_thermal_material_f64(
    *args: object, **kwargs: object
) -> object: ...
def solenoid_stress_fem_orthotropic_axisymmetric_thermal_material_f32(
    *args: object, **kwargs: object
) -> object: ...
def solenoid_stress_fem_orthotropic_axisymmetric_thermal_material_f64(
    *args: object, **kwargs: object
) -> object: ...

class HierarchicalDipoles:
    """Reusable single-source-tree hierarchical dipole field solver."""

    def __init__(self, theta: float = 0.01, construction_method: str = "recursive") -> None:
        """Create a reusable hierarchical dipole solver.

        Args:
            theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
            construction_method: Source-tree construction method.
        """
        ...

    def set_sources(
        self,
        loc: ArrayTriple,
        outer_radius: NDArray[float64] | None = None,
    ) -> None:
        """Build or replace the source tree for dipole geometry.

        Args:
            loc: Dipole source coordinates as component arrays.
            outer_radius: Radius for the magnetized-sphere near-field treatment.
        """
        ...

    def flux_density(
        self,
        target: ArrayTriple,
        moment: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic flux density for the current source and target geometry.

        This is an approximate method, and no particular accuracy level is guaranteed.
        Truncated methods like this one may average entire local loop structures out of
        existence; as a result, maximum relative error is unbounded. This method must be
        tuned to a given use-case in order to be useful, and should not be used to
        calculate safety-related field limits.

        Args:
            target: Target point coordinates as component arrays.
            moment: Dipole magnetic moment components.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def vector_potential(
        self,
        target: ArrayTriple,
        moment: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic vector potential for the current source and target geometry.

        This is an approximate method, and no particular accuracy level is guaranteed.
        Truncated methods like this one may average entire local loop structures out of
        existence; as a result, maximum relative error is unbounded. This method must be
        tuned to a given use-case in order to be useful, and should not be used to
        calculate safety-related field limits.

        Args:
            target: Target point coordinates as component arrays.
            moment: Dipole magnetic moment components.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def accepted_source_levels(
        self,
        target: ArrayTriple,
        moment: ArrayTriple,
        field: Literal["b", "a"] = "b",
    ) -> NDArray[float64]:
        """Return the accepted source-tree level diagnostic for each target.

        Args:
            target: Target point coordinates as component arrays.
            moment: Dipole magnetic moment components.
            field: Field kernel to use for the acceptance diagnostic.

        Returns:
            Accepted source-tree level for each observation point.
        """
        ...

    def source_tree_aabbs(self) -> tuple[NDArray[float64], ...]:
        """Return source-tree AABB bounds and levels.

        Returns:
            Component arrays for minimum bounds, maximum bounds, and tree levels.
        """
        ...

class HierarchicalLinearFilaments:
    """Reusable single-source-tree hierarchical linear-filament field solver."""

    def __init__(self, theta: float = 0.05, construction_method: str = "recursive") -> None:
        """Create a reusable hierarchical linear-filament solver.

        Args:
            theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
            construction_method: Source-tree construction method.
        """
        ...

    def set_sources(
        self,
        xyzfil: ArrayTriple,
        dlxyzfil: ArrayTriple,
        wire_radius: NDArray[float64],
    ) -> None:
        """Build or replace the source tree for filament geometry.

        Args:
            xyzfil: Filament segment start coordinates as component arrays.
            dlxyzfil: Filament segment displacement vectors as component arrays.
            wire_radius: Wire radius for each segment.
        """
        ...

    def flux_density(
        self,
        target: ArrayTriple,
        current: NDArray[float64],
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic flux density for the current source and target geometry.

        This is an approximate method, and no particular accuracy level is guaranteed.
        Truncated methods like this one may average entire local loop structures out of
        existence; as a result, maximum relative error is unbounded. This method must be
        tuned to a given use-case in order to be useful, and should not be used to
        calculate safety-related field limits.

        Args:
            target: Target point coordinates as component arrays.
            current: Current in each filament segment.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def vector_potential(
        self,
        target: ArrayTriple,
        current: NDArray[float64],
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic vector potential for the current source and target geometry.

        This is an approximate method, and no particular accuracy level is guaranteed.
        Truncated methods like this one may average entire local loop structures out of
        existence; as a result, maximum relative error is unbounded. This method must be
        tuned to a given use-case in order to be useful, and should not be used to
        calculate safety-related field limits.

        Args:
            target: Target point coordinates as component arrays.
            current: Current in each filament segment.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def accepted_source_levels(
        self,
        target: ArrayTriple,
        current: NDArray[float64],
        field: Literal["b", "a"] = "b",
    ) -> NDArray[float64]:
        """Return the accepted source-tree level diagnostic for each target.

        Args:
            target: Target point coordinates as component arrays.
            current: Current in each filament segment.
            field: Field kernel to use for the acceptance diagnostic.

        Returns:
            Accepted source-tree level for each observation point.
        """
        ...

    def source_tree_aabbs(self) -> tuple[NDArray[float64], ...]:
        """Return source-tree AABB bounds and levels.

        Returns:
            Component arrays for minimum bounds, maximum bounds, and tree levels.
        """
        ...

class HierarchicalBoundaryElements:
    """Reusable single-source-tree hierarchical triangular boundary-element field solver."""

    def __init__(
        self,
        theta: float = 0.05,
        quad: str = "dunavant3",
        construction_method: str = "recursive",
    ) -> None:
        """Create a reusable hierarchical boundary-element solver.

        Args:
            theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
            quad: Triangle quadrature rule.
            construction_method: Source-tree construction method.
        """
        ...

    def set_sources(
        self,
        nodes: NDArray[float64],
        triangles: NDArray[int64],
    ) -> None:
        """Build or replace the source tree for triangle geometry.

        Args:
            nodes: Mesh node coordinates with one node per row.
            triangles: Triangle node indices with one triangle per row.
        """
        ...

    def flux_density(
        self,
        target: ArrayTriple,
        current_density: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic flux density for the current source and target geometry.

        This is an approximate method, and no particular accuracy level is guaranteed.
        Truncated methods like this one may average entire local loop structures out of
        existence; as a result, maximum relative error is unbounded. This method must be
        tuned to a given use-case in order to be useful, and should not be used to
        calculate safety-related field limits.

        Args:
            target: Target point coordinates as component arrays.
            current_density: Triangle-local current-density components.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def vector_potential(
        self,
        target: ArrayTriple,
        current_density: ArrayTriple,
        par: bool = False,
        out: ArrayTriple | None = None,
    ) -> ArrayTriple:
        """Evaluate magnetic vector potential for the current source and target geometry.

        This is an approximate method, and no particular accuracy level is guaranteed.
        Truncated methods like this one may average entire local loop structures out of
        existence; as a result, maximum relative error is unbounded. This method must be
        tuned to a given use-case in order to be useful, and should not be used to
        calculate safety-related field limits.

        Args:
            target: Target point coordinates as component arrays.
            current_density: Triangle-local current-density components.
            par: Whether to evaluate target batches in parallel.
            out: Optional output component arrays to fill.

        Returns:
            Output component arrays. If `out` is provided, returns `out`.
        """
        ...

    def accepted_source_levels(
        self,
        target: ArrayTriple,
        current_density: ArrayTriple,
        field: Literal["b", "a"] = "b",
    ) -> NDArray[float64]:
        """Return the accepted source-tree level diagnostic for each target.

        Args:
            target: Target point coordinates as component arrays.
            current_density: Triangle-local current-density components.
            field: Field kernel to use for the acceptance diagnostic.

        Returns:
            Accepted source-tree level for each observation point.
        """
        ...

    def source_tree_aabbs(self) -> tuple[NDArray[float64], ...]:
        """Return source-tree AABB bounds and levels.

        Returns:
            Component arrays for minimum bounds, maximum bounds, and tree levels.
        """
        ...
