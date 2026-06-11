"""2D structural elasticity finite-element assembly.

This module provides a small displacement-based quadrilateral FEM solver for axisymmetric and
plane-strain structural reductions. The backend stores the reduced stiffness matrix and evaluates
loads and quadrature recovery matrix-free unless sparse operators are explicitly exported.

The element formulation follows the standard small-strain Galerkin construction

`K_e = integral(B^T D B c dA)` where `c` is `2*pi*r` for axisymmetric and the thickness of the
planar domain for plane strain.

with consistent body-force, surface-pressure, and surface-traction load vectors. The axisymmetric
engineering-strain vector is ordered as `[e_rr, e_zz, e_tt, g_rz]`. In Bower's terminology, the
underlying equations are the strain-displacement equation, the elastic stress-strain law, the
equation of static equilibrium for stresses, and the boundary conditions on displacement and
stress.

References:
    [1] Allan F. Bower,
        *Applied Mechanics of Solids*,
        CRC Press, 2009.
        See especially Section 8.1 and Table 8.3 for the general displacement-based
        finite-element construction and 2D interpolation functions.

    [2] E. L. Wilson,
        "Structural Analysis of Axisymmetric Solids,"
        *AIAA Journal*, 3(12), pp. 2269-2274, 1965.

    [3] R. A. Mitchell, R. M. Woolley, and C. R. Fisher,
        "Formulation and experimental verification of an axisymmetric finite-element structural
        analysis,"
        *Journal of Research of the National Bureau of Standards Section C*, 75C, 1971.

    [4] I. Fried,
        "Notes on the finite element analysis of the axisymmetric elastic solid,"
        *International Journal of Solids and Structures*, 10(3), 1974.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

import cfsem.cfsem as _cfsem_bindings

_assemble_model_2d_f64 = _cfsem_bindings.solenoid_stress_fem_assemble_model_2d_f64
_cfsem_radial_material_f64 = _cfsem_bindings.solenoid_stress_fem_cfsem_radial_material_f64
_infer_quad9_mesh_f64 = _cfsem_bindings.solenoid_stress_fem_infer_quad9_mesh_f64
_quad_mesh_interpolation_operator_f64 = (
    _cfsem_bindings.solenoid_stress_fem_quad_mesh_interpolation_operator_f64
)
_quad_mesh_query_f64 = _cfsem_bindings.solenoid_stress_fem_quad_mesh_query_f64
_quad_mesh_strain_operator_f64 = _cfsem_bindings.solenoid_stress_fem_quad_mesh_strain_operator_f64
_quad_mesh_stress_operator_f64 = _cfsem_bindings.solenoid_stress_fem_quad_mesh_stress_operator_f64
_isotropic_axisymmetric_material_f64 = _cfsem_bindings.solenoid_stress_fem_isotropic_axisymmetric_material_f64
_isotropic_plane_strain_material_f64 = _cfsem_bindings.solenoid_stress_fem_isotropic_plane_strain_material_f64
_isotropic_axisymmetric_thermal_material_f64 = (
    _cfsem_bindings.solenoid_stress_fem_isotropic_axisymmetric_thermal_material_f64
)
_isotropic_plane_strain_thermal_material_f64 = (
    _cfsem_bindings.solenoid_stress_fem_isotropic_plane_strain_thermal_material_f64
)
_orthotropic_axisymmetric_thermal_material_f64 = (
    _cfsem_bindings.solenoid_stress_fem_orthotropic_axisymmetric_thermal_material_f64
)

ArrayLike = npt.ArrayLike
Float64Array = npt.NDArray[np.float64]
UInt64Array = npt.NDArray[np.uint64]
_QUAD_FACE_NODE_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3), (3, 0))


def _as_float64_array(data: Any) -> Float64Array:
    """Convert binding output to a NumPy float64 array with an explicit static type."""

    return np.asarray(data, dtype=np.float64)


def _as_uint64_array(data: Any) -> UInt64Array:
    """Give static type checkers the binding dtype without changing runtime dtype."""

    return np.asarray(data)


def _csr_matrix_from_binding(
    binding: tuple[ArrayLike, ArrayLike, ArrayLike, int, int],
) -> sp.csr_matrix:
    vals, indices, indptr, nrow, ncol = binding
    return sp.csr_matrix(
        (
            np.asarray(vals, dtype=np.float64),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(int(nrow), int(ncol)),
    )


def _csc_matrix_from_binding(
    binding: tuple[ArrayLike, ArrayLike, ArrayLike, int, int],
) -> sp.csc_matrix:
    vals, indices, indptr, nrow, ncol = binding
    return sp.csc_matrix(
        (
            np.asarray(vals, dtype=np.float64),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(int(nrow), int(ncol)),
    )


@dataclass(frozen=True, slots=True)
class ElementMeasures:
    """Per-element cross-section area and represented volume.

    `areas` and `volumes` both have shape `(nelem,)`.
    `areas` has units `[area]` and `volumes` has units `[volume]`.
    """

    areas: Float64Array
    volumes: Float64Array


@dataclass(frozen=True, slots=True)
class PointLocations:
    """Element-owned physical and reference point locations.

    A location is a physical point together with the element that owns or is nearest to that point
    and the corresponding element-local reference coordinates. Recovery and sparse operator
    construction use `element_indices` and `reference_points` as the source of truth; `points` is
    included for caller inspection, plotting, and compatibility with mesh-query outputs.

    `points` has shape `(npoint, 2)` and units `[length]`. `element_indices` has shape
    `(npoint,)` and stores unitless analysis-element indices. `reference_points` has shape
    `(npoint, 2)` and stores unitless coordinates in the element's `[-1, 1]^2` reference domain.
    `element_type` records the element family that produced the locations so model methods can
    reject locations from an incompatible mesh.

    """

    points: Float64Array
    element_indices: npt.NDArray[np.int64]
    reference_points: Float64Array
    element_type: str


@dataclass(frozen=True, slots=True)
class Quadrature:
    """Element-major quadrature locations and mapped integration weights.

    `locations` stores the physical points, owning elements, and reference coordinates used by
    recovery and sparse operator methods. `weights_area` and `weights_volume` have shape
    `(npoint,)`; reshape them as `(nelem, points_per_element)` for integrating quantities over
    elements.
    """

    locations: PointLocations
    weights_area: Float64Array
    weights_volume: Float64Array
    points_per_element: int


@dataclass(frozen=True, slots=True)
class ElevatedQuad9Mesh:
    """Explicit 9-node analysis mesh inferred from a corner-only quad4 mesh.

    `analysis_elements` use the local quad9 ordering:
    - corners `0..3` in counter-clockwise order `[bottom-left, bottom-right, top-right, top-left]`
    - midsides `4..7` on faces `[bottom, right, top, left]`
    - center node `8`

    `input_nodes` and `analysis_nodes` have units `[length]`.
    """

    input_nodes: Float64Array
    input_elements: UInt64Array
    analysis_nodes: Float64Array
    analysis_elements: UInt64Array
    corner_node_indices: npt.NDArray[np.int64]
    midside_node_indices: npt.NDArray[np.int64]
    center_node_indices: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class QuadMeshInterpolation:
    """Interpolated nodal values and element-location metadata for query points.

    `values` has shape `(npoint, ...)`, where `...` is the trailing shape of the nodal values.
    `element_indices` stores the nearest element used for interpolation. `inside` reports whether
    the nearest-element distance was within the containment tolerance.
    """

    values: Float64Array
    element_indices: npt.NDArray[np.int64]
    reference_points: Float64Array
    inside: npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class QuadMeshQuery:
    """One-pass geometric query results for points in a 2D quadrilateral mesh.

    The query stores nearest-node, nearest-element, and nearest-face data for each query point.
    Interpolation and recovery operators can reuse this object without repeating the mesh search.
    For contained points, the nearest element is the containing element and
    `nearest_element_distances` is zero to numerical tolerance.
    """

    nodes: Float64Array
    elements: UInt64Array
    points: Float64Array
    element_type: str
    nearest_node_indices: npt.NDArray[np.int64]
    nearest_node_points: Float64Array
    nearest_node_distances: Float64Array
    nearest_element_indices: npt.NDArray[np.int64]
    nearest_element_reference_points: Float64Array
    nearest_element_points: Float64Array
    nearest_element_distances: Float64Array
    nearest_face_element_indices: npt.NDArray[np.int64]
    nearest_face_local_faces: npt.NDArray[np.int64]
    nearest_face_reference_coordinates: Float64Array
    nearest_face_points: Float64Array
    nearest_face_distances: Float64Array

    def point_locations(self) -> PointLocations:
        """Return nearest-element locations for recovery and sparse operators.

        The returned locations reuse the element ownership and reference coordinates already found
        by the mesh query. No additional mesh search or point projection is performed. Query points
        outside the mesh are represented by their nearest projected element points; callers that
        need strict containment should check `nearest_element_distances` before using the locations.

        Returns:
            PointLocations: Element-owned nearest-element locations.
        """

        return PointLocations(
            points=self.nearest_element_points,
            element_indices=self.nearest_element_indices,
            reference_points=self.nearest_element_reference_points,
            element_type=self.element_type,
        )


class Structural2DFEMModel:
    """Reusable 2D structural FEM model with sparse operators and reduced solve state.

    Structural FEM numeric arrays are `float64`; floating inputs must already use `float64` arrays.

    Load and stiffness operators are exported from the Rust backend on demand:
    - `body_force_to_rhs`, `pressure_to_rhs`, `traction_to_rhs`, and `temperature_to_rhs`
      map load amplitudes to the reduced structural right-hand side,
    Field recovery uses explicit `PointLocations` objects. Use `quadrature().locations` for quadrature
    locations, `locate_points(...)` for arbitrary physical points,
    `locate_points_in_elements(...)` when element ownership is already known, or
    `QuadMeshQuery.point_locations()` to reuse an existing mesh query.

    Matrix-free field methods (`strain`, `stress`, `thermal_strain`, `thermal_stress`) evaluate
    values directly at supplied locations. Sparse operator methods (`interpolation_operator`,
    `strain_operator`, `stress_operator`) materialize user-owned SciPy matrices for workflows that
    apply the same located recovery many times.

    Key public array shapes and units:
    - `stiffness` has shape `(ndof_reduced, ndof_reduced)` with entry units
      `[generalized force / displacement] = [energy / distance^2]`,
    - `body_force_to_rhs` has shape `(ndof_reduced, 2 * nelem)` with entry units `[volume]`,
    - `pressure_to_rhs` has shape `(ndof_reduced, n_pressure_faces)` with entry units `[area]`,
    - `traction_to_rhs` has shape `(ndof_reduced, 2 * n_traction_faces)` with entry units
      `[area]`,
    - `temperature_to_rhs` has shape `(ndof_reduced, n_temperature_nodes)` with entry units
      `[generalized force / temperature] = [energy / (distance * temperature)]`.

    `input_nodes` and `analysis_nodes` have shape `(nnode, 2)` and units `[length]`.
    `input_elements` and `analysis_elements` expose the original and analysis connectivity.
    """

    def __init__(
        self,
        *,
        backend: Any,
        input_nodes: Float64Array,
        input_elements: UInt64Array,
        analysis_nodes: Float64Array,
        analysis_elements: UInt64Array,
        elevated: ElevatedQuad9Mesh | None,
        material_ids: UInt64Array,
        material_table: Float64Array,
        thermal_material_table: Float64Array | None,
        material_orientation_angles: Float64Array,
        pressure_faces: UInt64Array,
        traction_faces: UInt64Array,
        formulation: str,
        thickness: float,
        element_type: str,
        free_dofs: npt.NDArray[np.int64],
        fixed_dofs: npt.NDArray[np.int64],
        fixed_values: Float64Array,
        ndof_full: int,
        ndof_reduced: int,
        nelem: int,
        nq_per_element: int,
        n_temperature_nodes: int,
    ) -> None:
        self._backend = backend
        self._input_nodes = input_nodes
        self._input_elements = input_elements
        self._elevated = elevated
        self._material_ids = material_ids
        self._material_table = material_table
        self._thermal_material_table = thermal_material_table
        self._material_orientation_angles = material_orientation_angles
        self.pressure_faces = pressure_faces
        self.traction_faces = traction_faces
        self.analysis_nodes = analysis_nodes
        self.analysis_elements = analysis_elements
        self.free_dofs = free_dofs
        self.fixed_dofs = fixed_dofs
        self.fixed_values = fixed_values
        self.formulation = formulation
        self.thickness = thickness
        self.element_type = element_type
        if formulation == "axisymmetric":
            self.coordinate_labels = ("r", "z")
            self.displacement_labels = ("u_r", "u_z")
            self.tensor_labels = ("rr", "zz", "tt", "rz")
            self.measure_label = "swept_volume"
        else:
            self.coordinate_labels = ("x", "y")
            self.displacement_labels = ("u_x", "u_y")
            self.tensor_labels = ("xx", "yy", "zz", "xy")
            self.measure_label = "volume"
        self.ndof_full = int(ndof_full)
        self.ndof_reduced = int(ndof_reduced)
        self.nelem = int(nelem)
        self.nq_per_element = int(nq_per_element)
        self.n_temperature_nodes = int(n_temperature_nodes)
        self._quadrature_cache: Quadrature | None = None
        self._element_measures_cache: ElementMeasures | None = None

    @property
    def input_nodes(self) -> Float64Array:
        """Corner-node input mesh coordinates with shape `(nnode, 2)` and units `[length]`."""

        return self._input_nodes

    @property
    def input_elements(self) -> npt.NDArray[np.uint64]:
        """Input mesh connectivity with shape `(nelem, 4)`."""

        return self._input_elements

    @cached_property
    def constant_rhs(self) -> Float64Array:
        """Load-independent reduced RHS contribution, exported from Rust on first access."""

        return np.asarray(self._backend.constant_rhs(), dtype=np.float64)

    def quadrature(self) -> Quadrature:
        """Return element-major quadrature locations and mapped integration weights.

        The returned locations are built directly from the model's quadrature rule and element
        geometry, so no global mesh query or point inversion is performed. Results are cached
        because the model mesh, quadrature rule, and geometry are immutable after assembly.

        Returns:
            Quadrature: Flat element-major locations with `nelem * nq_per_element` rows plus
            mapped area and volume integration weights. Pass `quadrature.locations` to recovery
            and sparse operator methods.
        """

        cache = self._quadrature_cache
        if cache is None:
            points, element_indices, reference_points, weights_area, weights_volume, points_per_element = (
                self._backend.quadrature()
            )
            locations = PointLocations(
                points=np.asarray(points, dtype=np.float64).reshape(-1, 2),
                element_indices=np.asarray(element_indices, dtype=np.int64),
                reference_points=np.asarray(reference_points, dtype=np.float64).reshape(-1, 2),
                element_type=self.element_type,
            )
            cache = Quadrature(
                locations=locations,
                weights_area=np.asarray(weights_area, dtype=np.float64),
                weights_volume=np.asarray(weights_volume, dtype=np.float64),
                points_per_element=int(points_per_element),
            )
            self._quadrature_cache = cache
        return cache

    def locate_points(
        self,
        points: ArrayLike,
        *,
        outside: str = "nearest",
        tolerance: float | None = None,
        max_iterations: int = 20,
    ) -> PointLocations:
        """Locate arbitrary physical points in this model's analysis mesh.

        This uses the current brute-force quadrilateral mesh query and returns the nearest element
        plus reference coordinates for each query point. Points outside the mesh are projected to
        the nearest element unless `outside="raise"` or `outside="error"` is supplied.

        Args:
            points: Physical coordinates with shape `(npoint, 2)` and units `[length]`.
            outside: Outside-mesh policy. `"nearest"` returns nearest-element projections;
                `"raise"` and `"error"` raise if any point is outside `tolerance`.
            tolerance: Nonnegative physical distance used to classify contained points. Defaults
                to `1e-10`.
            max_iterations: Maximum local inverse-map iterations per element during the query.

        Returns:
            PointLocations: Located points for recovery and sparse operator construction.

        Raises:
            ValueError: If `outside` requests an error and at least one point is outside the mesh.
        """

        query = query_quad_mesh(
            self.analysis_nodes,
            self.analysis_elements,
            points,
            element_type=self.element_type,
            max_iterations=max_iterations,
        )
        outside_policy = str(outside).strip().lower()
        assert outside_policy in {
            "nearest",
            "raise",
            "error",
        }, f"unsupported outside policy {outside!r}; use 'nearest' or 'raise'"
        tol = _normalize_query_tolerance(tolerance)
        inside = query.nearest_element_distances <= tol
        if outside_policy in {"raise", "error"} and not np.all(inside):
            first = int(np.flatnonzero(~inside)[0])
            raise ValueError(f"query point {first} is outside the quad mesh")
        return PointLocations(
            points=query.nearest_element_points,
            element_indices=query.nearest_element_indices,
            reference_points=query.nearest_element_reference_points,
            element_type=self.element_type,
        )

    def locate_points_in_elements(
        self,
        points: ArrayLike,
        element_indices: ArrayLike,
        *,
        max_iterations: int = 20,
    ) -> PointLocations:
        """Project physical points into caller-supplied owning elements.

        This is the fast path when the caller already knows element ownership, such as when
        reusing element indices returned from `quadrature()` or a previous mesh query. It performs
        one local element projection per point and does not scan the global mesh.

        Args:
            points: Physical coordinates with shape `(npoint, 2)` and units `[length]`.
            element_indices: Owning analysis-element indices with shape `(npoint,)`.
            max_iterations: Maximum inverse-map iterations for each local element projection.

        Returns:
            PointLocations: Projected physical points, caller-supplied element indices, and
            reference coordinates.
        """

        points_arr = _normalize_query_points(points)
        element_indices_arr = np.asarray(element_indices, dtype=np.uint64).reshape(-1)
        projected_points, projected_elements, reference_points = self._backend.locate_points_in_elements(
            points_arr,
            element_indices_arr,
            int(max_iterations),
        )
        return PointLocations(
            points=np.asarray(projected_points, dtype=np.float64).reshape(-1, 2),
            element_indices=np.asarray(projected_elements, dtype=np.int64),
            reference_points=np.asarray(reference_points, dtype=np.float64).reshape(-1, 2),
            element_type=self.element_type,
        )

    @cached_property
    def _temperature_elevation(self) -> sp.csr_matrix:
        """Return the cached input-to-analysis temperature elevation operator.

        Corner-node quad9 inputs are elevated inside the Rust backend before assembly, while the
        Python API still accepts temperatures on the original input nodes.  This operator bridges
        those two spaces for exported scipy operators.  It is cached because the input mesh and
        inferred analysis mesh are immutable for the lifetime of the model.
        """

        elevated = self._elevated
        assert elevated is not None, "temperature elevation is available only for inferred quad9 meshes"
        return _temperature_elevation_operator(elevated)

    @cached_property
    def stiffness(self) -> sp.csc_matrix:
        """Reduced stiffness matrix with shape `(ndof_reduced, ndof_reduced)`.

        Entries have units `[generalized force / displacement] = [energy / distance^2]`.
        The SciPy matrix is exported from the Rust backend on first access and then cached.
        """

        return _csc_matrix_from_binding(self._backend.stiffness_csc())

    @property
    def body_force_to_rhs(self) -> sp.csr_matrix:
        """Operator mapping per-element body-force density to the reduced RHS.

        Shape is `(ndof_reduced, 2 * nelem)`. Entries have units `[volume]`.
        """

        return _csr_matrix_from_binding(self._backend.body_force_to_rhs_csr())

    @property
    def pressure_to_rhs(self) -> sp.csr_matrix:
        """Operator mapping scalar pressure amplitudes to the reduced RHS.

        Shape is `(ndof_reduced, n_pressure_faces)`. Entries have units `[area]`.
        """

        return _csr_matrix_from_binding(self._backend.pressure_to_rhs_csr())

    @property
    def traction_to_rhs(self) -> sp.csr_matrix:
        """Operator mapping vector traction amplitudes to the reduced RHS.

        Shape is `(ndof_reduced, 2 * n_traction_faces)`. Entries have units `[area]`.
        """

        return _csr_matrix_from_binding(self._backend.traction_to_rhs_csr())

    @property
    def temperature_to_rhs(self) -> sp.csr_matrix:
        """Operator mapping input-node temperatures to the reduced RHS.

        Shape is `(ndof_reduced, n_temperature_nodes)`. Entries have units
        `[generalized force / temperature] = [energy / (distance * temperature)]`.
        """

        analysis_operator = _csr_matrix_from_binding(self._backend.temperature_to_rhs_csr())
        return (
            sp.csr_matrix(analysis_operator @ self._temperature_elevation)
            if self._elevated is not None and self.n_temperature_nodes > 0
            else analysis_operator
        )

    def interpolation_operator(self, locations: PointLocations) -> sp.csr_matrix:
        """Build a sparse interpolation operator for located points.

        Args:
            locations: Element-owned point locations from `quadrature().locations`,
                `locate_points(...)`, `locate_points_in_elements(...)`, or
                `QuadMeshQuery.point_locations()`.

        Returns:
            csr_matrix: Sparse operator with shape `(npoint, n_analysis_nodes)`. Multiplying by a
            scalar nodal field with shape `(n_analysis_nodes,)` returns interpolated values with
            shape `(npoint,)`; multiplying by `(n_analysis_nodes, ncomponent)` interpolates each
            component independently. Entries are unitless shape-function values.
        """

        locations = self._validate_locations(locations)
        return _coo_operator_from_binding(
            _quad_mesh_interpolation_operator_f64(
                self.analysis_nodes,
                self.analysis_elements,
                locations.element_indices.astype(np.uint64, copy=False),
                locations.reference_points,
                self.element_type,
            ),
        )

    def strain_operator(self, locations: PointLocations) -> sp.csr_matrix:
        """Build a sparse total-strain recovery operator for located points.

        Args:
            locations: Element-owned point locations from `quadrature().locations`,
                `locate_points(...)`, `locate_points_in_elements(...)`, or
                `QuadMeshQuery.point_locations()`.

        Returns:
            csr_matrix: Sparse operator with shape `(4 * npoint, 2 * n_analysis_nodes)`. Rows are
            grouped by point and tensor component. Multiplying by full analysis displacements with
            shape `(2 * n_analysis_nodes,)` returns flat strain samples with shape `(4 * npoint,)`.
            Entries have units `[1 / length]`.
        """

        locations = self._validate_locations(locations)
        return _coo_operator_from_binding(
            _quad_mesh_strain_operator_f64(
                self.analysis_nodes,
                self.analysis_elements,
                locations.element_indices.astype(np.uint64, copy=False),
                locations.reference_points,
                self.element_type,
                _formulation_code(self.formulation),
                self.thickness,
            ),
        )

    def stress_operator(self, locations: PointLocations) -> sp.csr_matrix:
        """Build a sparse elastic-stress recovery operator for located points.

        Args:
            locations: Element-owned point locations from `quadrature().locations`,
                `locate_points(...)`, `locate_points_in_elements(...)`, or
                `QuadMeshQuery.point_locations()`.

        Returns:
            csr_matrix: Sparse operator with shape `(4 * npoint, 2 * n_analysis_nodes)`. Rows are
            grouped by point and tensor component. Multiplying by full analysis displacements with
            shape `(2 * n_analysis_nodes,)` returns flat stress samples with shape `(4 * npoint,)`.
            Entries have units `[stress / length]`.
        """

        locations = self._validate_locations(locations)
        return _coo_operator_from_binding(
            _quad_mesh_stress_operator_f64(
                self.analysis_nodes,
                self.analysis_elements,
                locations.element_indices.astype(np.uint64, copy=False),
                locations.reference_points,
                self._material_ids,
                self._material_table,
                self._material_orientation_angles,
                self.element_type,
                _formulation_code(self.formulation),
                self.thickness,
            ),
        )

    def element_measures(self) -> ElementMeasures:
        """Return cross-section area and represented volume for each element.

        Returns:
            ElementMeasures: Per-element measures with:
                `areas` of shape `(nelem,)` and units `[area]`,
                `volumes` of shape `(nelem,)` and units `[volume]`.
        """

        cache = self._element_measures_cache
        if cache is not None:
            return cache
        quadrature = self.quadrature()
        weights_area = quadrature.weights_area.reshape(self.nelem, quadrature.points_per_element)
        weights_volume = quadrature.weights_volume.reshape(self.nelem, quadrature.points_per_element)
        cache = ElementMeasures(
            areas=np.asarray(weights_area.sum(axis=1), dtype=np.float64),
            volumes=np.asarray(weights_volume.sum(axis=1), dtype=np.float64),
        )
        self._element_measures_cache = cache
        return cache

    def _normalize_temperature_for_backend(
        self,
        nodal_temperature: ArrayLike | None,
    ) -> npt.NDArray[np.floating[Any]] | None:
        if self.n_temperature_nodes == 0:
            values = (
                np.zeros((0,), dtype=np.float64)
                if nodal_temperature is None
                else np.asarray(nodal_temperature).reshape(-1)
            )
            assert values.size == 0, "nodal_temperature was provided, but this model has no thermal operator"
            return None
        if nodal_temperature is None:
            raise ValueError("nodal_temperature is required because this model includes thermal materials")
        input_temperature = _normalize_nodal_temperature(nodal_temperature, self._input_nodes.shape[0])
        return (
            np.asarray(self._temperature_elevation @ input_temperature, dtype=np.float64)
            if self._elevated is not None
            else input_temperature
        )

    def _validate_locations(self, locations: PointLocations) -> PointLocations:
        assert isinstance(locations, PointLocations), "locations must be a PointLocations object"
        assert (
            locations.element_type == self.element_type
        ), f"locations use element_type {locations.element_type!r}, but model uses {self.element_type!r}"
        assert (
            locations.points.ndim == 2 and locations.points.shape[1] == 2
        ), f"locations.points must have shape (npoint, 2); got {locations.points.shape}"
        assert (
            locations.reference_points.ndim == 2 and locations.reference_points.shape[1] == 2
        ), f"locations.reference_points must have shape (npoint, 2); got {locations.reference_points.shape}"
        assert (
            locations.element_indices.ndim == 1
        ), f"locations.element_indices must have shape (npoint,); got {locations.element_indices.shape}"
        npoint = locations.element_indices.shape[0]
        assert (
            locations.points.shape[0] == npoint
        ), f"locations.points has {locations.points.shape[0]} rows, but element_indices has {npoint}"
        assert locations.reference_points.shape[0] == npoint, (
            "locations.reference_points has "
            f"{locations.reference_points.shape[0]} rows, but element_indices has {npoint}"
        )
        return locations

    def build_rhs(
        self,
        body_force: ArrayLike | None = None,
        pressure_values: ArrayLike | None = None,
        traction_values: ArrayLike | None = None,
        nodal_temperature: ArrayLike | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """Build one reduced structural right-hand side.

        Args:
            body_force: Elementwise body-force amplitudes with shape `(2,)` or `(nelem, 2)`.
                Components are `[b_r, b_z]` with units `[force / volume]`.
            pressure_values: Pressure amplitudes with shape `(n_pressure_faces,)` and units
                `[force / area]`. Positive values act in the inward normal direction.
            traction_values: Surface traction amplitudes with shape `(2,)` or
                `(n_traction_faces, 2)`. Components are `[t_r, t_z]` with units
                `[force / area]`.
            nodal_temperature: Input-node temperatures with shape `(n_input_nodes,)` and units
                `[temperature]`. Required only when the model includes thermal materials.

        Returns:
            NDArray: Reduced right-hand side with shape `(ndof_reduced,)` and units
            `[generalized force] = [energy / distance]`.

        Raises:
            ValueError: If thermal materials are present but `nodal_temperature` is omitted.
        """

        body_force_arr = None if body_force is None else _normalize_body_force(body_force, self.nelem)
        npressure = int(self.pressure_faces.shape[0])
        pressure_arr = (
            None if pressure_values is None else _normalize_pressure_values(pressure_values, npressure)
        )
        traction_arr = (
            None
            if traction_values is None
            else _normalize_traction_values(
                traction_values,
                int(self.traction_faces.shape[0]),
            )
        )
        temperature_arr = self._normalize_temperature_for_backend(nodal_temperature)
        rhs = self._backend.build_rhs(
            None if body_force_arr is None else body_force_arr.reshape(-1),
            pressure_arr,
            None if traction_arr is None else traction_arr.reshape(-1),
            temperature_arr,
        )
        return np.asarray(rhs, dtype=np.float64)

    def solve(self, rhs: ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """Solve the reduced system and recover the full displacement field.

        Args:
            rhs: Reduced right-hand side with shape `(ndof_reduced,)` and units
                `[generalized force] = [energy / distance]`.

        Returns:
            NDArray: Full displacement vector with shape `(ndof_full,)` and component ordering
            `[u_r0, u_z0, u_r1, u_z1, ...]`. Units are `[length]`.
        """

        rhs_arr = np.asarray(rhs).reshape(-1)
        assert (
            rhs_arr.shape[0] == self.ndof_reduced
        ), f"rhs must have length {self.ndof_reduced}; got {rhs_arr.shape}"
        return np.asarray(self._backend.solve(rhs_arr), dtype=np.float64)

    def recover_full(self, reduced_solution: ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """Reinsert prescribed Dirichlet values into a reduced displacement vector.

        Args:
            reduced_solution: Reduced displacement vector with shape `(ndof_reduced,)` and units
                `[length]`.

        Returns:
            NDArray: Full displacement vector with shape `(ndof_full,)` and component ordering
            `[u_r0, u_z0, u_r1, u_z1, ...]`. Units are `[length]`.
        """

        reduced_arr = np.asarray(reduced_solution).reshape(-1)
        assert (
            reduced_arr.shape[0] == self.ndof_reduced
        ), f"reduced_solution must have length {self.ndof_reduced}; got {reduced_arr.shape}"
        full = np.zeros((self.ndof_full,), dtype=np.float64)
        full[self.fixed_dofs] = self.fixed_values
        full[self.free_dofs] = reduced_arr
        return full

    def _full_displacement_for_backend(self, displacements: ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """Normalize reduced or full displacements to the backend's full flat vector."""

        arr = np.asarray(displacements)
        if arr.ndim == 1 and arr.shape == (self.ndof_reduced,):
            return self.recover_full(arr)
        return _normalize_displacements(displacements, self.analysis_nodes.shape[0]).reshape(-1)

    def strain(
        self,
        locations: PointLocations,
        displacements: ArrayLike,
    ) -> npt.NDArray[np.floating[Any]]:
        """Evaluate total strain at located points without materializing recovery matrices.

        Args:
            locations: Element-owned point locations from `quadrature().locations`,
                `locate_points(...)`, `locate_points_in_elements(...)`, or
                `QuadMeshQuery.point_locations()`.
            displacements: Either the reduced displacement solution with shape
                `(ndof_reduced,)`, or the full analysis displacement field with shape
                `(2 * n_analysis_nodes,)` or `(n_analysis_nodes, 2)`. Displacement units are
                `[length]`.

        Returns:
            NDArray: Total strain with shape `(npoint, 4)` and component ordering `[rr, zz, tt,
            rz]` for axisymmetric models or `[xx, yy, zz, xy]` for plane strain. Strain is
            unitless. For quadrature locations, reshape as `(nelem, nq_per_element, 4)` when an
            element-major view is needed.
        """

        locations = self._validate_locations(locations)
        displacements_full = self._full_displacement_for_backend(displacements)
        strain_flat = self._backend.strain(
            locations.element_indices.astype(np.uint64, copy=False),
            locations.reference_points,
            displacements_full,
        )
        return np.asarray(strain_flat, dtype=np.float64).reshape(-1, 4)

    def stress(
        self,
        locations: PointLocations,
        displacements: ArrayLike,
    ) -> npt.NDArray[np.floating[Any]]:
        """Evaluate stress at located points without materializing recovery matrices.

        Args:
            locations: Element-owned point locations from `quadrature().locations`,
                `locate_points(...)`, `locate_points_in_elements(...)`, or
                `QuadMeshQuery.point_locations()`.
            displacements: Either the reduced displacement solution with shape
                `(ndof_reduced,)`, or the full analysis displacement field with shape
                `(2 * n_analysis_nodes,)` or `(n_analysis_nodes, 2)`. Displacement units are
                `[length]`.

        Returns:
            NDArray: Stress with shape `(npoint, 4)` and component ordering `[rr, zz, tt, rz]` for
            axisymmetric models or `[xx, yy, zz, xy]` for plane strain. Units are `[stress]`. For
            quadrature locations, reshape as `(nelem, nq_per_element, 4)` when an element-major
            view is needed.
        """

        locations = self._validate_locations(locations)
        displacements_full = self._full_displacement_for_backend(displacements)
        stress_flat = self._backend.stress(
            locations.element_indices.astype(np.uint64, copy=False),
            locations.reference_points,
            displacements_full,
        )
        return np.asarray(stress_flat, dtype=np.float64).reshape(-1, 4)

    def thermal_strain(
        self,
        locations: PointLocations,
        nodal_temperature: ArrayLike | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """Evaluate thermal strain at located points without materializing recovery matrices.

        Args:
            locations: Element-owned point locations from `quadrature().locations`,
                `locate_points(...)`, `locate_points_in_elements(...)`, or
                `QuadMeshQuery.point_locations()`.
            nodal_temperature: Input-node temperatures with shape `(n_input_nodes,)` and units
                `[temperature]`. Required only when the model includes thermal materials.

        Returns:
            NDArray: Thermal strain with shape `(npoint, 4)` and component ordering `[rr, zz, tt,
            rz]` for axisymmetric models or `[xx, yy, zz, xy]` for plane strain. Strain is
            unitless. Models without thermal materials return zeros and do not require
            `nodal_temperature`.
        """

        locations = self._validate_locations(locations)
        temperature_arr = self._normalize_temperature_for_backend(nodal_temperature)
        thermal_strain_flat = self._backend.thermal_strain(
            locations.element_indices.astype(np.uint64, copy=False),
            locations.reference_points,
            temperature_arr,
        )
        return np.asarray(thermal_strain_flat, dtype=np.float64).reshape(-1, 4)

    def thermal_stress(
        self,
        locations: PointLocations,
        nodal_temperature: ArrayLike | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """Evaluate thermal stress at located points without materializing recovery matrices.

        Args:
            locations: Element-owned point locations from `quadrature().locations`,
                `locate_points(...)`, `locate_points_in_elements(...)`, or
                `QuadMeshQuery.point_locations()`.
            nodal_temperature: Input-node temperatures with shape `(n_input_nodes,)` and units
                `[temperature]`. Required only when the model includes thermal materials.

        Returns:
            NDArray: Thermal stress with shape `(npoint, 4)` and component ordering `[rr, zz, tt,
            rz]` for axisymmetric models or `[xx, yy, zz, xy]` for plane strain. Units are
            `[stress]`. Models without thermal materials return zeros and do not require
            `nodal_temperature`.
        """

        locations = self._validate_locations(locations)
        temperature_arr = self._normalize_temperature_for_backend(nodal_temperature)
        thermal_stress_flat = self._backend.thermal_stress(
            locations.element_indices.astype(np.uint64, copy=False),
            locations.reference_points,
            temperature_arr,
        )
        return np.asarray(thermal_stress_flat, dtype=np.float64).reshape(-1, 4)


def _quadrature_code(quadrature: str | int) -> int:
    normalized = str(quadrature).strip().lower()
    codes = {"3": 3, "gl3": 3, "4": 4, "gl4": 4}
    assert normalized in codes, f"unsupported quadrature {quadrature!r}; use 'gl3' or 'gl4'"
    return codes[normalized]


def _normalize_element_type(element_type: str) -> str:
    normalized = str(element_type).strip().lower()
    assert normalized in {
        "quad4",
        "quad9",
    }, f"unsupported element_type {element_type!r}; use 'quad4' or 'quad9'"
    return normalized


def _normalize_formulation(formulation: str) -> str:
    normalized = str(formulation).strip().lower()
    assert normalized in {
        "axisymmetric",
        "plane_strain",
    }, f"unsupported formulation {formulation!r}; use 'axisymmetric' or 'plane_strain'"
    return normalized


def _formulation_code(formulation: str) -> int:
    assert formulation in {"axisymmetric", "plane_strain"}, f"unsupported formulation {formulation!r}"
    return 0 if formulation == "axisymmetric" else 1


def _normalize_thickness(
    formulation: str,
    thickness: float | None,
) -> float:
    if formulation == "axisymmetric":
        assert thickness is None, "thickness is only valid for formulation='plane_strain'"
        return 0.0
    assert thickness is not None, "thickness is required for formulation='plane_strain'"
    value = float(thickness)
    assert value > 0.0, f"thickness must be positive; got {thickness!r}"
    return value


def _normalize_material_orientation_angles(
    material_orientation_angles: ArrayLike | None,
    nelem: int,
) -> Float64Array:
    if material_orientation_angles is None:
        return np.zeros((0,), dtype=np.float64)
    angles = np.asarray(material_orientation_angles)
    angles = np.broadcast_to(angles, (nelem,)).copy() if angles.ndim == 0 else angles
    assert angles.ndim == 1 and angles.shape[0] == nelem, (
        f"material_orientation_angles must be a scalar or have shape ({nelem},); " f"got {angles.shape}"
    )
    return angles


def _normalize_nodes(nodes: ArrayLike) -> Float64Array:
    arr = np.asarray(nodes)
    assert arr.ndim == 2 and arr.shape[1] == 2, f"nodes must have shape (nnode, 2); got {arr.shape}"
    return arr


def _normalize_elements(
    elements: ArrayLike,
    nodes_per_element: int | tuple[int, ...] = 4,
) -> npt.NDArray[np.uint64]:
    arr = _as_uint64_array(elements)
    expected = (nodes_per_element,) if isinstance(nodes_per_element, int) else nodes_per_element
    expected_text = " or ".join(f"(nelem, {count})" for count in expected)
    assert (
        arr.ndim == 2 and arr.shape[1] in expected
    ), f"elements must have shape {expected_text}; got {arr.shape}"
    return arr


def infer_quad9_mesh(nodes: ArrayLike, elements: ArrayLike) -> ElevatedQuad9Mesh:
    """Elevate a corner-only quad mesh to an explicit 9-node Lagrange mesh.

    Args:
        nodes: Corner-node coordinates with shape `(nnode, 2)` in `(r, z)` order. Units are
            `[length]`.
        elements: Quad4 connectivity with shape `(nelem, 4)`. Corner nodes must be ordered
            counter-clockwise in the `(r, z)` plane.

    Returns:
        ElevatedQuad9Mesh: Elevated analysis mesh with:
            `analysis_nodes` of shape `(n_analysis_nodes, 2)` and units `[length]`,
            `analysis_elements` of shape `(nelem, 9)`,
            `corner_node_indices`, `midside_node_indices`, and `center_node_indices` as
            one-dimensional index arrays.
    """

    nodes_arr = _normalize_nodes(nodes)
    elements_arr = _normalize_elements(elements)
    (
        analysis_nodes_flat,
        analysis_elements_flat,
        corner_node_indices,
        midside_node_indices,
        center_node_indices,
    ) = _infer_quad9_mesh_f64(nodes_arr, elements_arr)

    return ElevatedQuad9Mesh(
        input_nodes=nodes_arr,
        input_elements=elements_arr,
        analysis_nodes=np.asarray(analysis_nodes_flat, dtype=np.float64).reshape(-1, 2),
        analysis_elements=np.asarray(analysis_elements_flat, dtype=np.uint64).reshape(-1, 9),
        corner_node_indices=np.asarray(corner_node_indices, dtype=np.int64),
        midside_node_indices=np.asarray(midside_node_indices, dtype=np.int64),
        center_node_indices=np.asarray(center_node_indices, dtype=np.int64),
    )


def _normalize_query_points(
    points: ArrayLike,
) -> Float64Array:
    arr = np.asarray(points)
    assert arr.ndim == 2 and arr.shape[1] == 2, f"points must have shape (npoint, 2); got {arr.shape}"
    return arr


def _normalize_query_tolerance(
    tolerance: float | None,
) -> float:
    value = 1.0e-10 if tolerance is None else float(tolerance)
    assert value >= 0.0, f"tolerance must be nonnegative; got {tolerance!r}"
    return value


def _coo_operator_from_binding(
    binding: tuple[ArrayLike, ArrayLike, ArrayLike, int, int],
) -> sp.csr_matrix:
    vals, rows, cols, nrow, ncol = binding
    return sp.coo_matrix(
        (
            np.asarray(vals, dtype=np.float64),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(cols, dtype=np.int64),
            ),
        ),
        shape=(int(nrow), int(ncol)),
    ).tocsr()


def query_quad_mesh(
    nodes: ArrayLike,
    elements: ArrayLike,
    points: ArrayLike,
    *,
    element_type: str = "quad4",
    max_iterations: int = 20,
) -> QuadMeshQuery:
    """Query nearest node, nearest element, and nearest face in one pass.

    The Rust backend scans all nodes once and all elements once per query point. The element scan
    computes nearest-element and nearest-face metadata together so downstream interpolation and
    recovery operators do not repeat point location. Complexity is
    `O(npoint * (nnode + nelem * max_iterations))`. A point is contained when its
    nearest-element distance is zero to the caller's tolerance.

    Args:
        nodes: Mesh node coordinates with shape `(nnode, 2)` and units `[length]`.
        elements: Quad connectivity with shape `(nelem, 4)` for `quad4` or `(nelem, 9)` for
            `quad9`. Entries are unitless node indices.
        points: Query point coordinates with shape `(npoint, 2)` and units `[length]`.
        element_type: Element family, either `"quad4"` or `"quad9"`.
        max_iterations: Maximum Newton/projection iterations per element. Unitless.

    Returns:
        Query data with nearest-node, nearest-element, and nearest-face arrays. Coordinate arrays
        have units `[length]`, distances have units `[length]`, reference coordinates are unitless,
        and index arrays are unitless.
    """

    normalized_element_type = _normalize_element_type(element_type)
    nodes_arr = _normalize_nodes(nodes)
    elements_arr = _normalize_elements(
        elements,
        4 if normalized_element_type == "quad4" else 9,
    )
    points_arr = _normalize_query_points(points)
    data = _quad_mesh_query_f64(
        nodes_arr,
        elements_arr,
        points_arr,
        normalized_element_type,
        int(max_iterations),
    )

    return QuadMeshQuery(
        nodes=nodes_arr,
        elements=elements_arr,
        points=points_arr,
        element_type=normalized_element_type,
        nearest_node_indices=np.asarray(data["nearest_node_indices"], dtype=np.int64),
        nearest_node_points=np.asarray(data["nearest_node_points"], dtype=np.float64).reshape(-1, 2),
        nearest_node_distances=np.asarray(data["nearest_node_distances"], dtype=np.float64),
        nearest_element_indices=np.asarray(data["nearest_element_indices"], dtype=np.int64),
        nearest_element_reference_points=np.asarray(
            data["nearest_element_reference_points"],
            dtype=np.float64,
        ).reshape(-1, 2),
        nearest_element_points=np.asarray(data["nearest_element_points"], dtype=np.float64).reshape(-1, 2),
        nearest_element_distances=np.asarray(data["nearest_element_distances"], dtype=np.float64),
        nearest_face_element_indices=np.asarray(data["nearest_face_element_indices"], dtype=np.int64),
        nearest_face_local_faces=np.asarray(data["nearest_face_local_faces"], dtype=np.int64),
        nearest_face_reference_coordinates=np.asarray(
            data["nearest_face_reference_coordinates"],
            dtype=np.float64,
        ),
        nearest_face_points=np.asarray(data["nearest_face_points"], dtype=np.float64).reshape(-1, 2),
        nearest_face_distances=np.asarray(data["nearest_face_distances"], dtype=np.float64),
    )


def quad_mesh_interpolation_operator(
    query: QuadMeshQuery,
) -> sp.csr_matrix:
    """Return a reusable sparse operator mapping nodal scalar values to query-point values.

    The returned matrix has shape `(npoint, nnode)`. Applying it to a dense `(nnode,)` vector gives
    scalar values at the query points; applying it to `(nnode, ncomponent)` interpolates multiple
    nodal fields with the same operator. The operator always uses the query's nearest element.

    Args:
        query: Mesh query data from `query_quad_mesh(...)`.

    Returns:
        Sparse interpolation operator with shape `(npoint, nnode)`. Entries are unitless shape
        function values, so output values have the same units as the nodal values supplied during
        matrix multiplication.
    """

    return _coo_operator_from_binding(
        _quad_mesh_interpolation_operator_f64(
            query.nodes,
            query.elements,
            np.asarray(query.nearest_element_indices, dtype=np.uint64),
            query.nearest_element_reference_points,
            query.element_type,
        ),
    )


def interpolate_quad_mesh_values(
    nodes: ArrayLike,
    elements: ArrayLike,
    nodal_values: ArrayLike,
    points: ArrayLike,
    *,
    element_type: str = "quad4",
    outside: str = "raise",
    tolerance: float | None = None,
    max_iterations: int = 20,
) -> QuadMeshInterpolation:
    """Interpolate nodal values at arbitrary physical points in a quadrilateral mesh.

    The interpolation uses the element's actual shape functions. `nodal_values` may have shape
    `(nnode,)` or `(nnode, ...)`; the returned values have shape `(npoint,)` or `(npoint, ...)`.

    Point location is Rust-backed but brute-force and scans all elements once per query point.
    Outside policies are applied from the nearest-element distance: `"raise"` errors,
    `"nan"` masks outside values, and `"nearest"` returns the nearest-element interpolation.
    Complexity is `O(npoint * nelem * max_iterations)` for point location plus
    `O(npoint * nodes_per_element * ncomponent)` for interpolation.

    Args:
        nodes: Mesh node coordinates with shape `(nnode, 2)`.
        elements: Quad connectivity with shape `(nelem, 4)` or `(nelem, 9)`.
        nodal_values: Values at mesh nodes with shape `(nnode,)` or `(nnode, ...)`.
        points: Query point coordinates with shape `(npoint, 2)`.
        element_type: Element family, either `"quad4"` or `"quad9"`.
        outside: Outside-mesh policy: `"raise"`/`"error"`, `"nan"`, or `"nearest"`.
        tolerance: Physical and reference-space tolerance for point containment.
        max_iterations: Maximum Newton/projection iterations per element.

    Returns:
        Interpolated values plus element indices, reference coordinates, and inside flags for the
        query points. `values` has shape `(npoint,)` or `(npoint, ...)` and the same units as
        `nodal_values`; `element_indices` is unitless with shape `(npoint,)`; `reference_points`
        is unitless with shape `(npoint, 2)`; `inside` has shape `(npoint,)`.
    """

    query = query_quad_mesh(
        nodes,
        elements,
        points,
        element_type=element_type,
        max_iterations=max_iterations,
    )
    values_arr = np.asarray(nodal_values)
    assert (
        values_arr.ndim >= 1 and values_arr.shape[0] == query.nodes.shape[0]
    ), f"nodal_values must have shape (nnode,) or (nnode, ...); got {values_arr.shape}"
    values_shape = values_arr.shape[1:]
    values_2d = values_arr.reshape(query.nodes.shape[0], -1)
    outside_policy = str(outside).strip().lower()
    assert outside_policy in {
        "nearest",
        "nan",
        "raise",
        "error",
    }, f"unsupported outside policy {outside!r}; use 'raise', 'nan', or 'nearest'"
    tol = _normalize_query_tolerance(tolerance)
    inside = query.nearest_element_distances <= tol
    if outside_policy in {"raise", "error"} and not np.all(inside):
        first = int(np.flatnonzero(~inside)[0])
        raise ValueError(f"query point {first} is outside the quad mesh")
    operator = quad_mesh_interpolation_operator(query)
    values = np.asarray(operator @ values_2d).reshape((query.points.shape[0], *values_shape))
    if outside_policy == "nan" and np.any(~inside):
        values[~inside] = np.nan
    return QuadMeshInterpolation(
        values=values,
        element_indices=query.nearest_element_indices,
        reference_points=query.nearest_element_reference_points,
        inside=inside,
    )


def _temperature_elevation_operator(
    elevated: ElevatedQuad9Mesh,
) -> sp.csr_matrix:
    n_input_nodes = elevated.input_nodes.shape[0]
    n_analysis_nodes = elevated.analysis_nodes.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for node in range(n_input_nodes):
        rows.append(node)
        cols.append(node)
        vals.append(1.0)

    edge_to_midpoint: dict[tuple[int, int], int] = {}
    for element_index, conn in enumerate(elevated.input_elements):
        for local_edge, (local_a, local_b) in enumerate(_QUAD_FACE_NODE_PAIRS):
            node_a = int(conn[local_a])
            node_b = int(conn[local_b])
            edge_key = (node_a, node_b) if node_a < node_b else (node_b, node_a)
            if edge_key in edge_to_midpoint:
                continue
            midpoint_index = int(elevated.analysis_elements[element_index, 4 + local_edge])
            edge_to_midpoint[edge_key] = midpoint_index
            rows.extend([midpoint_index, midpoint_index])
            cols.extend([edge_key[0], edge_key[1]])
            vals.extend([0.5, 0.5])
        center_index = int(elevated.analysis_elements[element_index, 8])
        rows.extend([center_index] * 4)
        cols.extend([int(node) for node in conn])
        vals.extend([0.25] * 4)

    return sp.coo_matrix(
        (
            np.asarray(vals, dtype=np.float64),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(n_analysis_nodes, n_input_nodes),
    ).tocsr()


def _normalize_materials(
    material_ids: ArrayLike,
    material_table: ArrayLike,
) -> tuple[UInt64Array, Float64Array]:
    ids = _as_uint64_array(material_ids)
    assert ids.ndim == 1, f"material_ids must have shape (nelem,); got {ids.shape}"
    assert not isinstance(
        material_table, Mapping
    ), "material_table must be a dense array; use pack_material_tables_from_tags(...) for tagged inputs"
    table = np.asarray(material_table)
    assert table.ndim == 3 and table.shape[1:] == (
        4,
        4,
    ), f"material_table must have shape (nmat, 4, 4); got {table.shape}"
    return ids, table


def _normalize_thermal_material_table(
    thermal_material_table: ArrayLike | None,
) -> Float64Array | None:
    if thermal_material_table is None:
        return None
    assert not isinstance(thermal_material_table, Mapping), (
        "thermal_material_table must be a dense array; use pack_material_tables_from_tags(...) "
        "for tagged inputs"
    )
    table = np.asarray(thermal_material_table)
    assert (
        table.ndim == 2 and table.shape[1] == 5
    ), f"thermal_material_table must have shape (nmat, 5); got {table.shape}"
    assert not np.any(table[:, 3] != 0.0), "shear thermal expansion (alpha_rz) is not yet supported"
    return table


def pack_material_tables_from_tags(
    material_ids: ArrayLike,
    material_table_by_tag: Mapping[int, ArrayLike],
    thermal_material_table_by_tag: Mapping[int, ArrayLike] | None = None,
) -> tuple[
    npt.NDArray[np.uint64],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]] | None,
]:
    """Pack tagged material definitions into the dense FEM input format.

    Args:
        material_ids: Element material tags with shape `(nelem,)`.
        material_table_by_tag: Mapping from external material tag to elastic stress-strain matrix
            with shape `(4, 4)`. Matrix units are `[stress / strain] = [pressure]`.
        thermal_material_table_by_tag: Optional mapping from external material tag to thermal row
            with shape `(5,)` storing `[alpha_r, alpha_z, alpha_t, alpha_rz, T_ref]`. Thermal
            expansion coefficients have units `[strain / temperature]` and `T_ref` has units
            `[temperature]`.
    Returns:
        tuple: `(packed_material_ids, packed_material_table, packed_thermal_material_table)` where:
            `packed_material_ids` has shape `(nelem,)`,
            `packed_material_table` has shape `(nmat, 4, 4)`,
            `packed_thermal_material_table` has shape `(nmat, 5)` when provided, otherwise `None`.

    Raises:
        ValueError: If an element tag is missing from `material_table_by_tag`, or if thermal tags
            do not match the elastic tags exactly.
    """

    assert material_table_by_tag, "material_table_by_tag cannot be empty"
    resolved_dtype = np.dtype(np.float64)
    ids = np.asarray(material_ids, dtype=np.uint64)
    assert ids.ndim == 1, f"material_ids must have shape (nelem,); got {ids.shape}"

    material_tags = sorted(int(tag) for tag in material_table_by_tag)
    tag_to_index = {tag: index for index, tag in enumerate(material_tags)}
    try:
        packed_ids = np.asarray([tag_to_index[int(tag)] for tag in ids], dtype=np.uint64)
    except KeyError as exc:
        raise ValueError(
            f"material_ids contains tag {exc.args[0]} that is missing from material_table_by_tag"
        ) from exc

    material_rows = []
    for tag in material_tags:
        matrix = np.asarray(material_table_by_tag[tag], dtype=resolved_dtype)
        assert matrix.shape == (
            4,
            4,
        ), f"material_table_by_tag[{tag}] must have shape (4, 4); got {matrix.shape}"
        material_rows.append(matrix)
    packed_material_table = np.ascontiguousarray(np.stack(material_rows, axis=0), dtype=resolved_dtype)

    packed_thermal_table: npt.NDArray[np.floating[Any]] | None
    if thermal_material_table_by_tag is None:
        packed_thermal_table = None
    else:
        thermal_tags = {int(tag) for tag in thermal_material_table_by_tag}
        if thermal_tags != set(material_tags):
            raise ValueError(
                "thermal_material_table_by_tag must have exactly the same keys as material_table_by_tag"
            )
        thermal_rows = []
        for tag in material_tags:
            row = np.asarray(thermal_material_table_by_tag[tag], dtype=resolved_dtype)
            assert row.shape == (
                5,
            ), f"thermal_material_table_by_tag[{tag}] must have shape (5,); got {row.shape}"
            thermal_rows.append(row)
        packed_thermal_table = np.ascontiguousarray(np.stack(thermal_rows, axis=0), dtype=resolved_dtype)
        assert not np.any(
            packed_thermal_table[:, 3] != 0.0
        ), "shear thermal expansion (alpha_rz) is not yet supported"

    return packed_ids, packed_material_table, packed_thermal_table


def _normalize_nodal_temperature(
    nodal_temperature: ArrayLike,
    nnode: int,
) -> npt.NDArray[np.floating[Any]]:
    arr = np.asarray(nodal_temperature)
    assert (
        arr.ndim == 1 and arr.shape[0] == nnode
    ), f"nodal_temperature must have shape ({nnode},); got {arr.shape}"
    return arr


def _normalize_body_force(
    body_force: ArrayLike,
    nelem: int,
) -> npt.NDArray[np.floating[Any]]:
    arr = np.asarray(body_force)
    if arr.ndim == 1 and arr.shape == (2,):
        arr = np.broadcast_to(arr, (nelem, 2)).copy()
    assert arr.ndim == 2 and arr.shape == (
        nelem,
        2,
    ), f"body_force must have shape (2,) or (nelem, 2); got {arr.shape}"
    return arr


def _normalize_face_pairs(name: str, faces: ArrayLike | None) -> UInt64Array:
    faces = np.zeros((0, 2), dtype=np.uint64) if faces is None else _as_uint64_array(faces)
    assert faces.ndim == 2 and faces.shape[1] == 2, f"{name} must have shape (nload, 2); got {faces.shape}"
    return faces


def _normalize_pressure_values(
    pressure_values: ArrayLike | None,
    nload: int,
) -> npt.NDArray[np.floating[Any]]:
    values = np.zeros((nload,), dtype=np.float64) if pressure_values is None else np.asarray(pressure_values)
    assert values.ndim == 1, f"pressure_values must have shape (nload,); got {values.shape}"
    assert values.shape[0] == nload, f"pressure_values has {values.shape[0]} entries, but expected {nload}"
    return values


def _normalize_traction_values(
    traction_values: ArrayLike | None,
    nload: int,
) -> npt.NDArray[np.floating[Any]]:
    values = (
        np.zeros((nload, 2), dtype=np.float64) if traction_values is None else np.asarray(traction_values)
    )
    values = (
        np.broadcast_to(values, (nload, 2)).copy() if values.ndim == 1 and values.shape == (2,) else values
    )
    assert values.ndim == 2 and values.shape == (
        nload,
        2,
    ), f"traction_values must have shape (2,) or ({nload}, 2); got {values.shape}"
    return values


def _normalize_prescribed_dirichlet(
    prescribed: Mapping[int, float] | None,
) -> tuple[UInt64Array, Float64Array]:
    items = (
        [] if prescribed is None else sorted((int(dof), float(value)) for dof, value in prescribed.items())
    )
    return (
        np.asarray([dof for dof, _ in items], dtype=np.uint64),
        np.asarray([value for _, value in items], dtype=np.float64),
    )


def assemble_structural_2d(
    nodes: ArrayLike,
    elements: ArrayLike,
    material_ids: ArrayLike,
    material_table: ArrayLike,
    *,
    formulation: str = "axisymmetric",
    thickness: float | None = None,
    material_orientation_angles: ArrayLike | None = None,
    pressure_faces: ArrayLike | None = None,
    traction_faces: ArrayLike | None = None,
    thermal_material_table: ArrayLike | None = None,
    prescribed: Mapping[int, float] | None = None,
    quadrature: str | int = "gl3",
    element_type: str = "quad4",
    par: bool = True,
) -> Structural2DFEMModel:
    """Assemble the reusable 2D structural FEM model.

    Args:
        nodes: Corner-node coordinates with shape `(nnode, 2)`. Coordinates are `(r, z)` for
            `formulation="axisymmetric"` and `(x, y)` for `formulation="plane_strain"`.
            Units are `[length]`. Floating input arrays must have dtype `float64`.
        elements: Connectivity with shape `(nelem, 4)` for `element_type="quad4"`. For
            `element_type="quad9"`, pass either corner-only `(nelem, 4)` connectivity to infer a
            straight-sided quad9 mesh, or explicit `(nelem, 9)` connectivity in local order
            `[corner0, corner1, corner2, corner3, face0_mid, face1_mid, face2_mid, face3_mid,
            center]`. Corner nodes must be ordered counter-clockwise in the 2D analysis plane.
        material_ids: Dense material row indices with shape `(nelem,)`.
        material_table: Elastic stress-strain matrices with shape `(nmat, 4, 4)`. Matrix units
            are `[stress / strain] = [pressure]`.
        formulation: Symmetry reduction, either `"axisymmetric"` or `"plane_strain"`.
        thickness: Plane-strain out-of-plane thickness. Required only for
            `formulation="plane_strain"`.
        material_orientation_angles: Optional scalar or per-element angles, in radians, rotating
            local material axes into the global 2D frame before assembly.
        pressure_faces: Optional pressure-load topology with shape `(n_pressure_faces, 2)`. Each
            row is `[element_index, local_face]`.
        traction_faces: Optional traction-load topology with shape `(n_traction_faces, 2)`. Each
            row is `[element_index, local_face]`.
        thermal_material_table: Optional thermal material rows with shape `(nmat, 5)` storing
            `[alpha_r, alpha_z, alpha_t, alpha_rz, T_ref]`. Thermal expansion coefficients have
            units `[strain / temperature]` and `T_ref` has units `[temperature]`.
        prescribed: Optional mapping from full displacement DOF index to prescribed displacement
            value. Displacement units are `[length]`.
        quadrature: Quadrature rule selector, either `gl3`, `gl4`, `3`, or `4`.
        element_type: Analysis element family, either `quad4` or `quad9`.
        par: Whether to assemble stiffness and computed-on-call sparse exports using threaded
            element batches.

    Returns:
        Structural2DFEMModel: Reusable model with backend solve state and user-owned sparse
        operator exports.

    Raises:
        ValueError: If `quadrature` is unsupported.
        AssertionError: If array shapes are invalid or if mapping-style material inputs are passed
            instead of dense arrays.
    """

    nodes_arr = _normalize_nodes(nodes)
    material_ids_arr, material_table_arr = _normalize_materials(material_ids, material_table)
    thermal_material_table_arr = _normalize_thermal_material_table(
        thermal_material_table,
    )
    pressure_faces_arr = _normalize_face_pairs("pressure_faces", pressure_faces)
    traction_faces_arr = _normalize_face_pairs("traction_faces", traction_faces)
    prescribed_dofs, prescribed_values = _normalize_prescribed_dirichlet(prescribed)
    quadrature_code = _quadrature_code(quadrature)
    normalized_formulation = _normalize_formulation(formulation)
    thickness_value = _normalize_thickness(normalized_formulation, thickness)
    normalized_element_type = _normalize_element_type(element_type)
    if normalized_element_type == "quad4":
        elements_arr = _normalize_elements(elements, 4)
        analysis_nodes, analysis_elements, elevated = nodes_arr, elements_arr, None
    else:
        elements_arr = _normalize_elements(elements, (4, 9))
        if elements_arr.shape[1] == 4:
            elevated = infer_quad9_mesh(nodes_arr, elements_arr)
            analysis_nodes, analysis_elements = elevated.analysis_nodes, elevated.analysis_elements
        else:
            analysis_nodes, analysis_elements, elevated = nodes_arr, elements_arr, None
    material_orientation_angles_arr = _normalize_material_orientation_angles(
        material_orientation_angles,
        int(analysis_elements.shape[0]),
    )
    backend = _assemble_model_2d_f64(
        analysis_nodes,
        analysis_elements,
        material_ids_arr,
        material_table_arr,
        pressure_faces_arr,
        traction_faces_arr,
        (
            thermal_material_table_arr
            if thermal_material_table_arr is not None
            else np.zeros((0, 5), dtype=np.float64)
        ),
        material_orientation_angles_arr,
        prescribed_dofs,
        prescribed_values,
        4 if normalized_element_type == "quad4" else 9,
        _formulation_code(normalized_formulation),
        thickness_value,
        quadrature_code,
        par,
    )
    n_temperature_nodes = 0 if thermal_material_table_arr is None else nodes_arr.shape[0]

    model = Structural2DFEMModel(
        backend=backend,
        input_nodes=nodes_arr,
        input_elements=elements_arr,
        analysis_nodes=analysis_nodes,
        analysis_elements=analysis_elements,
        elevated=elevated,
        material_ids=material_ids_arr,
        material_table=material_table_arr,
        thermal_material_table=thermal_material_table_arr,
        material_orientation_angles=material_orientation_angles_arr,
        pressure_faces=pressure_faces_arr,
        traction_faces=traction_faces_arr,
        formulation=normalized_formulation,
        thickness=thickness_value,
        element_type=normalized_element_type,
        free_dofs=np.asarray(backend.free_dofs(), dtype=np.int64),
        fixed_dofs=np.asarray(backend.fixed_dofs(), dtype=np.int64),
        fixed_values=np.asarray(backend.fixed_values(), dtype=np.float64),
        ndof_full=int(backend.ndof_full),
        ndof_reduced=int(backend.ndof_reduced),
        nelem=elements_arr.shape[0],
        nq_per_element=int(backend.nq_per_element),
        n_temperature_nodes=n_temperature_nodes,
    )
    return model


def isotropic_axisymmetric_material(
    youngs_modulus: float,
    poisson_ratio: float,
) -> npt.NDArray[np.floating[Any]]:
    """Construct the isotropic axisymmetric elastic stress-strain matrix.

    Args:
        youngs_modulus: Young's modulus with units `[pressure]`.
        poisson_ratio: Poisson ratio with units `[dimensionless]`.

    Returns:
        NDArray: Elastic stress-strain matrix with shape `(4, 4)` in component order
        `[rr, zz, tt, rz]`. Units are `[stress / strain] = [pressure]`.
    """

    return _as_float64_array(_isotropic_axisymmetric_material_f64(youngs_modulus, poisson_ratio)).reshape(
        4, 4
    )


def isotropic_axisymmetric_thermal_material(
    alpha: float,
    reference_temperature: float = 0.0,
) -> npt.NDArray[np.floating[Any]]:
    """Construct isotropic thermal-expansion data.

    Args:
        alpha: Isotropic thermal expansion coefficient with units `[strain / temperature]`.
        reference_temperature: Stress-free reference temperature with units `[temperature]`.

    Returns:
        NDArray: Thermal material row with shape `(5,)` storing
        `[alpha_r, alpha_z, alpha_t, alpha_rz, T_ref]`. The first four entries have units
        `[strain / temperature]`; `T_ref` has units `[temperature]`.
    """

    return _as_float64_array(_isotropic_axisymmetric_thermal_material_f64(alpha, reference_temperature))


def isotropic_plane_strain_material(
    youngs_modulus: float,
    poisson_ratio: float,
) -> npt.NDArray[np.floating[Any]]:
    """Construct the isotropic plane-strain elastic stress-strain matrix.

    Returns a dense `(4, 4)` constitutive matrix in `[xx, yy, zz, xy]` order. The plane-strain
    solver sets `epsilon_zz = 0`, but this matrix still recovers the nonzero `sigma_zz` implied
    by the in-plane strains.
    """

    return _as_float64_array(_isotropic_plane_strain_material_f64(youngs_modulus, poisson_ratio)).reshape(
        4, 4
    )


def isotropic_plane_strain_thermal_material(
    alpha: float,
    reference_temperature: float = 0.0,
) -> npt.NDArray[np.floating[Any]]:
    """Construct isotropic plane-strain thermal-expansion data.

    Returns a row `[alpha_x, alpha_y, alpha_z, alpha_xy, T_ref]` with equal normal expansion
    coefficients and zero engineering shear expansion.
    """

    return _as_float64_array(_isotropic_plane_strain_thermal_material_f64(alpha, reference_temperature))


def orthotropic_axisymmetric_thermal_material(
    alpha_r: float,
    alpha_z: float,
    alpha_t: float,
    reference_temperature: float = 0.0,
) -> npt.NDArray[np.floating[Any]]:
    """Construct orthotropic thermal-expansion data.

    Args:
        alpha_r: Radial thermal expansion coefficient with units `[strain / temperature]`.
        alpha_z: Axial thermal expansion coefficient with units `[strain / temperature]`.
        alpha_t: Hoop thermal expansion coefficient with units `[strain / temperature]`.
        reference_temperature: Stress-free reference temperature with units `[temperature]`.

    Returns:
        NDArray: Thermal material row with shape `(5,)` storing
        `[alpha_r, alpha_z, alpha_t, alpha_rz, T_ref]`. The first four entries have units
        `[strain / temperature]`; `T_ref` has units `[temperature]`.
    """

    return _as_float64_array(
        _orthotropic_axisymmetric_thermal_material_f64(
            alpha_r,
            alpha_z,
            alpha_t,
            reference_temperature,
        )
    )


def orthotropic_plane_strain_thermal_material(
    alpha_x: float,
    alpha_y: float,
    alpha_z: float,
    reference_temperature: float = 0.0,
) -> npt.NDArray[np.floating[Any]]:
    """Construct orthotropic plane-strain thermal-expansion data.

    Returns a row `[alpha_x, alpha_y, alpha_z, alpha_xy, T_ref]` with zero engineering shear
    expansion. Use `material_orientation_angles` during assembly to rotate local orthotropic axes.
    """

    return orthotropic_axisymmetric_thermal_material(
        alpha_x,
        alpha_y,
        alpha_z,
        reference_temperature,
    )


def cfsem_radial_material(
    youngs_modulus: float,
    poisson_ratio: float,
) -> npt.NDArray[np.floating[Any]]:
    """Construct the reduced elastic matrix used by the 1D radial solver.

    Args:
        youngs_modulus: Young's modulus with units `[pressure]`.
        poisson_ratio: Poisson ratio with units `[dimensionless]`.

    Returns:
        NDArray: Elastic stress-strain matrix with shape `(4, 4)` in component order
        `[rr, zz, tt, rz]`. Units are `[stress / strain] = [pressure]`.
    """

    return _as_float64_array(_cfsem_radial_material_f64(youngs_modulus, poisson_ratio)).reshape(4, 4)


def _normalize_displacements(
    displacements: ArrayLike,
    nnode: int,
) -> npt.NDArray[np.floating[Any]]:
    arr = np.asarray(displacements)
    if arr.ndim == 1 and arr.shape == (2 * nnode,):
        return arr.reshape(nnode, 2)
    if arr.ndim == 2 and arr.shape == (nnode, 2):
        return arr
    raise ValueError(f"displacements must have shape (2*nnode,) or (nnode, 2); got {arr.shape}")


__all__ = [
    "Structural2DFEMModel",
    "ElementMeasures",
    "ElevatedQuad9Mesh",
    "PointLocations",
    "Quadrature",
    "QuadMeshInterpolation",
    "QuadMeshQuery",
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
    "query_quad_mesh",
]
