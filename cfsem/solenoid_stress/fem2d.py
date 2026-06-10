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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

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
_QUAD_FACE_NODE_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3), (3, 0))


def _to_csr_matrix(matrix: Any) -> sp.csr_matrix:
    """Normalize sparse results to the matrix API expected by this module."""

    return cast(sp.csr_matrix, sp.csr_matrix(matrix))


def _to_csc_matrix(matrix: Any) -> sp.csc_matrix:
    """Normalize sparse results to CSC matrices."""

    return cast(sp.csc_matrix, sp.csc_matrix(matrix))


def _as_float64_array(data: Any) -> npt.NDArray[np.float64]:
    """Convert binding output to a NumPy float64 array with an explicit static type."""

    return cast(npt.NDArray[np.float64], np.asarray(data, dtype=np.float64))


def _csr_matrix_from_binding(
    binding: tuple[ArrayLike, ArrayLike, ArrayLike, int, int],
) -> sp.csr_matrix:
    vals, indices, indptr, nrow, ncol = binding
    return _to_csr_matrix(
        sp.csr_matrix(
            (
                np.asarray(vals, dtype=np.float64),
                np.asarray(indices, dtype=np.int64),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(int(nrow), int(ncol)),
        )
    )


def _csc_matrix_from_binding(
    binding: tuple[ArrayLike, ArrayLike, ArrayLike, int, int],
) -> sp.csc_matrix:
    vals, indices, indptr, nrow, ncol = binding
    return _to_csc_matrix(
        sp.csc_matrix(
            (
                np.asarray(vals, dtype=np.float64),
                np.asarray(indices, dtype=np.int64),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(int(nrow), int(ncol)),
        )
    )


@dataclass(frozen=True, slots=True)
class ElementMeasures:
    """Per-element cross-section area and represented volume.

    `areas` and `volumes` both have shape `(nelem,)`.
    `areas` has units `[area]` and `volumes` has units `[volume]`.
    """

    areas: npt.NDArray[np.floating[Any]]
    volumes: npt.NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class ElementQuadrature:
    """Per-element physical quadrature points and mapped weights.

    `points` has shape `(nelem, nq_per_element, 2)`.
    `weights_area` and `weights_volume` have shape `(nelem, nq_per_element)`.
    `points` has units `[length]`, `weights_area` has units `[area]`, and
    `weights_volume` has units `[volume]`.
    """

    points: npt.NDArray[np.floating[Any]]
    weights_area: npt.NDArray[np.floating[Any]]
    weights_volume: npt.NDArray[np.floating[Any]]
    nq_per_element: int


@dataclass(frozen=True, slots=True)
class ElevatedQuad9Mesh:
    """Explicit 9-node analysis mesh inferred from a corner-only quad4 mesh.

    `analysis_elements` use the local quad9 ordering:
    - corners `0..3` in counter-clockwise order `[bottom-left, bottom-right, top-right, top-left]`
    - midsides `4..7` on faces `[bottom, right, top, left]`
    - center node `8`

    `input_nodes` and `analysis_nodes` have units `[length]`.
    """

    input_nodes: npt.NDArray[np.floating[Any]]
    input_elements: npt.NDArray[np.uint64]
    analysis_nodes: npt.NDArray[np.floating[Any]]
    analysis_elements: npt.NDArray[np.uint64]
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

    values: npt.NDArray[np.floating[Any]]
    element_indices: npt.NDArray[np.int64]
    reference_points: npt.NDArray[np.floating[Any]]
    inside: npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class QuadMeshQuery:
    """One-pass geometric query results for points in a 2D quadrilateral mesh.

    The query stores nearest-node, nearest-element, and nearest-face data for each query point.
    Interpolation and recovery operators can reuse this object without repeating the mesh search.
    For contained points, the nearest element is the containing element and
    `nearest_element_distances` is zero to numerical tolerance.
    """

    nodes: npt.NDArray[np.floating[Any]]
    elements: npt.NDArray[np.uint64]
    points: npt.NDArray[np.floating[Any]]
    element_type: str
    nearest_node_indices: npt.NDArray[np.int64]
    nearest_node_points: npt.NDArray[np.floating[Any]]
    nearest_node_distances: npt.NDArray[np.floating[Any]]
    nearest_element_indices: npt.NDArray[np.int64]
    nearest_element_reference_points: npt.NDArray[np.floating[Any]]
    nearest_element_points: npt.NDArray[np.floating[Any]]
    nearest_element_distances: npt.NDArray[np.floating[Any]]
    nearest_face_element_indices: npt.NDArray[np.int64]
    nearest_face_local_faces: npt.NDArray[np.int64]
    nearest_face_reference_coordinates: npt.NDArray[np.floating[Any]]
    nearest_face_points: npt.NDArray[np.floating[Any]]
    nearest_face_distances: npt.NDArray[np.floating[Any]]


class Structural2DFEMModel:
    """Reusable 2D structural FEM model with sparse operators and reduced solve state.

    Structural FEM numeric arrays are `float64`; floating inputs must already use `float64` arrays.

    The sparse operators are exported from the Rust backend on demand:
    - `body_force_to_rhs`, `pressure_to_rhs`, `traction_to_rhs`, and `temperature_to_rhs`
      map load amplitudes to the reduced structural right-hand side,
    - `strain_operator`, `stress_operator`, `thermal_strain_operator`, and
      `thermal_stress_operator` map solved displacements or nodal temperatures to quadrature-point
      fields.

    `build_rhs(...)`, `solve(...)`, `element_quadrature()`, `element_measures()`, and
    `evaluate_quadrature_strain(...)` call the Rust backend directly and do not materialize these
    sparse matrices. Sparse operator exports are user-owned artifacts; retain the returned SciPy
    matrix explicitly when a workflow needs reuse.

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
        input_nodes: npt.NDArray[np.floating[Any]],
        input_elements: npt.NDArray[np.uint64],
        analysis_nodes: npt.NDArray[np.floating[Any]],
        analysis_elements: npt.NDArray[np.uint64],
        elevated: ElevatedQuad9Mesh | None,
        pressure_faces: npt.NDArray[np.uint64],
        traction_faces: npt.NDArray[np.uint64],
        formulation: str,
        thickness: float,
        element_type: str,
        free_dofs: npt.NDArray[np.int64],
        fixed_dofs: npt.NDArray[np.int64],
        fixed_values: npt.NDArray[np.floating[Any]],
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
        self._element_quadrature_cache: ElementQuadrature | None = None
        self._element_measures_cache: ElementMeasures | None = None
        self._temperature_elevation_cache: sp.csr_matrix | None = None
        self._constant_rhs_cache: npt.NDArray[np.float64] | None = None
        self._quadrature_points_cache: npt.NDArray[np.float64] | None = None
        self._strain_constant_cache: npt.NDArray[np.float64] | None = None
        self._stress_constant_cache: npt.NDArray[np.float64] | None = None
        self._thermal_strain_constant_cache: npt.NDArray[np.float64] | None = None
        self._thermal_stress_constant_cache: npt.NDArray[np.float64] | None = None
        self._stiffness_cache: sp.csc_matrix | None = None

    @property
    def ndof(self) -> int:
        """Compatibility alias for `ndof_full`, the full displacement-vector length `(ndof_full,)`."""

        return self.ndof_full

    @property
    def input_nodes(self) -> npt.NDArray[np.floating[Any]]:
        """Corner-node input mesh coordinates with shape `(nnode, 2)` and units `[length]`."""

        return self._input_nodes

    @property
    def input_elements(self) -> npt.NDArray[np.uint64]:
        """Input mesh connectivity with shape `(nelem, 4)`."""

        return self._input_elements

    @property
    def constant_rhs(self) -> npt.NDArray[np.floating[Any]]:
        """Load-independent reduced RHS contribution, exported from Rust on first access."""

        cache = self._constant_rhs_cache
        if cache is None:
            cache = np.asarray(self._backend.constant_rhs(), dtype=np.float64)
            self._constant_rhs_cache = cache
        return cache

    @property
    def quadrature_points(self) -> npt.NDArray[np.floating[Any]]:
        """Quadrature-point coordinates with shape `(nelem, nq_per_element, 2)`.

        This property materializes point coordinates separately from field recovery so workflows
        that only need matrix-free values do not allocate point arrays.
        """

        cache = self._quadrature_points_cache
        if cache is None:
            cache = np.asarray(
                self._backend.quadrature_points_flat(),
                dtype=np.float64,
            ).reshape(self.analysis_elements.shape[0], self.nq_per_element, 2)
            self._quadrature_points_cache = cache
        return cache

    @property
    def strain_constant(self) -> npt.NDArray[np.floating[Any]]:
        """Strain offset from prescribed displacement DOFs, exported on first access."""

        cache = self._strain_constant_cache
        if cache is None:
            cache = np.asarray(self._backend.strain_constant(), dtype=np.float64)
            self._strain_constant_cache = cache
        return cache

    @property
    def stress_constant(self) -> npt.NDArray[np.floating[Any]]:
        """Stress offset from prescribed displacement DOFs, exported on first access."""

        cache = self._stress_constant_cache
        if cache is None:
            cache = np.asarray(self._backend.stress_constant(), dtype=np.float64)
            self._stress_constant_cache = cache
        return cache

    @property
    def thermal_strain_constant(self) -> npt.NDArray[np.floating[Any]]:
        """Thermal-strain reference-temperature offset, exported on first access."""

        cache = self._thermal_strain_constant_cache
        if cache is None:
            cache = np.asarray(self._backend.thermal_strain_constant(), dtype=np.float64)
            self._thermal_strain_constant_cache = cache
        return cache

    @property
    def thermal_stress_constant(self) -> npt.NDArray[np.floating[Any]]:
        """Thermal-stress reference-temperature offset, exported on first access."""

        cache = self._thermal_stress_constant_cache
        if cache is None:
            cache = np.asarray(self._backend.thermal_stress_constant(), dtype=np.float64)
            self._thermal_stress_constant_cache = cache
        return cache

    def _temperature_elevation(self) -> sp.csr_matrix:
        """Return the cached input-to-analysis temperature elevation operator.

        Corner-node quad9 inputs are elevated inside the Rust backend before assembly, while the
        Python API still accepts temperatures on the original input nodes.  This operator bridges
        those two spaces for exported scipy operators.  It is cached because the input mesh and
        inferred analysis mesh are immutable for the lifetime of the model.
        """

        elevated = self._elevated
        assert elevated is not None, "temperature elevation is available only for inferred quad9 meshes"
        cache = self._temperature_elevation_cache
        if cache is None:
            cache = _temperature_elevation_operator(elevated)
            self._temperature_elevation_cache = cache
        return cache

    def _cached_csc_export(self, attr: str, export: Callable[[], Any]) -> sp.csc_matrix:
        """Export one Rust-owned CSC operator to scipy on first access.

        The model is treated as immutable after assembly, so the cached scipy matrix remains valid
        for every later access to the corresponding read-only property.
        """

        cache = getattr(self, attr)
        if cache is None:
            cache = _csc_matrix_from_binding(export())
            setattr(self, attr, cache)
        return cast(sp.csc_matrix, cache)

    def _temperature_csr_export(
        self,
        export: Callable[[], Any],
    ) -> sp.csr_matrix:
        """Export a temperature-indexed CSR operator, including empty and elevated cases.

        Models without thermal materials expose zero-column operators for shape consistency.
        Inferred quad9 meshes export analysis-node operators from Rust, then postmultiply by the
        cached elevation operator so the public SciPy matrix acts on the original input nodes.
        """

        analysis_operator = _csr_matrix_from_binding(export())
        return (
            _to_csr_matrix(analysis_operator @ self._temperature_elevation())
            if self._elevated is not None and self.n_temperature_nodes > 0
            else analysis_operator
        )

    @property
    def stiffness(self) -> sp.csc_matrix:
        """Reduced stiffness matrix with shape `(ndof_reduced, ndof_reduced)`.

        Entries have units `[generalized force / displacement] = [energy / distance^2]`.
        The SciPy matrix is exported from the Rust backend on first access and then cached.
        """

        return self._cached_csc_export("_stiffness_cache", self._backend.stiffness_csc)

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

        return self._temperature_csr_export(
            self._backend.temperature_to_rhs_csr,
        )

    @property
    def strain_operator(self) -> sp.csr_matrix:
        """Operator mapping reduced displacement to quadrature-point total strain.

        Shape is `(4 * nelem * nq_per_element, ndof_reduced)`. Entries have units
        `[strain / displacement] = [1 / length]`.
        """

        return _csr_matrix_from_binding(self._backend.strain_operator_csr())

    @property
    def stress_operator(self) -> sp.csr_matrix:
        """Operator mapping reduced displacement to quadrature-point stress.

        Shape is `(4 * nelem * nq_per_element, ndof_reduced)`. Entries have units
        `[stress / displacement] = [pressure / length]`.
        """

        return _csr_matrix_from_binding(self._backend.stress_operator_csr())

    @property
    def thermal_strain_operator(self) -> sp.csr_matrix:
        """Operator mapping input-node temperatures to quadrature-point thermal strain.

        Shape is `(4 * nelem * nq_per_element, n_temperature_nodes)`. Entries have units
        `[strain / temperature]`.
        """

        return self._temperature_csr_export(
            self._backend.thermal_strain_operator_csr,
        )

    @property
    def thermal_stress_operator(self) -> sp.csr_matrix:
        """Operator mapping input-node temperatures to quadrature-point thermal stress.

        Shape is `(4 * nelem * nq_per_element, n_temperature_nodes)`. Entries have units
        `[stress / temperature]`.
        """

        return self._temperature_csr_export(
            self._backend.thermal_stress_operator_csr,
        )

    def element_quadrature(self) -> ElementQuadrature:
        """Return physical quadrature points and mapped weights for each element.

        Returns:
            ElementQuadrature: Quadrature data with:
                `points` of shape `(nelem, nq_per_element, 2)` and units `[length]`,
                `weights_area` of shape `(nelem, nq_per_element)` and units `[area]`,
                `weights_volume` of shape `(nelem, nq_per_element)` and units `[volume]`.
        """

        cache = self._element_quadrature_cache
        if cache is not None:
            return cache
        points_flat, weights_area_flat, weights_volume_flat, nq = self._backend.element_quadrature()
        nelem = self.analysis_elements.shape[0]
        points = np.asarray(points_flat, dtype=np.float64).reshape(nelem, int(nq), 2)
        weights_area = np.asarray(weights_area_flat, dtype=np.float64).reshape(nelem, int(nq))
        weights_volume = np.asarray(weights_volume_flat, dtype=np.float64).reshape(nelem, int(nq))
        cache = ElementQuadrature(
            points=points,
            weights_area=weights_area,
            weights_volume=weights_volume,
            nq_per_element=int(nq),
        )
        self._element_quadrature_cache = cache
        return cache

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
        areas, volumes = self._backend.element_measures()
        cache = ElementMeasures(
            areas=np.asarray(areas, dtype=np.float64),
            volumes=np.asarray(volumes, dtype=np.float64),
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
        return _analysis_temperature_for_element_type(
            nodal_temperature,
            self._input_nodes.shape[0],
            self._elevated,
        )

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

    def evaluate_quadrature_strain(
        self,
        displacements: ArrayLike,
    ) -> npt.NDArray[np.floating[Any]]:
        """Evaluate quadrature-point total strain without materializing recovery matrices.

        Args:
            displacements: Either the reduced displacement solution with shape
                `(ndof_reduced,)`, or the full analysis displacement field with shape
                `(2 * n_analysis_nodes,)` or `(n_analysis_nodes, 2)`. Displacement units are
                `[length]`.

        Returns:
            NDArray: Total strain with shape `(nelem, nq_per_element, 4)` and component ordering
            `[rr, zz, tt, rz]` for axisymmetric models or `[xx, yy, zz, xy]` for plane strain.
        """

        arr = np.asarray(displacements)
        if arr.ndim == 1 and arr.shape == (self.ndof_reduced,):
            displacements_full = self.recover_full(arr)
        else:
            displacements_full = _normalize_displacements(
                displacements, self.analysis_nodes.shape[0]
            ).reshape(-1)
        strain_flat, nq = self._backend.evaluate_quadrature_strain(displacements_full)
        nelem = self.analysis_elements.shape[0]
        nq = int(nq)
        return np.asarray(strain_flat, dtype=np.float64).reshape(nelem, nq, 4)


def _quadrature_code(quadrature: str | int) -> int:
    if quadrature in (3, "3", "gl3", "GL3"):
        return 3
    if quadrature in (4, "4", "gl4", "GL4"):
        return 4
    raise ValueError(f"unsupported quadrature {quadrature!r}; use 'gl3' or 'gl4'")


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
    if formulation == "axisymmetric":
        return 0
    if formulation == "plane_strain":
        return 1
    raise ValueError(f"unsupported formulation {formulation!r}")


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
) -> npt.NDArray[np.floating[Any]]:
    if material_orientation_angles is None:
        return np.zeros((0,), dtype=np.float64)
    angles = np.asarray(material_orientation_angles)
    if angles.ndim == 0:
        angles = np.broadcast_to(angles, (nelem,)).copy()
    assert angles.ndim == 1 and angles.shape[0] == nelem, (
        f"material_orientation_angles must be a scalar or have shape ({nelem},); " f"got {angles.shape}"
    )
    return angles


def _normalize_nodes(nodes: ArrayLike) -> npt.NDArray[np.floating[Any]]:
    arr = np.asarray(nodes)
    assert arr.ndim == 2 and arr.shape[1] == 2, f"nodes must have shape (nnode, 2); got {arr.shape}"
    return arr


def _normalize_elements(
    elements: ArrayLike,
    nodes_per_element: int | tuple[int, ...] = 4,
) -> npt.NDArray[np.uint64]:
    arr = np.asarray(elements)
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
) -> npt.NDArray[np.floating[Any]]:
    arr = np.asarray(points)
    assert arr.ndim == 2 and arr.shape[1] == 2, f"points must have shape (npoint, 2); got {arr.shape}"
    return arr


def _normalize_query_tolerance(
    tolerance: float | None,
) -> float:
    if tolerance is not None:
        value = float(tolerance)
        assert value >= 0.0, f"tolerance must be nonnegative; got {tolerance!r}"
        return value
    return 1.0e-10


def _coo_operator_from_binding(
    binding: tuple[ArrayLike, ArrayLike, ArrayLike, int, int],
) -> sp.csr_matrix:
    vals, rows, cols, nrow, ncol = binding
    return _to_csr_matrix(
        sp.coo_matrix(
            (
                np.asarray(vals, dtype=np.float64),
                (
                    np.asarray(rows, dtype=np.int64),
                    np.asarray(cols, dtype=np.int64),
                ),
            ),
            shape=(int(nrow), int(ncol)),
        ).tocsr()
    )


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


def quad_mesh_strain_operator(
    query: QuadMeshQuery,
    *,
    formulation: str,
    thickness: float | None = None,
) -> sp.csr_matrix:
    """Return a sparse operator mapping full nodal displacements to query-point strain.

    The returned matrix has shape `(4 * npoint, 2 * nnode)`. Rows are grouped by query point and
    use the same four-component strain ordering as the structural FEM formulation.

    Args:
        query: Mesh query data from `query_quad_mesh(...)`.
        formulation: Symmetry reduction, either `"axisymmetric"` or `"plane_strain"`.
        thickness: Plane-strain out-of-plane thickness with units `[length]`. Required only for
            `formulation="plane_strain"`; must be omitted for `formulation="axisymmetric"`.

    Returns:
        Sparse strain-recovery operator with shape `(4 * npoint, 2 * nnode)`. Entries have units
        `[1 / length]`, so multiplying by full nodal displacements with shape `(2 * nnode,)` and
        units `[length]` returns unitless strain samples with shape `(4 * npoint,)`.
    """

    normalized_formulation = _normalize_formulation(formulation)
    thickness_value = _normalize_thickness(normalized_formulation, thickness)
    return _coo_operator_from_binding(
        _quad_mesh_strain_operator_f64(
            query.nodes,
            query.elements,
            np.asarray(query.nearest_element_indices, dtype=np.uint64),
            query.nearest_element_reference_points,
            query.element_type,
            _formulation_code(normalized_formulation),
            thickness_value,
        ),
    )


def quad_mesh_stress_operator(
    query: QuadMeshQuery,
    material_ids: ArrayLike,
    material_table: ArrayLike,
    *,
    formulation: str,
    thickness: float | None = None,
    material_orientation_angles: ArrayLike | None = None,
) -> sp.csr_matrix:
    """Return a sparse operator mapping full nodal displacements to query-point stress.

    The returned matrix has shape `(4 * npoint, 2 * nnode)`. Rows are grouped by query point and
    use the same four-component stress ordering as the structural FEM formulation. Each query point
    uses the material row assigned to its nearest element; for points contained by the mesh, that is
    the containing element. Optional `material_orientation_angles` follow assembly semantics and
    rotate local anisotropic material axes into the global 2D frame per element.

    Args:
        query: Mesh query data from `query_quad_mesh(...)`.
        material_ids: Per-element material row indices with shape `(nelem,)`. Entries are unitless
            indices into `material_table`.
        material_table: Elastic stress-strain matrices with shape `(nmat, 4, 4)`. Entries have
            units `[stress / strain] = [pressure]`.
        formulation: Symmetry reduction, either `"axisymmetric"` or `"plane_strain"`.
        thickness: Plane-strain out-of-plane thickness with units `[length]`. Required only for
            `formulation="plane_strain"`; must be omitted for `formulation="axisymmetric"`.
        material_orientation_angles: Optional scalar or per-element angles with shape `(nelem,)`,
            in radians. Angles are unitless and rotate local material axes into the global 2D frame.

    Returns:
        Sparse stress-recovery operator with shape `(4 * npoint, 2 * nnode)`. Entries have units
        `[pressure / length]`, so multiplying by full nodal displacements with shape
        `(2 * nnode,)` and units `[length]` returns stress samples with shape `(4 * npoint,)` and
        units `[pressure]`.
    """

    material_ids_arr, material_table_arr = _normalize_materials(material_ids, material_table)
    assert material_ids_arr.shape == (
        query.elements.shape[0],
    ), f"material_ids must have shape ({query.elements.shape[0]},); got {material_ids_arr.shape}"
    material_orientation_angles_arr = _normalize_material_orientation_angles(
        material_orientation_angles,
        query.elements.shape[0],
    )
    normalized_formulation = _normalize_formulation(formulation)
    thickness_value = _normalize_thickness(normalized_formulation, thickness)
    return _coo_operator_from_binding(
        _quad_mesh_stress_operator_f64(
            query.nodes,
            query.elements,
            np.asarray(query.nearest_element_indices, dtype=np.uint64),
            query.nearest_element_reference_points,
            material_ids_arr,
            material_table_arr,
            material_orientation_angles_arr,
            query.element_type,
            _formulation_code(normalized_formulation),
            thickness_value,
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
    tol = _normalize_query_tolerance(tolerance)
    inside = query.nearest_element_distances <= tol
    if outside_policy in {"raise", "error"} and not np.all(inside):
        first = int(np.flatnonzero(~inside)[0])
        raise ValueError(f"query point {first} is outside the quad mesh")
    if outside_policy not in {"nearest", "nan", "raise", "error"}:
        raise ValueError(f"unsupported outside policy {outside!r}; use 'raise', 'nan', or 'nearest'")
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

    return _to_csr_matrix(
        sp.coo_matrix(
            (
                np.asarray(vals, dtype=np.float64),
                (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
            ),
            shape=(n_analysis_nodes, n_input_nodes),
        )
    )


def _analysis_temperature_for_element_type(
    nodal_temperature: ArrayLike,
    n_input_nodes: int,
    elevated: ElevatedQuad9Mesh | None,
) -> npt.NDArray[np.floating[Any]]:
    if elevated is None:
        return _normalize_nodal_temperature(nodal_temperature, n_input_nodes)
    input_temperature = _normalize_nodal_temperature(
        nodal_temperature,
        n_input_nodes,
    )
    elevation = _temperature_elevation_operator(elevated)
    return np.asarray(elevation @ input_temperature, dtype=np.float64)


def _normalize_materials(
    material_ids: ArrayLike,
    material_table: ArrayLike,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.floating[Any]]]:
    ids = np.asarray(material_ids)
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
) -> npt.NDArray[np.floating[Any]] | None:
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
        matrix = cast(
            npt.NDArray[np.floating[Any]],
            np.asarray(material_table_by_tag[tag], dtype=resolved_dtype),
        )
        assert matrix.shape == (
            4,
            4,
        ), f"material_table_by_tag[{tag}] must have shape (4, 4); got {matrix.shape}"
        material_rows.append(matrix)
    packed_material_table = cast(
        npt.NDArray[np.floating[Any]],
        np.ascontiguousarray(np.stack(material_rows, axis=0), dtype=resolved_dtype),
    )

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
            row = cast(
                npt.NDArray[np.floating[Any]],
                np.asarray(thermal_material_table_by_tag[tag], dtype=resolved_dtype),
            )
            assert row.shape == (
                5,
            ), f"thermal_material_table_by_tag[{tag}] must have shape (5,); got {row.shape}"
            thermal_rows.append(row)
        packed_thermal_table = cast(
            npt.NDArray[np.floating[Any]],
            np.ascontiguousarray(np.stack(thermal_rows, axis=0), dtype=resolved_dtype),
        )
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


def _normalize_face_pairs(name: str, faces: ArrayLike | None) -> npt.NDArray[np.uint64]:
    if faces is None:
        return np.zeros((0, 2), dtype=np.uint64)
    faces = np.asarray(faces)
    assert faces.ndim == 2 and faces.shape[1] == 2, f"{name} must have shape (nload, 2); got {faces.shape}"
    return faces


def _normalize_pressure_values(
    pressure_values: ArrayLike | None,
    nload: int,
) -> npt.NDArray[np.floating[Any]]:
    if pressure_values is None:
        return np.zeros((nload,), dtype=np.float64)
    values = np.asarray(pressure_values)
    assert values.ndim == 1, f"pressure_values must have shape (nload,); got {values.shape}"
    assert values.shape[0] == nload, f"pressure_values has {values.shape[0]} entries, but expected {nload}"
    return values


def _normalize_traction_values(
    traction_values: ArrayLike | None,
    nload: int,
) -> npt.NDArray[np.floating[Any]]:
    if traction_values is None:
        return np.zeros((nload, 2), dtype=np.float64)
    values = np.asarray(traction_values)
    if values.ndim == 1 and values.shape == (2,):
        values = np.broadcast_to(values, (nload, 2)).copy()
    assert values.ndim == 2 and values.shape == (
        nload,
        2,
    ), f"traction_values must have shape (2,) or ({nload}, 2); got {values.shape}"
    return values


def _normalize_prescribed_dirichlet(
    prescribed: Mapping[int, float] | None,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.floating[Any]]]:
    if prescribed is None:
        return np.zeros((0,), dtype=np.uint64), np.zeros((0,), dtype=np.float64)
    items = sorted((int(dof), float(value)) for dof, value in prescribed.items())
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
    "ElementQuadrature",
    "ElevatedQuad9Mesh",
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
    "quad_mesh_interpolation_operator",
    "quad_mesh_stress_operator",
    "quad_mesh_strain_operator",
    "query_quad_mesh",
]
