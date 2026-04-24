"""2D structural elasticity finite-element assembly.

This module provides a small displacement-based quadrilateral FEM solver for axisymmetric and
plane-strain structural reductions. The backend stores sparse load operators, sparse
quadrature-recovery operators, and a reduced stiffness matrix for repeated load solves.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

import cfsem.cfsem as _cfsem_bindings

_assemble_model_2d_f32 = _cfsem_bindings.solenoid_stress_fem_assemble_model_2d_f32
_assemble_model_2d_f64 = _cfsem_bindings.solenoid_stress_fem_assemble_model_2d_f64
_cfsem_radial_material_f32 = _cfsem_bindings.solenoid_stress_fem_cfsem_radial_material_f32
_cfsem_radial_material_f64 = _cfsem_bindings.solenoid_stress_fem_cfsem_radial_material_f64
_infer_quad9_mesh_f32 = _cfsem_bindings.solenoid_stress_fem_infer_quad9_mesh_f32
_infer_quad9_mesh_f64 = _cfsem_bindings.solenoid_stress_fem_infer_quad9_mesh_f64
_isotropic_axisymmetric_material_f32 = _cfsem_bindings.solenoid_stress_fem_isotropic_axisymmetric_material_f32
_isotropic_axisymmetric_material_f64 = _cfsem_bindings.solenoid_stress_fem_isotropic_axisymmetric_material_f64
_isotropic_plane_strain_material_f32 = _cfsem_bindings.solenoid_stress_fem_isotropic_plane_strain_material_f32
_isotropic_plane_strain_material_f64 = _cfsem_bindings.solenoid_stress_fem_isotropic_plane_strain_material_f64
_isotropic_axisymmetric_thermal_material_f32 = (
    _cfsem_bindings.solenoid_stress_fem_isotropic_axisymmetric_thermal_material_f32
)
_isotropic_axisymmetric_thermal_material_f64 = (
    _cfsem_bindings.solenoid_stress_fem_isotropic_axisymmetric_thermal_material_f64
)
_isotropic_plane_strain_thermal_material_f32 = (
    _cfsem_bindings.solenoid_stress_fem_isotropic_plane_strain_thermal_material_f32
)
_isotropic_plane_strain_thermal_material_f64 = (
    _cfsem_bindings.solenoid_stress_fem_isotropic_plane_strain_thermal_material_f64
)
_orthotropic_axisymmetric_thermal_material_f32 = (
    _cfsem_bindings.solenoid_stress_fem_orthotropic_axisymmetric_thermal_material_f32
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


def _sparse_shape(matrix: Any) -> tuple[int, int]:
    """Return a concrete 2D sparse shape for pyright and runtime callers."""

    return cast(tuple[int, int], matrix.shape)


def _as_float_array(data: Any, dtype: np.dtype[Any]) -> npt.NDArray[np.floating[Any]]:
    """Convert binding output to a NumPy floating array with an explicit static type."""

    return cast(npt.NDArray[np.floating[Any]], np.asarray(data, dtype=dtype))


def _csr_matrix_from_binding(
    binding: tuple[ArrayLike, ArrayLike, ArrayLike, int, int],
    dtype: np.dtype[Any],
) -> sp.csr_matrix:
    vals, indices, indptr, nrow, ncol = binding
    return _to_csr_matrix(
        sp.csr_matrix(
            (
                np.asarray(vals, dtype=dtype),
                np.asarray(indices, dtype=np.int64),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(int(nrow), int(ncol)),
        )
    )


def _csc_matrix_from_binding(
    binding: tuple[ArrayLike, ArrayLike, ArrayLike, int, int],
    dtype: np.dtype[Any],
) -> sp.csc_matrix:
    vals, indices, indptr, nrow, ncol = binding
    return _to_csc_matrix(
        sp.csc_matrix(
            (
                np.asarray(vals, dtype=dtype),
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


class Structural2DFEMModel:
    """Reusable 2D structural FEM model with sparse operators and reduced solve state.

    The stored sparse operators are the primary reusable objects:
    - `body_force_to_rhs`, `pressure_to_rhs`, `traction_to_rhs`, and `temperature_to_rhs`
      map load amplitudes to the reduced structural right-hand side,
    - `strain_operator`, `stress_operator`, `thermal_strain_operator`, and
      `thermal_stress_operator` map solved displacements or nodal temperatures to quadrature-point
      fields.

    `build_rhs(...)`, `solve(...)`, `element_quadrature()`, `element_measures()`, and
    `evaluate_quadrature(...)` are convenience methods layered on top of those stored operators
    and the reduced stiffness matrix.

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
        dtype: np.dtype[Any],
        input_nodes: npt.NDArray[np.floating[Any]],
        input_elements: npt.NDArray[np.uint64],
        analysis_nodes: npt.NDArray[np.floating[Any]],
        analysis_elements: npt.NDArray[np.uint64],
        elevated: ElevatedQuad9Mesh | None,
        pressure_faces: npt.NDArray[np.uint64],
        traction_faces: npt.NDArray[np.uint64],
        formulation: str,
        element_type: str,
        stiffness: sp.csc_matrix,
        body_force_to_rhs: sp.csr_matrix,
        pressure_to_rhs: sp.csr_matrix,
        traction_to_rhs: sp.csr_matrix,
        temperature_to_rhs: sp.csr_matrix,
        constant_rhs: npt.NDArray[np.floating[Any]],
        quadrature_points: npt.NDArray[np.floating[Any]],
        strain_operator: sp.csr_matrix,
        stress_operator: sp.csr_matrix,
        thermal_strain_operator: sp.csr_matrix,
        thermal_stress_operator: sp.csr_matrix,
        strain_constant: npt.NDArray[np.floating[Any]],
        stress_constant: npt.NDArray[np.floating[Any]],
        thermal_strain_constant: npt.NDArray[np.floating[Any]],
        thermal_stress_constant: npt.NDArray[np.floating[Any]],
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
        self._dtype = dtype
        self._input_nodes = input_nodes
        self._input_elements = input_elements
        self._elevated = elevated
        self.stiffness = stiffness
        self.body_force_to_rhs = body_force_to_rhs
        self.pressure_to_rhs = pressure_to_rhs
        self.traction_to_rhs = traction_to_rhs
        self.temperature_to_rhs = temperature_to_rhs
        self.constant_rhs = constant_rhs
        self.pressure_faces = pressure_faces
        self.traction_faces = traction_faces
        self.analysis_nodes = analysis_nodes
        self.analysis_elements = analysis_elements
        self.quadrature_points = quadrature_points
        self.strain_operator = strain_operator
        self.stress_operator = stress_operator
        self.thermal_strain_operator = thermal_strain_operator
        self.thermal_stress_operator = thermal_stress_operator
        self.strain_constant = strain_constant
        self.stress_constant = stress_constant
        self.thermal_strain_constant = thermal_strain_constant
        self.thermal_stress_constant = thermal_stress_constant
        self.free_dofs = free_dofs
        self.fixed_dofs = fixed_dofs
        self.fixed_values = fixed_values
        self.formulation = formulation
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

    @property
    def dtype(self) -> np.dtype[Any]:
        """Floating dtype used by all stored operators, arrays, and convenience-method outputs."""

        return self._dtype

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
        points = np.asarray(points_flat, dtype=self.dtype).reshape(nelem, int(nq), 2)
        weights_area = np.asarray(weights_area_flat, dtype=self.dtype).reshape(nelem, int(nq))
        weights_volume = np.asarray(weights_volume_flat, dtype=self.dtype).reshape(nelem, int(nq))
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
            areas=np.asarray(areas, dtype=self.dtype),
            volumes=np.asarray(volumes, dtype=self.dtype),
        )
        self._element_measures_cache = cache
        return cache

    def _normalize_temperature_for_backend(
        self,
        nodal_temperature: ArrayLike | None,
    ) -> npt.NDArray[np.floating[Any]] | None:
        if self.n_temperature_nodes == 0:
            values = (
                np.zeros((0,), dtype=self.dtype)
                if nodal_temperature is None
                else np.asarray(nodal_temperature, dtype=self.dtype).reshape(-1)
            )
            assert values.size == 0, "nodal_temperature was provided, but this model has no thermal operator"
            return None
        if nodal_temperature is None:
            raise ValueError("nodal_temperature is required because this model includes thermal materials")
        return _analysis_temperature_for_element_type(
            nodal_temperature,
            self._input_nodes.shape[0],
            self._elevated,
            self.dtype,
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

        body_force_arr = (
            np.zeros((self.nelem, 2), dtype=self.dtype)
            if body_force is None
            else _normalize_body_force(body_force, self.nelem, self.dtype)
        )
        _, nload = _sparse_shape(self.pressure_to_rhs)
        pressure_arr = _normalize_pressure_values(pressure_values, nload, self.dtype)
        _, ntraction_cols = _sparse_shape(self.traction_to_rhs)
        traction_arr = _normalize_traction_values(
            traction_values,
            ntraction_cols // 2,
            self.dtype,
        )
        temperature_arr = self._normalize_temperature_for_backend(nodal_temperature)
        rhs = self._backend.build_rhs(
            body_force_arr.reshape(-1),
            pressure_arr if pressure_arr.size else None,
            traction_arr.reshape(-1) if traction_arr.size else None,
            temperature_arr,
        )
        return np.asarray(rhs, dtype=self.dtype)

    def solve(self, rhs: ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """Solve the reduced system and recover the full displacement field.

        Args:
            rhs: Reduced right-hand side with shape `(ndof_reduced,)` and units
                `[generalized force] = [energy / distance]`.

        Returns:
            NDArray: Full displacement vector with shape `(ndof_full,)` and component ordering
            `[u_r0, u_z0, u_r1, u_z1, ...]`. Units are `[length]`.
        """

        rhs_arr = np.asarray(rhs, dtype=self.dtype).reshape(-1)
        assert (
            rhs_arr.shape[0] == self.ndof_reduced
        ), f"rhs must have length {self.ndof_reduced}; got {rhs_arr.shape}"
        return np.asarray(self._backend.solve(rhs_arr), dtype=self.dtype)

    def recover_full(self, reduced_solution: ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """Reinsert prescribed Dirichlet values into a reduced displacement vector.

        Args:
            reduced_solution: Reduced displacement vector with shape `(ndof_reduced,)` and units
                `[length]`.

        Returns:
            NDArray: Full displacement vector with shape `(ndof_full,)` and component ordering
            `[u_r0, u_z0, u_r1, u_z1, ...]`. Units are `[length]`.
        """

        reduced_arr = np.asarray(reduced_solution, dtype=self.dtype).reshape(-1)
        assert (
            reduced_arr.shape[0] == self.ndof_reduced
        ), f"reduced_solution must have length {self.ndof_reduced}; got {reduced_arr.shape}"
        full = np.zeros((self.ndof_full,), dtype=self.dtype)
        full[self.fixed_dofs] = self.fixed_values
        full[self.free_dofs] = reduced_arr
        return full

    def evaluate_quadrature(
        self,
        displacements: ArrayLike,
        nodal_temperature: ArrayLike | None = None,
    ) -> QuadratureFieldSamples:
        """Evaluate quadrature-point strain and stress fields.

        Args:
            displacements: Either the reduced displacement solution with shape
                `(ndof_reduced,)`, or the full analysis displacement field with shape
                `(2 * n_analysis_nodes,)` or `(n_analysis_nodes, 2)`. Displacement units are
                `[length]`.
            nodal_temperature: Input-node temperatures with shape `(n_input_nodes,)` and units
                `[temperature]`. Required only when the model includes thermal materials.

        Returns:
            QuadratureFieldSamples: Recovered quadrature fields where:
                `points` has shape `(nelem, nq_per_element, 2)` and units `[length]`,
                `strain`, `thermal_strain`, and `elastic_strain` have shape
                `(nelem, nq_per_element, 4)` and units `[strain]`,
                `stress` has shape `(nelem, nq_per_element, 4)` and units `[stress]`.

        Raises:
            ValueError: If thermal materials are present but `nodal_temperature` is omitted.
        """

        arr = np.asarray(displacements, dtype=self.dtype)
        if arr.ndim == 1 and arr.shape == (self.ndof_reduced,):
            displacements_full = self.recover_full(arr)
        else:
            displacements_full = _normalize_displacements(
                displacements, self.analysis_nodes.shape[0], self.dtype
            ).reshape(-1)
        temperature_arr = self._normalize_temperature_for_backend(nodal_temperature)
        (
            points_flat,
            strain_flat,
            thermal_strain_flat,
            elastic_strain_flat,
            stress_flat,
            nq,
        ) = self._backend.evaluate_quadrature(displacements_full, temperature_arr)
        nelem = self.analysis_elements.shape[0]
        nq = int(nq)
        return QuadratureFieldSamples(
            points=np.asarray(points_flat, dtype=self.dtype).reshape(nelem, nq, 2),
            strain=np.asarray(strain_flat, dtype=self.dtype).reshape(nelem, nq, 4),
            thermal_strain=np.asarray(thermal_strain_flat, dtype=self.dtype).reshape(nelem, nq, 4),
            elastic_strain=np.asarray(elastic_strain_flat, dtype=self.dtype).reshape(nelem, nq, 4),
            stress=np.asarray(stress_flat, dtype=self.dtype).reshape(nelem, nq, 4),
        )


@dataclass(frozen=True, slots=True)
class QuadratureFieldSamples:
    """Recovered strain and stress at element quadrature points.

    Each field has shape `(nelem, nq_per_element, 4)` with component ordering
    `[rr, zz, tt, rz]`. `points` has shape `(nelem, nq_per_element, 2)`.
    `points` has units `[length]`, `strain`, `thermal_strain`, and `elastic_strain`
    have units `[strain]`, and `stress` has units `[stress]`.
    """

    points: npt.NDArray[np.floating[Any]]
    strain: npt.NDArray[np.floating[Any]]
    thermal_strain: npt.NDArray[np.floating[Any]]
    elastic_strain: npt.NDArray[np.floating[Any]]
    stress: npt.NDArray[np.floating[Any]]

    @property
    def total_strain(self) -> npt.NDArray[np.floating[Any]]:
        """Alias for `strain`, with shape `(nelem, nq_per_element, 4)` and units `[strain]`."""

        return self.strain


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
    dtype: np.dtype[Any],
) -> float:
    if formulation == "axisymmetric":
        assert thickness is None, "thickness is only valid for formulation='plane_strain'"
        return float(np.asarray(0.0, dtype=dtype))
    assert thickness is not None, "thickness is required for formulation='plane_strain'"
    value = float(np.asarray(thickness, dtype=dtype))
    assert value > 0.0, f"thickness must be positive; got {thickness!r}"
    return value


def _normalize_material_orientation_angles(
    material_orientation_angles: ArrayLike | None,
    nelem: int,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    if material_orientation_angles is None:
        return np.zeros((0,), dtype=dtype)
    angles = np.asarray(material_orientation_angles, dtype=dtype)
    if angles.ndim == 0:
        angles = np.broadcast_to(angles, (nelem,)).copy()
    assert angles.ndim == 1 and angles.shape[0] == nelem, (
        f"material_orientation_angles must be a scalar or have shape ({nelem},); " f"got {angles.shape}"
    )
    return np.ascontiguousarray(angles)


def _resolve_float_dtype(*values: object) -> np.dtype[np.float32] | np.dtype[np.float64]:
    arrays: list[np.ndarray[Any, Any]] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, Mapping):
            arrays.extend(np.asarray(item) for item in value.values())
        else:
            arrays.append(np.asarray(value))
    result = np.dtype(np.result_type(*arrays, np.float32))
    if result.kind != "f" or result.itemsize <= 4:
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _normalize_nodes(nodes: ArrayLike, dtype: np.dtype[Any]) -> npt.NDArray[np.floating[Any]]:
    arr = np.asarray(nodes, dtype=dtype)
    assert arr.ndim == 2 and arr.shape[1] == 2, f"nodes must have shape (nnode, 2); got {arr.shape}"
    return np.ascontiguousarray(arr)


def _normalize_elements(elements: ArrayLike) -> npt.NDArray[np.uint64]:
    arr = np.asarray(elements, dtype=np.uint64)
    assert arr.ndim == 2 and arr.shape[1] == 4, f"elements must have shape (nelem, 4); got {arr.shape}"
    return np.ascontiguousarray(arr)


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

    dtype = _resolve_float_dtype(nodes)
    nodes_arr = _normalize_nodes(nodes, dtype)
    elements_arr = _normalize_elements(elements)
    binding = _dispatch_pair(dtype, _infer_quad9_mesh_f32, _infer_quad9_mesh_f64)
    (
        analysis_nodes_flat,
        analysis_elements_flat,
        corner_node_indices,
        midside_node_indices,
        center_node_indices,
    ) = binding(nodes_arr, elements_arr)

    return ElevatedQuad9Mesh(
        input_nodes=nodes_arr,
        input_elements=elements_arr,
        analysis_nodes=np.asarray(analysis_nodes_flat, dtype=dtype).reshape(-1, 2),
        analysis_elements=np.asarray(analysis_elements_flat, dtype=np.uint64).reshape(-1, 9),
        corner_node_indices=np.asarray(corner_node_indices, dtype=np.int64),
        midside_node_indices=np.asarray(midside_node_indices, dtype=np.int64),
        center_node_indices=np.asarray(center_node_indices, dtype=np.int64),
    )


def _temperature_elevation_operator(
    elevated: ElevatedQuad9Mesh,
    dtype: np.dtype[Any],
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
                np.asarray(vals, dtype=dtype),
                (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
            ),
            shape=(n_analysis_nodes, n_input_nodes),
        )
    )


def _analysis_temperature_for_element_type(
    nodal_temperature: ArrayLike,
    n_input_nodes: int,
    elevated: ElevatedQuad9Mesh | None,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    if elevated is None:
        return _normalize_nodal_temperature(nodal_temperature, n_input_nodes, dtype)
    input_temperature = _normalize_nodal_temperature(
        nodal_temperature,
        n_input_nodes,
        dtype,
    )
    elevation = _temperature_elevation_operator(elevated, dtype)
    return np.asarray(elevation @ input_temperature, dtype=dtype)


def _normalize_materials(
    material_ids: ArrayLike,
    material_table: ArrayLike,
    dtype: np.dtype[Any],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.floating[Any]]]:
    ids = np.asarray(material_ids, dtype=np.uint64)
    assert ids.ndim == 1, f"material_ids must have shape (nelem,); got {ids.shape}"
    assert not isinstance(
        material_table, Mapping
    ), "material_table must be a dense array; use pack_material_tables_from_tags(...) for tagged inputs"
    table = np.asarray(material_table, dtype=dtype)
    assert table.ndim == 3 and table.shape[1:] == (
        4,
        4,
    ), f"material_table must have shape (nmat, 4, 4); got {table.shape}"
    return np.ascontiguousarray(ids), np.ascontiguousarray(table)


def _normalize_thermal_material_table(
    material_ids: ArrayLike,
    thermal_material_table: ArrayLike | None,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]] | None:
    if thermal_material_table is None:
        return None
    ids = np.asarray(material_ids, dtype=np.uint64)
    assert ids.ndim == 1, f"material_ids must have shape (nelem,); got {ids.shape}"
    assert not isinstance(
        thermal_material_table, Mapping
    ), "thermal_material_table must be a dense array; use pack_material_tables_from_tags(...) for tagged inputs"
    table = np.asarray(thermal_material_table, dtype=dtype)
    assert (
        table.ndim == 2 and table.shape[1] == 5
    ), f"thermal_material_table must have shape (nmat, 5); got {table.shape}"
    table = np.ascontiguousarray(table)
    assert not np.any(table[:, 3] != 0.0), "shear thermal expansion (alpha_rz) is not yet supported"
    return table


def pack_material_tables_from_tags(
    material_ids: ArrayLike,
    material_table_by_tag: Mapping[int, ArrayLike],
    thermal_material_table_by_tag: Mapping[int, ArrayLike] | None = None,
    dtype: npt.DTypeLike | None = None,
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
        dtype: Optional output floating dtype. Defaults to the resolved dtype of the provided
            material rows.

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
    resolved_dtype = (
        np.dtype(dtype)
        if dtype is not None
        else _resolve_float_dtype(
            material_table_by_tag,
            thermal_material_table_by_tag,
        )
    )
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
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    arr = np.asarray(nodal_temperature, dtype=dtype)
    assert (
        arr.ndim == 1 and arr.shape[0] == nnode
    ), f"nodal_temperature must have shape ({nnode},); got {arr.shape}"
    return np.ascontiguousarray(arr)


def _normalize_body_force(
    body_force: ArrayLike,
    nelem: int,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    arr = np.asarray(body_force, dtype=dtype)
    if arr.ndim == 1 and arr.shape == (2,):
        arr = np.broadcast_to(arr, (nelem, 2)).copy()
    assert arr.ndim == 2 and arr.shape == (
        nelem,
        2,
    ), f"body_force must have shape (2,) or (nelem, 2); got {arr.shape}"
    return np.ascontiguousarray(arr)


def _normalize_face_pairs(name: str, faces: ArrayLike | None) -> npt.NDArray[np.uint64]:
    if faces is None:
        return np.zeros((0, 2), dtype=np.uint64)
    faces = np.asarray(faces, dtype=np.uint64)
    assert faces.ndim == 2 and faces.shape[1] == 2, f"{name} must have shape (nload, 2); got {faces.shape}"
    return np.ascontiguousarray(faces)


def _normalize_pressure_values(
    pressure_values: ArrayLike | None,
    nload: int,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    if pressure_values is None:
        return np.zeros((nload,), dtype=dtype)
    values = np.asarray(pressure_values, dtype=dtype)
    assert values.ndim == 1, f"pressure_values must have shape (nload,); got {values.shape}"
    assert values.shape[0] == nload, f"pressure_values has {values.shape[0]} entries, but expected {nload}"
    return np.ascontiguousarray(values)


def _normalize_traction_values(
    traction_values: ArrayLike | None,
    nload: int,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    if traction_values is None:
        return np.zeros((nload, 2), dtype=dtype)
    values = np.asarray(traction_values, dtype=dtype)
    if values.ndim == 1 and values.shape == (2,):
        values = np.broadcast_to(values, (nload, 2)).copy()
    assert values.ndim == 2 and values.shape == (
        nload,
        2,
    ), f"traction_values must have shape (2,) or ({nload}, 2); got {values.shape}"
    return np.ascontiguousarray(values)


def _normalize_prescribed_dirichlet(
    prescribed: Mapping[int, float] | None,
    dtype: np.dtype[Any],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.floating[Any]]]:
    if prescribed is None:
        return np.zeros((0,), dtype=np.uint64), np.zeros((0,), dtype=dtype)
    items = sorted((int(dof), float(value)) for dof, value in prescribed.items())
    return (
        np.asarray([dof for dof, _ in items], dtype=np.uint64),
        np.asarray([value for _, value in items], dtype=dtype),
    )


def _dispatch_pair(dtype: np.dtype[Any], f32: Any, f64: Any) -> Any:
    if dtype == np.float32:
        return f32
    return f64


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
) -> Structural2DFEMModel:
    """Assemble the reusable 2D structural FEM model.

    Args:
        nodes: Corner-node coordinates with shape `(nnode, 2)`. Coordinates are `(r, z)` for
            `formulation="axisymmetric"` and `(x, y)` for `formulation="plane_strain"`.
            Units are `[length]`.
        elements: Quad4 connectivity with shape `(nelem, 4)`. Corner nodes must be ordered
            counter-clockwise in the 2D analysis plane.
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

    Returns:
        Structural2DFEMModel: Reusable model storing the reduced stiffness matrix, sparse load
        operators, sparse quadrature-recovery operators, and cached solve state.

    Raises:
        ValueError: If `quadrature` is unsupported.
        AssertionError: If array shapes are invalid or if mapping-style material inputs are passed
            instead of dense arrays.
    """

    elements_arr = _normalize_elements(elements)
    dtype = _resolve_float_dtype(nodes, material_table, thermal_material_table)
    nodes_arr = _normalize_nodes(nodes, dtype)
    material_ids_arr, material_table_arr = _normalize_materials(material_ids, material_table, dtype)
    thermal_material_table_arr = _normalize_thermal_material_table(
        material_ids,
        thermal_material_table,
        dtype,
    )
    pressure_faces_arr = _normalize_face_pairs("pressure_faces", pressure_faces)
    traction_faces_arr = _normalize_face_pairs("traction_faces", traction_faces)
    prescribed_dofs, prescribed_values = _normalize_prescribed_dirichlet(prescribed, dtype)
    quadrature_code = _quadrature_code(quadrature)
    normalized_formulation = _normalize_formulation(formulation)
    thickness_value = _normalize_thickness(normalized_formulation, thickness, dtype)
    normalized_element_type = _normalize_element_type(element_type)
    if normalized_element_type == "quad4":
        analysis_nodes, analysis_elements, elevated = nodes_arr, elements_arr, None
    else:
        elevated = infer_quad9_mesh(nodes_arr, elements_arr)
        analysis_nodes, analysis_elements = elevated.analysis_nodes, elevated.analysis_elements
    material_orientation_angles_arr = _normalize_material_orientation_angles(
        material_orientation_angles,
        int(analysis_elements.shape[0]),
        dtype,
    )
    low_level = _dispatch_pair(
        dtype,
        _assemble_model_2d_f32,
        _assemble_model_2d_f64,
    )
    backend = low_level(
        analysis_nodes,
        analysis_elements,
        material_ids_arr,
        material_table_arr,
        pressure_faces_arr,
        traction_faces_arr,
        np.zeros((0, 5), dtype=dtype) if thermal_material_table_arr is None else thermal_material_table_arr,
        material_orientation_angles_arr,
        prescribed_dofs,
        prescribed_values,
        4 if normalized_element_type == "quad4" else 9,
        _formulation_code(normalized_formulation),
        thickness_value,
        quadrature_code,
    )

    stiffness = _csc_matrix_from_binding(backend.stiffness_csc(), dtype)
    body_force_to_rhs = _csr_matrix_from_binding(backend.body_force_to_rhs_csr(), dtype)
    pressure_to_rhs = _csr_matrix_from_binding(backend.pressure_to_rhs_csr(), dtype)
    traction_to_rhs = _csr_matrix_from_binding(backend.traction_to_rhs_csr(), dtype)
    analysis_temperature_to_rhs = _csr_matrix_from_binding(backend.temperature_to_rhs_csr(), dtype)
    strain_operator = _csr_matrix_from_binding(backend.strain_operator_csr(), dtype)
    stress_operator = _csr_matrix_from_binding(backend.stress_operator_csr(), dtype)
    analysis_thermal_strain_operator = _csr_matrix_from_binding(
        backend.thermal_strain_operator_csr(),
        dtype,
    )
    analysis_thermal_stress_operator = _csr_matrix_from_binding(
        backend.thermal_stress_operator_csr(),
        dtype,
    )
    if thermal_material_table_arr is None:
        temperature_to_rhs = sp.csr_matrix((int(backend.ndof_reduced), 0), dtype=dtype)
        thermal_strain_operator = sp.csr_matrix((_sparse_shape(strain_operator)[0], 0), dtype=dtype)
        thermal_stress_operator = sp.csr_matrix((_sparse_shape(stress_operator)[0], 0), dtype=dtype)
        n_temperature_nodes = 0
    else:
        if elevated is None:
            temperature_to_rhs = analysis_temperature_to_rhs
            thermal_strain_operator = analysis_thermal_strain_operator
            thermal_stress_operator = analysis_thermal_stress_operator
        else:
            temperature_elevation = _temperature_elevation_operator(elevated, dtype)
            temperature_to_rhs = _to_csr_matrix(analysis_temperature_to_rhs @ temperature_elevation)
            thermal_strain_operator = _to_csr_matrix(analysis_thermal_strain_operator @ temperature_elevation)
            thermal_stress_operator = _to_csr_matrix(analysis_thermal_stress_operator @ temperature_elevation)
        n_temperature_nodes = nodes_arr.shape[0]

    return Structural2DFEMModel(
        backend=backend,
        dtype=dtype,
        input_nodes=nodes_arr,
        input_elements=elements_arr,
        analysis_nodes=analysis_nodes,
        analysis_elements=analysis_elements,
        elevated=elevated,
        pressure_faces=pressure_faces_arr,
        traction_faces=traction_faces_arr,
        formulation=normalized_formulation,
        element_type=normalized_element_type,
        stiffness=stiffness,
        body_force_to_rhs=body_force_to_rhs,
        pressure_to_rhs=pressure_to_rhs,
        traction_to_rhs=traction_to_rhs,
        temperature_to_rhs=_to_csr_matrix(temperature_to_rhs),
        constant_rhs=np.asarray(backend.constant_rhs(), dtype=dtype),
        quadrature_points=np.asarray(
            backend.quadrature_points_flat(),
            dtype=dtype,
        ).reshape(analysis_elements.shape[0], int(backend.nq_per_element), 2),
        strain_operator=strain_operator,
        stress_operator=stress_operator,
        thermal_strain_operator=_to_csr_matrix(thermal_strain_operator),
        thermal_stress_operator=_to_csr_matrix(thermal_stress_operator),
        strain_constant=np.asarray(backend.strain_constant(), dtype=dtype),
        stress_constant=np.asarray(backend.stress_constant(), dtype=dtype),
        thermal_strain_constant=np.asarray(backend.thermal_strain_constant(), dtype=dtype),
        thermal_stress_constant=np.asarray(backend.thermal_stress_constant(), dtype=dtype),
        free_dofs=np.asarray(backend.free_dofs(), dtype=np.int64),
        fixed_dofs=np.asarray(backend.fixed_dofs(), dtype=np.int64),
        fixed_values=np.asarray(backend.fixed_values(), dtype=dtype),
        ndof_full=int(backend.ndof_full),
        ndof_reduced=int(backend.ndof_reduced),
        nelem=elements_arr.shape[0],
        nq_per_element=int(backend.nq_per_element),
        n_temperature_nodes=n_temperature_nodes,
    )


def isotropic_axisymmetric_material(
    youngs_modulus: float,
    poisson_ratio: float,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating[Any]]:
    """Construct the isotropic axisymmetric elastic stress-strain matrix.

    Args:
        youngs_modulus: Young's modulus with units `[pressure]`.
        poisson_ratio: Poisson ratio with units `[dimensionless]`.
        dtype: Output floating dtype.

    Returns:
        NDArray: Elastic stress-strain matrix with shape `(4, 4)` in component order
        `[rr, zz, tt, rz]`. Units are `[stress / strain] = [pressure]`.
    """

    resolved_dtype = np.dtype(dtype)
    binding = _dispatch_pair(
        resolved_dtype,
        _isotropic_axisymmetric_material_f32,
        _isotropic_axisymmetric_material_f64,
    )
    return _as_float_array(binding(youngs_modulus, poisson_ratio), resolved_dtype).reshape(4, 4)


def isotropic_axisymmetric_thermal_material(
    alpha: float,
    reference_temperature: float = 0.0,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating[Any]]:
    """Construct isotropic thermal-expansion data.

    Args:
        alpha: Isotropic thermal expansion coefficient with units `[strain / temperature]`.
        reference_temperature: Stress-free reference temperature with units `[temperature]`.
        dtype: Output floating dtype.

    Returns:
        NDArray: Thermal material row with shape `(5,)` storing
        `[alpha_r, alpha_z, alpha_t, alpha_rz, T_ref]`. The first four entries have units
        `[strain / temperature]`; `T_ref` has units `[temperature]`.
    """

    resolved_dtype = np.dtype(dtype)
    binding = _dispatch_pair(
        resolved_dtype,
        _isotropic_axisymmetric_thermal_material_f32,
        _isotropic_axisymmetric_thermal_material_f64,
    )
    return _as_float_array(binding(alpha, reference_temperature), resolved_dtype)


def isotropic_plane_strain_material(
    youngs_modulus: float,
    poisson_ratio: float,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating[Any]]:
    """Construct the isotropic plane-strain elastic stress-strain matrix."""

    resolved_dtype = np.dtype(dtype)
    binding = _dispatch_pair(
        resolved_dtype,
        _isotropic_plane_strain_material_f32,
        _isotropic_plane_strain_material_f64,
    )
    return _as_float_array(binding(youngs_modulus, poisson_ratio), resolved_dtype).reshape(4, 4)


def isotropic_plane_strain_thermal_material(
    alpha: float,
    reference_temperature: float = 0.0,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating[Any]]:
    """Construct isotropic plane-strain thermal-expansion data."""

    resolved_dtype = np.dtype(dtype)
    binding = _dispatch_pair(
        resolved_dtype,
        _isotropic_plane_strain_thermal_material_f32,
        _isotropic_plane_strain_thermal_material_f64,
    )
    return _as_float_array(binding(alpha, reference_temperature), resolved_dtype)


def orthotropic_axisymmetric_thermal_material(
    alpha_r: float,
    alpha_z: float,
    alpha_t: float,
    reference_temperature: float = 0.0,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating[Any]]:
    """Construct orthotropic thermal-expansion data.

    Args:
        alpha_r: Radial thermal expansion coefficient with units `[strain / temperature]`.
        alpha_z: Axial thermal expansion coefficient with units `[strain / temperature]`.
        alpha_t: Hoop thermal expansion coefficient with units `[strain / temperature]`.
        reference_temperature: Stress-free reference temperature with units `[temperature]`.
        dtype: Output floating dtype.

    Returns:
        NDArray: Thermal material row with shape `(5,)` storing
        `[alpha_r, alpha_z, alpha_t, alpha_rz, T_ref]`. The first four entries have units
        `[strain / temperature]`; `T_ref` has units `[temperature]`.
    """

    resolved_dtype = np.dtype(dtype)
    binding = _dispatch_pair(
        resolved_dtype,
        _orthotropic_axisymmetric_thermal_material_f32,
        _orthotropic_axisymmetric_thermal_material_f64,
    )
    return _as_float_array(binding(alpha_r, alpha_z, alpha_t, reference_temperature), resolved_dtype)


def orthotropic_plane_strain_thermal_material(
    alpha_x: float,
    alpha_y: float,
    alpha_z: float,
    reference_temperature: float = 0.0,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating[Any]]:
    """Construct orthotropic plane-strain thermal-expansion data."""

    return orthotropic_axisymmetric_thermal_material(
        alpha_x,
        alpha_y,
        alpha_z,
        reference_temperature,
        dtype,
    )


def cfsem_radial_material(
    youngs_modulus: float,
    poisson_ratio: float,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating[Any]]:
    """Construct the reduced elastic matrix used by the 1D radial solver.

    Args:
        youngs_modulus: Young's modulus with units `[pressure]`.
        poisson_ratio: Poisson ratio with units `[dimensionless]`.
        dtype: Output floating dtype.

    Returns:
        NDArray: Elastic stress-strain matrix with shape `(4, 4)` in component order
        `[rr, zz, tt, rz]`. Units are `[stress / strain] = [pressure]`.
    """

    resolved_dtype = np.dtype(dtype)
    binding = _dispatch_pair(
        resolved_dtype,
        _cfsem_radial_material_f32,
        _cfsem_radial_material_f64,
    )
    return _as_float_array(binding(youngs_modulus, poisson_ratio), resolved_dtype).reshape(4, 4)


def _normalize_displacements(
    displacements: ArrayLike,
    nnode: int,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    arr = np.asarray(displacements, dtype=dtype)
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
    "QuadratureFieldSamples",
    "assemble_structural_2d",
    "cfsem_radial_material",
    "infer_quad9_mesh",
    "isotropic_axisymmetric_material",
    "isotropic_axisymmetric_thermal_material",
    "isotropic_plane_strain_material",
    "isotropic_plane_strain_thermal_material",
    "orthotropic_axisymmetric_thermal_material",
    "orthotropic_plane_strain_thermal_material",
    "pack_material_tables_from_tags",
]
