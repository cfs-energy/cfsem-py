"""
2D-axisymmetric elasticity finite-element assembly for solenoid stress problems.

This module provides a small displacement-based axisymmetric finite-element solver
for the `(r, z)` meridian plane. The primary backend abstraction is an assembled,
reusable model that stores sparse load operators, sparse quadrature-recovery
operators, and the reduced stiffness matrix. The Rust backend also caches the LU
factorization and provides shared material, mesh-elevation, and quadrature-recovery
conveniences. Python wraps that model, normalizes NumPy-facing inputs, and reshapes
the returned arrays.

The element formulation follows the standard small-strain Galerkin construction

`K_e = integral(B^T D B 2*pi*r dA)`

with consistent body-force, surface-pressure, and surface-traction load vectors. The axisymmetric
engineering-strain vector is ordered as `[e_rr, e_zz, e_tt, g_rz]`.

References:
    [1] Thomas J. R. Hughes,
        *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*,
        Prentice-Hall, 1987.

    [2] Klaus-Juergen Bathe,
        *Finite Element Procedures*,
        Prentice Hall, 1996.

    [3] J. N. Reddy,
        *An Introduction to the Finite Element Method*, 3rd ed.,
        McGraw-Hill, 2005.

    [4] Stephen P. Timoshenko and J. N. Goodier,
        *Theory of Elasticity*, 3rd ed.,
        McGraw-Hill, 1970.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

import cfsem.cfsem as _cfsem_bindings

_assemble_model_axisymmetric_f32 = _cfsem_bindings.solenoid_stress_fem_assemble_model_axisymmetric_f32
_assemble_model_axisymmetric_f64 = _cfsem_bindings.solenoid_stress_fem_assemble_model_axisymmetric_f64
_cfsem_radial_material_f32 = _cfsem_bindings.solenoid_stress_fem_cfsem_radial_material_f32
_cfsem_radial_material_f64 = _cfsem_bindings.solenoid_stress_fem_cfsem_radial_material_f64
_infer_quad9_mesh_f32 = _cfsem_bindings.solenoid_stress_fem_infer_quad9_mesh_f32
_infer_quad9_mesh_f64 = _cfsem_bindings.solenoid_stress_fem_infer_quad9_mesh_f64
_isotropic_axisymmetric_material_f32 = _cfsem_bindings.solenoid_stress_fem_isotropic_axisymmetric_material_f32
_isotropic_axisymmetric_material_f64 = _cfsem_bindings.solenoid_stress_fem_isotropic_axisymmetric_material_f64
_isotropic_axisymmetric_thermal_material_f32 = (
    _cfsem_bindings.solenoid_stress_fem_isotropic_axisymmetric_thermal_material_f32
)
_isotropic_axisymmetric_thermal_material_f64 = (
    _cfsem_bindings.solenoid_stress_fem_isotropic_axisymmetric_thermal_material_f64
)
_orthotropic_axisymmetric_thermal_material_f32 = (
    _cfsem_bindings.solenoid_stress_fem_orthotropic_axisymmetric_thermal_material_f32
)
_orthotropic_axisymmetric_thermal_material_f64 = (
    _cfsem_bindings.solenoid_stress_fem_orthotropic_axisymmetric_thermal_material_f64
)

ArrayLike = npt.ArrayLike


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
    """Per-element meridian area and swept volume.

    `areas` and `swept_volumes` both have shape `(nelem,)`.
    """

    areas: npt.NDArray[np.floating[Any]]
    swept_volumes: npt.NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class ElementQuadrature:
    """Per-element physical quadrature points and mapped weights.

    `points_rz` has shape `(nelem, nq_per_element, 2)`.
    `weights_area` and `weights_volume` have shape `(nelem, nq_per_element)`.
    """

    points_rz: npt.NDArray[np.floating[Any]]
    weights_area: npt.NDArray[np.floating[Any]]
    weights_volume: npt.NDArray[np.floating[Any]]
    nq_per_element: int


@dataclass(frozen=True, slots=True)
class ElevatedQuad9Mesh:
    """Explicit 9-node analysis mesh inferred from a corner-only quad4 mesh."""

    input_nodes: npt.NDArray[np.floating[Any]]
    input_elements: npt.NDArray[np.uint64]
    analysis_nodes: npt.NDArray[np.floating[Any]]
    analysis_elements: npt.NDArray[np.uint64]
    corner_node_indices: npt.NDArray[np.int64]
    midside_node_indices: npt.NDArray[np.int64]
    center_node_indices: npt.NDArray[np.int64]


class AxisymmetricFEMModel:
    """Reusable axisymmetric FEM model with sparse operators and reduced solve state.

    The stored sparse operators are the primary reusable objects:
    - `body_force_to_rhs`, `pressure_to_rhs`, `traction_to_rhs`, and `temperature_to_rhs`
      map load amplitudes to the reduced structural right-hand side,
    - `strain_operator`, `stress_operator`, `thermal_strain_operator`, and
      `thermal_stress_operator` map solved displacements or nodal temperatures to quadrature-point
      fields.

    `build_rhs(...)`, `solve(...)`, `element_quadrature()`, `element_measures()`, and
    `evaluate_quadrature(...)` are convenience methods layered on top of those stored operators
    and the reduced stiffness matrix.

    `input_nodes` and `input_elements` expose the original corner-node mesh.
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
        element_type: str,
        stiffness: sp.csc_matrix,
        body_force_to_rhs: sp.csr_matrix,
        pressure_to_rhs: sp.csr_matrix,
        traction_to_rhs: sp.csr_matrix,
        temperature_to_rhs: sp.csr_matrix,
        constant_rhs: npt.NDArray[np.floating[Any]],
        quadrature_points_rz: npt.NDArray[np.floating[Any]],
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
        self.quadrature_points_rz = quadrature_points_rz
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
        self.element_type = element_type
        self.ndof_full = int(ndof_full)
        self.ndof_reduced = int(ndof_reduced)
        self.nelem = int(nelem)
        self.nq_per_element = int(nq_per_element)
        self.n_temperature_nodes = int(n_temperature_nodes)
        self._element_quadrature_cache: ElementQuadrature | None = None
        self._element_measures_cache: ElementMeasures | None = None

    @property
    def dtype(self) -> np.dtype[Any]:
        """Floating dtype used by the assembled operators and convenience methods."""

        return self._dtype

    @property
    def ndof(self) -> int:
        """Compatibility alias for `ndof_full`."""

        return self.ndof_full

    @property
    def input_nodes(self) -> npt.NDArray[np.floating[Any]]:
        """Corner-node input mesh coordinates with shape `(nnode, 2)`."""

        return self._input_nodes

    @property
    def input_elements(self) -> npt.NDArray[np.uint64]:
        """Input mesh connectivity with shape `(nelem, 4)`."""

        return self._input_elements

    def element_quadrature(self) -> ElementQuadrature:
        """Return physical quadrature points and mapped area/volume weights per element."""

        cache = self._element_quadrature_cache
        if cache is not None:
            return cache
        points_flat, weights_area_flat, weights_volume_flat, nq = self._backend.element_quadrature()
        nelem = self.analysis_elements.shape[0]
        points_rz = np.asarray(points_flat, dtype=self.dtype).reshape(nelem, int(nq), 2)
        weights_area = np.asarray(weights_area_flat, dtype=self.dtype).reshape(nelem, int(nq))
        weights_volume = np.asarray(weights_volume_flat, dtype=self.dtype).reshape(nelem, int(nq))
        cache = ElementQuadrature(
            points_rz=points_rz,
            weights_area=weights_area,
            weights_volume=weights_volume,
            nq_per_element=int(nq),
        )
        self._element_quadrature_cache = cache
        return cache

    def element_measures(self) -> ElementMeasures:
        """Return per-element meridian area and swept axisymmetric volume."""

        cache = self._element_measures_cache
        if cache is not None:
            return cache
        areas, swept_volumes = self._backend.element_measures()
        cache = ElementMeasures(
            areas=np.asarray(areas, dtype=self.dtype),
            swept_volumes=np.asarray(swept_volumes, dtype=self.dtype),
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
        """Build one reduced right-hand side from the stored sparse load operators.

        The returned vector has shape `(ndof_reduced,)`.
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
        """Solve the reduced system for one right-hand side and recover the full displacement."""

        rhs_arr = np.asarray(rhs, dtype=self.dtype).reshape(-1)
        assert (
            rhs_arr.shape[0] == self.ndof_reduced
        ), f"rhs must have length {self.ndof_reduced}; got {rhs_arr.shape}"
        return np.asarray(self._backend.solve(rhs_arr), dtype=self.dtype)

    def recover_full(self, reduced_solution: ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """Reinsert prescribed Dirichlet values into a reduced displacement vector."""

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
        """Evaluate quadrature-point strain and stress fields from displacements.

        `displacements` may be either the reduced solution with shape `(ndof_reduced,)` or the
        full analysis displacement field with shape `(2 * n_analysis_nodes,)` or
        `(n_analysis_nodes, 2)`.
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
            points_rz=np.asarray(points_flat, dtype=self.dtype).reshape(nelem, nq, 2),
            strain=np.asarray(strain_flat, dtype=self.dtype).reshape(nelem, nq, 4),
            thermal_strain=np.asarray(thermal_strain_flat, dtype=self.dtype).reshape(nelem, nq, 4),
            elastic_strain=np.asarray(elastic_strain_flat, dtype=self.dtype).reshape(nelem, nq, 4),
            stress=np.asarray(stress_flat, dtype=self.dtype).reshape(nelem, nq, 4),
        )


@dataclass(frozen=True, slots=True)
class QuadratureFieldSamples:
    """Recovered strain and stress at element quadrature points.

    Each field has shape `(nelem, nq_per_element, 4)` with component ordering
    `[rr, zz, tt, rz]`. `points_rz` has shape `(nelem, nq_per_element, 2)`.
    """

    points_rz: npt.NDArray[np.floating[Any]]
    strain: npt.NDArray[np.floating[Any]]
    thermal_strain: npt.NDArray[np.floating[Any]]
    elastic_strain: npt.NDArray[np.floating[Any]]
    stress: npt.NDArray[np.floating[Any]]

    @property
    def total_strain(self) -> npt.NDArray[np.floating[Any]]:
        """Alias for `strain`."""

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
    """Elevate a corner-only quad mesh to an explicit 9-node Lagrange mesh."""

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
    edge_nodes = ((0, 1), (1, 2), (2, 3), (3, 0))
    for element_index, conn in enumerate(elevated.input_elements):
        for local_edge, (local_a, local_b) in enumerate(edge_nodes):
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
    material_table: ArrayLike | Mapping[int, ArrayLike],
    dtype: np.dtype[Any],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.floating[Any]]]:
    ids = np.asarray(material_ids, dtype=np.uint64)
    assert ids.ndim == 1, f"material_ids must have shape (nelem,); got {ids.shape}"
    if isinstance(material_table, Mapping):
        assert material_table, "material_table mapping cannot be empty"
        keys = sorted(int(key) for key in material_table)
        dense_table = []
        tag_to_index = {key: index for index, key in enumerate(keys)}
        for key in keys:
            matrix = np.asarray(material_table[key], dtype=dtype)
            assert matrix.shape == (4, 4), f"material_table[{key}] must have shape (4, 4); got {matrix.shape}"
            dense_table.append(matrix)
        try:
            normalized_ids = np.asarray([tag_to_index[int(tag)] for tag in ids], dtype=np.uint64)
        except KeyError as exc:
            raise ValueError(
                f"material_ids contains tag {exc.args[0]} that is missing from material_table"
            ) from exc
        return normalized_ids, np.ascontiguousarray(np.stack(dense_table, axis=0), dtype=dtype)

    table = np.asarray(material_table, dtype=dtype)
    assert table.ndim == 3 and table.shape[1:] == (
        4,
        4,
    ), f"material_table must have shape (nmat, 4, 4); got {table.shape}"
    return np.ascontiguousarray(ids), np.ascontiguousarray(table)


def _normalize_thermal_material_table(
    material_ids: ArrayLike,
    thermal_material_table: ArrayLike | Mapping[int, ArrayLike] | None,
    dtype: np.dtype[Any],
    *,
    require_mapping: bool | None = None,
) -> tuple[npt.NDArray[np.uint64] | None, npt.NDArray[np.floating[Any]] | None]:
    if thermal_material_table is None:
        return None, None
    ids = np.asarray(material_ids, dtype=np.uint64)
    assert ids.ndim == 1, f"material_ids must have shape (nelem,); got {ids.shape}"
    is_mapping = isinstance(thermal_material_table, Mapping)
    assert (
        require_mapping is None or is_mapping == require_mapping
    ), "thermal_material_table must use the same mapping/dense convention as material_table"
    if is_mapping:
        mapping = thermal_material_table
        assert mapping, "thermal_material_table mapping cannot be empty"
        keys = sorted(int(key) for key in mapping)
        dense_table = []
        tag_to_index = {key: index for index, key in enumerate(keys)}
        for key in keys:
            row = np.asarray(mapping[key], dtype=dtype)
            assert row.shape == (5,), f"thermal_material_table[{key}] must have shape (5,); got {row.shape}"
            dense_table.append(row)
        try:
            normalized_ids = np.asarray([tag_to_index[int(tag)] for tag in ids], dtype=np.uint64)
        except KeyError as exc:
            raise ValueError(
                f"material_ids contains tag {exc.args[0]} that is missing from thermal_material_table"
            ) from exc
        table = np.ascontiguousarray(np.stack(dense_table, axis=0), dtype=dtype)
    else:
        table = np.asarray(thermal_material_table, dtype=dtype)
        assert (
            table.ndim == 2 and table.shape[1] == 5
        ), f"thermal_material_table must have shape (nmat, 5); got {table.shape}"
        normalized_ids = np.ascontiguousarray(ids)
        table = np.ascontiguousarray(table)
    assert np.allclose(
        table[:, 3], 0.0
    ), "thermal_material_table shear thermal expansion must be zero in phase 1"
    return normalized_ids, table


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


def assemble_axisymmetric(
    nodes: ArrayLike,
    elements: ArrayLike,
    material_ids: ArrayLike,
    material_table: ArrayLike | Mapping[int, ArrayLike],
    pressure_faces: ArrayLike | None = None,
    traction_faces: ArrayLike | None = None,
    thermal_material_table: ArrayLike | Mapping[int, ArrayLike] | None = None,
    prescribed: Mapping[int, float] | None = None,
    quadrature: str | int = "gl3",
    element_type: str = "quad4",
) -> AxisymmetricFEMModel:
    """Assemble the reusable axisymmetric FEM model.

    The returned model stores the reduced stiffness matrix, the sparse load operators, the sparse
    quadrature-recovery operators, and the fixed load topology associated with `pressure_faces`,
    `traction_faces`, and `thermal_material_table`.

    `element_type="quad4"` uses the input mesh directly. `element_type="quad9"` elevates the
    corner-only input mesh to an explicit 9-node analysis mesh for the backend while keeping the
    Python-side load and temperature inputs on the original corner nodes.
    """

    elements_arr = _normalize_elements(elements)
    dtype = _resolve_float_dtype(nodes, material_table, thermal_material_table)
    nodes_arr = _normalize_nodes(nodes, dtype)
    material_ids_arr, material_table_arr = _normalize_materials(material_ids, material_table, dtype)
    _thermal_ids_arr, thermal_material_table_arr = _normalize_thermal_material_table(
        material_ids,
        thermal_material_table,
        dtype,
        require_mapping=isinstance(material_table, Mapping) if thermal_material_table is not None else None,
    )
    pressure_faces_arr = _normalize_face_pairs("pressure_faces", pressure_faces)
    traction_faces_arr = _normalize_face_pairs("traction_faces", traction_faces)
    prescribed_dofs, prescribed_values = _normalize_prescribed_dirichlet(prescribed, dtype)
    quadrature_code = _quadrature_code(quadrature)
    normalized_element_type = _normalize_element_type(element_type)
    if normalized_element_type == "quad4":
        analysis_nodes, analysis_elements, elevated = nodes_arr, elements_arr, None
    else:
        elevated = infer_quad9_mesh(nodes_arr, elements_arr)
        analysis_nodes, analysis_elements = elevated.analysis_nodes, elevated.analysis_elements
    low_level = _dispatch_pair(
        dtype,
        _assemble_model_axisymmetric_f32,
        _assemble_model_axisymmetric_f64,
    )
    backend = low_level(
        analysis_nodes,
        analysis_elements,
        material_ids_arr,
        material_table_arr,
        pressure_faces_arr,
        traction_faces_arr,
        np.zeros((0, 5), dtype=dtype) if thermal_material_table_arr is None else thermal_material_table_arr,
        prescribed_dofs,
        prescribed_values,
        4 if normalized_element_type == "quad4" else 9,
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

    return AxisymmetricFEMModel(
        backend=backend,
        dtype=dtype,
        input_nodes=nodes_arr,
        input_elements=elements_arr,
        analysis_nodes=analysis_nodes,
        analysis_elements=analysis_elements,
        elevated=elevated,
        pressure_faces=pressure_faces_arr,
        traction_faces=traction_faces_arr,
        element_type=normalized_element_type,
        stiffness=stiffness,
        body_force_to_rhs=body_force_to_rhs,
        pressure_to_rhs=pressure_to_rhs,
        traction_to_rhs=traction_to_rhs,
        temperature_to_rhs=_to_csr_matrix(temperature_to_rhs),
        constant_rhs=np.asarray(backend.constant_rhs(), dtype=dtype),
        quadrature_points_rz=np.asarray(
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
    """Construct the full 3D isotropic axisymmetric constitutive matrix."""

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
    """Construct isotropic thermal-expansion data `[alpha_r, alpha_z, alpha_t, 0, T_ref]`."""

    resolved_dtype = np.dtype(dtype)
    binding = _dispatch_pair(
        resolved_dtype,
        _isotropic_axisymmetric_thermal_material_f32,
        _isotropic_axisymmetric_thermal_material_f64,
    )
    return _as_float_array(binding(alpha, reference_temperature), resolved_dtype)


def orthotropic_axisymmetric_thermal_material(
    alpha_r: float,
    alpha_z: float,
    alpha_t: float,
    reference_temperature: float = 0.0,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating[Any]]:
    """Construct orthotropic thermal-expansion data `[alpha_r, alpha_z, alpha_t, 0, T_ref]`."""

    resolved_dtype = np.dtype(dtype)
    binding = _dispatch_pair(
        resolved_dtype,
        _orthotropic_axisymmetric_thermal_material_f32,
        _orthotropic_axisymmetric_thermal_material_f64,
    )
    return _as_float_array(binding(alpha_r, alpha_z, alpha_t, reference_temperature), resolved_dtype)


def cfsem_radial_material(
    youngs_modulus: float,
    poisson_ratio: float,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating[Any]]:
    """Construct the reduced isotropic constitutive matrix matching `SolenoidStress1D`."""

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
    "AxisymmetricFEMModel",
    "ElementMeasures",
    "ElementQuadrature",
    "ElevatedQuad9Mesh",
    "QuadratureFieldSamples",
    "assemble_axisymmetric",
    "cfsem_radial_material",
    "infer_quad9_mesh",
    "isotropic_axisymmetric_material",
    "isotropic_axisymmetric_thermal_material",
    "orthotropic_axisymmetric_thermal_material",
]
