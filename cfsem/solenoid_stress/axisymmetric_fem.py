"""
2D-axisymmetric elasticity finite-element assembly for solenoid stress problems.

This module provides a small displacement-based axisymmetric finite-element solver
for the `(r, z)` meridian plane. The Rust backend assembles the reduced constrained
model, stores the sparse operators, caches the LU factorization, and runs the solve.
Python wraps that model, normalizes load data, and handles postprocessing workflows.

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

_assemble_model_axisymmetric_quad4_f32 = (
    _cfsem_bindings.solenoid_stress_fem_assemble_model_axisymmetric_quad4_f32
)
_assemble_model_axisymmetric_quad4_f64 = (
    _cfsem_bindings.solenoid_stress_fem_assemble_model_axisymmetric_quad4_f64
)
_assemble_model_axisymmetric_quad9_f32 = (
    _cfsem_bindings.solenoid_stress_fem_assemble_model_axisymmetric_quad9_f32
)
_assemble_model_axisymmetric_quad9_f64 = (
    _cfsem_bindings.solenoid_stress_fem_assemble_model_axisymmetric_quad9_f64
)

ArrayLike = npt.ArrayLike
ElementType = str


def _to_csr_matrix(matrix: Any) -> sp.csr_matrix:
    """Normalize sparse results to the matrix API expected by this module."""

    return cast(sp.csr_matrix, sp.csr_matrix(matrix))


def _to_csc_matrix(matrix: Any) -> sp.csc_matrix:
    """Normalize sparse results to CSC matrices."""

    return cast(sp.csc_matrix, sp.csc_matrix(matrix))


def _sparse_shape(matrix: Any) -> tuple[int, int]:
    """Return a concrete 2D sparse shape for pyright and runtime callers."""

    return cast(tuple[int, int], matrix.shape)


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
    """Per-element meridian area and swept volume."""

    areas: npt.NDArray[np.floating[Any]]
    swept_volumes: npt.NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class ElementQuadrature:
    """Per-element physical quadrature points and mapped weights."""

    points_rz: npt.NDArray[np.floating[Any]]
    weights_area: npt.NDArray[np.floating[Any]]
    weights_volume: npt.NDArray[np.floating[Any]]
    nq_per_element: int


@dataclass(frozen=True, slots=True)
class ElevatedQuad9Mesh:
    """Explicit 9-node analysis mesh inferred from a corner-only quad mesh."""

    input_nodes: npt.NDArray[np.floating[Any]]
    input_elements: npt.NDArray[np.uint64]
    analysis_nodes: npt.NDArray[np.floating[Any]]
    analysis_elements: npt.NDArray[np.uint64]
    corner_node_indices: npt.NDArray[np.int64]
    midside_node_indices: npt.NDArray[np.int64]
    center_node_indices: npt.NDArray[np.int64]


class AxisymmetricFEMModel:
    """Reduced axisymmetric FEM model assembled once and reused for load cases."""

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
        quadrature_code: int,
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
        self._quadrature_code = int(quadrature_code)
        self._element_quadrature_cache: ElementQuadrature | None = None
        self._element_measures_cache: ElementMeasures | None = None
        self.nodes = input_nodes
        self.elements = input_elements

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def ndof(self) -> int:
        return self.ndof_full

    @property
    def thermal_reference_rhs(self) -> npt.NDArray[np.floating[Any]]:
        return self.constant_rhs

    def element_quadrature(self) -> ElementQuadrature:
        """Return physical quadrature points and mapped area/volume weights per element."""

        cache = self._element_quadrature_cache
        if cache is not None:
            return cache
        nelem = self.analysis_elements.shape[0]
        nq = self.nq_per_element
        points_rz = np.zeros((nelem, nq, 2), dtype=self.dtype)
        weights_area = np.zeros((nelem, nq), dtype=self.dtype)
        weights_volume = np.zeros((nelem, nq), dtype=self.dtype)
        for element_index, conn in enumerate(self.analysis_elements):
            coords = self.analysis_nodes[conn]
            for sample_index, (_n, _grad_phys, det_j, point, weight) in enumerate(
                _volume_samples(coords, self.element_type, self._quadrature_code, self.dtype)
            ):
                points_rz[element_index, sample_index] = point
                weights_area[element_index, sample_index] = det_j * weight
                weights_volume[element_index, sample_index] = det_j * weight * (2.0 * np.pi * point[0])
        cache = ElementQuadrature(
            points_rz=points_rz,
            weights_area=weights_area,
            weights_volume=weights_volume,
            nq_per_element=nq,
        )
        self._element_quadrature_cache = cache
        return cache

    def element_measures(self) -> ElementMeasures:
        """Return per-element meridian area and swept axisymmetric volume."""

        cache = self._element_measures_cache
        if cache is None:
            quadrature = self.element_quadrature()
            cache = ElementMeasures(
                areas=np.sum(quadrature.weights_area, axis=1),
                swept_volumes=np.sum(quadrature.weights_volume, axis=1),
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
            self.element_type,
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
        body_force_arr = _normalize_body_force_or_zero(body_force, self.nelem, self.dtype)
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
        rhs_arr = np.asarray(rhs, dtype=self.dtype).reshape(-1)
        assert (
            rhs_arr.shape[0] == self.ndof_reduced
        ), f"rhs must have length {self.ndof_reduced}; got {rhs_arr.shape}"
        return np.asarray(self._backend.solve(rhs_arr), dtype=self.dtype)

    def recover_full(self, reduced_solution: ArrayLike) -> npt.NDArray[np.floating[Any]]:
        reduced_arr = np.asarray(reduced_solution, dtype=self.dtype).reshape(-1)
        assert (
            reduced_arr.shape[0] == self.ndof_reduced
        ), f"reduced_solution must have length {self.ndof_reduced}; got {reduced_arr.shape}"
        full = np.zeros((self.ndof_full,), dtype=self.dtype)
        full[self.fixed_dofs] = self.fixed_values
        full[self.free_dofs] = reduced_arr
        return full

    def _normalize_reduced_solution(
        self,
        displacements: ArrayLike,
    ) -> npt.NDArray[np.floating[Any]]:
        arr = np.asarray(displacements, dtype=self.dtype)
        if arr.ndim == 1 and arr.shape == (self.ndof_reduced,):
            return np.ascontiguousarray(arr)
        full = _normalize_displacements(displacements, self.analysis_nodes.shape[0], self.dtype).reshape(-1)
        return np.ascontiguousarray(full[self.free_dofs], dtype=self.dtype)

    def evaluate_quadrature(
        self,
        displacements: ArrayLike,
        nodal_temperature: ArrayLike | None = None,
    ) -> QuadratureFieldSamples:
        reduced = self._normalize_reduced_solution(displacements)
        if self.n_temperature_nodes == 0:
            temperature_arr = np.zeros((0,), dtype=self.dtype)
        else:
            if nodal_temperature is None:
                raise ValueError(
                    "nodal_temperature is required because this model includes thermal materials"
                )
            temperature_arr = _normalize_nodal_temperature(
                nodal_temperature,
                self.n_temperature_nodes,
                self.dtype,
            )
        nelem = self.analysis_elements.shape[0]
        nq = self.nq_per_element
        strain = (
            np.asarray(self.strain_operator @ reduced, dtype=self.dtype) + self.strain_constant
        ).reshape(nelem, nq, 4)
        thermal_strain = (
            np.asarray(self.thermal_strain_operator @ temperature_arr, dtype=self.dtype)
            + self.thermal_strain_constant
        ).reshape(nelem, nq, 4)
        stress = (
            np.asarray(self.stress_operator @ reduced, dtype=self.dtype)
            + self.stress_constant
            - (
                np.asarray(self.thermal_stress_operator @ temperature_arr, dtype=self.dtype)
                + self.thermal_stress_constant
            )
        ).reshape(nelem, nq, 4)
        return QuadratureFieldSamples(
            points_rz=self.quadrature_points_rz,
            strain=strain,
            thermal_strain=thermal_strain,
            elastic_strain=strain - thermal_strain,
            stress=stress,
        )


@dataclass(frozen=True, slots=True)
class QuadratureFieldSamples:
    """Recovered strain and stress at element quadrature points."""

    points_rz: npt.NDArray[np.floating[Any]]
    strain: npt.NDArray[np.floating[Any]]
    thermal_strain: npt.NDArray[np.floating[Any]]
    elastic_strain: npt.NDArray[np.floating[Any]]
    stress: npt.NDArray[np.floating[Any]]

    @property
    def total_strain(self) -> npt.NDArray[np.floating[Any]]:
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


def _validate_element_quadrature_combo(element_type: str, quadrature_code: int) -> None:
    assert quadrature_code in {3, 4}, f"unsupported quadrature code {quadrature_code}; use 3 or 4"
    _normalize_element_type(element_type)


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
    node_list = [nodes_arr[i].copy() for i in range(nodes_arr.shape[0])]
    analysis_elements = np.zeros((elements_arr.shape[0], 9), dtype=np.uint64)
    analysis_elements[:, :4] = elements_arr

    edge_to_midpoint: dict[tuple[int, int], int] = {}
    midside_indices: list[int] = []
    center_indices = np.zeros((elements_arr.shape[0],), dtype=np.int64)
    next_node_index = nodes_arr.shape[0]

    midside_parametric_points = ((0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))
    edge_nodes = ((0, 1), (1, 2), (2, 3), (3, 0))

    for element_index, conn in enumerate(elements_arr):
        coords = nodes_arr[conn]
        for local_edge, ((local_a, local_b), (xi, eta)) in enumerate(
            zip(edge_nodes, midside_parametric_points, strict=True)
        ):
            node_a = int(conn[local_a])
            node_b = int(conn[local_b])
            edge_key = (node_a, node_b) if node_a < node_b else (node_b, node_a)
            midpoint_index = edge_to_midpoint.get(edge_key)
            if midpoint_index is None:
                midpoint = (_quad4_shape(xi, eta).astype(dtype, copy=False) @ coords).astype(
                    dtype,
                    copy=False,
                )
                midpoint_index = next_node_index
                next_node_index += 1
                edge_to_midpoint[edge_key] = midpoint_index
                midside_indices.append(midpoint_index)
                node_list.append(np.asarray(midpoint, dtype=dtype))
            analysis_elements[element_index, 4 + local_edge] = midpoint_index

        center = (_quad4_shape(0.0, 0.0).astype(dtype, copy=False) @ coords).astype(dtype, copy=False)
        center_index = next_node_index
        next_node_index += 1
        node_list.append(np.asarray(center, dtype=dtype))
        center_indices[element_index] = center_index
        analysis_elements[element_index, 8] = center_index

    return ElevatedQuad9Mesh(
        input_nodes=nodes_arr,
        input_elements=elements_arr,
        analysis_nodes=np.asarray(node_list, dtype=dtype),
        analysis_elements=analysis_elements,
        corner_node_indices=np.arange(nodes_arr.shape[0], dtype=np.int64),
        midside_node_indices=np.asarray(midside_indices, dtype=np.int64),
        center_node_indices=center_indices,
    )


def _analysis_mesh_for_element_type(
    nodes: npt.NDArray[np.floating[Any]],
    elements: npt.NDArray[np.uint64],
    element_type: str,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.uint64], ElevatedQuad9Mesh | None]:
    normalized_type = _normalize_element_type(element_type)
    if normalized_type == "quad4":
        return nodes, elements, None
    elevated = infer_quad9_mesh(nodes, elements)
    return elevated.analysis_nodes, elevated.analysis_elements, elevated


def _temperature_elevation_operator(
    elevated: ElevatedQuad9Mesh | None,
    dtype: np.dtype[Any],
) -> sp.csr_matrix:
    if elevated is None:
        nnode = 0
        return sp.csr_matrix((nnode, nnode), dtype=dtype)
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
    element_type: str,
    elevated: ElevatedQuad9Mesh | None,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    del element_type
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


def _empty_thermal_material_table(dtype: np.dtype[Any]) -> npt.NDArray[np.floating[Any]]:
    return np.zeros((0, 5), dtype=dtype)


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


def _normalize_body_force_or_zero(
    body_force: ArrayLike | None,
    nelem: int,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    if body_force is None:
        return np.zeros((nelem, 2), dtype=dtype)
    return _normalize_body_force(body_force, nelem, dtype)


def _normalize_pressure_faces(pressure_faces: ArrayLike | None) -> npt.NDArray[np.uint64]:
    if pressure_faces is None:
        return np.zeros((0, 2), dtype=np.uint64)
    faces = np.asarray(pressure_faces, dtype=np.uint64)
    assert (
        faces.ndim == 2 and faces.shape[1] == 2
    ), f"pressure_faces must have shape (nload, 2); got {faces.shape}"
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


def _normalize_traction_faces(traction_faces: ArrayLike | None) -> npt.NDArray[np.uint64]:
    if traction_faces is None:
        return np.zeros((0, 2), dtype=np.uint64)
    faces = np.asarray(traction_faces, dtype=np.uint64)
    assert (
        faces.ndim == 2 and faces.shape[1] == 2
    ), f"traction_faces must have shape (nload, 2); got {faces.shape}"
    return np.ascontiguousarray(faces)


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


def _dispatch_by_element_type(
    element_type: str,
    quad4_impl: Any,
    quad9_impl: Any,
) -> Any:
    normalized_type = _normalize_element_type(element_type)
    if normalized_type == "quad4":
        return quad4_impl
    return quad9_impl


def _element_jacobian(
    coords: npt.NDArray[np.floating[Any]],
    grad_ref: npt.NDArray[np.floating[Any]],
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    return np.array(
        [
            [np.dot(coords[:, 0], grad_ref[:, 0]), np.dot(coords[:, 0], grad_ref[:, 1])],
            [np.dot(coords[:, 1], grad_ref[:, 0]), np.dot(coords[:, 1], grad_ref[:, 1])],
        ],
        dtype=dtype,
    )


def _quad_face_reference(local_face: int, s: float) -> tuple[float, float, tuple[float, float]]:
    if local_face == 0:
        return s, -1.0, (1.0, 0.0)
    if local_face == 1:
        return 1.0, s, (0.0, 1.0)
    if local_face == 2:
        return -s, 1.0, (-1.0, 0.0)
    if local_face == 3:
        return -1.0, -s, (0.0, -1.0)
    raise ValueError(f"invalid local face {local_face}; expected 0, 1, 2, or 3")


def _q2_lagrange_1d(x: float) -> npt.NDArray[np.float64]:
    return np.array([0.5 * x * (x - 1.0), 1.0 - x * x, 0.5 * x * (x + 1.0)], dtype=np.float64)


def _q2_lagrange_grad_1d(x: float) -> npt.NDArray[np.float64]:
    return np.array([x - 0.5, -2.0 * x, x + 0.5], dtype=np.float64)


def _quad9_shape(xi: float, eta: float) -> npt.NDArray[np.float64]:
    lx = _q2_lagrange_1d(xi)
    ly = _q2_lagrange_1d(eta)
    return np.array(
        [
            lx[0] * ly[0],
            lx[2] * ly[0],
            lx[2] * ly[2],
            lx[0] * ly[2],
            lx[1] * ly[0],
            lx[2] * ly[1],
            lx[1] * ly[2],
            lx[0] * ly[1],
            lx[1] * ly[1],
        ],
        dtype=np.float64,
    )


def _quad9_grad_ref(xi: float, eta: float) -> npt.NDArray[np.float64]:
    lx = _q2_lagrange_1d(xi)
    ly = _q2_lagrange_1d(eta)
    dlx = _q2_lagrange_grad_1d(xi)
    dly = _q2_lagrange_grad_1d(eta)
    return np.array(
        [
            [dlx[0] * ly[0], lx[0] * dly[0]],
            [dlx[2] * ly[0], lx[2] * dly[0]],
            [dlx[2] * ly[2], lx[2] * dly[2]],
            [dlx[0] * ly[2], lx[0] * dly[2]],
            [dlx[1] * ly[0], lx[1] * dly[0]],
            [dlx[2] * ly[1], lx[2] * dly[1]],
            [dlx[1] * ly[2], lx[1] * dly[2]],
            [dlx[0] * ly[1], lx[0] * dly[1]],
            [dlx[1] * ly[1], lx[1] * dly[1]],
        ],
        dtype=np.float64,
    )


def _element_shape(element_type: str, xi: float, eta: float) -> npt.NDArray[np.float64]:
    normalized_type = _normalize_element_type(element_type)
    if normalized_type == "quad4":
        return _quad4_shape(xi, eta)
    return _quad9_shape(xi, eta)


def _element_grad_ref(element_type: str, xi: float, eta: float) -> npt.NDArray[np.float64]:
    normalized_type = _normalize_element_type(element_type)
    if normalized_type == "quad4":
        return _quad4_grad_ref(xi, eta)
    return _quad9_grad_ref(xi, eta)


def _axisymmetric_b_matrix(
    n: npt.NDArray[np.floating[Any]],
    grad_phys: npt.NDArray[np.floating[Any]],
    radius: float,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    assert radius > np.finfo(dtype).eps, f"quadrature radius {radius} is too close to zero"
    nnodes = int(n.shape[0])
    b = np.zeros((4, 2 * nnodes), dtype=dtype)
    for i in range(nnodes):
        col_r = 2 * i
        col_z = col_r + 1
        b[0, col_r] = grad_phys[i, 0]
        b[1, col_z] = grad_phys[i, 1]
        b[2, col_r] = n[i] / dtype.type(radius)
        b[3, col_r] = grad_phys[i, 1]
        b[3, col_z] = grad_phys[i, 0]
    return b


def _volume_samples(
    coords: npt.NDArray[np.floating[Any]],
    element_type: str,
    quadrature_code: int,
    dtype: np.dtype[Any],
):
    for xi, wx in _gauss_1d(quadrature_code):
        for eta, wy in _gauss_1d(quadrature_code):
            n = _element_shape(element_type, xi, eta).astype(dtype, copy=False)
            grad_ref = _element_grad_ref(element_type, xi, eta).astype(dtype, copy=False)
            jac = _element_jacobian(coords, grad_ref, dtype)
            det_j = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
            assert det_j > 0.0, f"encountered non-positive element Jacobian determinant {float(det_j)!r}"
            point = n @ coords
            assert point[0] >= 0.0, f"quadrature point has negative radius {float(point[0])!r}"
            inv_j = np.linalg.inv(jac)
            grad_phys = np.column_stack(
                [
                    inv_j[0, 0] * grad_ref[:, 0] + inv_j[1, 0] * grad_ref[:, 1],
                    inv_j[0, 1] * grad_ref[:, 0] + inv_j[1, 1] * grad_ref[:, 1],
                ]
            )
            yield n, grad_phys, det_j, np.asarray(point, dtype=dtype), dtype.type(wx * wy)


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
    """Assemble a reduced axisymmetric FEM model with fixed load topology and Dirichlet data."""

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
    pressure_faces_arr = _normalize_pressure_faces(pressure_faces)
    traction_faces_arr = _normalize_traction_faces(traction_faces)
    prescribed_dofs, prescribed_values = _normalize_prescribed_dirichlet(prescribed, dtype)
    quadrature_code = _quadrature_code(quadrature)
    normalized_element_type = _normalize_element_type(element_type)
    _validate_element_quadrature_combo(normalized_element_type, quadrature_code)
    analysis_nodes, analysis_elements, elevated = _analysis_mesh_for_element_type(
        nodes_arr, elements_arr, normalized_element_type
    )
    low_level = _dispatch_pair(
        dtype,
        _dispatch_by_element_type(
            normalized_element_type,
            _assemble_model_axisymmetric_quad4_f32,
            _assemble_model_axisymmetric_quad9_f32,
        ),
        _dispatch_by_element_type(
            normalized_element_type,
            _assemble_model_axisymmetric_quad4_f64,
            _assemble_model_axisymmetric_quad9_f64,
        ),
    )
    backend = low_level(
        analysis_nodes,
        analysis_elements,
        material_ids_arr,
        material_table_arr,
        pressure_faces_arr,
        traction_faces_arr,
        _empty_thermal_material_table(dtype)
        if thermal_material_table_arr is None
        else thermal_material_table_arr,
        prescribed_dofs,
        prescribed_values,
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
        quadrature_code=quadrature_code,
    )


def isotropic_axisymmetric_material(
    youngs_modulus: float,
    poisson_ratio: float,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating[Any]]:
    """
    Construct the full 3D isotropic axisymmetric constitutive matrix.

    This is the conventional small-strain isotropic matrix specialized to the
    axisymmetric strain ordering `[rr, zz, tt, rz]`; see [1]-[3].
    """

    e = float(youngs_modulus)
    nu = float(poisson_ratio)
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = e / (2.0 * (1.0 + nu))
    return np.array(
        [
            [lam + 2.0 * mu, lam, lam, 0.0],
            [lam, lam + 2.0 * mu, lam, 0.0],
            [lam, lam, lam + 2.0 * mu, 0.0],
            [0.0, 0.0, 0.0, mu],
        ],
        dtype=dtype,
    )


def isotropic_axisymmetric_thermal_material(
    alpha: float,
    reference_temperature: float = 0.0,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating[Any]]:
    return np.array(
        [alpha, alpha, alpha, 0.0, reference_temperature],
        dtype=dtype,
    )


def orthotropic_axisymmetric_thermal_material(
    alpha_r: float,
    alpha_z: float,
    alpha_t: float,
    reference_temperature: float = 0.0,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating[Any]]:
    return np.array(
        [alpha_r, alpha_z, alpha_t, 0.0, reference_temperature],
        dtype=dtype,
    )


def cfsem_radial_material(
    youngs_modulus: float,
    poisson_ratio: float,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating[Any]]:
    """
    Construct the reduced isotropic constitutive matrix matching `SolenoidStress1D`.

    This matrix is chosen so that radial/hoop response in `u_r` agrees with the
    assumptions used by `cfsem.solenoid_stress.solenoid_1d.SolenoidStress1D`
    for the pressure-vessel validation cases.
    """

    e = float(youngs_modulus)
    nu = float(poisson_ratio)
    factor = e / (1.0 - nu**2)
    shear = e / (2.0 * (1.0 + nu))
    return np.array(
        [
            [factor, 0.0, factor * nu, 0.0],
            [0.0, e, 0.0, 0.0],
            [factor * nu, 0.0, factor, 0.0],
            [0.0, 0.0, 0.0, shear],
        ],
        dtype=dtype,
    )


def _gauss_1d(code: int) -> list[tuple[float, float]]:
    if code == 3:
        a = np.sqrt(3.0 / 5.0)
        return [(-a, 5.0 / 9.0), (0.0, 8.0 / 9.0), (a, 5.0 / 9.0)]
    if code == 4:
        a = np.sqrt((3.0 + 2.0 * np.sqrt(6.0 / 5.0)) / 7.0)
        b = np.sqrt((3.0 - 2.0 * np.sqrt(6.0 / 5.0)) / 7.0)
        w_a = (18.0 - np.sqrt(30.0)) / 36.0
        w_b = (18.0 + np.sqrt(30.0)) / 36.0
        return [(-a, w_a), (-b, w_b), (b, w_b), (a, w_a)]
    raise ValueError(f"unsupported quadrature code {code}")


def _quad4_shape(xi: float, eta: float) -> npt.NDArray[np.float64]:
    return 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )


def _quad4_grad_ref(xi: float, eta: float) -> npt.NDArray[np.float64]:
    return 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [1.0 - eta, -(1.0 + xi)],
            [1.0 + eta, 1.0 + xi],
            [-(1.0 + eta), 1.0 - xi],
        ]
    )


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
    "ElevatedQuad9Mesh",
    "QuadratureFieldSamples",
    "assemble_axisymmetric",
    "cfsem_radial_material",
    "infer_quad9_mesh",
    "isotropic_axisymmetric_material",
    "isotropic_axisymmetric_thermal_material",
    "orthotropic_axisymmetric_thermal_material",
]
