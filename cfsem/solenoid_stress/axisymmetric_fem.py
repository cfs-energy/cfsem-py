"""
2D-axisymmetric elasticity finite-element assembly for solenoid stress problems.

This module provides a small displacement-based axisymmetric finite-element solver
for the `(r, z)` meridian plane. The Rust backend assembles the global COO stiffness
matrix and sparse load operators, while Python handles load-operator application,
sparse linear algebra, postprocessing, and validation workflows.

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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import cfsem.cfsem as _cfsem_bindings
_assemble_axisymmetric_quad4_f32 = _cfsem_bindings.solenoid_stress_fem_assemble_axisymmetric_quad4_f32
_assemble_axisymmetric_quad4_f64 = _cfsem_bindings.solenoid_stress_fem_assemble_axisymmetric_quad4_f64
_assemble_axisymmetric_quad9_f32 = _cfsem_bindings.solenoid_stress_fem_assemble_axisymmetric_quad9_f32
_assemble_axisymmetric_quad9_f64 = _cfsem_bindings.solenoid_stress_fem_assemble_axisymmetric_quad9_f64
_element_measures_axisymmetric_quad4_f32 = (
    _cfsem_bindings.solenoid_stress_fem_element_measures_axisymmetric_quad4_f32
)
_element_measures_axisymmetric_quad4_f64 = (
    _cfsem_bindings.solenoid_stress_fem_element_measures_axisymmetric_quad4_f64
)
_element_measures_axisymmetric_quad9_f32 = (
    _cfsem_bindings.solenoid_stress_fem_element_measures_axisymmetric_quad9_f32
)
_element_measures_axisymmetric_quad9_f64 = (
    _cfsem_bindings.solenoid_stress_fem_element_measures_axisymmetric_quad9_f64
)
_element_quadrature_axisymmetric_quad4_f32 = (
    _cfsem_bindings.solenoid_stress_fem_element_quadrature_axisymmetric_quad4_f32
)
_element_quadrature_axisymmetric_quad4_f64 = (
    _cfsem_bindings.solenoid_stress_fem_element_quadrature_axisymmetric_quad4_f64
)
_element_quadrature_axisymmetric_quad9_f32 = (
    _cfsem_bindings.solenoid_stress_fem_element_quadrature_axisymmetric_quad9_f32
)
_element_quadrature_axisymmetric_quad9_f64 = (
    _cfsem_bindings.solenoid_stress_fem_element_quadrature_axisymmetric_quad9_f64
)
_quadrature_field_operators_axisymmetric_quad4_f32 = (
    _cfsem_bindings.solenoid_stress_fem_quadrature_field_operators_axisymmetric_quad4_f32
)
_quadrature_field_operators_axisymmetric_quad4_f64 = (
    _cfsem_bindings.solenoid_stress_fem_quadrature_field_operators_axisymmetric_quad4_f64
)
_quadrature_field_operators_axisymmetric_quad9_f32 = (
    _cfsem_bindings.solenoid_stress_fem_quadrature_field_operators_axisymmetric_quad9_f32
)
_quadrature_field_operators_axisymmetric_quad9_f64 = (
    _cfsem_bindings.solenoid_stress_fem_quadrature_field_operators_axisymmetric_quad9_f64
)
_body_force_operator_axisymmetric_quad4_f32 = (
    _cfsem_bindings.solenoid_stress_fem_body_force_operator_axisymmetric_quad4_f32
)
_body_force_operator_axisymmetric_quad4_f64 = (
    _cfsem_bindings.solenoid_stress_fem_body_force_operator_axisymmetric_quad4_f64
)
_body_force_operator_axisymmetric_quad9_f32 = (
    _cfsem_bindings.solenoid_stress_fem_body_force_operator_axisymmetric_quad9_f32
)
_body_force_operator_axisymmetric_quad9_f64 = (
    _cfsem_bindings.solenoid_stress_fem_body_force_operator_axisymmetric_quad9_f64
)
_pressure_operator_axisymmetric_quad4_f32 = (
    _cfsem_bindings.solenoid_stress_fem_pressure_operator_axisymmetric_quad4_f32
)
_pressure_operator_axisymmetric_quad4_f64 = (
    _cfsem_bindings.solenoid_stress_fem_pressure_operator_axisymmetric_quad4_f64
)
_pressure_operator_axisymmetric_quad9_f32 = (
    _cfsem_bindings.solenoid_stress_fem_pressure_operator_axisymmetric_quad9_f32
)
_pressure_operator_axisymmetric_quad9_f64 = (
    _cfsem_bindings.solenoid_stress_fem_pressure_operator_axisymmetric_quad9_f64
)
_traction_operator_axisymmetric_quad4_f32 = (
    _cfsem_bindings.solenoid_stress_fem_traction_operator_axisymmetric_quad4_f32
)
_traction_operator_axisymmetric_quad4_f64 = (
    _cfsem_bindings.solenoid_stress_fem_traction_operator_axisymmetric_quad4_f64
)
_traction_operator_axisymmetric_quad9_f32 = (
    _cfsem_bindings.solenoid_stress_fem_traction_operator_axisymmetric_quad9_f32
)
_traction_operator_axisymmetric_quad9_f64 = (
    _cfsem_bindings.solenoid_stress_fem_traction_operator_axisymmetric_quad9_f64
)
_temperature_operator_axisymmetric_quad4_f32 = (
    _cfsem_bindings.solenoid_stress_fem_temperature_operator_axisymmetric_quad4_f32
)
_temperature_operator_axisymmetric_quad4_f64 = (
    _cfsem_bindings.solenoid_stress_fem_temperature_operator_axisymmetric_quad4_f64
)
_temperature_operator_axisymmetric_quad9_f32 = (
    _cfsem_bindings.solenoid_stress_fem_temperature_operator_axisymmetric_quad9_f32
)
_temperature_operator_axisymmetric_quad9_f64 = (
    _cfsem_bindings.solenoid_stress_fem_temperature_operator_axisymmetric_quad9_f64
)

ArrayLike = npt.ArrayLike
ElementType = str


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    """Sparse system assembled in COO triplet form."""

    rows: npt.NDArray[np.int64]
    cols: npt.NDArray[np.int64]
    vals: npt.NDArray[np.floating[Any]]
    rhs: npt.NDArray[np.floating[Any]]
    ndof: int

    def to_coo(self) -> sp.coo_matrix:
        return sp.coo_matrix((self.vals, (self.rows, self.cols)), shape=(self.ndof, self.ndof))

    def to_csr(self) -> sp.csr_matrix:
        return self.to_coo().tocsr()


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
class QuadratureFieldOperators:
    """Sparse operators that map nodal displacements to quadrature-point strain/stress."""

    points_rz: npt.NDArray[np.floating[Any]]
    strain_operator: sp.csr_matrix
    stress_operator: sp.csr_matrix
    thermal_strain_operator: sp.csr_matrix
    thermal_stress_operator: sp.csr_matrix
    thermal_strain_constant: npt.NDArray[np.floating[Any]]
    thermal_stress_constant: npt.NDArray[np.floating[Any]]
    nq_per_element: int
    ndof: int
    ntemp: int


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


@dataclass(frozen=True, slots=True)
class ReducedSystem:
    """Linear system after eliminating prescribed Dirichlet dofs."""

    matrix: sp.csr_matrix
    rhs: npt.NDArray[np.floating[Any]]
    free_dofs: npt.NDArray[np.int64]
    fixed_dofs: npt.NDArray[np.int64]
    fixed_values: npt.NDArray[np.floating[Any]]
    ndof: int

    def recover(self, free_solution: ArrayLike) -> npt.NDArray[np.floating[Any]]:
        solution = np.zeros(self.ndof, dtype=self.rhs.dtype)
        solution[self.fixed_dofs] = self.fixed_values
        solution[self.free_dofs] = np.asarray(free_solution, dtype=self.rhs.dtype)
        return solution


@dataclass(frozen=True, slots=True)
class AxisymmetricFEMModel:
    """
    Reusable axisymmetric FEM model with fixed stiffness and linear load operators.

    `body_force_to_rhs` maps flattened per-element body-force data
    `[f_r(0), f_z(0), f_r(1), f_z(1), ...]` onto the global load vector.
    `pressure_to_rhs` maps the reusable `pressure_values` vector associated with
    `pressure_faces` onto the global load vector. `traction_to_rhs` maps flattened
    per-face traction vectors `[t_r(0), t_z(0), ...]` associated with `traction_faces`
    onto the same load vector.
    """

    stiffness: sp.csr_matrix
    body_force_to_rhs: sp.csr_matrix
    pressure_to_rhs: sp.csr_matrix
    traction_to_rhs: sp.csr_matrix
    temperature_to_rhs: sp.csr_matrix
    thermal_reference_rhs: npt.NDArray[np.floating[Any]]
    pressure_faces: npt.NDArray[np.uint64]
    traction_faces: npt.NDArray[np.uint64]
    analysis_nodes: npt.NDArray[np.floating[Any]]
    analysis_elements: npt.NDArray[np.uint64]
    element_type: str
    ndof: int
    nelem: int
    n_temperature_nodes: int
    dtype: np.dtype[Any]

    def body_force_rhs(self, body_force: ArrayLike | None = None) -> npt.NDArray[np.floating[Any]]:
        body_force_arr = _normalize_body_force_or_zero(body_force, self.nelem, self.dtype)
        return np.asarray(self.body_force_to_rhs @ body_force_arr.reshape(-1), dtype=self.dtype)

    def pressure_rhs(self, pressure_values: ArrayLike | None = None) -> npt.NDArray[np.floating[Any]]:
        pressure_arr = _normalize_pressure_values(pressure_values, self.pressure_to_rhs.shape[1], self.dtype)
        if pressure_arr.size == 0:
            return np.zeros((self.ndof,), dtype=self.dtype)
        return np.asarray(self.pressure_to_rhs @ pressure_arr, dtype=self.dtype)

    def traction_rhs(self, traction_values: ArrayLike | None = None) -> npt.NDArray[np.floating[Any]]:
        traction_arr = _normalize_traction_values(
            traction_values,
            self.traction_to_rhs.shape[1] // 2,
            self.dtype,
        )
        if traction_arr.size == 0:
            return np.zeros((self.ndof,), dtype=self.dtype)
        return np.asarray(self.traction_to_rhs @ traction_arr.reshape(-1), dtype=self.dtype)

    def temperature_rhs(
        self,
        nodal_temperature: ArrayLike | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        if self.temperature_to_rhs.shape[1] == 0:
            return np.zeros((self.ndof,), dtype=self.dtype)
        if nodal_temperature is None:
            raise ValueError("nodal_temperature is required because this model includes thermal materials")
        temperature_arr = _normalize_nodal_temperature(
            nodal_temperature,
            self.n_temperature_nodes,
            self.dtype,
        )
        return np.asarray(self.temperature_to_rhs @ temperature_arr, dtype=self.dtype)

    def rhs(
        self,
        body_force: ArrayLike | None = None,
        pressure_values: ArrayLike | None = None,
        traction_values: ArrayLike | None = None,
        nodal_temperature: ArrayLike | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        return (
            self.thermal_reference_rhs
            + self.body_force_rhs(body_force)
            + self.pressure_rhs(pressure_values)
            + self.traction_rhs(traction_values)
            + self.temperature_rhs(nodal_temperature)
        )

    def apply_dirichlet(
        self,
        prescribed: Mapping[int, float] | None = None,
    ) -> ReducedAxisymmetricFEMModel:
        reduced = apply_dirichlet(
            self.stiffness,
            np.zeros((self.ndof,), dtype=self.dtype),
            prescribed=prescribed,
        )
        return ReducedAxisymmetricFEMModel(
            matrix=reduced.matrix,
            body_force_to_rhs=self.body_force_to_rhs[reduced.free_dofs].tocsr(),
            pressure_to_rhs=self.pressure_to_rhs[reduced.free_dofs].tocsr(),
            traction_to_rhs=self.traction_to_rhs[reduced.free_dofs].tocsr(),
            temperature_to_rhs=self.temperature_to_rhs[reduced.free_dofs].tocsr(),
            thermal_reference_rhs=(
                apply_dirichlet(
                    self.stiffness,
                    self.thermal_reference_rhs,
                    prescribed=prescribed,
                ).rhs
                - reduced.rhs
            ),
            constant_rhs=reduced.rhs,
            pressure_faces=self.pressure_faces,
            traction_faces=self.traction_faces,
            analysis_nodes=self.analysis_nodes,
            analysis_elements=self.analysis_elements,
            element_type=self.element_type,
            free_dofs=reduced.free_dofs,
            fixed_dofs=reduced.fixed_dofs,
            fixed_values=reduced.fixed_values,
            ndof=self.ndof,
            nelem=self.nelem,
            n_temperature_nodes=self.n_temperature_nodes,
            dtype=self.dtype,
        )

    def solve_dirichlet(
        self,
        body_force: ArrayLike | None = None,
        pressure_values: ArrayLike | None = None,
        traction_values: ArrayLike | None = None,
        nodal_temperature: ArrayLike | None = None,
        prescribed: Mapping[int, float] | None = None,
        solver: Any | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        return self.apply_dirichlet(prescribed).solve(
            body_force=body_force,
            pressure_values=pressure_values,
            traction_values=traction_values,
            nodal_temperature=nodal_temperature,
            solver=solver,
        )

    def factorized_solver(
        self,
        prescribed: Mapping[int, float] | None = None,
    ) -> Callable[
        [ArrayLike | None, ArrayLike | None, ArrayLike | None, ArrayLike | None],
        npt.NDArray[np.floating[Any]],
    ]:
        return self.apply_dirichlet(prescribed).factorized_solver()


@dataclass(frozen=True, slots=True)
class ReducedAxisymmetricFEMModel:
    """Axisymmetric FEM model after eliminating prescribed Dirichlet dofs."""

    matrix: sp.csr_matrix
    body_force_to_rhs: sp.csr_matrix
    pressure_to_rhs: sp.csr_matrix
    traction_to_rhs: sp.csr_matrix
    temperature_to_rhs: sp.csr_matrix
    thermal_reference_rhs: npt.NDArray[np.floating[Any]]
    constant_rhs: npt.NDArray[np.floating[Any]]
    pressure_faces: npt.NDArray[np.uint64]
    traction_faces: npt.NDArray[np.uint64]
    analysis_nodes: npt.NDArray[np.floating[Any]]
    analysis_elements: npt.NDArray[np.uint64]
    element_type: str
    free_dofs: npt.NDArray[np.int64]
    fixed_dofs: npt.NDArray[np.int64]
    fixed_values: npt.NDArray[np.floating[Any]]
    ndof: int
    nelem: int
    n_temperature_nodes: int
    dtype: np.dtype[Any]

    def recover(self, free_solution: ArrayLike) -> npt.NDArray[np.floating[Any]]:
        solution = np.zeros(self.ndof, dtype=self.dtype)
        solution[self.fixed_dofs] = self.fixed_values
        solution[self.free_dofs] = np.asarray(free_solution, dtype=self.dtype)
        return solution

    def body_force_rhs(self, body_force: ArrayLike | None = None) -> npt.NDArray[np.floating[Any]]:
        body_force_arr = _normalize_body_force_or_zero(body_force, self.nelem, self.dtype)
        return np.asarray(self.body_force_to_rhs @ body_force_arr.reshape(-1), dtype=self.dtype)

    def pressure_rhs(self, pressure_values: ArrayLike | None = None) -> npt.NDArray[np.floating[Any]]:
        pressure_arr = _normalize_pressure_values(pressure_values, self.pressure_to_rhs.shape[1], self.dtype)
        if pressure_arr.size == 0:
            return np.zeros((self.matrix.shape[0],), dtype=self.dtype)
        return np.asarray(self.pressure_to_rhs @ pressure_arr, dtype=self.dtype)

    def traction_rhs(self, traction_values: ArrayLike | None = None) -> npt.NDArray[np.floating[Any]]:
        traction_arr = _normalize_traction_values(
            traction_values,
            self.traction_to_rhs.shape[1] // 2,
            self.dtype,
        )
        if traction_arr.size == 0:
            return np.zeros((self.matrix.shape[0],), dtype=self.dtype)
        return np.asarray(self.traction_to_rhs @ traction_arr.reshape(-1), dtype=self.dtype)

    def temperature_rhs(
        self,
        nodal_temperature: ArrayLike | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        if self.temperature_to_rhs.shape[1] == 0:
            return np.zeros((self.matrix.shape[0],), dtype=self.dtype)
        if nodal_temperature is None:
            raise ValueError("nodal_temperature is required because this model includes thermal materials")
        temperature_arr = _normalize_nodal_temperature(
            nodal_temperature,
            self.n_temperature_nodes,
            self.dtype,
        )
        return np.asarray(self.temperature_to_rhs @ temperature_arr, dtype=self.dtype)

    def rhs(
        self,
        body_force: ArrayLike | None = None,
        pressure_values: ArrayLike | None = None,
        traction_values: ArrayLike | None = None,
        nodal_temperature: ArrayLike | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        return (
            self.constant_rhs
            + self.thermal_reference_rhs
            + self.body_force_rhs(body_force)
            + self.pressure_rhs(pressure_values)
            + self.traction_rhs(traction_values)
            + self.temperature_rhs(nodal_temperature)
        )

    def solve(
        self,
        body_force: ArrayLike | None = None,
        pressure_values: ArrayLike | None = None,
        traction_values: ArrayLike | None = None,
        nodal_temperature: ArrayLike | None = None,
        solver: Any | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        rhs = self.rhs(
            body_force=body_force,
            pressure_values=pressure_values,
            traction_values=traction_values,
            nodal_temperature=nodal_temperature,
        )
        if self.matrix.shape[0] == 0:
            return self.recover(np.zeros((0,), dtype=self.dtype))
        if solver is None:
            free_solution = spla.factorized(self.matrix.tocsc())(rhs)
        else:
            free_solution = solver(self.matrix, rhs)
        return self.recover(free_solution)

    def factorized_solver(
        self,
    ) -> Callable[
        [ArrayLike | None, ArrayLike | None, ArrayLike | None, ArrayLike | None],
        npt.NDArray[np.floating[Any]],
    ]:
        if self.matrix.shape[0] == 0:

            def solve_empty(
                body_force: ArrayLike | None = None,
                pressure_values: ArrayLike | None = None,
                traction_values: ArrayLike | None = None,
                nodal_temperature: ArrayLike | None = None,
            ) -> npt.NDArray[np.floating[Any]]:
                del body_force, pressure_values, traction_values, nodal_temperature
                return self.recover(np.zeros((0,), dtype=self.dtype))

            return solve_empty

        solve_free = spla.factorized(self.matrix.tocsc())

        def solve_case(
            body_force: ArrayLike | None = None,
            pressure_values: ArrayLike | None = None,
            traction_values: ArrayLike | None = None,
            nodal_temperature: ArrayLike | None = None,
        ) -> npt.NDArray[np.floating[Any]]:
            return self.recover(
                solve_free(
                    self.rhs(
                        body_force=body_force,
                        pressure_values=pressure_values,
                        traction_values=traction_values,
                        nodal_temperature=nodal_temperature,
                    )
                )
            )

        return solve_case


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
    if quadrature in (3, "3", "gl3", "GL3", "gausslegendre3", "GaussLegendre3", "3x3", "gauss3x3", "Gauss3x3"):
        return 3
    if quadrature in (4, "4", "gl4", "GL4", "gausslegendre4", "GaussLegendre4", "4x4", "gauss4x4", "Gauss4x4"):
        return 4
    raise ValueError(f"unsupported quadrature {quadrature!r}; use 'gl3' or 'gl4'")


def _normalize_element_type(element_type: str) -> str:
    normalized = str(element_type).strip().lower()
    if normalized not in {"quad4", "quad9"}:
        raise ValueError(f"unsupported element_type {element_type!r}; use 'quad4' or 'quad9'")
    return normalized


def _validate_element_quadrature_combo(element_type: str, quadrature_code: int) -> None:
    if quadrature_code not in {3, 4}:
        raise ValueError(f"unsupported quadrature code {quadrature_code}; use 3 or 4")
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
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"nodes must have shape (nnode, 2); got {arr.shape}")
    return np.ascontiguousarray(arr)


def _normalize_elements(elements: ArrayLike) -> npt.NDArray[np.uint64]:
    arr = np.asarray(elements, dtype=np.uint64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"elements must have shape (nelem, 4); got {arr.shape}")
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
            edge_key = tuple(sorted((int(conn[local_a]), int(conn[local_b]))))
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
            edge_key = tuple(sorted((int(conn[local_a]), int(conn[local_b]))))
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
        (np.asarray(vals, dtype=dtype), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n_analysis_nodes, n_input_nodes),
    ).tocsr()


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
    if ids.ndim != 1:
        raise ValueError(f"material_ids must have shape (nelem,); got {ids.shape}")
    if isinstance(material_table, Mapping):
        if not material_table:
            raise ValueError("material_table mapping cannot be empty")
        keys = sorted(int(key) for key in material_table)
        dense_table = []
        tag_to_index = {key: index for index, key in enumerate(keys)}
        for key in keys:
            matrix = np.asarray(material_table[key], dtype=dtype)
            if matrix.shape != (4, 4):
                raise ValueError(f"material_table[{key}] must have shape (4, 4); got {matrix.shape}")
            dense_table.append(matrix)
        try:
            normalized_ids = np.asarray([tag_to_index[int(tag)] for tag in ids], dtype=np.uint64)
        except KeyError as exc:
            raise ValueError(
                f"material_ids contains tag {exc.args[0]} that is missing from material_table"
            ) from exc
        return normalized_ids, np.ascontiguousarray(np.stack(dense_table, axis=0), dtype=dtype)

    table = np.asarray(material_table, dtype=dtype)
    if table.ndim != 3 or table.shape[1:] != (4, 4):
        raise ValueError(f"material_table must have shape (nmat, 4, 4); got {table.shape}")
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
    if ids.ndim != 1:
        raise ValueError(f"material_ids must have shape (nelem,); got {ids.shape}")
    is_mapping = isinstance(thermal_material_table, Mapping)
    if require_mapping is not None and is_mapping != require_mapping:
        raise ValueError(
            "thermal_material_table must use the same mapping/dense convention as material_table"
        )
    if is_mapping:
        mapping = thermal_material_table
        if not mapping:
            raise ValueError("thermal_material_table mapping cannot be empty")
        keys = sorted(int(key) for key in mapping)
        dense_table = []
        tag_to_index = {key: index for index, key in enumerate(keys)}
        for key in keys:
            row = np.asarray(mapping[key], dtype=dtype)
            if row.shape != (5,):
                raise ValueError(
                    f"thermal_material_table[{key}] must have shape (5,); got {row.shape}"
                )
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
        if table.ndim != 2 or table.shape[1] != 5:
            raise ValueError(f"thermal_material_table must have shape (nmat, 5); got {table.shape}")
        normalized_ids = np.ascontiguousarray(ids)
        table = np.ascontiguousarray(table)
    if not np.allclose(table[:, 3], 0.0):
        raise ValueError("thermal_material_table shear thermal expansion must be zero in phase 1")
    return normalized_ids, table


def _empty_thermal_material_table(dtype: np.dtype[Any]) -> npt.NDArray[np.floating[Any]]:
    return np.zeros((0, 5), dtype=dtype)


def _normalize_nodal_temperature(
    nodal_temperature: ArrayLike,
    nnode: int,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    arr = np.asarray(nodal_temperature, dtype=dtype)
    if arr.ndim != 1 or arr.shape[0] != nnode:
        raise ValueError(f"nodal_temperature must have shape ({nnode},); got {arr.shape}")
    return np.ascontiguousarray(arr)


def _normalize_body_force(
    body_force: ArrayLike,
    nelem: int,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    arr = np.asarray(body_force, dtype=dtype)
    if arr.ndim == 1 and arr.shape == (2,):
        arr = np.broadcast_to(arr, (nelem, 2)).copy()
    if arr.ndim != 2 or arr.shape != (nelem, 2):
        raise ValueError(f"body_force must have shape (2,) or (nelem, 2); got {arr.shape}")
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
    if faces.ndim != 2 or faces.shape[1] != 2:
        raise ValueError(f"pressure_faces must have shape (nload, 2); got {faces.shape}")
    return np.ascontiguousarray(faces)


def _normalize_pressure_values(
    pressure_values: ArrayLike | None,
    nload: int,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    if pressure_values is None:
        return np.zeros((nload,), dtype=dtype)
    values = np.asarray(pressure_values, dtype=dtype)
    if values.ndim != 1:
        raise ValueError(f"pressure_values must have shape (nload,); got {values.shape}")
    if values.shape[0] != nload:
        raise ValueError(f"pressure_values has {values.shape[0]} entries, but expected {nload}")
    return np.ascontiguousarray(values)


def _normalize_pressure_loads(
    pressure_faces: ArrayLike | None,
    pressure_values: ArrayLike | None,
    dtype: np.dtype[Any],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.floating[Any]]]:
    if pressure_faces is None and pressure_values is None:
        return _normalize_pressure_faces(None), _normalize_pressure_values(None, 0, dtype)
    if pressure_faces is None or pressure_values is None:
        raise ValueError("pressure_faces and pressure_values must either both be provided or both be omitted")
    faces = _normalize_pressure_faces(pressure_faces)
    values = np.asarray(pressure_values, dtype=dtype)
    if values.ndim != 1:
        raise ValueError(f"pressure_values must have shape (nload,); got {values.shape}")
    if faces.shape[0] != values.shape[0]:
        raise ValueError(
            f"pressure_faces has {faces.shape[0]} rows, but pressure_values has {values.shape[0]} entries"
        )
    return faces, np.ascontiguousarray(values)


def _normalize_traction_faces(traction_faces: ArrayLike | None) -> npt.NDArray[np.uint64]:
    if traction_faces is None:
        return np.zeros((0, 2), dtype=np.uint64)
    faces = np.asarray(traction_faces, dtype=np.uint64)
    if faces.ndim != 2 or faces.shape[1] != 2:
        raise ValueError(f"traction_faces must have shape (nload, 2); got {faces.shape}")
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
    if values.ndim != 2 or values.shape != (nload, 2):
        raise ValueError(f"traction_values must have shape (2,) or ({nload}, 2); got {values.shape}")
    return np.ascontiguousarray(values)


def _normalize_traction_loads(
    traction_faces: ArrayLike | None,
    traction_values: ArrayLike | None,
    dtype: np.dtype[Any],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.floating[Any]]]:
    if traction_faces is None and traction_values is None:
        return _normalize_traction_faces(None), _normalize_traction_values(None, 0, dtype)
    if traction_faces is None or traction_values is None:
        raise ValueError("traction_faces and traction_values must either both be provided or both be omitted")
    faces = _normalize_traction_faces(traction_faces)
    values = _normalize_traction_values(traction_values, faces.shape[0], dtype)
    return faces, values


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
    if radius <= np.finfo(dtype).eps:
        raise ValueError(f"quadrature radius {radius} is too close to zero")
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
            if det_j <= 0.0:
                raise ValueError(f"encountered non-positive element Jacobian determinant {float(det_j)!r}")
            point = n @ coords
            if point[0] < 0.0:
                raise ValueError(f"quadrature point has negative radius {float(point[0])!r}")
            inv_j = np.linalg.inv(jac)
            grad_phys = np.column_stack(
                [
                    inv_j[0, 0] * grad_ref[:, 0] + inv_j[1, 0] * grad_ref[:, 1],
                    inv_j[0, 1] * grad_ref[:, 0] + inv_j[1, 1] * grad_ref[:, 1],
                ]
            )
            yield n, grad_phys, det_j, np.asarray(point, dtype=dtype), dtype.type(wx * wy)


def _face_samples(
    coords: npt.NDArray[np.floating[Any]],
    element_type: str,
    local_face: int,
    quadrature_code: int,
    dtype: np.dtype[Any],
):
    for s, weight in _gauss_1d(quadrature_code):
        xi, eta, ds_reference = _quad_face_reference(local_face, s)
        n = _element_shape(element_type, xi, eta).astype(dtype, copy=False)
        grad_ref = _element_grad_ref(element_type, xi, eta).astype(dtype, copy=False)
        jac = _element_jacobian(coords, grad_ref, dtype)
        point = n @ coords
        if point[0] < 0.0:
            raise ValueError(f"face quadrature point has negative radius {float(point[0])!r}")
        tangent = np.array(
            [
                jac[0, 0] * ds_reference[0] + jac[0, 1] * ds_reference[1],
                jac[1, 0] * ds_reference[0] + jac[1, 1] * ds_reference[1],
            ],
            dtype=dtype,
        )
        tangent_norm_sq = tangent[0] * tangent[0] + tangent[1] * tangent[1]
        if tangent_norm_sq <= 0.0:
            raise ValueError(
                f"degenerate face tangent on local face {local_face}; "
                f"tangent squared norm is {float(tangent_norm_sq)!r}"
            )
        yield n, tangent, np.asarray(point, dtype=dtype), dtype.type(weight)


def _assemble_axisymmetric_python(
    nodes: npt.NDArray[np.floating[Any]],
    elements: npt.NDArray[np.uint64],
    material_ids: npt.NDArray[np.uint64],
    material_table: npt.NDArray[np.floating[Any]],
    body_force: npt.NDArray[np.floating[Any]],
    pressure_faces: npt.NDArray[np.uint64],
    pressure_values: npt.NDArray[np.floating[Any]],
    traction_faces: npt.NDArray[np.uint64],
    traction_values: npt.NDArray[np.floating[Any]],
    quadrature_code: int,
    dtype: np.dtype[Any],
    element_type: str,
) -> AssemblyResult:
    nelem = elements.shape[0]
    ndof = nodes.shape[0] * 2
    nnodes_per_element = elements.shape[1]
    dof_per_element = 2 * nnodes_per_element
    rows: list[int] = []
    cols: list[int] = []
    vals: list[Any] = []
    rhs = np.zeros((ndof,), dtype=dtype)
    two_pi = dtype.type(2.0 * np.pi)

    for element_index, conn in enumerate(elements):
        coords = nodes[conn]
        material = material_table[int(material_ids[element_index])]
        ke = np.zeros((dof_per_element, dof_per_element), dtype=dtype)
        fe = np.zeros((dof_per_element,), dtype=dtype)
        for n, grad_phys, det_j, point, weight in _volume_samples(
            coords, element_type, quadrature_code, dtype
        ):
            b = _axisymmetric_b_matrix(n, grad_phys, float(point[0]), dtype)
            scale = two_pi * point[0] * det_j * weight
            ke += scale * (b.T @ material @ b)
            for local_node in range(nnodes_per_element):
                fe[2 * local_node] += scale * n[local_node] * body_force[element_index, 0]
                fe[2 * local_node + 1] += scale * n[local_node] * body_force[element_index, 1]

        local_dofs = np.empty((dof_per_element,), dtype=np.int64)
        for local_node, global_node in enumerate(conn):
            local_dofs[2 * local_node] = 2 * int(global_node)
            local_dofs[2 * local_node + 1] = 2 * int(global_node) + 1
        rhs[local_dofs] += fe
        for row_local, row_dof in enumerate(local_dofs):
            for col_local, col_dof in enumerate(local_dofs):
                rows.append(int(row_dof))
                cols.append(int(col_dof))
                vals.append(ke[row_local, col_local])

    for load_index, (element_index_u64, local_face_u64) in enumerate(pressure_faces):
        element_index = int(element_index_u64)
        if element_index < 0 or element_index >= nelem:
            raise ValueError(
                f"pressure_faces references element {element_index}, "
                f"but mesh has {elements.shape[0]} elements"
            )
        local_face = int(local_face_u64)
        conn = elements[element_index]
        coords = nodes[conn]
        fe = np.zeros((dof_per_element,), dtype=dtype)
        for n, tangent, point, weight in _face_samples(
            coords, element_type, local_face, quadrature_code, dtype
        ):
            normal_area = np.array([tangent[1], -tangent[0]], dtype=dtype)
            scale = -pressure_values[load_index] * two_pi * point[0] * weight
            for local_node in range(nnodes_per_element):
                fe[2 * local_node] += scale * n[local_node] * normal_area[0]
                fe[2 * local_node + 1] += scale * n[local_node] * normal_area[1]
        for local_node, global_node in enumerate(conn):
            rhs[2 * int(global_node)] += fe[2 * local_node]
            rhs[2 * int(global_node) + 1] += fe[2 * local_node + 1]

    for load_index, (element_index_u64, local_face_u64) in enumerate(traction_faces):
        element_index = int(element_index_u64)
        if element_index < 0 or element_index >= nelem:
            raise ValueError(
                f"traction_faces references element {element_index}, "
                f"but mesh has {elements.shape[0]} elements"
            )
        local_face = int(local_face_u64)
        conn = elements[element_index]
        coords = nodes[conn]
        fe = np.zeros((dof_per_element,), dtype=dtype)
        for n, tangent, point, weight in _face_samples(
            coords, element_type, local_face, quadrature_code, dtype
        ):
            tangent_norm = dtype.type(np.sqrt(tangent[0] * tangent[0] + tangent[1] * tangent[1]))
            scale = two_pi * point[0] * tangent_norm * weight
            for local_node in range(nnodes_per_element):
                fe[2 * local_node] += scale * n[local_node] * traction_values[load_index, 0]
                fe[2 * local_node + 1] += scale * n[local_node] * traction_values[load_index, 1]
        for local_node, global_node in enumerate(conn):
            rhs[2 * int(global_node)] += fe[2 * local_node]
            rhs[2 * int(global_node) + 1] += fe[2 * local_node + 1]

    return AssemblyResult(
        rows=np.asarray(rows, dtype=np.int64),
        cols=np.asarray(cols, dtype=np.int64),
        vals=np.asarray(vals, dtype=dtype),
        rhs=rhs,
        ndof=ndof,
    )


def _assemble_body_force_operator(
    nodes: npt.NDArray[np.floating[Any]],
    elements: npt.NDArray[np.uint64],
    quadrature_code: int,
    dtype: np.dtype[Any],
    element_type: str,
) -> sp.csr_matrix:
    return _assemble_body_force_operator_rust(nodes, elements, quadrature_code, dtype, element_type)


def _assemble_pressure_operator(
    nodes: npt.NDArray[np.floating[Any]],
    elements: npt.NDArray[np.uint64],
    pressure_faces: npt.NDArray[np.uint64],
    quadrature_code: int,
    dtype: np.dtype[Any],
    element_type: str,
) -> sp.csr_matrix:
    return _assemble_pressure_operator_rust(
        nodes,
        elements,
        pressure_faces,
        quadrature_code,
        dtype,
        element_type,
    )


def _assemble_traction_operator(
    nodes: npt.NDArray[np.floating[Any]],
    elements: npt.NDArray[np.uint64],
    traction_faces: npt.NDArray[np.uint64],
    quadrature_code: int,
    dtype: np.dtype[Any],
    element_type: str,
) -> sp.csr_matrix:
    return _assemble_traction_operator_rust(
        nodes,
        elements,
        traction_faces,
        quadrature_code,
        dtype,
        element_type,
    )


def _assemble_temperature_operator(
    nodes: npt.NDArray[np.floating[Any]],
    elements: npt.NDArray[np.uint64],
    material_ids: npt.NDArray[np.uint64],
    material_table: npt.NDArray[np.floating[Any]],
    thermal_material_table: npt.NDArray[np.floating[Any]] | None,
    quadrature_code: int,
    dtype: np.dtype[Any],
    element_type: str,
) -> tuple[sp.csr_matrix, npt.NDArray[np.floating[Any]]]:
    if thermal_material_table is None:
        return (
            sp.csr_matrix((2 * nodes.shape[0], 0), dtype=dtype),
            np.zeros((2 * nodes.shape[0],), dtype=dtype),
        )
    return _assemble_temperature_operator_rust(
        nodes,
        elements,
        material_ids,
        material_table,
        thermal_material_table,
        quadrature_code,
        dtype,
        element_type,
    )


def _coo_operator_from_triplets(
    rows: ArrayLike,
    cols: ArrayLike,
    vals: ArrayLike,
    shape: tuple[int, int],
    dtype: np.dtype[Any],
) -> sp.csr_matrix:
    return sp.coo_matrix(
        (
            np.asarray(vals, dtype=dtype),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(cols, dtype=np.int64),
            ),
        ),
        shape=shape,
    ).tocsr()


def _assemble_body_force_operator_rust(
    nodes: npt.NDArray[np.floating[Any]],
    elements: npt.NDArray[np.uint64],
    quadrature_code: int,
    dtype: np.dtype[Any],
    element_type: str,
) -> sp.csr_matrix:
    low_level = _dispatch_pair(
        dtype,
        _dispatch_by_element_type(
            element_type,
            _body_force_operator_axisymmetric_quad4_f32,
            _body_force_operator_axisymmetric_quad9_f32,
        ),
        _dispatch_by_element_type(
            element_type,
            _body_force_operator_axisymmetric_quad4_f64,
            _body_force_operator_axisymmetric_quad9_f64,
        ),
    )
    rows, cols, vals, nrow, ncol = low_level(nodes, elements, quadrature_code)
    return _coo_operator_from_triplets(rows, cols, vals, shape=(int(nrow), int(ncol)), dtype=dtype)


def _assemble_pressure_operator_rust(
    nodes: npt.NDArray[np.floating[Any]],
    elements: npt.NDArray[np.uint64],
    pressure_faces: npt.NDArray[np.uint64],
    quadrature_code: int,
    dtype: np.dtype[Any],
    element_type: str,
) -> sp.csr_matrix:
    if pressure_faces.shape[0] == 0:
        return sp.csr_matrix((2 * nodes.shape[0], 0), dtype=dtype)
    low_level = _dispatch_pair(
        dtype,
        _dispatch_by_element_type(
            element_type,
            _pressure_operator_axisymmetric_quad4_f32,
            _pressure_operator_axisymmetric_quad9_f32,
        ),
        _dispatch_by_element_type(
            element_type,
            _pressure_operator_axisymmetric_quad4_f64,
            _pressure_operator_axisymmetric_quad9_f64,
        ),
    )
    rows, cols, vals, nrow, ncol = low_level(nodes, elements, pressure_faces, quadrature_code)
    return _coo_operator_from_triplets(rows, cols, vals, shape=(int(nrow), int(ncol)), dtype=dtype)


def _assemble_temperature_operator_rust(
    nodes: npt.NDArray[np.floating[Any]],
    elements: npt.NDArray[np.uint64],
    material_ids: npt.NDArray[np.uint64],
    material_table: npt.NDArray[np.floating[Any]],
    thermal_material_table: npt.NDArray[np.floating[Any]],
    quadrature_code: int,
    dtype: np.dtype[Any],
    element_type: str,
) -> tuple[sp.csr_matrix, npt.NDArray[np.floating[Any]]]:
    low_level = _dispatch_pair(
        dtype,
        _dispatch_by_element_type(
            element_type,
            _temperature_operator_axisymmetric_quad4_f32,
            _temperature_operator_axisymmetric_quad9_f32,
        ),
        _dispatch_by_element_type(
            element_type,
            _temperature_operator_axisymmetric_quad4_f64,
            _temperature_operator_axisymmetric_quad9_f64,
        ),
    )
    rows, cols, vals, reference_rhs, nrow, ncol = low_level(
        nodes,
        elements,
        material_ids,
        material_table,
        thermal_material_table,
        quadrature_code,
    )
    return (
        _coo_operator_from_triplets(rows, cols, vals, shape=(int(nrow), int(ncol)), dtype=dtype),
        np.asarray(reference_rhs, dtype=dtype),
    )


def _assemble_stiffness_rust(
    nodes: npt.NDArray[np.floating[Any]],
    elements: npt.NDArray[np.uint64],
    material_ids: npt.NDArray[np.uint64],
    material_table: npt.NDArray[np.floating[Any]],
    quadrature_code: int,
    dtype: np.dtype[Any],
    element_type: str,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.floating[Any]], int]:
    low_level = _dispatch_pair(
        dtype,
        _dispatch_by_element_type(
            element_type,
            _assemble_axisymmetric_quad4_f32,
            _assemble_axisymmetric_quad9_f32,
        ),
        _dispatch_by_element_type(
            element_type,
            _assemble_axisymmetric_quad4_f64,
            _assemble_axisymmetric_quad9_f64,
        ),
    )
    rows, cols, vals, ndof = low_level(
        nodes,
        elements,
        material_ids,
        material_table,
        quadrature_code,
    )
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(vals, dtype=dtype),
        int(ndof),
    )


def _assemble_traction_operator_rust(
    nodes: npt.NDArray[np.floating[Any]],
    elements: npt.NDArray[np.uint64],
    traction_faces: npt.NDArray[np.uint64],
    quadrature_code: int,
    dtype: np.dtype[Any],
    element_type: str,
) -> sp.csr_matrix:
    if traction_faces.shape[0] == 0:
        return sp.csr_matrix((2 * nodes.shape[0], 0), dtype=dtype)
    low_level = _dispatch_pair(
        dtype,
        _dispatch_by_element_type(
            element_type,
            _traction_operator_axisymmetric_quad4_f32,
            _traction_operator_axisymmetric_quad9_f32,
        ),
        _dispatch_by_element_type(
            element_type,
            _traction_operator_axisymmetric_quad4_f64,
            _traction_operator_axisymmetric_quad9_f64,
        ),
    )
    rows, cols, vals, nrow, ncol = low_level(nodes, elements, traction_faces, quadrature_code)
    return _coo_operator_from_triplets(rows, cols, vals, shape=(int(nrow), int(ncol)), dtype=dtype)


def _quadrature_field_operators_rust(
    nodes: npt.NDArray[np.floating[Any]],
    elements: npt.NDArray[np.uint64],
    material_ids: npt.NDArray[np.uint64],
    material_table: npt.NDArray[np.floating[Any]],
    thermal_material_table: npt.NDArray[np.floating[Any]] | None,
    quadrature_code: int,
    dtype: np.dtype[Any],
    element_type: str,
) -> QuadratureFieldOperators:
    low_level = _dispatch_pair(
        dtype,
        _dispatch_by_element_type(
            element_type,
            _quadrature_field_operators_axisymmetric_quad4_f32,
            _quadrature_field_operators_axisymmetric_quad9_f32,
        ),
        _dispatch_by_element_type(
            element_type,
            _quadrature_field_operators_axisymmetric_quad4_f64,
            _quadrature_field_operators_axisymmetric_quad9_f64,
        ),
    )
    thermal_table_arr = (
        _empty_thermal_material_table(dtype)
        if thermal_material_table is None
        else thermal_material_table
    )
    (
        points_flat,
        (strain_rows, strain_cols, strain_vals),
        (stress_rows, stress_cols, stress_vals),
        (thermal_strain_rows, thermal_strain_cols, thermal_strain_vals),
        (thermal_stress_rows, thermal_stress_cols, thermal_stress_vals),
        (thermal_strain_constant, thermal_stress_constant),
        (nq_per_element, ndof, ntemp),
    ) = low_level(
        nodes,
        elements,
        material_ids,
        material_table,
        thermal_table_arr,
        quadrature_code,
    )
    nelem = elements.shape[0]
    nrow = nelem * int(nq_per_element) * 4
    return QuadratureFieldOperators(
        points_rz=np.asarray(points_flat, dtype=dtype).reshape(nelem, int(nq_per_element), 2),
        strain_operator=_coo_operator_from_triplets(
            strain_rows,
            strain_cols,
            strain_vals,
            shape=(nrow, int(ndof)),
            dtype=dtype,
        ),
        stress_operator=_coo_operator_from_triplets(
            stress_rows,
            stress_cols,
            stress_vals,
            shape=(nrow, int(ndof)),
            dtype=dtype,
        ),
        thermal_strain_operator=_coo_operator_from_triplets(
            thermal_strain_rows,
            thermal_strain_cols,
            thermal_strain_vals,
            shape=(nrow, int(ntemp)),
            dtype=dtype,
        ),
        thermal_stress_operator=_coo_operator_from_triplets(
            thermal_stress_rows,
            thermal_stress_cols,
            thermal_stress_vals,
            shape=(nrow, int(ntemp)),
            dtype=dtype,
        ),
        thermal_strain_constant=np.asarray(thermal_strain_constant, dtype=dtype),
        thermal_stress_constant=np.asarray(thermal_stress_constant, dtype=dtype),
        nq_per_element=int(nq_per_element),
        ndof=int(ndof),
        ntemp=int(ntemp),
    )


def _assemble_quadrature_field_operators_python(
    nodes: npt.NDArray[np.floating[Any]],
    elements: npt.NDArray[np.uint64],
    material_ids: npt.NDArray[np.uint64],
    material_table: npt.NDArray[np.floating[Any]],
    quadrature_code: int,
    dtype: np.dtype[Any],
    element_type: str,
) -> QuadratureFieldOperators:
    nelem = elements.shape[0]
    ndof = 2 * nodes.shape[0]
    q1d = _gauss_1d(quadrature_code)
    nq = len(q1d) ** 2
    points = np.zeros((nelem, nq, 2), dtype=dtype)
    strain_rows: list[int] = []
    strain_cols: list[int] = []
    strain_vals: list[Any] = []
    stress_rows: list[int] = []
    stress_cols: list[int] = []
    stress_vals: list[Any] = []

    for element_index, conn in enumerate(elements):
        coords = nodes[conn]
        material = material_table[int(material_ids[element_index])]
        local_dofs = np.empty((2 * conn.shape[0],), dtype=np.int64)
        for local_node, global_node in enumerate(conn):
            global_node_int = int(global_node)
            local_dofs[2 * local_node] = 2 * global_node_int
            local_dofs[2 * local_node + 1] = 2 * global_node_int + 1

        for q_local, (n, grad_phys, _det_j, point, _weight) in enumerate(
            _volume_samples(coords, element_type, quadrature_code, dtype)
        ):
            b = _axisymmetric_b_matrix(n, grad_phys, float(point[0]), dtype)
            db = np.asarray(material @ b, dtype=dtype)
            points[element_index, q_local] = point
            row_base = 4 * (element_index * nq + q_local)
            for component in range(4):
                global_row = row_base + component
                for local_dof, global_col in enumerate(local_dofs):
                    strain_value = b[component, local_dof]
                    if strain_value != 0.0:
                        strain_rows.append(global_row)
                        strain_cols.append(int(global_col))
                        strain_vals.append(strain_value)
                    stress_value = db[component, local_dof]
                    if stress_value != 0.0:
                        stress_rows.append(global_row)
                        stress_cols.append(int(global_col))
                        stress_vals.append(stress_value)

    nrow = nelem * nq * 4
    return QuadratureFieldOperators(
        points_rz=points,
        strain_operator=_coo_operator_from_triplets(
            strain_rows,
            strain_cols,
            strain_vals,
            shape=(nrow, ndof),
            dtype=dtype,
        ),
        stress_operator=_coo_operator_from_triplets(
            stress_rows,
            stress_cols,
            stress_vals,
            shape=(nrow, ndof),
            dtype=dtype,
        ),
        thermal_strain_operator=sp.csr_matrix((nrow, 0), dtype=dtype),
        thermal_stress_operator=sp.csr_matrix((nrow, 0), dtype=dtype),
        thermal_strain_constant=np.zeros((nrow,), dtype=dtype),
        thermal_stress_constant=np.zeros((nrow,), dtype=dtype),
        nq_per_element=nq,
        ndof=ndof,
        ntemp=0,
    )
def assemble_axisymmetric(
    nodes: ArrayLike,
    elements: ArrayLike,
    material_ids: ArrayLike,
    material_table: ArrayLike | Mapping[int, ArrayLike],
    body_force: ArrayLike,
    pressure_faces: ArrayLike | None = None,
    pressure_values: ArrayLike | None = None,
    traction_faces: ArrayLike | None = None,
    traction_values: ArrayLike | None = None,
    thermal_material_table: ArrayLike | Mapping[int, ArrayLike] | None = None,
    nodal_temperature: ArrayLike | None = None,
    quadrature: str | int = "gl3",
    element_type: str = "quad4",
) -> AssemblyResult:
    """
    Assemble the global axisymmetric elasticity system in COO form.

    The assembled element matrix uses the standard axisymmetric weak form
    `K_e = integral(B^T D B 2*pi*r dA)`; see [1]-[3] in the module references.
    The right-hand side is built by applying the sparse body-force, pressure,
    traction, and thermal operators. Traction values are specified in global
    `(r, z)` components.
    """

    dtype = _resolve_float_dtype(
        nodes,
        material_table,
        body_force,
        pressure_values,
        traction_values,
        thermal_material_table,
        nodal_temperature,
    )
    nodes_arr = _normalize_nodes(nodes, dtype)
    elements_arr = _normalize_elements(elements)
    material_ids_arr, material_table_arr = _normalize_materials(material_ids, material_table, dtype)
    _thermal_ids_arr, thermal_material_table_arr = _normalize_thermal_material_table(
        material_ids,
        thermal_material_table,
        dtype,
        require_mapping=isinstance(material_table, Mapping) if thermal_material_table is not None else None,
    )
    if material_ids_arr.shape[0] != elements_arr.shape[0]:
        raise ValueError(
            f"material_ids has length {material_ids_arr.shape[0]}, "
            f"but elements has {elements_arr.shape[0]} rows"
        )
    if thermal_material_table_arr is not None:
        if nodal_temperature is None:
            raise ValueError("nodal_temperature must be provided when thermal_material_table is provided")
    elif nodal_temperature is not None:
        raise ValueError("thermal_material_table must be provided when nodal_temperature is provided")
    body_force_arr = _normalize_body_force(body_force, elements_arr.shape[0], dtype)
    pressure_faces_arr, pressure_values_arr = _normalize_pressure_loads(
        pressure_faces, pressure_values, dtype
    )
    traction_faces_arr, traction_values_arr = _normalize_traction_loads(
        traction_faces, traction_values, dtype
    )
    quadrature_code = _quadrature_code(quadrature)
    normalized_element_type = _normalize_element_type(element_type)
    _validate_element_quadrature_combo(normalized_element_type, quadrature_code)
    analysis_nodes, analysis_elements, elevated = _analysis_mesh_for_element_type(
        nodes_arr, elements_arr, normalized_element_type
    )
    rows, cols, vals, ndof = _assemble_stiffness_rust(
        analysis_nodes,
        analysis_elements,
        material_ids_arr,
        material_table_arr,
        quadrature_code,
        dtype,
        normalized_element_type,
    )
    body_force_to_rhs = _assemble_body_force_operator(
        analysis_nodes,
        analysis_elements,
        quadrature_code,
        dtype,
        normalized_element_type,
    )
    pressure_to_rhs = _assemble_pressure_operator(
        analysis_nodes,
        analysis_elements,
        pressure_faces_arr,
        quadrature_code,
        dtype,
        normalized_element_type,
    )
    traction_to_rhs = _assemble_traction_operator(
        analysis_nodes,
        analysis_elements,
        traction_faces_arr,
        quadrature_code,
        dtype,
        normalized_element_type,
    )
    analysis_temperature_to_rhs, thermal_reference_rhs = _assemble_temperature_operator(
        analysis_nodes,
        analysis_elements,
        material_ids_arr,
        material_table_arr,
        thermal_material_table_arr,
        quadrature_code,
        dtype,
        normalized_element_type,
    )
    if thermal_material_table_arr is None:
        temperature_to_rhs = sp.csr_matrix((ndof, 0), dtype=dtype)
        thermal_reference_rhs = np.zeros((ndof,), dtype=dtype)
        temperature_arr = None
    else:
        temperature_arr = _normalize_nodal_temperature(nodal_temperature, nodes_arr.shape[0], dtype)
        if elevated is None:
            temperature_to_rhs = analysis_temperature_to_rhs
        else:
            temperature_to_rhs = analysis_temperature_to_rhs @ _temperature_elevation_operator(
                elevated, dtype
            )
    rhs = np.asarray(thermal_reference_rhs, dtype=dtype).copy()
    rhs += np.asarray(body_force_to_rhs @ body_force_arr.reshape(-1), dtype=dtype)
    if pressure_values_arr.size:
        rhs += np.asarray(pressure_to_rhs @ pressure_values_arr, dtype=dtype)
    if traction_values_arr.size:
        rhs += np.asarray(traction_to_rhs @ traction_values_arr.reshape(-1), dtype=dtype)
    if temperature_arr is not None:
        rhs += np.asarray(temperature_to_rhs @ temperature_arr, dtype=dtype)
    return AssemblyResult(
        rows=rows,
        cols=cols,
        vals=vals,
        rhs=rhs,
        ndof=ndof,
    )


def assemble_axisymmetric_model(
    nodes: ArrayLike,
    elements: ArrayLike,
    material_ids: ArrayLike,
    material_table: ArrayLike | Mapping[int, ArrayLike],
    pressure_faces: ArrayLike | None = None,
    traction_faces: ArrayLike | None = None,
    thermal_material_table: ArrayLike | Mapping[int, ArrayLike] | None = None,
    quadrature: str | int = "gl3",
    element_type: str = "quad4",
) -> AxisymmetricFEMModel:
    """
    Assemble a reusable axisymmetric FEM model for repeated load cases.

    The stiffness matrix and linear load operators are assembled once through the
    Rust backend. The returned load operators map per-element body-force data and
    reusable pressure-load amplitudes to the global right-hand side through sparse
    matrix-vector products on the Python side.

    Notes:
        `pressure_faces` defines the ordering of the reusable pressure load vector.
        `traction_faces` defines the ordering of the reusable traction vector list.
        Repeated solves may vary `pressure_values` and `traction_values`, but not
        the face lists themselves.
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
    pressure_faces_arr = _normalize_pressure_faces(pressure_faces)
    traction_faces_arr = _normalize_traction_faces(traction_faces)
    quadrature_code = _quadrature_code(quadrature)
    normalized_element_type = _normalize_element_type(element_type)
    _validate_element_quadrature_combo(normalized_element_type, quadrature_code)
    analysis_nodes, analysis_elements, elevated = _analysis_mesh_for_element_type(
        nodes_arr, elements_arr, normalized_element_type
    )
    rows, cols, vals, ndof = _assemble_stiffness_rust(
        analysis_nodes,
        analysis_elements,
        material_ids_arr,
        material_table_arr,
        quadrature_code,
        dtype,
        normalized_element_type,
    )
    stiffness = _coo_operator_from_triplets(rows, cols, vals, shape=(ndof, ndof), dtype=dtype)
    body_force_to_rhs = _assemble_body_force_operator(
        analysis_nodes,
        analysis_elements,
        quadrature_code,
        dtype,
        normalized_element_type,
    )
    pressure_to_rhs = _assemble_pressure_operator(
        analysis_nodes,
        analysis_elements,
        pressure_faces_arr,
        quadrature_code,
        dtype,
        normalized_element_type,
    )
    traction_to_rhs = _assemble_traction_operator(
        analysis_nodes,
        analysis_elements,
        traction_faces_arr,
        quadrature_code,
        dtype,
        normalized_element_type,
    )
    analysis_temperature_to_rhs, thermal_reference_rhs = _assemble_temperature_operator(
        analysis_nodes,
        analysis_elements,
        material_ids_arr,
        material_table_arr,
        thermal_material_table_arr,
        quadrature_code,
        dtype,
        normalized_element_type,
    )
    if thermal_material_table_arr is None:
        temperature_to_rhs = sp.csr_matrix((ndof, 0), dtype=dtype)
        thermal_reference_rhs = np.zeros((ndof,), dtype=dtype)
        n_temperature_nodes = 0
    else:
        if elevated is None:
            temperature_to_rhs = analysis_temperature_to_rhs
        else:
            temperature_to_rhs = analysis_temperature_to_rhs @ _temperature_elevation_operator(
                elevated, dtype
            )
        n_temperature_nodes = nodes_arr.shape[0]

    return AxisymmetricFEMModel(
        stiffness=stiffness,
        body_force_to_rhs=body_force_to_rhs,
        pressure_to_rhs=pressure_to_rhs,
        traction_to_rhs=traction_to_rhs,
        temperature_to_rhs=temperature_to_rhs.tocsr(),
        thermal_reference_rhs=np.asarray(thermal_reference_rhs, dtype=dtype),
        pressure_faces=pressure_faces_arr,
        traction_faces=traction_faces_arr,
        analysis_nodes=analysis_nodes,
        analysis_elements=analysis_elements,
        element_type=normalized_element_type,
        ndof=ndof,
        nelem=elements_arr.shape[0],
        n_temperature_nodes=n_temperature_nodes,
        dtype=dtype,
    )


def quadrature_field_operators_axisymmetric(
    nodes: ArrayLike,
    elements: ArrayLike,
    material_ids: ArrayLike,
    material_table: ArrayLike | Mapping[int, ArrayLike],
    thermal_material_table: ArrayLike | Mapping[int, ArrayLike] | None = None,
    quadrature: str | int = "gl3",
    element_type: str = "quad4",
) -> QuadratureFieldOperators:
    """
    Build sparse operators that map nodal displacements to quadrature strain/stress samples.

    The returned operators act on the global displacement vector ordered as
    `[u_r(0), u_z(0), u_r(1), u_z(1), ...]` and produce quadrature samples stacked in
    element-major order with four consecutive rows per sample:
    `[e_rr, e_zz, e_tt, g_rz]` for `strain_operator` and
    `[s_rr, s_zz, s_tt, t_rz]` for `stress_operator`.
    """

    dtype = _resolve_float_dtype(nodes, material_table, thermal_material_table)
    nodes_arr = _normalize_nodes(nodes, dtype)
    elements_arr = _normalize_elements(elements)
    material_ids_arr, material_table_arr = _normalize_materials(material_ids, material_table, dtype)
    _thermal_ids_arr, thermal_material_table_arr = _normalize_thermal_material_table(
        material_ids,
        thermal_material_table,
        dtype,
        require_mapping=isinstance(material_table, Mapping) if thermal_material_table is not None else None,
    )
    if material_ids_arr.shape[0] != elements_arr.shape[0]:
        raise ValueError(
            f"material_ids has length {material_ids_arr.shape[0]}, "
            f"but elements has {elements_arr.shape[0]} rows"
        )
    quadrature_code = _quadrature_code(quadrature)
    normalized_element_type = _normalize_element_type(element_type)
    _validate_element_quadrature_combo(normalized_element_type, quadrature_code)
    analysis_nodes, analysis_elements, _elevated = _analysis_mesh_for_element_type(
        nodes_arr, elements_arr, normalized_element_type
    )
    return _quadrature_field_operators_rust(
        analysis_nodes,
        analysis_elements,
        material_ids_arr,
        material_table_arr,
        thermal_material_table_arr,
        quadrature_code,
        dtype,
        normalized_element_type,
    )


def element_measures_axisymmetric(
    nodes: ArrayLike,
    elements: ArrayLike,
    quadrature: str | int = "gl3",
    element_type: str = "quad4",
) -> ElementMeasures:
    """Return per-element meridian areas and swept axisymmetric volumes."""

    dtype = _resolve_float_dtype(nodes)
    nodes_arr = _normalize_nodes(nodes, dtype)
    elements_arr = _normalize_elements(elements)
    quadrature_code = _quadrature_code(quadrature)
    normalized_element_type = _normalize_element_type(element_type)
    _validate_element_quadrature_combo(normalized_element_type, quadrature_code)
    analysis_nodes, analysis_elements, _elevated = _analysis_mesh_for_element_type(
        nodes_arr, elements_arr, normalized_element_type
    )
    low_level = _dispatch_pair(
        dtype,
        _dispatch_by_element_type(
            normalized_element_type,
            _element_measures_axisymmetric_quad4_f32,
            _element_measures_axisymmetric_quad9_f32,
        ),
        _dispatch_by_element_type(
            normalized_element_type,
            _element_measures_axisymmetric_quad4_f64,
            _element_measures_axisymmetric_quad9_f64,
        ),
    )
    areas, swept_volumes = low_level(analysis_nodes, analysis_elements, quadrature_code)
    return ElementMeasures(
        areas=np.asarray(areas, dtype=dtype),
        swept_volumes=np.asarray(swept_volumes, dtype=dtype),
    )


def element_quadrature_axisymmetric(
    nodes: ArrayLike,
    elements: ArrayLike,
    quadrature: str | int = "gl3",
    element_type: str = "quad4",
) -> ElementQuadrature:
    """Return physical quadrature points and mapped area/volume weights per element."""

    dtype = _resolve_float_dtype(nodes)
    nodes_arr = _normalize_nodes(nodes, dtype)
    elements_arr = _normalize_elements(elements)
    quadrature_code = _quadrature_code(quadrature)
    normalized_element_type = _normalize_element_type(element_type)
    _validate_element_quadrature_combo(normalized_element_type, quadrature_code)
    analysis_nodes, analysis_elements, _elevated = _analysis_mesh_for_element_type(
        nodes_arr, elements_arr, normalized_element_type
    )
    low_level = _dispatch_pair(
        dtype,
        _dispatch_by_element_type(
            normalized_element_type,
            _element_quadrature_axisymmetric_quad4_f32,
            _element_quadrature_axisymmetric_quad9_f32,
        ),
        _dispatch_by_element_type(
            normalized_element_type,
            _element_quadrature_axisymmetric_quad4_f64,
            _element_quadrature_axisymmetric_quad9_f64,
        ),
    )
    points_flat, weights_area, weights_volume, nq_per_element = low_level(
        analysis_nodes,
        analysis_elements,
        quadrature_code,
    )
    nelem = analysis_elements.shape[0]
    points_rz = np.asarray(points_flat, dtype=dtype).reshape(nelem, nq_per_element, 2)
    weights_area_arr = np.asarray(weights_area, dtype=dtype).reshape(nelem, nq_per_element)
    weights_volume_arr = np.asarray(weights_volume, dtype=dtype).reshape(nelem, nq_per_element)
    return ElementQuadrature(
        points_rz=points_rz,
        weights_area=weights_area_arr,
        weights_volume=weights_volume_arr,
        nq_per_element=int(nq_per_element),
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


def apply_dirichlet(
    matrix: sp.spmatrix,
    rhs: ArrayLike,
    prescribed: Mapping[int, float] | None = None,
) -> ReducedSystem:
    """Eliminate prescribed degrees of freedom by free-dof partitioning."""

    csr = matrix.tocsr()
    rhs_arr = np.asarray(rhs, dtype=csr.dtype).reshape(-1)
    if csr.shape[0] != csr.shape[1]:
        raise ValueError(f"matrix must be square; got {csr.shape}")
    if rhs_arr.shape[0] != csr.shape[0]:
        raise ValueError(f"rhs length {rhs_arr.shape[0]} does not match matrix size {csr.shape[0]}")
    prescribed = {} if prescribed is None else dict(prescribed)
    if prescribed:
        fixed_dofs = np.asarray(sorted(int(dof) for dof in prescribed), dtype=np.int64)
        fixed_values = np.asarray([prescribed[int(dof)] for dof in fixed_dofs], dtype=rhs_arr.dtype)
    else:
        fixed_dofs = np.zeros((0,), dtype=np.int64)
        fixed_values = np.zeros((0,), dtype=rhs_arr.dtype)
    if fixed_dofs.size and ((fixed_dofs < 0).any() or (fixed_dofs >= csr.shape[0]).any()):
        raise ValueError("prescribed DOF index is out of bounds")
    free_dofs = np.setdiff1d(np.arange(csr.shape[0], dtype=np.int64), fixed_dofs, assume_unique=True)
    reduced_rhs = rhs_arr[free_dofs].copy()
    if fixed_dofs.size:
        reduced_rhs -= csr[free_dofs][:, fixed_dofs] @ fixed_values
    reduced_matrix = csr[free_dofs][:, free_dofs].tocsr()
    return ReducedSystem(
        matrix=reduced_matrix,
        rhs=reduced_rhs,
        free_dofs=free_dofs,
        fixed_dofs=fixed_dofs,
        fixed_values=fixed_values,
        ndof=csr.shape[0],
    )


def solve_dirichlet(
    matrix: sp.spmatrix,
    rhs: ArrayLike,
    prescribed: Mapping[int, float] | None = None,
    solver: Any | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """Solve the reduced sparse system after applying prescribed Dirichlet values."""

    reduced = apply_dirichlet(matrix, rhs, prescribed)
    if reduced.matrix.shape[0] == 0:
        return reduced.recover(np.zeros((0,), dtype=reduced.rhs.dtype))
    if solver is None:
        free_solution = spla.factorized(reduced.matrix.tocsc())(reduced.rhs)
    else:
        free_solution = solver(reduced.matrix, reduced.rhs)
    return reduced.recover(free_solution)


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


def evaluate_axisymmetric_strain_stress_at_quadrature(
    nodes: ArrayLike,
    elements: ArrayLike,
    material_ids: ArrayLike,
    material_table: ArrayLike | Mapping[int, ArrayLike],
    displacements: ArrayLike,
    thermal_material_table: ArrayLike | Mapping[int, ArrayLike] | None = None,
    nodal_temperature: ArrayLike | None = None,
    quadrature: str | int = "gl3",
    element_type: str = "quad4",
) -> QuadratureFieldSamples:
    """
    Recover strain and stress at element quadrature points from nodal displacements.

    The recovery uses the same family-specific shape functions, Jacobian map, and
    axisymmetric `B` matrix used by the assembler, so the samples align directly
    with the discrete formulation from [1]-[3].
    """

    dtype = _resolve_float_dtype(nodes, material_table, displacements, nodal_temperature)
    operators = quadrature_field_operators_axisymmetric(
        nodes,
        elements,
        material_ids,
        material_table,
        thermal_material_table=thermal_material_table,
        quadrature=quadrature,
        element_type=element_type,
    )
    displacements_arr = _normalize_displacements(displacements, operators.ndof // 2, dtype)
    u_flat = displacements_arr.reshape(-1)
    if operators.ntemp == 0:
        temperature_flat = np.zeros((0,), dtype=dtype)
    else:
        if nodal_temperature is None:
            raise ValueError("nodal_temperature is required because thermal_material_table was provided")
        nodes_arr = _normalize_nodes(nodes, dtype)
        elements_arr = _normalize_elements(elements)
        normalized_element_type = _normalize_element_type(element_type)
        _analysis_nodes, _analysis_elements, elevated = _analysis_mesh_for_element_type(
            nodes_arr,
            elements_arr,
            normalized_element_type,
        )
        temperature_flat = _analysis_temperature_for_element_type(
            nodal_temperature,
            nodes_arr.shape[0],
            normalized_element_type,
            elevated,
            dtype,
        )
    nelem = operators.points_rz.shape[0]
    nq = operators.nq_per_element
    strain = np.asarray(operators.strain_operator @ u_flat, dtype=dtype).reshape(nelem, nq, 4)
    thermal_strain = (
        np.asarray(operators.thermal_strain_operator @ temperature_flat, dtype=dtype)
        + operators.thermal_strain_constant
    ).reshape(nelem, nq, 4)
    stress = (
        np.asarray(operators.stress_operator @ u_flat, dtype=dtype)
        - (
            np.asarray(operators.thermal_stress_operator @ temperature_flat, dtype=dtype)
            + operators.thermal_stress_constant
        )
    ).reshape(nelem, nq, 4)
    elastic_strain = strain - thermal_strain
    return QuadratureFieldSamples(
        points_rz=operators.points_rz,
        strain=strain,
        thermal_strain=thermal_strain,
        elastic_strain=elastic_strain,
        stress=stress,
    )


__all__ = [
    "AssemblyResult",
    "AxisymmetricFEMModel",
    "ElevatedQuad9Mesh",
    "ElementMeasures",
    "ElementQuadrature",
    "QuadratureFieldOperators",
    "QuadratureFieldSamples",
    "ReducedAxisymmetricFEMModel",
    "ReducedSystem",
    "apply_dirichlet",
    "assemble_axisymmetric",
    "assemble_axisymmetric_model",
    "cfsem_radial_material",
    "element_measures_axisymmetric",
    "element_quadrature_axisymmetric",
    "evaluate_axisymmetric_strain_stress_at_quadrature",
    "infer_quad9_mesh",
    "isotropic_axisymmetric_material",
    "isotropic_axisymmetric_thermal_material",
    "orthotropic_axisymmetric_thermal_material",
    "quadrature_field_operators_axisymmetric",
    "solve_dirichlet",
]
