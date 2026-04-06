"""
2D-axisymmetric elasticity finite-element assembly for solenoid stress problems.

This module provides a small displacement-based axisymmetric finite-element solver
for the `(r, z)` meridian plane.  The Rust backend assembles the global COO stiffness
matrix and load vector, while Python handles sparse linear algebra, postprocessing,
and validation workflows.

The element formulation follows the standard small-strain Galerkin construction

`K_e = integral(B^T D B 2*pi*r dA)`

with consistent body-force and surface-pressure load vectors.  The axisymmetric
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

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from cfsem.cfsem import (
    solenoid_stress_fem_assemble_axisymmetric_quad4_f32 as _assemble_axisymmetric_quad4_f32,
)
from cfsem.cfsem import (
    solenoid_stress_fem_assemble_axisymmetric_quad4_f64 as _assemble_axisymmetric_quad4_f64,
)
from cfsem.cfsem import (
    solenoid_stress_fem_element_measures_axisymmetric_quad4_f32 as _element_measures_axisymmetric_quad4_f32,
)
from cfsem.cfsem import (
    solenoid_stress_fem_element_measures_axisymmetric_quad4_f64 as _element_measures_axisymmetric_quad4_f64,
)
from cfsem.cfsem import (
    solenoid_stress_fem_element_quadrature_axisymmetric_quad4_f32 as _element_quadrature_axisymmetric_quad4_f32,
)
from cfsem.cfsem import (
    solenoid_stress_fem_element_quadrature_axisymmetric_quad4_f64 as _element_quadrature_axisymmetric_quad4_f64,
)

ArrayLike = npt.ArrayLike


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
class QuadratureFieldSamples:
    """Recovered strain and stress at element quadrature points."""

    points_rz: npt.NDArray[np.floating[Any]]
    strain: npt.NDArray[np.floating[Any]]
    stress: npt.NDArray[np.floating[Any]]


def _quadrature_code(quadrature: str | int) -> int:
    if quadrature in (2, "2", "2x2", "gauss2x2", "Gauss2x2"):
        return 2
    if quadrature in (3, "3", "3x3", "gauss3x3", "Gauss3x3"):
        return 3
    raise ValueError(f"unsupported quadrature {quadrature!r}; use '2x2' or '3x3'")


def _resolve_float_dtype(*values: object) -> np.dtype[np.float32] | np.dtype[np.float64]:
    arrays = [np.asarray(value) for value in values if value is not None and not isinstance(value, Mapping)]
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
            raise ValueError(f"material_ids contains tag {exc.args[0]} that is missing from material_table") from exc
        return normalized_ids, np.ascontiguousarray(np.stack(dense_table, axis=0), dtype=dtype)

    table = np.asarray(material_table, dtype=dtype)
    if table.ndim != 3 or table.shape[1:] != (4, 4):
        raise ValueError(f"material_table must have shape (nmat, 4, 4); got {table.shape}")
    return np.ascontiguousarray(ids), np.ascontiguousarray(table)


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


def _normalize_pressure_loads(
    pressure_faces: ArrayLike | None,
    pressure_values: ArrayLike | None,
    dtype: np.dtype[Any],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.floating[Any]]]:
    if pressure_faces is None and pressure_values is None:
        return (
            np.zeros((0, 2), dtype=np.uint64),
            np.zeros((0,), dtype=dtype),
        )
    if pressure_faces is None or pressure_values is None:
        raise ValueError("pressure_faces and pressure_values must either both be provided or both be omitted")
    faces = np.asarray(pressure_faces, dtype=np.uint64)
    values = np.asarray(pressure_values, dtype=dtype)
    if faces.ndim != 2 or faces.shape[1] != 2:
        raise ValueError(f"pressure_faces must have shape (nload, 2); got {faces.shape}")
    if values.ndim != 1:
        raise ValueError(f"pressure_values must have shape (nload,); got {values.shape}")
    if faces.shape[0] != values.shape[0]:
        raise ValueError(
            f"pressure_faces has {faces.shape[0]} rows, but pressure_values has {values.shape[0]} entries"
        )
    return np.ascontiguousarray(faces), np.ascontiguousarray(values)


def _dispatch_pair(dtype: np.dtype[Any], f32: Any, f64: Any) -> Any:
    if dtype == np.float32:
        return f32
    return f64


def assemble_axisymmetric(
    nodes: ArrayLike,
    elements: ArrayLike,
    material_ids: ArrayLike,
    material_table: ArrayLike | Mapping[int, ArrayLike],
    body_force: ArrayLike,
    pressure_faces: ArrayLike | None = None,
    pressure_values: ArrayLike | None = None,
    quadrature: str | int = "2x2",
) -> AssemblyResult:
    """
    Assemble the global axisymmetric elasticity system in COO form.

    The assembled element matrix uses the standard axisymmetric weak form
    `K_e = integral(B^T D B 2*pi*r dA)`; see [1]-[3] in the module references.
    """

    dtype = _resolve_float_dtype(nodes, body_force, pressure_values)
    nodes_arr = _normalize_nodes(nodes, dtype)
    elements_arr = _normalize_elements(elements)
    material_ids_arr, material_table_arr = _normalize_materials(material_ids, material_table, dtype)
    if material_ids_arr.shape[0] != elements_arr.shape[0]:
        raise ValueError(
            f"material_ids has length {material_ids_arr.shape[0]}, but elements has {elements_arr.shape[0]} rows"
        )
    body_force_arr = _normalize_body_force(body_force, elements_arr.shape[0], dtype)
    pressure_faces_arr, pressure_values_arr = _normalize_pressure_loads(pressure_faces, pressure_values, dtype)
    quadrature_code = _quadrature_code(quadrature)
    low_level = _dispatch_pair(dtype, _assemble_axisymmetric_quad4_f32, _assemble_axisymmetric_quad4_f64)
    rows, cols, vals, rhs, ndof = low_level(
        nodes_arr,
        elements_arr,
        material_ids_arr,
        material_table_arr,
        body_force_arr,
        pressure_faces_arr,
        pressure_values_arr,
        quadrature_code,
    )
    return AssemblyResult(
        rows=np.asarray(rows, dtype=np.int64),
        cols=np.asarray(cols, dtype=np.int64),
        vals=np.asarray(vals, dtype=dtype),
        rhs=np.asarray(rhs, dtype=dtype),
        ndof=int(ndof),
    )


def element_measures_axisymmetric(
    nodes: ArrayLike,
    elements: ArrayLike,
    quadrature: str | int = "2x2",
) -> ElementMeasures:
    """Return per-element meridian areas and swept axisymmetric volumes."""

    dtype = _resolve_float_dtype(nodes)
    nodes_arr = _normalize_nodes(nodes, dtype)
    elements_arr = _normalize_elements(elements)
    low_level = _dispatch_pair(
        dtype,
        _element_measures_axisymmetric_quad4_f32,
        _element_measures_axisymmetric_quad4_f64,
    )
    areas, swept_volumes = low_level(nodes_arr, elements_arr, _quadrature_code(quadrature))
    return ElementMeasures(
        areas=np.asarray(areas, dtype=dtype),
        swept_volumes=np.asarray(swept_volumes, dtype=dtype),
    )


def element_quadrature_axisymmetric(
    nodes: ArrayLike,
    elements: ArrayLike,
    quadrature: str | int = "2x2",
) -> ElementQuadrature:
    """Return physical quadrature points and mapped area/volume weights per element."""

    dtype = _resolve_float_dtype(nodes)
    nodes_arr = _normalize_nodes(nodes, dtype)
    elements_arr = _normalize_elements(elements)
    low_level = _dispatch_pair(
        dtype,
        _element_quadrature_axisymmetric_quad4_f32,
        _element_quadrature_axisymmetric_quad4_f64,
    )
    points_flat, weights_area, weights_volume, nq_per_element = low_level(
        nodes_arr,
        elements_arr,
        _quadrature_code(quadrature),
    )
    nelem = elements_arr.shape[0]
    return ElementQuadrature(
        points_rz=np.asarray(points_flat, dtype=dtype).reshape(nelem, nq_per_element, 2),
        weights_area=np.asarray(weights_area, dtype=dtype).reshape(nelem, nq_per_element),
        weights_volume=np.asarray(weights_volume, dtype=dtype).reshape(nelem, nq_per_element),
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
    """Solve the linear system after applying prescribed Dirichlet values."""

    reduced = apply_dirichlet(matrix, rhs, prescribed)
    if reduced.matrix.shape[0] == 0:
        return reduced.recover(np.zeros((0,), dtype=reduced.rhs.dtype))
    if solver is None:
        free_solution = spla.spsolve(reduced.matrix, reduced.rhs)
    else:
        free_solution = solver(reduced.matrix, reduced.rhs)
    return reduced.recover(free_solution)


def _gauss_1d(code: int) -> list[tuple[float, float]]:
    if code == 2:
        a = 1.0 / np.sqrt(3.0)
        return [(-a, 1.0), (a, 1.0)]
    if code == 3:
        a = np.sqrt(3.0 / 5.0)
        return [(-a, 5.0 / 9.0), (0.0, 8.0 / 9.0), (a, 5.0 / 9.0)]
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
    quadrature: str | int = "2x2",
) -> QuadratureFieldSamples:
    """
    Recover strain and stress at element quadrature points from nodal displacements.

    The recovery uses the same `Quad4` shape functions, Jacobian map, and axisymmetric
    `B` matrix used by the assembler, so the samples align directly with the discrete
    formulation from [1]-[3].
    """

    dtype = _resolve_float_dtype(nodes, material_table, displacements)
    nodes_arr = _normalize_nodes(nodes, dtype)
    elements_arr = _normalize_elements(elements)
    material_ids_arr, material_table_arr = _normalize_materials(material_ids, material_table, dtype)
    displacements_arr = _normalize_displacements(displacements, nodes_arr.shape[0], dtype)
    if material_ids_arr.shape[0] != elements_arr.shape[0]:
        raise ValueError(
            f"material_ids has length {material_ids_arr.shape[0]}, but elements has {elements_arr.shape[0]} rows"
        )

    q1d = _gauss_1d(_quadrature_code(quadrature))
    nq = len(q1d) ** 2
    points = np.zeros((elements_arr.shape[0], nq, 2), dtype=dtype)
    strain = np.zeros((elements_arr.shape[0], nq, 4), dtype=dtype)
    stress = np.zeros((elements_arr.shape[0], nq, 4), dtype=dtype)

    q_data = []
    for xi, wx in q1d:
        for eta, wy in q1d:
            q_data.append((xi, eta, wx * wy))

    for element_index, conn in enumerate(elements_arr):
        coords = nodes_arr[conn]
        u_local = displacements_arr[conn].reshape(8)
        material = material_table_arr[int(material_ids_arr[element_index])]
        for q_local, (xi, eta, _) in enumerate(q_data):
            n = _quad4_shape(xi, eta).astype(dtype, copy=False)
            grad_ref = _quad4_grad_ref(xi, eta).astype(dtype, copy=False)
            jac = np.array(
                [
                    [np.dot(coords[:, 0], grad_ref[:, 0]), np.dot(coords[:, 0], grad_ref[:, 1])],
                    [np.dot(coords[:, 1], grad_ref[:, 0]), np.dot(coords[:, 1], grad_ref[:, 1])],
                ],
                dtype=dtype,
            )
            inv_j = np.linalg.inv(jac)
            grad_phys = np.column_stack(
                [
                    inv_j[0, 0] * grad_ref[:, 0] + inv_j[1, 0] * grad_ref[:, 1],
                    inv_j[0, 1] * grad_ref[:, 0] + inv_j[1, 1] * grad_ref[:, 1],
                ]
            )
            point = n @ coords
            radius = float(point[0])
            if radius <= np.finfo(dtype).eps:
                raise ValueError(f"quadrature radius {radius} is too close to zero")
            b = np.zeros((4, 8), dtype=dtype)
            for i in range(4):
                col_r = 2 * i
                col_z = col_r + 1
                b[0, col_r] = grad_phys[i, 0]
                b[1, col_z] = grad_phys[i, 1]
                b[2, col_r] = n[i] / radius
                b[3, col_r] = grad_phys[i, 1]
                b[3, col_z] = grad_phys[i, 0]
            eps_q = b @ u_local
            sig_q = material @ eps_q
            points[element_index, q_local] = point
            strain[element_index, q_local] = eps_q
            stress[element_index, q_local] = sig_q

    return QuadratureFieldSamples(points_rz=points, strain=strain, stress=stress)


__all__ = [
    "AssemblyResult",
    "ElementMeasures",
    "ElementQuadrature",
    "QuadratureFieldSamples",
    "ReducedSystem",
    "apply_dirichlet",
    "assemble_axisymmetric",
    "cfsem_radial_material",
    "element_measures_axisymmetric",
    "element_quadrature_axisymmetric",
    "evaluate_axisymmetric_strain_stress_at_quadrature",
    "isotropic_axisymmetric_material",
    "solve_dirichlet",
]
