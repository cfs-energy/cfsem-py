from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from cfsem.solenoid_stress import axisymmetric_fem as fem
from cfsem.solenoid_stress.axisymmetric_fem import (
    cfsem_radial_material,
    isotropic_axisymmetric_material,
)
from cfsem.solenoid_stress.solenoid_1d import (
    SolenoidStress1D,
    solenoid_1d_structural_factor,
    solenoid_1d_structural_rhs,
)
from cfsem.solenoid_stress.thick_wall_cylinder_handcalc import (
    s_hoop_thick_wall_cylinder,
    s_radial_thick_wall_cylinder,
)


DType = type[np.float32] | type[np.float64]
QUADRATURES = ["gl3", "gl4"]
DTYPES: list[DType] = [np.float32, np.float64]
AREA_VOLUME_MESHES = [(1, 1), (3, 2)]
BODY_FORCE_MESHES = [(2, 1), (4, 2)]
PRESSURE_NR_CASES = [24, 48]
ELEMENT_TYPES = ["quad4", "quad9"]
ValidationError = (AssertionError, ValueError)


def build_annulus_strip_mesh(
    ri: float,
    ro: float,
    height: float,
    nr: int,
    nz: int,
    dtype: DType,
) -> tuple[np.ndarray, np.ndarray]:
    radii = np.linspace(ri, ro, nr + 1, dtype=dtype)
    zs = np.linspace(0.0, height, nz + 1, dtype=dtype)
    nodes = np.array([[r, z] for z in zs for r in radii], dtype=dtype)

    def node_id(i: int, j: int) -> int:
        return j * (nr + 1) + i

    elements = []
    for j in range(nz):
        for i in range(nr):
            elements.append(
                [
                    node_id(i, j),
                    node_id(i + 1, j),
                    node_id(i + 1, j + 1),
                    node_id(i, j + 1),
                ]
            )
    return nodes, np.asarray(elements, dtype=np.uint64)


def pressure_faces_for_strip(nr: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    inner = []
    outer = []
    for j in range(nz):
        element = j * nr
        inner.append([element, 3])
        outer.append([element + nr - 1, 1])
    return np.asarray(inner, dtype=np.uint64), np.asarray(outer, dtype=np.uint64)


def horizontal_faces_for_strip(nr: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    bottom = []
    top = []
    for i in range(nr):
        bottom.append([i, 0])
        top.append([(nz - 1) * nr + i, 2])
    return np.asarray(bottom, dtype=np.uint64), np.asarray(top, dtype=np.uint64)


def prescribed_z_dofs(node_count: int) -> dict[int, float]:
    return {2 * node + 1: 0.0 for node in range(node_count)}


def tolerance(dtype: DType) -> tuple[float, float]:
    if dtype is np.float32:
        return 5.0e-5, 5.0e-6
    return 1.0e-12, 1.0e-12


def solve_with_factorized_model(
    model: fem.AxisymmetricFEMModel,
    rhs: np.ndarray,
) -> np.ndarray:
    if model.stiffness.shape[0] == 0:
        return model.recover_full(np.zeros((0,), dtype=rhs.dtype))
    reduced_solution = spla.factorized(model.stiffness)(rhs)
    return model.recover_full(reduced_solution)


def assemble_model_and_rhs(
    *,
    nodes: np.ndarray,
    elements: np.ndarray,
    material_ids: np.ndarray,
    material_table: np.ndarray | dict[int, np.ndarray],
    body_force: np.ndarray | list[float] | tuple[float, float] | None = None,
    pressure_faces: np.ndarray | None = None,
    pressure_values: np.ndarray | None = None,
    traction_faces: np.ndarray | None = None,
    traction_values: np.ndarray | None = None,
    thermal_material_table: np.ndarray | dict[int, np.ndarray] | None = None,
    nodal_temperature: np.ndarray | None = None,
    prescribed: dict[int, float] | None = None,
    quadrature: str = "gl3",
    element_type: str = "quad4",
) -> tuple[fem.AxisymmetricFEMModel, np.ndarray]:
    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=material_ids,
        material_table=material_table,
        pressure_faces=pressure_faces,
        traction_faces=traction_faces,
        thermal_material_table=thermal_material_table,
        prescribed=prescribed,
        quadrature=quadrature,
        element_type=element_type,
    )
    rhs = model.build_rhs(
        body_force=body_force,
        pressure_values=pressure_values,
        traction_values=traction_values,
        nodal_temperature=nodal_temperature,
    )
    return model, rhs


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: dtype.__name__)
@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize(("nr", "nz"), AREA_VOLUME_MESHES, ids=["single", "refined"])
def test_element_measures_and_quadrature_match_exact_cylindrical_shell_values(
    dtype: DType,
    quadrature: str,
    nr: int,
    nz: int,
) -> None:
    nodes, elements = build_annulus_strip_mesh(1.0, 2.0, 1.0, nr=nr, nz=nz, dtype=dtype)
    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray(
            [isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)]
        ),
        quadrature=quadrature,
    )
    measures = model.element_measures()
    quadrature_data = model.element_quadrature()

    expected_area = dtype(1.0)
    expected_volume = dtype(np.pi * (2.0**2 - 1.0**2))
    rtol, atol = tolerance(dtype)

    assert np.all(measures.areas > 0.0)
    assert np.all(measures.swept_volumes > 0.0)
    assert np.allclose(measures.areas.sum(), expected_area, rtol=rtol, atol=atol)
    assert np.allclose(measures.swept_volumes.sum(), expected_volume, rtol=rtol, atol=atol)
    assert np.allclose(
        quadrature_data.weights_area.sum(axis=1),
        measures.areas,
        rtol=rtol,
        atol=atol,
    )
    assert np.allclose(
        quadrature_data.weights_volume.sum(axis=1),
        measures.swept_volumes,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: dtype.__name__)
@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize(("nr", "nz"), BODY_FORCE_MESHES, ids=["coarse", "refined"])
def test_body_force_total_matches_requested_total_force(
    dtype: DType,
    quadrature: str,
    nr: int,
    nz: int,
) -> None:
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=nr, nz=nz, dtype=dtype)
    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray(
            [isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)]
        ),
        quadrature=quadrature,
    )
    measures = model.element_measures()
    total_force = np.array([1234.0, -432.0], dtype=dtype)
    density = total_force / measures.swept_volumes.sum()

    model, rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)]),
        body_force=density,
        quadrature=quadrature,
    )

    assert model.stiffness.shape == (model.ndof_reduced, model.ndof_reduced)
    rhs = rhs.reshape(-1, 2)
    rtol, atol = tolerance(dtype)
    assert np.allclose(rhs[:, 0].sum(), total_force[0], rtol=rtol, atol=max(atol, 1.0e-4))
    assert np.allclose(rhs[:, 1].sum(), total_force[1], rtol=rtol, atol=max(atol, 1.0e-4))


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: dtype.__name__)
@pytest.mark.parametrize("quadrature", QUADRATURES)
def test_axisymmetric_model_rhs_matches_operator_sum(dtype: DType, quadrature: str) -> None:
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=3, nz=2, dtype=dtype)
    inner_faces, outer_faces = pressure_faces_for_strip(nr=3, nz=2)
    pressure_faces = np.vstack([inner_faces, outer_faces])
    pressure_values = np.linspace(1.0e5, 6.0e5, pressure_faces.shape[0], dtype=dtype)
    body_force = np.column_stack(
        [
            np.linspace(-3.0e4, 7.0e4, elements.shape[0], dtype=dtype),
            np.linspace(4.0e4, -5.0e4, elements.shape[0], dtype=dtype),
        ]
    )
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)

    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        pressure_faces=pressure_faces,
        quadrature=quadrature,
    )
    rhs = model.build_rhs(body_force=body_force, pressure_values=pressure_values)
    body_force_arr = fem._normalize_body_force(body_force, elements.shape[0], np.dtype(dtype))
    pressure_arr = fem._normalize_pressure_values(pressure_values, pressure_faces.shape[0], np.dtype(dtype))

    expected_rhs = np.asarray(model.constant_rhs, dtype=dtype).copy()
    expected_rhs += np.asarray(model.body_force_to_rhs @ body_force_arr.reshape(-1), dtype=dtype)
    expected_rhs += np.asarray(model.pressure_to_rhs @ pressure_arr, dtype=dtype)

    assert np.allclose(rhs, expected_rhs)


def test_model_dtype_resolution_includes_material_tables() -> None:
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=2, nz=1, dtype=np.float32)
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=np.float64)
    thermal_material = fem.isotropic_axisymmetric_thermal_material(
        1.1e-5,
        reference_temperature=293.15,
        dtype=np.float64,
    )
    nodal_temperature = np.linspace(294.0, 301.0, nodes.shape[0], dtype=np.float32)

    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        thermal_material_table=np.asarray([thermal_material]),
    )

    assert model.dtype == np.dtype(np.float64)
    assert model.stiffness.dtype == np.float64
    assert model.build_rhs(
        body_force=np.array([0.0, 0.0], dtype=np.float32),
        nodal_temperature=nodal_temperature,
    ).dtype == np.float64


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: dtype.__name__)
@pytest.mark.parametrize("quadrature", QUADRATURES)
def test_axisymmetric_model_reuses_factorization_across_load_cases(dtype: DType, quadrature: str) -> None:
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=4, nz=2, dtype=dtype)
    inner_faces, outer_faces = pressure_faces_for_strip(nr=4, nz=2)
    pressure_faces = np.vstack([inner_faces, outer_faces])
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)
    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        pressure_faces=pressure_faces,
        prescribed=prescribed_z_dofs(nodes.shape[0]),
        quadrature=quadrature,
    )
    solve_free = spla.factorized(model.stiffness)

    cases = [
        (
            np.column_stack(
                [
                    np.linspace(0.0, 2.0e4, elements.shape[0], dtype=dtype),
                    np.linspace(-1.0e4, 1.0e4, elements.shape[0], dtype=dtype),
                ]
            ),
            np.linspace(2.0e5, 5.0e5, pressure_faces.shape[0], dtype=dtype),
        ),
        (
            np.column_stack(
                [
                    np.linspace(-3.0e4, 1.0e4, elements.shape[0], dtype=dtype),
                    np.linspace(2.5e4, -2.5e4, elements.shape[0], dtype=dtype),
                ]
            ),
            np.linspace(-1.5e5, 3.5e5, pressure_faces.shape[0], dtype=dtype),
        ),
    ]

    rtol, atol = tolerance(dtype)
    for body_force, pressure_values in cases:
        rhs = model.build_rhs(body_force=body_force, pressure_values=pressure_values)
        expected = model.solve(rhs)
        actual = model.recover_full(solve_free(rhs))
        assert np.allclose(actual, expected, rtol=rtol, atol=max(atol, 1.0e-6))


@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
def test_traction_model_rhs_matches_operator_sum(quadrature: str, element_type: str) -> None:
    dtype = np.float64
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=3, nz=2, dtype=dtype)
    inner_faces, outer_faces = pressure_faces_for_strip(nr=3, nz=2)
    _bottom_faces, top_faces = horizontal_faces_for_strip(nr=3, nz=2)
    pressure_faces = outer_faces
    traction_faces = np.vstack([inner_faces, top_faces])
    pressure_values = np.linspace(1.0e5, 2.5e5, pressure_faces.shape[0], dtype=dtype)
    traction_values = np.column_stack(
        [
            np.linspace(-2.0e5, 1.0e5, traction_faces.shape[0], dtype=dtype),
            np.linspace(3.0e5, -1.0e5, traction_faces.shape[0], dtype=dtype),
        ]
    )
    body_force = np.column_stack(
        [
            np.linspace(-3.0e4, 7.0e4, elements.shape[0], dtype=dtype),
            np.linspace(4.0e4, -5.0e4, elements.shape[0], dtype=dtype),
        ]
    )
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)

    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        pressure_faces=pressure_faces,
        traction_faces=traction_faces,
        quadrature=quadrature,
        element_type=element_type,
    )
    rhs = model.build_rhs(
        body_force=body_force,
        pressure_values=pressure_values,
        traction_values=traction_values,
    )
    body_force_arr = fem._normalize_body_force(body_force, elements.shape[0], np.dtype(dtype))
    pressure_arr = fem._normalize_pressure_values(pressure_values, pressure_faces.shape[0], np.dtype(dtype))
    traction_arr = fem._normalize_traction_values(traction_values, traction_faces.shape[0], np.dtype(dtype))
    expected_rhs = np.asarray(model.thermal_reference_rhs, dtype=dtype).copy()
    expected_rhs += np.asarray(model.body_force_to_rhs @ body_force_arr.reshape(-1), dtype=dtype)
    expected_rhs += np.asarray(model.pressure_to_rhs @ pressure_arr, dtype=dtype)
    expected_rhs += np.asarray(model.traction_to_rhs @ traction_arr.reshape(-1), dtype=dtype)
    assert np.allclose(rhs, expected_rhs)


@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
def test_pressure_and_traction_superpose_linearly(quadrature: str, element_type: str) -> None:
    dtype = np.float64
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=2, nz=1, dtype=dtype)
    _inner_faces, outer_faces = pressure_faces_for_strip(nr=2, nz=1)
    _bottom_faces, top_faces = horizontal_faces_for_strip(nr=2, nz=1)
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)
    pressure_values = np.full(outer_faces.shape[0], 2.0e5, dtype=dtype)
    traction_values = np.column_stack(
        [
            np.full(top_faces.shape[0], 1.2e5, dtype=dtype),
            np.full(top_faces.shape[0], -0.8e5, dtype=dtype),
        ]
    )

    pressure_model, pressure_rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=np.array([0.0, 0.0], dtype=dtype),
        pressure_faces=outer_faces,
        pressure_values=pressure_values,
        quadrature=quadrature,
        element_type=element_type,
    )
    traction_model, traction_rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=np.array([0.0, 0.0], dtype=dtype),
        traction_faces=top_faces,
        traction_values=traction_values,
        quadrature=quadrature,
        element_type=element_type,
    )
    combined_model, combined_rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=np.array([0.0, 0.0], dtype=dtype),
        pressure_faces=outer_faces,
        pressure_values=pressure_values,
        traction_faces=top_faces,
        traction_values=traction_values,
        quadrature=quadrature,
        element_type=element_type,
    )

    assert np.allclose(combined_model.stiffness.toarray(), pressure_model.stiffness.toarray())
    assert np.allclose(combined_model.stiffness.toarray(), traction_model.stiffness.toarray())
    assert np.allclose(combined_rhs, pressure_rhs + traction_rhs)


@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
def test_radial_traction_on_outer_face_matches_expected_total_force(
    quadrature: str,
    element_type: str,
) -> None:
    dtype = np.float64
    ri, ro, height = 0.5, 1.0, 0.2
    nodes, elements = build_annulus_strip_mesh(ri, ro, height, nr=2, nz=1, dtype=dtype)
    _inner_faces, outer_faces = pressure_faces_for_strip(nr=2, nz=1)
    traction = 3.5e5
    model, rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)]),
        body_force=np.array([0.0, 0.0], dtype=dtype),
        traction_faces=outer_faces,
        traction_values=np.array([traction, 0.0], dtype=dtype),
        quadrature=quadrature,
        element_type=element_type,
    )

    assert model.stiffness.shape == (model.ndof, model.ndof)
    rhs = rhs.reshape(-1, 2)
    expected_force = traction * 2.0 * np.pi * ro * height
    assert np.allclose(rhs[:, 1], 0.0)
    assert np.isclose(rhs[:, 0].sum(), expected_force)


@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
def test_axial_traction_on_top_face_matches_expected_total_force(quadrature: str, element_type: str) -> None:
    dtype = np.float64
    ri, ro, height = 0.5, 1.0, 0.2
    nodes, elements = build_annulus_strip_mesh(ri, ro, height, nr=2, nz=1, dtype=dtype)
    _bottom_faces, top_faces = horizontal_faces_for_strip(nr=2, nz=1)
    traction = -4.0e5
    model, rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)]),
        body_force=np.array([0.0, 0.0], dtype=dtype),
        traction_faces=top_faces,
        traction_values=np.array([0.0, traction], dtype=dtype),
        quadrature=quadrature,
        element_type=element_type,
    )

    assert model.stiffness.shape == (model.ndof, model.ndof)
    rhs = rhs.reshape(-1, 2)
    expected_force = traction * np.pi * (ro**2 - ri**2)
    assert np.allclose(rhs[:, 0], 0.0)
    assert np.isclose(rhs[:, 1].sum(), expected_force)


@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
def test_pressure_matches_equivalent_normal_traction_on_straight_faces(
    quadrature: str,
    element_type: str,
) -> None:
    dtype = np.float64
    pressure = 2.5e5
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=2, nz=1, dtype=dtype)
    _inner_faces, outer_faces = pressure_faces_for_strip(nr=2, nz=1)
    _bottom_faces, top_faces = horizontal_faces_for_strip(nr=2, nz=1)
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)

    _outer_pressure_model, outer_pressure_rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=np.array([0.0, 0.0], dtype=dtype),
        pressure_faces=outer_faces,
        pressure_values=np.full(outer_faces.shape[0], pressure, dtype=dtype),
        quadrature=quadrature,
        element_type=element_type,
    )
    _outer_traction_model, outer_traction_rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=np.array([0.0, 0.0], dtype=dtype),
        traction_faces=outer_faces,
        traction_values=np.array([-pressure, 0.0], dtype=dtype),
        quadrature=quadrature,
        element_type=element_type,
    )
    _top_pressure_model, top_pressure_rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=np.array([0.0, 0.0], dtype=dtype),
        pressure_faces=top_faces,
        pressure_values=np.full(top_faces.shape[0], pressure, dtype=dtype),
        quadrature=quadrature,
        element_type=element_type,
    )
    _top_traction_model, top_traction_rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=np.array([0.0, 0.0], dtype=dtype),
        traction_faces=top_faces,
        traction_values=np.array([0.0, -pressure], dtype=dtype),
        quadrature=quadrature,
        element_type=element_type,
    )

    assert np.allclose(outer_pressure_rhs, outer_traction_rhs)
    assert np.allclose(top_pressure_rhs, top_traction_rhs)


@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
def test_zero_surface_traction_matches_natural_free_boundary(
    quadrature: str,
    element_type: str,
) -> None:
    dtype = np.float64
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=3, nz=2, dtype=dtype)
    inner_faces, outer_faces = pressure_faces_for_strip(nr=3, nz=2)
    bottom_faces, top_faces = horizontal_faces_for_strip(nr=3, nz=2)
    boundary_faces = np.vstack([inner_faces, outer_faces, bottom_faces, top_faces])
    zero_traction = np.zeros((boundary_faces.shape[0], 2), dtype=dtype)
    body_force = np.column_stack(
        [
            np.linspace(2.0e4, 6.0e4, elements.shape[0], dtype=dtype),
            np.linspace(-1.5e4, 1.5e4, elements.shape[0], dtype=dtype),
        ]
    )
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)
    prescribed = {1: 0.0}

    free_model, free_rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=body_force,
        prescribed=prescribed,
        quadrature=quadrature,
        element_type=element_type,
    )
    zero_traction_model, zero_traction_rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=body_force,
        traction_faces=boundary_faces,
        traction_values=zero_traction,
        prescribed=prescribed,
        quadrature=quadrature,
        element_type=element_type,
    )

    assert np.allclose(free_model.stiffness.toarray(), zero_traction_model.stiffness.toarray())
    assert np.allclose(free_rhs, zero_traction_rhs)
    assert np.allclose(free_model.solve(free_rhs), zero_traction_model.solve(zero_traction_rhs))


@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
def test_factorized_solve_reuses_stiffness_with_varying_traction(
    quadrature: str, element_type: str
) -> None:
    dtype = np.float64
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=3, nz=2, dtype=dtype)
    _inner_faces, outer_faces = pressure_faces_for_strip(nr=3, nz=2)
    _bottom_faces, top_faces = horizontal_faces_for_strip(nr=3, nz=2)
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)
    prescribed = prescribed_z_dofs(nodes.shape[0] if element_type == "quad4" else fem.infer_quad9_mesh(nodes, elements).analysis_nodes.shape[0])
    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        pressure_faces=outer_faces,
        traction_faces=top_faces,
        prescribed=prescribed,
        quadrature=quadrature,
        element_type=element_type,
    )
    solve_free = spla.factorized(model.stiffness)

    cases = [
        (
            np.linspace(1.0e5, 2.0e5, outer_faces.shape[0], dtype=dtype),
            np.column_stack(
                [
                    np.linspace(0.0, 1.5e5, top_faces.shape[0], dtype=dtype),
                    np.linspace(-2.0e5, -1.0e5, top_faces.shape[0], dtype=dtype),
                ]
            ),
        ),
        (
            np.linspace(-5.0e4, 1.0e5, outer_faces.shape[0], dtype=dtype),
            np.column_stack(
                [
                    np.linspace(0.5e5, -0.5e5, top_faces.shape[0], dtype=dtype),
                    np.linspace(1.2e5, -0.3e5, top_faces.shape[0], dtype=dtype),
                ]
            ),
        ),
    ]

    for pressure_values, traction_values in cases:
        rhs = model.build_rhs(
            body_force=np.array([0.0, 0.0], dtype=dtype),
            pressure_values=pressure_values,
            traction_values=traction_values,
        )
        expected = model.solve(rhs)
        actual = model.recover_full(solve_free(rhs))
        assert np.allclose(actual, expected)


@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
def test_thermal_model_rhs_matches_operator_sum(
    quadrature: str,
    element_type: str,
) -> None:
    dtype = np.float64
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=3, nz=2, dtype=dtype)
    inner_faces, outer_faces = pressure_faces_for_strip(nr=3, nz=2)
    pressure_faces = np.vstack([inner_faces, outer_faces])
    pressure_values = np.linspace(1.0e5, 6.0e5, pressure_faces.shape[0], dtype=dtype)
    body_force = np.column_stack(
        [
            np.linspace(-3.0e4, 7.0e4, elements.shape[0], dtype=dtype),
            np.linspace(4.0e4, -5.0e4, elements.shape[0], dtype=dtype),
        ]
    )
    material_table = np.asarray(
        [
            isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype),
            isotropic_axisymmetric_material(180.0e9, 0.24, dtype=dtype),
        ]
    )
    thermal_material_table = np.asarray(
        [
            fem.isotropic_axisymmetric_thermal_material(1.2e-5, reference_temperature=293.15, dtype=dtype),
            fem.isotropic_axisymmetric_thermal_material(0.8e-5, reference_temperature=301.15, dtype=dtype),
        ]
    )
    material_ids = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.uint64)
    radial_span = np.ptp(nodes[:, 0])
    axial_span = np.ptp(nodes[:, 1])
    nodal_temperature = np.asarray(
        296.15
        + 12.0 * (nodes[:, 0] - nodes[:, 0].min()) / radial_span
        + 4.0 * (nodes[:, 1] - nodes[:, 1].min()) / max(axial_span, 1.0e-12),
        dtype=dtype,
    )

    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=material_ids,
        material_table=material_table,
        pressure_faces=pressure_faces,
        thermal_material_table=thermal_material_table,
        quadrature=quadrature,
        element_type=element_type,
    )
    rhs = model.build_rhs(
        body_force=body_force,
        pressure_values=pressure_values,
        nodal_temperature=nodal_temperature,
    )
    body_force_arr = fem._normalize_body_force(body_force, elements.shape[0], np.dtype(dtype))
    pressure_arr = fem._normalize_pressure_values(pressure_values, pressure_faces.shape[0], np.dtype(dtype))
    temperature_arr = fem._normalize_nodal_temperature(nodal_temperature, nodes.shape[0], np.dtype(dtype))
    expected_rhs = np.asarray(model.thermal_reference_rhs, dtype=dtype).copy()
    expected_rhs += np.asarray(model.body_force_to_rhs @ body_force_arr.reshape(-1), dtype=dtype)
    expected_rhs += np.asarray(model.pressure_to_rhs @ pressure_arr, dtype=dtype)
    expected_rhs += np.asarray(model.temperature_to_rhs @ temperature_arr, dtype=dtype)
    assert np.allclose(rhs, expected_rhs)


@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
def test_uniform_temperature_recovery_matches_fully_constrained_thermal_stress(
    quadrature: str,
    element_type: str,
) -> None:
    dtype = np.float64
    alpha = 1.1e-5
    reference_temperature = 293.15
    delta_temperature = 40.0
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=2, nz=1, dtype=dtype)
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)
    thermal_material = fem.isotropic_axisymmetric_thermal_material(
        alpha,
        reference_temperature=reference_temperature,
        dtype=dtype,
    )
    nodal_temperature = np.full(nodes.shape[0], reference_temperature + delta_temperature, dtype=dtype)
    model, rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=np.array([0.0, 0.0], dtype=dtype),
        thermal_material_table=np.asarray([thermal_material]),
        nodal_temperature=nodal_temperature,
        prescribed={dof: 0.0 for dof in range(2 * model.analysis_nodes.shape[0])} if False else None,
        quadrature=quadrature,
        element_type=element_type,
    )
    all_fixed = {dof: 0.0 for dof in range(2 * model.analysis_nodes.shape[0])}
    model, rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=np.array([0.0, 0.0], dtype=dtype),
        thermal_material_table=np.asarray([thermal_material]),
        nodal_temperature=nodal_temperature,
        prescribed=all_fixed,
        quadrature=quadrature,
        element_type=element_type,
    )
    displacement = model.recover_full(np.zeros((0,), dtype=dtype))
    samples = model.evaluate_quadrature(displacement, nodal_temperature=nodal_temperature)

    expected_thermal_strain = np.asarray([alpha, alpha, alpha, 0.0], dtype=dtype) * delta_temperature
    expected_stress = -(material @ expected_thermal_strain)
    expected_thermal_strain_grid = np.broadcast_to(expected_thermal_strain, samples.total_strain.shape)
    expected_stress_grid = np.broadcast_to(expected_stress, samples.stress.shape)

    assert np.allclose(displacement, 0.0)
    assert np.allclose(samples.total_strain, 0.0)
    assert np.allclose(samples.thermal_strain, expected_thermal_strain_grid)
    assert np.allclose(samples.elastic_strain, -expected_thermal_strain_grid)
    assert np.allclose(samples.stress, expected_stress_grid)


@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
def test_quadrature_recovery_splits_total_elastic_and_thermal_strain_consistently(
    quadrature: str,
    element_type: str,
) -> None:
    dtype = np.float64
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=2, nz=1, dtype=dtype)
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)
    thermal_material = fem.isotropic_axisymmetric_thermal_material(
        9.0e-6,
        reference_temperature=290.0,
        dtype=dtype,
    )
    analysis_nnode = (
        nodes.shape[0]
        if element_type == "quad4"
        else fem.infer_quad9_mesh(nodes, elements).analysis_nodes.shape[0]
    )
    displacement = np.linspace(-2.0e-4, 3.0e-4, 2 * analysis_nnode, dtype=dtype)
    nodal_temperature = np.linspace(292.0, 307.0, nodes.shape[0], dtype=dtype)

    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        thermal_material_table=np.asarray([thermal_material]),
        quadrature=quadrature,
        element_type=element_type,
    )
    samples = model.evaluate_quadrature(displacement, nodal_temperature=nodal_temperature)

    assert np.allclose(samples.total_strain, samples.elastic_strain + samples.thermal_strain)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: dtype.__name__)
@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("nr", PRESSURE_NR_CASES, ids=["coarse", "refined"])
def test_pressure_vessel_stresses_match_lame_reference(dtype: DType, quadrature: str, nr: int) -> None:
    ri, ro = 0.5, 1.0
    pin, pout = 2.0e6, 0.4e6
    nodes, elements = build_annulus_strip_mesh(ri, ro, height=0.1, nr=nr, nz=1, dtype=dtype)
    inner_faces, outer_faces = pressure_faces_for_strip(nr=nr, nz=1)
    pressure_faces = np.vstack([inner_faces, outer_faces])
    pressure_values = np.concatenate(
        [
            np.full(inner_faces.shape[0], pin, dtype=dtype),
            np.full(outer_faces.shape[0], pout, dtype=dtype),
        ]
    )

    material = cfsem_radial_material(200.0e9, 0.27, dtype=dtype)
    prescribed = prescribed_z_dofs(nodes.shape[0])
    model_fe, rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=np.array([0.0, 0.0], dtype=dtype),
        pressure_faces=pressure_faces,
        pressure_values=pressure_values,
        prescribed=prescribed,
        quadrature=quadrature,
    )
    displacement = solve_with_factorized_model(model_fe, rhs)
    samples = model_fe.evaluate_quadrature(displacement)
    radii = samples.points_rz[..., 0]
    radial_exact = s_radial_thick_wall_cylinder(radii, ri, ro, pin, pout)
    hoop_exact = s_hoop_thick_wall_cylinder(radii, ri, ro, pin, pout)

    if nr <= 24 and dtype is np.float32:
        radial_rtol, hoop_rtol, atol = 7.0e-2, 6.0e-2, 1.8e4
    elif nr <= 24:
        radial_rtol, hoop_rtol, atol = 5.0e-2, 4.0e-2, 1.0e4
    elif dtype is np.float32:
        radial_rtol, hoop_rtol, atol = 3.0e-2, 2.5e-2, 8.0e3
    else:
        radial_rtol, hoop_rtol, atol = 2.5e-2, 2.0e-2, 2.5e3
    assert np.allclose(samples.stress[..., 0], radial_exact, rtol=radial_rtol, atol=atol)
    assert np.allclose(samples.stress[..., 2], hoop_exact, rtol=hoop_rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: dtype.__name__)
@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("nr", PRESSURE_NR_CASES, ids=["coarse", "refined"])
def test_pressure_vessel_radial_displacement_matches_cfsem_1d_solver(
    dtype: DType,
    quadrature: str,
    nr: int,
) -> None:
    ri, ro = 0.5, 1.0
    pin, pout = 1.5e6, 0.2e6
    nodes, elements = build_annulus_strip_mesh(ri, ro, height=0.1, nr=nr, nz=1, dtype=dtype)
    inner_faces, outer_faces = pressure_faces_for_strip(nr=nr, nz=1)
    material = cfsem_radial_material(200.0e9, 0.27, dtype=dtype)
    model_fe, rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=np.array([0.0, 0.0], dtype=dtype),
        pressure_faces=np.vstack([inner_faces, outer_faces]),
        pressure_values=np.concatenate(
            [
                np.full(inner_faces.shape[0], pin, dtype=dtype),
                np.full(outer_faces.shape[0], pout, dtype=dtype),
            ]
        ),
        prescribed=prescribed_z_dofs(nodes.shape[0]),
        quadrature=quadrature,
    )
    displacement = solve_with_factorized_model(model_fe, rhs).reshape(model_fe.analysis_nodes.shape[0], 2)

    bottom = displacement[: nr + 1, 0]
    top = displacement[nr + 1 :, 0]
    radial_fe = 0.5 * (bottom + top)

    r_actual = nodes[: nr + 1, 0]
    nudge = 1.0e-6
    rgrid = np.concatenate([[ri - nudge], r_actual, [ro + nudge]])
    zeros = np.zeros_like(rgrid)
    c = solenoid_1d_structural_factor(200.0e9, 0.27)
    rhs = solenoid_1d_structural_rhs(c, zeros, zeros, pin, pout)
    model = SolenoidStress1D(
        rgrid=rgrid,
        elasticity_modulus=200.0e9,
        poisson_ratio=0.27,
        direct_inverse=False,
    )
    radial_cfsem = model.displacement_solver(rhs)[1:-1]

    if nr <= 24 and dtype is np.float32:
        rtol, atol = 7.0e-2, 2.5e-6
    elif nr <= 24:
        rtol, atol = 5.0e-2, 8.0e-7
    elif dtype is np.float32:
        rtol, atol = 4.0e-2, 1.0e-6
    else:
        rtol, atol = 3.0e-2, 2.0e-7
    assert np.allclose(radial_fe, radial_cfsem, rtol=rtol, atol=atol)


def test_axisymmetric_fem_helper_validation_branches() -> None:
    material = isotropic_axisymmetric_material(200.0e9, 0.27)

    assert fem._quadrature_code("gl3") == 3
    assert fem._quadrature_code("gl4") == 4
    with pytest.raises(ValueError, match="unsupported quadrature"):
        fem._quadrature_code("2x2")
    with pytest.raises(ValueError, match="unsupported quadrature"):
        fem._quadrature_code("bad")

    with pytest.raises(ValueError, match="missing from material_table"):
        fem._normalize_materials(
            np.array([1], dtype=np.uint64),
            {0: material},
            np.dtype(np.float64),
        )

    normalized_ids, dense_table = fem._normalize_materials(
        np.array([4, 2, 4], dtype=np.uint64),
        {2: material, 4: 2.0 * material},
        np.dtype(np.float64),
    )
    assert np.array_equal(normalized_ids, np.array([1, 0, 1], dtype=np.uint64))
    assert dense_table.shape == (2, 4, 4)
    assert np.allclose(dense_table[0], material)
    assert np.allclose(dense_table[1], 2.0 * material)

    with pytest.raises(ValidationError, match="material_table must have shape"):
        fem._normalize_materials(
            np.zeros((1,), dtype=np.uint64),
            np.zeros((1, 3, 3)),
            np.dtype(np.float64),
        )

    body_force = fem._normalize_body_force(np.array([1.0, 2.0]), 3, np.dtype(np.float64))
    assert body_force.shape == (3, 2)

    traction = fem._normalize_traction_values(np.array([1.0, 2.0]), 3, np.dtype(np.float64))
    assert traction.shape == (3, 2)

    assert len(fem._gauss_1d(3)) == 3
    assert len(fem._gauss_1d(4)) == 4
    with pytest.raises(ValueError, match="unsupported quadrature code"):
        fem._gauss_1d(2)


def test_model_recovery_and_fixed_dof_branches() -> None:
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=1, nz=1, dtype=np.float64)
    material = isotropic_axisymmetric_material(200.0e9, 0.27)
    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        prescribed={0: 1.25, 1: -0.5},
    )

    assert np.array_equal(model.fixed_dofs, np.array([0, 1], dtype=np.int64))
    assert np.allclose(model.fixed_values, np.array([1.25, -0.5]))

    recovered = model.recover_full(np.zeros((model.ndof_reduced,), dtype=np.float64))
    assert np.allclose(recovered[:2], np.array([1.25, -0.5]))


def test_assembly_and_postprocessing_validation_branches() -> None:
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=1, nz=1, dtype=np.float64)
    material = isotropic_axisymmetric_material(200.0e9, 0.27)

    with pytest.raises(ValueError, match="material_ids has length 0"):
        fem.assemble_axisymmetric(
            nodes=nodes,
            elements=elements,
            material_ids=np.zeros((0,), dtype=np.uint64),
            material_table=np.asarray([material]),
        )

    with pytest.raises(ValueError, match="unsupported quadrature"):
        fem.assemble_axisymmetric(
            nodes=nodes,
            elements=elements,
            material_ids=np.zeros((1,), dtype=np.uint64),
            material_table=np.asarray([material]),
            quadrature="2x2",
        )

    reshaped = fem._normalize_displacements(
        np.zeros((nodes.shape[0], 2)),
        nodes.shape[0],
        np.dtype(np.float64),
    )
    assert reshaped.shape == (nodes.shape[0], 2)
    with pytest.raises(ValueError, match="displacements must have shape"):
        fem._normalize_displacements(np.zeros((nodes.shape[0], 3)), nodes.shape[0], np.dtype(np.float64))

    near_axis_nodes = np.array(
        [
            [0.0, 0.0],
            [1.0e-8, 0.0],
            [1.0e-8, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="too close to zero"):
        model = fem.assemble_axisymmetric(
            near_axis_nodes,
            np.array([[0, 1, 2, 3]], dtype=np.uint64),
            np.zeros((1,), dtype=np.uint64),
            np.asarray([isotropic_axisymmetric_material(200.0e9, 0.27, dtype=np.float32)]),
        )
        model.element_quadrature()


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: dtype.__name__)
def test_quad4_quadrature_field_operators_match_manual_recovery(dtype: DType) -> None:
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=2, nz=1, dtype=dtype)
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)
    material_ids = np.zeros(elements.shape[0], dtype=np.uint64)
    displacement = np.linspace(-2.0e-4, 3.0e-4, 2 * nodes.shape[0], dtype=dtype)
    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=material_ids,
        material_table=np.asarray([material]),
        quadrature="gl3",
        element_type="quad4",
    )

    dtype_res = np.dtype(dtype)
    nodes_arr = fem._normalize_nodes(nodes, dtype_res)
    elements_arr = fem._normalize_elements(elements)
    material_ids_arr, material_table_arr = fem._normalize_materials(
        material_ids,
        np.asarray([material]),
        dtype_res,
    )
    displacements_arr = fem._normalize_displacements(displacement, nodes.shape[0], dtype_res)
    manual_points: list[np.ndarray] = []
    manual_strain: list[np.ndarray] = []
    manual_stress: list[np.ndarray] = []
    for element_index, conn in enumerate(elements_arr):
        coords = nodes_arr[conn]
        u_local = displacements_arr[conn].reshape(-1)
        material_local = material_table_arr[int(material_ids_arr[element_index])]
        for n, grad_phys, _det_j, point, _weight in fem._volume_samples(coords, "quad4", 3, dtype_res):
            b = fem._axisymmetric_b_matrix(n, grad_phys, float(point[0]), dtype_res)
            eps = b @ u_local
            sig = material_local @ eps
            manual_points.append(np.asarray(point, dtype=dtype_res))
            manual_strain.append(np.asarray(eps, dtype=dtype_res))
            manual_stress.append(np.asarray(sig, dtype=dtype_res))

    actual_strain = np.asarray(model.strain_operator @ displacement, dtype=dtype_res).reshape(-1, 4)
    actual_stress = np.asarray(model.stress_operator @ displacement, dtype=dtype_res).reshape(-1, 4)
    expected_points = np.asarray(manual_points, dtype=dtype_res).reshape(elements.shape[0], 9, 2)
    expected_strain = np.asarray(manual_strain, dtype=dtype_res)
    expected_stress = np.asarray(manual_stress, dtype=dtype_res)
    rtol, atol = tolerance(dtype)

    assert model.quadrature_points_rz.shape == (elements.shape[0], 9, 2)
    assert model.strain_operator.shape == (elements.shape[0] * 9 * 4, displacement.size)
    assert model.stress_operator.shape == (elements.shape[0] * 9 * 4, displacement.size)
    assert np.allclose(model.quadrature_points_rz, expected_points, rtol=rtol, atol=atol)
    assert np.allclose(actual_strain, expected_strain, rtol=rtol, atol=max(atol, 1.0e-9))
    assert np.allclose(actual_stress, expected_stress, rtol=max(rtol, 2.0e-6), atol=max(atol, 1.0e-2))


def test_infer_quad9_mesh_preserves_corner_nodes_and_shares_edge_midpoints() -> None:
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=2, nz=1, dtype=np.float64)
    elevated = fem.infer_quad9_mesh(nodes, elements)

    assert elevated.input_elements.shape == (2, 4)
    assert elevated.analysis_elements.shape == (2, 9)
    assert np.array_equal(elevated.corner_node_indices, np.arange(nodes.shape[0], dtype=np.int64))
    assert elevated.midside_node_indices.size == 7
    assert elevated.center_node_indices.size == 2
    assert elevated.analysis_elements[0, 5] == elevated.analysis_elements[1, 7]

    shared_midpoint_index = int(elevated.analysis_elements[0, 5])
    shared_midpoint = elevated.analysis_nodes[shared_midpoint_index]
    expected_midpoint = 0.5 * (nodes[1] + nodes[4])
    assert np.allclose(shared_midpoint, expected_midpoint)


@pytest.mark.parametrize("quadrature", QUADRATURES)
def test_quad9_measures_match_quad4_for_inferred_geometry(quadrature: str) -> None:
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=3, nz=2, dtype=np.float64)
    model_quad4 = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([isotropic_axisymmetric_material(200.0e9, 0.27)]),
        quadrature=quadrature,
        element_type="quad4",
    )
    model_quad9 = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([isotropic_axisymmetric_material(200.0e9, 0.27)]),
        quadrature=quadrature,
        element_type="quad9",
    )
    measures_quad4 = model_quad4.element_measures()
    measures_quad9 = model_quad9.element_measures()
    quadrature_quad9 = model_quad9.element_quadrature()

    assert np.allclose(measures_quad9.areas, measures_quad4.areas)
    assert np.allclose(measures_quad9.swept_volumes, measures_quad4.swept_volumes)
    assert np.allclose(quadrature_quad9.weights_area.sum(axis=1), measures_quad9.areas)
    assert np.allclose(quadrature_quad9.weights_volume.sum(axis=1), measures_quad9.swept_volumes)


@pytest.mark.parametrize("quadrature", QUADRATURES)
def test_quad9_model_rhs_matches_operator_sum(quadrature: str) -> None:
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=3, nz=2, dtype=np.float64)
    inner_faces, outer_faces = pressure_faces_for_strip(nr=3, nz=2)
    pressure_faces = np.vstack([inner_faces, outer_faces])
    pressure_values = np.linspace(1.0e5, 6.0e5, pressure_faces.shape[0], dtype=np.float64)
    body_force = np.column_stack(
        [
            np.linspace(-3.0e4, 7.0e4, elements.shape[0], dtype=np.float64),
            np.linspace(4.0e4, -5.0e4, elements.shape[0], dtype=np.float64),
        ]
    )
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=np.float64)

    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        pressure_faces=pressure_faces,
        quadrature=quadrature,
        element_type="quad9",
    )
    assert model.analysis_elements.shape[1] == 9
    assert model.analysis_nodes.shape[0] > nodes.shape[0]
    assert model.ndof == 2 * model.analysis_nodes.shape[0]
    rhs = model.build_rhs(body_force=body_force, pressure_values=pressure_values)
    expected_rhs = np.asarray(model.thermal_reference_rhs, dtype=np.float64).copy()
    expected_rhs += np.asarray(
        model.body_force_to_rhs
        @ fem._normalize_body_force(body_force, elements.shape[0], np.dtype(np.float64)).reshape(-1),
        dtype=np.float64,
    )
    expected_rhs += np.asarray(
        model.pressure_to_rhs
        @ fem._normalize_pressure_values(pressure_values, pressure_faces.shape[0], np.dtype(np.float64)),
        dtype=np.float64,
    )
    assert np.allclose(rhs, expected_rhs)


@pytest.mark.parametrize("quadrature", QUADRATURES)
def test_quad9_quadrature_recovery_shapes(quadrature: str) -> None:
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=2, nz=1, dtype=np.float64)
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=np.float64)
    model, rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=np.array([0.0, 0.0], dtype=np.float64),
        prescribed=prescribed_z_dofs(
            fem.infer_quad9_mesh(nodes, elements).analysis_nodes.shape[0]
        ),
        quadrature=quadrature,
        element_type="quad9",
    )
    displacement = solve_with_factorized_model(model, rhs)
    samples = model.evaluate_quadrature(displacement)

    nq = 9 if quadrature == "gl3" else 16
    assert samples.points_rz.shape == (elements.shape[0], nq, 2)
    assert samples.strain.shape == (elements.shape[0], nq, 4)
    assert samples.stress.shape == (elements.shape[0], nq, 4)


@pytest.mark.parametrize("quadrature", QUADRATURES)
def test_quad9_quadrature_field_operator_shapes(quadrature: str) -> None:
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=2, nz=1, dtype=np.float64)
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=np.float64)
    elevated = fem.infer_quad9_mesh(nodes, elements)
    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        quadrature=quadrature,
        element_type="quad9",
    )

    nq = 9 if quadrature == "gl3" else 16
    assert model.quadrature_points_rz.shape == (elements.shape[0], nq, 2)
    assert model.ndof == 2 * elevated.analysis_nodes.shape[0]
    assert model.strain_operator.shape == (elements.shape[0] * nq * 4, model.ndof)
    assert model.stress_operator.shape == (elements.shape[0] * nq * 4, model.ndof)


def test_model_zero_load_and_empty_reduction_branches() -> None:
    dtype = np.float64
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=1, nz=1, dtype=dtype)
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)

    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
    )
    rhs = model.build_rhs(body_force=np.array([0.0, 0.0], dtype=dtype))
    assert np.allclose(rhs, 0.0)

    displacement = solve_with_factorized_model(model, rhs)
    assert displacement.shape == (model.ndof_full,)

    all_fixed = {dof: 0.0 for dof in range(2 * nodes.shape[0])}
    fixed_model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        prescribed=all_fixed,
    )
    fixed_rhs = fixed_model.build_rhs(body_force=np.array([0.0, 0.0], dtype=dtype))
    assert fixed_model.stiffness.shape == (0, 0)
    assert np.allclose(fixed_model.solve(fixed_rhs), 0.0)


def test_thermal_model_missing_temperature_and_alignment_validation_branches() -> None:
    dtype = np.float64
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=1, nz=1, dtype=dtype)
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)
    thermal_material = fem.isotropic_axisymmetric_thermal_material(1.2e-5, 293.15, dtype=dtype)

    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        thermal_material_table=np.asarray([thermal_material]),
    )
    nodal_temperature = np.full(nodes.shape[0], 300.0, dtype=dtype)
    assert model.build_rhs(nodal_temperature=nodal_temperature).shape == (model.ndof_reduced,)
    with pytest.raises(ValueError, match="nodal_temperature is required"):
        model.build_rhs()

    with pytest.raises(ValueError, match="missing from thermal_material_table"):
        fem.assemble_axisymmetric(
            nodes=nodes,
            elements=elements,
            material_ids=np.array([0], dtype=np.uint64),
            material_table={0: material},
            thermal_material_table={1: thermal_material},
        )

    with pytest.raises(ValueError, match="nodal_temperature is required"):
        zero_displacement = np.zeros((model.ndof_reduced,), dtype=dtype)
        model.evaluate_quadrature(zero_displacement)


def test_private_helper_and_validation_branches_not_hit_by_public_paths() -> None:
    dtype = np.dtype(np.float64)

    assert fem._resolve_float_dtype({"a": np.array([1.0], dtype=np.float64)}) == np.dtype(np.float64)
    empty_elevation = fem._temperature_elevation_operator(None, dtype)
    assert empty_elevation.shape == (0, 0)
    normalized_ids, thermal_table = fem._normalize_thermal_material_table(
        np.array([4, 2, 4], dtype=np.uint64),
        {2: np.array([1.0, 2.0, 3.0, 0.0, 4.0]), 4: np.array([5.0, 6.0, 7.0, 0.0, 8.0])},
        dtype,
    )
    assert np.array_equal(normalized_ids, np.array([1, 0, 1], dtype=np.uint64))
    assert thermal_table.shape == (2, 5)

    assert fem._quad_face_reference(0, 0.25) == (0.25, -1.0, (1.0, 0.0))
    assert fem._quad_face_reference(1, 0.25) == (1.0, 0.25, (0.0, 1.0))
    assert fem._quad_face_reference(2, 0.25) == (-0.25, 1.0, (-1.0, 0.0))
    assert fem._quad_face_reference(3, 0.25) == (-1.0, -0.25, (0.0, -1.0))
    with pytest.raises(ValueError, match="invalid local face"):
        fem._quad_face_reference(4, 0.0)

    shape = fem._quad9_shape(0.0, 0.0)
    grad = fem._quad9_grad_ref(0.0, 0.0)
    assert shape.shape == (9,)
    assert np.isclose(shape.sum(), 1.0)
    assert grad.shape == (9, 2)
    assert fem._element_shape("quad9", 0.0, 0.0).shape == (9,)
    assert fem._element_grad_ref("quad9", 0.0, 0.0).shape == (9, 2)

    ortho = fem.orthotropic_axisymmetric_thermal_material(1.0, 2.0, 3.0, reference_temperature=4.0, dtype=dtype)
    assert np.allclose(ortho, np.array([1.0, 2.0, 3.0, 0.0, 4.0], dtype=dtype))
