from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
import scipy.interpolate as spi
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
from cfsem.solenoid_stress.thermal_handcalc import (
    s_thermal_long_cylinder_linear_temperature,
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


class MultiMaterialCheckerboardCase(NamedTuple):
    nodes: np.ndarray
    elements: np.ndarray
    analysis_nodes: np.ndarray
    material_ids: np.ndarray
    material_table: np.ndarray
    thermal_material_table: np.ndarray
    ri: float
    ro: float
    height: float
    nr: int
    nz: int


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


def distort_annulus_strip_mesh(
    nodes: np.ndarray,
    ri: float,
    ro: float,
    height: float,
    nr: int,
    nz: int,
    distortion: float,
) -> np.ndarray:
    distortion = float(distortion)
    if distortion <= 0.0 or nr <= 0 or nz <= 0:
        return np.asarray(nodes, dtype=np.float64).copy()

    z_min = 0.0
    z_max = height
    width = max(ro - ri, 1.0e-12)
    dr_cell = width / max(nr, 1)
    dz_cell = max(height, 1.0e-12) / max(nz, 1)

    xi = (nodes[:, 0] - ri) / width
    eta = (nodes[:, 1] - z_min) / max(z_max - z_min, 1.0e-12)
    interior_mode = np.sin(np.pi * xi) * np.sin(2.0 * np.pi * eta)

    distorted = np.asarray(nodes, dtype=np.float64).copy()
    distorted[:, 0] += distortion * dr_cell * interior_mode * np.cos(np.pi * xi)
    distorted[:, 1] += distortion * dz_cell * interior_mode * np.sin(np.pi * xi)

    pinned = (
        np.isclose(nodes[:, 0], ri)
        | np.isclose(nodes[:, 0], ro)
        | np.isclose(nodes[:, 1], z_min)
        | np.isclose(nodes[:, 1], z_max)
        | np.isclose(nodes[:, 1], 0.5 * height)
    )
    distorted[pinned] = nodes[pinned]
    return distorted.astype(nodes.dtype, copy=False)


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


def analysis_nodes_for_element_type(
    nodes: np.ndarray,
    elements: np.ndarray,
    element_type: str,
) -> np.ndarray:
    return nodes if element_type == "quad4" else fem.infer_quad9_mesh(nodes, elements).analysis_nodes


def analysis_node_count_for_element_type(
    nodes: np.ndarray,
    elements: np.ndarray,
    element_type: str,
) -> int:
    return int(analysis_nodes_for_element_type(nodes, elements, element_type).shape[0])


def checkerboard_material_ids(nr: int, nz: int) -> np.ndarray:
    element_i = np.tile(np.arange(nr), nz)
    element_j = np.repeat(np.arange(nz), nr)
    return ((element_i + 2 * element_j) % 2).astype(np.uint64)


def build_multimaterial_checkerboard_case(
    *,
    dtype: DType,
    element_type: str,
    thermal_reference_temperatures: tuple[float, float],
    ri: float = 0.5,
    ro: float = 1.0,
    height: float = 0.4,
    nr: int = 5,
    nz: int = 5,
) -> MultiMaterialCheckerboardCase:
    nodes, elements = build_annulus_strip_mesh(ri, ro, height, nr=nr, nz=nz, dtype=dtype)
    material_table = np.asarray(
        [
            isotropic_axisymmetric_material(205.0e9, 0.28, dtype=dtype),
            isotropic_axisymmetric_material(145.0e9, 0.32, dtype=dtype),
        ],
        dtype=dtype,
    )
    thermal_material_table = np.asarray(
        [
            fem.isotropic_axisymmetric_thermal_material(
                1.1e-5, thermal_reference_temperatures[0], dtype=dtype
            ),
            fem.isotropic_axisymmetric_thermal_material(
                1.9e-5, thermal_reference_temperatures[1], dtype=dtype
            ),
        ],
        dtype=dtype,
    )
    return MultiMaterialCheckerboardCase(
        nodes=nodes,
        elements=elements,
        analysis_nodes=analysis_nodes_for_element_type(nodes, elements, element_type),
        material_ids=checkerboard_material_ids(nr, nz),
        material_table=material_table,
        thermal_material_table=thermal_material_table,
        ri=ri,
        ro=ro,
        height=height,
        nr=nr,
        nz=nz,
    )


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


def prescribed_bottom_supports(analysis_nodes: np.ndarray) -> dict[int, float]:
    bottom_z = float(np.min(analysis_nodes[:, 1]))
    bottom_nodes = np.flatnonzero(np.isclose(analysis_nodes[:, 1], bottom_z))
    anchor = int(bottom_nodes[np.argmin(analysis_nodes[bottom_nodes, 0])])
    prescribed = {2 * int(node) + 1: 0.0 for node in bottom_nodes}
    prescribed[2 * anchor] = 0.0
    return prescribed


def interpolate_field(
    points: np.ndarray,
    values: np.ndarray,
    sample_points: np.ndarray,
) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    interpolator = spi.RBFInterpolator(
        np.asarray(points, dtype=np.float64),
        values_arr,
        kernel="linear",
        neighbors=4,
    )
    out = np.asarray(interpolator(np.asarray(sample_points, dtype=np.float64)), dtype=np.float64)
    assert np.all(np.isfinite(out))
    return out


def normalized_peak_error(test: np.ndarray, reference: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(reference))), 1.0e-30)
    return float(np.max(np.abs(np.asarray(test) - np.asarray(reference))) / scale)


def solve_pressurized_region_1d(
    r_actual: np.ndarray,
    elasticity_modulus: float,
    poisson_ratio: float,
    pi: float,
    po: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dr_target = 1.0e-3
    span = float(r_actual[-1] - r_actual[0])
    nr_dense = max(int(np.ceil(span / dr_target)), 1) + 1
    r_dense = np.linspace(float(r_actual[0]), float(r_actual[-1]), nr_dense)
    dr_min = float(np.min(np.diff(r_dense)))
    nudge = min(1.0e-6, 1.0e-3 * dr_min)
    rgrid = np.concatenate([[r_dense[0] - nudge], r_dense, [r_dense[-1] + nudge]])
    zeros = np.zeros_like(rgrid)
    c = solenoid_1d_structural_factor(elasticity_modulus, poisson_ratio)
    rhs = solenoid_1d_structural_rhs(c, zeros, zeros, pi, po)
    model = SolenoidStress1D(
        rgrid=rgrid,
        elasticity_modulus=elasticity_modulus,
        poisson_ratio=poisson_ratio,
        direct_inverse=False,
    )
    displacement = np.asarray(model.displacement_solver(rhs))
    strain = model.operators.a_eu @ displacement
    stress = model.operators.a_se @ strain
    nr = rgrid.shape[0]
    stress_r = np.asarray(stress[:nr])[1:-1]
    stress_t = np.asarray(stress[nr:])[1:-1]
    return rgrid[1:-1], displacement[1:-1], stress_r, stress_t


def solve_two_region_pressure_vessel_1d(
    r_actual: np.ndarray,
    interface_index: int,
    inner_material: tuple[float, float],
    outer_material: tuple[float, float],
    pi: float,
    po: float,
) -> tuple[
    float,
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    r_inner = r_actual[: interface_index + 1]
    r_outer = r_actual[interface_index:]
    e_inner, nu_inner = inner_material
    e_outer, nu_outer = outer_material

    def interface_mismatch(interface_pressure: float) -> float:
        _r0, u_inner, _sr0, _st0 = solve_pressurized_region_1d(
            r_inner, e_inner, nu_inner, pi, interface_pressure
        )
        _r1, u_outer, _sr1, _st1 = solve_pressurized_region_1d(
            r_outer, e_outer, nu_outer, interface_pressure, po
        )
        return float(u_inner[-1] - u_outer[0])

    pressure_probe = max(abs(pi), abs(po), 1.0)
    mismatch_0 = interface_mismatch(0.0)
    mismatch_1 = interface_mismatch(pressure_probe)
    interface_pressure = -mismatch_0 * pressure_probe / (mismatch_1 - mismatch_0)

    inner_solution = solve_pressurized_region_1d(r_inner, e_inner, nu_inner, pi, interface_pressure)
    outer_solution = solve_pressurized_region_1d(r_outer, e_outer, nu_outer, interface_pressure, po)
    return interface_pressure, inner_solution, outer_solution


def piecewise_interp_two_region(
    sample_r: np.ndarray,
    interface_radius: float,
    inner_r: np.ndarray,
    inner_values: np.ndarray,
    outer_r: np.ndarray,
    outer_values: np.ndarray,
) -> np.ndarray:
    sample_r = np.asarray(sample_r, dtype=np.float64)
    out = np.empty_like(sample_r)
    inner_mask = sample_r <= interface_radius
    out[inner_mask] = np.interp(sample_r[inner_mask], inner_r, inner_values)
    out[~inner_mask] = np.interp(sample_r[~inner_mask], outer_r, outer_values)
    return out


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
        material_table=np.asarray([isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)]),
        quadrature=quadrature,
    )
    measures = model.element_measures()
    quadrature_data = model.element_quadrature()
    assert model.element_measures() is measures
    assert model.element_quadrature() is quadrature_data

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
        material_table=np.asarray([isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)]),
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
    assert (
        model.build_rhs(
            body_force=np.array([0.0, 0.0], dtype=np.float32),
            nodal_temperature=nodal_temperature,
        ).dtype
        == np.float64
    )


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
def test_factorized_solve_reuses_stiffness_with_varying_traction(quadrature: str, element_type: str) -> None:
    dtype = np.float64
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=3, nz=2, dtype=dtype)
    _inner_faces, outer_faces = pressure_faces_for_strip(nr=3, nz=2)
    _bottom_faces, top_faces = horizontal_faces_for_strip(nr=3, nz=2)
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)
    prescribed = prescribed_z_dofs(analysis_node_count_for_element_type(nodes, elements, element_type))
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
    all_fixed = {
        dof: 0.0 for dof in range(2 * analysis_node_count_for_element_type(nodes, elements, element_type))
    }
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
    displacement = model.solve(rhs)
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


def test_linear_radial_temperature_long_cylinder_matches_analytic_midplane_stress() -> None:
    dtype = np.float64
    ri = 0.5
    ro = 1.0
    height = 10.0
    nr = 24
    nz = 81
    quadrature = "gl4"
    element_type = "quad9"
    elasticity_modulus = 200.0e9
    poisson_ratio = 0.27
    alpha = 1.2e-5
    reference_temperature = 0.0
    temperature_inner = 80.0
    temperature_outer = 20.0

    nodes, elements = build_annulus_strip_mesh(ri, ro, height, nr=nr, nz=nz, dtype=dtype)
    material = isotropic_axisymmetric_material(
        elasticity_modulus,
        poisson_ratio,
        dtype=dtype,
    )
    thermal_material = fem.isotropic_axisymmetric_thermal_material(
        alpha,
        reference_temperature=reference_temperature,
        dtype=dtype,
    )
    nodal_temperature = temperature_inner + (temperature_outer - temperature_inner) * (
        (nodes[:, 0] - ri) / (ro - ri)
    )

    model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        thermal_material_table=np.asarray([thermal_material]),
        prescribed={1: 0.0},
        quadrature=quadrature,
        element_type=element_type,
    )
    rhs = model.build_rhs(nodal_temperature=nodal_temperature)
    displacement = model.solve(rhs)
    samples = model.evaluate_quadrature(displacement, nodal_temperature=nodal_temperature)

    dz = height / nz
    points = samples.points_rz.reshape(-1, 2)
    stress = samples.stress.reshape(-1, 4)
    center_band = np.abs(points[:, 1] - 0.5 * height) <= 0.5 * dz
    assert np.count_nonzero(center_band) > 0

    radius = points[center_band, 0]
    sigma_rr_ref, sigma_tt_ref, sigma_zz_ref = s_thermal_long_cylinder_linear_temperature(
        radius,
        ri,
        ro,
        elasticity_modulus,
        poisson_ratio,
        alpha,
        temperature_inner,
        temperature_outer,
    )

    sigma_rr = stress[center_band, 0]
    sigma_zz = stress[center_band, 1]
    sigma_tt = stress[center_band, 2]
    tau_rz = stress[center_band, 3]

    assert normalized_peak_error(sigma_rr, sigma_rr_ref) < 1.0e-2
    assert normalized_peak_error(sigma_tt, sigma_tt_ref) < 1.0e-2
    assert normalized_peak_error(sigma_zz, sigma_zz_ref) < 1.0e-2
    assert np.max(np.abs(tau_rz)) < 1.0e-3 * max(
        float(np.max(np.abs(sigma_rr_ref))),
        float(np.max(np.abs(sigma_tt_ref))),
        float(np.max(np.abs(sigma_zz_ref))),
    )


@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
def test_multimaterial_uniform_temperature_recovery_matches_fully_constrained_thermal_stress(
    quadrature: str,
    element_type: str,
) -> None:
    dtype = np.float64
    case = build_multimaterial_checkerboard_case(
        dtype=dtype,
        element_type=element_type,
        thermal_reference_temperatures=(293.15, 315.0),
    )
    nodal_temperature = np.full(case.nodes.shape[0], 333.15, dtype=dtype)
    all_fixed = {dof: 0.0 for dof in range(2 * case.analysis_nodes.shape[0])}
    model, rhs = assemble_model_and_rhs(
        nodes=case.nodes,
        elements=case.elements,
        material_ids=case.material_ids,
        material_table=case.material_table,
        body_force=np.array([0.0, 0.0], dtype=dtype),
        thermal_material_table=case.thermal_material_table,
        nodal_temperature=nodal_temperature,
        prescribed=all_fixed,
        quadrature=quadrature,
        element_type=element_type,
    )

    displacement = model.solve(rhs)
    samples = model.evaluate_quadrature(displacement, nodal_temperature=nodal_temperature)

    expected_thermal_strain = np.zeros_like(samples.thermal_strain)
    expected_stress = np.zeros_like(samples.stress)
    for element_index, material_id in enumerate(case.material_ids):
        thermal_row = case.thermal_material_table[int(material_id)]
        material = case.material_table[int(material_id)]
        delta_temperature = nodal_temperature[0] - thermal_row[4]
        thermal_strain = thermal_row[:4] * delta_temperature
        stress = -(material @ thermal_strain)
        expected_thermal_strain[element_index, :, :] = thermal_strain
        expected_stress[element_index, :, :] = stress

    assert np.allclose(displacement, 0.0)
    assert np.allclose(samples.total_strain, 0.0)
    assert np.allclose(samples.thermal_strain, expected_thermal_strain)
    assert np.allclose(samples.elastic_strain, -expected_thermal_strain)
    assert np.allclose(samples.stress, expected_stress)


@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
def test_multimaterial_loads_superpose_linearly(
    quadrature: str,
    element_type: str,
) -> None:
    dtype = np.float64
    reference_temperature = 300.0
    case = build_multimaterial_checkerboard_case(
        dtype=dtype,
        element_type=element_type,
        thermal_reference_temperatures=(reference_temperature, reference_temperature),
    )

    inner_faces, outer_faces = pressure_faces_for_strip(nr=case.nr, nz=case.nz)
    _bottom_faces, top_faces = horizontal_faces_for_strip(nr=case.nr, nz=case.nz)
    pressure_faces = np.vstack([inner_faces, outer_faces])
    pressure_values = np.concatenate(
        [
            np.linspace(6.0e5, 1.1e6, inner_faces.shape[0], dtype=dtype),
            np.linspace(1.5e5, 4.5e5, outer_faces.shape[0], dtype=dtype),
        ]
    )
    traction_coordinate = np.linspace(0.0, 1.0, top_faces.shape[0], dtype=dtype)
    traction_values = np.column_stack(
        [
            1.2e5 * np.sin(np.pi * traction_coordinate),
            -1.8e5 * (0.6 + 0.4 * np.cos(np.pi * traction_coordinate)),
        ]
    )
    element_i = np.tile(np.arange(case.nr, dtype=dtype), case.nz)
    element_j = np.repeat(np.arange(case.nz, dtype=dtype), case.nr)
    xi = (element_i + 0.5) / case.nr
    eta = (element_j + 0.5) / case.nz
    body_force = np.column_stack(
        [
            2.2e4 * (0.5 + eta),
            -1.6e4 * np.cos(np.pi * xi) * (0.3 + eta),
        ]
    )
    nodal_temperature_ref = np.full(case.nodes.shape[0], reference_temperature, dtype=dtype)
    rfrac = (case.nodes[:, 0] - case.ri) / (case.ro - case.ri)
    zfrac = case.nodes[:, 1] / case.height
    nodal_temperature_hot = reference_temperature + 18.0 * rfrac + 11.0 * zfrac

    model = fem.assemble_axisymmetric(
        nodes=case.nodes,
        elements=case.elements,
        material_ids=case.material_ids,
        material_table=case.material_table,
        pressure_faces=pressure_faces,
        traction_faces=top_faces,
        thermal_material_table=case.thermal_material_table,
        prescribed=prescribed_bottom_supports(case.analysis_nodes),
        quadrature=quadrature,
        element_type=element_type,
    )

    rhs_body = model.build_rhs(body_force=body_force, nodal_temperature=nodal_temperature_ref)
    rhs_pressure = model.build_rhs(pressure_values=pressure_values, nodal_temperature=nodal_temperature_ref)
    rhs_traction = model.build_rhs(traction_values=traction_values, nodal_temperature=nodal_temperature_ref)
    rhs_thermal = model.build_rhs(nodal_temperature=nodal_temperature_hot)
    rhs_combined = model.build_rhs(
        body_force=body_force,
        pressure_values=pressure_values,
        traction_values=traction_values,
        nodal_temperature=nodal_temperature_hot,
    )

    displacement_body = model.solve(rhs_body)
    displacement_pressure = model.solve(rhs_pressure)
    displacement_traction = model.solve(rhs_traction)
    displacement_thermal = model.solve(rhs_thermal)
    displacement_combined = model.solve(rhs_combined)

    samples_body = model.evaluate_quadrature(displacement_body, nodal_temperature=nodal_temperature_ref)
    samples_pressure = model.evaluate_quadrature(
        displacement_pressure, nodal_temperature=nodal_temperature_ref
    )
    samples_traction = model.evaluate_quadrature(
        displacement_traction, nodal_temperature=nodal_temperature_ref
    )
    samples_thermal = model.evaluate_quadrature(displacement_thermal, nodal_temperature=nodal_temperature_hot)
    samples_combined = model.evaluate_quadrature(
        displacement_combined, nodal_temperature=nodal_temperature_hot
    )

    rhs_sum = rhs_body + rhs_pressure + rhs_traction + rhs_thermal
    displacement_sum = (
        displacement_body + displacement_pressure + displacement_traction + displacement_thermal
    )
    strain_sum = (
        samples_body.strain + samples_pressure.strain + samples_traction.strain + samples_thermal.strain
    )
    elastic_strain_sum = (
        samples_body.elastic_strain
        + samples_pressure.elastic_strain
        + samples_traction.elastic_strain
        + samples_thermal.elastic_strain
    )
    stress_sum = (
        samples_body.stress + samples_pressure.stress + samples_traction.stress + samples_thermal.stress
    )

    rhs_atol = 1.0e-10 * float(np.max(np.abs(rhs_sum)))
    displacement_atol = 1.0e-10 * float(np.max(np.abs(displacement_sum)))
    strain_atol = 1.0e-10 * float(np.max(np.abs(strain_sum)))
    elastic_strain_atol = 1.0e-10 * float(np.max(np.abs(elastic_strain_sum)))
    thermal_strain_atol = 1.0e-10 * float(np.max(np.abs(samples_thermal.thermal_strain)))
    stress_atol = 1.0e-10 * float(np.max(np.abs(stress_sum)))

    assert np.allclose(rhs_combined, rhs_sum, rtol=1.0e-10, atol=rhs_atol)
    assert np.allclose(displacement_combined, displacement_sum, rtol=1.0e-10, atol=displacement_atol)
    assert np.allclose(samples_combined.strain, strain_sum, rtol=1.0e-10, atol=strain_atol)
    assert np.allclose(
        samples_combined.elastic_strain,
        elastic_strain_sum,
        rtol=1.0e-10,
        atol=elastic_strain_atol,
    )
    assert np.allclose(
        samples_combined.thermal_strain,
        samples_thermal.thermal_strain,
        rtol=1.0e-10,
        atol=thermal_strain_atol,
    )
    assert np.allclose(samples_combined.stress, stress_sum, rtol=1.0e-10, atol=stress_atol)


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
    analysis_nnode = analysis_node_count_for_element_type(nodes, elements, element_type)
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
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
@pytest.mark.parametrize("nr", PRESSURE_NR_CASES, ids=["coarse", "refined"])
def test_pressure_vessel_stresses_match_lame_reference(
    dtype: DType, quadrature: str, element_type: str, nr: int
) -> None:
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
    prescribed = prescribed_z_dofs(analysis_node_count_for_element_type(nodes, elements, element_type))
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
        element_type=element_type,
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
    # Stress is a recovered field built from displacement gradients, so it
    # carries more numerical error than the displacement comparison below.
    assert np.allclose(samples.stress[..., 0], radial_exact, rtol=radial_rtol, atol=atol)
    assert np.allclose(samples.stress[..., 2], hoop_exact, rtol=hoop_rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: dtype.__name__)
@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
@pytest.mark.parametrize("nr", PRESSURE_NR_CASES, ids=["coarse", "refined"])
def test_pressure_vessel_radial_displacement_matches_cfsem_1d_solver(
    dtype: DType,
    quadrature: str,
    element_type: str,
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
        prescribed=prescribed_z_dofs(analysis_node_count_for_element_type(nodes, elements, element_type)),
        quadrature=quadrature,
        element_type=element_type,
    )
    displacement = solve_with_factorized_model(model_fe, rhs).reshape(model_fe.analysis_nodes.shape[0], 2)
    corner_displacement = displacement[: nodes.shape[0], 0]
    bottom = corner_displacement[: nr + 1]
    top = corner_displacement[nr + 1 :]
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


@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
def test_two_material_pressure_vessel_matches_chained_1d_solver(element_type: str) -> None:
    dtype = np.float64
    ri, rm, ro = 0.5, 0.525, 0.55
    pin, pout = 1.0e7, 2.0e6
    nr = 224
    nodes, elements = build_annulus_strip_mesh(ri, ro, height=0.1, nr=nr, nz=1, dtype=dtype)
    r_actual = nodes[: nr + 1, 0]
    interface_index = int(np.argmin(np.abs(r_actual - rm)))
    interface_radius = float(r_actual[interface_index])

    inner_material = (205.0e9, 0.27)
    outer_material = (135.0e9, 0.31)
    material_table = np.asarray(
        [
            cfsem_radial_material(*inner_material, dtype=dtype),
            cfsem_radial_material(*outer_material, dtype=dtype),
        ]
    )
    material_ids = np.concatenate(
        [
            np.zeros(interface_index, dtype=np.uint64),
            np.ones(nr - interface_index, dtype=np.uint64),
        ]
    )

    inner_faces, outer_faces = pressure_faces_for_strip(nr=nr, nz=1)
    pressure_faces = np.vstack([inner_faces, outer_faces])
    pressure_values = np.concatenate(
        [
            np.full(inner_faces.shape[0], pin, dtype=dtype),
            np.full(outer_faces.shape[0], pout, dtype=dtype),
        ]
    )
    analysis_nnode = analysis_node_count_for_element_type(nodes, elements, element_type)
    model_fe, rhs = assemble_model_and_rhs(
        nodes=nodes,
        elements=elements,
        material_ids=material_ids,
        material_table=material_table,
        body_force=np.array([0.0, 0.0], dtype=dtype),
        pressure_faces=pressure_faces,
        pressure_values=pressure_values,
        prescribed=prescribed_z_dofs(analysis_nnode),
        quadrature="gl4",
        element_type=element_type,
    )
    displacement = solve_with_factorized_model(model_fe, rhs).reshape(model_fe.analysis_nodes.shape[0], 2)
    corner_displacement = displacement[: nodes.shape[0], 0]
    radial_fe = 0.5 * (corner_displacement[: nr + 1] + corner_displacement[nr + 1 :])
    samples = model_fe.evaluate_quadrature(displacement)

    (
        interface_pressure,
        (r_inner, u_inner, sr_inner, st_inner),
        (r_outer, u_outer, sr_outer, st_outer),
    ) = solve_two_region_pressure_vessel_1d(
        r_actual,
        interface_index,
        inner_material,
        outer_material,
        pin,
        pout,
    )

    assert pin >= interface_pressure >= pout

    radial_1d = piecewise_interp_two_region(
        r_actual,
        interface_radius,
        r_inner,
        u_inner,
        r_outer,
        u_outer,
    )
    sample_r = samples.points_rz[..., 0]
    stress_r_1d = piecewise_interp_two_region(
        sample_r,
        interface_radius,
        r_inner,
        sr_inner,
        r_outer,
        sr_outer,
    )
    stress_t_1d = piecewise_interp_two_region(
        sample_r,
        interface_radius,
        r_inner,
        st_inner,
        r_outer,
        st_outer,
    )

    radial_atol = 1.0e-6 * float(np.max(np.abs(radial_1d)))
    stress_r_atol = 1.0e-6 * float(np.max(np.abs(stress_r_1d)))
    stress_t_atol = 1.0e-6 * float(np.max(np.abs(stress_t_1d)))

    assert np.allclose(radial_fe, radial_1d, rtol=1.0e-2, atol=radial_atol)
    # Stress recovery introduces more numerical error than the primary
    # displacement solve, so the stress checks use their own field-scaled
    # absolute tolerances.
    assert np.allclose(samples.stress[..., 0], stress_r_1d, rtol=1.0e-2, atol=stress_r_atol)
    assert np.allclose(samples.stress[..., 2], stress_t_1d, rtol=1.0e-2, atol=stress_t_atol)


@pytest.mark.parametrize("quadrature", QUADRATURES)
@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
def test_distorted_2d_mesh_matches_regular_solution_at_common_points(
    quadrature: str,
    element_type: str,
) -> None:
    dtype = np.float64
    ri, ro, height = 0.5, 1.0, 0.24
    nr, nz = 42, 14
    regular_nodes, elements = build_annulus_strip_mesh(ri, ro, height, nr=nr, nz=nz, dtype=dtype)
    distorted_nodes = distort_annulus_strip_mesh(regular_nodes, ri, ro, height, nr=nr, nz=nz, distortion=0.05)
    inner_faces, outer_faces = pressure_faces_for_strip(nr=nr, nz=nz)
    _bottom_faces, top_faces = horizontal_faces_for_strip(nr=nr, nz=nz)
    pressure_faces = np.vstack([inner_faces, outer_faces])
    pressure_values = np.concatenate(
        [
            np.full(inner_faces.shape[0], 1.2e6, dtype=dtype),
            np.full(outer_faces.shape[0], 0.25e6, dtype=dtype),
        ]
    )
    face_coordinate = np.linspace(0.0, 1.0, top_faces.shape[0], dtype=dtype)
    traction_values = np.column_stack(
        [
            1.8e5 * np.sin(np.pi * face_coordinate),
            -2.2e5 * (0.4 + 0.6 * np.cos(0.5 * np.pi * face_coordinate)),
        ]
    )
    element_i = np.tile(np.arange(nr, dtype=dtype), nz)
    element_j = np.repeat(np.arange(nz, dtype=dtype), nr)
    xi = (element_i + 0.5) / nr
    eta = (element_j + 0.5) / nz
    body_force = np.column_stack(
        [
            4.0e4 * (0.5 + eta),
            -3.0e4 * np.cos(np.pi * xi) * np.sin(np.pi * eta),
        ]
    )
    material = isotropic_axisymmetric_material(205.0e9, 0.29, dtype=dtype)
    regular_analysis_nodes = analysis_nodes_for_element_type(regular_nodes, elements, element_type)
    prescribed = prescribed_bottom_supports(regular_analysis_nodes)

    regular_model, regular_rhs = assemble_model_and_rhs(
        nodes=regular_nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=body_force,
        pressure_faces=pressure_faces,
        pressure_values=pressure_values,
        traction_faces=top_faces,
        traction_values=traction_values,
        prescribed=prescribed,
        quadrature=quadrature,
        element_type=element_type,
    )
    distorted_model, distorted_rhs = assemble_model_and_rhs(
        nodes=distorted_nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=body_force,
        pressure_faces=pressure_faces,
        pressure_values=pressure_values,
        traction_faces=top_faces,
        traction_values=traction_values,
        prescribed=prescribed,
        quadrature=quadrature,
        element_type=element_type,
    )

    regular_u = regular_model.solve(regular_rhs).reshape(regular_model.analysis_nodes.shape[0], 2)
    distorted_u = distorted_model.solve(distorted_rhs).reshape(distorted_model.analysis_nodes.shape[0], 2)
    regular_samples = regular_model.evaluate_quadrature(regular_u)
    distorted_samples = distorted_model.evaluate_quadrature(distorted_u)

    dr = (ro - ri) / nr
    dz = height / nz
    sample_r = np.linspace(ri + 3.5 * dr, ro - 3.5 * dr, 5)
    sample_z = np.linspace(2.5 * dz, height - 2.5 * dz, 3)
    sample_rr, sample_zz = np.meshgrid(sample_r, sample_z, indexing="ij")
    sample_points = np.column_stack([sample_rr.reshape(-1), sample_zz.reshape(-1)])

    regular_displacement = interpolate_field(regular_model.analysis_nodes, regular_u, sample_points)
    distorted_displacement = interpolate_field(distorted_model.analysis_nodes, distorted_u, sample_points)
    regular_stress = interpolate_field(
        regular_samples.points_rz.reshape(-1, 2),
        regular_samples.stress.reshape(-1, 4),
        sample_points,
    )
    distorted_stress = interpolate_field(
        distorted_samples.points_rz.reshape(-1, 2),
        distorted_samples.stress.reshape(-1, 4),
        sample_points,
    )

    assert np.max(np.abs(regular_displacement[:, 0])) > 1.0e-8
    assert np.max(np.abs(regular_displacement[:, 1])) > 1.0e-8
    assert np.max(np.abs(regular_stress[:, 3])) > 1.0e3

    assert normalized_peak_error(distorted_displacement[:, 0], regular_displacement[:, 0]) < 1.0e-2
    assert normalized_peak_error(distorted_displacement[:, 1], regular_displacement[:, 1]) < 1.0e-2
    assert normalized_peak_error(distorted_stress[:, 0], regular_stress[:, 0]) < 1.0e-2
    assert normalized_peak_error(distorted_stress[:, 1], regular_stress[:, 1]) < 1.0e-2
    assert normalized_peak_error(distorted_stress[:, 2], regular_stress[:, 2]) < 1.0e-2
    assert normalized_peak_error(distorted_stress[:, 3], regular_stress[:, 3]) < 1.0e-2


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

    with pytest.raises(ValueError, match="displacements must have shape"):
        model = fem.assemble_axisymmetric(
            nodes=nodes,
            elements=elements,
            material_ids=np.zeros((1,), dtype=np.uint64),
            material_table=np.asarray([material]),
        )
        model.evaluate_quadrature(np.zeros((nodes.shape[0], 3), dtype=np.float64))

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
    assert model.input_nodes.shape == nodes.shape
    assert model.input_elements.shape == elements.shape
    assert model.build_rhs(nodal_temperature=nodal_temperature).shape == (model.ndof_reduced,)
    with pytest.raises(ValueError, match="nodal_temperature is required"):
        model.build_rhs()

    packed_ids, packed_material_table, packed_thermal_table = fem.pack_material_tables_from_tags(
        material_ids=np.array([7], dtype=np.uint64),
        material_table_by_tag={7: material},
        thermal_material_table_by_tag={7: thermal_material},
    )
    packed_model = fem.assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=packed_ids,
        material_table=packed_material_table,
        thermal_material_table=packed_thermal_table,
    )
    packed_rhs = packed_model.build_rhs(nodal_temperature=nodal_temperature)
    assert packed_rhs.shape == (packed_model.ndof_reduced,)

    with pytest.raises(AssertionError, match="pack_material_tables_from_tags"):
        fem.assemble_axisymmetric(
            nodes=nodes,
            elements=elements,
            material_ids=np.array([0], dtype=np.uint64),
            material_table={0: material},
        )

    with pytest.raises(ValueError, match="missing from material_table_by_tag"):
        fem.pack_material_tables_from_tags(
            material_ids=np.array([1], dtype=np.uint64),
            material_table_by_tag={0: material},
        )

    with pytest.raises(ValueError, match="same keys as material_table_by_tag"):
        fem.pack_material_tables_from_tags(
            material_ids=np.array([1], dtype=np.uint64),
            material_table_by_tag={0: material, 1: material},
            thermal_material_table_by_tag={1: thermal_material, 2: thermal_material},
        )

    with pytest.raises(AssertionError, match="alpha_rz"):
        fem.assemble_axisymmetric(
            nodes=nodes,
            elements=elements,
            material_ids=np.array([0], dtype=np.uint64),
            material_table=np.asarray([material]),
            thermal_material_table=np.asarray([[1.2e-5, 1.2e-5, 1.2e-5, 1.0e-9, 293.15]], dtype=dtype),
        )

    with pytest.raises(ValueError, match="nodal_temperature is required"):
        zero_displacement = np.zeros((model.ndof_reduced,), dtype=dtype)
        model.evaluate_quadrature(zero_displacement)

    samples = model.evaluate_quadrature(
        np.zeros((model.analysis_nodes.shape[0], 2), dtype=dtype),
        nodal_temperature=nodal_temperature,
    )
    assert samples.stress.shape[-1] == 4


def test_pack_material_tables_from_tags_sorts_tags_and_rewrites_ids() -> None:
    dtype = np.float64
    material_a = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=dtype)
    material_b = isotropic_axisymmetric_material(150.0e9, 0.31, dtype=dtype)
    thermal_a = fem.isotropic_axisymmetric_thermal_material(1.2e-5, 293.15, dtype=dtype)
    thermal_b = fem.isotropic_axisymmetric_thermal_material(1.8e-5, 310.0, dtype=dtype)

    packed_ids_no_thermal, packed_material_table_no_thermal, packed_thermal_table_no_thermal = (
        fem.pack_material_tables_from_tags(
            material_ids=np.array([20, 10, 20], dtype=np.uint64),
            material_table_by_tag={20: material_b, 10: material_a},
        )
    )
    assert np.array_equal(packed_ids_no_thermal, np.array([1, 0, 1], dtype=np.uint64))
    assert np.allclose(packed_material_table_no_thermal[0], material_a)
    assert np.allclose(packed_material_table_no_thermal[1], material_b)
    assert packed_thermal_table_no_thermal is None

    packed_ids, packed_material_table, packed_thermal_table = fem.pack_material_tables_from_tags(
        material_ids=np.array([20, 10, 20], dtype=np.uint64),
        material_table_by_tag={20: material_b, 10: material_a},
        thermal_material_table_by_tag={20: thermal_b, 10: thermal_a},
    )

    assert np.array_equal(packed_ids, np.array([1, 0, 1], dtype=np.uint64))
    assert np.allclose(packed_material_table[0], material_a)
    assert np.allclose(packed_material_table[1], material_b)
    assert packed_thermal_table is not None
    assert np.allclose(packed_thermal_table[0], thermal_a)
    assert np.allclose(packed_thermal_table[1], thermal_b)


def test_python_convenience_wrappers_preserve_dtype_and_shapes() -> None:
    dtype = np.dtype(np.float32)
    iso = fem.isotropic_axisymmetric_material(200.0e9, 0.3, dtype=dtype)
    reduced = fem.cfsem_radial_material(200.0e9, 0.3, dtype=dtype)
    thermal = fem.isotropic_axisymmetric_thermal_material(1.0e-5, reference_temperature=293.15, dtype=dtype)
    ortho = fem.orthotropic_axisymmetric_thermal_material(
        1.0e-5, 2.0e-5, 3.0e-5, reference_temperature=293.15, dtype=dtype
    )
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=2, nz=1, dtype=np.float32)
    elevated = fem.infer_quad9_mesh(nodes, elements)

    assert iso.shape == (4, 4)
    assert reduced.shape == (4, 4)
    assert thermal.shape == (5,)
    assert ortho.shape == (5,)
    assert iso.dtype == dtype
    assert reduced.dtype == dtype
    assert thermal.dtype == dtype
    assert ortho.dtype == dtype
    assert elevated.analysis_elements.shape[1] == 9
    assert elevated.analysis_nodes.dtype == dtype


def test_quad9_temperature_elevation_reproduces_affine_temperature_field() -> None:
    dtype = np.dtype(np.float64)
    nodes = np.asarray(
        [
            [0.52, 0.00],
            [0.81, 0.04],
            [1.07, -0.01],
            [0.56, 0.27],
            [0.84, 0.33],
            [1.10, 0.29],
        ],
        dtype=dtype,
    )
    elements = np.asarray([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=np.uint64)
    elevated = fem.infer_quad9_mesh(nodes, elements)

    a_r, b_z, c0 = 3.25, -1.75, 4.5
    corner_temperature = a_r * nodes[:, 0] + b_z * nodes[:, 1] + c0
    analysis_temperature = fem._analysis_temperature_for_element_type(
        corner_temperature,
        nodes.shape[0],
        elevated,
        dtype,
    )
    expected_temperature = a_r * elevated.analysis_nodes[:, 0] + b_z * elevated.analysis_nodes[:, 1] + c0

    assert np.allclose(analysis_temperature, expected_temperature, rtol=0.0, atol=1.0e-14)
