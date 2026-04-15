from __future__ import annotations

import os

import numpy as np
import scipy.sparse.linalg as spla

from cfsem.solenoid_stress import (
    apply_dirichlet,
    assemble_axisymmetric_model,
    evaluate_axisymmetric_strain_stress_at_quadrature,
    isotropic_axisymmetric_material,
    isotropic_axisymmetric_thermal_material,
)


"""
All-in-one repeated-load example for the axisymmetric FEM API.

This script assembles a reusable model once, with fixed:
- mesh
- material stiffness
- pressure face list
- traction face list
- thermal material data

It then updates the load values only:
- body-force density per element
- pressure value per loaded face
- traction vector per loaded face
- nodal temperature

and rebuilds the right-hand side entirely through the sparse load operators.
"""


def build_annulus_strip_mesh(
    ri: float,
    ro: float,
    height: float,
    nr: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray]:
    radii = np.linspace(ri, ro, nr + 1, dtype=np.float64)
    zs = np.linspace(0.0, height, nz + 1, dtype=np.float64)
    nodes = np.array([[r, z] for z in zs for r in radii], dtype=np.float64)

    def node_id(i: int, j: int) -> int:
        return j * (nr + 1) + i

    elements = []
    for j in range(nz):
        for i in range(nr):
            elements.append([node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1), node_id(i, j + 1)])
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


def prescribed_dofs(nodes: np.ndarray) -> dict[int, float]:
    fixed = {
        2 * node + 1: 0.0
        for node, point in enumerate(nodes)
        if np.isclose(point[1], 0.0)
    }
    fixed[0] = 0.0
    return fixed


def von_mises(stress: np.ndarray) -> np.ndarray:
    return np.sqrt(
        0.5
        * (
            (stress[..., 0] - stress[..., 1]) ** 2
            + (stress[..., 1] - stress[..., 2]) ** 2
            + (stress[..., 2] - stress[..., 0]) ** 2
            + 6.0 * stress[..., 3] ** 2
        )
    )


def make_body_force(nelem: int, radial_scale: float, axial_scale: float) -> np.ndarray:
    return np.column_stack(
        [
            np.linspace(-1.0, 1.0, nelem, dtype=np.float64) * radial_scale,
            np.linspace(1.0, -1.0, nelem, dtype=np.float64) * axial_scale,
        ]
    )


def make_temperature_field(
    nodes: np.ndarray,
    base_temperature: float,
    radial_rise: float,
    axial_rise: float,
) -> np.ndarray:
    radial_coord = nodes[:, 0]
    axial_coord = nodes[:, 1]
    radial_span = max(np.ptp(radial_coord), 1.0e-12)
    axial_span = max(np.ptp(axial_coord), 1.0e-12)
    return (
        base_temperature
        + radial_rise * (radial_coord - radial_coord.min()) / radial_span
        + axial_rise * (axial_coord - axial_coord.min()) / axial_span
    )


def summarize_case(
    name: str,
    model,
    solve_free,
    input_nodes: np.ndarray,
    elements: np.ndarray,
    material: np.ndarray,
    thermal_material: np.ndarray,
    quadrature: str,
    element_type: str,
    prescribed: dict[int, float],
    body_force: np.ndarray,
    pressure_values: np.ndarray,
    traction_values: np.ndarray,
    nodal_temperature: np.ndarray,
) -> None:
    rhs_body = np.asarray(model.body_force_to_rhs @ body_force.reshape(-1), dtype=np.float64)
    rhs_pressure = np.asarray(model.pressure_to_rhs @ pressure_values, dtype=np.float64)
    rhs_traction = np.asarray(model.traction_to_rhs @ traction_values.reshape(-1), dtype=np.float64)
    rhs_temperature = np.asarray(model.temperature_to_rhs @ nodal_temperature, dtype=np.float64)
    rhs_manual = (
        model.thermal_reference_rhs
        + rhs_body
        + rhs_pressure
        + rhs_traction
        + rhs_temperature
    )
    rhs_full = model.rhs(
        body_force=body_force,
        pressure_values=pressure_values,
        traction_values=traction_values,
        nodal_temperature=nodal_temperature,
    )
    reduced = apply_dirichlet(model.stiffness, rhs_full, prescribed=prescribed)
    displacement = reduced.recover(solve_free(reduced.rhs))
    samples = evaluate_axisymmetric_strain_stress_at_quadrature(
        input_nodes,
        elements,
        np.zeros(elements.shape[0], dtype=np.uint64),
        np.asarray([material]),
        displacement,
        thermal_material_table=np.asarray([thermal_material]),
        nodal_temperature=nodal_temperature,
        quadrature=quadrature,
        element_type=element_type,
    )
    vm = von_mises(samples.stress)
    displacement_2d = displacement.reshape(-1, 2)

    if not np.allclose(rhs_manual, rhs_full):
        raise AssertionError(f"{name}: operator sum did not match model.rhs(...)")

    print(name)
    print(
        "  rhs pieces:"
        f" ||body||={np.linalg.norm(rhs_body):.3e},"
        f" ||pressure||={np.linalg.norm(rhs_pressure):.3e},"
        f" ||traction||={np.linalg.norm(rhs_traction):.3e},"
        f" ||thermal||={np.linalg.norm(rhs_temperature):.3e},"
        f" ||reference||={np.linalg.norm(model.thermal_reference_rhs):.3e}"
    )
    print(
        "  checks:"
        f" manual-vs-model={np.linalg.norm(rhs_manual - rhs_full):.3e},"
        f" reduced-rhs={np.linalg.norm(reduced.rhs):.3e}"
    )
    print(
        "  response:"
        f" max|u_r|={np.max(np.abs(displacement_2d[:, 0])):.6e} m,"
        f" max|u_z|={np.max(np.abs(displacement_2d[:, 1])):.6e} m,"
        f" max VM={np.max(vm):.6e} Pa"
    )


def main() -> None:
    testing = os.environ.get("CFSEM_TESTING") == "True"
    nr = 4 if testing else 10
    nz = 2 if testing else 4
    quadrature = "gl4"
    element_type = "quad9"

    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=nr, nz=nz)
    _inner_faces, outer_faces = pressure_faces_for_strip(nr=nr, nz=nz)
    _bottom_faces, top_faces = horizontal_faces_for_strip(nr=nr, nz=nz)

    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=np.float64)
    thermal_material = isotropic_axisymmetric_thermal_material(
        1.1e-5,
        reference_temperature=293.15,
        dtype=np.float64,
    )

    model = assemble_axisymmetric_model(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        pressure_faces=outer_faces,
        traction_faces=top_faces,
        thermal_material_table=np.asarray([thermal_material]),
        quadrature=quadrature,
        element_type=element_type,
    )
    prescribed = prescribed_dofs(model.analysis_nodes)
    reduced_zero = apply_dirichlet(model.stiffness, np.zeros(model.ndof, dtype=np.float64), prescribed=prescribed)
    solve_free = spla.factorized(reduced_zero.matrix.tocsc())

    print("Reusable axisymmetric FEM model")
    print(
        "  mesh:"
        f" nelem={model.nelem},"
        f" ndof={model.ndof},"
        f" element_type={model.element_type},"
        f" quadrature={quadrature}"
    )
    print(
        "  operators:"
        f" body_force_to_rhs={model.body_force_to_rhs.shape},"
        f" pressure_to_rhs={model.pressure_to_rhs.shape},"
        f" traction_to_rhs={model.traction_to_rhs.shape},"
        f" temperature_to_rhs={model.temperature_to_rhs.shape}"
    )
    print("  face lists are fixed once; only load values change between cases")

    case_1_body_force = make_body_force(model.nelem, radial_scale=4.0e4, axial_scale=2.0e4)
    case_1_pressure = np.linspace(1.2e5, 2.0e5, model.pressure_faces.shape[0], dtype=np.float64)
    case_1_traction = np.column_stack(
        [
            np.linspace(0.0, 8.0e4, model.traction_faces.shape[0], dtype=np.float64),
            np.linspace(-1.5e5, -8.0e4, model.traction_faces.shape[0], dtype=np.float64),
        ]
    )
    case_1_temperature = make_temperature_field(
        nodes,
        base_temperature=298.15,
        radial_rise=10.0,
        axial_rise=4.0,
    )

    case_2_body_force = make_body_force(model.nelem, radial_scale=-2.5e4, axial_scale=5.0e4)
    case_2_pressure = np.linspace(-5.0e4, 1.5e5, model.pressure_faces.shape[0], dtype=np.float64)
    case_2_traction = np.column_stack(
        [
            np.linspace(6.0e4, -6.0e4, model.traction_faces.shape[0], dtype=np.float64),
            np.linspace(1.2e5, -4.0e4, model.traction_faces.shape[0], dtype=np.float64),
        ]
    )
    case_2_temperature = make_temperature_field(
        nodes,
        base_temperature=301.15,
        radial_rise=6.0,
        axial_rise=12.0,
    )

    summarize_case(
        "Case 1: outward pressure, top traction, moderate thermal gradient",
        model,
        solve_free,
        nodes,
        elements,
        material,
        thermal_material,
        quadrature,
        element_type,
        prescribed,
        case_1_body_force,
        case_1_pressure,
        case_1_traction,
        case_1_temperature,
    )
    summarize_case(
        "Case 2: updated load values reusing the same operators and factorization",
        model,
        solve_free,
        nodes,
        elements,
        material,
        thermal_material,
        quadrature,
        element_type,
        prescribed,
        case_2_body_force,
        case_2_pressure,
        case_2_traction,
        case_2_temperature,
    )


if __name__ == "__main__":
    main()
