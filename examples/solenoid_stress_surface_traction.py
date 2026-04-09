from __future__ import annotations

import numpy as np

from cfsem.solenoid_stress import (
    assemble_axisymmetric,
    evaluate_axisymmetric_strain_stress_at_quadrature,
    isotropic_axisymmetric_material,
    solve_dirichlet,
)


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


def traction_case_summary(
    name: str,
    nodes: np.ndarray,
    elements: np.ndarray,
    pressure_faces: np.ndarray | None,
    pressure_values: np.ndarray | None,
    traction_faces: np.ndarray | None,
    traction_values: np.ndarray | None,
) -> None:
    material = isotropic_axisymmetric_material(200.0e9, 0.27, dtype=np.float64)
    assembly = assemble_axisymmetric(
        nodes=nodes,
        elements=elements,
        material_ids=np.zeros(elements.shape[0], dtype=np.uint64),
        material_table=np.asarray([material]),
        body_force=np.array([0.0, 0.0], dtype=np.float64),
        pressure_faces=pressure_faces,
        pressure_values=pressure_values,
        traction_faces=traction_faces,
        traction_values=traction_values,
        quadrature="4x4",
        element_type="quad9",
    )
    displacement = solve_dirichlet(
        assembly.to_csr(),
        assembly.rhs,
        prescribed=prescribed_dofs(nodes),
    ).reshape(-1, 2)
    samples = evaluate_axisymmetric_strain_stress_at_quadrature(
        nodes,
        elements,
        np.zeros(elements.shape[0], dtype=np.uint64),
        np.asarray([material]),
        displacement.reshape(-1),
        quadrature="4x4",
        element_type="quad9",
    )
    disp_mag = np.linalg.norm(displacement, axis=1)
    vm = np.sqrt(
        0.5
        * (
            (samples.stress[..., 0] - samples.stress[..., 1]) ** 2
            + (samples.stress[..., 1] - samples.stress[..., 2]) ** 2
            + (samples.stress[..., 2] - samples.stress[..., 0]) ** 2
            + 6.0 * samples.stress[..., 3] ** 2
        )
    )
    print(
        f"{name}: max|u_r|={np.max(np.abs(displacement[:, 0])):.6e} m, "
        f"max|u_z|={np.max(np.abs(displacement[:, 1])):.6e} m, "
        f"max|u|={np.max(disp_mag):.6e} m, "
        f"max VM={np.max(vm):.6e} Pa"
    )


def main() -> None:
    nodes, elements = build_annulus_strip_mesh(0.5, 1.0, 0.2, nr=10, nz=4)
    _inner_faces, outer_faces = pressure_faces_for_strip(nr=10, nz=4)
    _bottom_faces, top_faces = horizontal_faces_for_strip(nr=10, nz=4)

    traction_case_summary(
        "Top axial traction",
        nodes,
        elements,
        pressure_faces=None,
        pressure_values=None,
        traction_faces=top_faces,
        traction_values=np.array([0.0, 2.0e5], dtype=np.float64),
    )
    traction_case_summary(
        "Outer radial traction",
        nodes,
        elements,
        pressure_faces=None,
        pressure_values=None,
        traction_faces=outer_faces,
        traction_values=np.array([1.5e5, 0.0], dtype=np.float64),
    )
    traction_case_summary(
        "Outer pressure plus top shear-like traction",
        nodes,
        elements,
        pressure_faces=outer_faces,
        pressure_values=np.full(outer_faces.shape[0], 2.0e5, dtype=np.float64),
        traction_faces=top_faces,
        traction_values=np.array([1.0e5, -5.0e4], dtype=np.float64),
    )


if __name__ == "__main__":
    main()
