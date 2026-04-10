//! Consistent element load-vector construction.
//!
//! Body-force, pressure, and traction terms are assembled as consistent nodal loads from the same
//! weak form used for the stiffness matrix. The axisymmetric face contributions include the
//! `2*pi*r` measure factor on the revolved face; see Hughes (1987), Bathe (1996), and Reddy (2005).

use crate::mesh::QuadratureRule;
use crate::mesh::elements::quad2d::{quad4, quad9};
use crate::physics::solenoid_stress::geometry::{
    FaceSample, face_samples_quad4, face_samples_quad9,
};
use crate::physics::solenoid_stress::types::{DOF_PER_NODE, Real, dof_per_element, two_pi};

/// Accumulate the consistent nodal load vector for a uniform body-force density on one element.
pub fn accumulate_body_force<
    F: Real,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    fe: &mut [F; DOF_PER_ELEMENT],
    body_force: [F; 2],
    shape: &[F; NODES_PER_ELEMENT],
    scale: F,
) {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    for (i, n) in shape.iter().enumerate() {
        fe[2 * i] = fe[2 * i] + scale * *n * body_force[0];
        fe[2 * i + 1] = fe[2 * i + 1] + scale * *n * body_force[1];
    }
}

fn pressure_element_load_generic<
    F: Real,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    coords: &[[F; 2]; NODES_PER_ELEMENT],
    local_face: u8,
    pressure: F,
    quadrature: QuadratureRule,
    face_samples_fn: fn(
        &[[F; 2]; NODES_PER_ELEMENT],
        u8,
        QuadratureRule,
    ) -> Result<Vec<FaceSample<F, NODES_PER_ELEMENT>>, String>,
) -> Result<[F; DOF_PER_ELEMENT], String> {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    let mut fe = [F::zero(); DOF_PER_ELEMENT];
    let two_pi = two_pi::<F>();
    for sample in face_samples_fn(coords, local_face, quadrature)? {
        let normal_area = [sample.tangent[1], -sample.tangent[0]];
        let scale = -pressure * two_pi * sample.point[0] * sample.weight;
        for (i, n) in sample.n.iter().enumerate() {
            fe[2 * i] = fe[2 * i] + scale * *n * normal_area[0];
            fe[2 * i + 1] = fe[2 * i + 1] + scale * *n * normal_area[1];
        }
    }
    Ok(fe)
}

fn traction_element_load_generic<
    F: Real,
    const NODES_PER_ELEMENT: usize,
    const DOF_PER_ELEMENT: usize,
>(
    coords: &[[F; 2]; NODES_PER_ELEMENT],
    local_face: u8,
    traction: [F; 2],
    quadrature: QuadratureRule,
    face_samples_fn: fn(
        &[[F; 2]; NODES_PER_ELEMENT],
        u8,
        QuadratureRule,
    ) -> Result<Vec<FaceSample<F, NODES_PER_ELEMENT>>, String>,
) -> Result<[F; DOF_PER_ELEMENT], String> {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    let mut fe = [F::zero(); DOF_PER_ELEMENT];
    let two_pi = two_pi::<F>();
    for sample in face_samples_fn(coords, local_face, quadrature)? {
        let tangent_norm =
            (sample.tangent[0] * sample.tangent[0] + sample.tangent[1] * sample.tangent[1]).sqrt();
        let scale = two_pi * sample.point[0] * tangent_norm * sample.weight;
        for (i, n) in sample.n.iter().enumerate() {
            fe[2 * i] = fe[2 * i] + scale * *n * traction[0];
            fe[2 * i + 1] = fe[2 * i + 1] + scale * *n * traction[1];
        }
    }
    Ok(fe)
}

/// Integrate the consistent nodal load vector for a constant pressure on one Quad4 element face.
pub fn pressure_element_load_quad4<F: Real>(
    coords: &[[F; 2]; quad4::NODES_PER_ELEMENT],
    local_face: u8,
    pressure: F,
    quadrature: QuadratureRule,
) -> Result<[F; dof_per_element(quad4::NODES_PER_ELEMENT)], String> {
    pressure_element_load_generic::<
        F,
        { quad4::NODES_PER_ELEMENT },
        { dof_per_element(quad4::NODES_PER_ELEMENT) },
    >(
        coords,
        local_face,
        pressure,
        quadrature,
        face_samples_quad4::<F>,
    )
}

/// Integrate the consistent nodal load vector for a constant pressure on one Quad9 element face.
pub fn pressure_element_load_quad9<F: Real>(
    coords: &[[F; 2]; quad9::NODES_PER_ELEMENT],
    local_face: u8,
    pressure: F,
    quadrature: QuadratureRule,
) -> Result<[F; dof_per_element(quad9::NODES_PER_ELEMENT)], String> {
    pressure_element_load_generic::<
        F,
        { quad9::NODES_PER_ELEMENT },
        { dof_per_element(quad9::NODES_PER_ELEMENT) },
    >(
        coords,
        local_face,
        pressure,
        quadrature,
        face_samples_quad9::<F>,
    )
}

/// Integrate the consistent nodal load vector for a constant traction on one Quad4 element face.
pub fn traction_element_load_quad4<F: Real>(
    coords: &[[F; 2]; quad4::NODES_PER_ELEMENT],
    local_face: u8,
    traction: [F; 2],
    quadrature: QuadratureRule,
) -> Result<[F; dof_per_element(quad4::NODES_PER_ELEMENT)], String> {
    traction_element_load_generic::<
        F,
        { quad4::NODES_PER_ELEMENT },
        { dof_per_element(quad4::NODES_PER_ELEMENT) },
    >(
        coords,
        local_face,
        traction,
        quadrature,
        face_samples_quad4::<F>,
    )
}

/// Integrate the consistent nodal load vector for a constant traction on one Quad9 element face.
pub fn traction_element_load_quad9<F: Real>(
    coords: &[[F; 2]; quad9::NODES_PER_ELEMENT],
    local_face: u8,
    traction: [F; 2],
    quadrature: QuadratureRule,
) -> Result<[F; dof_per_element(quad9::NODES_PER_ELEMENT)], String> {
    traction_element_load_generic::<
        F,
        { quad9::NODES_PER_ELEMENT },
        { dof_per_element(quad9::NODES_PER_ELEMENT) },
    >(
        coords,
        local_face,
        traction,
        quadrature,
        face_samples_quad9::<F>,
    )
}
