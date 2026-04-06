//! Consistent element load-vector construction.
//!
//! Body-force and pressure terms are assembled as consistent nodal loads from the same weak form
//! used for the stiffness matrix.  The axisymmetric pressure contribution includes the `2*pi*r`
//! measure factor on the revolved face; see Hughes (1987), Bathe (1996), and Reddy (2005).

use crate::physics::solenoid_stress::geometry::face_samples;
use crate::physics::solenoid_stress::quad4::DOF_PER_ELEMENT;
use crate::physics::solenoid_stress::quadrature::QuadratureRule;
use crate::physics::solenoid_stress::types::{Real, two_pi};

pub fn accumulate_body_force<F: Real>(
    fe: &mut [F; DOF_PER_ELEMENT],
    body_force: [F; 2],
    shape: &[F; 4],
    scale: F,
) {
    for (i, n) in shape.iter().enumerate() {
        fe[2 * i] = fe[2 * i] + scale * *n * body_force[0];
        fe[2 * i + 1] = fe[2 * i + 1] + scale * *n * body_force[1];
    }
}

pub fn pressure_element_load<F: Real>(
    coords: &[[F; 2]; 4],
    local_face: u8,
    pressure: F,
    quadrature: QuadratureRule,
) -> Result<[F; DOF_PER_ELEMENT], String> {
    let mut fe = [F::zero(); DOF_PER_ELEMENT];
    let two_pi = two_pi::<F>();
    for sample in face_samples(coords, local_face, quadrature)? {
        let normal_area = [sample.tangent[1], -sample.tangent[0]];
        let scale = -pressure * two_pi * sample.point[0] * sample.weight;
        for (i, n) in sample.n.iter().enumerate() {
            fe[2 * i] = fe[2 * i] + scale * *n * normal_area[0];
            fe[2 * i + 1] = fe[2 * i + 1] + scale * *n * normal_area[1];
        }
    }
    Ok(fe)
}
