//! Axisymmetric strain-displacement utilities.
//!
//! The `B` matrix uses the standard small-strain axisymmetric ordering
//! `[e_rr, e_zz, e_tt, g_rz]` with `e_tt = u_r / r`.  The resulting element stiffness is the
//! conventional `B^T D B` construction; see Hughes (1987), Bathe (1996), and Reddy (2005).

use crate::physics::solenoid_stress::quad4::{DOF_PER_ELEMENT, NODES_PER_ELEMENT};
use crate::physics::solenoid_stress::types::Real;

pub fn build_b_matrix<F: Real>(
    n: &[F; NODES_PER_ELEMENT],
    grad_phys: &[[F; 2]; NODES_PER_ELEMENT],
    radius: F,
) -> Result<[[F; DOF_PER_ELEMENT]; 4], String> {
    if radius <= F::epsilon() {
        return Err(format!(
            "quadrature radius {radius:?} is too close to zero for the axisymmetric hoop-strain term"
        ));
    }
    let mut b = [[F::zero(); DOF_PER_ELEMENT]; 4];
    for i in 0..NODES_PER_ELEMENT {
        let col_r = 2 * i;
        let col_z = col_r + 1;
        let dndr = grad_phys[i][0];
        let dndz = grad_phys[i][1];
        b[0][col_r] = dndr;
        b[1][col_z] = dndz;
        b[2][col_r] = n[i] / radius;
        b[3][col_r] = dndz;
        b[3][col_z] = dndr;
    }
    Ok(b)
}

pub fn accumulate_stiffness<F: Real>(
    ke: &mut [[F; DOF_PER_ELEMENT]; DOF_PER_ELEMENT],
    d: &[[F; 4]; 4],
    b: &[[F; DOF_PER_ELEMENT]; 4],
    scale: F,
) {
    let mut db = [[F::zero(); DOF_PER_ELEMENT]; 4];
    for row in 0..4 {
        for col in 0..DOF_PER_ELEMENT {
            let mut value = F::zero();
            for k in 0..4 {
                value = value + d[row][k] * b[k][col];
            }
            db[row][col] = value;
        }
    }

    for row in 0..DOF_PER_ELEMENT {
        for col in 0..DOF_PER_ELEMENT {
            let mut value = F::zero();
            for k in 0..4 {
                value = value + b[k][row] * db[k][col];
            }
            ke[row][col] = ke[row][col] + scale * value;
        }
    }
}
