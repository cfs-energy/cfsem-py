//! Axisymmetric strain-displacement utilities.
//!
//! The `B` matrix uses the small-strain axisymmetric ordering
//! `[e_rr, e_zz, e_tt, g_rz]` with `e_tt = u_r / r`.  The resulting element stiffness uses the
//! `B^T D B` construction.  For the general displacement-based finite-element formulation used
//! here, the primary textbook reference is Bower's *Applied Mechanics of Solids*, especially
//! Section 8.1.  The axisymmetric reduction itself follows the axisymmetric finite-element papers
//! cited below.
//!
//! In this formulation each node carries only two displacement unknowns: radial `u_r` and axial
//! `u_z`.  Circumferential displacement is omitted by the axisymmetric assumption, but the hoop
//! normal strain `e_tt` still appears because moving a ring outward changes its circumference.

use crate::physics::solenoid_stress::types::{DOF_PER_NODE, Real};

/// Build the axisymmetric strain-displacement matrix `B` at one quadrature point.
///
/// Multiplying this matrix by the element displacement vector
/// `[u_r1, u_z1, u_r2, u_z2, ...]^T` gives the strain vector
/// `[e_rr, e_zz, e_tt, g_rz]^T` at that point.
///
/// # References
/// - Allan F. Bower, *Applied Mechanics of Solids*, CRC Press, 2009, Section 8.1.
/// - E. L. Wilson, "Structural Analysis of Axisymmetric Solids," *AIAA Journal*, 3(12), pp. 2269-2274, December 1965. doi:10.2514/3.3356.
/// - R. A. Mitchell, R. M. Woolley, and C. R. Fisher, "Formulation and experimental verification of an axisymmetric finite-element structural analysis," *Journal of Research of the National Bureau of Standards Section C*, 75C, 1971.
pub fn build_b_matrix<F: Real, const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    n: &[F; NODES_PER_ELEMENT],
    grad_phys: &[[F; 2]; NODES_PER_ELEMENT],
    radius: F,
) -> Result<[[F; DOF_PER_ELEMENT]; 4], String> {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
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
        // e_rr = du_r/dr
        b[0][col_r] = dndr;
        // e_zz = du_z/dz
        b[1][col_z] = dndz;
        // e_tt = u_r/r, so only the radial displacement shape function enters.
        b[2][col_r] = n[i] / radius;
        // g_rz = du_r/dz + du_z/dr in engineering-shear convention.
        b[3][col_r] = dndz;
        b[3][col_z] = dndr;
    }
    Ok(b)
}

/// Accumulate `scale * B^T D B` into the element stiffness matrix.
///
/// The caller supplies `scale = 2*pi*r*det(J)*w`, so this routine is purely the dense local
/// linear-algebra kernel for one quadrature point.  The constitutive action is applied directly
/// through the supplied per-material `4 x 4` matrix `d`; no global constitutive operator is ever
/// assembled.
///
/// # References
/// - Allan F. Bower, *Applied Mechanics of Solids*, CRC Press, 2009, Section 8.1.
/// - E. L. Wilson, "Structural Analysis of Axisymmetric Solids," *AIAA Journal*, 3(12), pp. 2269-2274, December 1965. doi:10.2514/3.3356.
/// - R. A. Mitchell, R. M. Woolley, and C. R. Fisher, "Formulation and experimental verification of an axisymmetric finite-element structural analysis," *Journal of Research of the National Bureau of Standards Section C*, 75C, 1971.
/// - I. Fried, "Notes on the finite element analysis of the axisymmetric elastic solid," *International Journal of Solids and Structures*, 10(3), 1974.
pub fn accumulate_stiffness<F: Real, const DOF_PER_ELEMENT: usize>(
    ke: &mut [[F; DOF_PER_ELEMENT]; DOF_PER_ELEMENT],
    d: &[[F; 4]; 4],
    b: &[[F; DOF_PER_ELEMENT]; 4],
    scale: F,
) {
    let db = constitutive_times_b(d, b);

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

/// Multiply the constitutive matrix `D` by the axisymmetric strain operator `B`.
///
/// The returned matrix maps element displacement DOFs directly to stresses:
/// `sigma = (D B) u_e`.
///
/// This is the local matrix-free constitutive application used in both stiffness assembly and
/// stress recovery.  `D` is the per-material `4 x 4` constitutive matrix; it is not assembled
/// into any larger global matrix.
pub fn constitutive_times_b<F: Real, const DOF_PER_ELEMENT: usize>(
    d: &[[F; 4]; 4],
    b: &[[F; DOF_PER_ELEMENT]; 4],
) -> [[F; DOF_PER_ELEMENT]; 4] {
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
    db
}

/// Multiply the constitutive matrix `D` by one strain vector.
///
/// This is the local matrix-free constitutive application used for thermal stress construction and
/// other quadrature-point stress calculations.
pub fn constitutive_times_strain<F: Real>(d: &[[F; 4]; 4], strain: &[F; 4]) -> [F; 4] {
    let mut out = [F::zero(); 4];
    for row in 0..4 {
        let mut value = F::zero();
        for k in 0..4 {
            value = value + d[row][k] * strain[k];
        }
        out[row] = value;
    }
    out
}

/// Accumulate `scale * B^T sigma` into one element load vector.
pub fn accumulate_b_transpose_vector<F: Real, const DOF_PER_ELEMENT: usize>(
    fe: &mut [F; DOF_PER_ELEMENT],
    b: &[[F; DOF_PER_ELEMENT]; 4],
    sigma: &[F; 4],
    scale: F,
) {
    for dof in 0..DOF_PER_ELEMENT {
        let mut value = F::zero();
        for component in 0..4 {
            value = value + b[component][dof] * sigma[component];
        }
        fe[dof] = fe[dof] + scale * value;
    }
}
