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

use crate::physics::solenoid_stress::types::{DOF_PER_NODE, Structural2dFormulation};

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
pub fn build_axisymmetric_b_matrix<const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    n: &[f64; NODES_PER_ELEMENT],
    grad_phys: &[[f64; 2]; NODES_PER_ELEMENT],
    radius: f64,
) -> Result<[[f64; DOF_PER_ELEMENT]; 4], String> {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    if radius <= f64::EPSILON {
        return Err(format!(
            "quadrature radius {radius:?} is too close to zero for the axisymmetric hoop-strain term"
        ));
    }
    let mut b = [[0.0; DOF_PER_ELEMENT]; 4];
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

/// Build the plane-strain strain-displacement matrix `B` at one quadrature point.
///
/// Multiplying this matrix by the element displacement vector
/// `[u_x1, u_y1, u_x2, u_y2, ...]^T` gives the strain vector
/// `[e_xx, e_yy, e_zz, g_xy]^T` at that point, with `e_zz = 0`.
pub fn build_plane_strain_b_matrix<const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    grad_phys: &[[f64; 2]; NODES_PER_ELEMENT],
) -> [[f64; DOF_PER_ELEMENT]; 4] {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    let mut b = [[0.0; DOF_PER_ELEMENT]; 4];
    for (i, grad) in grad_phys.iter().enumerate() {
        let col_x = 2 * i;
        let col_y = col_x + 1;
        let dndx = grad[0];
        let dndy = grad[1];
        // e_xx = du_x/dx
        b[0][col_x] = dndx;
        // e_yy = du_y/dy
        b[1][col_y] = dndy;
        // e_zz = 0 by the plane-strain kinematic assumption.
        // g_xy = du_x/dy + du_y/dx in engineering-shear convention.
        b[3][col_x] = dndy;
        b[3][col_y] = dndx;
    }
    b
}

/// Build the formulation-specific strain-displacement matrix `B` at one quadrature point.
pub fn build_b_matrix<const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    formulation: Structural2dFormulation,
    n: &[f64; NODES_PER_ELEMENT],
    grad_phys: &[[f64; 2]; NODES_PER_ELEMENT],
    point: [f64; 2],
) -> Result<[[f64; DOF_PER_ELEMENT]; 4], String> {
    match formulation {
        Structural2dFormulation::Axisymmetric => build_axisymmetric_b_matrix::<
            NODES_PER_ELEMENT,
            DOF_PER_ELEMENT,
        >(n, grad_phys, point[0]),
        Structural2dFormulation::PlaneStrain { .. } => Ok(build_plane_strain_b_matrix::<
            NODES_PER_ELEMENT,
            DOF_PER_ELEMENT,
        >(grad_phys)),
    }
}

/// Accumulate `scale * B^T D B` into the element stiffness matrix.
///
/// The caller supplies `scale = 2*pi*r*det(J)*w`, so this routine is purely the dense local
/// linear-algebra kernel for one quadrature point.  The elastic stress-strain matrix is applied
/// directly through the supplied per-material `4 x 4` matrix `d`; no global stress-strain matrix
/// is ever assembled.
///
/// # References
/// - Allan F. Bower, *Applied Mechanics of Solids*, CRC Press, 2009, Section 8.1.
/// - E. L. Wilson, "Structural Analysis of Axisymmetric Solids," *AIAA Journal*, 3(12), pp. 2269-2274, December 1965. doi:10.2514/3.3356.
/// - R. A. Mitchell, R. M. Woolley, and C. R. Fisher, "Formulation and experimental verification of an axisymmetric finite-element structural analysis," *Journal of Research of the National Bureau of Standards Section C*, 75C, 1971.
/// - I. Fried, "Notes on the finite element analysis of the axisymmetric elastic solid," *International Journal of Solids and Structures*, 10(3), 1974.
pub fn accumulate_stiffness<const DOF_PER_ELEMENT: usize>(
    ke: &mut [[f64; DOF_PER_ELEMENT]; DOF_PER_ELEMENT],
    d: &[[f64; 4]; 4],
    b: &[[f64; DOF_PER_ELEMENT]; 4],
    scale: f64,
) {
    let db = constitutive_times_b(d, b);

    for row in 0..DOF_PER_ELEMENT {
        for col in 0..DOF_PER_ELEMENT {
            let mut value = 0.0;
            for k in 0..4 {
                value = value + b[k][row] * db[k][col];
            }
            ke[row][col] = ke[row][col] + scale * value;
        }
    }
}

/// Multiply the constitutive matrix `D` by the axisymmetric strain-displacement matrix `B`.
///
/// The returned matrix maps element displacement DOFs directly to stresses:
/// `sigma = (D B) u_e`.
///
/// This is the local matrix-free application of the elastic stress-strain law used in both
/// stiffness assembly and stress recovery.  `D` is the per-material `4 x 4` constitutive matrix;
/// it is not assembled into any larger global matrix.
pub fn constitutive_times_b<const DOF_PER_ELEMENT: usize>(
    d: &[[f64; 4]; 4],
    b: &[[f64; DOF_PER_ELEMENT]; 4],
) -> [[f64; DOF_PER_ELEMENT]; 4] {
    let mut db = [[0.0; DOF_PER_ELEMENT]; 4];
    for row in 0..4 {
        for col in 0..DOF_PER_ELEMENT {
            let mut value = 0.0;
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
/// This is the local matrix-free application of the elastic stress-strain law used for thermal
/// stress construction and other quadrature-point stress calculations.
pub fn constitutive_times_strain(d: &[[f64; 4]; 4], strain: &[f64; 4]) -> [f64; 4] {
    let mut out = [0.0; 4];
    for row in 0..4 {
        let mut value = 0.0;
        for k in 0..4 {
            value = value + d[row][k] * strain[k];
        }
        out[row] = value;
    }
    out
}

/// Accumulate `scale * B^T sigma` into one element load vector.
pub fn accumulate_b_transpose_vector<const DOF_PER_ELEMENT: usize>(
    fe: &mut [f64; DOF_PER_ELEMENT],
    b: &[[f64; DOF_PER_ELEMENT]; 4],
    sigma: &[f64; 4],
    scale: f64,
) {
    for dof in 0..DOF_PER_ELEMENT {
        let mut value = 0.0;
        for component in 0..4 {
            value = value + b[component][dof] * sigma[component];
        }
        fe[dof] = fe[dof] + scale * value;
    }
}
