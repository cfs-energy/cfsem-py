//! Boundary-element fields and formulas.
//!
//! References:
//! * \[1\] S. E. Mousavi and N. Sukumar, “Generalized Duffy transformation for integrating vertex singularities,” Comput Mech, vol. 45, no. 2–3, pp. 127–140, Jan. 2010, doi: 10.1007/s00466-009-0424-1.
//! * \[2\] R. D. Graglia, “On the numerical integration of the linear shape functions times the 3-D Green’s function or its gradient on a plane triangle,” IEEE Transactions on Antennas and Propagation, vol. 41, no. 10, pp. 1448–1455, Oct. 1993, doi: 10.1109/8.247786.
//! * \[3\] D. Wilton, S. Rao, A. Glisson, D. Schaubert, O. Al-Bundak, and C. Butler, “Potential integrals for uniform and linear source distributions on polygonal and polyhedral domains,” IEEE Transactions on Antennas and Propagation, vol. 32, no. 3, pp. 276–281, Mar. 1984, doi: 10.1109/TAP.1984.1143304.
//! * \[4\] M. G. Duffy, “Quadrature Over a Pyramid or Cube of Integrands with a Singularity at a Vertex,” SIAM Journal on Numerical Analysis, vol. 19, no. 6, pp. 1260–1262, 1982.
//! * \[5\] G. N. Peeren, “Stream function approach for determining optimal surface currents,” Phd Thesis 2 (Research NOT TU/e / Graduation TU/e), Technische Universiteit Eindhoven, Eindhoven, 2003. doi: 10.6100/IR570424.
//! * \[6\] F. Hussain, M. S. Karim, and R. Ahamad, “Appropriate Gaussian quadrature formulae for triangles”.

use crate::math::{cross3, rss3};

mod flux_density;
mod inductance;
#[cfg(test)]
mod test;
mod vector_potential;

pub use flux_density::{
    flux_density_triangle, flux_density_triangle_mesh, flux_density_triangle_mesh_par,
    triangle_flux_density_basis,
};
pub use inductance::{
    triangle_basis_mutual_inductance, triangle_basis_mutual_inductance_block,
    triangle_geometric_coupling, triangle_geometric_coupling_regular,
    triangle_inductance_from_potential_vectors,
};
pub use vector_potential::{
    triangle_vector_potential_basis, vector_potential_triangle, vector_potential_triangle_mesh,
    vector_potential_triangle_mesh_par,
};

/// Second-order quadrature integration weights on a triangular surface
/// Format is [Weights, U, V]
///
/// References:
/// * F. Hussain, M. S. Karim, and R. Ahamad, “Appropriate Gaussian quadrature formulae for triangles”.
const TABLE_GAUSS_LEGENDRE_2: [[f64; 3]; 4] = [
    [0.5283121635e-01, 0.1666666667e+00, 0.7886751346e+00],
    [0.1971687836e+00, 0.6220084679e+00, 0.2113248654e+00],
    [0.5283121635e-01, 0.4465819874e-01, 0.7886751346e+00],
    [0.1971687836e+00, 0.1666666667e+00, 0.2113248654e+00],
];

/// Third-order quadrature integration weights on a triangular surface
/// Format is [Weights, U, V]
///
/// References:
/// * F. Hussain, M. S. Karim, and R. Ahamad, “Appropriate Gaussian quadrature formulae for triangles”.
const TABLE_GAUSS_LEGENDRE_3: [[f64; 3]; 9] = [
    [0.9876542474e-01, 0.2500000000e+00, 0.5000000000e+00],
    [0.1391378575e-01, 0.5635083269e-01, 0.8872983346e+00],
    [0.1095430035e+00, 0.4436491673e+00, 0.1127016654e+00],
    [0.6172839460e-01, 0.4436491673e+00, 0.5000000000e+00],
    [0.8696116674e-02, 0.1000000000e+00, 0.8872983346e+00],
    [0.6846438175e-01, 0.7872983346e+00, 0.1127016654e+00],
    [0.6172839460e-01, 0.5635083269e-01, 0.5000000000e+00],
    [0.8696116674e-02, 0.1270166538e-01, 0.8872983346e+00],
    [0.6846438175e-01, 0.1000000000e+00, 0.1127016654e+00],
];

/// Midpoint-rule samples used for the 1D edge integral in the Duffy-style
/// triangle self kernel.
const TRIANGLE_SELF_DUFFY_SAMPLES: usize = 16;

#[derive(Clone, Copy)]
pub enum QuadratureKind {
    GaussLegendre2,
    GaussLegendre3,
    Dunavant,
}

/// Isoparametric mapping of a point on a 3D triangle
/// from U-V coordinates on the triangle's surface.
#[inline]
pub fn map_tri_uv(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3], pin_uv: [f64; 2]) -> [f64; 3] {
    let mut pout = [0.0; 3];
    let w = 1.0 - pin_uv[0] - pin_uv[1];
    pout[0] = n0[0] * w + n1[0] * pin_uv[0] + n2[0] * pin_uv[1];
    pout[1] = n0[1] * w + n1[1] * pin_uv[0] + n2[1] * pin_uv[1];
    pout[2] = n0[2] * w + n1[2] * pin_uv[0] + n2[2] * pin_uv[1];
    pout
}

/// Area of a 3D triangle.
#[inline]
pub fn calc_tri_area(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> f64 {
    let v01 = [n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]];
    let v02 = [n2[0] - n0[0], n2[1] - n0[1], n2[2] - n0[2]];
    let cross = cross3(v01[0], v01[1], v01[2], v02[0], v02[1], v02[2]);
    0.5 * rss3(cross.0, cross.1, cross.2)
}

/// Normal vector of a triangle.
///
/// Direction is non-unique; the order of the points determines whether
/// the returned normal points "up" or "down" relative to the triangle.
#[inline]
pub fn calc_tri_normal(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> [f64; 3] {
    let v01 = [n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]];
    let v02 = [n2[0] - n0[0], n2[1] - n0[1], n2[2] - n0[2]];
    let cross = cross3(v01[0], v01[1], v01[2], v02[0], v02[1], v02[2]);
    let norm = rss3(cross.0, cross.1, cross.2);
    [cross.0 / norm, cross.1 / norm, cross.2 / norm]
}

#[inline]
fn triangle_basis_current_density(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> (f64, [f64; 3]) {
    let v01 = [n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]];
    let v02 = [n2[0] - n0[0], n2[1] - n0[1], n2[2] - n0[2]];
    let tri_area = calc_tri_area(n0, n1, n2);
    let jref = [
        (v02[0] - v01[0]) / (2.0 * tri_area),
        (v02[1] - v01[1]) / (2.0 * tri_area),
        (v02[2] - v01[2]) / (2.0 * tri_area),
    ];

    (tri_area, jref)
}

#[inline]
fn triangle_basis_current_densities(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> [[f64; 3]; 3] {
    [
        triangle_basis_current_density(n0, n1, n2).1,
        triangle_basis_current_density(n1, n2, n0).1,
        triangle_basis_current_density(n2, n0, n1).1,
    ]
}

#[inline]
fn triangle_quadrature_points(quad_kind: QuadratureKind) -> &'static [[f64; 3]] {
    match quad_kind {
        QuadratureKind::GaussLegendre2 => &TABLE_GAUSS_LEGENDRE_2,
        QuadratureKind::GaussLegendre3 => &TABLE_GAUSS_LEGENDRE_3,
        _ => panic!(),
    }
}

#[inline]
fn points_match(a: [f64; 3], b: [f64; 3]) -> bool {
    rss3(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1e-12
}

#[inline]
fn triangles_identical(
    src0: [f64; 3],
    src1: [f64; 3],
    src2: [f64; 3],
    tgt0: [f64; 3],
    tgt1: [f64; 3],
    tgt2: [f64; 3],
) -> bool {
    let src = [src0, src1, src2];
    let tgt = [tgt0, tgt1, tgt2];

    src.iter().all(|&s| tgt.iter().any(|&t| points_match(s, t)))
}
