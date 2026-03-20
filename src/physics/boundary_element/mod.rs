//! Boundary-element fields and formulas.
//!
//! References:
//! * \[1\] S. E. Mousavi and N. Sukumar, “Generalized Duffy transformation for integrating vertex singularities,” Comput Mech, vol. 45, no. 2–3, pp. 127–140, Jan. 2010, doi: 10.1007/s00466-009-0424-1.
//! * \[2\] R. D. Graglia, “On the numerical integration of the linear shape functions times the 3-D Green’s function or its gradient on a plane triangle,” IEEE Transactions on Antennas and Propagation, vol. 41, no. 10, pp. 1448–1455, Oct. 1993, doi: 10.1109/8.247786.
//! * \[3\] D. Wilton, S. Rao, A. Glisson, D. Schaubert, O. Al-Bundak, and C. Butler, “Potential integrals for uniform and linear source distributions on polygonal and polyhedral domains,” IEEE Transactions on Antennas and Propagation, vol. 32, no. 3, pp. 276–281, Mar. 1984, doi: 10.1109/TAP.1984.1143304.
//! * \[4\] M. G. Duffy, “Quadrature Over a Pyramid or Cube of Integrands with a Singularity at a Vertex,” SIAM Journal on Numerical Analysis, vol. 19, no. 6, pp. 1260–1262, 1982.
//! * \[5\] G. N. Peeren, “Stream function approach for determining optimal surface currents,” Phd Thesis 2 (Research NOT TU/e / Graduation TU/e), Technische Universiteit Eindhoven, Eindhoven, 2003. doi: 10.6100/IR570424.
//! * \[6\] F. Hussain, M. S. Karim, and R. Ahamad, “Appropriate Gaussian quadrature formulae for triangles”.
//! * \[7\] D. A. Dunavant, “High Degree Efficient Symmetrical Gaussian Quadrature Rules for the Triangle,” International Journal for Numerical Methods in Engineering, vol. 21, no. 6, pp. 1129-1148, 1985, doi: 10.1002/nme.1620210612.

use crate::math::{cross3, rss3};
use crate::mesh::{TriangleMeshView, validate_triangle_mesh_geometry};

mod body_force_density;
mod flux_density;
mod inductance;
mod vector_potential;

#[cfg(test)]
mod test;

pub use flux_density::{
    flux_density_triangle, flux_density_triangle_mesh, flux_density_triangle_mesh_mapping,
    flux_density_triangle_mesh_mapping_par, flux_density_triangle_mesh_par,
    triangle_flux_density_basis, triangle_mesh_flux_density_from_potential_vectors,
};

pub use body_force_density::{
    triangle_basis_force_block, triangle_force_from_potential_vectors,
    triangle_mesh_force_from_potential_vectors, triangle_mesh_force_mapping,
    triangle_mesh_force_mapping_from_circular_filaments,
    triangle_mesh_force_mapping_from_circular_filaments_par,
    triangle_mesh_force_mapping_from_dipoles, triangle_mesh_force_mapping_from_dipoles_par,
    triangle_mesh_force_mapping_from_linear_filaments,
    triangle_mesh_force_mapping_from_linear_filaments_par, triangle_mesh_force_mapping_par,
    triangle_mesh_self_force_mapping, triangle_mesh_self_force_mapping_par,
    triangle_mesh_triangle_forces_from_potential_vectors,
};

pub use inductance::{
    triangle_basis_mutual_inductance, triangle_basis_mutual_inductance_block,
    triangle_geometric_coupling, triangle_geometric_coupling_regular,
    triangle_inductance_from_potential_vectors,
    triangle_mesh_flux_linkage_from_source_coefficients,
    triangle_mesh_flux_linkage_mapping_from_dipoles,
    triangle_mesh_flux_linkage_mapping_from_dipoles_par,
    triangle_mesh_inductance_from_potential_vectors,
    triangle_mesh_inductance_mapping_from_circular_filaments,
    triangle_mesh_inductance_mapping_from_circular_filaments_par,
    triangle_mesh_inductance_mapping_from_linear_filaments,
    triangle_mesh_inductance_mapping_from_linear_filaments_par, triangle_mesh_inductance_matrix,
    triangle_mesh_inductance_matrix_par, triangle_mesh_inductive_energy,
    triangle_mesh_interaction_energy_from_source_coefficients,
};

pub use vector_potential::{
    triangle_mesh_vector_potential_from_potential_vectors, triangle_vector_potential_basis,
    vector_potential_triangle, vector_potential_triangle_mesh,
    vector_potential_triangle_mesh_mapping, vector_potential_triangle_mesh_mapping_par,
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

/// Dunavant's 7-point degree-5 symmetric quadrature rule on a triangle.
/// Format is [Weights, U, V].
///
/// The published rule is given in barycentric form and normalized so the
/// weights sum to 1 over a physical triangle area factor. This backend stores
/// quadrature weights in the same reference-triangle convention as the existing
/// Gauss-Legendre tables, so the published weights are halved here.
///
/// References:
/// * [7], Appendix II, rule with `p = 5`, `n_g = 7`.
const TABLE_DUNAVANT_7: [[f64; 3]; 7] = [
    [0.112500000000000, 0.333333333333333, 0.333333333333333],
    [0.066197076394253, 0.470142064105115, 0.470142064105115],
    [0.066197076394253, 0.059715871789770, 0.470142064105115],
    [0.066197076394253, 0.470142064105115, 0.059715871789770],
    [0.062969590272414, 0.101286507323456, 0.101286507323456],
    [0.062969590272414, 0.797426985353087, 0.101286507323456],
    [0.062969590272414, 0.101286507323456, 0.797426985353087],
];

/// Midpoint-rule samples used for the 1D edge integral in the Duffy-style
/// triangle self kernel.
const TRIANGLE_SELF_DUFFY_SAMPLES: usize = 16;

#[derive(Clone, Copy)]
pub enum QuadratureKind {
    GaussLegendre2,
    GaussLegendre3,
    Dunavant5,
}

/// Isoparametric mapping of a point on a 3D triangle
/// from U-V coordinates on the triangle's surface.
///
/// Args:
///     n0: Triangle vertex 0 coordinates `[x, y, z]` (m).
///     n1: Triangle vertex 1 coordinates `[x, y, z]` (m).
///     n2: Triangle vertex 2 coordinates `[x, y, z]` (m).
///     pin_uv: Reference-triangle coordinates `[u, v]` (dimensionless).
///
/// Returns:
///     Cartesian point `[x, y, z]` on the triangle surface (m).
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
///
/// Args:
///     n0: Triangle vertex 0 coordinates `[x, y, z]` (m).
///     n1: Triangle vertex 1 coordinates `[x, y, z]` (m).
///     n2: Triangle vertex 2 coordinates `[x, y, z]` (m).
///
/// Returns:
///     Triangle area (m^2).
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
///
/// Args:
///     n0: Triangle vertex 0 coordinates `[x, y, z]` (m).
///     n1: Triangle vertex 1 coordinates `[x, y, z]` (m).
///     n2: Triangle vertex 2 coordinates `[x, y, z]` (m).
///
/// Returns:
///     Unit normal vector `[nx, ny, nz]` (dimensionless).
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
    let v01 = [n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]]; // [m]
    let v02 = [n2[0] - n0[0], n2[1] - n0[1], n2[2] - n0[2]]; // [m]
    let tri_area = calc_tri_area(n0, n1, n2); // [m^2]
    let jref = [
        (v02[0] - v01[0]) / (2.0 * tri_area), // [1/m]
        (v02[1] - v01[1]) / (2.0 * tri_area), // [1/m]
        (v02[2] - v01[2]) / (2.0 * tri_area), // [1/m]
    ];

    (tri_area, jref)
}

#[inline]
fn triangle_basis_current_densities(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> [[f64; 3]; 3] {
    [
        triangle_basis_current_density(n0, n1, n2).1, // [1/m]
        triangle_basis_current_density(n1, n2, n0).1, // [1/m]
        triangle_basis_current_density(n2, n0, n1).1, // [1/m]
    ]
}

/// Physical surface current density induced on one triangle by its nodal
/// stream-function values.
///
/// Args:
///     n0: Triangle vertex 0 coordinates `[x, y, z]` (m).
///     n1: Triangle vertex 1 coordinates `[x, y, z]` (m).
///     n2: Triangle vertex 2 coordinates `[x, y, z]` (m).
///     s: Nodal current-potential values `[s0, s1, s2]` (A).
///
/// Returns:
///     Constant surface current density `[jx, jy, jz]` on the triangle (A/m).
#[inline]
pub fn triangle_current_density(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3], s: [f64; 3]) -> [f64; 3] {
    let basis = triangle_basis_current_densities(n0, n1, n2);
    [
        s[0] * basis[0][0] + s[1] * basis[1][0] + s[2] * basis[2][0], // [A/m]
        s[0] * basis[0][1] + s[1] * basis[1][1] + s[2] * basis[2][1], // [A/m]
        s[0] * basis[0][2] + s[1] * basis[1][2] + s[2] * basis[2][2], // [A/m]
    ]
}

#[inline]
fn triangle_quadrature_points(quad_kind: QuadratureKind) -> &'static [[f64; 3]] {
    match quad_kind {
        QuadratureKind::GaussLegendre2 => &TABLE_GAUSS_LEGENDRE_2,
        QuadratureKind::GaussLegendre3 => &TABLE_GAUSS_LEGENDRE_3,
        QuadratureKind::Dunavant5 => &TABLE_DUNAVANT_7,
    }
}

/// Number of quadrature points used by a given triangle rule.
///
/// Args:
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///
/// Returns:
///     Number of quadrature points in the selected rule (dimensionless).
#[inline]
pub fn triangle_quadrature_count(quad_kind: QuadratureKind) -> usize {
    triangle_quadrature_points(quad_kind).len()
}

/// Extract the constant physical surface current density on each triangle of a mesh.
///
/// Args:
///     nodes: Node-coordinate component slices `(x, y, z)` (m).
///     triangles: Triangle-node index component slices `(i0, i1, i2)` (dimensionless).
///     s: Nodal current-potential values (A).
///     out: Output buffers for triangle current-density components `(jx, jy, jz)` (A/m).
///
/// Returns:
///     `Ok(())` after writing one current-density vector per triangle to `out`, or an
///     error if the mesh geometry or output dimensions are inconsistent.
#[inline]
pub fn triangle_mesh_current_density(
    nodes: (&[f64], &[f64], &[f64]),
    triangles: (&[usize], &[usize], &[usize]),
    s: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let mesh = TriangleMeshView::new(nodes, triangles, s)?;
    let ntri = mesh.len();
    if out.0.len() != ntri || out.1.len() != ntri || out.2.len() != ntri {
        return Err("Output dimension mismatch");
    }

    for i in 0..ntri {
        let (tri_nodes, tri_s) = mesh.triangle_nodes(i);
        let j = triangle_current_density(tri_nodes[0], tri_nodes[1], tri_nodes[2], tri_s); // [A/m]
        out.0[i] = j[0]; // [A/m]
        out.1[i] = j[1]; // [A/m]
        out.2[i] = j[2]; // [A/m]
    }

    Ok(())
}

/// Extract physical quadrature-point coordinates and area weights for each triangle
/// in triangle-major order.
///
/// Args:
///     nodes: Node-coordinate component slices `(x, y, z)` (m).
///     triangles: Triangle-node index component slices `(i0, i1, i2)` (dimensionless).
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///     out: Output buffers for quadrature-point coordinates `(xq, yq, zq)` (m).
///     weights: Output buffer for physical quadrature weights `ΔS_q` (m^2).
///
/// Returns:
///     `Ok(())` after writing triangle-major quadrature coordinates and weights, or an
///     error if the mesh geometry or output dimensions are inconsistent.
#[inline]
pub fn triangle_mesh_quadrature_points(
    nodes: (&[f64], &[f64], &[f64]),
    triangles: (&[usize], &[usize], &[usize]),
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
    weights: &mut [f64],
) -> Result<(), &'static str> {
    let (_nnode, ntri) = validate_triangle_mesh_geometry(nodes, triangles)?;
    let quad_points = triangle_quadrature_points(quad_kind);
    let nqp = quad_points.len();
    let nout = ntri * nqp;
    if out.0.len() != nout || out.1.len() != nout || out.2.len() != nout || weights.len() != nout {
        return Err("Output dimension mismatch");
    }

    for i in 0..ntri {
        let n0 = [
            nodes.0[triangles.0[i]],
            nodes.1[triangles.0[i]],
            nodes.2[triangles.0[i]],
        ];
        let n1 = [
            nodes.0[triangles.1[i]],
            nodes.1[triangles.1[i]],
            nodes.2[triangles.1[i]],
        ];
        let n2 = [
            nodes.0[triangles.2[i]],
            nodes.1[triangles.2[i]],
            nodes.2[triangles.2[i]],
        ];
        let tri_area = calc_tri_area(n0, n1, n2); // [m^2]

        for (k, qp) in quad_points.iter().enumerate() {
            let idx = i * nqp + k;
            let point = map_tri_uv(n0, n1, n2, [qp[1], qp[2]]); // [m]
            out.0[idx] = point[0]; // [m]
            out.1[idx] = point[1]; // [m]
            out.2[idx] = point[2]; // [m]
            weights[idx] = qp[0] * tri_area; // [m^2]
        }
    }

    Ok(())
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
