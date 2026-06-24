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

use crate::math::Scalar;
use crate::math::norm3;
use crate::mesh::TriangleMeshView;
pub use crate::mesh::elements::tri::mapping::{
    area as calc_tri_area, map_point as map_tri_uv, normal as calc_tri_normal,
};
pub use crate::mesh::elements::tri::quadrature::{QuadratureKind, triangle_quadrature_count};
pub(crate) use crate::mesh::elements::tri::quadrature::{
    TRIANGLE_MAX_QUADRATURE_POINTS, triangle_quadrature_points,
};

mod body_force_density;
mod flux_density;
mod inductance;
mod vector_potential;

#[cfg(test)]
mod test;

/// Near-field distance threshold relative to the triangle's maximum edge length
/// for one level of closest-point subdivision in the B- and A-field kernels.
pub(crate) const TRIANGLE_NEAR_SUBDIVISION_DISTANCE_FACTOR: f64 = 1.0;

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
    triangle_mesh_inductance_matrix_par, triangle_mesh_interaction_energy_from_source_coefficients,
};

pub use vector_potential::{
    triangle_mesh_vector_potential_from_potential_vectors, triangle_vector_potential_basis,
    vector_potential_triangle, vector_potential_triangle_mesh,
    vector_potential_triangle_mesh_mapping, vector_potential_triangle_mesh_mapping_par,
    vector_potential_triangle_mesh_par,
};

/// Midpoint-rule samples used for the 1D edge integral in the Duffy-style
/// triangle self kernel.
const TRIANGLE_SELF_DUFFY_SAMPLES: usize = 16;

#[inline]
/// Return the triangle area and constant basis-current density vector.
fn triangle_basis_current_density<T: Scalar>(n0: [T; 3], n1: [T; 3], n2: [T; 3]) -> (T, [T; 3]) {
    let v01 = [n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]]; // [m]
    let v02 = [n2[0] - n0[0], n2[1] - n0[1], n2[2] - n0[2]]; // [m]
    let tri_area = calc_tri_area(n0, n1, n2); // [m^2]
    let two_area = tri_area + tri_area; // [m^2]
    let jref = [
        (v01[0] - v02[0]) / two_area, // [1/m]
        (v01[1] - v02[1]) / two_area, // [1/m]
        (v01[2] - v02[2]) / two_area, // [1/m]
    ];

    (tri_area, jref)
}

/// Constant surface-current-density vectors induced by the three nodal basis
/// functions on one triangle.
///
/// Each returned vector is the physical surface current density generated by a
/// unit nodal stream-function value on one triangle node and zero values on the
/// other two nodes. The stream-function convention is `J_s = n_hat x grad(phi)`,
/// where a stream-function jump across a cut equals the total current through
/// that cut.
///
/// Args:
///     n0: Triangle vertex 0 coordinates `[x, y, z]` (m).
///     n1: Triangle vertex 1 coordinates `[x, y, z]` (m).
///     n2: Triangle vertex 2 coordinates `[x, y, z]` (m).
///
/// Returns:
///     Three basis current-density vectors `[[jx, jy, jz]; 3]` [1/m], ordered
///     to match nodal basis functions `(n0, n1, n2)`.
#[inline]
pub fn triangle_basis_current_densities<T: Scalar>(
    n0: [T; 3],
    n1: [T; 3],
    n2: [T; 3],
) -> [[T; 3]; 3] {
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
pub fn triangle_current_density<T: Scalar>(
    n0: [T; 3],
    n1: [T; 3],
    n2: [T; 3],
    s: [T; 3],
) -> [T; 3] {
    let basis = triangle_basis_current_densities(n0, n1, n2);
    [
        s[0] * basis[0][0] + s[1] * basis[1][0] + s[2] * basis[2][0], // [A/m]
        s[0] * basis[0][1] + s[1] * basis[1][1] + s[2] * basis[2][1], // [A/m]
        s[0] * basis[0][2] + s[1] * basis[1][2] + s[2] * basis[2][2], // [A/m]
    ]
}

/// Extract the constant physical surface current density on each triangle of a mesh.
///
/// Args:
///     mesh: Borrowed triangle-mesh geometry view.
///     s: Nodal current-potential values (A).
///     out: Output buffers for triangle current-density components `(jx, jy, jz)` (A/m).
///
/// Returns:
///     `Ok(())` after writing one current-density vector per triangle to `out`, or an
///     error if the mesh geometry or output dimensions are inconsistent.
#[inline]
pub fn triangle_mesh_current_density(
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    mesh.validate_nodal_values(s)?;
    let ntri = mesh.len();
    if out.0.len() != ntri || out.1.len() != ntri || out.2.len() != ntri {
        return Err("Output dimension mismatch");
    }

    for i in 0..ntri {
        let tri_nodes = mesh.triangle_nodes(i);
        let tri_s = mesh.triangle_scalars(i, s);
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
///     mesh: Borrowed triangle-mesh geometry view.
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///     out: Output buffers for quadrature-point coordinates `(xq, yq, zq)` (m).
///     weights: Output buffer for physical quadrature weights `ΔS_q` (m^2).
///
/// Returns:
///     `Ok(())` after writing triangle-major quadrature coordinates and weights, or an
///     error if the mesh geometry or output dimensions are inconsistent.
#[inline]
pub fn triangle_mesh_quadrature_points(
    mesh: &TriangleMeshView<'_>,
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
    weights: &mut [f64],
) -> Result<(), &'static str> {
    let quad_points = triangle_quadrature_points(quad_kind);
    let nqp = quad_points.len();
    let nout = mesh.len() * nqp;
    if out.0.len() != nout || out.1.len() != nout || out.2.len() != nout || weights.len() != nout {
        return Err("Output dimension mismatch");
    }

    for i in 0..mesh.len() {
        let [n0, n1, n2] = mesh.triangle_nodes(i);
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

/// Return whether two points match to the geometric tolerance.
#[inline]
fn points_match(a: [f64; 3], b: [f64; 3]) -> bool {
    norm3([a[0] - b[0], a[1] - b[1], a[2] - b[2]]) < 1e-12
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
