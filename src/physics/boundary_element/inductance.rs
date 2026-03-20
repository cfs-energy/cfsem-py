use super::{
    QuadratureKind, TRIANGLE_SELF_DUFFY_SAMPLES, calc_tri_area, map_tri_uv,
    triangle_basis_current_densities, triangle_quadrature_points, triangles_identical,
};
use crate::MU0_OVER_4PI;
use crate::math::{dot3, rss3};

/// Regular triangle evaluation of the scalar kernel integral `∫ dS / R` using plain
/// quadrature.
///
/// References:
/// - [3], pp. 276-281, for `1 / R` potential integrals on flat polygonal elements.
/// - [2], pp. 1448-1455, for triangle integration of Green-function kernels with linear
///   shape functions.
#[inline]
fn triangle_scalar_potential_regular(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    obs: [f64; 3],
    quad_kind: QuadratureKind,
) -> f64 {
    let tri_area = calc_tri_area(n0, n1, n2); // [m^2]
    let quad_points = triangle_quadrature_points(quad_kind);

    let mut out = 0.0; // [m]
    for qp in quad_points {
        let src = map_tri_uv(n0, n1, n2, [qp[1], qp[2]]); // [m]
        let dist = rss3(obs[0] - src[0], obs[1] - src[1], obs[2] - src[2]); // [m]
        out += qp[0] * tri_area / dist; // [m]
    }

    out
}

/// Weakly singular single-triangle `∫ dS / R` evaluation for a target point on the
/// triangle itself.
///
/// Method:
/// - Split the parent triangle into three sub-triangles sharing `obs` as a vertex.
/// - On each sub-triangle, use a Duffy-style collapse of the radial coordinate so the
///   `1 / R` singularity is canceled by the surface Jacobian.
///     - Because the new triangles each end at `obs`, the local dS area (and local
///       contribution to current) of each triangle goes to zero linearly (like `R`) as it
///       approaches `obs`, while the vector potential becomes singular like `1/R`. So, the
///       local contribution to the field at `obs` now goes to `R/R` at `obs` instead of
///       diverging to a div/0.
/// - The remaining 1D integral along the opposite edge is smooth and is evaluated by a
///   midpoint rule.
///
/// References:
/// - [4], pp. 1260-1262, for the original Duffy transform for vertex singularities on
///   simplices.
/// - [1], abstract and Sec. 2, for the generalized Duffy mapping and the note that the
///   standard `1 / r` case corresponds to the classical Duffy choice.
/// - [2], pp. 1448-1455, and [3], pp. 276-281, for related weakly singular triangle
///   Green-function integrals.
#[inline]
fn triangle_scalar_potential_self_duffy(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    obs: [f64; 3],
) -> f64 {
    let edges = [(n0, n1), (n1, n2), (n2, n0)];
    let mut out = 0.0; // [m]

    // For each edge [nx, ny] in the triangle, treat [obs, nx, ny] as a
    // new sub-triangle for nonsingular integration.
    for (va, vb) in edges {
        let area_sub = calc_tri_area(obs, va, vb); // [m^2]
        if area_sub == 0.0 {
            continue;
        }

        // Integrate the transverse direction (across the triangle).
        let mut line_integral = 0.0; // [1/m]
        for i in 0..TRIANGLE_SELF_DUFFY_SAMPLES {
            let eta = (i as f64 + 0.5) / TRIANGLE_SELF_DUFFY_SAMPLES as f64;
            let edge_vec = [
                (1.0 - eta).mul_add(va[0] - obs[0], eta * (vb[0] - obs[0])),
                (1.0 - eta).mul_add(va[1] - obs[1], eta * (vb[1] - obs[1])),
                (1.0 - eta).mul_add(va[2] - obs[2], eta * (vb[2] - obs[2])),
            ];
            line_integral += 1.0 / rss3(edge_vec[0], edge_vec[1], edge_vec[2]); // [1/m]
        }

        out += area_sub * line_integral / TRIANGLE_SELF_DUFFY_SAMPLES as f64; // [m]
    }

    out
}

/// Double-surface geometric coupling
/// `∫_target ∫_source 1 / |r - r'| dS' dS`
/// for a well-separated triangle pair using plain nested quadrature.
///
/// References:
/// - [5], Eq. (3.16) on p. 68 for mutual inductance via `A · j`, Eq. (4.6) on p. 93
///   for constant triangle current density, and Eqs. (5.3)-(5.5) on pp. 107-108 for
///   triangle vector-potential integrals.
/// - [3], pp. 276-281.
/// - [2], pp. 1448-1455.
///
/// Args:
///     src0: Source triangle vertex 0 `[x, y, z]` (m).
///     src1: Source triangle vertex 1 `[x, y, z]` (m).
///     src2: Source triangle vertex 2 `[x, y, z]` (m).
///     tgt0: Target triangle vertex 0 `[x, y, z]` (m).
///     tgt1: Target triangle vertex 1 `[x, y, z]` (m).
///     tgt2: Target triangle vertex 2 `[x, y, z]` (m).
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///
/// Returns:
///     Double-surface geometric coupling `∫∫ dS' dS / R` (m^3).
#[inline]
pub fn triangle_geometric_coupling_regular(
    src0: [f64; 3],
    src1: [f64; 3],
    src2: [f64; 3],
    tgt0: [f64; 3],
    tgt1: [f64; 3],
    tgt2: [f64; 3],
    quad_kind: QuadratureKind,
) -> f64 {
    let tri_area_tgt = calc_tri_area(tgt0, tgt1, tgt2); // [m^2]
    let quad_points_tgt = triangle_quadrature_points(quad_kind);

    let mut out = 0.0; // [m^3]
    for qp in quad_points_tgt {
        let obs = map_tri_uv(tgt0, tgt1, tgt2, [qp[1], qp[2]]); // [m]
        out += qp[0]
            * tri_area_tgt
            * triangle_scalar_potential_regular(src0, src1, src2, obs, quad_kind); // [m^3]
    }

    out
}

/// Double-surface self coupling for a triangle.
///
/// Method:
/// - Integrate over target quadrature points on the triangle.
/// - At each target point, evaluate the source-side weakly singular `∫ dS / R` term
///   with the Duffy-style helper above.
///
/// References:
/// - [5], discussion on p. 106 and Eqs. (5.3)-(5.5) on pp. 107-108 for evaluation of
///   vector potential on the source support.
/// - [4], pp. 1260-1262.
/// - [1], abstract and Sec. 2.
/// - [2], pp. 1448-1455, and [3], pp. 276-281.
#[inline]
fn triangle_geometric_coupling_self(
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
    quad_kind: QuadratureKind,
) -> f64 {
    let tri_area = calc_tri_area(n0, n1, n2); // [m^2]
    let quad_points = triangle_quadrature_points(quad_kind);

    let mut out = 0.0; // [m^3]
    for qp in quad_points {
        let obs = map_tri_uv(n0, n1, n2, [qp[1], qp[2]]); // [m]
        out += qp[0] * tri_area * triangle_scalar_potential_self_duffy(n0, n1, n2, obs); // [m^3]
    }

    out
}

/// Double-surface geometric coupling
/// `∫_target ∫_source 1 / |r - r'| dS' dS`
/// between two triangles.
///
/// Method:
/// - Use the dedicated self-term path when the two triangles are identical.
/// - Otherwise evaluate the pair with regular nested quadrature in both
///   source/target directions and average the two results so the numerical coupling is
///   explicitly symmetric.
///
/// References:
/// - [5], Eq. (3.16) on p. 68 and Sec. 3.5.1 on p. 85 for the symmetry of mutual
///   inductance, together with Eqs. (5.3)-(5.5) on pp. 107-108 for triangle
///   vector-potential evaluation.
/// - [2], pp. 1448-1455.
/// - [3], pp. 276-281.
/// - [1] and [4], for weakly singular integration background for the dedicated self term.
///
/// Args:
///     src0: Source triangle vertex 0 `[x, y, z]` (m).
///     src1: Source triangle vertex 1 `[x, y, z]` (m).
///     src2: Source triangle vertex 2 `[x, y, z]` (m).
///     tgt0: Target triangle vertex 0 `[x, y, z]` (m).
///     tgt1: Target triangle vertex 1 `[x, y, z]` (m).
///     tgt2: Target triangle vertex 2 `[x, y, z]` (m).
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///
/// Returns:
///     Symmetric double-surface geometric coupling `∫∫ dS' dS / R` (m^3).
#[inline]
pub fn triangle_geometric_coupling(
    src0: [f64; 3],
    src1: [f64; 3],
    src2: [f64; 3],
    tgt0: [f64; 3],
    tgt1: [f64; 3],
    tgt2: [f64; 3],
    quad_kind: QuadratureKind,
) -> f64 {
    if triangles_identical(src0, src1, src2, tgt0, tgt1, tgt2) {
        return triangle_geometric_coupling_self(src0, src1, src2, quad_kind);
    }

    0.5 * (triangle_geometric_coupling_regular(src0, src1, src2, tgt0, tgt1, tgt2, quad_kind)
        + triangle_geometric_coupling_regular(tgt0, tgt1, tgt2, src0, src1, src2, quad_kind))
}

/// Mutual-inductance block for the three nodal basis functions on a source triangle and
/// the three nodal basis functions on a target triangle.
///
/// Method:
/// - Linear triangle basis current densities are constant over each triangle.
/// - Compute one scalar geometric coupling `G = ∫∫ 1 / R dS' dS`.
/// - Form the full `3x3` block as `μ0 / 4π * G * (K_src_i · K_tgt_j)`.
///
/// References:
/// - [5], Eq. (3.16) on p. 68 for `M_mn = ∬ A_m · j_n dS`, Eq. (3.24) on p. 70 for the
///   stream-function current representation, and Eq. (4.6) on p. 93 for the constant
///   current density induced by linear triangle nodal values.
/// - [3], pp. 276-281.
/// - [2], pp. 1448-1455.
///
/// Args:
///     src0: Source triangle vertex 0 `[x, y, z]` (m).
///     src1: Source triangle vertex 1 `[x, y, z]` (m).
///     src2: Source triangle vertex 2 `[x, y, z]` (m).
///     tgt0: Target triangle vertex 0 `[x, y, z]` (m).
///     tgt1: Target triangle vertex 1 `[x, y, z]` (m).
///     tgt2: Target triangle vertex 2 `[x, y, z]` (m).
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///
/// Returns:
///     Mutual-inductance block `[[M_ij]; 3]` for the source and target triangle bases (H).
#[inline]
pub fn triangle_basis_mutual_inductance_block(
    src0: [f64; 3],
    src1: [f64; 3],
    src2: [f64; 3],
    tgt0: [f64; 3],
    tgt1: [f64; 3],
    tgt2: [f64; 3],
    quad_kind: QuadratureKind,
) -> [[f64; 3]; 3] {
    let g = triangle_geometric_coupling(src0, src1, src2, tgt0, tgt1, tgt2, quad_kind);
    let ksrc = triangle_basis_current_densities(src0, src1, src2);
    let ktgt = triangle_basis_current_densities(tgt0, tgt1, tgt2);

    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = MU0_OVER_4PI
                * g
                * dot3(
                    ksrc[i][0], ksrc[i][1], ksrc[i][2], ktgt[j][0], ktgt[j][1], ktgt[j][2],
                );
        }
    }

    out
}

/// Single entry from the triangle-basis mutual-inductance block.
///
/// References:
/// - [5], Eq. (3.16) on p. 68, Eq. (3.24) on p. 70, and Eq. (4.6) on p. 93.
///
/// Args:
///     src0: Source triangle vertex 0 `[x, y, z]` (m).
///     src1: Source triangle vertex 1 `[x, y, z]` (m).
///     src2: Source triangle vertex 2 `[x, y, z]` (m).
///     src_basis: Source basis-function index in `{0, 1, 2}` (dimensionless).
///     tgt0: Target triangle vertex 0 `[x, y, z]` (m).
///     tgt1: Target triangle vertex 1 `[x, y, z]` (m).
///     tgt2: Target triangle vertex 2 `[x, y, z]` (m).
///     tgt_basis: Target basis-function index in `{0, 1, 2}` (dimensionless).
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///
/// Returns:
///     Triangle-basis mutual-inductance entry `M_ij` (H).
#[inline]
pub fn triangle_basis_mutual_inductance(
    src0: [f64; 3],
    src1: [f64; 3],
    src2: [f64; 3],
    src_basis: usize,
    tgt0: [f64; 3],
    tgt1: [f64; 3],
    tgt2: [f64; 3],
    tgt_basis: usize,
    quad_kind: QuadratureKind,
) -> f64 {
    triangle_basis_mutual_inductance_block(src0, src1, src2, tgt0, tgt1, tgt2, quad_kind)[src_basis]
        [tgt_basis]
}

/// Contract a triangle-pair inductance block with source and target nodal potential
/// vectors to obtain the total inductive coupling between the two triangle current
/// distributions.
///
/// References:
/// - [5], Eq. (3.16) on p. 68 for the mutual-inductance bilinear form, together with
///   Eq. (4.6) on p. 93 for the linear dependence of triangle current density on nodal
///   stream-function values.
///
/// Args:
///     m_block: Triangle-pair mutual-inductance block `[[M_ij]; 3]` (H).
///     s_src: Source nodal current-potential vector `[s0, s1, s2]` (A).
///     s_tgt: Target nodal current-potential vector `[s0, s1, s2]` (A).
///
/// Returns:
///     Bilinear coupling `s_src^T M s_tgt` (H*A^2).
#[inline]
pub fn triangle_inductance_from_potential_vectors(
    m_block: [[f64; 3]; 3],
    s_src: [f64; 3],
    s_tgt: [f64; 3],
) -> f64 {
    let mut out = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            out += s_src[i] * m_block[i][j] * s_tgt[j];
        }
    }
    out
}
