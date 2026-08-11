use super::triangle_potential::UniformTriangle;
use super::{
    QuadratureKind, calc_tri_area, map_tri_uv, triangle_basis_current_densities,
    triangle_quadrature_points, triangles_identical,
};
use crate::MU0_OVER_4PI;
use crate::chunksize;
use crate::math::{cartesian_to_cylindrical, dot3, norm3};
use crate::mesh::TriangleMeshView;
use crate::physics::circular_filament::vector_potential_circular_filament_scalar;
use crate::physics::linear_filament::vector_potential_linear_filament_scalar;
use crate::physics::point_source::dipole::vector_potential_dipole_scalar;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Regular triangle evaluation of the scalar kernel integral `∫ dS / R` using plain
/// quadrature.
///
/// References:
/// - \[3\], pp. 276-281, for `1 / R` potential integrals on flat polygonal elements.
/// - \[2\], pp. 1448-1455, for triangle integration of Green-function kernels with linear
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
        let dist = norm3([obs[0] - src[0], obs[1] - src[1], obs[2] - src[2]]); // [m]
        out += qp[0] * tri_area / dist; // [m]
    }

    out
}

/// Double-surface geometric coupling
/// `∫_target ∫_source 1 / |r - r'| dS' dS`
/// for a well-separated triangle pair using plain nested quadrature.
///
/// References:
/// - \[5\], Eq. (3.16) on p. 68 for mutual inductance via `A · j`, Eq. (4.6) on p. 93
///   for constant triangle current density, and Eqs. (5.3)-(5.5) on pp. 107-108 for
///   triangle vector-potential integrals.
/// - \[3\], pp. 276-281.
/// - \[2\], pp. 1448-1455.
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

#[inline]
fn triangle_geometric_coupling_regular_symmetric(
    source: [[f64; 3]; 3],
    target: [[f64; 3]; 3],
    quad_kind: QuadratureKind,
) -> f64 {
    0.5 * (triangle_geometric_coupling_regular(
        source[0], source[1], source[2], target[0], target[1], target[2], quad_kind,
    ) + triangle_geometric_coupling_regular(
        target[0], target[1], target[2], source[0], source[1], source[2], quad_kind,
    ))
}

/// Integrate the exact source-triangle potential over one target triangle.
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21).
#[inline]
pub(super) fn triangle_geometric_coupling_exact_directed(
    source: &UniformTriangle<f64>,
    target_nodes: [[f64; 3]; 3],
    quad_kind: QuadratureKind,
) -> f64 {
    let area = calc_tri_area(target_nodes[0], target_nodes[1], target_nodes[2]); // [m^2]
    let mut out = 0.0; // [m^3]
    for qp in triangle_quadrature_points(quad_kind) {
        let obs = map_tri_uv(
            target_nodes[0],
            target_nodes[1],
            target_nodes[2],
            [qp[1], qp[2]],
        ); // [m]
        out += qp[0] * area * source.scalar_potential(obs); // [m^3]
    }
    out
}

const COUPLING_NEAR_SUBDIVISION_DEPTH: usize = 2;

#[inline]
pub(super) fn longest_edge_bisection(nodes: [[f64; 3]; 3]) -> [[[f64; 3]; 3]; 2] {
    let lengths = [
        dot3(
            [
                nodes[1][0] - nodes[0][0],
                nodes[1][1] - nodes[0][1],
                nodes[1][2] - nodes[0][2],
            ],
            [
                nodes[1][0] - nodes[0][0],
                nodes[1][1] - nodes[0][1],
                nodes[1][2] - nodes[0][2],
            ],
        ),
        dot3(
            [
                nodes[2][0] - nodes[1][0],
                nodes[2][1] - nodes[1][1],
                nodes[2][2] - nodes[1][2],
            ],
            [
                nodes[2][0] - nodes[1][0],
                nodes[2][1] - nodes[1][1],
                nodes[2][2] - nodes[1][2],
            ],
        ),
        dot3(
            [
                nodes[0][0] - nodes[2][0],
                nodes[0][1] - nodes[2][1],
                nodes[0][2] - nodes[2][2],
            ],
            [
                nodes[0][0] - nodes[2][0],
                nodes[0][1] - nodes[2][1],
                nodes[0][2] - nodes[2][2],
            ],
        ),
    ];
    let edge = if lengths[1] > lengths[0] && lengths[1] >= lengths[2] {
        1
    } else if lengths[2] > lengths[0] {
        2
    } else {
        0
    };
    let i = edge;
    let j = (edge + 1) % 3;
    let k = (edge + 2) % 3;
    let midpoint = [
        0.5 * (nodes[i][0] + nodes[j][0]),
        0.5 * (nodes[i][1] + nodes[j][1]),
        0.5 * (nodes[i][2] + nodes[j][2]),
    ];
    [
        [nodes[i], midpoint, nodes[k]],
        [midpoint, nodes[j], nodes[k]],
    ]
}

#[inline]
fn point_triangle_distance_squared(point: [f64; 3], tri: [[f64; 3]; 3]) -> f64 {
    let closest = crate::mesh::elements::tri::tri3::closest_point(point, tri[0], tri[1], tri[2]);
    dot3(
        [
            point[0] - closest[0],
            point[1] - closest[1],
            point[2] - closest[2],
        ],
        [
            point[0] - closest[0],
            point[1] - closest[1],
            point[2] - closest[2],
        ],
    )
}

#[inline]
fn segment_distance_squared(p0: [f64; 3], p1: [f64; 3], q0: [f64; 3], q1: [f64; 3]) -> f64 {
    let d1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let d2 = [q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]];
    let r = [p0[0] - q0[0], p0[1] - q0[1], p0[2] - q0[2]];
    let a = dot3(d1, d1);
    let e = dot3(d2, d2);
    let b = dot3(d1, d2);
    let c = dot3(d1, r);
    let f = dot3(d2, r);
    let denominator = a * e - b * b;

    let mut s = if denominator > 0.0 {
        ((b * f - c * e) / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }

    let delta = [
        r[0] + s * d1[0] - t * d2[0],
        r[1] + s * d1[1] - t * d2[1],
        r[2] + s * d1[2] - t * d2[2],
    ];
    dot3(delta, delta)
}

#[inline]
fn triangle_distance_squared(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> f64 {
    let mut distance = f64::INFINITY;
    for vertex in a {
        distance = distance.min(point_triangle_distance_squared(vertex, b));
    }
    for vertex in b {
        distance = distance.min(point_triangle_distance_squared(vertex, a));
    }
    for edge_a in 0..3 {
        for edge_b in 0..3 {
            distance = distance.min(segment_distance_squared(
                a[edge_a],
                a[(edge_a + 1) % 3],
                b[edge_b],
                b[(edge_b + 1) % 3],
            ));
        }
    }
    distance
}

#[inline]
fn triangle_aabb_distance_squared(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> f64 {
    let mut distance = 0.0;
    for axis in 0..3 {
        let a_min = a[0][axis].min(a[1][axis]).min(a[2][axis]);
        let a_max = a[0][axis].max(a[1][axis]).max(a[2][axis]);
        let b_min = b[0][axis].min(b[1][axis]).min(b[2][axis]);
        let b_max = b[0][axis].max(b[1][axis]).max(b[2][axis]);
        let gap = (a_min - b_max).max(b_min - a_max).max(0.0);
        distance += gap * gap;
    }
    distance
}

#[inline]
fn triangle_max_edge_squared(nodes: [[f64; 3]; 3]) -> f64 {
    let mut maximum: f64 = 0.0;
    for edge in 0..3 {
        let delta = [
            nodes[(edge + 1) % 3][0] - nodes[edge][0],
            nodes[(edge + 1) % 3][1] - nodes[edge][1],
            nodes[(edge + 1) % 3][2] - nodes[edge][2],
        ];
        maximum = maximum.max(dot3(delta, delta));
    }
    maximum
}

/// Integrate an exact source-triangle potential over a target triangle after
/// a fixed number of longest-edge bisections.
///
/// The source surface integral is analytic. Each target leaf uses the fixed
/// seven-point Dunavant degree-5 rule, independent of the caller's far-field
/// quadrature selection.
///
/// Args:
///     source: Cached nondegenerate uniform source triangle.
///     target_nodes: Target triangle vertices in metres.
///     depth: Remaining fixed bisection levels.
///
/// Returns:
///     Directed geometric coupling in cubic metres.
///
/// References:
/// - D. R. Wilton, J. Rivero, W. A. Johnson, and F. Vipiana, “Evaluation of
///   Static Potential Integrals on Triangular Domains,” IEEE Access, vol. 8,
///   pp. 99806–99819, 2020. doi:10.1109/ACCESS.2020.2997287.
/// - D. A. Dunavant, “High Degree Efficient Symmetrical Gaussian Quadrature
///   Rules for the Triangle,” International Journal for Numerical Methods in
///   Engineering, vol. 21, no. 6, pp. 1129–1148, 1985.
///   doi:10.1002/nme.1620210612.
/// - N. A. Gumerov, S. Kaneko, and R. Duraiswami, “Analytical Galerkin
///   Boundary Integrals of Laplace Kernel Layer Potentials in R^3,” SIAM
///   Journal on Scientific Computing, vol. 46, no. 2, pp. A974–A997, 2024.
///   doi:10.1137/23M1547688. Provides an analytic double-integral formulation
///   suitable for independent validation.
pub(super) fn triangle_geometric_coupling_exact_fixed(
    source: &UniformTriangle<f64>,
    target_nodes: [[f64; 3]; 3],
    depth: usize,
) -> f64 {
    if depth == 0 {
        return triangle_geometric_coupling_exact_directed(
            source,
            target_nodes,
            QuadratureKind::Dunavant5,
        );
    }

    let children = longest_edge_bisection(target_nodes);
    triangle_geometric_coupling_exact_fixed(source, children[0], depth - 1)
        + triangle_geometric_coupling_exact_fixed(source, children[1], depth - 1)
}

/// Double-surface geometric coupling using the exact source-triangle potential
/// and fixed D5 target integration at subdivision depth 2 for self and near pairs.
/// Well-separated pairs use reciprocal nested quadrature; this keeps the
/// exact source potential focused on interactions that need near-singular treatment.
///
/// Near and self evaluations are directed from the supplied source to target.
/// Reversing a direct pair call can therefore change the approximation within
/// the 0.25% validation tolerance established for the representative near-pair
/// test suite. This is a validation bound, not a runtime error estimate. Mesh
/// assembly remains exactly symmetric because each computed upper-triangular
/// block is scattered with its transpose.
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for the exact source-triangle
///   potential and its limiting cases.
///
/// Args:
///     src0: Source triangle vertex 0 `[x, y, z]` (m).
///     src1: Source triangle vertex 1 `[x, y, z]` (m).
///     src2: Source triangle vertex 2 `[x, y, z]` (m).
///     tgt0: Target triangle vertex 0 `[x, y, z]` (m).
///     tgt1: Target triangle vertex 1 `[x, y, z]` (m).
///     tgt2: Target triangle vertex 2 `[x, y, z]` (m).
///     quad_kind: Quadrature rule for well-separated pairs; self and near pairs
///         always use fixed D5 integration (dimensionless).
///
/// Returns:
///     Directed geometric coupling in cubic metres.
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
    let source_nodes = [src0, src1, src2];
    let target_nodes = [tgt0, tgt1, tgt2];
    if triangles_identical(src0, src1, src2, tgt0, tgt1, tgt2) {
        let source = UniformTriangle::new(src0, src1, src2);
        return triangle_geometric_coupling_exact_fixed(
            &source,
            target_nodes,
            COUPLING_NEAR_SUBDIVISION_DEPTH,
        );
    }
    let max_edge_squared =
        triangle_max_edge_squared(source_nodes).max(triangle_max_edge_squared(target_nodes));
    if triangle_aabb_distance_squared(source_nodes, target_nodes) > max_edge_squared {
        return triangle_geometric_coupling_regular_symmetric(
            source_nodes,
            target_nodes,
            quad_kind,
        );
    }
    let pair_distance_sq = triangle_distance_squared(source_nodes, target_nodes);
    if pair_distance_sq > max_edge_squared {
        return triangle_geometric_coupling_regular_symmetric(
            source_nodes,
            target_nodes,
            quad_kind,
        );
    }
    let source = UniformTriangle::new(src0, src1, src2);
    triangle_geometric_coupling_exact_fixed(&source, target_nodes, COUPLING_NEAR_SUBDIVISION_DEPTH)
}

/// Mutual-inductance block for the three nodal basis functions on a source triangle and
/// the three nodal basis functions on a target triangle.
///
/// Method:
/// - Linear triangle basis current densities are constant over each triangle.
/// - Compute one scalar geometric coupling `G = ∫∫ 1 / R dS' dS`.
/// - Form the full `3x3` block as `μ0 / 4π * G * (K_src_i · K_tgt_j)`.
/// - This block is the elemental nodal-basis contribution used to assemble a full mesh
///   inductance matrix.
///
/// For self and near pairs this is a directed approximation: reversing source
/// and target can change the block within the representative-suite tolerance
/// described by [`triangle_geometric_coupling`].
///
/// References:
/// - \[5\], Eq. (3.16) on p. 68 for `M_mn = ∬ A_m · j_n dS`, Eq. (3.24) on p. 70 for the
///   stream-function current representation, and Eq. (4.6) on p. 93 for the constant
///   current density induced by linear triangle nodal values.
/// - \[3\], pp. 276-281.
/// - \[2\], pp. 1448-1455.
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for near and self interactions.
///
/// Args:
///     src0: Source triangle vertex 0 `[x, y, z]` (m).
///     src1: Source triangle vertex 1 `[x, y, z]` (m).
///     src2: Source triangle vertex 2 `[x, y, z]` (m).
///     tgt0: Target triangle vertex 0 `[x, y, z]` (m).
///     tgt1: Target triangle vertex 1 `[x, y, z]` (m).
///     tgt2: Target triangle vertex 2 `[x, y, z]` (m).
///     quad_kind: Quadrature rule for well-separated pairs; self and near pairs
///         always use fixed D5 integration (dimensionless).
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
            out[i][j] = MU0_OVER_4PI * g * dot3(ksrc[i], ktgt[j]);
        }
    }

    out
}

#[inline]
fn scatter_triangle_block(
    out: &mut [f64],
    nnode: usize,
    src_idx: [usize; 3],
    tgt_idx: [usize; 3],
    block: [[f64; 3]; 3],
) {
    for i in 0..3 {
        let row = src_idx[i] * nnode; // [-]
        for j in 0..3 {
            out[row + tgt_idx[j]] += block[i][j]; // [H]
        }
    }
}

#[inline]
fn scatter_triangle_block_transpose(
    out: &mut [f64],
    nnode: usize,
    src_idx: [usize; 3],
    tgt_idx: [usize; 3],
    block: [[f64; 3]; 3],
) {
    for i in 0..3 {
        for j in 0..3 {
            out[tgt_idx[j] * nnode + src_idx[i]] += block[i][j]; // [H]
        }
    }
}

#[inline]
fn validate_inductance_matrix_inputs(
    lmat: &[f64],
    s_src: &[f64],
    s_tgt: &[f64],
) -> Result<usize, &'static str> {
    let nnode = s_src.len(); // [-]
    if s_tgt.len() != nnode {
        return Err("Nodal scalar dimension mismatch");
    }
    if lmat.len() != nnode * nnode {
        return Err("Inductance matrix dimension mismatch");
    }
    Ok(nnode)
}

#[inline]
fn validate_inductance_mapping_inputs(
    map: &[f64],
    nnode_tgt: usize,
    nsrc: usize,
) -> Result<(), &'static str> {
    let expected = nnode_tgt
        .checked_mul(nsrc)
        .ok_or("Inductance mapping size overflow")?;
    if map.len() != expected {
        return Err("Output dimension mismatch");
    }
    Ok(())
}

#[inline]
fn validate_flux_linkage_mapping_vector_inputs(
    map: &[f64],
    coeffs_src: &[f64],
    out: &[f64],
) -> Result<(usize, usize), &'static str> {
    let nsrc = coeffs_src.len(); // [-]
    if nsrc == 0 {
        if map.is_empty() && out.is_empty() {
            return Ok((0, 0));
        }
        return Err("Source coefficient dimension mismatch");
    }
    if !map.len().is_multiple_of(nsrc) {
        return Err("Inductance mapping dimension mismatch");
    }
    let nnode_tgt = map.len() / nsrc; // [-]
    if out.len() != nnode_tgt {
        return Err("Flux-linkage output dimension mismatch");
    }
    Ok((nnode_tgt, nsrc))
}

#[inline]
fn triangle_mesh_source_mapping_from_vector_potential_fn<F>(
    mesh_tgt: &TriangleMeshView<'_>,
    nsrc: usize,
    quad_kind: QuadratureKind,
    out: &mut [f64],
    eval_a: F,
) -> Result<(), &'static str>
where
    F: Fn(usize, [f64; 3]) -> [f64; 3] + Sync,
{
    validate_inductance_mapping_inputs(out, mesh_tgt.nnode(), nsrc)?;

    out.fill(0.0); // [H] or source-dependent interaction units

    for itgt in 0..mesh_tgt.len() {
        let (tgt_nodes, tgt_idx) = mesh_tgt.triangle_nodes_and_indices(itgt);
        let tri_area = calc_tri_area(tgt_nodes[0], tgt_nodes[1], tgt_nodes[2]); // [m^2]
        let ktgt = triangle_basis_current_densities(tgt_nodes[0], tgt_nodes[1], tgt_nodes[2]); // [1/m]

        for qp in triangle_quadrature_points(quad_kind) {
            let obs = map_tri_uv(tgt_nodes[0], tgt_nodes[1], tgt_nodes[2], [qp[1], qp[2]]); // [m]
            let w = qp[0] * tri_area; // [m^2]

            for isrc in 0..nsrc {
                let a = eval_a(isrc, obs); // [V*s/(m*source-unit)]
                for ibasis in 0..3 {
                    out[tgt_idx[ibasis] * nsrc + isrc] += dot3(ktgt[ibasis], a) * w; // [H] or source-dependent interaction units
                }
            }
        }
    }

    Ok(())
}

#[inline]
fn triangle_mesh_source_mapping_from_vector_potential_fn_par<F>(
    mesh_tgt: &TriangleMeshView<'_>,
    nsrc: usize,
    quad_kind: QuadratureKind,
    out: &mut [f64],
    eval_a: F,
) -> Result<(), &'static str>
where
    F: Fn(usize, [f64; 3]) -> [f64; 3] + Sync + Send,
{
    let matrix_len = mesh_tgt
        .nnode()
        .checked_mul(nsrc)
        .ok_or("Inductance mapping size overflow")?;
    if out.len() != matrix_len {
        return Err("Output dimension mismatch");
    }

    let chunk = chunksize(mesh_tgt.len().max(1)); // [-]
    let starts: Vec<usize> = (0..mesh_tgt.len()).step_by(chunk).collect();
    let mut partial_buffers = Vec::with_capacity(starts.len());
    for _ in 0..starts.len() {
        let mut local = Vec::new();
        if local.try_reserve_exact(matrix_len).is_err() {
            return triangle_mesh_source_mapping_from_vector_potential_fn(
                mesh_tgt, nsrc, quad_kind, out, eval_a,
            );
        }
        local.resize(matrix_len, 0.0); // [H] or source-dependent interaction units
        partial_buffers.push(local);
    }

    let partials: Vec<Vec<f64>> = starts
        .into_par_iter()
        .zip(partial_buffers.into_par_iter())
        .map(|(start, mut local)| {
            let end = (start + chunk).min(mesh_tgt.len());
            for itgt in start..end {
                let (tgt_nodes, tgt_idx) = mesh_tgt.triangle_nodes_and_indices(itgt);
                let tri_area = calc_tri_area(tgt_nodes[0], tgt_nodes[1], tgt_nodes[2]); // [m^2]
                let ktgt =
                    triangle_basis_current_densities(tgt_nodes[0], tgt_nodes[1], tgt_nodes[2]); // [1/m]

                for qp in triangle_quadrature_points(quad_kind) {
                    let obs = map_tri_uv(tgt_nodes[0], tgt_nodes[1], tgt_nodes[2], [qp[1], qp[2]]); // [m]
                    let w = qp[0] * tri_area; // [m^2]

                    for isrc in 0..nsrc {
                        let a = eval_a(isrc, obs); // [V*s/(m*source-unit)]
                        for ibasis in 0..3 {
                            local[tgt_idx[ibasis] * nsrc + isrc] += dot3(ktgt[ibasis], a) * w; // [H] or source-dependent interaction units
                        }
                    }
                }
            }
            local
        })
        .collect();

    out.fill(0.0); // [H] or source-dependent interaction units
    for partial in partials {
        for (dst, val) in out.iter_mut().zip(partial) {
            *dst += val;
        }
    }

    Ok(())
}

/// Assemble the dense nodal-basis inductance matrix for one triangle mesh.
///
/// Method:
/// - Treat the triangle-pair `3x3` mutual-inductance block as the elemental
///   nodal-basis kernel.
/// - Loop over the upper triangle of source-target element pairs.
/// - Scatter-add each elemental block and its transpose into a row-major global
///   node-node matrix.
///
/// The resulting matrix acts on nodal current-potential values `s_a` and represents the
/// bilinear form
/// `L_ab = μ0 / 4π ∬ K_a(r) · K_b(r') / |r - r'| dS dS'`.
/// Matrix entries are exactly symmetric by construction. Because each near pair
/// is evaluated in upper-triangular element order, permuting mesh cells can change
/// its directed near-field approximation within the representative-suite 0.25%
/// validation tolerance.
///
/// Args:
///     mesh: Borrowed triangle-mesh geometry view.
///     quad_kind: Quadrature rule for well-separated pairs; self and near pairs
///         always use fixed D5 integration (dimensionless).
///     out: Row-major output matrix buffer of length `nnode * nnode` (H).
///
/// Returns:
///     `Ok(())` after writing the dense nodal inductance matrix to `out`, or an error if
///     the mesh geometry or output dimensions are inconsistent.
///
/// References:
/// - \[5\], Eq. (3.16) on p. 68, Eq. (3.24) on p. 70, and Eq. (4.6) on p. 93.
/// - \[3\], pp. 276-281.
/// - \[2\], pp. 1448-1455.
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for near and self interactions.
#[inline]
pub fn triangle_mesh_inductance_matrix(
    mesh: &TriangleMeshView<'_>,
    quad_kind: QuadratureKind,
    out: &mut [f64],
) -> Result<(), &'static str> {
    let nnode = mesh.nnode();
    if out.len() != nnode * nnode {
        return Err("Output dimension mismatch");
    }

    out.fill(0.0); // [H]

    for isrc in 0..mesh.len() {
        let (src_nodes, src_idx) = mesh.triangle_nodes_and_indices(isrc);
        for itgt in isrc..mesh.len() {
            let (tgt_nodes, tgt_idx) = mesh.triangle_nodes_and_indices(itgt);
            let block = triangle_basis_mutual_inductance_block(
                src_nodes[0],
                src_nodes[1],
                src_nodes[2],
                tgt_nodes[0],
                tgt_nodes[1],
                tgt_nodes[2],
                quad_kind,
            );
            scatter_triangle_block(out, nnode, src_idx, tgt_idx, block);
            if itgt != isrc {
                scatter_triangle_block_transpose(out, nnode, src_idx, tgt_idx, block);
            }
        }
    }

    Ok(())
}

/// Assemble the dense nodal-basis inductance matrix for one triangle mesh.
/// This variant is parallelized over chunks of source triangles and reduced into the
/// final dense matrix.
/// It has the same directed near-pair and structural-symmetry behavior as
/// [`triangle_mesh_inductance_matrix`].
///
/// If the per-worker scratch matrices cannot be allocated, this routine falls back to
/// the serial implementation rather than failing outright.
///
/// Args:
///     mesh: Borrowed triangle-mesh geometry view.
///     quad_kind: Quadrature rule for well-separated pairs; self and near pairs
///         always use fixed D5 integration (dimensionless).
///     out: Row-major output matrix buffer of length `nnode * nnode` (H).
///
/// Returns:
///     `Ok(())` after writing the dense nodal inductance matrix to `out`, or an error if
///     the mesh geometry or output dimensions are inconsistent.
///
/// References:
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for near and self interactions.
#[inline]
pub fn triangle_mesh_inductance_matrix_par(
    mesh: &TriangleMeshView<'_>,
    quad_kind: QuadratureKind,
    out: &mut [f64],
) -> Result<(), &'static str> {
    let nnode = mesh.nnode();
    let matrix_len = mesh
        .nnode()
        .checked_mul(mesh.nnode())
        .ok_or("Inductance matrix size overflow")?;
    if out.len() != matrix_len {
        return Err("Output dimension mismatch");
    }

    let chunk = chunksize(mesh.len().max(1)); // [-]
    let starts: Vec<usize> = (0..mesh.len()).step_by(chunk).collect();
    let mut partial_buffers = Vec::with_capacity(starts.len());
    for _ in 0..starts.len() {
        let mut local = Vec::new();
        if local.try_reserve_exact(matrix_len).is_err() {
            return triangle_mesh_inductance_matrix(mesh, quad_kind, out);
        }
        local.resize(matrix_len, 0.0); // [H]
        partial_buffers.push(local);
    }

    let partials: Vec<Vec<f64>> = starts
        .into_par_iter()
        .zip(partial_buffers.into_par_iter())
        .map(|(start, mut local)| {
            let end = (start + chunk).min(mesh.len());
            for isrc in start..end {
                let (src_nodes, src_idx) = mesh.triangle_nodes_and_indices(isrc);
                for itgt in isrc..mesh.len() {
                    let (tgt_nodes, tgt_idx) = mesh.triangle_nodes_and_indices(itgt);
                    let block = triangle_basis_mutual_inductance_block(
                        src_nodes[0],
                        src_nodes[1],
                        src_nodes[2],
                        tgt_nodes[0],
                        tgt_nodes[1],
                        tgt_nodes[2],
                        quad_kind,
                    );
                    scatter_triangle_block(&mut local, nnode, src_idx, tgt_idx, block);
                    if itgt != isrc {
                        scatter_triangle_block_transpose(
                            &mut local, nnode, src_idx, tgt_idx, block,
                        );
                    }
                }
            }
            local
        })
        .collect();

    out.fill(0.0); // [H]
    for partial in partials {
        for (dst, val) in out.iter_mut().zip(partial) {
            *dst += val; // [H]
        }
    }

    Ok(())
}

/// Contract a dense nodal inductance matrix with source and target nodal current-potential
/// vectors.
///
/// Args:
///     lmat: Row-major nodal inductance matrix of length `nnode * nnode` (H).
///     s_src: Source nodal current-potential values (A).
///     s_tgt: Target nodal current-potential values (A).
///
/// Returns:
///     Bilinear inductive coupling `s_src^T L s_tgt` (H*A^2 = J).
///     This can be used to compute the mutual inductive energy between two collections
///     of nodes represented in the same nodal vector space.
#[inline]
pub fn triangle_mesh_inductance_from_potential_vectors(
    lmat: &[f64],
    s_src: &[f64],
    s_tgt: &[f64],
) -> Result<f64, &'static str> {
    let nnode = validate_inductance_matrix_inputs(lmat, s_src, s_tgt)?;

    let mut out = 0.0; // [H*A^2]
    for i in 0..nnode {
        let row = &lmat[i * nnode..(i + 1) * nnode];
        for j in 0..nnode {
            out += s_src[i] * row[j] * s_tgt[j]; // [H*A^2]
        }
    }

    Ok(out)
}

/// Internal helper used by boundary-element tests of dense inductance contractions.
///
/// Args:
///     lmat: Row-major nodal inductance matrix of length `nnode * nnode` (H).
///     s: Nodal current-potential values (A).
///
/// Returns:
///     Magnetic energy `0.5 * s^T L s` (J).
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
pub(crate) fn triangle_mesh_inductive_energy(lmat: &[f64], s: &[f64]) -> Result<f64, &'static str> {
    Ok(0.5 * triangle_mesh_inductance_from_potential_vectors(lmat, s, s)?) // [J]
}

/// Apply a dense source-to-node flux-linkage or inductance mapping to source coefficients.
///
/// Args:
///     map: Row-major mapping of length `nnode_tgt * nsrc` in `(target node, source index)` order.
///     coeffs_src: Source coefficients.
///     out: Output nodal flux-linkage vector.
///
/// Returns:
///     `Ok(())` after writing the target nodal flux-linkage vector to `out`, or an error if
///     the mapping dimensions are inconsistent.
#[inline]
pub fn triangle_mesh_flux_linkage_from_source_coefficients(
    map: &[f64],
    coeffs_src: &[f64],
    out: &mut [f64],
) -> Result<(), &'static str> {
    let (nnode_tgt, nsrc) = validate_flux_linkage_mapping_vector_inputs(map, coeffs_src, out)?;
    if nsrc == 0 {
        return Ok(());
    }

    for inode in 0..nnode_tgt {
        let row = &map[inode * nsrc..(inode + 1) * nsrc];
        out[inode] = 0.0;
        for isrc in 0..nsrc {
            out[inode] += row[isrc] * coeffs_src[isrc];
        }
    }

    Ok(())
}

/// Interaction energy obtained by contracting a source-to-node mapping with target nodal
/// current-potential values and source coefficients.
///
/// Args:
///     map: Row-major mapping of length `nnode_tgt * nsrc` in `(target node, source index)` order.
///     s_tgt: Target nodal current-potential values.
///     coeffs_src: Source coefficients.
///
/// Returns:
///     Interaction energy `s_tgt^T (map @ coeffs_src)`.
#[inline]
pub fn triangle_mesh_interaction_energy_from_source_coefficients(
    map: &[f64],
    s_tgt: &[f64],
    coeffs_src: &[f64],
) -> Result<f64, &'static str> {
    let (nnode_tgt, nsrc) = validate_flux_linkage_mapping_vector_inputs(map, coeffs_src, s_tgt)?;
    if nsrc == 0 {
        return Ok(0.0);
    }

    let mut out = 0.0;
    for inode in 0..nnode_tgt {
        let row = &map[inode * nsrc..(inode + 1) * nsrc];
        let mut psi = 0.0;
        for isrc in 0..nsrc {
            psi += row[isrc] * coeffs_src[isrc];
        }
        out += s_tgt[inode] * psi;
    }

    Ok(out)
}

/// Assemble the source-current to target-node inductance mapping from linear filaments.
///
/// Args:
///     xyzfil: Filament segment start coordinates `(x, y, z)` (m).
///     dlxyzfil: Filament segment deltas `(dx, dy, dz)` (m).
///     wire_radius: Filament radii (m).
///     mesh_tgt: Borrowed target triangle-mesh geometry view.
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///     out: Row-major output mapping buffer of length `nnode_tgt * nfil` (H).
///
/// Returns:
///     `Ok(())` after writing the inductance mapping to `out`, or an error if the source,
///     target, or output dimensions are inconsistent.
#[inline]
pub fn triangle_mesh_inductance_mapping_from_linear_filaments(
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    mesh_tgt: &TriangleMeshView<'_>,
    quad_kind: QuadratureKind,
    out: &mut [f64],
) -> Result<(), &'static str> {
    let nfil = xyzfil.0.len(); // [-]
    if xyzfil.1.len() != nfil
        || xyzfil.2.len() != nfil
        || dlxyzfil.0.len() != nfil
        || dlxyzfil.1.len() != nfil
        || dlxyzfil.2.len() != nfil
        || wire_radius.len() != nfil
    {
        return Err("Source dimension mismatch");
    }

    triangle_mesh_source_mapping_from_vector_potential_fn(
        mesh_tgt,
        nfil,
        quad_kind,
        out,
        |ifil, obs| {
            let start = (xyzfil.0[ifil], xyzfil.1[ifil], xyzfil.2[ifil]);
            let end = (
                xyzfil.0[ifil] + dlxyzfil.0[ifil],
                xyzfil.1[ifil] + dlxyzfil.1[ifil],
                xyzfil.2[ifil] + dlxyzfil.2[ifil],
            );
            let a = vector_potential_linear_filament_scalar(
                (start, end, 1.0),
                wire_radius[ifil],
                (obs[0], obs[1], obs[2]),
            );
            [a.0, a.1, a.2]
        },
    )
}

/// Parallel variant of [`triangle_mesh_inductance_mapping_from_linear_filaments`].
#[inline]
pub fn triangle_mesh_inductance_mapping_from_linear_filaments_par(
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    mesh_tgt: &TriangleMeshView<'_>,
    quad_kind: QuadratureKind,
    out: &mut [f64],
) -> Result<(), &'static str> {
    let nfil = xyzfil.0.len(); // [-]
    if xyzfil.1.len() != nfil
        || xyzfil.2.len() != nfil
        || dlxyzfil.0.len() != nfil
        || dlxyzfil.1.len() != nfil
        || dlxyzfil.2.len() != nfil
        || wire_radius.len() != nfil
    {
        return Err("Source dimension mismatch");
    }

    triangle_mesh_source_mapping_from_vector_potential_fn_par(
        mesh_tgt,
        nfil,
        quad_kind,
        out,
        |ifil, obs| {
            let start = (xyzfil.0[ifil], xyzfil.1[ifil], xyzfil.2[ifil]);
            let end = (
                xyzfil.0[ifil] + dlxyzfil.0[ifil],
                xyzfil.1[ifil] + dlxyzfil.1[ifil],
                xyzfil.2[ifil] + dlxyzfil.2[ifil],
            );
            let a = vector_potential_linear_filament_scalar(
                (start, end, 1.0),
                wire_radius[ifil],
                (obs[0], obs[1], obs[2]),
            );
            [a.0, a.1, a.2]
        },
    )
}

/// Assemble the source-current to target-node inductance mapping from circular filaments.
///
/// Args:
///     rfil: Circular filament radii (m).
///     zfil: Circular filament axial coordinates (m).
///     mesh_tgt: Borrowed target triangle-mesh geometry view.
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///     out: Row-major output mapping buffer of length `nnode_tgt * nfil` (H).
///
/// Returns:
///     `Ok(())` after writing the inductance mapping to `out`, or an error if the source,
///     target, or output dimensions are inconsistent.
#[inline]
pub fn triangle_mesh_inductance_mapping_from_circular_filaments(
    rfil: &[f64],
    zfil: &[f64],
    mesh_tgt: &TriangleMeshView<'_>,
    quad_kind: QuadratureKind,
    out: &mut [f64],
) -> Result<(), &'static str> {
    let nfil = rfil.len(); // [-]
    if zfil.len() != nfil {
        return Err("Source dimension mismatch");
    }

    triangle_mesh_source_mapping_from_vector_potential_fn(
        mesh_tgt,
        nfil,
        quad_kind,
        out,
        |ifil, obs| {
            let [robs, phiobs, zobs] = cartesian_to_cylindrical(obs);
            let a_phi = vector_potential_circular_filament_scalar(
                (rfil[ifil], zfil[ifil], 1.0),
                (robs, zobs),
            );
            [-a_phi * libm::sin(phiobs), a_phi * libm::cos(phiobs), 0.0]
        },
    )
}

/// Parallel variant of [`triangle_mesh_inductance_mapping_from_circular_filaments`].
#[inline]
pub fn triangle_mesh_inductance_mapping_from_circular_filaments_par(
    rfil: &[f64],
    zfil: &[f64],
    mesh_tgt: &TriangleMeshView<'_>,
    quad_kind: QuadratureKind,
    out: &mut [f64],
) -> Result<(), &'static str> {
    let nfil = rfil.len(); // [-]
    if zfil.len() != nfil {
        return Err("Source dimension mismatch");
    }

    triangle_mesh_source_mapping_from_vector_potential_fn_par(
        mesh_tgt,
        nfil,
        quad_kind,
        out,
        |ifil, obs| {
            let [robs, phiobs, zobs] = cartesian_to_cylindrical(obs);
            let a_phi = vector_potential_circular_filament_scalar(
                (rfil[ifil], zfil[ifil], 1.0),
                (robs, zobs),
            );
            [-a_phi * libm::sin(phiobs), a_phi * libm::cos(phiobs), 0.0]
        },
    )
}

/// Assemble the source-amplitude to target-node flux-linkage mapping from dipoles.
///
/// Args:
///     loc: Dipole locations `(x, y, z)` (m).
///     moment_dir: Dipole moment direction vectors `(mx, my, mz)`.
///     outer_radius: Dipole finite-core radii (m).
///     mesh_tgt: Borrowed target triangle-mesh geometry view.
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///     out: Row-major output mapping buffer of length `nnode_tgt * ndip`.
///
/// Returns:
///     `Ok(())` after writing the flux-linkage mapping to `out`, or an error if the source,
///     target, or output dimensions are inconsistent.
#[inline]
pub fn triangle_mesh_flux_linkage_mapping_from_dipoles(
    loc: (&[f64], &[f64], &[f64]),
    moment_dir: (&[f64], &[f64], &[f64]),
    outer_radius: &[f64],
    mesh_tgt: &TriangleMeshView<'_>,
    quad_kind: QuadratureKind,
    out: &mut [f64],
) -> Result<(), &'static str> {
    let ndip = loc.0.len(); // [-]
    if loc.1.len() != ndip
        || loc.2.len() != ndip
        || moment_dir.0.len() != ndip
        || moment_dir.1.len() != ndip
        || moment_dir.2.len() != ndip
        || outer_radius.len() != ndip
    {
        return Err("Source dimension mismatch");
    }

    triangle_mesh_source_mapping_from_vector_potential_fn(
        mesh_tgt,
        ndip,
        quad_kind,
        out,
        |idip, obs| {
            let a = vector_potential_dipole_scalar(
                (loc.0[idip], loc.1[idip], loc.2[idip]),
                (moment_dir.0[idip], moment_dir.1[idip], moment_dir.2[idip]),
                outer_radius[idip],
                (obs[0], obs[1], obs[2]),
            );
            [a.0, a.1, a.2]
        },
    )
}

/// Parallel variant of [`triangle_mesh_flux_linkage_mapping_from_dipoles`].
#[inline]
pub fn triangle_mesh_flux_linkage_mapping_from_dipoles_par(
    loc: (&[f64], &[f64], &[f64]),
    moment_dir: (&[f64], &[f64], &[f64]),
    outer_radius: &[f64],
    mesh_tgt: &TriangleMeshView<'_>,
    quad_kind: QuadratureKind,
    out: &mut [f64],
) -> Result<(), &'static str> {
    let ndip = loc.0.len(); // [-]
    if loc.1.len() != ndip
        || loc.2.len() != ndip
        || moment_dir.0.len() != ndip
        || moment_dir.1.len() != ndip
        || moment_dir.2.len() != ndip
        || outer_radius.len() != ndip
    {
        return Err("Source dimension mismatch");
    }

    triangle_mesh_source_mapping_from_vector_potential_fn_par(
        mesh_tgt,
        ndip,
        quad_kind,
        out,
        |idip, obs| {
            let a = vector_potential_dipole_scalar(
                (loc.0[idip], loc.1[idip], loc.2[idip]),
                (moment_dir.0[idip], moment_dir.1[idip], moment_dir.2[idip]),
                outer_radius[idip],
                (obs[0], obs[1], obs[2]),
            );
            [a.0, a.1, a.2]
        },
    )
}

/// Single entry from the triangle-basis mutual-inductance block.
///
/// References:
/// - \[5\], Eq. (3.16) on p. 68, Eq. (3.24) on p. 70, and Eq. (4.6) on p. 93.
/// - \[8\], Eqs. (5)-(9), (15), and (19)-(21), for near and self interactions.
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
///     quad_kind: Quadrature rule for well-separated pairs; self and near pairs
///         always use fixed D5 integration (dimensionless).
///
/// Returns:
///     Triangle-basis mutual-inductance entry `M_ij` (H).
#[inline]
#[allow(clippy::too_many_arguments)]
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
/// - \[5\], Eq. (3.16) on p. 68 for the mutual-inductance bilinear form, together with
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
