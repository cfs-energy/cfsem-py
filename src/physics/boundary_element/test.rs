use core::f64::consts::PI;

use super::{
    QuadratureKind, calc_tri_area, calc_tri_normal, flux_density_triangle,
    flux_density_triangle_mesh, flux_density_triangle_mesh_par, map_tri_uv,
    triangle_basis_current_densities, triangle_basis_mutual_inductance_block,
    triangle_inductance_from_potential_vectors, triangle_quadrature_points,
    triangle_vector_potential_basis, vector_potential_triangle, vector_potential_triangle_mesh,
    vector_potential_triangle_mesh_par,
};
use crate::MU0_OVER_4PI;
use crate::math::{cartesian_to_cylindrical, dot3};
use crate::physics::circular_filament::{
    flux_circular_filament_scalar, flux_density_circular_filament_cartesian_scalar,
    vector_potential_circular_filament_scalar,
};
use crate::testing::approx;

#[derive(Clone, Copy)]
struct TrianglePatch {
    nodes: [[f64; 3]; 3],
    s: [f64; 3],
}

struct TriangleMeshData {
    nodes: (Vec<f64>, Vec<f64>, Vec<f64>),
    triangles: (Vec<usize>, Vec<usize>, Vec<usize>),
    s: Vec<f64>,
}

fn circular_strip_triangles_at_z(
    radius: f64,
    height: f64,
    s0: f64,
    nphi: usize,
    z_center: f64,
) -> Vec<TrianglePatch> {
    assert!(nphi >= 3);

    let dphi = 2.0 * PI / nphi as f64;
    let mut tris = Vec::with_capacity(2 * nphi);

    for i in 0..nphi {
        let phi0 = i as f64 * dphi;
        let phi1 = (i + 1) as f64 * dphi;

        let lower0 = [
            radius * phi0.cos(),
            radius * phi0.sin(),
            z_center - height / 2.0,
        ];
        let lower1 = [
            radius * phi1.cos(),
            radius * phi1.sin(),
            z_center - height / 2.0,
        ];
        let upper0 = [
            radius * phi0.cos(),
            radius * phi0.sin(),
            z_center + height / 2.0,
        ];
        let upper1 = [
            radius * phi1.cos(),
            radius * phi1.sin(),
            z_center + height / 2.0,
        ];

        let tri0 = TrianglePatch {
            nodes: [lower0, lower1, upper1],
            s: [-s0, -s0, s0],
        };
        let tri1 = TrianglePatch {
            nodes: [lower0, upper1, upper0],
            s: [-s0, s0, s0],
        };

        let radial = [(phi0 + 0.5 * dphi).cos(), (phi0 + 0.5 * dphi).sin(), 0.0];
        for tri in [tri0, tri1] {
            let normal = calc_tri_normal(tri.nodes[0], tri.nodes[1], tri.nodes[2]);
            let alignment = normal[0] * radial[0] + normal[1] * radial[1];
            assert!(
                alignment > 0.0,
                "triangle winding is not radially consistent"
            );
            tris.push(tri);
        }
    }

    tris
}

fn circular_strip_triangles(radius: f64, height: f64, s0: f64, nphi: usize) -> Vec<TrianglePatch> {
    circular_strip_triangles_at_z(radius, height, s0, nphi, 0.0)
}

fn strip_flux_density(tris: &[TrianglePatch], obs: [f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for tri in tris {
        let contrib = flux_density_triangle(
            tri.nodes[0],
            tri.nodes[1],
            tri.nodes[2],
            tri.s,
            obs,
            QuadratureKind::GaussLegendre3,
        );
        out[0] += contrib[0];
        out[1] += contrib[1];
        out[2] += contrib[2];
    }
    out
}

fn strip_vector_potential(tris: &[TrianglePatch], obs: [f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for tri in tris {
        let contrib = vector_potential_triangle(
            tri.nodes[0],
            tri.nodes[1],
            tri.nodes[2],
            tri.s,
            obs,
            QuadratureKind::GaussLegendre3,
        );
        out[0] += contrib[0];
        out[1] += contrib[1];
        out[2] += contrib[2];
    }
    out
}

fn triangle_patches_to_mesh(tris: &[TrianglePatch]) -> TriangleMeshData {
    let mut nodes_xyz: Vec<[f64; 3]> = Vec::new();
    let mut node_s: Vec<f64> = Vec::new();
    let mut tri0 = Vec::with_capacity(tris.len());
    let mut tri1 = Vec::with_capacity(tris.len());
    let mut tri2 = Vec::with_capacity(tris.len());

    for tri in tris {
        let mut tri_idx = [0usize; 3];
        for local in 0..3 {
            let node = tri.nodes[local];
            if let Some(idx) = nodes_xyz.iter().position(|&existing| existing == node) {
                assert!(
                    approx(node_s[idx], tri.s[local], 0.0, 1e-14),
                    "shared node had inconsistent scalar value"
                );
                tri_idx[local] = idx;
            } else {
                nodes_xyz.push(node);
                node_s.push(tri.s[local]);
                tri_idx[local] = nodes_xyz.len() - 1;
            }
        }
        tri0.push(tri_idx[0]);
        tri1.push(tri_idx[1]);
        tri2.push(tri_idx[2]);
    }

    let mut x = Vec::with_capacity(nodes_xyz.len());
    let mut y = Vec::with_capacity(nodes_xyz.len());
    let mut z = Vec::with_capacity(nodes_xyz.len());
    for node in nodes_xyz {
        x.push(node[0]);
        y.push(node[1]);
        z.push(node[2]);
    }

    TriangleMeshData {
        nodes: (x, y, z),
        triangles: (tri0, tri1, tri2),
        s: node_s,
    }
}

fn obs_components(obs: &[[f64; 3]]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(obs.len());
    let mut y = Vec::with_capacity(obs.len());
    let mut z = Vec::with_capacity(obs.len());
    for point in obs {
        x.push(point[0]);
        y.push(point[1]);
        z.push(point[2]);
    }
    (x, y, z)
}

fn mesh_flux_density(mesh: &TriangleMeshData, obs: &[[f64; 3]], par: bool) -> Vec<[f64; 3]> {
    let obs_xyz = obs_components(obs);
    let mut bx = vec![0.0; obs.len()];
    let mut by = vec![0.0; obs.len()];
    let mut bz = vec![0.0; obs.len()];

    let result = match par {
        true => flux_density_triangle_mesh_par(
            (&obs_xyz.0, &obs_xyz.1, &obs_xyz.2),
            (&mesh.nodes.0, &mesh.nodes.1, &mesh.nodes.2),
            (&mesh.triangles.0, &mesh.triangles.1, &mesh.triangles.2),
            &mesh.s,
            QuadratureKind::GaussLegendre3,
            (&mut bx, &mut by, &mut bz),
        ),
        false => flux_density_triangle_mesh(
            (&obs_xyz.0, &obs_xyz.1, &obs_xyz.2),
            (&mesh.nodes.0, &mesh.nodes.1, &mesh.nodes.2),
            (&mesh.triangles.0, &mesh.triangles.1, &mesh.triangles.2),
            &mesh.s,
            QuadratureKind::GaussLegendre3,
            (&mut bx, &mut by, &mut bz),
        ),
    };
    result.unwrap();

    (0..obs.len()).map(|i| [bx[i], by[i], bz[i]]).collect()
}

fn mesh_vector_potential(mesh: &TriangleMeshData, obs: &[[f64; 3]], par: bool) -> Vec<[f64; 3]> {
    let obs_xyz = obs_components(obs);
    let mut ax = vec![0.0; obs.len()];
    let mut ay = vec![0.0; obs.len()];
    let mut az = vec![0.0; obs.len()];

    let result = match par {
        true => vector_potential_triangle_mesh_par(
            (&obs_xyz.0, &obs_xyz.1, &obs_xyz.2),
            (&mesh.nodes.0, &mesh.nodes.1, &mesh.nodes.2),
            (&mesh.triangles.0, &mesh.triangles.1, &mesh.triangles.2),
            &mesh.s,
            QuadratureKind::GaussLegendre3,
            (&mut ax, &mut ay, &mut az),
        ),
        false => vector_potential_triangle_mesh(
            (&obs_xyz.0, &obs_xyz.1, &obs_xyz.2),
            (&mesh.nodes.0, &mesh.nodes.1, &mesh.nodes.2),
            (&mesh.triangles.0, &mesh.triangles.1, &mesh.triangles.2),
            &mesh.s,
            QuadratureKind::GaussLegendre3,
            (&mut ax, &mut ay, &mut az),
        ),
    };
    result.unwrap();

    (0..obs.len()).map(|i| [ax[i], ay[i], az[i]]).collect()
}

fn max_abs_component(vectors: &[[f64; 3]]) -> f64 {
    vectors
        .iter()
        .flat_map(|v| v.iter())
        .map(|v| v.abs())
        .fold(0.0, f64::max)
}

fn triangle_nodes_for_basis(tri: [[f64; 3]; 3], basis_idx: usize) -> [[f64; 3]; 3] {
    match basis_idx {
        0 => [tri[0], tri[1], tri[2]],
        1 => [tri[1], tri[2], tri[0]],
        2 => [tri[2], tri[0], tri[1]],
        _ => panic!(),
    }
}

fn strip_mutual_inductance(src: &[TrianglePatch], tgt: &[TrianglePatch]) -> f64 {
    let mut out = 0.0;
    for source in src {
        for target in tgt {
            let block = triangle_basis_mutual_inductance_block(
                source.nodes[0],
                source.nodes[1],
                source.nodes[2],
                target.nodes[0],
                target.nodes[1],
                target.nodes[2],
                QuadratureKind::GaussLegendre3,
            );
            out += triangle_inductance_from_potential_vectors(block, source.s, target.s);
        }
    }
    out
}

#[test]
fn test_triangle_mesh_collection_matches_single_triangle_kernels() {
    let tri = TrianglePatch {
        nodes: [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.2, 0.8, 0.1]],
        s: [1.2, -0.4, 0.7],
    };
    let mesh = triangle_patches_to_mesh(&[tri]);
    let obs = [[0.3, -0.2, 1.1], [0.8, 0.4, 0.6], [-0.4, 0.7, 0.9]];

    let b_mesh = mesh_flux_density(&mesh, &obs, false);
    let b_mesh_par = mesh_flux_density(&mesh, &obs, true);
    let a_mesh = mesh_vector_potential(&mesh, &obs, false);
    let a_mesh_par = mesh_vector_potential(&mesh, &obs, true);

    for (i, point) in obs.iter().enumerate() {
        let b_direct = flux_density_triangle(
            tri.nodes[0],
            tri.nodes[1],
            tri.nodes[2],
            tri.s,
            *point,
            QuadratureKind::GaussLegendre3,
        );
        let a_direct = vector_potential_triangle(
            tri.nodes[0],
            tri.nodes[1],
            tri.nodes[2],
            tri.s,
            *point,
            QuadratureKind::GaussLegendre3,
        );

        for axis in 0..3 {
            assert!(
                approx(b_mesh[i][axis], b_direct[axis], 1e-12, 1e-14),
                "single-triangle mesh B mismatch at point {i}, axis {axis}"
            );
            assert!(
                approx(b_mesh_par[i][axis], b_direct[axis], 1e-12, 1e-14),
                "single-triangle mesh parallel B mismatch at point {i}, axis {axis}"
            );
            assert!(
                approx(a_mesh[i][axis], a_direct[axis], 1e-12, 1e-14),
                "single-triangle mesh A mismatch at point {i}, axis {axis}"
            );
            assert!(
                approx(a_mesh_par[i][axis], a_direct[axis], 1e-12, 1e-14),
                "single-triangle mesh parallel A mismatch at point {i}, axis {axis}"
            );
        }
    }
}

#[test]
fn test_triangle_basis_mutual_inductance_block_matches_vector_potential_for_disjoint_triangles() {
    let src = [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.2, 0.8, 0.0]];
    let tgt = [[0.3, -0.2, 1.1], [1.1, 0.1, 1.4], [0.2, 0.9, 1.2]];
    let quad_kind = QuadratureKind::GaussLegendre3;

    let block = triangle_basis_mutual_inductance_block(
        src[0], src[1], src[2], tgt[0], tgt[1], tgt[2], quad_kind,
    );
    let block_t = triangle_basis_mutual_inductance_block(
        tgt[0], tgt[1], tgt[2], src[0], src[1], src[2], quad_kind,
    );

    let tri_area_tgt = calc_tri_area(tgt[0], tgt[1], tgt[2]);
    let quad_points_tgt = triangle_quadrature_points(quad_kind);
    let ktgt = triangle_basis_current_densities(tgt[0], tgt[1], tgt[2]);

    for i in 0..3 {
        let src_basis = triangle_nodes_for_basis(src, i);
        for j in 0..3 {
            let mut via_a_dot_k = 0.0;
            for qp in quad_points_tgt {
                let obs = map_tri_uv(tgt[0], tgt[1], tgt[2], [qp[1], qp[2]]);
                let a_src = triangle_vector_potential_basis(
                    src_basis[0],
                    src_basis[1],
                    src_basis[2],
                    obs,
                    quad_kind,
                );
                via_a_dot_k += qp[0]
                    * tri_area_tgt
                    * MU0_OVER_4PI
                    * dot3(
                        a_src[0], a_src[1], a_src[2], ktgt[j][0], ktgt[j][1], ktgt[j][2],
                    );
            }

            assert!(
                approx(block[i][j], via_a_dot_k, 1e-10, 1e-12),
                "A·K mismatch for block[{i}][{j}]: direct={:.6e}, via_A={:.6e}",
                block[i][j],
                via_a_dot_k,
            );
            assert!(
                approx(block[i][j], block_t[j][i], 1e-10, 1e-12),
                "reciprocity mismatch for block[{i}][{j}]: M12={:.6e}, M21^T={:.6e}",
                block[i][j],
                block_t[j][i],
            );
        }
    }
}

#[test]
fn test_triangle_basis_self_inductance_block_is_symmetric_and_finite() {
    let tri = [[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.2, 0.9, 0.2]];
    let block = triangle_basis_mutual_inductance_block(
        tri[0],
        tri[1],
        tri[2],
        tri[0],
        tri[1],
        tri[2],
        QuadratureKind::GaussLegendre3,
    );

    let mut max_entry: f64 = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                block[i][j].is_finite(),
                "self block contains non-finite entry at ({i},{j})"
            );
            assert!(
                approx(block[i][j], block[j][i], 1e-10, 1e-12),
                "self block is not symmetric at ({i},{j}): {:.6e} vs {:.6e}",
                block[i][j],
                block[j][i],
            );
            max_entry = max_entry.max(block[i][j].abs());
        }
    }

    for s in [[1.0, -1.0, 0.0], [PI, -1.0, 0.5], [2.0, -0.75, -1.25]] {
        let energy_like = triangle_inductance_from_potential_vectors(block, s, s);
        assert!(
            energy_like > -(1e-10 * max_entry.max(1.0)),
            "self block is not positive semidefinite enough for s={s:?}: {:.6e}",
            energy_like,
        );
    }
}

#[test]
fn test_triangle_basis_mutual_inductance_touching_pairs_are_finite_and_reciprocal() {
    let tri0 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let shared_edge = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    let shared_vertex = [[0.0, 1.0, 0.0], [0.2, 1.7, 0.4], [-0.3, 1.4, -0.2]];

    for other in [shared_edge, shared_vertex] {
        let block12 = triangle_basis_mutual_inductance_block(
            tri0[0],
            tri0[1],
            tri0[2],
            other[0],
            other[1],
            other[2],
            QuadratureKind::GaussLegendre3,
        );
        let block21 = triangle_basis_mutual_inductance_block(
            other[0],
            other[1],
            other[2],
            tri0[0],
            tri0[1],
            tri0[2],
            QuadratureKind::GaussLegendre3,
        );

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    block12[i][j].is_finite(),
                    "touching-pair entry is non-finite at ({i},{j})"
                );
                assert!(
                    approx(block12[i][j], block21[j][i], 1e-8, 1e-11),
                    "touching-pair reciprocity mismatch at ({i},{j}): {:.6e} vs {:.6e}",
                    block12[i][j],
                    block21[j][i],
                );
            }
        }
    }
}

#[test]
fn test_triangle_strip_mutual_inductance_against_circular_filament() {
    let radius = 0.71;
    let height = radius * 1e-3;
    let nphi = 128;
    let current = 1.0;
    let z_src = -0.37;
    let z_tgt = 0.41;

    let strip_src = circular_strip_triangles_at_z(radius, height, current, nphi, z_src);
    let strip_tgt = circular_strip_triangles_at_z(radius, height, current, nphi, z_tgt);

    let m_strip = strip_mutual_inductance(&strip_src, &strip_tgt);
    let m_strip_reverse = strip_mutual_inductance(&strip_tgt, &strip_src);
    let m_loop = flux_circular_filament_scalar((radius, z_src, 1.0), (radius, z_tgt));

    assert!(
        approx(m_loop, m_strip, 1e-3, 1e-12),
        "strip mutual inductance mismatch: strip={:.6e}, circular={:.6e}",
        m_strip,
        m_loop,
    );
    assert!(
        approx(m_strip, m_strip_reverse, 1e-10, 1e-12),
        "strip reciprocity mismatch: M12={:.6e}, M21={:.6e}",
        m_strip,
        m_strip_reverse,
    );
}

#[test]
fn test_flux_density_triangle_circular_strip_matches_circular_filament_far_field() {
    let radius = 0.7312345987;
    let height = radius * 1e-3;
    let nphi = 256;

    let loop_current = 1.7;
    let s0 = loop_current;

    let strip = circular_strip_triangles(radius, height, s0, nphi);
    let strip_mesh = triangle_patches_to_mesh(&strip);
    let obs = [
        [2.70, 0.95, 0.85],
        [3.10, -1.15, 1.05],
        [3.45, 0.75, -1.25],
        [3.80, 1.30, 1.60],
        [4.20, -0.90, -1.55],
        [4.55, 1.10, -2.05],
    ];
    let b_mesh = mesh_flux_density(&strip_mesh, &obs, false);
    let b_mesh_par = mesh_flux_density(&strip_mesh, &obs, true);
    let a_mesh = mesh_vector_potential(&strip_mesh, &obs, false);
    let a_mesh_par = mesh_vector_potential(&strip_mesh, &obs, true);

    let mut b_strip = Vec::with_capacity(obs.len());
    let mut b_loop = Vec::with_capacity(obs.len());
    let mut a_strip = Vec::with_capacity(obs.len());
    let mut a_loop = Vec::with_capacity(obs.len());
    for point in obs {
        b_strip.push(strip_flux_density(&strip, point));
        let b_ref = flux_density_circular_filament_cartesian_scalar(
            (radius, 0.0, loop_current),
            (point[0], point[1], point[2]),
        );
        b_loop.push([b_ref.0, b_ref.1, b_ref.2]);

        a_strip.push(strip_vector_potential(&strip, point));
        let (r_obs, phi_obs, z_obs) = cartesian_to_cylindrical(point[0], point[1], point[2]);
        let a_phi =
            vector_potential_circular_filament_scalar((radius, 0.0, loop_current), (r_obs, z_obs));
        a_loop.push([-a_phi * libm::sin(phi_obs), a_phi * libm::cos(phi_obs), 0.0]);
    }

    let b_axis_names = ["Bx", "By", "Bz"];
    let a_axis_names = ["Ax", "Ay", "Az"];
    let bfield_rtol = 1e-3;
    let bfield_atol = max_abs_component(&b_loop) * 1e-12;
    let afield_rtol = 1e-3;
    let afield_atol = max_abs_component(&a_loop) * 1e-12;

    for i in 0..obs.len() {
        for axis in 0..3 {
            assert!(
                approx(b_loop[i][axis], b_strip[i][axis], bfield_rtol, bfield_atol),
                "{} mismatch at point {}: strip={:.6e}, reference={:.6e}, obs={:?}",
                b_axis_names[axis],
                i,
                b_strip[i][axis],
                b_loop[i][axis],
                obs[i],
            );
            assert!(
                approx(b_strip[i][axis], b_mesh[i][axis], 1e-12, bfield_atol),
                "{} mesh serial mismatch at point {}: mesh={:.6e}, strip={:.6e}, obs={:?}",
                b_axis_names[axis],
                i,
                b_mesh[i][axis],
                b_strip[i][axis],
                obs[i],
            );
            assert!(
                approx(b_strip[i][axis], b_mesh_par[i][axis], 1e-12, bfield_atol),
                "{} mesh parallel mismatch at point {}: mesh={:.6e}, strip={:.6e}, obs={:?}",
                b_axis_names[axis],
                i,
                b_mesh_par[i][axis],
                b_strip[i][axis],
                obs[i],
            );

            assert!(
                approx(a_loop[i][axis], a_strip[i][axis], afield_rtol, afield_atol),
                "{} mismatch at point {}: strip={:.6e}, reference={:.6e}, obs={:?}",
                a_axis_names[axis],
                i,
                a_strip[i][axis],
                a_loop[i][axis],
                obs[i],
            );
            assert!(
                approx(a_strip[i][axis], a_mesh[i][axis], 1e-12, afield_atol),
                "{} mesh serial mismatch at point {}: mesh={:.6e}, strip={:.6e}, obs={:?}",
                a_axis_names[axis],
                i,
                a_mesh[i][axis],
                a_strip[i][axis],
                obs[i],
            );
            assert!(
                approx(a_strip[i][axis], a_mesh_par[i][axis], 1e-12, afield_atol),
                "{} mesh parallel mismatch at point {}: mesh={:.6e}, strip={:.6e}, obs={:?}",
                a_axis_names[axis],
                i,
                a_mesh_par[i][axis],
                a_strip[i][axis],
                obs[i],
            );
        }
    }
}

#[test]
fn test_flux_density_triangle_circular_strip_matches_circular_filament_near_axis() {
    let radius = 0.7312345987;
    let height = radius * 1e-3;
    let nphi = 256;
    let loop_current = 1.7;
    let strip = circular_strip_triangles(radius, height, loop_current, nphi);
    let strip_mesh = triangle_patches_to_mesh(&strip);

    let mut obs = Vec::with_capacity(202);
    for i in -100..=100 {
        obs.push([1e-8, 0.0, i as f64 * 0.01]);
    }
    obs.push([0.0, 1e-8, 0.0]);

    let bz_rtol = 1e-3;
    let bxy_atol = 1e-12;
    let b_mesh = mesh_flux_density(&strip_mesh, &obs, false);
    let b_mesh_par = mesh_flux_density(&strip_mesh, &obs, true);

    for (i, point) in obs.iter().copied().enumerate() {
        let b_strip = strip_flux_density(&strip, point);
        let b_ref = flux_density_circular_filament_cartesian_scalar(
            (radius, 0.0, loop_current),
            (point[0], point[1], point[2]),
        );
        let b_loop = [b_ref.0, b_ref.1, b_ref.2];

        assert!(
            approx(b_loop[2], b_strip[2], bz_rtol, 1e-15),
            "Bz mismatch near axis: strip={:.6e}, reference={:.6e}, obs={:?}",
            b_strip[2],
            b_loop[2],
            point,
        );
        assert!(
            approx(b_strip[2], b_mesh[i][2], 1e-12, 1e-15),
            "mesh serial Bz mismatch near axis: mesh={:.6e}, strip={:.6e}, obs={:?}",
            b_mesh[i][2],
            b_strip[2],
            point,
        );
        assert!(
            approx(b_strip[2], b_mesh_par[i][2], 1e-12, 1e-15),
            "mesh parallel Bz mismatch near axis: mesh={:.6e}, strip={:.6e}, obs={:?}",
            b_mesh_par[i][2],
            b_strip[2],
            point,
        );
        assert!(
            b_strip[0].abs() <= bxy_atol,
            "Bx not near zero on axis: {:.6e} at {:?}",
            b_strip[0],
            point,
        );
        assert!(
            b_mesh[i][0].abs() <= bxy_atol && b_mesh_par[i][0].abs() <= bxy_atol,
            "mesh Bx not near zero on axis at {:?}",
            point,
        );
        assert!(
            b_strip[1].abs() <= bxy_atol,
            "By not near zero on axis: {:.6e} at {:?}",
            b_strip[1],
            point,
        );
        assert!(
            b_mesh[i][1].abs() <= bxy_atol && b_mesh_par[i][1].abs() <= bxy_atol,
            "mesh By not near zero on axis at {:?}",
            point,
        );
    }
}
