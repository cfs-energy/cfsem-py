use core::f64::consts::PI;

use super::{
    QuadratureKind, calc_tri_area, calc_tri_normal, flux_density_triangle,
    flux_density_triangle_mesh, flux_density_triangle_mesh_par, map_tri_uv,
    triangle_basis_current_densities, triangle_basis_current_density, triangle_basis_force_block,
    triangle_basis_mutual_inductance_block, triangle_current_density,
    triangle_force_from_potential_vectors, triangle_inductance_from_potential_vectors,
    triangle_mesh_current_density, triangle_mesh_force_from_potential_vectors,
    triangle_mesh_force_mapping, triangle_mesh_force_mapping_par,
    triangle_mesh_inductance_from_potential_vectors, triangle_mesh_inductance_matrix,
    triangle_mesh_inductance_matrix_par, triangle_mesh_inductive_energy,
    triangle_mesh_quadrature_points, triangle_mesh_self_force_mapping,
    triangle_mesh_self_force_mapping_par, triangle_quadrature_count, triangle_quadrature_points,
    triangle_vector_potential_basis, vector_potential_triangle, vector_potential_triangle_mesh,
    vector_potential_triangle_mesh_par,
};
use crate::math::{cartesian_to_cylindrical, cross3, dot3};
use crate::physics::circular_filament::{
    flux_circular_filament_scalar, flux_density_circular_filament_cartesian_scalar,
    vector_potential_circular_filament_scalar,
};
use crate::physics::point_source::current_element::{
    flux_density_current_element_scalar, vector_potential_current_element_scalar,
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

fn mesh_inductance_matrix(mesh: &TriangleMeshData, par: bool) -> Vec<f64> {
    let nnode = mesh.nodes.0.len();
    let mut out = vec![0.0; nnode * nnode];
    let result = match par {
        true => triangle_mesh_inductance_matrix_par(
            (&mesh.nodes.0, &mesh.nodes.1, &mesh.nodes.2),
            (&mesh.triangles.0, &mesh.triangles.1, &mesh.triangles.2),
            QuadratureKind::GaussLegendre3,
            &mut out,
        ),
        false => triangle_mesh_inductance_matrix(
            (&mesh.nodes.0, &mesh.nodes.1, &mesh.nodes.2),
            (&mesh.triangles.0, &mesh.triangles.1, &mesh.triangles.2),
            QuadratureKind::GaussLegendre3,
            &mut out,
        ),
    };
    result.unwrap();
    out
}

fn mesh_force_mapping(
    mesh_src: &TriangleMeshData,
    mesh_tgt: &TriangleMeshData,
    par: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nnode_src = mesh_src.nodes.0.len();
    let ntri_tgt = mesh_tgt.triangles.0.len();
    let nout = nnode_src * ntri_tgt;
    let (mut fx, mut fy, mut fz) = (vec![0.0; nout], vec![0.0; nout], vec![0.0; nout]);

    let result = match par {
        true => triangle_mesh_force_mapping_par(
            (&mesh_src.nodes.0, &mesh_src.nodes.1, &mesh_src.nodes.2),
            (
                &mesh_src.triangles.0,
                &mesh_src.triangles.1,
                &mesh_src.triangles.2,
            ),
            (&mesh_tgt.nodes.0, &mesh_tgt.nodes.1, &mesh_tgt.nodes.2),
            (
                &mesh_tgt.triangles.0,
                &mesh_tgt.triangles.1,
                &mesh_tgt.triangles.2,
            ),
            &mesh_tgt.s,
            QuadratureKind::GaussLegendre3,
            (&mut fx, &mut fy, &mut fz),
        ),
        false => triangle_mesh_force_mapping(
            (&mesh_src.nodes.0, &mesh_src.nodes.1, &mesh_src.nodes.2),
            (
                &mesh_src.triangles.0,
                &mesh_src.triangles.1,
                &mesh_src.triangles.2,
            ),
            (&mesh_tgt.nodes.0, &mesh_tgt.nodes.1, &mesh_tgt.nodes.2),
            (
                &mesh_tgt.triangles.0,
                &mesh_tgt.triangles.1,
                &mesh_tgt.triangles.2,
            ),
            &mesh_tgt.s,
            QuadratureKind::GaussLegendre3,
            (&mut fx, &mut fy, &mut fz),
        ),
    };
    result.unwrap();

    (fx, fy, fz)
}

fn mesh_self_force_mapping(mesh: &TriangleMeshData, par: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nnode = mesh.nodes.0.len();
    let ntri = mesh.triangles.0.len();
    let nout = nnode * ntri;
    let (mut fx, mut fy, mut fz) = (vec![0.0; nout], vec![0.0; nout], vec![0.0; nout]);

    let result = match par {
        true => triangle_mesh_self_force_mapping_par(
            (&mesh.nodes.0, &mesh.nodes.1, &mesh.nodes.2),
            (&mesh.triangles.0, &mesh.triangles.1, &mesh.triangles.2),
            &mesh.s,
            QuadratureKind::GaussLegendre3,
            (&mut fx, &mut fy, &mut fz),
        ),
        false => triangle_mesh_self_force_mapping(
            (&mesh.nodes.0, &mesh.nodes.1, &mesh.nodes.2),
            (&mesh.triangles.0, &mesh.triangles.1, &mesh.triangles.2),
            &mesh.s,
            QuadratureKind::GaussLegendre3,
            (&mut fx, &mut fy, &mut fz),
        ),
    };
    result.unwrap();

    (fx, fy, fz)
}

fn explicit_force_on_target_triangle_from_source_mesh(
    mesh_src: &TriangleMeshData,
    tri_nodes: [[f64; 3]; 3],
    tri_s: [f64; 3],
) -> [f64; 3] {
    let tri_area = calc_tri_area(tri_nodes[0], tri_nodes[1], tri_nodes[2]);
    let k_tgt = triangle_current_density(tri_nodes[0], tri_nodes[1], tri_nodes[2], tri_s);
    let mut out = [0.0; 3];
    for qp in triangle_quadrature_points(QuadratureKind::GaussLegendre3) {
        let obs = map_tri_uv(tri_nodes[0], tri_nodes[1], tri_nodes[2], [qp[1], qp[2]]);
        let mut bx = [0.0];
        let mut by = [0.0];
        let mut bz = [0.0];
        flux_density_triangle_mesh(
            (&[obs[0]], &[obs[1]], &[obs[2]]),
            (&mesh_src.nodes.0, &mesh_src.nodes.1, &mesh_src.nodes.2),
            (
                &mesh_src.triangles.0,
                &mesh_src.triangles.1,
                &mesh_src.triangles.2,
            ),
            &mesh_src.s,
            QuadratureKind::GaussLegendre3,
            (&mut bx, &mut by, &mut bz),
        )
        .unwrap();
        let jf = cross3(k_tgt[0], k_tgt[1], k_tgt[2], bx[0], by[0], bz[0]);
        let w = qp[0] * tri_area;
        out[0] += jf.0 * w;
        out[1] += jf.1 * w;
        out[2] += jf.2 * w;
    }
    out
}

fn combine_disconnected_meshes(
    src: &TriangleMeshData,
    tgt: &TriangleMeshData,
) -> (TriangleMeshData, Vec<f64>, Vec<f64>) {
    let nsrc = src.nodes.0.len();
    let ntgt = tgt.nodes.0.len();

    let mut nodes = (
        Vec::with_capacity(nsrc + ntgt),
        Vec::with_capacity(nsrc + ntgt),
        Vec::with_capacity(nsrc + ntgt),
    );
    nodes.0.extend_from_slice(&src.nodes.0);
    nodes.0.extend_from_slice(&tgt.nodes.0);
    nodes.1.extend_from_slice(&src.nodes.1);
    nodes.1.extend_from_slice(&tgt.nodes.1);
    nodes.2.extend_from_slice(&src.nodes.2);
    nodes.2.extend_from_slice(&tgt.nodes.2);

    let mut triangles = (
        Vec::with_capacity(src.triangles.0.len() + tgt.triangles.0.len()),
        Vec::with_capacity(src.triangles.1.len() + tgt.triangles.1.len()),
        Vec::with_capacity(src.triangles.2.len() + tgt.triangles.2.len()),
    );
    triangles.0.extend_from_slice(&src.triangles.0);
    triangles.1.extend_from_slice(&src.triangles.1);
    triangles.2.extend_from_slice(&src.triangles.2);
    triangles
        .0
        .extend(tgt.triangles.0.iter().map(|&idx| idx + nsrc));
    triangles
        .1
        .extend(tgt.triangles.1.iter().map(|&idx| idx + nsrc));
    triangles
        .2
        .extend(tgt.triangles.2.iter().map(|&idx| idx + nsrc));

    let mut s = Vec::with_capacity(nsrc + ntgt);
    s.extend_from_slice(&src.s);
    s.extend_from_slice(&tgt.s);

    let mut s_src = vec![0.0; nsrc + ntgt];
    s_src[..nsrc].copy_from_slice(&src.s);
    let mut s_tgt = vec![0.0; nsrc + ntgt];
    s_tgt[nsrc..].copy_from_slice(&tgt.s);

    (
        TriangleMeshData {
            nodes,
            triangles,
            s,
        },
        s_src,
        s_tgt,
    )
}

fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0, |acc, k| acc * k as f64)
}

fn reference_triangle_monomial_integral(p: usize, q: usize) -> f64 {
    factorial(p) * factorial(q) / factorial(p + q + 2)
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
fn test_triangle_basis_fields_match_current_element_quadrature_sum() {
    let tri = [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.2, 0.8, 0.1]];
    let obs = [0.35, -0.22, 1.15];
    let quad_kind = QuadratureKind::GaussLegendre3;

    let (tri_area, jref) = triangle_basis_current_density(tri[0], tri[1], tri[2]);
    let mut b_via_elements = [0.0; 3];
    let mut a_via_elements = [0.0; 3];

    for qp in triangle_quadrature_points(quad_kind) {
        let src = map_tri_uv(tri[0], tri[1], tri[2], [qp[1], qp[2]]);
        let moment = [
            jref[0] * qp[0] * tri_area,
            jref[1] * qp[0] * tri_area,
            jref[2] * qp[0] * tri_area,
        ];
        let bq = flux_density_current_element_scalar(src, moment, obs);
        let aq = vector_potential_current_element_scalar(src, moment, obs);
        for axis in 0..3 {
            b_via_elements[axis] += bq[axis];
            a_via_elements[axis] += aq[axis];
        }
    }

    let b_basis = super::triangle_flux_density_basis(tri[0], tri[1], tri[2], obs, quad_kind);
    let a_basis = triangle_vector_potential_basis(tri[0], tri[1], tri[2], obs, quad_kind);

    for axis in 0..3 {
        assert!(
            approx(b_basis[axis], b_via_elements[axis], 0.0, 1e-15),
            "basis B/current-element mismatch at axis {axis}: basis={:.16e}, via_elements={:.16e}",
            b_basis[axis],
            b_via_elements[axis],
        );
        assert!(
            approx(a_basis[axis], a_via_elements[axis], 0.0, 1e-15),
            "basis A/current-element mismatch at axis {axis}: basis={:.16e}, via_elements={:.16e}",
            a_basis[axis],
            a_via_elements[axis],
        );
    }
}

#[test]
fn test_triangle_mesh_quadrature_points_and_current_density_extractors() {
    let tris = circular_strip_triangles(0.73, 7.3e-4, 1.7, 24);
    let mesh = triangle_patches_to_mesh(&tris);
    let quad_kind = QuadratureKind::GaussLegendre2;
    let ntri = mesh.triangles.0.len();
    let nqp = triangle_quadrature_count(quad_kind);

    let (mut jx, mut jy, mut jz) = (vec![0.0; ntri], vec![0.0; ntri], vec![0.0; ntri]);
    triangle_mesh_current_density(
        (&mesh.nodes.0, &mesh.nodes.1, &mesh.nodes.2),
        (&mesh.triangles.0, &mesh.triangles.1, &mesh.triangles.2),
        &mesh.s,
        (&mut jx, &mut jy, &mut jz),
    )
    .unwrap();

    let (mut xq, mut yq, mut zq) = (
        vec![0.0; ntri * nqp],
        vec![0.0; ntri * nqp],
        vec![0.0; ntri * nqp],
    );
    let mut wq = vec![0.0; ntri * nqp];
    triangle_mesh_quadrature_points(
        (&mesh.nodes.0, &mesh.nodes.1, &mesh.nodes.2),
        (&mesh.triangles.0, &mesh.triangles.1, &mesh.triangles.2),
        quad_kind,
        (&mut xq, &mut yq, &mut zq),
        &mut wq,
    )
    .unwrap();

    let quad_points = triangle_quadrature_points(quad_kind);
    for (i, tri) in tris.iter().enumerate() {
        let j_expected = triangle_current_density(tri.nodes[0], tri.nodes[1], tri.nodes[2], tri.s);
        for axis in 0..3 {
            let j = [jx[i], jy[i], jz[i]][axis];
            assert!(
                approx(j, j_expected[axis], 1e-13, 1e-14),
                "triangle current density mismatch for triangle {i}, axis {axis}"
            );
        }

        let tri_area = calc_tri_area(tri.nodes[0], tri.nodes[1], tri.nodes[2]);
        for (k, qp) in quad_points.iter().enumerate() {
            let idx = i * nqp + k;
            let expected_point =
                map_tri_uv(tri.nodes[0], tri.nodes[1], tri.nodes[2], [qp[1], qp[2]]);
            for axis in 0..3 {
                let q = [xq[idx], yq[idx], zq[idx]][axis];
                assert!(
                    approx(q, expected_point[axis], 0.0, 1e-14),
                    "quadrature point mismatch for triangle {i}, point {k}, axis {axis}"
                );
            }
            assert!(
                approx(wq[idx], qp[0] * tri_area, 0.0, 1e-14),
                "quadrature weight mismatch for triangle {i}, point {k}"
            );
        }
    }
}

#[test]
fn test_dunavant_rule_integrates_reference_triangle_monomials_to_degree_five() {
    let quad_points = triangle_quadrature_points(QuadratureKind::Dunavant5);

    assert_eq!(quad_points.len(), 7);

    for p in 0..=5 {
        for q in 0..=(5 - p) {
            let approx_int = quad_points
                .iter()
                .map(|qp| qp[0] * qp[1].powi(p as i32) * qp[2].powi(q as i32))
                .sum::<f64>();
            let exact_int = reference_triangle_monomial_integral(p, q);
            assert!(
                approx(approx_int, exact_int, 0.0, 1e-14),
                "Dunavant rule failed for u^{p} v^{q}: approx={approx_int:.16e}, exact={exact_int:.16e}"
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
fn test_triangle_mesh_inductance_matrix_matches_single_triangle_block() {
    let tri = TrianglePatch {
        nodes: [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.2, 0.8, 0.1]],
        s: [1.2, -0.4, 0.7],
    };
    let mesh = triangle_patches_to_mesh(&[tri]);
    let block = triangle_basis_mutual_inductance_block(
        tri.nodes[0],
        tri.nodes[1],
        tri.nodes[2],
        tri.nodes[0],
        tri.nodes[1],
        tri.nodes[2],
        QuadratureKind::GaussLegendre3,
    );
    let lmat = mesh_inductance_matrix(&mesh, false);
    let lmat_par = mesh_inductance_matrix(&mesh, true);

    for i in 0..3 {
        for j in 0..3 {
            let idx = i * 3 + j;
            assert!(
                approx(lmat[idx], block[i][j], 1e-12, 1e-14),
                "single-triangle nodal matrix mismatch at ({i},{j}): matrix={:.16e}, block={:.16e}",
                lmat[idx],
                block[i][j],
            );
            assert!(
                approx(lmat_par[idx], block[i][j], 1e-12, 1e-14),
                "single-triangle parallel nodal matrix mismatch at ({i},{j}): matrix={:.16e}, block={:.16e}",
                lmat_par[idx],
                block[i][j],
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
fn test_triangle_mesh_inductance_matrix_is_symmetric_and_matches_direct_contraction() {
    let patches = [
        TrianglePatch {
            nodes: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            s: [1.2, -0.7, 0.4],
        },
        TrianglePatch {
            nodes: [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            s: [1.2, 0.4, -0.9],
        },
    ];
    let mesh = triangle_patches_to_mesh(&patches);
    let lmat = mesh_inductance_matrix(&mesh, false);
    let lmat_par = mesh_inductance_matrix(&mesh, true);
    let nnode = mesh.s.len();

    for i in 0..nnode {
        for j in 0..nnode {
            let idx = i * nnode + j;
            let idx_t = j * nnode + i;
            assert!(
                approx(lmat[idx], lmat[idx_t], 1e-10, 1e-12),
                "global nodal matrix is not symmetric at ({i},{j}): {:.6e} vs {:.6e}",
                lmat[idx],
                lmat[idx_t],
            );
            assert!(
                approx(lmat[idx], lmat_par[idx], 1e-12, 1e-14),
                "serial/parallel nodal matrix mismatch at ({i},{j}): serial={:.16e}, parallel={:.16e}",
                lmat[idx],
                lmat_par[idx],
            );
        }
    }

    let direct = strip_mutual_inductance(&patches, &patches);
    let via_l = triangle_mesh_inductance_from_potential_vectors(&lmat, &mesh.s, &mesh.s).unwrap();
    let energy = triangle_mesh_inductive_energy(&lmat, &mesh.s).unwrap();

    assert!(
        approx(via_l, direct, 1e-10, 1e-12),
        "global nodal bilinear form mismatch: matrix={:.6e}, direct={:.6e}",
        via_l,
        direct,
    );
    assert!(
        approx(energy, 0.5 * direct, 1e-10, 1e-12),
        "global nodal energy mismatch: matrix={:.6e}, expected={:.6e}",
        energy,
        0.5 * direct,
    );
}

#[test]
fn test_triangle_mesh_inductance_matrix_has_constant_potential_null_mode() {
    let patches = [
        TrianglePatch {
            nodes: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            s: [0.0, 0.0, 0.0],
        },
        TrianglePatch {
            nodes: [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            s: [0.0, 0.0, 0.0],
        },
    ];
    let mesh = triangle_patches_to_mesh(&patches);
    let const_s = vec![2.3; mesh.s.len()];
    let lmat = mesh_inductance_matrix(&mesh, false);
    let mut jx = vec![0.0; mesh.triangles.0.len()];
    let mut jy = vec![0.0; mesh.triangles.0.len()];
    let mut jz = vec![0.0; mesh.triangles.0.len()];

    triangle_mesh_current_density(
        (&mesh.nodes.0, &mesh.nodes.1, &mesh.nodes.2),
        (&mesh.triangles.0, &mesh.triangles.1, &mesh.triangles.2),
        &const_s,
        (&mut jx, &mut jy, &mut jz),
    )
    .unwrap();

    for i in 0..mesh.triangles.0.len() {
        for (axis, comp) in [jx[i], jy[i], jz[i]].into_iter().enumerate() {
            assert!(
                approx(comp, 0.0, 0.0, 1e-14),
                "constant-potential current density is nonzero for triangle {i}, axis {axis}: {:.16e}",
                comp,
            );
        }
    }

    let bilinear =
        triangle_mesh_inductance_from_potential_vectors(&lmat, &const_s, &const_s).unwrap();
    let energy = triangle_mesh_inductive_energy(&lmat, &const_s).unwrap();
    assert!(
        approx(bilinear, 0.0, 0.0, 1e-12),
        "constant-potential bilinear form is nonzero: {:.16e}",
        bilinear,
    );
    assert!(
        approx(energy, 0.0, 0.0, 1e-12),
        "constant-potential energy is nonzero: {:.16e}",
        energy,
    );
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
fn test_triangle_mesh_inductance_matrix_mutual_coupling_matches_triangle_pair_sum() {
    let radius = 0.71;
    let height = radius * 1e-3;
    let nphi = 24;
    let current = 1.0;
    let z_src = -0.37;
    let z_tgt = 0.41;

    let strip_src = circular_strip_triangles_at_z(radius, height, current, nphi, z_src);
    let strip_tgt = circular_strip_triangles_at_z(radius, height, current, nphi, z_tgt);
    let mesh_src = triangle_patches_to_mesh(&strip_src);
    let mesh_tgt = triangle_patches_to_mesh(&strip_tgt);
    let (mesh, s_src, s_tgt) = combine_disconnected_meshes(&mesh_src, &mesh_tgt);
    let lmat = mesh_inductance_matrix(&mesh, false);

    let direct = strip_mutual_inductance(&strip_src, &strip_tgt);
    let via_l = triangle_mesh_inductance_from_potential_vectors(&lmat, &s_src, &s_tgt).unwrap();

    assert!(
        approx(via_l, direct, 1e-10, 1e-12),
        "disconnected-mesh mutual coupling mismatch: matrix={:.6e}, direct={:.6e}",
        via_l,
        direct,
    );
}

#[test]
fn test_triangle_basis_force_block_matches_direct_contraction() {
    let src0 = [0.0, 0.0, 0.0];
    let src1 = [0.4, 0.1, 0.0];
    let src2 = [0.1, 0.5, 0.0];
    let tgt0 = [0.2, -0.1, 0.7];
    let tgt1 = [0.6, 0.0, 0.8];
    let tgt2 = [0.1, 0.4, 0.9];
    let s_src = [0.7, -0.2, 0.5];
    let s_tgt = [-0.3, 0.4, 0.8];

    let block = triangle_basis_force_block(
        src0,
        src1,
        src2,
        tgt0,
        tgt1,
        tgt2,
        QuadratureKind::GaussLegendre3,
    );
    let via_block = triangle_force_from_potential_vectors(block, s_src, s_tgt);

    let tri_area = calc_tri_area(tgt0, tgt1, tgt2);
    let k_tgt = triangle_current_density(tgt0, tgt1, tgt2, s_tgt);
    let mut direct = [0.0; 3];
    for qp in triangle_quadrature_points(QuadratureKind::GaussLegendre3) {
        let obs = map_tri_uv(tgt0, tgt1, tgt2, [qp[1], qp[2]]);
        let b = flux_density_triangle(src0, src1, src2, s_src, obs, QuadratureKind::GaussLegendre3);
        let jf = cross3(k_tgt[0], k_tgt[1], k_tgt[2], b[0], b[1], b[2]);
        let w = qp[0] * tri_area;
        direct[0] += jf.0 * w;
        direct[1] += jf.1 * w;
        direct[2] += jf.2 * w;
    }

    for axis in 0..3 {
        assert!(
            approx(via_block[axis], direct[axis], 1e-12, 1e-12),
            "force block mismatch on axis {axis}: block={:.16e}, direct={:.16e}",
            via_block[axis],
            direct[axis],
        );
    }
}

#[test]
fn test_triangle_mesh_force_mapping_matches_direct_target_integration() {
    let radius = 0.61;
    let height = radius * 1e-3;
    let nphi = 32;
    let src = triangle_patches_to_mesh(&circular_strip_triangles_at_z(
        radius, height, 1.3, nphi, -0.17,
    ));
    let tgt = triangle_patches_to_mesh(&circular_strip_triangles_at_z(
        radius, height, 0.9, nphi, 0.23,
    ));

    let (fx, fy, fz) = mesh_force_mapping(&src, &tgt, false);
    let via_mapping = triangle_mesh_force_from_potential_vectors(&fx, &fy, &fz, &src.s).unwrap();
    let via_mapping_par = {
        let (fxp, fyp, fzp) = mesh_force_mapping(&src, &tgt, true);
        triangle_mesh_force_from_potential_vectors(&fxp, &fyp, &fzp, &src.s).unwrap()
    };

    let mut direct = [0.0; 3];
    for itgt in 0..tgt.triangles.0.len() {
        let idx = [
            tgt.triangles.0[itgt],
            tgt.triangles.1[itgt],
            tgt.triangles.2[itgt],
        ];
        let tri_nodes = idx.map(|k| [tgt.nodes.0[k], tgt.nodes.1[k], tgt.nodes.2[k]]);
        let tri_s = idx.map(|k| tgt.s[k]);
        let force = explicit_force_on_target_triangle_from_source_mesh(&src, tri_nodes, tri_s);
        direct[0] += force[0];
        direct[1] += force[1];
        direct[2] += force[2];
    }

    for axis in 0..3 {
        assert!(
            approx(via_mapping[axis], direct[axis], 1e-11, 1e-12),
            "mesh force mapping mismatch on axis {axis}: mapping={:.16e}, direct={:.16e}",
            via_mapping[axis],
            direct[axis],
        );
        assert!(
            approx(via_mapping_par[axis], direct[axis], 1e-11, 1e-12),
            "parallel mesh force mapping mismatch on axis {axis}: mapping={:.16e}, direct={:.16e}",
            via_mapping_par[axis],
            direct[axis],
        );
    }
}

#[test]
fn test_triangle_mesh_self_force_mapping_serial_matches_parallel() {
    let radius = 0.73;
    let height = radius * 1e-3;
    let mesh = triangle_patches_to_mesh(&circular_strip_triangles(radius, height, 1.0, 48));

    let (fx, fy, fz) = mesh_self_force_mapping(&mesh, false);
    let total = triangle_mesh_force_from_potential_vectors(&fx, &fy, &fz, &mesh.s).unwrap();
    let total_par = {
        let (fxp, fyp, fzp) = mesh_self_force_mapping(&mesh, true);
        triangle_mesh_force_from_potential_vectors(&fxp, &fyp, &fzp, &mesh.s).unwrap()
    };

    for axis in 0..3 {
        assert!(
            total[axis].is_finite(),
            "serial self force is not finite on axis {axis}: {:.6e}",
            total[axis]
        );
        assert!(
            total_par[axis].is_finite(),
            "parallel self force is not finite on axis {axis}: {:.6e}",
            total_par[axis]
        );
        assert!(
            approx(total[axis], total_par[axis], 1e-12, 1e-12),
            "self force serial/parallel mismatch on axis {axis}: serial={:.16e}, parallel={:.16e}",
            total[axis],
            total_par[axis],
        );
    }
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
