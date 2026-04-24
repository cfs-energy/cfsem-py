use core::f64::consts::PI;

use super::inductance::triangle_mesh_inductive_energy;
use super::{
    QuadratureKind, calc_tri_area, calc_tri_normal, flux_density_triangle,
    flux_density_triangle_mesh, flux_density_triangle_mesh_mapping,
    flux_density_triangle_mesh_mapping_par, flux_density_triangle_mesh_par, map_tri_uv,
    triangle_basis_current_densities, triangle_basis_current_density, triangle_basis_force_block,
    triangle_basis_mutual_inductance_block, triangle_current_density,
    triangle_force_from_potential_vectors, triangle_inductance_from_potential_vectors,
    triangle_mesh_current_density, triangle_mesh_flux_density_from_potential_vectors,
    triangle_mesh_flux_linkage_from_source_coefficients,
    triangle_mesh_flux_linkage_mapping_from_dipoles,
    triangle_mesh_flux_linkage_mapping_from_dipoles_par,
    triangle_mesh_force_from_potential_vectors, triangle_mesh_force_mapping,
    triangle_mesh_force_mapping_par, triangle_mesh_inductance_from_potential_vectors,
    triangle_mesh_inductance_mapping_from_circular_filaments,
    triangle_mesh_inductance_mapping_from_circular_filaments_par,
    triangle_mesh_inductance_mapping_from_linear_filaments,
    triangle_mesh_inductance_mapping_from_linear_filaments_par, triangle_mesh_inductance_matrix,
    triangle_mesh_inductance_matrix_par, triangle_mesh_interaction_energy_from_source_coefficients,
    triangle_mesh_quadrature_points, triangle_mesh_self_force_mapping,
    triangle_mesh_self_force_mapping_par, triangle_mesh_triangle_forces_from_potential_vectors,
    triangle_mesh_vector_potential_from_potential_vectors, triangle_quadrature_count,
    triangle_quadrature_points, triangle_vector_potential_basis, vector_potential_triangle,
    vector_potential_triangle_mesh, vector_potential_triangle_mesh_mapping,
    vector_potential_triangle_mesh_mapping_par, vector_potential_triangle_mesh_par,
};
use crate::math::{cartesian_to_cylindrical, cross3, dot3};
use crate::mesh::TriangleMeshView;
use crate::physics::circular_filament::{
    flux_circular_filament_scalar, flux_density_circular_filament_cartesian_scalar,
    vector_potential_circular_filament_scalar,
};
use crate::physics::linear_filament::vector_potential_linear_filament_scalar;
use crate::physics::point_source::current_element::{
    flux_density_current_element_scalar, vector_potential_current_element_scalar,
};
use crate::physics::point_source::dipole::vector_potential_dipole_scalar;
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

fn mesh_view(mesh: &TriangleMeshData) -> TriangleMeshView<'_> {
    TriangleMeshView::new(
        (&mesh.nodes.0, &mesh.nodes.1, &mesh.nodes.2),
        (&mesh.triangles.0, &mesh.triangles.1, &mesh.triangles.2),
    )
    .unwrap()
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
            s: [0.5 * s0, 0.5 * s0, -0.5 * s0],
        };
        let tri1 = TrianglePatch {
            nodes: [lower0, upper1, upper0],
            s: [0.5 * s0, -0.5 * s0, -0.5 * s0],
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
    let view = mesh_view(mesh);

    let result = match par {
        true => flux_density_triangle_mesh_par(
            (&obs_xyz.0, &obs_xyz.1, &obs_xyz.2),
            &view,
            &mesh.s,
            QuadratureKind::GaussLegendre3,
            (&mut bx, &mut by, &mut bz),
        ),
        false => flux_density_triangle_mesh(
            (&obs_xyz.0, &obs_xyz.1, &obs_xyz.2),
            &view,
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
    let view = mesh_view(mesh);

    let result = match par {
        true => vector_potential_triangle_mesh_par(
            (&obs_xyz.0, &obs_xyz.1, &obs_xyz.2),
            &view,
            &mesh.s,
            QuadratureKind::GaussLegendre3,
            (&mut ax, &mut ay, &mut az),
        ),
        false => vector_potential_triangle_mesh(
            (&obs_xyz.0, &obs_xyz.1, &obs_xyz.2),
            &view,
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
    let view = mesh_view(mesh);
    let result = match par {
        true => {
            triangle_mesh_inductance_matrix_par(&view, QuadratureKind::GaussLegendre3, &mut out)
        }
        false => triangle_mesh_inductance_matrix(&view, QuadratureKind::GaussLegendre3, &mut out),
    };
    result.unwrap();
    out
}

fn explicit_nodal_flux_linkage_from_vector_potential_fn<F>(
    mesh_tgt: &TriangleMeshData,
    eval_a: F,
) -> Vec<f64>
where
    F: Fn([f64; 3]) -> [f64; 3],
{
    let nnode = mesh_tgt.nodes.0.len();
    let mut out = vec![0.0; nnode];

    for itgt in 0..mesh_tgt.triangles.0.len() {
        let idx = [
            mesh_tgt.triangles.0[itgt],
            mesh_tgt.triangles.1[itgt],
            mesh_tgt.triangles.2[itgt],
        ];
        let tri_nodes = idx.map(|k| {
            [
                mesh_tgt.nodes.0[k],
                mesh_tgt.nodes.1[k],
                mesh_tgt.nodes.2[k],
            ]
        });
        let tri_area = calc_tri_area(tri_nodes[0], tri_nodes[1], tri_nodes[2]);
        let ktgt = triangle_basis_current_densities(tri_nodes[0], tri_nodes[1], tri_nodes[2]);

        for qp in triangle_quadrature_points(QuadratureKind::GaussLegendre3) {
            let obs = map_tri_uv(tri_nodes[0], tri_nodes[1], tri_nodes[2], [qp[1], qp[2]]);
            let a = eval_a(obs);
            let w = qp[0] * tri_area;
            for ibasis in 0..3 {
                out[idx[ibasis]] += dot3(
                    ktgt[ibasis][0],
                    ktgt[ibasis][1],
                    ktgt[ibasis][2],
                    a[0],
                    a[1],
                    a[2],
                ) * w;
            }
        }
    }

    out
}

fn mesh_inductance_mapping_from_linear_filaments(
    mesh_tgt: &TriangleMeshData,
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    par: bool,
) -> Vec<f64> {
    let nnode = mesh_tgt.nodes.0.len();
    let nfil = xyzfil.0.len();
    let mut out = vec![0.0; nnode * nfil];
    let view = mesh_view(mesh_tgt);
    let result = match par {
        true => triangle_mesh_inductance_mapping_from_linear_filaments_par(
            xyzfil,
            dlxyzfil,
            wire_radius,
            &view,
            QuadratureKind::GaussLegendre3,
            &mut out,
        ),
        false => triangle_mesh_inductance_mapping_from_linear_filaments(
            xyzfil,
            dlxyzfil,
            wire_radius,
            &view,
            QuadratureKind::GaussLegendre3,
            &mut out,
        ),
    };
    result.unwrap();
    out
}

fn mesh_inductance_mapping_from_circular_filaments(
    mesh_tgt: &TriangleMeshData,
    rfil: &[f64],
    zfil: &[f64],
    par: bool,
) -> Vec<f64> {
    let nnode = mesh_tgt.nodes.0.len();
    let nfil = rfil.len();
    let mut out = vec![0.0; nnode * nfil];
    let view = mesh_view(mesh_tgt);
    let result = match par {
        true => triangle_mesh_inductance_mapping_from_circular_filaments_par(
            rfil,
            zfil,
            &view,
            QuadratureKind::GaussLegendre3,
            &mut out,
        ),
        false => triangle_mesh_inductance_mapping_from_circular_filaments(
            rfil,
            zfil,
            &view,
            QuadratureKind::GaussLegendre3,
            &mut out,
        ),
    };
    result.unwrap();
    out
}

fn mesh_flux_linkage_mapping_from_dipoles(
    mesh_tgt: &TriangleMeshData,
    loc: (&[f64], &[f64], &[f64]),
    moment_dir: (&[f64], &[f64], &[f64]),
    outer_radius: &[f64],
    par: bool,
) -> Vec<f64> {
    let nnode = mesh_tgt.nodes.0.len();
    let ndip = loc.0.len();
    let mut out = vec![0.0; nnode * ndip];
    let view = mesh_view(mesh_tgt);
    let result = match par {
        true => triangle_mesh_flux_linkage_mapping_from_dipoles_par(
            loc,
            moment_dir,
            outer_radius,
            &view,
            QuadratureKind::GaussLegendre3,
            &mut out,
        ),
        false => triangle_mesh_flux_linkage_mapping_from_dipoles(
            loc,
            moment_dir,
            outer_radius,
            &view,
            QuadratureKind::GaussLegendre3,
            &mut out,
        ),
    };
    result.unwrap();
    out
}

fn mesh_flux_density_mapping(
    mesh: &TriangleMeshData,
    obs: &[[f64; 3]],
    par: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let obs_xyz = obs_components(obs);
    let nnode = mesh.nodes.0.len();
    let nout = obs.len() * nnode;
    let (mut bx, mut by, mut bz) = (vec![0.0; nout], vec![0.0; nout], vec![0.0; nout]);
    let view = mesh_view(mesh);

    let result = match par {
        true => flux_density_triangle_mesh_mapping_par(
            (&obs_xyz.0, &obs_xyz.1, &obs_xyz.2),
            &view,
            QuadratureKind::GaussLegendre3,
            (&mut bx, &mut by, &mut bz),
        ),
        false => flux_density_triangle_mesh_mapping(
            (&obs_xyz.0, &obs_xyz.1, &obs_xyz.2),
            &view,
            QuadratureKind::GaussLegendre3,
            (&mut bx, &mut by, &mut bz),
        ),
    };
    result.unwrap();

    (bx, by, bz)
}

fn mesh_vector_potential_mapping(
    mesh: &TriangleMeshData,
    obs: &[[f64; 3]],
    par: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let obs_xyz = obs_components(obs);
    let nnode = mesh.nodes.0.len();
    let nout = obs.len() * nnode;
    let (mut ax, mut ay, mut az) = (vec![0.0; nout], vec![0.0; nout], vec![0.0; nout]);
    let view = mesh_view(mesh);

    let result = match par {
        true => vector_potential_triangle_mesh_mapping_par(
            (&obs_xyz.0, &obs_xyz.1, &obs_xyz.2),
            &view,
            QuadratureKind::GaussLegendre3,
            (&mut ax, &mut ay, &mut az),
        ),
        false => vector_potential_triangle_mesh_mapping(
            (&obs_xyz.0, &obs_xyz.1, &obs_xyz.2),
            &view,
            QuadratureKind::GaussLegendre3,
            (&mut ax, &mut ay, &mut az),
        ),
    };
    result.unwrap();

    (ax, ay, az)
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
    let view_src = mesh_view(mesh_src);
    let view_tgt = mesh_view(mesh_tgt);

    let result = match par {
        true => triangle_mesh_force_mapping_par(
            &view_src,
            &view_tgt,
            &mesh_tgt.s,
            QuadratureKind::GaussLegendre3,
            (&mut fx, &mut fy, &mut fz),
        ),
        false => triangle_mesh_force_mapping(
            &view_src,
            &view_tgt,
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
    let view = mesh_view(mesh);

    let result = match par {
        true => triangle_mesh_self_force_mapping_par(
            &view,
            &mesh.s,
            QuadratureKind::GaussLegendre3,
            (&mut fx, &mut fy, &mut fz),
        ),
        false => triangle_mesh_self_force_mapping(
            &view,
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
    let view = mesh_view(mesh_src);
    for qp in triangle_quadrature_points(QuadratureKind::GaussLegendre3) {
        let obs = map_tri_uv(tri_nodes[0], tri_nodes[1], tri_nodes[2], [qp[1], qp[2]]);
        let mut bx = [0.0];
        let mut by = [0.0];
        let mut bz = [0.0];
        flux_density_triangle_mesh(
            (&[obs[0]], &[obs[1]], &[obs[2]]),
            &view,
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

/// Checks that the collection kernels reproduce the single-triangle field kernels.
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
    let (bx_map, by_map, bz_map) = mesh_flux_density_mapping(&mesh, &obs, false);
    let (bx_map_par, by_map_par, bz_map_par) = mesh_flux_density_mapping(&mesh, &obs, true);
    let (ax_map, ay_map, az_map) = mesh_vector_potential_mapping(&mesh, &obs, false);
    let (ax_map_par, ay_map_par, az_map_par) = mesh_vector_potential_mapping(&mesh, &obs, true);
    let mut b_from_map = (
        vec![0.0; obs.len()],
        vec![0.0; obs.len()],
        vec![0.0; obs.len()],
    );
    let mut b_from_map_par = (
        vec![0.0; obs.len()],
        vec![0.0; obs.len()],
        vec![0.0; obs.len()],
    );
    let mut a_from_map = (
        vec![0.0; obs.len()],
        vec![0.0; obs.len()],
        vec![0.0; obs.len()],
    );
    let mut a_from_map_par = (
        vec![0.0; obs.len()],
        vec![0.0; obs.len()],
        vec![0.0; obs.len()],
    );
    triangle_mesh_flux_density_from_potential_vectors(
        &bx_map,
        &by_map,
        &bz_map,
        &mesh.s,
        (&mut b_from_map.0, &mut b_from_map.1, &mut b_from_map.2),
    )
    .unwrap();
    triangle_mesh_flux_density_from_potential_vectors(
        &bx_map_par,
        &by_map_par,
        &bz_map_par,
        &mesh.s,
        (
            &mut b_from_map_par.0,
            &mut b_from_map_par.1,
            &mut b_from_map_par.2,
        ),
    )
    .unwrap();
    triangle_mesh_vector_potential_from_potential_vectors(
        &ax_map,
        &ay_map,
        &az_map,
        &mesh.s,
        (&mut a_from_map.0, &mut a_from_map.1, &mut a_from_map.2),
    )
    .unwrap();
    triangle_mesh_vector_potential_from_potential_vectors(
        &ax_map_par,
        &ay_map_par,
        &az_map_par,
        &mesh.s,
        (
            &mut a_from_map_par.0,
            &mut a_from_map_par.1,
            &mut a_from_map_par.2,
        ),
    )
    .unwrap();

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
            let b_from_map_axis = match axis {
                0 => b_from_map.0[i],
                1 => b_from_map.1[i],
                2 => b_from_map.2[i],
                _ => unreachable!(),
            };
            let b_from_map_par_axis = match axis {
                0 => b_from_map_par.0[i],
                1 => b_from_map_par.1[i],
                2 => b_from_map_par.2[i],
                _ => unreachable!(),
            };
            let a_from_map_axis = match axis {
                0 => a_from_map.0[i],
                1 => a_from_map.1[i],
                2 => a_from_map.2[i],
                _ => unreachable!(),
            };
            let a_from_map_par_axis = match axis {
                0 => a_from_map_par.0[i],
                1 => a_from_map_par.1[i],
                2 => a_from_map_par.2[i],
                _ => unreachable!(),
            };
            assert!(
                approx(b_from_map_axis, b_direct[axis], 1e-12, 1e-14),
                "single-triangle mapped B mismatch at point {i}, axis {axis}"
            );
            assert!(
                approx(b_from_map_par_axis, b_direct[axis], 1e-12, 1e-14),
                "single-triangle parallel mapped B mismatch at point {i}, axis {axis}"
            );
            assert!(
                approx(a_from_map_axis, a_direct[axis], 1e-12, 1e-14),
                "single-triangle mapped A mismatch at point {i}, axis {axis}"
            );
            assert!(
                approx(a_from_map_par_axis, a_direct[axis], 1e-12, 1e-14),
                "single-triangle parallel mapped A mismatch at point {i}, axis {axis}"
            );
        }
    }
}

/// Checks that field mapping matrices contract to the same fields as direct mesh evaluation.
#[test]
fn test_triangle_mesh_field_mappings_match_collection_fields() {
    let radius = 0.71;
    let height = radius * 1e-3;
    let mesh = triangle_patches_to_mesh(&circular_strip_triangles(radius, height, 1.4, 48));
    let obs = [
        [0.35, -0.18, 1.05],
        [1.40, 0.55, -0.62],
        [2.10, -0.44, 0.83],
        [2.85, 0.91, -1.11],
    ];

    let b_direct = mesh_flux_density(&mesh, &obs, false);
    let a_direct = mesh_vector_potential(&mesh, &obs, false);
    let (bx_map, by_map, bz_map) = mesh_flux_density_mapping(&mesh, &obs, false);
    let (bx_map_par, by_map_par, bz_map_par) = mesh_flux_density_mapping(&mesh, &obs, true);
    let (ax_map, ay_map, az_map) = mesh_vector_potential_mapping(&mesh, &obs, false);
    let (ax_map_par, ay_map_par, az_map_par) = mesh_vector_potential_mapping(&mesh, &obs, true);

    let mut bx = vec![0.0; obs.len()];
    let mut by = vec![0.0; obs.len()];
    let mut bz = vec![0.0; obs.len()];
    let mut bx_par = vec![0.0; obs.len()];
    let mut by_par = vec![0.0; obs.len()];
    let mut bz_par = vec![0.0; obs.len()];
    let mut ax = vec![0.0; obs.len()];
    let mut ay = vec![0.0; obs.len()];
    let mut az = vec![0.0; obs.len()];
    let mut ax_par = vec![0.0; obs.len()];
    let mut ay_par = vec![0.0; obs.len()];
    let mut az_par = vec![0.0; obs.len()];

    triangle_mesh_flux_density_from_potential_vectors(
        &bx_map,
        &by_map,
        &bz_map,
        &mesh.s,
        (&mut bx, &mut by, &mut bz),
    )
    .unwrap();
    triangle_mesh_flux_density_from_potential_vectors(
        &bx_map_par,
        &by_map_par,
        &bz_map_par,
        &mesh.s,
        (&mut bx_par, &mut by_par, &mut bz_par),
    )
    .unwrap();
    triangle_mesh_vector_potential_from_potential_vectors(
        &ax_map,
        &ay_map,
        &az_map,
        &mesh.s,
        (&mut ax, &mut ay, &mut az),
    )
    .unwrap();
    triangle_mesh_vector_potential_from_potential_vectors(
        &ax_map_par,
        &ay_map_par,
        &az_map_par,
        &mesh.s,
        (&mut ax_par, &mut ay_par, &mut az_par),
    )
    .unwrap();

    for i in 0..bx_map.len() {
        assert!(
            approx(bx_map[i], bx_map_par[i], 1e-12, 1e-14),
            "serial/parallel Bx mapping mismatch at flattened index {i}"
        );
        assert!(
            approx(by_map[i], by_map_par[i], 1e-12, 1e-14),
            "serial/parallel By mapping mismatch at flattened index {i}"
        );
        assert!(
            approx(bz_map[i], bz_map_par[i], 1e-12, 1e-14),
            "serial/parallel Bz mapping mismatch at flattened index {i}"
        );
        assert!(
            approx(ax_map[i], ax_map_par[i], 1e-12, 1e-14),
            "serial/parallel Ax mapping mismatch at flattened index {i}"
        );
        assert!(
            approx(ay_map[i], ay_map_par[i], 1e-12, 1e-14),
            "serial/parallel Ay mapping mismatch at flattened index {i}"
        );
        assert!(
            approx(az_map[i], az_map_par[i], 1e-12, 1e-14),
            "serial/parallel Az mapping mismatch at flattened index {i}"
        );
    }

    for iobs in 0..obs.len() {
        let b_map = [bx[iobs], by[iobs], bz[iobs]];
        let b_map_par = [bx_par[iobs], by_par[iobs], bz_par[iobs]];
        let a_map = [ax[iobs], ay[iobs], az[iobs]];
        let a_map_par = [ax_par[iobs], ay_par[iobs], az_par[iobs]];

        for axis in 0..3 {
            assert!(
                approx(b_map[axis], b_direct[iobs][axis], 1e-12, 1e-14),
                "contracted B mapping mismatch at point {iobs}, axis {axis}"
            );
            assert!(
                approx(b_map_par[axis], b_direct[iobs][axis], 1e-12, 1e-14),
                "parallel contracted B mapping mismatch at point {iobs}, axis {axis}"
            );
            assert!(
                approx(a_map[axis], a_direct[iobs][axis], 1e-12, 1e-14),
                "contracted A mapping mismatch at point {iobs}, axis {axis}"
            );
            assert!(
                approx(a_map_par[axis], a_direct[iobs][axis], 1e-12, 1e-14),
                "parallel contracted A mapping mismatch at point {iobs}, axis {axis}"
            );
        }
    }
}

/// Checks triangle basis fields against a direct current-element quadrature sum.
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

/// Checks that summing the three one-hot triangle bases cancels B and A fields.
#[test]
fn test_single_triangle_basis_contributions_cancel_for_constant_potential() {
    let tri = [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.2, 0.8, 0.1]];
    let basis_vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let obs_far = [10.0, -7.5, 6.0];
    let obs_on = map_tri_uv(tri[0], tri[1], tri[2], [0.23, 0.41]);
    let normal = calc_tri_normal(tri[0], tri[1], tri[2]);
    let eps = 1e-12;
    let obs_near = [
        obs_on[0] + eps * normal[0],
        obs_on[1] + eps * normal[1],
        obs_on[2] + eps * normal[2],
    ];

    for (obs_name, obs) in [("far", obs_far), ("near", obs_near)] {
        let mut b_sum = [0.0; 3];
        let mut a_sum = [0.0; 3];
        let mut b_scale: f64 = 0.0;
        let mut a_scale: f64 = 0.0;

        for s_basis in basis_vectors {
            let b = flux_density_triangle(
                tri[0],
                tri[1],
                tri[2],
                s_basis,
                obs,
                QuadratureKind::GaussLegendre3,
            );
            let a = vector_potential_triangle(
                tri[0],
                tri[1],
                tri[2],
                s_basis,
                obs,
                QuadratureKind::GaussLegendre3,
            );
            for axis in 0..3 {
                b_sum[axis] += b[axis];
                a_sum[axis] += a[axis];
                b_scale = b_scale.max(b[axis].abs());
                a_scale = a_scale.max(a[axis].abs());
            }
        }

        let (b_rtol, a_rtol) = match obs_name {
            "far" => (1e-8, 1e-8),
            "near" => (1e-12, 1e-12),
            _ => unreachable!(),
        };
        let b_atol = b_rtol * b_scale;
        let a_atol = a_rtol * a_scale;
        for axis in 0..3 {
            assert!(
                approx(b_sum[axis], 0.0, 0.0, b_atol),
                "basis B cancellation failed at {obs_name} point, axis {axis}: sum={:.16e}, scale={:.16e}, rtol={:.16e}, atol={:.16e}",
                b_sum[axis],
                b_scale,
                b_rtol,
                b_atol,
            );
            assert!(
                approx(a_sum[axis], 0.0, 0.0, a_atol),
                "basis A cancellation failed at {obs_name} point, axis {axis}: sum={:.16e}, scale={:.16e}, rtol={:.16e}, atol={:.16e}",
                a_sum[axis],
                a_scale,
                a_rtol,
                a_atol,
            );
        }
    }
}

/// Checks mesh quadrature-point extraction and triangle current-density reconstruction.
#[test]
fn test_triangle_mesh_quadrature_points_and_current_density_extractors() {
    let tris = circular_strip_triangles(0.73, 7.3e-4, 1.7, 24);
    let mesh = triangle_patches_to_mesh(&tris);
    let quad_kind = QuadratureKind::GaussLegendre2;
    let ntri = mesh.triangles.0.len();
    let nqp = triangle_quadrature_count(quad_kind);

    let (mut jx, mut jy, mut jz) = (vec![0.0; ntri], vec![0.0; ntri], vec![0.0; ntri]);
    triangle_mesh_current_density(&mesh_view(&mesh), &mesh.s, (&mut jx, &mut jy, &mut jz)).unwrap();

    let (mut xq, mut yq, mut zq) = (
        vec![0.0; ntri * nqp],
        vec![0.0; ntri * nqp],
        vec![0.0; ntri * nqp],
    );
    let mut wq = vec![0.0; ntri * nqp];
    triangle_mesh_quadrature_points(
        &mesh_view(&mesh),
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

/// Checks that the Dunavant rule integrates reference-triangle monomials through degree five.
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

/// Checks disjoint-triangle inductance blocks against vector-potential contraction.
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

/// Checks a one-triangle mesh inductance matrix against the direct single-triangle block.
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

/// Checks that the self-inductance block stays finite, symmetric, and nearly PSD.
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

/// Checks mesh inductance symmetry and agreement with direct bilinear contraction.
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

/// Checks that a constant nodal potential lies in the inductance matrix null mode.
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

    triangle_mesh_current_density(&mesh_view(&mesh), &const_s, (&mut jx, &mut jy, &mut jz))
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

/// Checks linear-filament inductance mapping against direct target-triangle quadrature.
#[test]
fn test_triangle_mesh_inductance_mapping_from_linear_filaments_matches_direct_target_quadrature() {
    let radius = 0.62;
    let height = radius * 1e-3;
    let mesh_tgt = triangle_patches_to_mesh(&circular_strip_triangles_at_z(
        radius, height, 0.9, 32, 0.21,
    ));
    let xyzfil = (&[0.15, -0.12][..], &[-0.35, 0.28][..], &[0.42, -0.31][..]);
    let dlxyzfil = (&[0.27, -0.18][..], &[0.16, 0.21][..], &[-0.11, 0.14][..]);
    let wire_radius = &[2.5e-3, 1.5e-3][..];
    let coeffs = &[1.4, -0.7][..];

    let map = mesh_inductance_mapping_from_linear_filaments(
        &mesh_tgt,
        xyzfil,
        dlxyzfil,
        wire_radius,
        false,
    );
    let map_par = mesh_inductance_mapping_from_linear_filaments(
        &mesh_tgt,
        xyzfil,
        dlxyzfil,
        wire_radius,
        true,
    );
    for i in 0..map.len() {
        assert!(
            approx(map[i], map_par[i], 1e-12, 1e-14),
            "linear-filament inductance mapping serial/parallel mismatch at {i}"
        );
    }

    let mut psi = vec![0.0; mesh_tgt.nodes.0.len()];
    triangle_mesh_flux_linkage_from_source_coefficients(&map, coeffs, &mut psi).unwrap();
    let energy =
        triangle_mesh_interaction_energy_from_source_coefficients(&map, &mesh_tgt.s, coeffs)
            .unwrap();

    let psi_ref = explicit_nodal_flux_linkage_from_vector_potential_fn(&mesh_tgt, |obs| {
        let mut out = [0.0; 3];
        for i in 0..coeffs.len() {
            let start = (xyzfil.0[i], xyzfil.1[i], xyzfil.2[i]);
            let end = (
                xyzfil.0[i] + dlxyzfil.0[i],
                xyzfil.1[i] + dlxyzfil.1[i],
                xyzfil.2[i] + dlxyzfil.2[i],
            );
            let a = vector_potential_linear_filament_scalar(
                (start, end, coeffs[i]),
                wire_radius[i],
                (obs[0], obs[1], obs[2]),
            );
            out[0] += a.0;
            out[1] += a.1;
            out[2] += a.2;
        }
        out
    });
    let energy_ref: f64 = mesh_tgt
        .s
        .iter()
        .zip(psi_ref.iter())
        .map(|(s, p)| s * p)
        .sum();

    for i in 0..psi.len() {
        assert!(
            approx(psi[i], psi_ref[i], 1e-11, 1e-12),
            "linear-filament nodal flux linkage mismatch at node {i}: map={:.16e}, direct={:.16e}",
            psi[i],
            psi_ref[i]
        );
    }
    assert!(
        approx(energy, energy_ref, 1e-11, 1e-12),
        "linear-filament interaction energy mismatch: map={:.16e}, direct={:.16e}",
        energy,
        energy_ref
    );
}

/// Checks circular-filament inductance mapping against direct target-triangle quadrature.
#[test]
fn test_triangle_mesh_inductance_mapping_from_circular_filaments_matches_direct_target_quadrature()
{
    let radius = 0.58;
    let height = radius * 1e-3;
    let mesh_tgt = triangle_patches_to_mesh(&circular_strip_triangles_at_z(
        radius, height, 0.85, 32, 0.17,
    ));
    let rfil = &[0.47, 0.63][..];
    let zfil = &[-0.16, 0.29][..];
    let coeffs = &[0.8, -1.1][..];

    let map = mesh_inductance_mapping_from_circular_filaments(&mesh_tgt, rfil, zfil, false);
    let map_par = mesh_inductance_mapping_from_circular_filaments(&mesh_tgt, rfil, zfil, true);
    for i in 0..map.len() {
        assert!(
            approx(map[i], map_par[i], 1e-12, 1e-14),
            "circular-filament inductance mapping serial/parallel mismatch at {i}"
        );
    }

    let mut psi = vec![0.0; mesh_tgt.nodes.0.len()];
    triangle_mesh_flux_linkage_from_source_coefficients(&map, coeffs, &mut psi).unwrap();
    let energy =
        triangle_mesh_interaction_energy_from_source_coefficients(&map, &mesh_tgt.s, coeffs)
            .unwrap();

    let psi_ref = explicit_nodal_flux_linkage_from_vector_potential_fn(&mesh_tgt, |obs| {
        let (robs, phiobs, zobs) = cartesian_to_cylindrical(obs[0], obs[1], obs[2]);
        let mut out = [0.0; 3];
        for i in 0..coeffs.len() {
            let a_phi = vector_potential_circular_filament_scalar(
                (rfil[i], zfil[i], coeffs[i]),
                (robs, zobs),
            );
            out[0] += -a_phi * libm::sin(phiobs);
            out[1] += a_phi * libm::cos(phiobs);
        }
        out
    });
    let energy_ref: f64 = mesh_tgt
        .s
        .iter()
        .zip(psi_ref.iter())
        .map(|(s, p)| s * p)
        .sum();

    for i in 0..psi.len() {
        assert!(
            approx(psi[i], psi_ref[i], 1e-11, 1e-12),
            "circular-filament nodal flux linkage mismatch at node {i}: map={:.16e}, direct={:.16e}",
            psi[i],
            psi_ref[i]
        );
    }
    assert!(
        approx(energy, energy_ref, 1e-11, 1e-12),
        "circular-filament interaction energy mismatch: map={:.16e}, direct={:.16e}",
        energy,
        energy_ref
    );
}

/// Checks dipole flux-linkage mapping against direct target-triangle quadrature.
#[test]
fn test_triangle_mesh_flux_linkage_mapping_from_dipoles_matches_direct_target_quadrature() {
    let radius = 0.55;
    let height = radius * 1e-3;
    let mesh_tgt = triangle_patches_to_mesh(&circular_strip_triangles_at_z(
        radius, height, 0.75, 28, 0.12,
    ));
    let loc = (&[0.21, -0.38][..], &[0.12, 0.25][..], &[-0.27, 0.34][..]);
    let moment_dir = (&[0.0, 0.8][..], &[0.0, -0.4][..], &[1.0, 0.45][..]);
    let outer_radius = &[0.0, 0.0][..];
    let coeffs = &[0.35, -0.6][..];

    let map =
        mesh_flux_linkage_mapping_from_dipoles(&mesh_tgt, loc, moment_dir, outer_radius, false);
    let map_par =
        mesh_flux_linkage_mapping_from_dipoles(&mesh_tgt, loc, moment_dir, outer_radius, true);
    for i in 0..map.len() {
        assert!(
            approx(map[i], map_par[i], 1e-12, 1e-14),
            "dipole flux-linkage mapping serial/parallel mismatch at {i}"
        );
    }

    let mut psi = vec![0.0; mesh_tgt.nodes.0.len()];
    triangle_mesh_flux_linkage_from_source_coefficients(&map, coeffs, &mut psi).unwrap();
    let energy =
        triangle_mesh_interaction_energy_from_source_coefficients(&map, &mesh_tgt.s, coeffs)
            .unwrap();

    let psi_ref = explicit_nodal_flux_linkage_from_vector_potential_fn(&mesh_tgt, |obs| {
        let mut out = [0.0; 3];
        for i in 0..coeffs.len() {
            let a = vector_potential_dipole_scalar(
                (loc.0[i], loc.1[i], loc.2[i]),
                (
                    coeffs[i] * moment_dir.0[i],
                    coeffs[i] * moment_dir.1[i],
                    coeffs[i] * moment_dir.2[i],
                ),
                outer_radius[i],
                (obs[0], obs[1], obs[2]),
            );
            out[0] += a.0;
            out[1] += a.1;
            out[2] += a.2;
        }
        out
    });
    let energy_ref: f64 = mesh_tgt
        .s
        .iter()
        .zip(psi_ref.iter())
        .map(|(s, p)| s * p)
        .sum();

    for i in 0..psi.len() {
        assert!(
            approx(psi[i], psi_ref[i], 1e-11, 1e-12),
            "dipole nodal flux linkage mismatch at node {i}: map={:.16e}, direct={:.16e}",
            psi[i],
            psi_ref[i]
        );
    }
    assert!(
        approx(energy, energy_ref, 1e-11, 1e-12),
        "dipole interaction energy mismatch: map={:.16e}, direct={:.16e}",
        energy,
        energy_ref
    );
}

/// Checks that touching-triangle inductance pairs remain finite and reciprocal.
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

/// Checks strip mutual inductance against a circular-filament reference loop.
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

/// Checks disconnected-mesh mutual coupling against an explicit triangle-pair sum.
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

/// Checks the triangle force block against direct contraction over the target triangle.
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

/// Checks mesh force mapping against direct target-triangle force integration.
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

/// Checks per-triangle force contraction against explicit target-triangle forces.
#[test]
fn test_triangle_mesh_triangle_forces_from_potential_vectors_match_direct_target_forces() {
    let radius = 0.64;
    let height = radius * 1e-3;
    let nphi = 24;
    let src = triangle_patches_to_mesh(&circular_strip_triangles_at_z(
        radius, height, 1.1, nphi, -0.21,
    ));
    let tgt = triangle_patches_to_mesh(&circular_strip_triangles_at_z(
        radius, height, 0.8, nphi, 0.19,
    ));

    let (fx, fy, fz) = mesh_force_mapping(&src, &tgt, false);
    let (fxp, fyp, fzp) = mesh_force_mapping(&src, &tgt, true);
    let ntri_tgt = tgt.triangles.0.len();

    let mut tri_fx = vec![0.0; ntri_tgt];
    let mut tri_fy = vec![0.0; ntri_tgt];
    let mut tri_fz = vec![0.0; ntri_tgt];
    triangle_mesh_triangle_forces_from_potential_vectors(
        &fx,
        &fy,
        &fz,
        &src.s,
        (&mut tri_fx, &mut tri_fy, &mut tri_fz),
    )
    .unwrap();

    let mut tri_fx_par = vec![0.0; ntri_tgt];
    let mut tri_fy_par = vec![0.0; ntri_tgt];
    let mut tri_fz_par = vec![0.0; ntri_tgt];
    triangle_mesh_triangle_forces_from_potential_vectors(
        &fxp,
        &fyp,
        &fzp,
        &src.s,
        (&mut tri_fx_par, &mut tri_fy_par, &mut tri_fz_par),
    )
    .unwrap();

    let total = triangle_mesh_force_from_potential_vectors(&fx, &fy, &fz, &src.s).unwrap();
    let total_par = triangle_mesh_force_from_potential_vectors(&fxp, &fyp, &fzp, &src.s).unwrap();
    let mut direct_total = [0.0; 3];
    for itgt in 0..ntri_tgt {
        let idx = [
            tgt.triangles.0[itgt],
            tgt.triangles.1[itgt],
            tgt.triangles.2[itgt],
        ];
        let tri_nodes = idx.map(|k| [tgt.nodes.0[k], tgt.nodes.1[k], tgt.nodes.2[k]]);
        let tri_s = idx.map(|k| tgt.s[k]);
        let direct = explicit_force_on_target_triangle_from_source_mesh(&src, tri_nodes, tri_s);
        direct_total[0] += direct[0];
        direct_total[1] += direct[1];
        direct_total[2] += direct[2];

        for axis in 0..3 {
            let via_tri = [tri_fx[itgt], tri_fy[itgt], tri_fz[itgt]][axis];
            let via_tri_par = [tri_fx_par[itgt], tri_fy_par[itgt], tri_fz_par[itgt]][axis];
            assert!(
                approx(via_tri, direct[axis], 1e-11, 1e-12),
                "triangle force contraction mismatch for triangle {itgt}, axis {axis}: contraction={:.16e}, direct={:.16e}",
                via_tri,
                direct[axis],
            );
            assert!(
                approx(via_tri_par, direct[axis], 1e-11, 1e-12),
                "parallel triangle force contraction mismatch for triangle {itgt}, axis {axis}: contraction={:.16e}, direct={:.16e}",
                via_tri_par,
                direct[axis],
            );
        }
    }

    let tri_total = [
        tri_fx.iter().sum::<f64>(),
        tri_fy.iter().sum::<f64>(),
        tri_fz.iter().sum::<f64>(),
    ];
    let tri_total_par = [
        tri_fx_par.iter().sum::<f64>(),
        tri_fy_par.iter().sum::<f64>(),
        tri_fz_par.iter().sum::<f64>(),
    ];
    for axis in 0..3 {
        assert!(
            approx(tri_total[axis], total[axis], 1e-12, 1e-12),
            "triangle-force sum mismatch on axis {axis}: summed={:.16e}, total={:.16e}",
            tri_total[axis],
            total[axis],
        );
        assert!(
            approx(tri_total_par[axis], total_par[axis], 1e-12, 1e-12),
            "parallel triangle-force sum mismatch on axis {axis}: summed={:.16e}, total={:.16e}",
            tri_total_par[axis],
            total_par[axis],
        );
        assert!(
            approx(total[axis], direct_total[axis], 1e-11, 1e-12),
            "triangle-force total mismatch on axis {axis}: total={:.16e}, direct={:.16e}",
            total[axis],
            direct_total[axis],
        );
    }
}

/// Checks that disjoint-mesh force mappings satisfy Newton's third-law reciprocity.
#[test]
fn test_triangle_mesh_force_mapping_obeys_newtons_third_law_for_disjoint_meshes() {
    let radius = 0.61;
    let height = radius * 1e-3;
    let nphi = 32;
    let src = triangle_patches_to_mesh(&circular_strip_triangles_at_z(
        radius, height, 1.3, nphi, -0.17,
    ));
    let tgt = triangle_patches_to_mesh(&circular_strip_triangles_at_z(
        radius, height, 0.9, nphi, 0.23,
    ));

    let (fx_tgt, fy_tgt, fz_tgt) = mesh_force_mapping(&src, &tgt, false);
    let force_on_tgt =
        triangle_mesh_force_from_potential_vectors(&fx_tgt, &fy_tgt, &fz_tgt, &src.s).unwrap();

    let (fx_src, fy_src, fz_src) = mesh_force_mapping(&tgt, &src, false);
    let force_on_src =
        triangle_mesh_force_from_potential_vectors(&fx_src, &fy_src, &fz_src, &tgt.s).unwrap();

    for axis in 0..3 {
        assert!(
            approx(force_on_tgt[axis], -force_on_src[axis], 1e-9, 1e-12),
            "Newton's third law mismatch on axis {axis}: F_tgt={:.16e}, F_src={:.16e}",
            force_on_tgt[axis],
            force_on_src[axis],
        );
    }
}

/// Checks that self-force mapping gives the same result in serial and parallel.
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

/// Checks that triangle B and A remain finite on and very near the source surface.
#[test]
fn test_triangle_fields_are_finite_on_and_very_near_triangle_surface() {
    let tri = TrianglePatch {
        nodes: [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.2, 0.8, 0.1]],
        s: [1.2, -0.4, 0.7],
    };
    let mesh = triangle_patches_to_mesh(&[tri]);
    let obs_on = map_tri_uv(tri.nodes[0], tri.nodes[1], tri.nodes[2], [0.25, 0.5]);
    let normal = calc_tri_normal(tri.nodes[0], tri.nodes[1], tri.nodes[2]);
    let eps = 1e-12;
    let obs_above = [
        obs_on[0] + eps * normal[0],
        obs_on[1] + eps * normal[1],
        obs_on[2] + eps * normal[2],
    ];
    let obs_below = [
        obs_on[0] - eps * normal[0],
        obs_on[1] - eps * normal[1],
        obs_on[2] - eps * normal[2],
    ];
    let obs = [obs_on, obs_above, obs_below];

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
                b_direct[axis].is_finite(),
                "direct near-surface B is not finite at point {i}, axis {axis}: {:.16e}",
                b_direct[axis],
            );
            assert!(
                a_direct[axis].is_finite(),
                "direct near-surface A is not finite at point {i}, axis {axis}: {:.16e}",
                a_direct[axis],
            );
            assert!(
                approx(b_mesh[i][axis], b_direct[axis], 1e-12, 1e-12),
                "mesh near-surface B mismatch at point {i}, axis {axis}: mesh={:.16e}, direct={:.16e}",
                b_mesh[i][axis],
                b_direct[axis],
            );
            assert!(
                approx(b_mesh_par[i][axis], b_direct[axis], 1e-12, 1e-12),
                "parallel mesh near-surface B mismatch at point {i}, axis {axis}: mesh={:.16e}, direct={:.16e}",
                b_mesh_par[i][axis],
                b_direct[axis],
            );
            assert!(
                approx(a_mesh[i][axis], a_direct[axis], 1e-12, 1e-12),
                "mesh near-surface A mismatch at point {i}, axis {axis}: mesh={:.16e}, direct={:.16e}",
                a_mesh[i][axis],
                a_direct[axis],
            );
            assert!(
                approx(a_mesh_par[i][axis], a_direct[axis], 1e-12, 1e-12),
                "parallel mesh near-surface A mismatch at point {i}, axis {axis}: mesh={:.16e}, direct={:.16e}",
                a_mesh_par[i][axis],
                a_direct[axis],
            );
        }
    }

    for axis in 0..3 {
        assert!(
            approx(a_mesh[1][axis], a_mesh[2][axis], 1e-9, 1e-12),
            "vector potential is not continuous across the surface on axis {axis}: above={:.16e}, below={:.16e}",
            a_mesh[1][axis],
            a_mesh[2][axis],
        );
    }
}

/// Checks far-field strip flux density and vector potential against a circular filament.
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

/// Checks near-axis strip flux density against a circular-filament reference solution.
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
