#![allow(clippy::all)] // Criterion requires black-boxed benchmark inputs and outputs.

use cfsem::MU0_OVER_4PI;
use cfsem::mesh::TriangleMeshView;
use cfsem::physics::boundary_element::{
    QuadratureKind, flux_density_triangle, triangle_geometric_coupling,
    triangle_geometric_coupling_regular, triangle_mesh_inductance_matrix_par,
    vector_potential_triangle,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

fn pair(aspect: f64, separation: f64) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    let area = 1.0e-3;
    let length = (2.0 * area * aspect).sqrt();
    let width = length / aspect;
    let source = [
        [0.0, 0.0, 0.0],
        [length, 0.0, 0.0],
        [0.5 * length, width, 0.0],
    ];
    let target = [
        [0.0, -separation, 0.0],
        [0.5 * length, -width - separation, 0.0],
        [length, -separation, 0.0],
    ];
    (source, target)
}

fn additional_near_pairs(aspect: f64) -> [(&'static str, [[f64; 3]; 3], [[f64; 3]; 3], bool); 3] {
    let (source, _) = pair(aspect, 0.0);
    let length = source[1][0];
    let width = source[2][1];
    let shared_vertex = [
        source[0],
        [0.25 * length, -width, 0.2 * width],
        [-0.2 * length, -0.5 * width, -0.1 * width],
    ];
    let close_disjoint = source.map(|[x, y, z]| [x, y, z + 0.05 * width]);
    [
        ("self", source, source, false),
        ("shared-vertex", source, shared_vertex, true),
        ("close-disjoint", source, close_disjoint, true),
    ]
}

struct AnnulusMesh {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    tri0: Vec<usize>,
    tri1: Vec<usize>,
    tri2: Vec<usize>,
}

impl AnnulusMesh {
    fn view(&self) -> TriangleMeshView<'_> {
        TriangleMeshView::new(
            (&self.x, &self.y, &self.z),
            (&self.tri0, &self.tri1, &self.tri2),
        )
        .unwrap()
    }
}

fn structured_annulus(nradial: usize, nphi: usize) -> AnnulusMesh {
    let mut x = Vec::with_capacity((nradial + 1) * nphi);
    let mut y = Vec::with_capacity((nradial + 1) * nphi);
    let mut z = Vec::with_capacity((nradial + 1) * nphi);
    for iradial in 0..=nradial {
        let radius = 0.5 + 0.5 * iradial as f64 / nradial as f64;
        for iphi in 0..nphi {
            let phi = 2.0 * std::f64::consts::PI * iphi as f64 / nphi as f64;
            x.push(radius * phi.cos());
            y.push(radius * phi.sin());
            z.push(0.0);
        }
    }

    let mut tri0 = Vec::with_capacity(2 * nradial * nphi);
    let mut tri1 = Vec::with_capacity(2 * nradial * nphi);
    let mut tri2 = Vec::with_capacity(2 * nradial * nphi);
    for iradial in 0..nradial {
        for iphi in 0..nphi {
            let next_phi = (iphi + 1) % nphi;
            let inner0 = iradial * nphi + iphi;
            let inner1 = iradial * nphi + next_phi;
            let outer0 = (iradial + 1) * nphi + iphi;
            let outer1 = (iradial + 1) * nphi + next_phi;
            tri0.extend([inner0, inner0]);
            tri1.extend([outer0, outer1]);
            tri2.extend([outer1, inner1]);
        }
    }

    AnnulusMesh {
        x,
        y,
        z,
        tri0,
        tri1,
        tri2,
    }
}

fn bench_triangle_coupling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Triangle geometric coupling");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));

    let mut cases = Vec::new();
    for (label, aspect, separation) in [
        ("far aspect 1".to_owned(), 1.0, 10.0),
        ("far aspect 67".to_owned(), 67.0, 10.0),
        ("edge-sharing aspect 1".to_owned(), 1.0, 0.0),
        ("edge-sharing aspect 4".to_owned(), 4.0, 0.0),
        ("edge-sharing aspect 16".to_owned(), 16.0, 0.0),
        ("edge-sharing aspect 67".to_owned(), 67.0, 0.0),
    ] {
        let (source, target) = pair(aspect, separation);
        cases.push((label, source, target, separation == 0.0, true));
    }
    for aspect in [1.0, 4.0, 16.0, 67.0] {
        for (kind, source, target, regular_is_finite) in additional_near_pairs(aspect) {
            cases.push((
                format!("{kind} aspect {aspect}"),
                source,
                target,
                true,
                regular_is_finite,
            ));
        }
    }

    for (label, source, target, is_near, regular_is_finite) in cases {
        group.bench_with_input(
            BenchmarkId::new("production", label.as_str()),
            &label,
            |b, _| {
                b.iter(|| {
                    let source = black_box(source);
                    let target = black_box(target);
                    black_box(triangle_geometric_coupling(
                        source[0],
                        source[1],
                        source[2],
                        target[0],
                        target[1],
                        target[2],
                        QuadratureKind::Dunavant3,
                    ))
                });
            },
        );
        if is_near {
            group.bench_with_input(
                BenchmarkId::new("production D1", label.as_str()),
                &label,
                |b, _| {
                    b.iter(|| {
                        let source = black_box(source);
                        let target = black_box(target);
                        black_box(triangle_geometric_coupling(
                            source[0],
                            source[1],
                            source[2],
                            target[0],
                            target[1],
                            target[2],
                            QuadratureKind::Dunavant1,
                        ))
                    });
                },
            );
        }
        if regular_is_finite {
            group.bench_with_input(
                BenchmarkId::new("legacy nested", label.as_str()),
                &label,
                |b, _| {
                    b.iter(|| {
                        let source = black_box(source);
                        let target = black_box(target);
                        black_box(triangle_geometric_coupling_regular(
                            source[0],
                            source[1],
                            source[2],
                            target[0],
                            target[1],
                            target[2],
                            QuadratureKind::Dunavant3,
                        ))
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_inductance_assembly(c: &mut Criterion) {
    let mut group = c.benchmark_group("Structured annulus inductance assembly");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    let mut cases = vec![(648, 3, 108), (1152, 4, 144)];
    if std::env::var_os("CFSEM_BENCH_FULL_ASSEMBLY").is_some() {
        cases.extend([(2178, 9, 121), (4050, 15, 135), (7938, 21, 189)]);
    }

    for (ntriangles, nradial, nphi) in cases {
        let mesh = structured_annulus(nradial, nphi);
        assert_eq!(mesh.tri0.len(), ntriangles);
        let nnode = mesh.x.len();
        let mesh_view = mesh.view();
        for quadrature in [QuadratureKind::Dunavant1, QuadratureKind::Dunavant3] {
            group.bench_function(
                BenchmarkId::new(format!("parallel {quadrature:?}"), ntriangles),
                |b| {
                    let mut out = vec![0.0; nnode * nnode];
                    b.iter(|| {
                        triangle_mesh_inductance_matrix_par(
                            &mesh_view,
                            quadrature,
                            black_box(&mut out),
                        )
                        .unwrap();
                        black_box(&out);
                    });
                },
            );
        }
    }
    group.finish();
}

fn legacy_triangle_flux_density(
    nodes: [[f64; 3]; 3],
    current_density: [f64; 3],
    obs: [f64; 3],
) -> [f64; 3] {
    let ab = [
        nodes[1][0] - nodes[0][0],
        nodes[1][1] - nodes[0][1],
        nodes[1][2] - nodes[0][2],
    ];
    let ac = [
        nodes[2][0] - nodes[0][0],
        nodes[2][1] - nodes[0][1],
        nodes[2][2] - nodes[0][2],
    ];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let area = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    let rule = [
        (-27.0 / 48.0, 1.0 / 3.0, 1.0 / 3.0),
        (25.0 / 48.0, 0.2, 0.2),
        (25.0 / 48.0, 0.6, 0.2),
        (25.0 / 48.0, 0.2, 0.6),
    ];
    let mut out = [0.0; 3];
    for (weight, u, v) in rule {
        let source = [
            nodes[0][0] + u * ab[0] + v * ac[0],
            nodes[0][1] + u * ab[1] + v * ac[1],
            nodes[0][2] + u * ab[2] + v * ac[2],
        ];
        let r = [obs[0] - source[0], obs[1] - source[1], obs[2] - source[2]];
        let inv_r3 = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).powf(-1.5);
        let contribution = [
            current_density[1] * r[2] - current_density[2] * r[1],
            current_density[2] * r[0] - current_density[0] * r[2],
            current_density[0] * r[1] - current_density[1] * r[0],
        ];
        for axis in 0..3 {
            out[axis] += MU0_OVER_4PI * area * weight * contribution[axis] * inv_r3;
        }
    }
    out
}

fn legacy_triangle_vector_potential_basis(nodes: [[f64; 3]; 3], obs: [f64; 3]) -> [f64; 3] {
    let ab = [
        nodes[1][0] - nodes[0][0],
        nodes[1][1] - nodes[0][1],
        nodes[1][2] - nodes[0][2],
    ];
    let ac = [
        nodes[2][0] - nodes[0][0],
        nodes[2][1] - nodes[0][1],
        nodes[2][2] - nodes[0][2],
    ];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let area = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    let current_density = [
        (nodes[1][0] - nodes[2][0]) / (2.0 * area),
        (nodes[1][1] - nodes[2][1]) / (2.0 * area),
        (nodes[1][2] - nodes[2][2]) / (2.0 * area),
    ];
    let rule = [
        (-27.0 / 48.0, 1.0 / 3.0, 1.0 / 3.0),
        (25.0 / 48.0, 0.2, 0.2),
        (25.0 / 48.0, 0.6, 0.2),
        (25.0 / 48.0, 0.2, 0.6),
    ];
    let mut scalar = 0.0;
    for (weight, u, v) in rule {
        let source = [
            nodes[0][0] + u * ab[0] + v * ac[0],
            nodes[0][1] + u * ab[1] + v * ac[1],
            nodes[0][2] + u * ab[2] + v * ac[2],
        ];
        let r = [obs[0] - source[0], obs[1] - source[1], obs[2] - source[2]];
        scalar += weight / (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    }
    let factor = MU0_OVER_4PI * area * scalar;
    current_density.map(|component| factor * component)
}

fn legacy_triangle_vector_potential(
    nodes: [[f64; 3]; 3],
    nodal_values: [f64; 3],
    obs: [f64; 3],
) -> [f64; 3] {
    let a0 = legacy_triangle_vector_potential_basis(nodes, obs);
    let a1 = legacy_triangle_vector_potential_basis([nodes[1], nodes[2], nodes[0]], obs);
    let a2 = legacy_triangle_vector_potential_basis([nodes[2], nodes[0], nodes[1]], obs);
    [
        nodal_values[0] * a0[0] + nodal_values[1] * a1[0] + nodal_values[2] * a2[0],
        nodal_values[0] * a0[1] + nodal_values[1] * a1[1] + nodal_values[2] * a2[1],
        nodal_values[0] * a0[2] + nodal_values[1] * a1[2] + nodal_values[2] * a2[2],
    ]
}

fn bench_triangle_flux_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("Triangle flux density far field");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));
    let nodes = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let nodal_values = [0.0, 1.0, 0.0];
    let obs = [0.25, 0.25, 10.0];

    group.bench_function("analytic", |b| {
        b.iter(|| {
            let nodes = black_box(nodes);
            black_box(flux_density_triangle(
                nodes[0],
                nodes[1],
                nodes[2],
                black_box(nodal_values),
                black_box(obs),
            ))
        });
    });
    group.bench_function("legacy dunavant3", |b| {
        b.iter(|| {
            black_box(legacy_triangle_flux_density(
                black_box(nodes),
                black_box(nodal_values),
                black_box(obs),
            ))
        });
    });
    group.finish();
}

fn bench_triangle_vector_potential(c: &mut Criterion) {
    let mut group = c.benchmark_group("Triangle vector potential far field");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));
    let nodes = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let nodal_values = [0.0, 1.0, 0.0];
    let current_density = [0.0, 1.0, 0.0];
    let obs = [0.25, 0.25, 10.0];

    group.bench_function("analytic", |b| {
        b.iter(|| {
            let nodes = black_box(nodes);
            black_box(vector_potential_triangle(
                nodes[0],
                nodes[1],
                nodes[2],
                black_box(nodal_values),
                black_box(obs),
            ))
        });
    });
    group.bench_function("legacy dunavant3", |b| {
        b.iter(|| {
            black_box(legacy_triangle_vector_potential(
                black_box(nodes),
                black_box(current_density),
                black_box(obs),
            ))
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_triangle_coupling,
    bench_inductance_assembly,
    bench_triangle_flux_density,
    bench_triangle_vector_potential
);
criterion_main!(benches);
