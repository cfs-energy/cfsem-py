#![allow(clippy::all)] // Criterion requires black-boxed benchmark inputs and outputs.

use cfsem::MU0_OVER_4PI;
use cfsem::physics::boundary_element::{
    QuadratureKind, flux_density_triangle, triangle_geometric_coupling,
    triangle_geometric_coupling_regular,
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

fn bench_triangle_coupling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Triangle geometric coupling");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));

    for (label, aspect, separation) in [
        ("far aspect 1", 1.0, 10.0),
        ("far aspect 67", 67.0, 10.0),
        ("edge-sharing aspect 1", 1.0, 0.0),
        ("edge-sharing aspect 67", 67.0, 0.0),
    ] {
        let (source, target) = pair(aspect, separation);
        group.bench_with_input(BenchmarkId::new("production", label), &label, |b, _| {
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
        });
        group.bench_with_input(BenchmarkId::new("legacy nested", label), &label, |b, _| {
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
        });
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

fn bench_triangle_flux_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("Triangle flux density far field");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));
    let nodes = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let nodal_values = [0.0, 1.0, 0.0];
    let current_density = [0.0, 1.0, 0.0];
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
    bench_triangle_flux_density
);
criterion_main!(benches);
