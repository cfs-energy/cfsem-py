#![allow(clippy::all)] // Criterion requires black-boxed benchmark inputs and outputs.

use cfsem::math::{hyp2f1, hyp2f1_par, hyp2f1_scalar};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use num_complex::Complex64;
use std::hint::black_box;
use std::time::Duration;

type Case = (&'static str, Complex64, Complex64, Complex64, Complex64);

fn cases() -> [Case; 7] {
    [
        (
            "direct",
            Complex64::new(0.5, 0.25),
            Complex64::new(1.25, -0.5),
            Complex64::new(2.0, 0.75),
            Complex64::new(0.1, 0.2),
        ),
        (
            "polynomial",
            Complex64::new(-2.0, 0.0),
            Complex64::new(1.2, 0.4),
            Complex64::new(3.5, -0.2),
            Complex64::new(2.0, 0.5),
        ),
        (
            "pfaff",
            Complex64::new(0.7, 0.2),
            Complex64::new(1.3, -0.1),
            Complex64::new(2.4, 0.3),
            Complex64::new(-0.5, 0.1),
        ),
        (
            "one",
            Complex64::new(0.4, 0.2),
            Complex64::new(1.1, 0.3),
            Complex64::new(2.5, 0.5),
            Complex64::new(0.98, 0.03),
        ),
        (
            "infinity",
            Complex64::new(0.4, 0.2),
            Complex64::new(1.1, 0.3),
            Complex64::new(2.7, -0.2),
            Complex64::new(4.0, 2.0),
        ),
        (
            "taylor",
            Complex64::new(0.7, 0.2),
            Complex64::new(1.2, -0.3),
            Complex64::new(2.1, 0.1),
            Complex64::new(0.5, 0.866_025_403_784_438_6),
        ),
        (
            "near-integer",
            Complex64::new(0.4, 0.2),
            Complex64::new(0.9, -0.1),
            Complex64::new(3.3, 0.100_000_001),
            Complex64::new(0.97, 0.02),
        ),
    ]
}

fn bench_scalar_regions(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hyp2f1 scalar regions");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(2));
    for (label, a, b, c, z) in cases() {
        group.bench_function(label, |bencher| {
            bencher.iter(|| {
                black_box(hyp2f1_scalar(
                    black_box(a),
                    black_box(b),
                    black_box(c),
                    black_box(z),
                ))
            });
        });
    }
    group.finish();
}

fn bench_vectors(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hyp2f1 vectors");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));
    let regions = cases();
    for length in [1, 64, 4096, 65_536] {
        let mut a = Vec::with_capacity(length);
        let mut b = Vec::with_capacity(length);
        let mut c = Vec::with_capacity(length);
        let mut z = Vec::with_capacity(length);
        for index in 0..length {
            let (_, ai, bi, ci, zi) = regions[index % regions.len()];
            a.push(ai);
            b.push(bi);
            c.push(ci);
            z.push(zi);
        }
        let mut out = vec![Complex64::ZERO; length];

        group.bench_with_input(
            BenchmarkId::new("scalar loop", length),
            &length,
            |bench, _| {
                bench.iter(|| {
                    for index in 0..length {
                        out[index] = hyp2f1_scalar(a[index], b[index], c[index], z[index]);
                    }
                    black_box(&out);
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("serial", length), &length, |bench, _| {
            bench.iter(|| black_box(hyp2f1(&a, &b, &c, &z, &mut out).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("parallel", length), &length, |bench, _| {
            bench.iter(|| black_box(hyp2f1_par(&a, &b, &c, &z, &mut out).unwrap()));
        });
    }
    group.finish();
}

criterion_group!(hyp2f1_benches, bench_scalar_regions, bench_vectors);
criterion_main!(hyp2f1_benches);
