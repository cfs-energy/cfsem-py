#![allow(clippy::all)] // Clippy will attempt to remove black_box() internals

use cfsem::physics::hierarchical::tree::BuildMethod;
use cfsem::physics::hierarchical::{
    flux_density_linear_filament_hierarchical, vector_potential_linear_filament_hierarchical,
};
use cfsem::physics::linear_filament::{
    flux_density_linear_filament, flux_density_linear_filament_par,
    vector_potential_linear_filament, vector_potential_linear_filament_par,
};
use criterion::*;
use std::time::Duration;

use std::hint::black_box;

const HIERARCHICAL_THETA: f64 = 0.05;
const LOOP_RADIUS: f64 = 1.0;
const LOOP_OBS_FRACTION_OFFSET: f64 = 0.027;
const LOOP_CURRENT: f64 = 0.5;
const LOOP_WIRE_RADIUS: f64 = 0.002;

struct LinearFilamentBenchInput {
    xfil: Vec<f64>,
    yfil: Vec<f64>,
    zfil: Vec<f64>,
    dlxfil: Vec<f64>,
    dlyfil: Vec<f64>,
    dlzfil: Vec<f64>,
    ifil: Vec<f64>,
    wire_radius: Vec<f64>,
    xobs: Vec<f64>,
    yobs: Vec<f64>,
    zobs: Vec<f64>,
}

fn circular_loop_linear_filament_bench_input(
    nfils: usize,
    nobs: usize,
) -> LinearFilamentBenchInput {
    let mut xfil = Vec::with_capacity(nfils);
    let mut yfil = Vec::with_capacity(nfils);
    let mut zfil = Vec::with_capacity(nfils);
    let mut dlxfil = Vec::with_capacity(nfils);
    let mut dlyfil = Vec::with_capacity(nfils);
    let mut dlzfil = Vec::with_capacity(nfils);
    let mut ifil = Vec::with_capacity(nfils);
    let mut wire_radius = Vec::with_capacity(nfils);

    for i in 0..nfils {
        let t0 = i as f64 / nfils as f64;
        let t1 = (i + 1) as f64 / nfils as f64;
        let p0 = loop_point(t0);
        let p1 = loop_point(t1);
        xfil.push(p0[0]);
        yfil.push(p0[1]);
        zfil.push(p0[2]);
        dlxfil.push(p1[0] - p0[0]);
        dlyfil.push(p1[1] - p0[1]);
        dlzfil.push(p1[2] - p0[2]);
        ifil.push(LOOP_CURRENT);
        wire_radius.push(LOOP_WIRE_RADIUS);
    }

    let mut xobs = Vec::with_capacity(nobs);
    let mut yobs = Vec::with_capacity(nobs);
    let mut zobs = Vec::with_capacity(nobs);
    for i in 0..nobs {
        let t = if nobs > 1 {
            i as f64 / (nobs - 1) as f64
        } else {
            0.5
        };
        let point = loop_point(t + LOOP_OBS_FRACTION_OFFSET);
        xobs.push(point[0]);
        yobs.push(point[1]);
        zobs.push(point[2]);
    }

    LinearFilamentBenchInput {
        xfil,
        yfil,
        zfil,
        dlxfil,
        dlyfil,
        dlzfil,
        ifil,
        wire_radius,
        xobs,
        yobs,
        zobs,
    }
}

fn loop_point(t: f64) -> [f64; 3] {
    let theta = core::f64::consts::TAU * t;
    [LOOP_RADIUS * theta.cos(), LOOP_RADIUS * theta.sin(), 0.0]
}

fn bench_flux_density_linear_filament(c: &mut Criterion) {
    let mut group = c.benchmark_group("Flux Density of Linear Filaments");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    // Examine logspace with fixed total throughput
    for nfac in [1, 10, 100, 1000].iter() {
        for nfils in (1_usize..=5).map(|i| 10_usize.pow(i as u32)) {
            let nfils = nfils * nfac;
            let nobs = 1000;
            let nobs = nobs / nfac;
            let input = circular_loop_linear_filament_bench_input(nfils, nobs);

            let ntot = nobs * nfils;
            group.throughput(Throughput::Elements(ntot as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("Flux Density of Linear Filaments\n{} Obs. Point(s)", nobs),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        let n = input.xobs.len();
                        let (mut bx, mut by, mut bz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
                        black_box(
                            flux_density_linear_filament(
                                (&input.xobs[..], &input.yobs[..], &input.zobs[..]),
                                (&input.xfil[..], &input.yfil[..], &input.zfil[..]),
                                (&input.dlxfil[..], &input.dlyfil[..], &input.dlzfil[..]),
                                &input.ifil[..],
                                &input.wire_radius,
                                (&mut bx, &mut by, &mut bz),
                            )
                            .unwrap(),
                        )
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Flux Density of Linear Filaments, Parallel\n{} Obs. Point(s)",
                        nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        let n = input.xobs.len();
                        let (mut bx, mut by, mut bz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
                        black_box(
                            flux_density_linear_filament_par(
                                (&input.xobs[..], &input.yobs[..], &input.zobs[..]),
                                (&input.xfil[..], &input.yfil[..], &input.zfil[..]),
                                (&input.dlxfil[..], &input.dlyfil[..], &input.dlzfil[..]),
                                &input.ifil[..],
                                &input.wire_radius,
                                (&mut bx, &mut by, &mut bz),
                            )
                            .unwrap(),
                        )
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Flux Density of Linear Filaments, Hierarchical\n{} Obs. Point(s)",
                        nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        let n = input.xobs.len();
                        let (mut bx, mut by, mut bz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
                        black_box(
                            flux_density_linear_filament_hierarchical(
                                (&input.xobs, &input.yobs, &input.zobs),
                                (&input.xfil, &input.yfil, &input.zfil),
                                (&input.dlxfil, &input.dlyfil, &input.dlzfil),
                                &input.ifil,
                                &input.wire_radius,
                                BuildMethod::LongestAxis,
                                HIERARCHICAL_THETA,
                                false,
                                (&mut bx, &mut by, &mut bz),
                            )
                            .unwrap(),
                        )
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Flux Density of Linear Filaments, Hierarchical Parallel\n{} Obs. Point(s)",
                        nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        let n = input.xobs.len();
                        let (mut bx, mut by, mut bz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
                        black_box(
                            flux_density_linear_filament_hierarchical(
                                (&input.xobs, &input.yobs, &input.zobs),
                                (&input.xfil, &input.yfil, &input.zfil),
                                (&input.dlxfil, &input.dlyfil, &input.dlzfil),
                                &input.ifil,
                                &input.wire_radius,
                                BuildMethod::LongestAxis,
                                HIERARCHICAL_THETA,
                                true,
                                (&mut bx, &mut by, &mut bz),
                            )
                            .unwrap(),
                        )
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_vector_potential_linear_filament(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vector Potential of Linear Filaments");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    // Examine logspace with fixed total throughput
    for nfac in [1, 10, 100, 1000].iter() {
        for nfils in (1_usize..=5).map(|i| 10_usize.pow(i as u32)) {
            let nfils = nfils * nfac;
            let nobs = 1000;
            let nobs = nobs / nfac;
            let input = circular_loop_linear_filament_bench_input(nfils, nobs);

            let ntot = nobs * nfils;
            group.throughput(Throughput::Elements(ntot as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Vector Potential of Linear Filaments\n{} Obs. Point(s)",
                        nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        let n = input.xobs.len();
                        let (mut bx, mut by, mut bz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
                        black_box(
                            vector_potential_linear_filament(
                                (&input.xobs[..], &input.yobs[..], &input.zobs[..]),
                                (&input.xfil[..], &input.yfil[..], &input.zfil[..]),
                                (&input.dlxfil[..], &input.dlyfil[..], &input.dlzfil[..]),
                                &input.ifil[..],
                                &input.wire_radius,
                                (&mut bx, &mut by, &mut bz),
                            )
                            .unwrap(),
                        )
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Vector Potential of Linear Filaments, Parallel\n{} Obs. Point(s)",
                        nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        let n = input.xobs.len();
                        let (mut bx, mut by, mut bz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
                        black_box(
                            vector_potential_linear_filament_par(
                                (&input.xobs[..], &input.yobs[..], &input.zobs[..]),
                                (&input.xfil[..], &input.yfil[..], &input.zfil[..]),
                                (&input.dlxfil[..], &input.dlyfil[..], &input.dlzfil[..]),
                                &input.ifil[..],
                                &input.wire_radius,
                                (&mut bx, &mut by, &mut bz),
                            )
                            .unwrap(),
                        )
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Vector Potential of Linear Filaments, Hierarchical\n{} Obs. Point(s)",
                        nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        let n = input.xobs.len();
                        let (mut ax, mut ay, mut az) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
                        black_box(
                            vector_potential_linear_filament_hierarchical(
                                (&input.xobs, &input.yobs, &input.zobs),
                                (&input.xfil, &input.yfil, &input.zfil),
                                (&input.dlxfil, &input.dlyfil, &input.dlzfil),
                                &input.ifil,
                                &input.wire_radius,
                                BuildMethod::LongestAxis,
                                HIERARCHICAL_THETA,
                                false,
                                (&mut ax, &mut ay, &mut az),
                            )
                            .unwrap(),
                        )
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Vector Potential of Linear Filaments, Hierarchical Parallel\n{} Obs. Point(s)",
                        nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        let n = input.xobs.len();
                        let (mut ax, mut ay, mut az) =
                            (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
                        black_box(
                            vector_potential_linear_filament_hierarchical(
                                (&input.xobs, &input.yobs, &input.zobs),
                                (&input.xfil, &input.yfil, &input.zfil),
                                (&input.dlxfil, &input.dlyfil, &input.dlzfil),
                                &input.ifil,
                                &input.wire_radius,
                                BuildMethod::LongestAxis,
                                HIERARCHICAL_THETA,
                                true,
                                (&mut ax, &mut ay, &mut az),
                            )
                            .unwrap(),
                        )
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    group_bench_flux_density_linear_filament,
    bench_flux_density_linear_filament
);
criterion_group!(
    group_bench_vector_potential_linear_filament,
    bench_vector_potential_linear_filament
);

criterion_main!(
    group_bench_flux_density_linear_filament,
    group_bench_vector_potential_linear_filament
);
