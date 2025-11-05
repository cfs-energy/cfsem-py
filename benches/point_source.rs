#![allow(clippy::all)] // Clippy will attempt to remove black_box() internals

use cfsem::physics::point_source::{
    flux_density_dipole, flux_density_dipole_par, vector_potential_dipole,
    vector_potential_dipole_par,
};
use criterion::*;
use std::time::Duration;

use std::hint::black_box;

fn bench_flux_density_dipole(c: &mut Criterion) {
    let mut group = c.benchmark_group("Flux Density of a Magnetic Dipole");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    for ndipoles in [1, 1000] {
        for nobs in [10, 10_000] {
            let locx = vec![0.01_f64; ndipoles];
            let locy = vec![0.02_f64; ndipoles];
            let locz = vec![0.03_f64; ndipoles];

            let momx = vec![0.17_f64; ndipoles];
            let momy = vec![0.077_f64; ndipoles];
            let momz = vec![1.0_f64; ndipoles];

            let outer_radius = vec![0.001_f64; ndipoles];

            let obsx = vec![0.7_f64; nobs];
            let obsy = vec![-0.4_f64; nobs];
            let obsz = vec![0.9_f64; nobs];

            let mut outx = vec![0.0_f64; nobs];
            let mut outy = vec![0.0_f64; nobs];
            let mut outz = vec![0.0_f64; nobs];

            let ntot = nobs * ndipoles;
            group.throughput(Throughput::Elements(ntot as u64));

            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Flux Density of a Magnetic Dipole\n{} src × {} obs",
                        ndipoles, nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(
                            flux_density_dipole(
                                (&locx, &locy, &locz),
                                (&momx, &momy, &momz),
                                &outer_radius,
                                (&obsx, &obsy, &obsz),
                                (&mut outx, &mut outy, &mut outz),
                            )
                            .unwrap(),
                        )
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Flux Density of a Magnetic Dipole, Parallel\n{} src × {} obs",
                        ndipoles, nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(
                            flux_density_dipole_par(
                                (&locx, &locy, &locz),
                                (&momx, &momy, &momz),
                                &outer_radius,
                                (&obsx, &obsy, &obsz),
                                (&mut outx, &mut outy, &mut outz),
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

fn bench_vector_potential_dipole(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vector Potential of a Magnetic Dipole");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    for ndipoles in [1, 1000] {
        for nobs in [10, 10_000] {
            let locx = vec![0.01_f64; ndipoles];
            let locy = vec![0.02_f64; ndipoles];
            let locz = vec![0.03_f64; ndipoles];

            let momx = vec![0.17_f64; ndipoles];
            let momy = vec![0.077_f64; ndipoles];
            let momz = vec![1.0_f64; ndipoles];

            let outer_radius = vec![0.001_f64; ndipoles];

            let obsx = vec![0.7_f64; nobs];
            let obsy = vec![-0.4_f64; nobs];
            let obsz = vec![0.9_f64; nobs];

            let mut outx = vec![0.0_f64; nobs];
            let mut outy = vec![0.0_f64; nobs];
            let mut outz = vec![0.0_f64; nobs];

            let ntot = nobs * ndipoles;
            group.throughput(Throughput::Elements(ntot as u64));

            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Vector Potential of a Magnetic Dipole\n{} src × {} obs",
                        ndipoles, nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(
                            vector_potential_dipole(
                                (&locx, &locy, &locz),
                                (&momx, &momy, &momz),
                                &outer_radius,
                                (&obsx, &obsy, &obsz),
                                (&mut outx, &mut outy, &mut outz),
                            )
                            .unwrap(),
                        )
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Vector Potential of a Magnetic Dipole, Parallel\n{} src × {} obs",
                        ndipoles, nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(
                            vector_potential_dipole_par(
                                (&locx, &locy, &locz),
                                (&momx, &momy, &momz),
                                &outer_radius,
                                (&obsx, &obsy, &obsz),
                                (&mut outx, &mut outy, &mut outz),
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

criterion_group!(group_flux_density, bench_flux_density_dipole);
criterion_group!(group_vector_potential, bench_vector_potential_dipole);
criterion_main!(group_flux_density, group_vector_potential);
