#![allow(clippy::all)] // Clippy will attempt to remove black_box() internals

use cfsem::physics::{
    circular_filament::{
        flux_circular_filament_par, flux_density_circular_filament_par,
        vector_potential_circular_filament, vector_potential_circular_filament_par,
    },
    flux_circular_filament, flux_density_circular_filament,
};
use criterion::*;
use std::time::Duration;

fn bench_flux_circular_filament(c: &mut Criterion) {
    let mut group = c.benchmark_group("Poloidal Flux of a Circular Filament");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    // Examine logspace with fixed total throughput
    for nfac in [1, 10, 100, 1000].iter() {
        for nfils in (0_usize..=5).map(|i| 10_usize.pow(i as u32)) {
            // Filament inputs
            let nfils = nfils * nfac;
            let rfil = vec![1.0 / 7.0_f64; nfils];
            let zfil = vec![1.0 / 11.0_f64; nfils];
            let current = vec![0.5_f64; nfils];

            // Observation points
            let nobs = 1000;
            let nobs = nobs / nfac;
            let robs = vec![2.0 / 7.0_f64; nobs];
            let zobs = vec![2.0 / 11.0_f64; nobs];

            // Output
            let mut out = vec![0.0_f64; nobs];

            let ntot = nobs * nfils;
            group.throughput(Throughput::Elements(ntot as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Poloidal Flux of a Circular Filament\n{} Obs. Point(s)",
                        nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(
                            flux_circular_filament(
                                (&rfil, &zfil, &current),
                                (&robs, &zobs),
                                &mut out,
                            )
                            .unwrap(),
                        )
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Poloidal Flux of a Circular Filament, Parallel\n{} Obs. Point(s)",
                        nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(
                            flux_circular_filament_par(
                                (&rfil, &zfil, &current),
                                (&robs, &zobs),
                                &mut out,
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

fn bench_vector_potential_circular_filament(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vector Potential of a Circular Filament");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    // Examine logspace with fixed total throughput
    for nfac in [1, 10, 100, 1000].iter() {
        for nfils in (0_usize..=5).map(|i| 10_usize.pow(i as u32)) {
            // Filament inputs
            let nfils = nfils * nfac;
            let rfil = vec![1.0 / 7.0_f64; nfils];
            let zfil = vec![1.0 / 11.0_f64; nfils];
            let current = vec![0.5_f64; nfils];

            // Observation points
            let nobs = 1000;
            let nobs = nobs / nfac;
            let robs = vec![2.0 / 7.0_f64; nobs];
            let zobs = vec![2.0 / 11.0_f64; nobs];

            // Output
            let mut out = vec![0.0_f64; nobs];

            let ntot = nobs * nfils;
            group.throughput(Throughput::Elements(ntot as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Vector Potential of a Circular Filament\n{} Obs. Point(s)",
                        nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(
                            vector_potential_circular_filament(
                                (&rfil, &zfil, &current),
                                (&robs, &zobs),
                                &mut out,
                            )
                            .unwrap(),
                        )
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Vector Potential of a Circular Filament, Parallel\n{} Obs. Point(s)",
                        nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(
                            vector_potential_circular_filament_par(
                                (&rfil, &zfil, &current),
                                (&robs, &zobs),
                                &mut out,
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

fn bench_flux_density_circular_filament(c: &mut Criterion) {
    let mut group = c.benchmark_group("Poloidal Flux Density of a Circular Filament");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    // Examine logspace with fixed total throughput
    for nfac in [1, 10, 100, 1000].iter() {
        for nfils in (0_usize..=4).map(|i| 10_usize.pow(i as u32)) {
            // Filament inputs
            let nfils = nfils * nfac;
            let rfil = vec![1.0 / 7.0_f64; nfils];
            let zfil = vec![1.0 / 11.0_f64; nfils];
            let current = vec![0.5_f64; nfils];

            // Observation points
            let nobs = 1000;
            let nobs = nobs / nfac;
            let robs = vec![2.0 / 7.0_f64; nobs];
            let zobs = vec![2.0 / 11.0_f64; nobs];

            // Output
            let mut out = vec![0.0_f64; nobs];
            let mut out1 = vec![0.0_f64; nobs];

            let ntot = nobs * nfils;
            group.throughput(Throughput::Elements(ntot as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Poloidal Flux Density of a Circular Filament\n{} Obs. Point(s)",
                        nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(flux_density_circular_filament(
                            (&rfil, &zfil, &current),
                            (&robs, &zobs),
                            (&mut out, &mut out1),
                        ))
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Poloidal Flux Density of a Circular Filament, Parallel\n{} Obs. Point(s)",
                        nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(flux_density_circular_filament_par(
                            (&rfil, &zfil, &current),
                            (&robs, &zobs),
                            (&mut out, &mut out1),
                        ))
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(group_flux, bench_flux_circular_filament);
criterion_group!(
    group_vector_potential,
    bench_vector_potential_circular_filament
);
criterion_group!(group_flux_density, bench_flux_density_circular_filament);
criterion_main!(group_flux, group_vector_potential, group_flux_density);
