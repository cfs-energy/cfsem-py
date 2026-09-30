#![allow(clippy::all)] // Clippy will attempt to remove black_box() internals

use cfsem::physics::{
    circular_filament::{
        flux_circular_filament_par, flux_density_circular_filament_finite_radius_scalar,
        flux_density_circular_filament_finite_radius_scalar_near,
        flux_density_circular_filament_par, flux_density_circular_filament_scalar,
        vector_potential_circular_filament, vector_potential_circular_filament_par,
    },
    flux_circular_filament, flux_density_circular_filament,
};
use criterion::*;
use std::time::Duration;

use std::hint::black_box;

const WIRE_RADIUS: f64 = 1e-3; // [m] 1 mm conductor radius

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
            let wire_radius = vec![WIRE_RADIUS; nfils];

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
                                &wire_radius,
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
                                &wire_radius,
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
            let wire_radius = vec![WIRE_RADIUS; nfils];

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
                                &wire_radius,
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
                                &wire_radius,
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
            let wire_radius = vec![WIRE_RADIUS; nfils];

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
                            &wire_radius,
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
                            &wire_radius,
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

fn bench_vector_potential_near_conductor(c: &mut Criterion) {
    let mut group = c.benchmark_group("Circular Filament Near Vector Potential");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(10_000));

    // Nonsingular points in the finite-thickness model's domain of validity.
    // Compare zero and positive per-source radii with identical geometry.
    let wire_radius = WIRE_RADIUS;
    for (region, u, v) in [("interior", 0.3, 0.4), ("near exterior", 0.6, 1.0)] {
        for nobs in [1000, 100, 10, 1] {
            let nfils = 10_000 / nobs;
            let rfil = vec![1.0; nfils];
            let zfil = vec![0.25; nfils];
            let current = vec![3.0; nfils];
            let robs = vec![1.0 + u * wire_radius; nobs];
            let zobs = vec![0.25 + v * wire_radius; nobs];
            let mut out = vec![0.0; nobs];
            for (model, radius) in [("thin", 0.0), ("finite", wire_radius)] {
                let radii = vec![radius; nfils];
                group.bench_function(BenchmarkId::new(format!("{region}/{model}"), nobs), |b| {
                    b.iter(|| {
                        vector_potential_circular_filament(
                            black_box((&rfil, &zfil, &current)),
                            black_box(&radii),
                            black_box((&robs, &zobs)),
                            black_box(&mut out),
                        )
                        .unwrap();
                        black_box(&out);
                    });
                });
            }
        }
    }
    group.finish();
}

fn bench_flux_density_finite_radius_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("Circular Filament Scalar Field");
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));

    // Compare both scalar kernels at identical near-conductor observations.
    // The thin-filament field remains nonsingular at these points, but differs
    // physically from the uniform-current finite-radius field inside the wire.
    let filament = (1.0, 0.25, 3.0);
    let wire_radius = WIRE_RADIUS;
    for (name, u, v) in [
        ("interior", 0.3, 0.4),
        ("near exterior", 0.6, 1.0),
        ("blend", 1.2, 1.6),
        ("far", 0.0, 1000.0),
        ("axis", -filament.0 / wire_radius, 1000.0),
    ] {
        let obs = (filament.0 + u * wire_radius, filament.1 + v * wire_radius);
        let input = (filament, wire_radius, obs);
        group.bench_with_input(BenchmarkId::new(name, "thin"), &input, |b, &input| {
            b.iter(|| {
                let (filament, _, obs) = black_box(input);
                black_box(flux_density_circular_filament_scalar(filament, obs))
            });
        });
        if name == "interior" || name == "near exterior" || name == "blend" {
            group.bench_with_input(BenchmarkId::new(name, "near only"), &input, |b, &input| {
                b.iter(|| {
                    let (filament, radius, obs) = black_box(input);
                    black_box(flux_density_circular_filament_finite_radius_scalar_near(
                        filament, radius, obs,
                    ))
                });
            });
        }
        group.bench_with_input(
            BenchmarkId::new(name, "finite radius"),
            &input,
            |b, &input| {
                b.iter(|| {
                    // Black-box the radius too, so the positive-radius branch and
                    // source-dependent elliptic integrals are evaluated each call.
                    let (filament, radius, obs) = black_box(input);
                    black_box(flux_density_circular_filament_finite_radius_scalar(
                        filament, radius, obs,
                    ))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(group_flux, bench_flux_circular_filament);
criterion_group!(
    group_vector_potential,
    bench_vector_potential_circular_filament
);
criterion_group!(group_flux_density, bench_flux_density_circular_filament);
criterion_group!(
    group_finite_radius_scalar,
    bench_flux_density_finite_radius_scalar
);
criterion_group!(
    group_near_vector_potential,
    bench_vector_potential_near_conductor
);
criterion_main!(
    group_flux,
    group_vector_potential,
    group_flux_density,
    group_finite_radius_scalar,
    group_near_vector_potential
);
