#![allow(clippy::all)] // Clippy will attempt to remove black_box() internals

use cfsem::physics::hierarchical::evaluator::{
    EvaluationScratch, SourceNodeSummaries, eval, eval_par, scratch_len_par, update_summaries,
};
use cfsem::physics::hierarchical::kernel::{HierarchicalError, HierarchicalKernel};
use cfsem::physics::hierarchical::kernels::{
    DipoleFluxDensityKernel, DipoleMoments, DipoleSource, DipoleSources, DipoleTarget,
    DipoleTargets, DipoleVectorPotentialKernel,
};
use cfsem::physics::hierarchical::tree::ClusterTree;
use cfsem::physics::point_source::{
    flux_density_dipole, flux_density_dipole_par, vector_potential_dipole,
    vector_potential_dipole_par,
};
use criterion::*;
use std::time::Duration;

use std::hint::black_box;

const HIERARCHICAL_THETA: f64 = 0.01;

struct HierarchicalDipoleSolve<
    'a,
    K: HierarchicalKernel<
            Scalar = f64,
            SourceGeometry = DipoleSource<f64>,
            TargetGeometry = DipoleTarget<f64>,
            SourceMoment = [f64; 3],
            Output = [f64; 3],
        > + Sync,
> {
    kernel: K,
    sources: DipoleSources<'a, f64>,
    targets: DipoleTargets<'a, f64>,
    moments: DipoleMoments<'a, f64>,
    source_tree: ClusterTree<f64>,
    source_summaries: SourceNodeSummaries<K>,
    scratch_value: [[f64; 3]; 1],
    parallel_scratch_value: Vec<[f64; 3]>,
}

impl<'a, K> HierarchicalDipoleSolve<'a, K>
where
    K: HierarchicalKernel<
            Scalar = f64,
            SourceGeometry = DipoleSource<f64>,
            TargetGeometry = DipoleTarget<f64>,
            SourceMoment = [f64; 3],
            Output = [f64; 3],
        > + Sync,
{
    fn new(
        kernel: K,
        loc: (&'a [f64], &'a [f64], &'a [f64]),
        moment: (&'a [f64], &'a [f64], &'a [f64]),
        outer_radius: &'a [f64],
        obs: (&'a [f64], &'a [f64], &'a [f64]),
    ) -> Self {
        let sources = DipoleSources::new(loc.0, loc.1, loc.2, outer_radius);
        let targets = DipoleTargets::new(obs.0, obs.1, obs.2);
        let moments = DipoleMoments::new(moment.0, moment.1, moment.2);

        let source_tree = ClusterTree::build_morton_lbvh(sources).unwrap();
        let source_summaries = SourceNodeSummaries::<K>::new(source_tree.as_view());
        let target_count = obs.0.len();
        let parallel_scratch_value = vec![[0.0; 3]; scratch_len_par(target_count)];

        Self {
            kernel,
            sources,
            targets,
            moments,
            source_tree,
            source_summaries,
            scratch_value: [[0.0; 3]; 1],
            parallel_scratch_value,
        }
    }

    fn solve_into(&mut self, out: (&mut [f64], &mut [f64], &mut [f64])) {
        assert_eq!(
            update_summaries(
                &self.kernel,
                self.source_tree.as_view(),
                self.sources,
                self.moments,
                &mut self.source_summaries.node_summaries,
            ),
            HierarchicalError::Ok
        );

        let mut scratch = EvaluationScratch {
            contribution: &mut self.scratch_value,
        };
        let out_components = [out.0, out.1, out.2];
        assert_eq!(
            eval(
                &self.kernel,
                self.source_tree.as_view(),
                &self.source_summaries.node_summaries,
                self.sources,
                self.targets,
                self.moments,
                HIERARCHICAL_THETA,
                out_components,
                &mut scratch,
            ),
            HierarchicalError::Ok
        );
    }

    fn solve_into_par(&mut self, out: (&mut [f64], &mut [f64], &mut [f64])) {
        assert_eq!(
            update_summaries(
                &self.kernel,
                self.source_tree.as_view(),
                self.sources,
                self.moments,
                &mut self.source_summaries.node_summaries,
            ),
            HierarchicalError::Ok
        );

        let mut scratch = EvaluationScratch {
            contribution: &mut self.parallel_scratch_value,
        };
        let out_components = [out.0, out.1, out.2];
        assert_eq!(
            eval_par(
                &self.kernel,
                self.source_tree.as_view(),
                &self.source_summaries.node_summaries,
                self.sources,
                self.targets,
                self.moments,
                HIERARCHICAL_THETA,
                out_components,
                &mut scratch,
            ),
            HierarchicalError::Ok
        );
    }
}

fn hierarchical_dipole_build_and_solve<K>(
    kernel: K,
    loc: (&[f64], &[f64], &[f64]),
    moment: (&[f64], &[f64], &[f64]),
    outer_radius: &[f64],
    obs: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) where
    K: HierarchicalKernel<
            Scalar = f64,
            SourceGeometry = DipoleSource<f64>,
            TargetGeometry = DipoleTarget<f64>,
            SourceMoment = [f64; 3],
            Output = [f64; 3],
        > + Sync,
{
    let mut solve = HierarchicalDipoleSolve::new(kernel, loc, moment, outer_radius, obs);
    solve.solve_into(out);
}

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

            let mut hierarchical = HierarchicalDipoleSolve::new(
                DipoleFluxDensityKernel::<f64>::new(),
                (&locx, &locy, &locz),
                (&momx, &momy, &momz),
                &outer_radius,
                (&obsx, &obsy, &obsz),
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Flux Density of a Magnetic Dipole, Hierarchical\n{} src × {} obs",
                        ndipoles, nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(hierarchical.solve_into((&mut outx, &mut outy, &mut outz)))
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Flux Density of a Magnetic Dipole, Hierarchical Parallel\n{} src × {} obs",
                        ndipoles, nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(hierarchical.solve_into_par((&mut outx, &mut outy, &mut outz)))
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Flux Density of a Magnetic Dipole, Hierarchical Build+Solve\n{} src × {} obs",
                        ndipoles, nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(hierarchical_dipole_build_and_solve(
                            DipoleFluxDensityKernel::<f64>::new(),
                            (&locx, &locy, &locz),
                            (&momx, &momy, &momz),
                            &outer_radius,
                            (&obsx, &obsy, &obsz),
                            (&mut outx, &mut outy, &mut outz),
                        ))
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

            let mut hierarchical_moment = HierarchicalDipoleSolve::new(
                DipoleVectorPotentialKernel::<f64>::new(),
                (&locx, &locy, &locz),
                (&momx, &momy, &momz),
                &outer_radius,
                (&obsx, &obsy, &obsz),
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Vector Potential of a Magnetic Dipole, Hierarchical\n{} src × {} obs",
                        ndipoles, nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(hierarchical_moment.solve_into((&mut outx, &mut outy, &mut outz)))
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Vector Potential of a Magnetic Dipole, Hierarchical Parallel\n{} src × {} obs",
                        ndipoles, nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(
                            hierarchical_moment.solve_into_par((&mut outx, &mut outy, &mut outz)),
                        )
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Vector Potential of a Magnetic Dipole, Hierarchical Build+Solve\n{} src × {} obs",
                        ndipoles, nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        black_box(hierarchical_dipole_build_and_solve(
                            DipoleVectorPotentialKernel::<f64>::new(),
                            (&locx, &locy, &locz),
                            (&momx, &momy, &momz),
                            &outer_radius,
                            (&obsx, &obsy, &obsz),
                            (&mut outx, &mut outy, &mut outz),
                        ))
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
