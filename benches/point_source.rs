#![allow(clippy::all)] // Clippy will attempt to remove black_box() internals

use cfsem::physics::hierarchical::kernels::{
    DipoleFluxDensityKernel, DipoleSource, DipoleTarget, DipoleVectorPotentialKernel,
};
use cfsem::physics::hierarchical::{
    ClusterTree, DualInteractionPlan, DualTreeError, DualTreeKernel, EvaluationScratch,
    SourceNodeSummaries, TargetNodeSummaries, evaluate_into, evaluate_into_par,
    update_plan_target_summaries_into, update_source_summaries_into,
};
use cfsem::physics::point_source::{
    flux_density_dipole, flux_density_dipole_par, vector_potential_dipole,
    vector_potential_dipole_par,
};
use criterion::*;
use std::time::Duration;

use std::hint::black_box;

const HIERARCHICAL_LEAF_SIZE: usize = 16;
const HIERARCHICAL_THETA: f64 = 0.7;
const HIERARCHICAL_NUM_CHUNKS: usize = 8;

struct HierarchicalDipoleSolve<
    K: DualTreeKernel<Scalar = f64, SourceMoment = [f64; 3], Output = [f64; 3]> + Sync,
> where
    K::SourceGeometry: From<DipoleSource<f64>>,
    K::TargetGeometry: From<DipoleTarget<f64>>,
{
    kernel: K,
    sources: Vec<K::SourceGeometry>,
    targets: Vec<K::TargetGeometry>,
    moments: Vec<[f64; 3]>,
    source_tree: ClusterTree<f64>,
    plan: DualInteractionPlan<f64>,
    source_summaries: SourceNodeSummaries<K>,
    target_summaries: TargetNodeSummaries<K>,
    vector_out: Vec<[f64; 3]>,
    scratch_value: [[f64; 3]; 1],
    parallel_scratch_value: Vec<[f64; 3]>,
}

impl<K> HierarchicalDipoleSolve<K>
where
    K: DualTreeKernel<Scalar = f64, SourceMoment = [f64; 3], Output = [f64; 3]> + Sync,
    K::SourceGeometry: From<DipoleSource<f64>>,
    K::TargetGeometry: From<DipoleTarget<f64>>,
{
    fn new(
        kernel: K,
        loc: (&[f64], &[f64], &[f64]),
        moment: (&[f64], &[f64], &[f64]),
        outer_radius: &[f64],
        obs: (&[f64], &[f64], &[f64]),
    ) -> Self {
        let mut sources = Vec::with_capacity(loc.0.len());
        for i in 0..loc.0.len() {
            sources.push(
                DipoleSource {
                    position: [loc.0[i], loc.1[i], loc.2[i]],
                    outer_radius: outer_radius[i],
                }
                .into(),
            );
        }

        let mut targets = Vec::with_capacity(obs.0.len());
        for i in 0..obs.0.len() {
            targets.push(
                DipoleTarget {
                    position: [obs.0[i], obs.1[i], obs.2[i]],
                }
                .into(),
            );
        }

        let mut moments = Vec::with_capacity(moment.0.len());
        for i in 0..moment.0.len() {
            moments.push([moment.0[i], moment.1[i], moment.2[i]]);
        }

        let source_tree = ClusterTree::build_morton_lbvh(&sources, HIERARCHICAL_LEAF_SIZE).unwrap();
        let plan = DualInteractionPlan::build(
            source_tree.as_view(),
            &targets,
            HIERARCHICAL_LEAF_SIZE,
            HIERARCHICAL_THETA,
            HIERARCHICAL_NUM_CHUNKS,
            false,
        )
        .unwrap();

        let mut target_summaries = TargetNodeSummaries::<K>::new_for_plan(plan.as_view());
        assert_eq!(
            update_plan_target_summaries_into(
                &kernel,
                plan.as_view(),
                &targets,
                &mut target_summaries,
            ),
            DualTreeError::Ok
        );

        let source_summaries = SourceNodeSummaries::<K>::new(source_tree.as_view());
        let vector_out = vec![[0.0; 3]; targets.len()];
        let parallel_scratch_value = vec![[0.0; 3]; plan.chunks.len()];

        Self {
            kernel,
            sources,
            targets,
            moments,
            source_tree,
            plan,
            source_summaries,
            target_summaries,
            vector_out,
            scratch_value: [[0.0; 3]; 1],
            parallel_scratch_value,
        }
    }

    fn solve_into(&mut self, out: (&mut [f64], &mut [f64], &mut [f64])) {
        assert_eq!(
            update_source_summaries_into(
                &self.kernel,
                self.source_tree.as_view(),
                &self.sources,
                &self.moments,
                &mut self.source_summaries.node_summaries,
            ),
            DualTreeError::Ok
        );

        let mut scratch = EvaluationScratch {
            contribution: &mut self.scratch_value,
        };
        assert_eq!(
            evaluate_into(
                &self.kernel,
                self.plan.as_view(),
                self.source_tree.as_view(),
                &self.source_summaries.node_summaries,
                &self.target_summaries,
                &self.sources,
                &self.targets,
                &self.moments,
                &mut self.vector_out,
                &mut scratch,
            ),
            DualTreeError::Ok
        );

        for i in 0..self.vector_out.len() {
            out.0[i] = self.vector_out[i][0];
            out.1[i] = self.vector_out[i][1];
            out.2[i] = self.vector_out[i][2];
        }
    }

    fn solve_into_par(&mut self, out: (&mut [f64], &mut [f64], &mut [f64])) {
        assert_eq!(
            update_source_summaries_into(
                &self.kernel,
                self.source_tree.as_view(),
                &self.sources,
                &self.moments,
                &mut self.source_summaries.node_summaries,
            ),
            DualTreeError::Ok
        );

        let mut scratch = EvaluationScratch {
            contribution: &mut self.parallel_scratch_value,
        };
        assert_eq!(
            evaluate_into_par(
                &self.kernel,
                self.plan.as_view(),
                self.source_tree.as_view(),
                &self.source_summaries.node_summaries,
                &self.target_summaries,
                &self.sources,
                &self.targets,
                &self.moments,
                &mut self.vector_out,
                &mut scratch,
            ),
            DualTreeError::Ok
        );

        for i in 0..self.vector_out.len() {
            out.0[i] = self.vector_out[i][0];
            out.1[i] = self.vector_out[i][1];
            out.2[i] = self.vector_out[i][2];
        }
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
    K: DualTreeKernel<Scalar = f64, SourceMoment = [f64; 3], Output = [f64; 3]> + Sync,
    K::SourceGeometry: From<DipoleSource<f64>>,
    K::TargetGeometry: From<DipoleTarget<f64>>,
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
