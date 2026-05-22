#![allow(clippy::all)] // Clippy will attempt to remove black_box() internals

use cfsem::physics::hierarchical::kernels::{
    DipoleTarget, DipoleTargets, LinearFilamentFluxDensityKernel, LinearFilamentSource,
    LinearFilamentSources, LinearFilamentVectorPotentialKernel,
};
use cfsem::physics::hierarchical::{
    ClusterTree, EvaluationScratch, HierarchicalError, HierarchicalKernel, SourceNodeSummaries,
    eval, eval_par, scratch_len_par, update_summaries,
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

struct HierarchicalLinearFilamentSolve<'a, K>
where
    K: HierarchicalKernel<
            Scalar = f64,
            SourceGeometry = LinearFilamentSource<f64>,
            TargetGeometry = DipoleTarget<f64>,
            SourceMoment = f64,
            Output = [f64; 3],
        > + Sync,
{
    kernel: K,
    sources: LinearFilamentSources<'a, f64>,
    targets: DipoleTargets<'a, f64>,
    currents: &'a [f64],
    source_tree: ClusterTree<f64>,
    source_summaries: SourceNodeSummaries<K>,
    vector_out: Vec<[f64; 3]>,
    scratch_value: [[f64; 3]; 1],
    parallel_scratch_value: Vec<[f64; 3]>,
}

impl<'a, K> HierarchicalLinearFilamentSolve<'a, K>
where
    K: HierarchicalKernel<
            Scalar = f64,
            SourceGeometry = LinearFilamentSource<f64>,
            TargetGeometry = DipoleTarget<f64>,
            SourceMoment = f64,
            Output = [f64; 3],
        > + Sync,
{
    fn new(
        kernel: K,
        xyzfil: (&'a [f64], &'a [f64], &'a [f64]),
        dlxyzfil: (&'a [f64], &'a [f64], &'a [f64]),
        currents: &'a [f64],
        wire_radius: &'a [f64],
        xyzobs: (&'a [f64], &'a [f64], &'a [f64]),
    ) -> Self {
        let sources = LinearFilamentSources::new(xyzfil, dlxyzfil, wire_radius);
        let targets = DipoleTargets::new(xyzobs.0, xyzobs.1, xyzobs.2);

        let source_tree = ClusterTree::build_morton_lbvh(sources).unwrap();
        let source_summaries = SourceNodeSummaries::<K>::new(source_tree.as_view());
        let target_count = xyzobs.0.len();
        let vector_out = vec![[0.0; 3]; target_count];
        let parallel_scratch_value = vec![[0.0; 3]; scratch_len_par(target_count)];

        Self {
            kernel,
            sources,
            targets,
            currents,
            source_tree,
            source_summaries,
            vector_out,
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
                self.currents,
                &mut self.source_summaries.node_summaries,
            ),
            HierarchicalError::Ok
        );
        let mut scratch = EvaluationScratch {
            contribution: &mut self.scratch_value,
        };
        assert_eq!(
            eval(
                &self.kernel,
                self.source_tree.as_view(),
                &self.source_summaries.node_summaries,
                self.sources,
                self.targets,
                self.currents,
                HIERARCHICAL_THETA,
                &mut self.vector_out,
                &mut scratch,
            ),
            HierarchicalError::Ok
        );

        for i in 0..self.vector_out.len() {
            out.0[i] = self.vector_out[i][0];
            out.1[i] = self.vector_out[i][1];
            out.2[i] = self.vector_out[i][2];
        }
    }

    fn solve_into_par(&mut self, out: (&mut [f64], &mut [f64], &mut [f64])) {
        assert_eq!(
            update_summaries(
                &self.kernel,
                self.source_tree.as_view(),
                self.sources,
                self.currents,
                &mut self.source_summaries.node_summaries,
            ),
            HierarchicalError::Ok
        );
        let mut scratch = EvaluationScratch {
            contribution: &mut self.parallel_scratch_value,
        };
        assert_eq!(
            eval_par(
                &self.kernel,
                self.source_tree.as_view(),
                &self.source_summaries.node_summaries,
                self.sources,
                self.targets,
                self.currents,
                HIERARCHICAL_THETA,
                &mut self.vector_out,
                &mut scratch,
            ),
            HierarchicalError::Ok
        );

        for i in 0..self.vector_out.len() {
            out.0[i] = self.vector_out[i][0];
            out.1[i] = self.vector_out[i][1];
            out.2[i] = self.vector_out[i][2];
        }
    }
}

fn hierarchical_linear_filament_build_and_solve<K>(
    kernel: K,
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    currents: &[f64],
    wire_radius: &[f64],
    xyzobs: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) where
    K: HierarchicalKernel<
            Scalar = f64,
            SourceGeometry = LinearFilamentSource<f64>,
            TargetGeometry = DipoleTarget<f64>,
            SourceMoment = f64,
            Output = [f64; 3],
        > + Sync,
{
    let mut solve = HierarchicalLinearFilamentSolve::new(
        kernel,
        xyzfil,
        dlxyzfil,
        currents,
        wire_radius,
        xyzobs,
    );
    solve.solve_into(out);
}

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

            let mut hierarchical = HierarchicalLinearFilamentSolve::new(
                LinearFilamentFluxDensityKernel::<f64>::new(),
                (&input.xfil, &input.yfil, &input.zfil),
                (&input.dlxfil, &input.dlyfil, &input.dlzfil),
                &input.ifil,
                &input.wire_radius,
                (&input.xobs, &input.yobs, &input.zobs),
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
                        black_box(hierarchical.solve_into((&mut bx, &mut by, &mut bz)))
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
                        black_box(hierarchical.solve_into_par((&mut bx, &mut by, &mut bz)))
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Flux Density of Linear Filaments, Hierarchical Build+Solve\n{} Obs. Point(s)",
                        nobs
                    ),
                    ntot,
                ),
                &ntot,
                |b, &_| {
                    b.iter(|| {
                        let n = input.xobs.len();
                        let (mut bx, mut by, mut bz) =
                            (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
                        black_box(hierarchical_linear_filament_build_and_solve(
                            LinearFilamentFluxDensityKernel::<f64>::new(),
                            (&input.xfil, &input.yfil, &input.zfil),
                            (&input.dlxfil, &input.dlyfil, &input.dlzfil),
                            &input.ifil,
                            &input.wire_radius,
                            (&input.xobs, &input.yobs, &input.zobs),
                            (&mut bx, &mut by, &mut bz),
                        ))
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

            let mut hierarchical = HierarchicalLinearFilamentSolve::new(
                LinearFilamentVectorPotentialKernel::<f64>::new(),
                (&input.xfil, &input.yfil, &input.zfil),
                (&input.dlxfil, &input.dlyfil, &input.dlzfil),
                &input.ifil,
                &input.wire_radius,
                (&input.xobs, &input.yobs, &input.zobs),
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
                        black_box(hierarchical.solve_into((&mut ax, &mut ay, &mut az)))
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
                        black_box(hierarchical.solve_into_par((&mut ax, &mut ay, &mut az)))
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!(
                        "Vector Potential of Linear Filaments, Hierarchical Build+Solve\n{} Obs. Point(s)",
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
                        black_box(hierarchical_linear_filament_build_and_solve(
                            LinearFilamentVectorPotentialKernel::<f64>::new(),
                            (&input.xfil, &input.yfil, &input.zfil),
                            (&input.dlxfil, &input.dlyfil, &input.dlzfil),
                            &input.ifil,
                            &input.wire_radius,
                            (&input.xobs, &input.yobs, &input.zobs),
                            (&mut ax, &mut ay, &mut az),
                        ))
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
