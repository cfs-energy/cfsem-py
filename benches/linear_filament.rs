#![allow(clippy::all)] // Clippy will attempt to remove black_box() internals

use cfsem::physics::hierarchical::kernels::{
    DipoleTarget, LinearFilamentFluxDensityKernel, LinearFilamentSource,
};
use cfsem::physics::hierarchical::{
    ClusterTree, DualInteractionPlan, DualTreeError, EvaluationScratch, SourceNodeSummaries,
    TargetNodeSummaries, evaluate_into, update_source_summaries_into, update_target_summaries_into,
};
use cfsem::physics::linear_filament::{
    flux_density_linear_filament, flux_density_linear_filament_par,
    vector_potential_linear_filament, vector_potential_linear_filament_par,
};
use criterion::*;
use std::time::Duration;

use std::hint::black_box;

const HIERARCHICAL_LEAF_SIZE: usize = 16;
const HIERARCHICAL_THETA: f64 = 0.7;
const LOOP_RADIUS: f64 = 1.0;
const LOOP_OBS_FRACTION_OFFSET: f64 = 0.027;
const LOOP_CURRENT: f64 = 0.5;
const LOOP_WIRE_RADIUS: f64 = 0.002;

struct HierarchicalLinearFilamentSolve {
    kernel: LinearFilamentFluxDensityKernel<f64>,
    sources: Vec<LinearFilamentSource<f64>>,
    targets: Vec<DipoleTarget<f64>>,
    currents: Vec<f64>,
    source_tree: ClusterTree<f64>,
    target_tree: ClusterTree<f64>,
    plan: DualInteractionPlan,
    source_summaries: SourceNodeSummaries<LinearFilamentFluxDensityKernel<f64>>,
    target_summaries: TargetNodeSummaries<LinearFilamentFluxDensityKernel<f64>>,
    vector_out: Vec<[f64; 3]>,
    scratch_value: [[f64; 3]; 1],
}

impl HierarchicalLinearFilamentSolve {
    fn new(
        xyzfil: (&[f64], &[f64], &[f64]),
        dlxyzfil: (&[f64], &[f64], &[f64]),
        currents: &[f64],
        wire_radius: &[f64],
        xyzobs: (&[f64], &[f64], &[f64]),
    ) -> Self {
        let kernel = LinearFilamentFluxDensityKernel::<f64>::new();
        let mut sources = Vec::with_capacity(xyzfil.0.len());
        for i in 0..xyzfil.0.len() {
            let start = [xyzfil.0[i], xyzfil.1[i], xyzfil.2[i]];
            let end = [
                xyzfil.0[i] + dlxyzfil.0[i],
                xyzfil.1[i] + dlxyzfil.1[i],
                xyzfil.2[i] + dlxyzfil.2[i],
            ];
            sources.push(LinearFilamentSource {
                start,
                end,
                wire_radius: wire_radius[i],
            });
        }

        let mut targets = Vec::with_capacity(xyzobs.0.len());
        for i in 0..xyzobs.0.len() {
            targets.push(DipoleTarget {
                position: [xyzobs.0[i], xyzobs.1[i], xyzobs.2[i]],
            });
        }

        let source_tree = ClusterTree::build_morton_lbvh(&sources, HIERARCHICAL_LEAF_SIZE).unwrap();
        let target_tree = ClusterTree::build_morton_lbvh(&targets, HIERARCHICAL_LEAF_SIZE).unwrap();
        let plan = DualInteractionPlan::build(
            source_tree.as_view(),
            target_tree.as_view(),
            HIERARCHICAL_THETA,
        )
        .unwrap();

        let mut target_summaries =
            TargetNodeSummaries::<LinearFilamentFluxDensityKernel<f64>>::new(target_tree.as_view());
        assert_eq!(
            update_target_summaries_into(
                &kernel,
                target_tree.as_view(),
                &targets,
                &mut target_summaries.node_summaries,
            ),
            DualTreeError::Ok
        );

        let source_summaries =
            SourceNodeSummaries::<LinearFilamentFluxDensityKernel<f64>>::new(source_tree.as_view());
        let vector_out = vec![[0.0; 3]; targets.len()];

        Self {
            kernel,
            sources,
            targets,
            currents: currents.to_vec(),
            source_tree,
            target_tree,
            plan,
            source_summaries,
            target_summaries,
            vector_out,
            scratch_value: [[0.0; 3]; 1],
        }
    }

    fn solve_into(&mut self, out: (&mut [f64], &mut [f64], &mut [f64])) {
        assert_eq!(
            update_source_summaries_into(
                &self.kernel,
                self.source_tree.as_view(),
                &self.sources,
                &self.currents,
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
                self.target_tree.as_view(),
                &self.source_summaries.node_summaries,
                &self.target_summaries.node_summaries,
                &self.sources,
                &self.targets,
                &self.currents,
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

fn hierarchical_linear_filament_build_and_solve(
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    currents: &[f64],
    wire_radius: &[f64],
    xyzobs: (&[f64], &[f64], &[f64]),
    out: (&mut [f64], &mut [f64], &mut [f64]),
) {
    let mut solve =
        HierarchicalLinearFilamentSolve::new(xyzfil, dlxyzfil, currents, wire_radius, xyzobs);
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
