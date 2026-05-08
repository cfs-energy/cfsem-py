use super::evaluator::{
    TargetNodeSummaries, evaluate_into, evaluate_into_par, output_len,
    parallel_evaluation_scratch_len, serial_evaluation_scratch_len,
    update_plan_target_summaries_into,
};
use super::plan::DualInteractionPlan;
use super::*;
use crate::physics::boundary_element::{
    QuadratureKind, flux_density_triangle, vector_potential_triangle,
};
use crate::physics::hierarchical::kernels::{
    BoundaryElementFluxDensityKernel, BoundaryElementTriangle,
    BoundaryElementVectorPotentialKernel, DipoleFluxDensityKernel, DipoleSource, DipoleTarget,
    DipoleVectorPotentialKernel, LinearFilamentFluxDensityKernel, LinearFilamentSource,
    LinearFilamentVectorPotentialKernel,
};
use crate::physics::point_source::segment::flux_density_point_segment_scalar;

#[derive(Clone, Copy)]
struct MockPoint<T: DualTreeScalar> {
    point: [T; 3],
}

impl<T: DualTreeScalar> BoundedGeometry for MockPoint<T> {
    type Scalar = T;

    fn aabb(&self) -> Aabb<Self::Scalar> {
        Aabb::from_point(self.point)
    }

    fn representative_point(&self) -> [Self::Scalar; 3] {
        self.point
    }
}

#[derive(Clone, Copy, Default)]
struct SourceSummary<T: DualTreeScalar> {
    centroid: [T; 3],
    moment: T,
    count: T,
}

#[derive(Clone, Copy, Default)]
struct TargetSummary<T: DualTreeScalar> {
    centroid: [T; 3],
    count: T,
}

struct MockKernel<T: DualTreeScalar> {
    _marker: core::marker::PhantomData<T>,
}

impl<T: DualTreeScalar> MockKernel<T> {
    fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl<T: DualTreeScalar> DualTreeKernel for MockKernel<T> {
    type Scalar = T;
    type SourceGeometry = MockPoint<T>;
    type TargetGeometry = MockPoint<T>;
    type SourceMoment = T;
    type SourceSummary = SourceSummary<T>;
    type TargetSummary = TargetSummary<T>;
    type Output = T;

    fn summarize_leaf_sources(
        &self,
        source_ids: &[u32],
        sources: &[Self::SourceGeometry],
        moments: &[Self::SourceMoment],
        out: &mut Self::SourceSummary,
    ) -> DualTreeError {
        *out = SourceSummary::default();
        for i in 0..source_ids.len() {
            let id = source_ids[i] as usize;
            out.count = out.count + T::ONE;
            out.moment = out.moment + moments[id];
            for axis in 0..3 {
                out.centroid[axis] = out.centroid[axis] + sources[id].point[axis];
            }
        }
        if out.count > T::ZERO {
            for axis in 0..3 {
                out.centroid[axis] = out.centroid[axis] / out.count;
            }
        }
        DualTreeError::Ok
    }

    fn combine_source_summaries(
        &self,
        children: &[Self::SourceSummary],
        _child_ids: &[u32],
        out: &mut Self::SourceSummary,
    ) -> DualTreeError {
        *out = SourceSummary::default();
        for i in 0..children.len() {
            out.count = out.count + children[i].count;
            out.moment = out.moment + children[i].moment;
            for axis in 0..3 {
                out.centroid[axis] =
                    out.centroid[axis] + children[i].centroid[axis] * children[i].count;
            }
        }
        if out.count > T::ZERO {
            for axis in 0..3 {
                out.centroid[axis] = out.centroid[axis] / out.count;
            }
        }
        DualTreeError::Ok
    }

    fn summarize_leaf_targets(
        &self,
        target_ids: &[u32],
        targets: &[Self::TargetGeometry],
        out: &mut Self::TargetSummary,
    ) -> DualTreeError {
        *out = TargetSummary::default();
        for i in 0..target_ids.len() {
            let id = target_ids[i] as usize;
            out.count = out.count + T::ONE;
            for axis in 0..3 {
                out.centroid[axis] = out.centroid[axis] + targets[id].point[axis];
            }
        }
        if out.count > T::ZERO {
            for axis in 0..3 {
                out.centroid[axis] = out.centroid[axis] / out.count;
            }
        }
        DualTreeError::Ok
    }

    fn combine_target_summaries(
        &self,
        children: &[Self::TargetSummary],
        _child_ids: &[u32],
        out: &mut Self::TargetSummary,
    ) -> DualTreeError {
        *out = TargetSummary::default();
        for i in 0..children.len() {
            out.count = out.count + children[i].count;
            for axis in 0..3 {
                out.centroid[axis] =
                    out.centroid[axis] + children[i].centroid[axis] * children[i].count;
            }
        }
        if out.count > T::ZERO {
            for axis in 0..3 {
                out.centroid[axis] = out.centroid[axis] / out.count;
            }
        }
        DualTreeError::Ok
    }

    fn eval_exact(
        &self,
        target: &Self::TargetGeometry,
        source: &Self::SourceGeometry,
        moment: &Self::SourceMoment,
        out: &mut Self::Output,
    ) -> DualTreeError {
        let r2 = dist2(target.point, source.point);
        *out = *moment / (T::ONE + r2);
        DualTreeError::Ok
    }

    fn eval_far(
        &self,
        target: &Self::TargetSummary,
        source: &Self::SourceSummary,
        out: &mut Self::Output,
    ) -> DualTreeError {
        let r2 = dist2(target.centroid, source.centroid);
        *out = source.moment / (T::ONE + r2);
        DualTreeError::Ok
    }

    fn zero_output(&self, out: &mut Self::Output) {
        *out = T::ZERO;
    }

    fn accumulate(&self, out: &mut Self::Output, contribution: &Self::Output) {
        *out = *out + *contribution;
    }
}

fn dist2<T: DualTreeScalar>(a: [T; 3], b: [T; 3]) -> T {
    let mut out = T::ZERO;
    for axis in 0..3 {
        let d = a[axis] - b[axis];
        out = out + d * d;
    }
    out
}

fn vec_norm3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn points_f64(values: &[[f64; 3]]) -> Vec<MockPoint<f64>> {
    let mut out = Vec::new();
    for i in 0..values.len() {
        out.push(MockPoint { point: values[i] });
    }
    out
}

fn points_f32(values: &[[f32; 3]]) -> Vec<MockPoint<f32>> {
    let mut out = Vec::new();
    for i in 0..values.len() {
        out.push(MockPoint { point: values[i] });
    }
    out
}

#[test]
fn aabb_union_and_gap() {
    let a = Aabb::from_point([0.0_f64, 0.0, 0.0]);
    let b = Aabb::from_point([2.0_f64, 3.0, 0.0]);
    let c = a.union(b);
    assert_eq!(c.min, [0.0, 0.0, 0.0]);
    assert_eq!(c.max, [2.0, 3.0, 0.0]);
    assert_eq!(c.diameter_sq(), 13.0);

    let d = Aabb {
        min: [4.0, 3.0, 0.0],
        max: [5.0, 4.0, 1.0],
    };
    assert_eq!(c.gap_distance_sq(&d), 4.0);
}

#[test]
fn dipole_source_aabb_bounds_magnetized_sphere() {
    let source = DipoleSource {
        position: [1.0_f64, -2.0, 3.0],
        outer_radius: 0.25,
    };
    let aabb = source.aabb();
    assert_eq!(aabb.min, [0.75, -2.25, 2.75]);
    assert_eq!(aabb.max, [1.25, -1.75, 3.25]);

    let point_source = DipoleSource {
        position: [1.0_f64, -2.0, 3.0],
        outer_radius: 0.0,
    };
    assert_eq!(point_source.aabb(), Aabb::from_point(point_source.position));
}

#[test]
fn dipole_exact_uses_magnetized_sphere_radius() {
    let kernel = DipoleFluxDensityKernel::<f64>::new();
    let source = DipoleSource {
        position: [0.0, 0.0, 0.0],
        outer_radius: 2.0,
    };
    let target = DipoleTarget {
        position: [0.5, 0.0, 0.0],
    };
    let moment = [0.0, 0.0, 3.0];
    let mut out = [0.0; 3];

    assert_eq!(
        kernel.eval_exact(&target, &source, &moment, &mut out),
        DualTreeError::Ok
    );

    let expected = crate::physics::point_source::dipole::flux_density_dipole_scalar(
        (0.0, 0.0, 0.0),
        (moment[0], moment[1], moment[2]),
        source.outer_radius,
        (target.position[0], target.position[1], target.position[2]),
    );
    assert_eq!(out, [expected.0, expected.1, expected.2]);
}

#[test]
fn dipole_b_and_a_kernels_reuse_tree_and_plan_against_point_source() {
    let b_kernel = DipoleFluxDensityKernel::<f64>::new();
    let a_kernel = DipoleVectorPotentialKernel::<f64>::new();
    let sources = [
        DipoleSource {
            position: [0.0, 0.0, 0.0],
            outer_radius: 0.5,
        },
        DipoleSource {
            position: [1.0, 0.5, 0.0],
            outer_radius: 0.0,
        },
    ];
    let targets = [
        DipoleTarget {
            position: [0.25, 0.0, 0.0],
        },
        DipoleTarget {
            position: [3.0, 1.0, 0.5],
        },
    ];
    let moments = [[0.0, 0.0, 2.0], [0.0, 1.0, 0.5]];

    let source_tree = ClusterTree::build(&sources, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 1, 0.0, 1, false).unwrap();

    let mut b_source_summaries =
        SourceNodeSummaries::<DipoleFluxDensityKernel<f64>>::new(source_tree.as_view());
    let mut b_target_summaries =
        TargetNodeSummaries::<DipoleFluxDensityKernel<f64>>::new_for_plan(plan.as_view());
    let mut a_source_summaries =
        SourceNodeSummaries::<DipoleVectorPotentialKernel<f64>>::new(source_tree.as_view());
    let mut a_target_summaries =
        TargetNodeSummaries::<DipoleVectorPotentialKernel<f64>>::new_for_plan(plan.as_view());

    assert_eq!(
        update_source_summaries_into(
            &b_kernel,
            source_tree.as_view(),
            &sources,
            &moments,
            &mut b_source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_plan_target_summaries_into(
            &b_kernel,
            plan.as_view(),
            &targets,
            &mut b_target_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_source_summaries_into(
            &a_kernel,
            source_tree.as_view(),
            &sources,
            &moments,
            &mut a_source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_plan_target_summaries_into(
            &a_kernel,
            plan.as_view(),
            &targets,
            &mut a_target_summaries,
        ),
        DualTreeError::Ok
    );

    let mut scratch_value = [[0.0; 3]];
    let mut scratch = EvaluationScratch {
        contribution: &mut scratch_value,
    };
    let mut b_out = [[0.0; 3]; 2];
    let mut a_out = [[0.0; 3]; 2];

    assert_eq!(
        evaluate_into(
            &b_kernel,
            plan.as_view(),
            source_tree.as_view(),
            &b_source_summaries.node_summaries,
            &b_target_summaries,
            &sources,
            &targets,
            &moments,
            &mut b_out,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        evaluate_into(
            &a_kernel,
            plan.as_view(),
            source_tree.as_view(),
            &a_source_summaries.node_summaries,
            &a_target_summaries,
            &sources,
            &targets,
            &moments,
            &mut a_out,
            &mut scratch,
        ),
        DualTreeError::Ok
    );

    for target_id in 0..targets.len() {
        let mut expected_b = [0.0; 3];
        let mut expected_a = [0.0; 3];
        for source_id in 0..sources.len() {
            let source = sources[source_id];
            let moment = moments[source_id];
            let target = targets[target_id];
            let b = crate::physics::point_source::dipole::flux_density_dipole_scalar(
                (source.position[0], source.position[1], source.position[2]),
                (moment[0], moment[1], moment[2]),
                source.outer_radius,
                (target.position[0], target.position[1], target.position[2]),
            );
            let a = crate::physics::point_source::dipole::vector_potential_dipole_scalar(
                (source.position[0], source.position[1], source.position[2]),
                (moment[0], moment[1], moment[2]),
                source.outer_radius,
                (target.position[0], target.position[1], target.position[2]),
            );
            expected_b[0] += b.0;
            expected_b[1] += b.1;
            expected_b[2] += b.2;
            expected_a[0] += a.0;
            expected_a[1] += a.1;
            expected_a[2] += a.2;
        }
        for axis in 0..3 {
            assert!((b_out[target_id][axis] - expected_b[axis]).abs() < 1.0e-20);
            assert!((a_out[target_id][axis] - expected_a[axis]).abs() < 1.0e-20);
        }
    }
}

#[test]
fn linear_filament_source_aabb_bounds_full_segment() {
    let source = LinearFilamentSource {
        start: [1.0_f64, -2.0, 3.0],
        end: [-1.0, 4.0, 2.5],
        wire_radius: 0.25,
    };
    let aabb = source.aabb();
    assert_eq!(aabb.min, [-1.25, -2.25, 2.25]);
    assert_eq!(aabb.max, [1.25, 4.25, 3.25]);
    assert_eq!(source.representative_point(), [0.0, 1.0, 2.75]);
}

#[test]
fn linear_filament_exact_matches_scalar_and_supports_f32() {
    let kernel = LinearFilamentFluxDensityKernel::<f64>::new();
    let source = LinearFilamentSource {
        start: [0.0, 0.0, 0.0],
        end: [0.0, 0.0, 1.0],
        wire_radius: 0.01,
    };
    let target = DipoleTarget {
        position: [0.2, 0.0, 0.5],
    };
    let current = 3.0;
    let mut out = [0.0; 3];

    assert_eq!(
        kernel.eval_exact(&target, &source, &current, &mut out),
        DualTreeError::Ok
    );
    let expected = crate::physics::linear_filament::flux_density_linear_filament_scalar(
        (
            (source.start[0], source.start[1], source.start[2]),
            (source.end[0], source.end[1], source.end[2]),
            current,
        ),
        source.wire_radius,
        (target.position[0], target.position[1], target.position[2]),
    );
    assert_eq!(out, [expected.0, expected.1, expected.2]);

    let bf32 = crate::physics::linear_filament::flux_density_linear_filament_scalar(
        ((0.0_f32, 0.0, 0.0), (0.0, 0.0, 1.0), 3.0),
        0.01,
        (0.2, 0.0, 0.5),
    );
    assert!(bf32.1.abs() > 0.0);
}

#[test]
fn linear_filament_theta_zero_matches_dense_and_serial_direct() {
    let kernel = LinearFilamentFluxDensityKernel::<f64>::new();
    let sources = [
        LinearFilamentSource {
            start: [0.0, 0.0, 0.0],
            end: [0.0, 0.0, 1.0],
            wire_radius: 0.01,
        },
        LinearFilamentSource {
            start: [0.5, 0.0, 0.0],
            end: [0.5, 0.2, 1.0],
            wire_radius: 0.02,
        },
    ];
    let targets = [
        DipoleTarget {
            position: [1.0, 0.0, 0.5],
        },
        DipoleTarget {
            position: [0.25, 0.8, 0.25],
        },
    ];
    let currents = [2.0, -1.5];

    let source_tree = ClusterTree::build(&sources, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 1, 0.0, 1, false).unwrap();
    let mut source_summaries =
        SourceNodeSummaries::<LinearFilamentFluxDensityKernel<f64>>::new(source_tree.as_view());
    let mut target_summaries =
        TargetNodeSummaries::<LinearFilamentFluxDensityKernel<f64>>::new_for_plan(plan.as_view());

    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources,
            &currents,
            &mut source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_plan_target_summaries_into(&kernel, plan.as_view(), &targets, &mut target_summaries,),
        DualTreeError::Ok
    );

    let mut scratch_value = [[0.0; 3]];
    let mut scratch = EvaluationScratch {
        contribution: &mut scratch_value,
    };
    let mut bh = [[0.0; 3]; 2];
    let mut dense = [[0.0; 3]; 2];
    assert_eq!(
        evaluate_into(
            &kernel,
            plan.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries,
            &sources,
            &targets,
            &currents,
            &mut bh,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        dense_direct_evaluate_into(
            &kernel,
            &sources,
            &targets,
            &currents,
            &mut dense,
            &mut scratch,
        ),
        DualTreeError::Ok
    );

    let xp = [targets[0].position[0], targets[1].position[0]];
    let yp = [targets[0].position[1], targets[1].position[1]];
    let zp = [targets[0].position[2], targets[1].position[2]];
    let xfil = [sources[0].start[0], sources[1].start[0]];
    let yfil = [sources[0].start[1], sources[1].start[1]];
    let zfil = [sources[0].start[2], sources[1].start[2]];
    let dlx = [
        sources[0].end[0] - sources[0].start[0],
        sources[1].end[0] - sources[1].start[0],
    ];
    let dly = [
        sources[0].end[1] - sources[0].start[1],
        sources[1].end[1] - sources[1].start[1],
    ];
    let dlz = [
        sources[0].end[2] - sources[0].start[2],
        sources[1].end[2] - sources[1].start[2],
    ];
    let wire_radius = [sources[0].wire_radius, sources[1].wire_radius];
    let mut bx = [0.0; 2];
    let mut by = [0.0; 2];
    let mut bz = [0.0; 2];
    crate::physics::linear_filament::flux_density_linear_filament(
        (&xp, &yp, &zp),
        (&xfil, &yfil, &zfil),
        (&dlx, &dly, &dlz),
        &currents,
        &wire_radius,
        (&mut bx, &mut by, &mut bz),
    )
    .unwrap();

    for i in 0..bh.len() {
        for axis in 0..3 {
            assert!((bh[i][axis] - dense[i][axis]).abs() < 1.0e-20);
        }
        assert!((bh[i][0] - bx[i]).abs() < 1.0e-20);
        assert!((bh[i][1] - by[i]).abs() < 1.0e-20);
        assert!((bh[i][2] - bz[i]).abs() < 1.0e-20);
    }
}

#[test]
fn linear_filament_vector_potential_theta_zero_matches_dense_and_serial_direct() {
    let kernel = LinearFilamentVectorPotentialKernel::<f64>::new();
    let sources = [
        LinearFilamentSource {
            start: [0.0, 0.0, 0.0],
            end: [0.0, 0.0, 1.0],
            wire_radius: 0.01,
        },
        LinearFilamentSource {
            start: [0.5, 0.0, 0.0],
            end: [0.5, 0.2, 1.0],
            wire_radius: 0.02,
        },
    ];
    let targets = [
        DipoleTarget {
            position: [1.0, 0.0, 0.5],
        },
        DipoleTarget {
            position: [0.25, 0.8, 0.25],
        },
    ];
    let currents = [2.0, -1.5];

    let source_tree = ClusterTree::build(&sources, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 1, 0.0, 1, false).unwrap();
    let mut source_summaries =
        SourceNodeSummaries::<LinearFilamentVectorPotentialKernel<f64>>::new(source_tree.as_view());
    let mut target_summaries =
        TargetNodeSummaries::<LinearFilamentVectorPotentialKernel<f64>>::new_for_plan(
            plan.as_view(),
        );

    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources,
            &currents,
            &mut source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_plan_target_summaries_into(&kernel, plan.as_view(), &targets, &mut target_summaries,),
        DualTreeError::Ok
    );

    let mut scratch_value = [[0.0; 3]];
    let mut scratch = EvaluationScratch {
        contribution: &mut scratch_value,
    };
    let mut bh = [[0.0; 3]; 2];
    let mut dense = [[0.0; 3]; 2];
    assert_eq!(
        evaluate_into(
            &kernel,
            plan.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries,
            &sources,
            &targets,
            &currents,
            &mut bh,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        dense_direct_evaluate_into(
            &kernel,
            &sources,
            &targets,
            &currents,
            &mut dense,
            &mut scratch,
        ),
        DualTreeError::Ok
    );

    let xp = [targets[0].position[0], targets[1].position[0]];
    let yp = [targets[0].position[1], targets[1].position[1]];
    let zp = [targets[0].position[2], targets[1].position[2]];
    let xfil = [sources[0].start[0], sources[1].start[0]];
    let yfil = [sources[0].start[1], sources[1].start[1]];
    let zfil = [sources[0].start[2], sources[1].start[2]];
    let dlx = [
        sources[0].end[0] - sources[0].start[0],
        sources[1].end[0] - sources[1].start[0],
    ];
    let dly = [
        sources[0].end[1] - sources[0].start[1],
        sources[1].end[1] - sources[1].start[1],
    ];
    let dlz = [
        sources[0].end[2] - sources[0].start[2],
        sources[1].end[2] - sources[1].start[2],
    ];
    let wire_radius = [sources[0].wire_radius, sources[1].wire_radius];
    let mut ax = [0.0; 2];
    let mut ay = [0.0; 2];
    let mut az = [0.0; 2];
    crate::physics::linear_filament::vector_potential_linear_filament(
        (&xp, &yp, &zp),
        (&xfil, &yfil, &zfil),
        (&dlx, &dly, &dlz),
        &currents,
        &wire_radius,
        (&mut ax, &mut ay, &mut az),
    )
    .unwrap();

    for i in 0..bh.len() {
        for axis in 0..3 {
            assert!((bh[i][axis] - dense[i][axis]).abs() < 1.0e-20);
        }
        assert!((bh[i][0] - ax[i]).abs() < 1.0e-20);
        assert!((bh[i][1] - ay[i]).abs() < 1.0e-20);
        assert!((bh[i][2] - az[i]).abs() < 1.0e-20);
    }
}

#[test]
fn boundary_element_exact_matches_scalar_and_supports_f32() {
    let quad_kind = QuadratureKind::Dunavant3;
    let source = BoundaryElementTriangle {
        n0: [0.0_f64, 0.0, 0.0],
        n1: [1.0, 0.0, 0.0],
        n2: [0.0, 1.0, 0.0],
    };
    let target = DipoleTarget {
        position: [0.25, 0.3, 0.8],
    };
    let moment = [0.0, 1.0, -0.25];

    let b_kernel = BoundaryElementFluxDensityKernel::<f64>::new(quad_kind);
    let mut b_out = [0.0; 3];
    assert_eq!(
        b_kernel.eval_exact(&target, &source, &moment, &mut b_out),
        DualTreeError::Ok
    );
    assert_eq!(
        b_out,
        flux_density_triangle(
            source.n0,
            source.n1,
            source.n2,
            moment,
            target.position,
            quad_kind
        )
    );

    let a_kernel = BoundaryElementVectorPotentialKernel::<f64>::new(quad_kind);
    let mut a_out = [0.0; 3];
    assert_eq!(
        a_kernel.eval_exact(&target, &source, &moment, &mut a_out),
        DualTreeError::Ok
    );
    assert_eq!(
        a_out,
        vector_potential_triangle(
            source.n0,
            source.n1,
            source.n2,
            moment,
            target.position,
            quad_kind
        )
    );

    let bf32 = flux_density_triangle(
        [0.0_f32, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, -0.25],
        [0.25, 0.3, 0.8],
        quad_kind,
    );
    assert!(bf32[0].is_finite());
}

#[test]
fn boundary_element_zero_current_source_does_not_shift_far_summary() {
    let quad_kind = QuadratureKind::Dunavant3;
    let kernel = BoundaryElementFluxDensityKernel::<f64>::new(quad_kind);
    let active_source = BoundaryElementTriangle {
        n0: [0.0, 0.0, 0.0],
        n1: [1.0, 0.0, 0.0],
        n2: [0.0, 1.0, 0.0],
    };
    let inactive_source = BoundaryElementTriangle {
        n0: [40.0, 0.0, 0.0],
        n1: [41.0, 0.0, 0.0],
        n2: [40.0, 1.0, 0.0],
    };
    let targets = [DipoleTarget {
        position: [120.0, 15.0, 25.0],
    }];
    let active_moment = [0.0, 1.0, -0.25];
    let inactive_moment = [3.0, 3.0, 3.0];

    let active_only_sources = [active_source];
    let active_only_moments = [active_moment];
    let source_tree = ClusterTree::build(&active_only_sources, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 1, 1.0e9, 1, false).unwrap();
    assert_eq!(plan.near_target_ids.len(), 0);
    assert_eq!(plan.far_source_node_ids.len(), 1);
    let mut source_summaries =
        SourceNodeSummaries::<BoundaryElementFluxDensityKernel<f64>>::new(source_tree.as_view());
    let mut target_summaries =
        TargetNodeSummaries::<BoundaryElementFluxDensityKernel<f64>>::new_for_plan(plan.as_view());
    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &active_only_sources,
            &active_only_moments,
            &mut source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_plan_target_summaries_into(&kernel, plan.as_view(), &targets, &mut target_summaries),
        DualTreeError::Ok
    );
    let mut scratch_value = [[0.0; 3]];
    let mut scratch = EvaluationScratch {
        contribution: &mut scratch_value,
    };
    let mut active_only = [[0.0; 3]; 1];
    assert_eq!(
        evaluate_into(
            &kernel,
            plan.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries,
            &active_only_sources,
            &targets,
            &active_only_moments,
            &mut active_only,
            &mut scratch,
        ),
        DualTreeError::Ok
    );

    let sources_with_inactive = [active_source, inactive_source];
    let moments_with_inactive = [active_moment, inactive_moment];
    let source_tree = ClusterTree::build(&sources_with_inactive, 2).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 1, 1.0e9, 1, false).unwrap();
    assert_eq!(plan.near_target_ids.len(), 0);
    assert_eq!(plan.far_source_node_ids.len(), 1);
    let mut source_summaries =
        SourceNodeSummaries::<BoundaryElementFluxDensityKernel<f64>>::new(source_tree.as_view());
    let mut target_summaries =
        TargetNodeSummaries::<BoundaryElementFluxDensityKernel<f64>>::new_for_plan(plan.as_view());
    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources_with_inactive,
            &moments_with_inactive,
            &mut source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_plan_target_summaries_into(&kernel, plan.as_view(), &targets, &mut target_summaries),
        DualTreeError::Ok
    );
    let mut with_inactive = [[0.0; 3]; 1];
    assert_eq!(
        evaluate_into(
            &kernel,
            plan.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries,
            &sources_with_inactive,
            &targets,
            &moments_with_inactive,
            &mut with_inactive,
            &mut scratch,
        ),
        DualTreeError::Ok
    );

    for axis in 0..3 {
        assert!((active_only[0][axis] - with_inactive[0][axis]).abs() < 1.0e-30);
    }
}

#[test]
fn boundary_element_forced_far_matches_direct_for_bent_strip_asymptotically() {
    let quad_kind = QuadratureKind::Dunavant3;
    let kernel = BoundaryElementFluxDensityKernel::<f64>::new(quad_kind);
    let sources = [
        BoundaryElementTriangle {
            n0: [0.0, -0.05, 0.0],
            n1: [1.0, -0.05, 0.0],
            n2: [0.0, 0.05, 0.0],
        },
        BoundaryElementTriangle {
            n0: [0.0, 0.05, 0.0],
            n1: [1.0, -0.05, 0.0],
            n2: [1.0, 0.05, 0.0],
        },
        BoundaryElementTriangle {
            n0: [1.0, -0.05, 0.0],
            n1: [1.05, 1.0, 0.0],
            n2: [1.0, 0.05, 0.0],
        },
        BoundaryElementTriangle {
            n0: [1.0, 0.05, 0.0],
            n1: [1.05, 1.0, 0.0],
            n2: [0.95, 1.0, 0.0],
        },
    ];
    let moments = [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let targets = [DipoleTarget {
        position: [20.0, 10.0, 7.0],
    }];

    let source_tree = ClusterTree::build(&sources, sources.len()).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 1, 1.0e9, 1, false).unwrap();
    assert_eq!(plan.near_target_ids.len(), 0);
    assert_eq!(plan.far_source_node_ids.len(), 1);
    let mut source_summaries =
        SourceNodeSummaries::<BoundaryElementFluxDensityKernel<f64>>::new(source_tree.as_view());
    let mut target_summaries =
        TargetNodeSummaries::<BoundaryElementFluxDensityKernel<f64>>::new_for_plan(plan.as_view());
    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources,
            &moments,
            &mut source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_plan_target_summaries_into(&kernel, plan.as_view(), &targets, &mut target_summaries),
        DualTreeError::Ok
    );

    let mut scratch_value = [[0.0; 3]];
    let mut scratch = EvaluationScratch {
        contribution: &mut scratch_value,
    };
    let mut far = [[0.0; 3]; 1];
    let mut direct = [[0.0; 3]; 1];
    assert_eq!(
        evaluate_into(
            &kernel,
            plan.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries,
            &sources,
            &targets,
            &moments,
            &mut far,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        dense_direct_evaluate_into(
            &kernel,
            &sources,
            &targets,
            &moments,
            &mut direct,
            &mut scratch,
        ),
        DualTreeError::Ok
    );

    let err = vec_norm3([
        far[0][0] - direct[0][0],
        far[0][1] - direct[0][1],
        far[0][2] - direct[0][2],
    ]);
    let rel = err / vec_norm3(direct[0]);
    assert!(
        rel < 1.0e-1,
        "relative error {rel:e}, far={:?}, direct={:?}",
        far[0],
        direct[0]
    );
}

#[test]
fn boundary_element_theta_zero_matches_dense_and_scalar_direct() {
    let quad_kind = QuadratureKind::Dunavant3;
    let kernel = BoundaryElementFluxDensityKernel::<f64>::new(quad_kind);
    let sources = [
        BoundaryElementTriangle {
            n0: [0.0, 0.0, 0.0],
            n1: [1.0, 0.0, 0.0],
            n2: [0.0, 1.0, 0.0],
        },
        BoundaryElementTriangle {
            n0: [1.0, 0.0, 0.0],
            n1: [1.0, 1.0, 0.0],
            n2: [0.0, 1.0, 0.0],
        },
    ];
    let targets = [
        DipoleTarget {
            position: [0.25, 0.25, 0.5],
        },
        DipoleTarget {
            position: [1.5, -0.2, 0.8],
        },
    ];
    let moments = [[0.0, 1.0, -0.25], [0.25, 1.0, 0.0]];

    let source_tree = ClusterTree::build(&sources, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 1, 0.0, 1, false).unwrap();
    let mut source_summaries =
        SourceNodeSummaries::<BoundaryElementFluxDensityKernel<f64>>::new(source_tree.as_view());
    let mut target_summaries =
        TargetNodeSummaries::<BoundaryElementFluxDensityKernel<f64>>::new_for_plan(plan.as_view());

    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources,
            &moments,
            &mut source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_plan_target_summaries_into(&kernel, plan.as_view(), &targets, &mut target_summaries,),
        DualTreeError::Ok
    );

    let mut scratch_value = [[0.0; 3]];
    let mut scratch = EvaluationScratch {
        contribution: &mut scratch_value,
    };
    let mut bh = [[0.0; 3]; 2];
    let mut dense = [[0.0; 3]; 2];
    assert_eq!(
        evaluate_into(
            &kernel,
            plan.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries,
            &sources,
            &targets,
            &moments,
            &mut bh,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        dense_direct_evaluate_into(
            &kernel,
            &sources,
            &targets,
            &moments,
            &mut dense,
            &mut scratch,
        ),
        DualTreeError::Ok
    );

    for target_id in 0..targets.len() {
        let mut direct = [0.0; 3];
        for source_id in 0..sources.len() {
            let contrib = flux_density_triangle(
                sources[source_id].n0,
                sources[source_id].n1,
                sources[source_id].n2,
                moments[source_id],
                targets[target_id].position,
                quad_kind,
            );
            for axis in 0..3 {
                direct[axis] += contrib[axis];
            }
        }
        for axis in 0..3 {
            assert!((bh[target_id][axis] - dense[target_id][axis]).abs() < 1.0e-20);
            assert!((bh[target_id][axis] - direct[axis]).abs() < 1.0e-20);
        }
    }
}

#[test]
fn boundary_element_vector_potential_theta_zero_matches_dense_and_scalar_direct() {
    let quad_kind = QuadratureKind::Dunavant3;
    let kernel = BoundaryElementVectorPotentialKernel::<f64>::new(quad_kind);
    let sources = [
        BoundaryElementTriangle {
            n0: [0.0, 0.0, 0.0],
            n1: [1.0, 0.0, 0.0],
            n2: [0.0, 1.0, 0.0],
        },
        BoundaryElementTriangle {
            n0: [1.0, 0.0, 0.0],
            n1: [1.0, 1.0, 0.0],
            n2: [0.0, 1.0, 0.0],
        },
    ];
    let targets = [
        DipoleTarget {
            position: [0.25, 0.25, 0.5],
        },
        DipoleTarget {
            position: [1.5, -0.2, 0.8],
        },
    ];
    let moments = [[0.0, 1.0, -0.25], [0.25, 1.0, 0.0]];

    let source_tree = ClusterTree::build(&sources, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 1, 0.0, 1, false).unwrap();
    let mut source_summaries =
        SourceNodeSummaries::<BoundaryElementVectorPotentialKernel<f64>>::new(
            source_tree.as_view(),
        );
    let mut target_summaries =
        TargetNodeSummaries::<BoundaryElementVectorPotentialKernel<f64>>::new_for_plan(
            plan.as_view(),
        );

    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources,
            &moments,
            &mut source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_plan_target_summaries_into(&kernel, plan.as_view(), &targets, &mut target_summaries,),
        DualTreeError::Ok
    );

    let mut scratch_value = [[0.0; 3]];
    let mut scratch = EvaluationScratch {
        contribution: &mut scratch_value,
    };
    let mut bh = [[0.0; 3]; 2];
    let mut dense = [[0.0; 3]; 2];
    assert_eq!(
        evaluate_into(
            &kernel,
            plan.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries,
            &sources,
            &targets,
            &moments,
            &mut bh,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        dense_direct_evaluate_into(
            &kernel,
            &sources,
            &targets,
            &moments,
            &mut dense,
            &mut scratch,
        ),
        DualTreeError::Ok
    );

    for target_id in 0..targets.len() {
        let mut direct = [0.0; 3];
        for source_id in 0..sources.len() {
            let contrib = vector_potential_triangle(
                sources[source_id].n0,
                sources[source_id].n1,
                sources[source_id].n2,
                moments[source_id],
                targets[target_id].position,
                quad_kind,
            );
            for axis in 0..3 {
                direct[axis] += contrib[axis];
            }
        }
        for axis in 0..3 {
            assert!((bh[target_id][axis] - dense[target_id][axis]).abs() < 1.0e-20);
            assert!((bh[target_id][axis] - direct[axis]).abs() < 1.0e-20);
        }
    }
}

#[test]
fn linear_filament_reuses_tree_for_current_updates() {
    let kernel = LinearFilamentFluxDensityKernel::<f64>::new();
    let sources = [
        LinearFilamentSource {
            start: [0.0, 0.0, 0.0],
            end: [0.0, 0.0, 1.0],
            wire_radius: 0.01,
        },
        LinearFilamentSource {
            start: [0.5, 0.0, 0.0],
            end: [0.5, 0.0, 1.0],
            wire_radius: 0.01,
        },
    ];
    let targets = [DipoleTarget {
        position: [1.0, 0.0, 0.5],
    }];
    let currents0 = [1.0, 1.0];
    let currents1 = [2.0, -1.0];
    let source_tree = ClusterTree::build(&sources, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 1, 0.0, 1, false).unwrap();
    let mut source_summaries =
        SourceNodeSummaries::<LinearFilamentFluxDensityKernel<f64>>::new(source_tree.as_view());
    let mut target_summaries =
        TargetNodeSummaries::<LinearFilamentFluxDensityKernel<f64>>::new_for_plan(plan.as_view());
    assert_eq!(
        update_plan_target_summaries_into(&kernel, plan.as_view(), &targets, &mut target_summaries,),
        DualTreeError::Ok
    );

    let mut scratch_value = [[0.0; 3]];
    let mut scratch = EvaluationScratch {
        contribution: &mut scratch_value,
    };
    let mut out0 = [[0.0; 3]; 1];
    let mut out1 = [[0.0; 3]; 1];

    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources,
            &currents0,
            &mut source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        evaluate_into(
            &kernel,
            plan.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries,
            &sources,
            &targets,
            &currents0,
            &mut out0,
            &mut scratch,
        ),
        DualTreeError::Ok
    );

    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources,
            &currents1,
            &mut source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        evaluate_into(
            &kernel,
            plan.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries,
            &sources,
            &targets,
            &currents1,
            &mut out1,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    assert_ne!(out0, out1);
}

#[test]
fn linear_filament_far_cluster_uses_point_segment_source_term() {
    let kernel = LinearFilamentFluxDensityKernel::<f64>::new();
    let sources = [
        LinearFilamentSource {
            start: [-1.0, 0.0, 0.0],
            end: [-1.0, 0.0, 1.0],
            wire_radius: 0.01,
        },
        LinearFilamentSource {
            start: [1.0, 0.0, 0.0],
            end: [1.0, 0.0, 1.0],
            wire_radius: 0.01,
        },
    ];
    let targets = [DipoleTarget {
        position: [30.0, 2.0, 0.5],
    }];
    let currents = [1.0, 1.0];
    let source_tree = ClusterTree::build(&sources, 2).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 1, 1.0, 1, false).unwrap();
    assert_eq!(plan.far_target_node_ids.len(), 1);
    assert!(plan.near_target_ids.is_empty());

    let mut source_summaries =
        SourceNodeSummaries::<LinearFilamentFluxDensityKernel<f64>>::new(source_tree.as_view());
    let mut target_summaries =
        TargetNodeSummaries::<LinearFilamentFluxDensityKernel<f64>>::new_for_plan(plan.as_view());
    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources,
            &currents,
            &mut source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_plan_target_summaries_into(&kernel, plan.as_view(), &targets, &mut target_summaries,),
        DualTreeError::Ok
    );

    let mut scratch_value = [[0.0; 3]];
    let mut scratch = EvaluationScratch {
        contribution: &mut scratch_value,
    };
    let mut out = [[0.0; 3]; 1];
    assert_eq!(
        evaluate_into(
            &kernel,
            plan.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries,
            &sources,
            &targets,
            &currents,
            &mut out,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    assert!(out[0][0].is_finite());
    assert!(out[0][1].is_finite());
    assert!(out[0][2].is_finite());
    assert!(vec_norm3(out[0]) > 0.0);

    let expected = flux_density_point_segment_scalar(
        ((0.0, 0.0, -0.5), (0.0, 0.0, 1.5), 1.0),
        (
            targets[0].position[0],
            targets[0].position[1],
            targets[0].position[2],
        ),
    );
    assert!((out[0][0] - expected.0).abs() < 1e-18);
    assert!((out[0][1] - expected.1).abs() < 1e-18);
    assert!((out[0][2] - expected.2).abs() < 1e-18);
}

#[test]
fn tree_covers_each_input_once() {
    let points = points_f64(&[
        [3.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ]);
    let tree = ClusterTree::build(&points, 2).unwrap();
    assert_eq!(tree.sorted_indices.len(), points.len());

    let mut seen = vec![false; points.len()];
    for i in 0..tree.sorted_indices.len() {
        let id = tree.sorted_indices[i] as usize;
        assert!(!seen[id]);
        seen[id] = true;
    }
    for item in seen {
        assert!(item);
    }

    for i in 0..tree.leaf_node_ids.len() {
        let node = tree.leaf_node_ids[i] as usize;
        assert!(tree.leaf_count[node] <= 2);
        assert!(tree.leaf_start[node] != ClusterTreeView::<f64>::invalid_index());
    }
}

#[test]
fn morton_tree_covers_each_input_once() {
    let points = points_f64(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.25],
        [10.0, 0.0, 0.0],
        [10.5, 0.25, 0.5],
        [11.0, 1.0, 0.75],
        [2.0, 5.0, 1.0],
        [2.5, 5.5, 1.25],
    ]);
    let tree = ClusterTree::build_morton_lbvh(&points, 2).unwrap();
    assert_eq!(tree.sorted_indices.len(), points.len());
    assert_eq!(tree.node_range_start[0], 0);
    assert_eq!(tree.node_range_count[0] as usize, points.len());

    let mut seen = vec![false; points.len()];
    for i in 0..tree.sorted_indices.len() {
        let id = tree.sorted_indices[i] as usize;
        assert!(!seen[id]);
        seen[id] = true;
    }
    for item in seen {
        assert!(item);
    }

    for i in 0..tree.leaf_node_ids.len() {
        let node = tree.leaf_node_ids[i] as usize;
        assert!(tree.leaf_count[node] <= 2);
        assert!(tree.leaf_start[node] != ClusterTreeView::<f64>::invalid_index());
    }

    for i in 0..points.len() {
        let point = points[i].point;
        for axis in 0..3 {
            assert!(point[axis] >= tree.node_aabb[0].min[axis]);
            assert!(point[axis] <= tree.node_aabb[0].max[axis]);
        }
    }
}

#[test]
fn morton_tree_handles_degenerate_representative_extent() {
    let points = points_f32(&[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    let tree = ClusterTree::build_morton_lbvh(&points, 1).unwrap();
    assert_eq!(tree.sorted_indices, vec![0, 1, 2]);
    assert_eq!(tree.leaf_node_ids.len(), 3);
    assert_eq!(tree.max_depth, 2);
}

#[test]
fn theta_zero_plan_is_all_exact() {
    let sources = points_f64(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    let targets = points_f64(&[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]);
    let source_tree = ClusterTree::build(&sources, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 1, 0.0, 1, false).unwrap();
    assert_eq!(plan.near_target_ids.len(), sources.len() * targets.len());
    assert_eq!(plan.near_source_ids.len(), sources.len() * targets.len());
    assert!(plan.far_target_node_ids.is_empty());
    assert!(plan.far_source_node_ids.is_empty());
}

#[test]
fn separated_plan_has_far_pair() {
    let sources = points_f64(&[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let targets = points_f64(&[[100.0, 0.0, 0.0], [100.0, 1.0, 0.0]]);
    let source_tree = ClusterTree::build(&sources, 2).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 1, 1.0, 1, false).unwrap();
    assert_eq!(plan.far_target_node_ids.len(), 1);
    assert_eq!(plan.far_source_node_ids.len(), 1);
    assert!(plan.near_target_ids.is_empty());
}

#[test]
fn source_summary_update_tracks_moments() {
    let kernel = MockKernel::<f64>::new();
    let sources = points_f64(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]);
    let source_tree = ClusterTree::build(&sources, 1).unwrap();
    let mut summaries = SourceNodeSummaries::<MockKernel<f64>>::new(source_tree.as_view());

    let moments_a = [1.0, 2.0, 3.0];
    let err = update_source_summaries_into(
        &kernel,
        source_tree.as_view(),
        &sources,
        &moments_a,
        &mut summaries.node_summaries,
    );
    assert_eq!(err, DualTreeError::Ok);
    assert_eq!(summaries.node_summaries[0].moment, 6.0);

    let moments_b = [2.0, 4.0, 6.0];
    let err = update_source_summaries_into(
        &kernel,
        source_tree.as_view(),
        &sources,
        &moments_b,
        &mut summaries.node_summaries,
    );
    assert_eq!(err, DualTreeError::Ok);
    assert_eq!(summaries.node_summaries[0].moment, 12.0);
}

#[test]
fn dipole_source_summary_centroid_tracks_moment_weights() {
    let kernel = DipoleFluxDensityKernel::<f64>::new();
    let sources = [
        DipoleSource {
            position: [0.0, 0.0, 0.0],
            outer_radius: 0.0,
        },
        DipoleSource {
            position: [10.0, 0.0, 0.0],
            outer_radius: 0.0,
        },
    ];
    let source_tree = ClusterTree::build(&sources, 2).unwrap();
    let mut summaries =
        SourceNodeSummaries::<DipoleFluxDensityKernel<f64>>::new(source_tree.as_view());

    let moments_a = [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources,
            &moments_a,
            &mut summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert!((summaries.node_summaries[0].centroid[0] - 7.5).abs() < 1.0e-14);

    let moments_b = [[4.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources,
            &moments_b,
            &mut summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert!((summaries.node_summaries[0].centroid[0] - 2.0).abs() < 1.0e-14);
}

#[test]
fn theta_zero_matches_dense_direct_f64() {
    run_theta_zero_matches_dense_direct_f64();
}

#[test]
fn chunked_plan_serial_and_parallel_match_dense_direct() {
    let kernel = MockKernel::<f64>::new();
    let sources = points_f64(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.2, 0.0],
        [2.0, -0.1, 0.0],
        [3.0, 0.4, 0.0],
    ]);
    let targets = points_f64(&[
        [0.1, 0.3, 0.0],
        [0.9, -0.2, 0.0],
        [1.3, 0.5, 0.0],
        [1.8, -0.3, 0.0],
        [2.4, 0.7, 0.0],
        [2.9, -0.5, 0.0],
        [3.2, 0.1, 0.0],
    ]);
    let moments = [1.0_f64, 2.0, -0.5, 3.0];

    let source_tree = ClusterTree::build(&sources, 2).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 2, 0.0, 3, false).unwrap();
    let plan_built_in_parallel =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 2, 0.0, 3, true).unwrap();
    assert_eq!(plan.chunks.len(), 3);
    assert_eq!(plan_built_in_parallel.chunks.len(), plan.chunks.len());
    assert_eq!(
        plan_built_in_parallel.near_source_ids.len(),
        plan.near_source_ids.len()
    );
    assert_eq!(
        plan_built_in_parallel.far_source_node_ids.len(),
        plan.far_source_node_ids.len()
    );
    assert_eq!(output_len(plan.as_view()), targets.len());
    assert_eq!(serial_evaluation_scratch_len(plan.as_view()), 1);
    assert_eq!(
        parallel_evaluation_scratch_len(plan.as_view()),
        plan.chunks.len()
    );

    let mut source_summaries = SourceNodeSummaries::<MockKernel<f64>>::new(source_tree.as_view());
    let mut target_summaries = TargetNodeSummaries::<MockKernel<f64>>::new_for_plan(plan.as_view());
    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources,
            &moments,
            &mut source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_plan_target_summaries_into(&kernel, plan.as_view(), &targets, &mut target_summaries),
        DualTreeError::Ok
    );

    let mut dense = vec![0.0_f64; output_len(plan.as_view())];
    let mut serial = vec![0.0_f64; output_len(plan.as_view())];
    let mut parallel = vec![0.0_f64; output_len(plan.as_view())];
    let mut parallel_build = vec![0.0_f64; output_len(plan_built_in_parallel.as_view())];
    let mut dense_scratch_value = [0.0_f64; 1];
    let mut serial_scratch_value = [0.0_f64; 1];
    let mut parallel_scratch_value = vec![0.0_f64; parallel_evaluation_scratch_len(plan.as_view())];
    let mut parallel_build_scratch_value = [0.0_f64; 1];

    assert_eq!(
        dense_direct_evaluate_into(
            &kernel,
            &sources,
            &targets,
            &moments,
            &mut dense,
            &mut EvaluationScratch {
                contribution: &mut dense_scratch_value
            },
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        evaluate_into(
            &kernel,
            plan.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries,
            &sources,
            &targets,
            &moments,
            &mut serial,
            &mut EvaluationScratch {
                contribution: &mut serial_scratch_value
            },
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        evaluate_into_par(
            &kernel,
            plan.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries,
            &sources,
            &targets,
            &moments,
            &mut parallel,
            &mut EvaluationScratch {
                contribution: &mut parallel_scratch_value
            },
        ),
        DualTreeError::Ok
    );
    let mut parallel_build_target_summaries =
        TargetNodeSummaries::<MockKernel<f64>>::new_for_plan(plan_built_in_parallel.as_view());
    assert_eq!(
        update_plan_target_summaries_into(
            &kernel,
            plan_built_in_parallel.as_view(),
            &targets,
            &mut parallel_build_target_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        evaluate_into(
            &kernel,
            plan_built_in_parallel.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &parallel_build_target_summaries,
            &sources,
            &targets,
            &moments,
            &mut parallel_build,
            &mut EvaluationScratch {
                contribution: &mut parallel_build_scratch_value
            },
        ),
        DualTreeError::Ok
    );

    assert_eq!(serial, dense);
    assert_eq!(parallel, dense);
    assert_eq!(parallel_build, dense);
}

#[test]
fn theta_zero_matches_dense_direct_f32() {
    let kernel = MockKernel::<f32>::new();
    let sources = points_f32(&[[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]]);
    let targets = points_f32(&[[3.0, 0.0, 0.0], [4.0, 1.0, 0.0]]);
    let moments = [1.0_f32, 2.0, 3.0];
    let source_tree = ClusterTree::build(&sources, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 1, 0.0, 1, false).unwrap();
    let mut source_summaries = SourceNodeSummaries::<MockKernel<f32>>::new(source_tree.as_view());
    let mut target_summaries = TargetNodeSummaries::<MockKernel<f32>>::new_for_plan(plan.as_view());
    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources,
            &moments,
            &mut source_summaries.node_summaries
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_plan_target_summaries_into(&kernel, plan.as_view(), &targets, &mut target_summaries),
        DualTreeError::Ok
    );

    let mut scratch_value = [0.0_f32];
    let mut scratch = EvaluationScratch {
        contribution: &mut scratch_value,
    };
    let mut bh = [0.0_f32; 2];
    let mut dense = [0.0_f32; 2];
    assert_eq!(
        evaluate_into(
            &kernel,
            plan.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries,
            &sources,
            &targets,
            &moments,
            &mut bh,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        dense_direct_evaluate_into(
            &kernel,
            &sources,
            &targets,
            &moments,
            &mut dense,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    for i in 0..bh.len() {
        assert!((bh[i] - dense[i]).abs() < 1e-5);
    }
}

#[test]
fn source_tree_theta_zero_matches_dense_direct() {
    let kernel = MockKernel::<f64>::new();
    let sources = points_f64(&[[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, -0.1, 0.0]]);
    let targets = points_f64(&[[0.3, 0.0, 0.0], [3.0, 0.4, 0.0]]);
    let moments = [1.0, -2.0, 0.5];
    let source_tree = ClusterTree::build(&sources, 1).unwrap();
    let mut source_summaries = SourceNodeSummaries::<MockKernel<f64>>::new(source_tree.as_view());
    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources,
            &moments,
            &mut source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );

    let mut source_tree_out = [0.0; 2];
    let mut source_tree_out_par = [0.0; 2];
    let mut dense = [0.0; 2];
    let mut scratch_value = [0.0];
    let mut scratch = EvaluationScratch {
        contribution: &mut scratch_value,
    };
    assert_eq!(
        evaluate_source_tree_into(
            &kernel,
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &sources,
            &targets,
            &moments,
            0.0,
            &mut source_tree_out,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    let mut par_scratch_value =
        vec![0.0; parallel_source_tree_evaluation_scratch_len(targets.len())];
    let mut par_scratch = EvaluationScratch {
        contribution: &mut par_scratch_value,
    };
    assert_eq!(
        evaluate_source_tree_into_par(
            &kernel,
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &sources,
            &targets,
            &moments,
            0.0,
            &mut source_tree_out_par,
            &mut par_scratch,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        dense_direct_evaluate_into(
            &kernel,
            &sources,
            &targets,
            &moments,
            &mut dense,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    for i in 0..targets.len() {
        assert!((source_tree_out[i] - dense[i]).abs() < 1.0e-14);
        assert!((source_tree_out_par[i] - dense[i]).abs() < 1.0e-14);
    }
}

#[test]
fn dense_direct_reports_empty_scratch() {
    let kernel = MockKernel::<f64>::new();
    let sources = points_f64(&[[0.0, 0.0, 0.0]]);
    let targets = points_f64(&[[1.0, 0.0, 0.0]]);
    let moments = [1.0];
    let mut out = [0.0];
    let mut scratch = EvaluationScratch {
        contribution: &mut [],
    };
    assert_eq!(
        dense_direct_evaluate_into(
            &kernel,
            &sources,
            &targets,
            &moments,
            &mut out,
            &mut scratch
        ),
        DualTreeError::ScratchTooSmall
    );
}

fn run_theta_zero_matches_dense_direct_f64() {
    let kernel = MockKernel::<f64>::new();
    let sources = points_f64(&[[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]]);
    let targets = points_f64(&[[3.0, 0.0, 0.0], [4.0, 1.0, 0.0]]);
    let moments = [1.0_f64, 2.0, 3.0];
    let source_tree = ClusterTree::build(&sources, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 1, 0.0, 1, false).unwrap();
    let mut source_summaries = SourceNodeSummaries::<MockKernel<f64>>::new(source_tree.as_view());
    let mut target_summaries = TargetNodeSummaries::<MockKernel<f64>>::new_for_plan(plan.as_view());
    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources,
            &moments,
            &mut source_summaries.node_summaries
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_plan_target_summaries_into(&kernel, plan.as_view(), &targets, &mut target_summaries),
        DualTreeError::Ok
    );

    let mut scratch_value = [0.0_f64];
    let mut scratch = EvaluationScratch {
        contribution: &mut scratch_value,
    };
    let mut bh = [0.0_f64; 2];
    let mut dense = [0.0_f64; 2];
    assert_eq!(
        evaluate_into(
            &kernel,
            plan.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries,
            &sources,
            &targets,
            &moments,
            &mut bh,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        dense_direct_evaluate_into(
            &kernel,
            &sources,
            &targets,
            &moments,
            &mut dense,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    for i in 0..bh.len() {
        assert!((bh[i] - dense[i]).abs() < 1e-12);
    }
}

#[test]
fn dipole_flux_density_kernel_theta_zero_matches_dense() {
    let kernel = DipoleFluxDensityKernel::<f64>::new();
    let sources = [
        DipoleSource {
            position: [0.0, 0.0, 0.0],
            outer_radius: 0.0,
        },
        DipoleSource {
            position: [1.0, 0.5, 0.0],
            outer_radius: 0.0,
        },
        DipoleSource {
            position: [2.0, 0.0, 0.0],
            outer_radius: 0.0,
        },
    ];
    let targets = [
        DipoleTarget {
            position: [3.0, 0.0, 0.0],
        },
        DipoleTarget {
            position: [4.0, 1.0, 0.5],
        },
    ];
    let moments = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.5], [1.0, 0.0, 0.0]];

    let source_tree = ClusterTree::build(&sources, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), &targets, 1, 0.0, 1, false).unwrap();
    let mut source_summaries =
        SourceNodeSummaries::<DipoleFluxDensityKernel<f64>>::new(source_tree.as_view());
    let mut target_summaries =
        TargetNodeSummaries::<DipoleFluxDensityKernel<f64>>::new_for_plan(plan.as_view());

    assert_eq!(
        update_source_summaries_into(
            &kernel,
            source_tree.as_view(),
            &sources,
            &moments,
            &mut source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_plan_target_summaries_into(&kernel, plan.as_view(), &targets, &mut target_summaries,),
        DualTreeError::Ok
    );

    let mut scratch_value = [[0.0; 3]];
    let mut scratch = EvaluationScratch {
        contribution: &mut scratch_value,
    };
    let mut bh = [[0.0; 3]; 2];
    let mut dense = [[0.0; 3]; 2];

    assert_eq!(
        evaluate_into(
            &kernel,
            plan.as_view(),
            source_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries,
            &sources,
            &targets,
            &moments,
            &mut bh,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        dense_direct_evaluate_into(
            &kernel,
            &sources,
            &targets,
            &moments,
            &mut dense,
            &mut scratch,
        ),
        DualTreeError::Ok
    );

    for i in 0..bh.len() {
        for axis in 0..3 {
            assert!((bh[i][axis] - dense[i][axis]).abs() < 1.0e-20);
        }
    }
}
