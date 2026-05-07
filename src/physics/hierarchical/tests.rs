use super::*;
use crate::physics::hierarchical::kernels::{
    DipoleMomentKernel, DipoleMultipoleKernel, DipoleSource, DipoleTarget,
};

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
    let kernel = DipoleMomentKernel::<f64>::new();
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
fn theta_zero_plan_is_all_exact() {
    let sources = points_f64(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    let targets = points_f64(&[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]);
    let source_tree = ClusterTree::build(&sources, 1).unwrap();
    let target_tree = ClusterTree::build(&targets, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), target_tree.as_view(), 0.0).unwrap();
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
    let target_tree = ClusterTree::build(&targets, 2).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), target_tree.as_view(), 1.0).unwrap();
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
fn theta_zero_matches_dense_direct_f64() {
    run_theta_zero_matches_dense_direct_f64();
}

#[test]
fn theta_zero_matches_dense_direct_f32() {
    let kernel = MockKernel::<f32>::new();
    let sources = points_f32(&[[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]]);
    let targets = points_f32(&[[3.0, 0.0, 0.0], [4.0, 1.0, 0.0]]);
    let moments = [1.0_f32, 2.0, 3.0];
    let source_tree = ClusterTree::build(&sources, 1).unwrap();
    let target_tree = ClusterTree::build(&targets, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), target_tree.as_view(), 0.0).unwrap();
    let mut source_summaries = SourceNodeSummaries::<MockKernel<f32>>::new(source_tree.as_view());
    let mut target_summaries = TargetNodeSummaries::<MockKernel<f32>>::new(target_tree.as_view());
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
        update_target_summaries_into(
            &kernel,
            target_tree.as_view(),
            &targets,
            &mut target_summaries.node_summaries
        ),
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
            target_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries.node_summaries,
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
    let target_tree = ClusterTree::build(&targets, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), target_tree.as_view(), 0.0).unwrap();
    let mut source_summaries = SourceNodeSummaries::<MockKernel<f64>>::new(source_tree.as_view());
    let mut target_summaries = TargetNodeSummaries::<MockKernel<f64>>::new(target_tree.as_view());
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
        update_target_summaries_into(
            &kernel,
            target_tree.as_view(),
            &targets,
            &mut target_summaries.node_summaries
        ),
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
            target_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries.node_summaries,
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
fn dipole_moment_kernel_theta_zero_matches_dense() {
    let kernel = DipoleMomentKernel::<f64>::new();
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
    let target_tree = ClusterTree::build(&targets, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), target_tree.as_view(), 0.0).unwrap();
    let mut source_summaries =
        SourceNodeSummaries::<DipoleMomentKernel<f64>>::new(source_tree.as_view());
    let mut target_summaries =
        TargetNodeSummaries::<DipoleMomentKernel<f64>>::new(target_tree.as_view());

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
        update_target_summaries_into(
            &kernel,
            target_tree.as_view(),
            &targets,
            &mut target_summaries.node_summaries,
        ),
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
            target_tree.as_view(),
            &source_summaries.node_summaries,
            &target_summaries.node_summaries,
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

#[test]
fn dipole_multipole_far_summary_improves_over_single_moment() {
    let moment_kernel = DipoleMomentKernel::<f64>::new();
    let multipole_kernel = DipoleMultipoleKernel::<f64>::new();
    let sources = [
        DipoleSource {
            position: [-1.0, 0.0, 0.0],
            outer_radius: 0.0,
        },
        DipoleSource {
            position: [1.0, 0.0, 0.0],
            outer_radius: 0.0,
        },
    ];
    let targets = [DipoleTarget {
        position: [20.0, 3.0, 1.0],
    }];
    let moments = [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]];

    let source_tree = ClusterTree::build(&sources, 2).unwrap();
    let target_tree = ClusterTree::build(&targets, 1).unwrap();
    let plan =
        DualInteractionPlan::build(source_tree.as_view(), target_tree.as_view(), 1.0).unwrap();
    assert_eq!(plan.far_target_node_ids.len(), 1);
    assert!(plan.near_target_ids.is_empty());

    let mut moment_source_summaries =
        SourceNodeSummaries::<DipoleMomentKernel<f64>>::new(source_tree.as_view());
    let mut moment_target_summaries =
        TargetNodeSummaries::<DipoleMomentKernel<f64>>::new(target_tree.as_view());
    let mut multipole_source_summaries =
        SourceNodeSummaries::<DipoleMultipoleKernel<f64>>::new(source_tree.as_view());
    let mut multipole_target_summaries =
        TargetNodeSummaries::<DipoleMultipoleKernel<f64>>::new(target_tree.as_view());

    assert_eq!(
        update_source_summaries_into(
            &moment_kernel,
            source_tree.as_view(),
            &sources,
            &moments,
            &mut moment_source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_target_summaries_into(
            &moment_kernel,
            target_tree.as_view(),
            &targets,
            &mut moment_target_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_source_summaries_into(
            &multipole_kernel,
            source_tree.as_view(),
            &sources,
            &moments,
            &mut multipole_source_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        update_target_summaries_into(
            &multipole_kernel,
            target_tree.as_view(),
            &targets,
            &mut multipole_target_summaries.node_summaries,
        ),
        DualTreeError::Ok
    );

    let mut scratch_value = [[0.0; 3]];
    let mut scratch = EvaluationScratch {
        contribution: &mut scratch_value,
    };
    let mut moment_out = [[0.0; 3]; 1];
    let mut multipole_out = [[0.0; 3]; 1];
    let mut dense = [[0.0; 3]; 1];

    assert_eq!(
        evaluate_into(
            &moment_kernel,
            plan.as_view(),
            source_tree.as_view(),
            target_tree.as_view(),
            &moment_source_summaries.node_summaries,
            &moment_target_summaries.node_summaries,
            &sources,
            &targets,
            &moments,
            &mut moment_out,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        evaluate_into(
            &multipole_kernel,
            plan.as_view(),
            source_tree.as_view(),
            target_tree.as_view(),
            &multipole_source_summaries.node_summaries,
            &multipole_target_summaries.node_summaries,
            &sources,
            &targets,
            &moments,
            &mut multipole_out,
            &mut scratch,
        ),
        DualTreeError::Ok
    );
    assert_eq!(
        dense_direct_evaluate_into(
            &moment_kernel,
            &sources,
            &targets,
            &moments,
            &mut dense,
            &mut scratch,
        ),
        DualTreeError::Ok
    );

    let moment_err = vec_norm(sub_vec3(moment_out[0], dense[0]));
    let multipole_err = vec_norm(sub_vec3(multipole_out[0], dense[0]));
    assert!(multipole_err < moment_err);
}

fn sub_vec3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec_norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
