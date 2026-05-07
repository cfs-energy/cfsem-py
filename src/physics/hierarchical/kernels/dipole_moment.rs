use core::marker::PhantomData;

use super::dipole::{
    DipoleSource, DipoleTarget, DipoleTargetSummary, add3_in_place, combine_target, dipole_field,
    summarize_centroid, summarize_target_leaf,
};
use crate::physics::hierarchical::{DualTreeError, DualTreeKernel, DualTreeScalar};

/// Source summary containing only total dipole moment at a representative centroid.
#[derive(Clone, Copy, Debug, Default)]
pub struct DipoleMomentSummary<T: DualTreeScalar> {
    pub centroid: [T; 3],
    pub moment: [T; 3],
    pub count: T,
}

/// Dipole Barnes-Hut kernel that summarizes source clusters as one total dipole moment.
#[derive(Clone, Copy, Debug, Default)]
pub struct DipoleMomentKernel<T: DualTreeScalar> {
    marker: PhantomData<T>,
}

impl<T: DualTreeScalar> DipoleMomentKernel<T> {
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<T: DualTreeScalar> DualTreeKernel for DipoleMomentKernel<T> {
    type Scalar = T;
    type SourceGeometry = DipoleSource<T>;
    type TargetGeometry = DipoleTarget<T>;
    type SourceMoment = [T; 3];
    type SourceSummary = DipoleMomentSummary<T>;
    type TargetSummary = DipoleTargetSummary<T>;
    type Output = [T; 3];

    fn summarize_leaf_sources(
        &self,
        source_ids: &[u32],
        sources: &[Self::SourceGeometry],
        moments: &[Self::SourceMoment],
        out: &mut Self::SourceSummary,
    ) -> DualTreeError {
        *out = DipoleMomentSummary::default();
        summarize_centroid(source_ids, sources, &mut out.centroid, &mut out.count);
        for i in 0..source_ids.len() {
            let source_id = source_ids[i] as usize;
            add3_in_place(&mut out.moment, moments[source_id]);
        }
        DualTreeError::Ok
    }

    fn combine_source_summaries(
        &self,
        children: &[Self::SourceSummary],
        _child_ids: &[u32],
        out: &mut Self::SourceSummary,
    ) -> DualTreeError {
        *out = DipoleMomentSummary::default();
        combine_source_centroid_moment(
            children,
            &mut out.centroid,
            &mut out.moment,
            &mut out.count,
        );
        DualTreeError::Ok
    }

    fn summarize_leaf_targets(
        &self,
        target_ids: &[u32],
        targets: &[Self::TargetGeometry],
        out: &mut Self::TargetSummary,
    ) -> DualTreeError {
        summarize_target_leaf(target_ids, targets, out)
    }

    fn combine_target_summaries(
        &self,
        children: &[Self::TargetSummary],
        _child_ids: &[u32],
        out: &mut Self::TargetSummary,
    ) -> DualTreeError {
        combine_target(children, out)
    }

    fn eval_exact(
        &self,
        target: &Self::TargetGeometry,
        source: &Self::SourceGeometry,
        moment: &Self::SourceMoment,
        out: &mut Self::Output,
    ) -> DualTreeError {
        dipole_field(
            target.position,
            source.position,
            *moment,
            source.outer_radius,
            out,
        )
    }

    fn eval_far(
        &self,
        target: &Self::TargetSummary,
        source: &Self::SourceSummary,
        out: &mut Self::Output,
    ) -> DualTreeError {
        dipole_field(
            target.centroid,
            source.centroid,
            source.moment,
            T::ZERO,
            out,
        )
    }

    fn zero_output(&self, out: &mut Self::Output) {
        *out = [T::ZERO; 3];
    }

    fn accumulate(&self, out: &mut Self::Output, contribution: &Self::Output) {
        add3_in_place(out, *contribution);
    }
}

fn combine_source_centroid_moment<T: DualTreeScalar>(
    children: &[DipoleMomentSummary<T>],
    centroid: &mut [T; 3],
    moment: &mut [T; 3],
    count: &mut T,
) {
    *centroid = [T::ZERO; 3];
    *moment = [T::ZERO; 3];
    *count = T::ZERO;
    for i in 0..children.len() {
        *count = *count + children[i].count;
        add3_in_place(moment, children[i].moment);
        for axis in 0..3 {
            centroid[axis] = centroid[axis] + children[i].centroid[axis] * children[i].count;
        }
    }
    if *count > T::ZERO {
        for axis in 0..3 {
            centroid[axis] = centroid[axis] / *count;
        }
    }
}
