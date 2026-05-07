use core::marker::PhantomData;

use super::dipole::{
    DipoleSource, DipoleTarget, DipoleTargetSummary, add3_in_place, combine_target,
    dipole_vector_potential, summarize_centroid, summarize_target_leaf,
};
use crate::physics::hierarchical::{DualTreeError, DualTreeKernel, DualTreeScalar};

/// Source summary containing only total dipole moment at a representative centroid.
#[derive(Clone, Copy, Debug, Default)]
pub struct DipoleVectorPotentialSummary<T: DualTreeScalar> {
    pub centroid: [T; 3],
    pub moment: [T; 3],
    pub count: T,
}

/// Dipole vector-potential Barnes-Hut kernel that summarizes source clusters as one total dipole.
#[derive(Clone, Copy, Debug, Default)]
pub struct DipoleVectorPotentialKernel<T: DualTreeScalar> {
    marker: PhantomData<T>,
}

impl<T: DualTreeScalar> DipoleVectorPotentialKernel<T> {
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<T: DualTreeScalar> DualTreeKernel for DipoleVectorPotentialKernel<T> {
    type Scalar = T;
    type SourceGeometry = DipoleSource<T>;
    type TargetGeometry = DipoleTarget<T>;
    type SourceMoment = [T; 3];
    type SourceSummary = DipoleVectorPotentialSummary<T>;
    type TargetSummary = DipoleTargetSummary<T>;
    type Output = [T; 3];

    fn summarize_leaf_sources(
        &self,
        source_ids: &[u32],
        sources: &[Self::SourceGeometry],
        moments: &[Self::SourceMoment],
        out: &mut Self::SourceSummary,
    ) -> DualTreeError {
        *out = DipoleVectorPotentialSummary::default();
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
        *out = DipoleVectorPotentialSummary::default();
        for i in 0..children.len() {
            out.count = out.count + children[i].count;
            add3_in_place(&mut out.moment, children[i].moment);
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
        dipole_vector_potential(
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
        dipole_vector_potential(
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
