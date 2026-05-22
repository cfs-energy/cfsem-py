use core::marker::PhantomData;

use super::dipole::{
    DipoleSource, DipoleSummary, DipoleTarget, DipoleTargetSummary,
    combine_dipole_source_summaries, dipole_vector_potential, summarize_dipole_leaf_sources,
    summarize_target_leaf,
};
use crate::math::add3_in_place;
use crate::physics::hierarchical::{
    HierarchicalError, HierarchicalKernel, Scalar, SourceCollection, SourceMomentCollection,
};

/// Source summary for dipole vector-potential clusters.
pub type DipoleVectorPotentialSummary<T> = DipoleSummary<T>;

/// Dipole vector-potential Barnes-Hut kernel.
///
/// This is not a full multipole treatment. Far evaluation uses one total dipole
/// at the moment-weighted source centroid.
#[derive(Clone, Copy, Debug, Default)]
pub struct DipoleVectorPotentialKernel<T: Scalar> {
    marker: PhantomData<T>,
}

impl<T: Scalar> DipoleVectorPotentialKernel<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<T: Scalar> HierarchicalKernel for DipoleVectorPotentialKernel<T> {
    type Scalar = T;
    type SourceGeometry = DipoleSource<T>;
    type TargetGeometry = DipoleTarget<T>;
    type SourceMoment = [T; 3];
    type SourceSummary = DipoleVectorPotentialSummary<T>;
    type TargetSummary = DipoleTargetSummary<T>;
    type Output = [T; 3];

    #[inline]
    fn summarize_leaf_sources<S, M>(
        &self,
        source_ids: &[u32],
        sources: S,
        moments: M,
        out: &mut Self::SourceSummary,
    ) -> HierarchicalError
    where
        S: SourceCollection<Self>,
        M: SourceMomentCollection<Self>,
    {
        summarize_dipole_leaf_sources(
            source_ids,
            sources,
            |source_id| moments.moment(source_id),
            out,
        )
    }

    #[inline]
    fn combine_source_summaries(
        &self,
        children: &[Self::SourceSummary],
        _child_ids: &[u32],
        out: &mut Self::SourceSummary,
    ) -> HierarchicalError {
        combine_dipole_source_summaries(children, out)
    }

    #[inline]
    fn summarize_leaf_targets(
        &self,
        target_ids: &[u32],
        targets: &[Self::TargetGeometry],
        out: &mut Self::TargetSummary,
    ) -> HierarchicalError {
        summarize_target_leaf(target_ids, targets, out)
    }

    #[inline]
    fn eval_exact(
        &self,
        target: &Self::TargetGeometry,
        source: &Self::SourceGeometry,
        moment: &Self::SourceMoment,
        out: &mut Self::Output,
    ) {
        dipole_vector_potential(
            target.position,
            source.position,
            *moment,
            source.outer_radius,
            out,
        )
    }

    #[inline]
    fn eval_far(
        &self,
        target: &Self::TargetSummary,
        source: &Self::SourceSummary,
        out: &mut Self::Output,
    ) {
        dipole_vector_potential(
            target.centroid,
            source.centroid,
            source.moment,
            T::ZERO,
            out,
        );
    }

    #[inline]
    fn zero_output(&self, out: &mut Self::Output) {
        *out = [T::ZERO; 3];
    }

    #[inline]
    fn accumulate(&self, out: &mut Self::Output, contribution: &Self::Output) {
        add3_in_place(out, *contribution);
    }
}
