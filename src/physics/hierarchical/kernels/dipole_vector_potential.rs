use core::marker::PhantomData;

use super::dipole::{
    DipoleSource, DipoleTarget, DipoleTargetSummary, dipole_vector_potential,
    summarize_target_leaf, summarize_weighted_source_centroid,
};
use crate::math::add3_in_place;
use crate::physics::hierarchical::{HierarchicalError, HierarchicalKernel, Scalar};

/// Source summary for dipole vector-potential clusters.
#[derive(Clone, Copy, Debug, Default)]
pub struct DipoleVectorPotentialSummary<T: Scalar> {
    pub centroid: [T; 3],
    pub moment: [T; 3],
    pub weight: T,
}

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
    fn summarize_leaf_sources(
        &self,
        source_ids: &[u32],
        sources: &[Self::SourceGeometry],
        moments: &[Self::SourceMoment],
        out: &mut Self::SourceSummary,
    ) -> HierarchicalError {
        *out = DipoleVectorPotentialSummary::default();
        summarize_weighted_source_centroid(
            source_ids,
            sources,
            moments,
            &mut out.centroid,
            &mut out.weight,
        );
        for i in 0..source_ids.len() {
            let source_id = source_ids[i] as usize;
            add3_in_place(&mut out.moment, moments[source_id]);
        }
        HierarchicalError::Ok
    }

    #[inline]
    fn combine_source_summaries(
        &self,
        children: &[Self::SourceSummary],
        _child_ids: &[u32],
        out: &mut Self::SourceSummary,
    ) -> HierarchicalError {
        *out = DipoleVectorPotentialSummary::default();
        for i in 0..children.len() {
            out.weight = out.weight + children[i].weight;
            for axis in 0..3 {
                out.centroid[axis] =
                    children[i].centroid[axis].mul_add(children[i].weight, out.centroid[axis]);
            }
        }
        if out.weight > T::ZERO {
            for axis in 0..3 {
                out.centroid[axis] = out.centroid[axis] / out.weight;
            }
        }

        for i in 0..children.len() {
            add3_in_place(&mut out.moment, children[i].moment);
        }

        HierarchicalError::Ok
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
