use core::marker::PhantomData;

use super::dipole::{
    DipoleTarget, DipoleTargetSummary, dipole_vector_potential, summarize_target_leaf,
};
use super::linear_filament::{
    LinearFilamentSource, LinearFilamentSummary, array_to_tuple,
    combine_linear_filament_source_summaries, half, linear_filament_accept_far,
    summarize_linear_filament_leaf_sources, tuple_to_array,
};
use crate::math::{add3_in_place, scale3, sub3};
use crate::physics::hierarchical::{
    Aabb, HierarchicalError, HierarchicalKernel, Scalar, SourceCollection, SourceMomentCollection,
};
use crate::physics::linear_filament::vector_potential_linear_filament_scalar;
use crate::physics::point_source::segment::vector_potential_point_segment_scalar;

/// Source summary for finite linear filament vector-potential clusters.
pub type LinearFilamentVectorPotentialSummary<T> = LinearFilamentSummary<T>;

/// Linear filament vector-potential Barnes-Hut kernel.
///
/// Tree construction still uses each finite source segment's full AABB, so the
/// near/far plan is based on the full span of the included filaments. Once a
/// source cluster is accepted as far, the source term is represented as a point
/// current element with a current-element-weighted origin, unit direction, and
/// `I*dL` magnitude. A magnetic dipole term with its own weighted origin is
/// also included so closed or locally cancelling current paths can still
/// contribute to the far vector potential.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearFilamentVectorPotentialKernel<T: Scalar> {
    marker: PhantomData<T>,
}

impl<T: Scalar> LinearFilamentVectorPotentialKernel<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<T: Scalar> HierarchicalKernel for LinearFilamentVectorPotentialKernel<T> {
    type Scalar = T;
    type SourceGeometry = LinearFilamentSource<T>;
    type TargetGeometry = DipoleTarget<T>;
    type SourceMoment = T;
    type SourceSummary = LinearFilamentVectorPotentialSummary<T>;
    type TargetSummary = DipoleTargetSummary<T>;
    type Output = [T; 3];

    #[inline]
    fn summarize_leaf_sources<S, M>(
        &self,
        source_ids: &[u32],
        sources: S,
        currents: M,
        out: &mut Self::SourceSummary,
    ) -> HierarchicalError
    where
        S: SourceCollection<Self>,
        M: SourceMomentCollection<Self>,
    {
        summarize_linear_filament_leaf_sources(
            source_ids,
            |source_id| sources.source(source_id),
            |source_id| currents.moment(source_id),
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
        combine_linear_filament_source_summaries(children, out)
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
        current: &Self::SourceMoment,
        out: &mut Self::Output,
    ) {
        *out = tuple_to_array(vector_potential_linear_filament_scalar(
            (
                array_to_tuple(source.start),
                array_to_tuple(source.end),
                *current,
            ),
            source.wire_radius,
            array_to_tuple(target.position),
        ));
    }

    #[inline]
    fn eval_far(
        &self,
        target: &Self::TargetSummary,
        source: &Self::SourceSummary,
        out: &mut Self::Output,
    ) {
        *out = [T::ZERO; 3];
        if source.weight <= T::ZERO {
            return;
        }

        if source.magnitude > T::ZERO {
            *out = point_segment_source_term(
                source.origin,
                source.direction,
                source.magnitude,
                target.centroid,
            );
        }

        let mut dipole_out = [T::ZERO; 3];
        dipole_vector_potential(
            target.centroid,
            source.dipole_origin,
            source.dipole_moment,
            T::ZERO,
            &mut dipole_out,
        );
        add3_in_place(out, dipole_out);
    }

    #[inline]
    fn accept_far(
        &self,
        target_aabb: Aabb<Self::Scalar>,
        source_aabb: Aabb<Self::Scalar>,
        source: &Self::SourceSummary,
        theta: Self::Scalar,
    ) -> bool {
        linear_filament_accept_far(target_aabb, source_aabb, source, theta)
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

#[inline]
fn point_segment_source_term<T: Scalar>(
    origin: [T; 3],
    direction: [T; 3],
    magnitude: T,
    target: [T; 3],
) -> [T; 3] {
    let half_direction = scale3(direction, half::<T>());
    let start = sub3(origin, half_direction);
    let end = [
        origin[0] + half_direction[0],
        origin[1] + half_direction[1],
        origin[2] + half_direction[2],
    ];
    tuple_to_array(vector_potential_point_segment_scalar(
        (array_to_tuple(start), array_to_tuple(end), magnitude),
        array_to_tuple(target),
    ))
}
