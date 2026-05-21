use core::marker::PhantomData;

use super::boundary_element::{
    BoundaryElementSummary, BoundaryElementTriangle, combine_source_summaries, has_current,
    summarize_leaf_sources,
};
use super::dipole::{
    DipoleTarget, DipoleTargetSummary, dipole_vector_potential, summarize_target_leaf,
};
use crate::math::add3_in_place;
use crate::physics::boundary_element::{QuadratureKind, vector_potential_triangle};
use crate::physics::hierarchical::{HierarchicalError, HierarchicalKernel, Scalar};
use crate::physics::point_source::current_element::vector_potential_current_element_scalar;

/// Boundary-element vector-potential Barnes-Hut kernel.
///
/// Exact evaluation uses the upstream triangle quadrature. Far evaluation
/// summarizes accepted source clusters as one point current element plus a
/// shifted magnetic-dipole term so locally closed current paths retain their
/// leading loop behavior.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryElementVectorPotentialKernel<T: Scalar> {
    quad_kind: QuadratureKind,
    marker: PhantomData<T>,
}

impl<T: Scalar> BoundaryElementVectorPotentialKernel<T> {
    #[inline]
    pub fn new(quad_kind: QuadratureKind) -> Self {
        Self {
            quad_kind,
            marker: PhantomData,
        }
    }
}

impl<T: Scalar> Default for BoundaryElementVectorPotentialKernel<T> {
    #[inline]
    fn default() -> Self {
        Self::new(QuadratureKind::Dunavant3)
    }
}

impl<T: Scalar> HierarchicalKernel for BoundaryElementVectorPotentialKernel<T> {
    type Scalar = T;
    type SourceGeometry = BoundaryElementTriangle<T>;
    type TargetGeometry = DipoleTarget<T>;
    type SourceMoment = [T; 3];
    type SourceSummary = BoundaryElementSummary<T>;
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
        summarize_leaf_sources(source_ids, sources, moments, out)
    }

    #[inline]
    fn combine_source_summaries(
        &self,
        children: &[Self::SourceSummary],
        _child_ids: &[u32],
        out: &mut Self::SourceSummary,
    ) -> HierarchicalError {
        combine_source_summaries(children, out)
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
        *out = vector_potential_triangle(
            source.n0,
            source.n1,
            source.n2,
            *moment,
            target.position,
            self.quad_kind,
        );
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

        if has_current(source) {
            *out = vector_potential_current_element_scalar(
                source.origin,
                source.current_element,
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
    fn zero_output(&self, out: &mut Self::Output) {
        *out = [T::ZERO; 3];
    }

    #[inline]
    fn accumulate(&self, out: &mut Self::Output, contribution: &Self::Output) {
        add3_in_place(out, *contribution);
    }
}
