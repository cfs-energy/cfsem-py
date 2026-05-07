use core::marker::PhantomData;

use super::dipole::{
    DipoleSource, DipoleTarget, DipoleTargetSummary, add_matrix_in_place, add_outer_in_place,
    combine_target, dipole_vector_potential, dipole_vector_potential_derivative_component,
    summarize_centroid, summarize_target_leaf,
};
use crate::MU0_OVER_4PI;
use crate::math::{add3_in_place, sub3};
use crate::physics::hierarchical::{DualTreeError, DualTreeKernel, DualTreeScalar};

/// Source summary for dipole vector-potential clusters.
///
/// `first_moment[a][b] = sum_i (x_i[a] - centroid[a]) * moment_i[b]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DipoleVectorPotentialSummary<T: DualTreeScalar> {
    pub centroid: [T; 3],
    pub moment: [T; 3],
    pub first_moment: [[T; 3]; 3],
    pub count: T,
}

/// Dipole vector-potential Barnes-Hut kernel.
///
/// This is not a full multipole treatment. Far evaluation starts with one total
/// dipole at the source centroid and adds the first spatial moment correction
/// from the source cluster.
#[derive(Clone, Copy, Debug, Default)]
pub struct DipoleVectorPotentialKernel<T: DualTreeScalar> {
    marker: PhantomData<T>,
}

impl<T: DualTreeScalar> DipoleVectorPotentialKernel<T> {
    #[inline]
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

    #[inline]
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
            let delta = sub3(sources[source_id].position, out.centroid);
            add3_in_place(&mut out.moment, moments[source_id]);
            add_outer_in_place(&mut out.first_moment, delta, moments[source_id]);
        }
        DualTreeError::Ok
    }

    #[inline]
    fn combine_source_summaries(
        &self,
        children: &[Self::SourceSummary],
        _child_ids: &[u32],
        out: &mut Self::SourceSummary,
    ) -> DualTreeError {
        *out = DipoleVectorPotentialSummary::default();
        for i in 0..children.len() {
            out.count = out.count + children[i].count;
            for axis in 0..3 {
                out.centroid[axis] =
                    children[i].centroid[axis].mul_add(children[i].count, out.centroid[axis]);
            }
        }
        if out.count > T::ZERO {
            for axis in 0..3 {
                out.centroid[axis] = out.centroid[axis] / out.count;
            }
        }

        for i in 0..children.len() {
            add3_in_place(&mut out.moment, children[i].moment);
            add_matrix_in_place(&mut out.first_moment, children[i].first_moment);
            let shift = sub3(children[i].centroid, out.centroid);
            add_outer_in_place(&mut out.first_moment, shift, children[i].moment);
        }

        DualTreeError::Ok
    }

    #[inline]
    fn summarize_leaf_targets(
        &self,
        target_ids: &[u32],
        targets: &[Self::TargetGeometry],
        out: &mut Self::TargetSummary,
    ) -> DualTreeError {
        summarize_target_leaf(target_ids, targets, out)
    }

    #[inline]
    fn combine_target_summaries(
        &self,
        children: &[Self::TargetSummary],
        _child_ids: &[u32],
        out: &mut Self::TargetSummary,
    ) -> DualTreeError {
        combine_target(children, out)
    }

    #[inline]
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

    #[inline]
    fn eval_far(
        &self,
        target: &Self::TargetSummary,
        source: &Self::SourceSummary,
        out: &mut Self::Output,
    ) -> DualTreeError {
        let err = dipole_vector_potential(
            target.centroid,
            source.centroid,
            source.moment,
            T::ZERO,
            out,
        );
        if err != DualTreeError::Ok {
            return err;
        }

        let r = sub3(target.centroid, source.centroid);
        let c = T::from_f64(MU0_OVER_4PI);
        for source_axis in 0..3 {
            for moment_axis in 0..3 {
                let coeff = source.first_moment[source_axis][moment_axis];
                if coeff == T::ZERO {
                    continue;
                }
                let deriv =
                    dipole_vector_potential_derivative_component(r, moment_axis, source_axis, c);
                for out_axis in 0..3 {
                    out[out_axis] = (T::ZERO - coeff).mul_add(deriv[out_axis], out[out_axis]);
                }
            }
        }

        DualTreeError::Ok
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
