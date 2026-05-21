use core::marker::PhantomData;

use super::dipole::{
    DipoleTarget, DipoleTargetSummary, dipole_vector_potential, summarize_target_leaf,
};
use super::linear_filament_flux_density::LinearFilamentSource;
use crate::math::{add3_in_place, cross3, norm3, scale3, sub3};
use crate::physics::hierarchical::{
    Aabb, BoundedGeometry, HierarchicalError, HierarchicalKernel, Scalar, SourceCollection,
    SourceMomentCollection, geometric_accept_far,
};
use crate::physics::linear_filament::vector_potential_linear_filament_scalar;
use crate::physics::point_source::segment::vector_potential_point_segment_scalar;

/// Lower closure-ratio bound for source summaries that can be represented as loops.
const CLOSED_SUMMARY_CLOSURE_RATIO_MAX: f64 = 0.2;
/// Upper closure-ratio bound for source summaries that can be represented as segments.
const OPEN_SUMMARY_CLOSURE_RATIO_MIN: f64 = 0.8;
/// Linear filament opening-angle scale factor for stricter geometric acceptance.
const LINEAR_FILAMENT_THETA_SCALE: f64 = 0.5;

/// Source summary for finite linear filament vector-potential clusters.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearFilamentVectorPotentialSummary<T: Scalar> {
    /// `|I*dL|`-weighted origin for the net current-element source term.
    pub origin: [T; 3],
    /// Unit direction of the net current element after finalization.
    pub direction: [T; 3],
    /// Magnitude of the net current element `|sum(I*dL)|`.
    pub magnitude: T,
    /// Origin used for the residual magnetic dipole correction.
    pub dipole_origin: [T; 3],
    /// Magnetic dipole moment translated to `dipole_origin`.
    pub dipole_moment: [T; 3],
    /// Total current-element weight `sum(|I*dL|)`.
    pub weight: T,
}

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
        *out = LinearFilamentVectorPotentialSummary::default();
        for i in 0..source_ids.len() {
            let source_id = source_ids[i] as usize;
            let source = sources.source(source_id);
            add_source_to_summary(&source, currents.moment(source_id), out);
        }
        finalize_leaf_source_summary(out);
        HierarchicalError::Ok
    }

    #[inline]
    fn combine_source_summaries(
        &self,
        children: &[Self::SourceSummary],
        _child_ids: &[u32],
        out: &mut Self::SourceSummary,
    ) -> HierarchicalError {
        *out = LinearFilamentVectorPotentialSummary::default();
        for i in 0..children.len() {
            out.weight = out.weight + children[i].weight;
            add3_in_place(
                &mut out.origin,
                scale3(children[i].origin, children[i].weight),
            );
            add3_in_place(
                &mut out.direction,
                scale3(children[i].direction, children[i].magnitude),
            );
            add3_in_place(
                &mut out.dipole_origin,
                scale3(children[i].dipole_origin, children[i].weight),
            );
        }
        if out.weight > T::ZERO {
            out.origin = scale3(out.origin, T::ONE / out.weight);
            out.dipole_origin = scale3(out.dipole_origin, T::ONE / out.weight);
        }

        for i in 0..children.len() {
            let child_current = scale3(children[i].direction, children[i].magnitude);
            add3_in_place(&mut out.dipole_moment, children[i].dipole_moment);
            add3_in_place(
                &mut out.dipole_moment,
                scale3(
                    cross3(
                        sub3(children[i].dipole_origin, out.dipole_origin),
                        child_current,
                    ),
                    half::<T>(),
                ),
            );
        }
        finalize_current_element(out);
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
        if source.weight > T::ZERO {
            let closure_ratio = source.magnitude / source.weight;
            if closure_ratio >= T::from_f64(CLOSED_SUMMARY_CLOSURE_RATIO_MAX)
                && closure_ratio <= T::from_f64(OPEN_SUMMARY_CLOSURE_RATIO_MIN)
            {
                return false;
            }
        }
        geometric_accept_far(
            target_aabb,
            source_aabb,
            theta * T::from_f64(LINEAR_FILAMENT_THETA_SCALE),
        )
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
fn add_source_to_summary<T: Scalar>(
    source: &LinearFilamentSource<T>,
    current: T,
    out: &mut LinearFilamentVectorPotentialSummary<T>,
) {
    let dl = sub3(source.end, source.start);
    let length = norm3(dl);
    if length <= T::ZERO {
        return;
    }

    let current_element = scale3(dl, current);
    let current_element_weight = norm3(current_element);
    if current_element_weight <= T::ZERO {
        return;
    }

    out.weight = out.weight + current_element_weight;
    add3_in_place(
        &mut out.origin,
        scale3(source.representative_point(), current_element_weight),
    );
    add3_in_place(
        &mut out.dipole_origin,
        scale3(source.representative_point(), current_element_weight),
    );
    add3_in_place(&mut out.direction, current_element);
    add3_in_place(
        &mut out.dipole_moment,
        scale3(
            cross3(source.representative_point(), current_element),
            half::<T>(),
        ),
    );
}

#[inline]
fn finalize_leaf_source_summary<T: Scalar>(summary: &mut LinearFilamentVectorPotentialSummary<T>) {
    if summary.weight > T::ZERO {
        summary.origin = scale3(summary.origin, T::ONE / summary.weight);
        summary.dipole_origin = scale3(summary.dipole_origin, T::ONE / summary.weight);
    }

    add3_in_place(
        &mut summary.dipole_moment,
        scale3(
            cross3(summary.dipole_origin, summary.direction),
            T::ZERO - half::<T>(),
        ),
    );
    finalize_current_element(summary);
}

#[inline]
fn finalize_current_element<T: Scalar>(summary: &mut LinearFilamentVectorPotentialSummary<T>) {
    summary.magnitude = norm3(summary.direction);
    if summary.magnitude > T::ZERO {
        summary.direction = scale3(summary.direction, T::ONE / summary.magnitude);
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

#[inline]
fn half<T: Scalar>() -> T {
    T::from_f64(0.5)
}

#[inline]
fn array_to_tuple<T: Scalar>(value: [T; 3]) -> (T, T, T) {
    (value[0], value[1], value[2])
}

#[inline]
fn tuple_to_array<T: Scalar>(value: (T, T, T)) -> [T; 3] {
    [value.0, value.1, value.2]
}
