use core::marker::PhantomData;

use super::dipole::{
    DipoleTarget, DipoleTargetSummary, combine_target, dipole_field, summarize_target_leaf,
};
use crate::math::{add3_in_place, cross3, norm3, scale3, sub3};
use crate::physics::hierarchical::{
    Aabb, BoundedGeometry, DualTreeError, DualTreeKernel, DualTreeScalar,
};
use crate::physics::linear_filament::flux_density_linear_filament_scalar;
use crate::physics::point_source::segment::flux_density_point_segment_scalar;

/// Finite linear filament source geometry.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearFilamentSource<T: DualTreeScalar> {
    pub start: [T; 3],
    pub end: [T; 3],
    pub wire_radius: T,
}

impl<T: DualTreeScalar> BoundedGeometry for LinearFilamentSource<T> {
    type Scalar = T;

    #[inline]
    fn aabb(&self) -> Aabb<Self::Scalar> {
        let r = if self.wire_radius > T::ZERO {
            self.wire_radius
        } else {
            T::ZERO
        };
        let mut min = [T::ZERO; 3];
        let mut max = [T::ZERO; 3];
        for axis in 0..3 {
            if self.start[axis] < self.end[axis] {
                min[axis] = self.start[axis] - r;
                max[axis] = self.end[axis] + r;
            } else {
                min[axis] = self.end[axis] - r;
                max[axis] = self.start[axis] + r;
            }
        }
        Aabb { min, max }
    }

    #[inline]
    fn representative_point(&self) -> [Self::Scalar; 3] {
        let half = T::from_f64(0.5);
        [
            half.mul_add(self.start[0] + self.end[0], T::ZERO),
            half.mul_add(self.start[1] + self.end[1], T::ZERO),
            half.mul_add(self.start[2] + self.end[2], T::ZERO),
        ]
    }
}

/// Source summary for finite linear filament flux-density clusters.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearFilamentFluxDensitySummary<T: DualTreeScalar> {
    pub origin: [T; 3],
    pub direction: [T; 3],
    pub magnitude: T,
    pub dipole_origin: [T; 3],
    pub dipole_moment: [T; 3],
    pub weight: T,
}

/// Linear filament flux-density Barnes-Hut kernel.
///
/// Tree construction still uses each finite source segment's full AABB, so the
/// near/far plan is based on the full span of the included filaments. Once a
/// source cluster is accepted as far, the source term is represented as a point
/// current element with a length-weighted origin, unit direction, and
/// `I*dL` magnitude. A magnetic dipole term with its own weighted origin is
/// also included so closed or locally cancelling current paths can still
/// contribute to the far field.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearFilamentFluxDensityKernel<T: DualTreeScalar> {
    marker: PhantomData<T>,
}

impl<T: DualTreeScalar> LinearFilamentFluxDensityKernel<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<T: DualTreeScalar> DualTreeKernel for LinearFilamentFluxDensityKernel<T> {
    type Scalar = T;
    type SourceGeometry = LinearFilamentSource<T>;
    type TargetGeometry = DipoleTarget<T>;
    type SourceMoment = T;
    type SourceSummary = LinearFilamentFluxDensitySummary<T>;
    type TargetSummary = DipoleTargetSummary<T>;
    type Output = [T; 3];

    #[inline]
    fn summarize_leaf_sources(
        &self,
        source_ids: &[u32],
        sources: &[Self::SourceGeometry],
        currents: &[Self::SourceMoment],
        out: &mut Self::SourceSummary,
    ) -> DualTreeError {
        *out = LinearFilamentFluxDensitySummary::default();
        for i in 0..source_ids.len() {
            let source_id = source_ids[i] as usize;
            add_source_to_summary(&sources[source_id], currents[source_id], out);
        }
        finalize_leaf_source_summary(out);
        DualTreeError::Ok
    }

    #[inline]
    fn combine_source_summaries(
        &self,
        children: &[Self::SourceSummary],
        _child_ids: &[u32],
        out: &mut Self::SourceSummary,
    ) -> DualTreeError {
        *out = LinearFilamentFluxDensitySummary::default();
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
        current: &Self::SourceMoment,
        out: &mut Self::Output,
    ) -> DualTreeError {
        *out = tuple_to_array(flux_density_linear_filament_scalar(
            (
                array_to_tuple(source.start),
                array_to_tuple(source.end),
                *current,
            ),
            source.wire_radius,
            array_to_tuple(target.position),
        ));
        DualTreeError::Ok
    }

    #[inline]
    fn eval_far(
        &self,
        target: &Self::TargetSummary,
        source: &Self::SourceSummary,
        out: &mut Self::Output,
    ) -> DualTreeError {
        *out = [T::ZERO; 3];
        if source.weight <= T::ZERO {
            return DualTreeError::Ok;
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
        let err = dipole_field(
            target.centroid,
            source.dipole_origin,
            source.dipole_moment,
            T::ZERO,
            &mut dipole_out,
        );
        if err != DualTreeError::Ok {
            return err;
        }
        add3_in_place(out, dipole_out);

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

#[inline]
fn add_source_to_summary<T: DualTreeScalar>(
    source: &LinearFilamentSource<T>,
    current: T,
    out: &mut LinearFilamentFluxDensitySummary<T>,
) {
    let dl = sub3(source.end, source.start);
    let length = norm3(dl);
    if length <= T::ZERO {
        return;
    }

    out.weight = out.weight + length;
    add3_in_place(
        &mut out.origin,
        scale3(source.representative_point(), length),
    );
    add3_in_place(
        &mut out.dipole_origin,
        scale3(source.representative_point(), length),
    );
    let current_element = scale3(dl, current);
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
fn finalize_leaf_source_summary<T: DualTreeScalar>(
    summary: &mut LinearFilamentFluxDensitySummary<T>,
) {
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
fn finalize_current_element<T: DualTreeScalar>(summary: &mut LinearFilamentFluxDensitySummary<T>) {
    summary.magnitude = norm3(summary.direction);
    if summary.magnitude > T::ZERO {
        summary.direction = scale3(summary.direction, T::ONE / summary.magnitude);
    }
}

#[inline]
fn point_segment_source_term<T: DualTreeScalar>(
    origin: [T; 3],
    direction: [T; 3],
    magnitude: T,
    target: [T; 3],
) -> [T; 3] {
    let half = T::from_f64(0.5);
    let half_direction = scale3(direction, half);
    let start = sub3(origin, half_direction);
    let end = [
        origin[0] + half_direction[0],
        origin[1] + half_direction[1],
        origin[2] + half_direction[2],
    ];
    tuple_to_array(flux_density_point_segment_scalar(
        (array_to_tuple(start), array_to_tuple(end), magnitude),
        array_to_tuple(target),
    ))
}

#[inline]
fn half<T: DualTreeScalar>() -> T {
    T::from_f64(0.5)
}

#[inline]
fn array_to_tuple<T: DualTreeScalar>(value: [T; 3]) -> (T, T, T) {
    (value[0], value[1], value[2])
}

#[inline]
fn tuple_to_array<T: DualTreeScalar>(value: (T, T, T)) -> [T; 3] {
    [value.0, value.1, value.2]
}
