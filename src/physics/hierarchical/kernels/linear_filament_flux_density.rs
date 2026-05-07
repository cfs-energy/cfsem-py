use core::marker::PhantomData;

use super::dipole::{
    DipoleTarget, DipoleTargetSummary, add3_in_place, combine_target, summarize_target_leaf,
};
use crate::physics::hierarchical::{
    Aabb, BoundedGeometry, DualTreeError, DualTreeKernel, DualTreeScalar,
};
use crate::physics::linear_filament::flux_density_linear_filament_scalar;

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
            (self.start[0] + self.end[0]) * half,
            (self.start[1] + self.end[1]) * half,
            (self.start[2] + self.end[2]) * half,
        ]
    }
}

/// Source summary for finite linear filament flux-density clusters.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearFilamentFluxDensitySummary<T: DualTreeScalar> {
    pub start_accum: [T; 3],
    pub end_accum: [T; 3],
    pub weight: T,
    pub current_element: [T; 3],
    pub wire_radius: T,
}

/// Linear filament flux-density Barnes-Hut kernel.
///
/// Far evaluation represents each accepted source cluster as one equivalent
/// finite filament segment. The equivalent segment uses length-weighted averaged
/// endpoints for finite extent and the exact net `I*dL` vector for direction and
/// current magnitude.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearFilamentFluxDensityKernel<T: DualTreeScalar> {
    marker: PhantomData<T>,
}

impl<T: DualTreeScalar> LinearFilamentFluxDensityKernel<T> {
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
        DualTreeError::Ok
    }

    fn combine_source_summaries(
        &self,
        children: &[Self::SourceSummary],
        _child_ids: &[u32],
        out: &mut Self::SourceSummary,
    ) -> DualTreeError {
        *out = LinearFilamentFluxDensitySummary::default();
        for i in 0..children.len() {
            out.weight = out.weight + children[i].weight;
            add3_in_place(&mut out.start_accum, children[i].start_accum);
            add3_in_place(&mut out.end_accum, children[i].end_accum);
            add3_in_place(&mut out.current_element, children[i].current_element);
            if children[i].wire_radius > out.wire_radius {
                out.wire_radius = children[i].wire_radius;
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

        let inv_weight = T::ONE / source.weight;
        let avg_start = scale3(source.start_accum, inv_weight);
        let avg_end = scale3(source.end_accum, inv_weight);
        let center = scale3(add3(avg_start, avg_end), T::from_f64(0.5));
        let length_eq = norm3(sub3(avg_end, avg_start));
        let current_element_norm = norm3(source.current_element);
        if length_eq <= T::ZERO || current_element_norm <= T::ZERO {
            return DualTreeError::Ok;
        }

        let dir = scale3(source.current_element, T::ONE / current_element_norm);
        let half_length = length_eq * T::from_f64(0.5);
        let half_segment = scale3(dir, half_length);
        let start = sub3(center, half_segment);
        let end = add3(center, half_segment);
        let current = current_element_norm / length_eq;

        *out = tuple_to_array(flux_density_linear_filament_scalar(
            (array_to_tuple(start), array_to_tuple(end), current),
            source.wire_radius,
            array_to_tuple(target.centroid),
        ));
        DualTreeError::Ok
    }

    fn zero_output(&self, out: &mut Self::Output) {
        *out = [T::ZERO; 3];
    }

    fn accumulate(&self, out: &mut Self::Output, contribution: &Self::Output) {
        add3_in_place(out, *contribution);
    }
}

fn add_source_to_summary<T: DualTreeScalar>(
    source: &LinearFilamentSource<T>,
    current: T,
    out: &mut LinearFilamentFluxDensitySummary<T>,
) {
    let dl = sub3(source.end, source.start);
    let length = norm3(dl);
    if length <= T::ZERO {
        if source.wire_radius > out.wire_radius {
            out.wire_radius = source.wire_radius;
        }
        return;
    }

    out.weight = out.weight + length;
    add3_in_place(&mut out.start_accum, scale3(source.start, length));
    add3_in_place(&mut out.end_accum, scale3(source.end, length));
    add3_in_place(&mut out.current_element, scale3(dl, current));
    if source.wire_radius > out.wire_radius {
        out.wire_radius = source.wire_radius;
    }
}

#[inline]
fn array_to_tuple<T: DualTreeScalar>(value: [T; 3]) -> (T, T, T) {
    (value[0], value[1], value[2])
}

#[inline]
fn tuple_to_array<T: DualTreeScalar>(value: (T, T, T)) -> [T; 3] {
    [value.0, value.1, value.2]
}

#[inline]
fn add3<T: DualTreeScalar>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn sub3<T: DualTreeScalar>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn scale3<T: DualTreeScalar>(value: [T; 3], scale: T) -> [T; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

#[inline]
fn dot3<T: DualTreeScalar>(a: [T; 3], b: [T; 3]) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm3<T: DualTreeScalar>(value: [T; 3]) -> T {
    dot3(value, value).sqrt()
}
