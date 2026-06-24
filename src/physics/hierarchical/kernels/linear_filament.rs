use crate::math::{add3_in_place, cross3, norm3, scale3, sub3};
use crate::physics::hierarchical::{
    Aabb, BoundedGeometry, BoundedGeometryCollection, HierarchicalError, HierarchicalKernel,
    Scalar, SourceCollection, geometric_accept_far,
};

/// Lower closure-ratio bound for source summaries that can be represented as loops.
const CLOSED_SUMMARY_CLOSURE_RATIO_MAX: f64 = 0.2;
/// Upper closure-ratio bound for source summaries that can be represented as segments.
const OPEN_SUMMARY_CLOSURE_RATIO_MIN: f64 = 0.8;

/// Finite linear filament source geometry.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearFilamentSource<T: Scalar> {
    /// Segment start point.
    pub start: [T; 3],
    /// Segment end point.
    pub end: [T; 3],
    /// Wire radius used by the exact near-field filament kernel.
    pub wire_radius: T,
}

/// Borrowed component-column linear filament source geometry.
#[derive(Clone, Copy, Debug)]
pub struct LinearFilamentSources<'a, T: Scalar> {
    pub x: &'a [T],
    pub y: &'a [T],
    pub z: &'a [T],
    pub dlx: &'a [T],
    pub dly: &'a [T],
    pub dlz: &'a [T],
    pub wire_radius: &'a [T],
}

impl<'a, T: Scalar> LinearFilamentSources<'a, T> {
    /// Create borrowed linear filament source columns.
    #[inline]
    pub fn new(
        xyz: (&'a [T], &'a [T], &'a [T]),
        dlxyz: (&'a [T], &'a [T], &'a [T]),
        wire_radius: &'a [T],
    ) -> Self {
        Self {
            x: xyz.0,
            y: xyz.1,
            z: xyz.2,
            dlx: dlxyz.0,
            dly: dlxyz.1,
            dlz: dlxyz.2,
            wire_radius,
        }
    }

    /// Return one scalar source geometry value.
    #[inline]
    pub fn source_value(self, index: usize) -> LinearFilamentSource<T> {
        let start = [self.x[index], self.y[index], self.z[index]];
        LinearFilamentSource {
            start,
            end: [
                start[0] + self.dlx[index],
                start[1] + self.dly[index],
                start[2] + self.dlz[index],
            ],
            wire_radius: self.wire_radius[index],
        }
    }
}

impl<'a, T: Scalar> BoundedGeometryCollection<T> for LinearFilamentSources<'a, T> {
    #[inline]
    /// Return the number of items in this collection view.
    fn len(self) -> usize {
        self.x.len()
    }

    #[inline]
    /// Return whether all backing slices have compatible lengths.
    fn valid_lengths(self) -> bool {
        let n = self.x.len();
        self.y.len() == n
            && self.z.len() == n
            && self.dlx.len() == n
            && self.dly.len() == n
            && self.dlz.len() == n
            && self.wire_radius.len() == n
    }

    #[inline]
    /// Return the axis-aligned bounds for one geometry item.
    fn aabb(self, index: usize) -> Aabb<T> {
        self.source_value(index).aabb()
    }

    #[inline]
    /// Return the representative point used for tree construction.
    fn representative_point(self, index: usize) -> [T; 3] {
        self.source_value(index).representative_point()
    }
}

impl<'a, K, T> SourceCollection<K> for LinearFilamentSources<'a, T>
where
    K: HierarchicalKernel<Scalar = T, SourceGeometry = LinearFilamentSource<T>>,
    T: Scalar,
{
    #[inline]
    /// Return one source geometry item by index.
    fn source(self, index: usize) -> LinearFilamentSource<T> {
        self.source_value(index)
    }
}

impl<T: Scalar> BoundedGeometry for LinearFilamentSource<T> {
    type Scalar = T;

    #[inline]
    /// Return the axis-aligned bounds for one geometry item.
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
    /// Return the representative point used for tree construction.
    fn representative_point(&self) -> [Self::Scalar; 3] {
        let half = crate::math::cast::<T>(0.5);
        [
            half.mul_add(self.start[0] + self.end[0], T::ZERO),
            half.mul_add(self.start[1] + self.end[1], T::ZERO),
            half.mul_add(self.start[2] + self.end[2], T::ZERO),
        ]
    }
}

/// Source summary shared by finite linear filament B-field and A-field kernels.
///
/// The summary carries one point-segment source term plus a residual dipole
/// correction so both open current elements and locally closed current paths
/// have a common representation.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearFilamentSummary<T: Scalar> {
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

/// Build a shared linear-filament source summary from leaf source indices.
#[inline]
pub(super) fn summarize_linear_filament_leaf_sources<T, F, G>(
    source_ids: &[u32],
    source_at: F,
    current_at: G,
    out: &mut LinearFilamentSummary<T>,
) -> HierarchicalError
where
    T: Scalar,
    F: Fn(usize) -> LinearFilamentSource<T>,
    G: Fn(usize) -> T,
{
    *out = LinearFilamentSummary::default();
    for i in 0..source_ids.len() {
        let source_id = source_ids[i] as usize;
        let source = source_at(source_id);
        add_source_to_summary(&source, current_at(source_id), out);
    }
    finalize_leaf_source_summary(out);
    HierarchicalError::Ok
}

/// Combine child linear-filament summaries into one parent summary.
#[inline]
pub(super) fn combine_linear_filament_source_summaries<T: Scalar>(
    children: &[LinearFilamentSummary<T>],
    out: &mut LinearFilamentSummary<T>,
) -> HierarchicalError {
    *out = LinearFilamentSummary::default();
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

/// Apply the linear-filament closure-ratio guard and geometric acceptance test.
#[inline]
pub(super) fn linear_filament_accept_far<T: Scalar>(
    target_aabb: Aabb<T>,
    source_aabb: Aabb<T>,
    source: &LinearFilamentSummary<T>,
    theta: T,
) -> bool {
    if source.weight > T::ZERO {
        let closure_ratio = source.magnitude / source.weight;
        if closure_ratio >= crate::math::cast::<T>(CLOSED_SUMMARY_CLOSURE_RATIO_MAX)
            && closure_ratio <= crate::math::cast::<T>(OPEN_SUMMARY_CLOSURE_RATIO_MIN)
        {
            return false;
        }
    }
    geometric_accept_far(target_aabb, source_aabb, theta)
}

/// Accumulate one exact linear filament source into an unfinalized summary.
#[inline]
fn add_source_to_summary<T: Scalar>(
    source: &LinearFilamentSource<T>,
    current: T,
    out: &mut LinearFilamentSummary<T>,
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

/// Normalize leaf summary origins and translate the residual dipole moment.
#[inline]
fn finalize_leaf_source_summary<T: Scalar>(summary: &mut LinearFilamentSummary<T>) {
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

/// Convert the accumulated current element to direction and magnitude form.
#[inline]
fn finalize_current_element<T: Scalar>(summary: &mut LinearFilamentSummary<T>) {
    summary.magnitude = norm3(summary.direction);
    if summary.magnitude > T::ZERO {
        summary.direction = scale3(summary.direction, T::ONE / summary.magnitude);
    }
}

/// Return one half in the kernel scalar type.
#[inline]
pub(super) fn half<T: Scalar>() -> T {
    crate::math::cast::<T>(0.5)
}

/// Convert a fixed-size coordinate array to the tuple interface used upstream.
#[inline]
pub(super) fn array_to_tuple<T: Scalar>(value: [T; 3]) -> (T, T, T) {
    (value[0], value[1], value[2])
}

/// Convert an upstream tuple result back to a fixed-size coordinate array.
#[inline]
pub(super) fn tuple_to_array<T: Scalar>(value: (T, T, T)) -> [T; 3] {
    [value.0, value.1, value.2]
}
