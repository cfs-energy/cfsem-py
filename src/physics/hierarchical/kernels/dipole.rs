use crate::math::{add3_in_place, norm3};
use crate::physics::hierarchical::{
    Aabb, BoundedGeometry, BoundedGeometryCollection, HierarchicalError, HierarchicalKernel,
    Scalar, SourceCollection, SourceMomentCollection, TargetCollection,
};
use crate::physics::point_source::dipole::{
    flux_density_dipole_scalar_generic, vector_potential_dipole_scalar_generic,
};

/// Point source location for generic dipole kernels.
#[derive(Clone, Copy, Debug, Default)]
pub struct DipoleSource<T: Scalar> {
    pub position: [T; 3],
    pub outer_radius: T,
}

/// Borrowed component-column dipole source geometry.
#[derive(Clone, Copy, Debug)]
pub struct DipoleSources<'a, T: Scalar> {
    pub x: &'a [T],
    pub y: &'a [T],
    pub z: &'a [T],
    pub outer_radius: &'a [T],
}

impl<'a, T: Scalar> DipoleSources<'a, T> {
    /// Create borrowed dipole source columns.
    #[inline]
    pub fn new(x: &'a [T], y: &'a [T], z: &'a [T], outer_radius: &'a [T]) -> Self {
        Self {
            x,
            y,
            z,
            outer_radius,
        }
    }

    /// Return one scalar source geometry value.
    #[inline]
    pub fn source_value(self, index: usize) -> DipoleSource<T> {
        DipoleSource {
            position: [self.x[index], self.y[index], self.z[index]],
            outer_radius: self.outer_radius[index],
        }
    }
}

/// Point target location for generic dipole kernels.
#[derive(Clone, Copy, Debug, Default)]
pub struct DipoleTarget<T: Scalar> {
    pub position: [T; 3],
}

/// Borrowed component-column target points for dipole-like 3D target kernels.
#[derive(Clone, Copy, Debug)]
pub struct DipoleTargets<'a, T: Scalar> {
    pub x: &'a [T],
    pub y: &'a [T],
    pub z: &'a [T],
}

/// Borrowed row-major target points for 3D target kernels.
#[derive(Clone, Copy, Debug)]
pub struct DipoleTargetRows<'a, T: Scalar> {
    pub xyz: &'a [T],
}

/// Borrowed component-column dipole magnetic moments.
#[derive(Clone, Copy, Debug)]
pub struct DipoleMoments<'a, T: Scalar> {
    pub x: &'a [T],
    pub y: &'a [T],
    pub z: &'a [T],
}

impl<'a, T: Scalar> DipoleMoments<'a, T> {
    /// Create borrowed dipole moment columns.
    #[inline]
    pub fn new(x: &'a [T], y: &'a [T], z: &'a [T]) -> Self {
        Self { x, y, z }
    }
}

impl<'a, T: Scalar> DipoleTargets<'a, T> {
    /// Create borrowed target columns.
    #[inline]
    pub fn new(x: &'a [T], y: &'a [T], z: &'a [T]) -> Self {
        Self { x, y, z }
    }
}

impl<'a, T: Scalar> DipoleTargetRows<'a, T> {
    /// Create borrowed row-major target coordinates with shape `(n, 3)`.
    #[inline]
    pub fn new(xyz: &'a [T]) -> Self {
        Self { xyz }
    }
}

impl<'a, K, T> TargetCollection<K> for DipoleTargets<'a, T>
where
    K: HierarchicalKernel<Scalar = T, TargetGeometry = DipoleTarget<T>>,
    T: Scalar,
{
    #[inline]
    fn len(self) -> usize {
        self.x.len()
    }

    #[inline]
    fn valid_lengths(self) -> bool {
        self.x.len() == self.y.len() && self.x.len() == self.z.len()
    }

    #[inline]
    fn target(self, index: usize) -> DipoleTarget<T> {
        DipoleTarget {
            position: [self.x[index], self.y[index], self.z[index]],
        }
    }

    #[inline]
    fn slice(self, start: usize, end: usize) -> Self {
        Self {
            x: &self.x[start..end],
            y: &self.y[start..end],
            z: &self.z[start..end],
        }
    }
}

impl<'a, K, T> TargetCollection<K> for DipoleTargetRows<'a, T>
where
    K: HierarchicalKernel<Scalar = T, TargetGeometry = DipoleTarget<T>>,
    T: Scalar,
{
    #[inline]
    fn len(self) -> usize {
        self.xyz.len() / 3
    }

    #[inline]
    fn valid_lengths(self) -> bool {
        self.xyz.len().is_multiple_of(3)
    }

    #[inline]
    fn target(self, index: usize) -> DipoleTarget<T> {
        let start = 3 * index;
        DipoleTarget {
            position: [self.xyz[start], self.xyz[start + 1], self.xyz[start + 2]],
        }
    }

    #[inline]
    fn slice(self, start: usize, end: usize) -> Self {
        Self {
            xyz: &self.xyz[3 * start..3 * end],
        }
    }
}

impl<'a, T: Scalar> BoundedGeometryCollection<T> for DipoleSources<'a, T> {
    #[inline]
    fn len(self) -> usize {
        self.x.len()
    }

    #[inline]
    fn valid_lengths(self) -> bool {
        self.x.len() == self.y.len()
            && self.x.len() == self.z.len()
            && self.x.len() == self.outer_radius.len()
    }

    #[inline]
    fn aabb(self, index: usize) -> Aabb<T> {
        self.source_value(index).aabb()
    }

    #[inline]
    fn representative_point(self, index: usize) -> [T; 3] {
        [self.x[index], self.y[index], self.z[index]]
    }
}

impl<'a, K, T> SourceCollection<K> for DipoleSources<'a, T>
where
    K: HierarchicalKernel<Scalar = T, SourceGeometry = DipoleSource<T>>,
    T: Scalar,
{
    #[inline]
    fn source(self, index: usize) -> DipoleSource<T> {
        self.source_value(index)
    }
}

impl<'a, K, T> SourceMomentCollection<K> for DipoleMoments<'a, T>
where
    K: HierarchicalKernel<Scalar = T, SourceMoment = [T; 3]>,
    T: Scalar,
{
    #[inline]
    fn len(self) -> usize {
        self.x.len()
    }

    #[inline]
    fn valid_lengths(self) -> bool {
        self.x.len() == self.y.len() && self.x.len() == self.z.len()
    }

    #[inline]
    fn moment(self, index: usize) -> [T; 3] {
        [self.x[index], self.y[index], self.z[index]]
    }
}

impl<T: Scalar> BoundedGeometry for DipoleSource<T> {
    type Scalar = T;

    #[inline]
    fn aabb(&self) -> Aabb<Self::Scalar> {
        if self.outer_radius > T::ZERO {
            Aabb {
                min: [
                    self.position[0] - self.outer_radius,
                    self.position[1] - self.outer_radius,
                    self.position[2] - self.outer_radius,
                ],
                max: [
                    self.position[0] + self.outer_radius,
                    self.position[1] + self.outer_radius,
                    self.position[2] + self.outer_radius,
                ],
            }
        } else {
            Aabb::from_point(self.position)
        }
    }

    #[inline]
    fn representative_point(&self) -> [Self::Scalar; 3] {
        self.position
    }
}

impl<T: Scalar> BoundedGeometry for DipoleTarget<T> {
    type Scalar = T;

    #[inline]
    fn aabb(&self) -> Aabb<Self::Scalar> {
        Aabb::from_point(self.position)
    }

    #[inline]
    fn representative_point(&self) -> [Self::Scalar; 3] {
        self.position
    }
}

/// Target summary for point targets.
#[derive(Clone, Copy, Debug, Default)]
pub struct DipoleTargetSummary<T: Scalar> {
    pub centroid: [T; 3],
    pub count: T,
}

/// Source summary shared by dipole flux-density and vector-potential kernels.
///
/// Both fields use the same Barnes-Hut source approximation: one total dipole
/// moment located at the moment-magnitude-weighted source centroid.
#[derive(Clone, Copy, Debug, Default)]
pub struct DipoleSummary<T: Scalar> {
    /// Moment-magnitude-weighted source location used by the far-field term.
    pub centroid: [T; 3],
    /// Total dipole moment for all sources in the node.
    pub moment: [T; 3],
    /// Sum of source moment magnitudes used for centroid weighting.
    pub weight: T,
}

/// Build a shared dipole source summary from leaf source indices.
#[inline]
pub(super) fn summarize_dipole_leaf_sources<T, S, F>(
    source_ids: &[u32],
    sources: S,
    moment_at: F,
    out: &mut DipoleSummary<T>,
) -> HierarchicalError
where
    T: Scalar,
    S: BoundedGeometryCollection<T>,
    F: Fn(usize) -> [T; 3],
{
    *out = DipoleSummary::default();
    summarize_weighted_source_centroid(
        source_ids,
        sources,
        &moment_at,
        &mut out.centroid,
        &mut out.weight,
    );
    for i in 0..source_ids.len() {
        let source_id = source_ids[i] as usize;
        add3_in_place(&mut out.moment, moment_at(source_id));
    }
    HierarchicalError::Ok
}

/// Combine child dipole summaries into one parent summary.
#[inline]
pub(super) fn combine_dipole_source_summaries<T: Scalar>(
    children: &[DipoleSummary<T>],
    out: &mut DipoleSummary<T>,
) -> HierarchicalError {
    *out = DipoleSummary::default();
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
pub(super) fn summarize_weighted_source_centroid<T, S, F>(
    source_ids: &[u32],
    sources: S,
    moment_at: F,
    centroid: &mut [T; 3],
    weight: &mut T,
) where
    T: Scalar,
    S: BoundedGeometryCollection<T>,
    F: Fn(usize) -> [T; 3],
{
    *centroid = [T::ZERO; 3];
    *weight = T::ZERO;
    for i in 0..source_ids.len() {
        let source_id = source_ids[i] as usize;
        let moment = moment_at(source_id);
        let source_weight = norm3(moment);
        if source_weight <= T::ZERO {
            continue;
        }
        *weight = *weight + source_weight;
        let position = sources.representative_point(source_id);
        for axis in 0..3 {
            centroid[axis] = position[axis].mul_add(source_weight, centroid[axis]);
        }
    }
    if *weight > T::ZERO {
        for axis in 0..3 {
            centroid[axis] = centroid[axis] / *weight;
        }
    }
}

#[inline]
pub(super) fn summarize_target_leaf<T: Scalar>(
    target_ids: &[u32],
    targets: &[DipoleTarget<T>],
    out: &mut DipoleTargetSummary<T>,
) -> HierarchicalError {
    *out = DipoleTargetSummary::default();
    for i in 0..target_ids.len() {
        let target_id = target_ids[i] as usize;
        out.count = out.count + T::ONE;
        add3_in_place(&mut out.centroid, targets[target_id].position);
    }
    if out.count > T::ZERO {
        for axis in 0..3 {
            out.centroid[axis] = out.centroid[axis] / out.count;
        }
    }
    HierarchicalError::Ok
}

#[inline]
pub(super) fn dipole_field<T: Scalar>(
    target: [T; 3],
    source: [T; 3],
    moment: [T; 3],
    outer_radius: T,
    out: &mut [T; 3],
) {
    *out = flux_density_dipole_scalar_generic(source, moment, outer_radius, target);
}

#[inline]
pub(super) fn dipole_vector_potential<T: Scalar>(
    target: [T; 3],
    source: [T; 3],
    moment: [T; 3],
    outer_radius: T,
    out: &mut [T; 3],
) {
    *out = vector_potential_dipole_scalar_generic(source, moment, outer_radius, target);
}
