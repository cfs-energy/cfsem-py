use crate::math::{add3_in_place, norm3};
use crate::physics::hierarchical::{
    Aabb, BoundedGeometry, HierarchicalError, HierarchicalKernel, Scalar, TargetCollection,
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

impl<'a, T: Scalar> DipoleTargets<'a, T> {
    /// Create borrowed target columns.
    #[inline]
    pub fn new(x: &'a [T], y: &'a [T], z: &'a [T]) -> Self {
        Self { x, y, z }
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
    fn has_consistent_lengths(self) -> bool {
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

#[inline]
pub(super) fn summarize_weighted_source_centroid<T: Scalar>(
    source_ids: &[u32],
    sources: &[DipoleSource<T>],
    moments: &[[T; 3]],
    centroid: &mut [T; 3],
    weight: &mut T,
) {
    *centroid = [T::ZERO; 3];
    *weight = T::ZERO;
    for i in 0..source_ids.len() {
        let source_id = source_ids[i] as usize;
        let source_weight = norm3(moments[source_id]);
        if source_weight <= T::ZERO {
            continue;
        }
        *weight = *weight + source_weight;
        for axis in 0..3 {
            centroid[axis] =
                sources[source_id].position[axis].mul_add(source_weight, centroid[axis]);
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
