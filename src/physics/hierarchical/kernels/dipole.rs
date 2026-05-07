use crate::physics::hierarchical::{Aabb, BoundedGeometry, DualTreeError, DualTreeScalar};
use crate::physics::point_source::dipole::{
    flux_density_dipole_scalar_generic, vector_potential_dipole_scalar_generic,
};

/// Point source location for generic dipole kernels.
#[derive(Clone, Copy, Debug, Default)]
pub struct DipoleSource<T: DualTreeScalar> {
    pub position: [T; 3],
    pub outer_radius: T,
}

/// Point target location for generic dipole kernels.
#[derive(Clone, Copy, Debug, Default)]
pub struct DipoleTarget<T: DualTreeScalar> {
    pub position: [T; 3],
}

impl<T: DualTreeScalar> BoundedGeometry for DipoleSource<T> {
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

impl<T: DualTreeScalar> BoundedGeometry for DipoleTarget<T> {
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
pub struct DipoleTargetSummary<T: DualTreeScalar> {
    pub centroid: [T; 3],
    pub count: T,
}

pub(super) fn summarize_centroid<T: DualTreeScalar>(
    source_ids: &[u32],
    sources: &[DipoleSource<T>],
    centroid: &mut [T; 3],
    count: &mut T,
) {
    *centroid = [T::ZERO; 3];
    *count = T::ZERO;
    for i in 0..source_ids.len() {
        let source_id = source_ids[i] as usize;
        *count = *count + T::ONE;
        add3_in_place(centroid, sources[source_id].position);
    }
    if *count > T::ZERO {
        for axis in 0..3 {
            centroid[axis] = centroid[axis] / *count;
        }
    }
}

pub(super) fn summarize_target_leaf<T: DualTreeScalar>(
    target_ids: &[u32],
    targets: &[DipoleTarget<T>],
    out: &mut DipoleTargetSummary<T>,
) -> DualTreeError {
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
    DualTreeError::Ok
}

pub(super) fn combine_target<T: DualTreeScalar>(
    children: &[DipoleTargetSummary<T>],
    out: &mut DipoleTargetSummary<T>,
) -> DualTreeError {
    *out = DipoleTargetSummary::default();
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
    DualTreeError::Ok
}

pub(super) fn dipole_field<T: DualTreeScalar>(
    target: [T; 3],
    source: [T; 3],
    moment: [T; 3],
    outer_radius: T,
    out: &mut [T; 3],
) -> DualTreeError {
    *out = flux_density_dipole_scalar_generic(source, moment, outer_radius, target);
    DualTreeError::Ok
}

pub(super) fn dipole_field_derivative_component<T: DualTreeScalar>(
    r: [T; 3],
    moment_axis: usize,
    derivative_axis: usize,
    c: T,
) -> [T; 3] {
    let r2 = dot3(r, r);
    let rmag = r2.sqrt();
    let r5 = r2 * r2 * rmag;
    let r7 = r5 * r2;
    let three = T::from_f64(3.0);
    let mut out = [T::ZERO; 3];
    let r_m = r[moment_axis];
    let r_a = r[derivative_axis];

    for out_axis in 0..3 {
        let delta_im = if out_axis == moment_axis {
            T::ONE
        } else {
            T::ZERO
        };
        let delta_ia = if out_axis == derivative_axis {
            T::ONE
        } else {
            T::ZERO
        };
        let delta_ma = if moment_axis == derivative_axis {
            T::ONE
        } else {
            T::ZERO
        };
        let numerator = delta_ia.mul_add(r_m, r[out_axis].mul_add(delta_ma, delta_im * r_a));
        let second = r[out_axis] * r_m * r_a / r7;
        out[out_axis] = c * T::from_f64(-15.0).mul_add(second, three * numerator / r5);
    }
    out
}

pub(super) fn dipole_vector_potential<T: DualTreeScalar>(
    target: [T; 3],
    source: [T; 3],
    moment: [T; 3],
    outer_radius: T,
    out: &mut [T; 3],
) -> DualTreeError {
    *out = vector_potential_dipole_scalar_generic(source, moment, outer_radius, target);
    DualTreeError::Ok
}

pub(super) fn dipole_vector_potential_derivative_component<T: DualTreeScalar>(
    r: [T; 3],
    moment_axis: usize,
    derivative_axis: usize,
    c: T,
) -> [T; 3] {
    let r2 = dot3(r, r);
    let rmag = r2.sqrt();
    let r3 = r2 * rmag;
    let r5 = r3 * r2;
    let three = T::from_f64(3.0);
    let mut out = [T::ZERO; 3];

    for out_axis in 0..3 {
        let mut sum = T::ZERO;
        for r_axis in 0..3 {
            let epsilon = levi_civita(out_axis, moment_axis, r_axis);
            if epsilon == 0 {
                continue;
            }
            let delta = if r_axis == derivative_axis {
                T::ONE
            } else {
                T::ZERO
            };
            let term = (T::ZERO - three).mul_add(r[r_axis] * r[derivative_axis] / r5, delta / r3);
            if epsilon > 0 {
                sum = sum + term;
            } else {
                sum = sum - term;
            }
        }
        out[out_axis] = c * sum;
    }

    out
}

#[inline]
pub(super) fn add3_in_place<T: DualTreeScalar>(out: &mut [T; 3], value: [T; 3]) {
    for axis in 0..3 {
        out[axis] = out[axis] + value[axis];
    }
}

#[inline]
pub(super) fn add_matrix_in_place<T: DualTreeScalar>(out: &mut [[T; 3]; 3], value: [[T; 3]; 3]) {
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = out[i][j] + value[i][j];
        }
    }
}

#[inline]
pub(super) fn add_outer_in_place<T: DualTreeScalar>(out: &mut [[T; 3]; 3], a: [T; 3], b: [T; 3]) {
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i].mul_add(b[j], out[i][j]);
        }
    }
}

#[inline]
pub(super) fn sub3<T: DualTreeScalar>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot3<T: DualTreeScalar>(a: [T; 3], b: [T; 3]) -> T {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

#[inline]
fn levi_civita(a: usize, b: usize, c: usize) -> i32 {
    if a == b || b == c || a == c {
        0
    } else if (a == 0 && b == 1 && c == 2)
        || (a == 1 && b == 2 && c == 0)
        || (a == 2 && b == 0 && c == 1)
    {
        1
    } else {
        -1
    }
}
