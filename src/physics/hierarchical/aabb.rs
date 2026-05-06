use super::DualTreeScalar;

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb<T: DualTreeScalar> {
    pub min: [T; 3],
    pub max: [T; 3],
}

impl<T: DualTreeScalar> Aabb<T> {
    /// Empty bounding box.
    #[inline]
    pub fn empty() -> Self {
        let inf = T::from_f64(f64::INFINITY);
        let neg_inf = T::from_f64(f64::NEG_INFINITY);
        Self {
            min: [inf; 3],
            max: [neg_inf; 3],
        }
    }

    /// Bounding box around one point.
    #[inline]
    pub fn from_point(point: [T; 3]) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    /// Union of two bounding boxes.
    #[inline]
    pub fn union(self, other: Self) -> Self {
        let mut out = self;
        for axis in 0..3 {
            if other.min[axis] < out.min[axis] {
                out.min[axis] = other.min[axis];
            }
            if other.max[axis] > out.max[axis] {
                out.max[axis] = other.max[axis];
            }
        }
        out
    }

    /// Squared diagonal length.
    #[inline]
    pub fn diameter_sq(&self) -> T {
        let mut sum = T::ZERO;
        for axis in 0..3 {
            let d = self.max[axis] - self.min[axis];
            sum = sum + d * d;
        }
        sum
    }

    /// Squared gap distance between two AABBs.
    #[inline]
    pub fn gap_distance_sq(&self, other: &Self) -> T {
        let mut sum = T::ZERO;
        for axis in 0..3 {
            let gap = if self.min[axis] > other.max[axis] {
                self.min[axis] - other.max[axis]
            } else if other.min[axis] > self.max[axis] {
                other.min[axis] - self.max[axis]
            } else {
                T::ZERO
            };
            sum = sum + gap * gap;
        }
        sum
    }

    /// Center of the AABB.
    #[inline]
    pub fn centroid(&self) -> [T; 3] {
        let half = T::from_f64(0.5);
        [
            (self.min[0] + self.max[0]) * half,
            (self.min[1] + self.max[1]) * half,
            (self.min[2] + self.max[2]) * half,
        ]
    }

    #[inline]
    pub(crate) fn extent(&self, axis: usize) -> T {
        self.max[axis] - self.min[axis]
    }
}
