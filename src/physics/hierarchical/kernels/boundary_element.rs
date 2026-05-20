use crate::math::{add3_in_place, cross3, norm3, scale3, sub3};
use crate::physics::boundary_element::{calc_tri_area, triangle_current_density};
use crate::physics::hierarchical::{Aabb, BoundedGeometry, DualTreeError, DualTreeScalar};

/// Triangular boundary-element source geometry.
#[derive(Clone, Copy, Debug, Default)]
pub struct BoundaryElementTriangle<T: DualTreeScalar> {
    pub n0: [T; 3],
    pub n1: [T; 3],
    pub n2: [T; 3],
}

impl<T: DualTreeScalar> BoundedGeometry for BoundaryElementTriangle<T> {
    type Scalar = T;

    #[inline]
    fn aabb(&self) -> Aabb<Self::Scalar> {
        let mut min = [T::ZERO; 3];
        let mut max = [T::ZERO; 3];
        for axis in 0..3 {
            min[axis] = self.n0[axis];
            max[axis] = self.n0[axis];
            if self.n1[axis] < min[axis] {
                min[axis] = self.n1[axis];
            }
            if self.n2[axis] < min[axis] {
                min[axis] = self.n2[axis];
            }
            if self.n1[axis] > max[axis] {
                max[axis] = self.n1[axis];
            }
            if self.n2[axis] > max[axis] {
                max[axis] = self.n2[axis];
            }
        }
        Aabb { min, max }
    }

    #[inline]
    fn representative_point(&self) -> [Self::Scalar; 3] {
        let third = T::ONE / T::from_f64(3.0);
        [
            (self.n0[0] + self.n1[0] + self.n2[0]) * third,
            (self.n0[1] + self.n1[1] + self.n2[1]) * third,
            (self.n0[2] + self.n1[2] + self.n2[2]) * third,
        ]
    }
}

/// Source summary for boundary-element clusters.
#[derive(Clone, Copy, Debug, Default)]
pub struct BoundaryElementSummary<T: DualTreeScalar> {
    /// Position used by the collapsed point-current element term.
    pub origin: [T; 3],
    /// Net `K dS` current element for the accepted source cluster.
    pub current_element: [T; 3],
    /// Reference position for the magnetic-dipole correction.
    pub dipole_origin: [T; 3],
    /// Magnetic-dipole moment about `dipole_origin`.
    pub dipole_moment: [T; 3],
    /// Current-element magnitude weight used for source-position averages.
    pub weight: T,
}

#[inline]
pub(super) fn summarize_leaf_sources<T: DualTreeScalar>(
    source_ids: &[u32],
    sources: &[BoundaryElementTriangle<T>],
    moments: &[[T; 3]],
    out: &mut BoundaryElementSummary<T>,
) -> DualTreeError {
    *out = BoundaryElementSummary::default();
    for i in 0..source_ids.len() {
        let source_id = source_ids[i] as usize;
        add_source_to_summary(&sources[source_id], moments[source_id], out);
    }
    finalize_leaf_source_summary(out);
    DualTreeError::Ok
}

#[inline]
pub(super) fn combine_source_summaries<T: DualTreeScalar>(
    children: &[BoundaryElementSummary<T>],
    out: &mut BoundaryElementSummary<T>,
) -> DualTreeError {
    *out = BoundaryElementSummary::default();
    for i in 0..children.len() {
        out.weight = out.weight + children[i].weight;
        add3_in_place(
            &mut out.origin,
            scale3(children[i].origin, children[i].weight),
        );
        add3_in_place(&mut out.current_element, children[i].current_element);
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
        add3_in_place(&mut out.dipole_moment, children[i].dipole_moment);
        add3_in_place(
            &mut out.dipole_moment,
            scale3(
                cross3(
                    sub3(children[i].dipole_origin, out.dipole_origin),
                    children[i].current_element,
                ),
                half::<T>(),
            ),
        );
    }

    DualTreeError::Ok
}

#[inline]
fn add_source_to_summary<T: DualTreeScalar>(
    source: &BoundaryElementTriangle<T>,
    moment: [T; 3],
    out: &mut BoundaryElementSummary<T>,
) {
    let physical_area = calc_tri_area(source.n0, source.n1, source.n2);
    if physical_area <= T::ZERO {
        return;
    }
    // The upstream triangle kernels multiply physical area by Dunavant
    // reference-triangle weights that sum to 0.5. Use the same effective area
    // here so accepted far-field summaries stay normalized to the direct path.
    let area = physical_area * T::from_f64(0.5);
    let centroid = source.representative_point();
    let current_density = triangle_current_density(source.n0, source.n1, source.n2, moment);
    let current_element = scale3(current_density, area);
    let current_weight = norm3(current_element);
    if current_weight <= T::ZERO {
        return;
    }

    // The far-field surrogate represents the physical current distribution,
    // not just the mesh geometry. Current-element weighting prevents inactive
    // or weak-current triangles from moving the collapsed source position as
    // much as active elements.
    out.weight = out.weight + current_weight;
    add3_in_place(&mut out.origin, scale3(centroid, current_weight));
    add3_in_place(&mut out.dipole_origin, scale3(centroid, current_weight));
    add3_in_place(&mut out.current_element, current_element);
    add3_in_place(
        &mut out.dipole_moment,
        scale3(cross3(centroid, current_element), half::<T>()),
    );
}

#[inline]
fn finalize_leaf_source_summary<T: DualTreeScalar>(summary: &mut BoundaryElementSummary<T>) {
    if summary.weight > T::ZERO {
        summary.origin = scale3(summary.origin, T::ONE / summary.weight);
        summary.dipole_origin = scale3(summary.dipole_origin, T::ONE / summary.weight);
    }

    add3_in_place(
        &mut summary.dipole_moment,
        scale3(
            cross3(summary.dipole_origin, summary.current_element),
            T::ZERO - half::<T>(),
        ),
    );
}

#[inline]
pub(super) fn has_current<T: DualTreeScalar>(summary: &BoundaryElementSummary<T>) -> bool {
    norm3(summary.current_element) > T::ZERO
}

#[inline]
fn half<T: DualTreeScalar>() -> T {
    T::from_f64(0.5)
}
