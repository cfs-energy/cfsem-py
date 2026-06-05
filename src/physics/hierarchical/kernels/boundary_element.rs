use crate::math::{add3_in_place, cross3, norm3, scale3, sub3};
use crate::mesh::TriangleMeshView;
use crate::physics::boundary_element::{calc_tri_area, triangle_current_density};
use crate::physics::hierarchical::{
    Aabb, BoundedGeometry, BoundedGeometryCollection, HierarchicalError, HierarchicalKernel,
    Scalar, SourceCollection, SourceMomentCollection, geometric_accept_far,
};

/// Lower closure-ratio bound for source summaries that can be represented as loops.
const CLOSED_SUMMARY_CLOSURE_RATIO_MAX: f64 = 0.2;
/// Upper closure-ratio bound for source summaries that can be represented as open current patches.
const OPEN_SUMMARY_CLOSURE_RATIO_MIN: f64 = 0.8;

/// Triangular boundary-element source geometry.
#[derive(Clone, Copy, Debug, Default)]
pub struct BoundaryElementTriangle<T: Scalar> {
    pub n0: [T; 3],
    pub n1: [T; 3],
    pub n2: [T; 3],
}

/// Borrowed component-column boundary-element triangle geometry.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryElementTriangles<'a, T: Scalar> {
    pub mesh: &'a TriangleMeshView<'a>,
    marker: core::marker::PhantomData<T>,
}

impl<'a> BoundaryElementTriangles<'a, f64> {
    /// Create borrowed triangle mesh geometry for hierarchical BEM sources.
    #[inline]
    pub fn new(mesh: &'a TriangleMeshView<'a>) -> Self {
        Self {
            mesh,
            marker: core::marker::PhantomData,
        }
    }

    /// Return one scalar triangle geometry value.
    #[inline]
    pub fn source_value(self, index: usize) -> BoundaryElementTriangle<f64> {
        let [n0, n1, n2] = self.mesh.triangle_nodes(index);
        BoundaryElementTriangle { n0, n1, n2 }
    }
}

impl<'a> BoundedGeometryCollection<f64> for BoundaryElementTriangles<'a, f64> {
    #[inline]
    fn len(self) -> usize {
        self.mesh.len()
    }

    #[inline]
    fn valid_lengths(self) -> bool {
        true
    }

    #[inline]
    fn aabb(self, index: usize) -> Aabb<f64> {
        self.source_value(index).aabb()
    }

    #[inline]
    fn representative_point(self, index: usize) -> [f64; 3] {
        self.source_value(index).representative_point()
    }
}

impl<'a, K> SourceCollection<K> for BoundaryElementTriangles<'a, f64>
where
    K: HierarchicalKernel<Scalar = f64, SourceGeometry = BoundaryElementTriangle<f64>>,
{
    #[inline]
    fn source(self, index: usize) -> BoundaryElementTriangle<f64> {
        self.source_value(index)
    }
}

/// Borrowed nodal stream-function values for boundary-element source moments.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryElementNodalValues<'a, T: Scalar> {
    pub triangles: BoundaryElementTriangles<'a, T>,
    pub s: &'a [T],
}

impl<'a> BoundaryElementNodalValues<'a, f64> {
    /// Create borrowed nodal stream-function values for triangle moments.
    #[inline]
    pub fn new(triangles: BoundaryElementTriangles<'a, f64>, s: &'a [f64]) -> Self {
        Self { triangles, s }
    }
}

impl<'a, K> SourceMomentCollection<K> for BoundaryElementNodalValues<'a, f64>
where
    K: HierarchicalKernel<Scalar = f64, SourceMoment = [f64; 3]>,
{
    #[inline]
    fn len(self) -> usize {
        self.triangles.len()
    }

    #[inline]
    fn valid_lengths(self) -> bool {
        self.s.len() == self.triangles.mesh.nnode() && self.triangles.valid_lengths()
    }

    #[inline]
    fn moment(self, index: usize) -> [f64; 3] {
        self.triangles.mesh.triangle_scalars(index, self.s)
    }
}

impl<T: Scalar> BoundedGeometry for BoundaryElementTriangle<T> {
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
        let third = T::ONE / crate::math::cast::<T>(3.0);
        [
            (self.n0[0] + self.n1[0] + self.n2[0]) * third,
            (self.n0[1] + self.n1[1] + self.n2[1]) * third,
            (self.n0[2] + self.n1[2] + self.n2[2]) * third,
        ]
    }
}

/// Source summary for boundary-element clusters.
#[derive(Clone, Copy, Debug, Default)]
pub struct BoundaryElementSummary<T: Scalar> {
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
pub(super) fn summarize_leaf_sources<K, T, S, M>(
    source_ids: &[u32],
    sources: S,
    moments: M,
    out: &mut BoundaryElementSummary<T>,
) -> HierarchicalError
where
    K: HierarchicalKernel<
            Scalar = T,
            SourceGeometry = BoundaryElementTriangle<T>,
            SourceMoment = [T; 3],
        >,
    T: Scalar,
    S: SourceCollection<K>,
    M: SourceMomentCollection<K>,
{
    *out = BoundaryElementSummary::default();
    for i in 0..source_ids.len() {
        let source_id = source_ids[i] as usize;
        let source = sources.source(source_id);
        add_source_to_summary(&source, moments.moment(source_id), out);
    }
    finalize_leaf_source_summary(out);
    HierarchicalError::Ok
}

#[inline]
pub(super) fn combine_source_summaries<T: Scalar>(
    children: &[BoundaryElementSummary<T>],
    out: &mut BoundaryElementSummary<T>,
) -> HierarchicalError {
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

    HierarchicalError::Ok
}

#[inline]
fn add_source_to_summary<T: Scalar>(
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
    let area = physical_area * crate::math::cast::<T>(0.5);
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
fn finalize_leaf_source_summary<T: Scalar>(summary: &mut BoundaryElementSummary<T>) {
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
pub(super) fn has_current<T: Scalar>(summary: &BoundaryElementSummary<T>) -> bool {
    norm3(summary.current_element) > T::ZERO
}

/// Apply the BEM current-closure guard and geometric acceptance test.
#[inline]
pub(super) fn boundary_element_accept_far<T: Scalar>(
    target_aabb: Aabb<T>,
    source_aabb: Aabb<T>,
    source: &BoundaryElementSummary<T>,
    theta: T,
) -> bool {
    if source.weight > T::ZERO {
        let closure_ratio = norm3(source.current_element) / source.weight;
        if closure_ratio >= crate::math::cast::<T>(CLOSED_SUMMARY_CLOSURE_RATIO_MAX)
            && closure_ratio <= crate::math::cast::<T>(OPEN_SUMMARY_CLOSURE_RATIO_MIN)
        {
            return false;
        }
    }
    geometric_accept_far(target_aabb, source_aabb, theta)
}

#[inline]
fn half<T: Scalar>() -> T {
    crate::math::cast::<T>(0.5)
}
