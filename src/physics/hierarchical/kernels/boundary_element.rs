use crate::math::{add3_in_place, cross3, norm3, scale3, sub3};
use crate::physics::boundary_element::{calc_tri_area, triangle_current_density};
use crate::physics::hierarchical::{
    Aabb, BoundedGeometry, BoundedGeometryCollection, HierarchicalError, HierarchicalKernel,
    Scalar, SourceCollection, SourceMomentCollection,
};

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
    pub node_x: &'a [T],
    pub node_y: &'a [T],
    pub node_z: &'a [T],
    pub tri0: &'a [usize],
    pub tri1: &'a [usize],
    pub tri2: &'a [usize],
}

impl<'a, T: Scalar> BoundaryElementTriangles<'a, T> {
    /// Create borrowed triangle mesh columns for hierarchical BEM sources.
    #[inline]
    pub fn new(
        nodes: (&'a [T], &'a [T], &'a [T]),
        triangles: (&'a [usize], &'a [usize], &'a [usize]),
    ) -> Self {
        Self {
            node_x: nodes.0,
            node_y: nodes.1,
            node_z: nodes.2,
            tri0: triangles.0,
            tri1: triangles.1,
            tri2: triangles.2,
        }
    }

    /// Return one scalar triangle geometry value.
    #[inline]
    pub fn source_value(self, index: usize) -> BoundaryElementTriangle<T> {
        let i0 = self.tri0[index];
        let i1 = self.tri1[index];
        let i2 = self.tri2[index];
        BoundaryElementTriangle {
            n0: [self.node_x[i0], self.node_y[i0], self.node_z[i0]],
            n1: [self.node_x[i1], self.node_y[i1], self.node_z[i1]],
            n2: [self.node_x[i2], self.node_y[i2], self.node_z[i2]],
        }
    }

    /// Return whether every triangle index refers to an existing node.
    #[inline]
    pub fn indices_in_bounds(self) -> bool {
        let nnode = self.node_x.len();
        for i in 0..self.tri0.len() {
            if self.tri0[i] >= nnode || self.tri1[i] >= nnode || self.tri2[i] >= nnode {
                return false;
            }
        }
        true
    }
}

impl<'a, T: Scalar> BoundedGeometryCollection<T> for BoundaryElementTriangles<'a, T> {
    #[inline]
    fn geometry_len(self) -> usize {
        self.tri0.len()
    }

    #[inline]
    fn has_consistent_geometry_lengths(self) -> bool {
        self.node_x.len() == self.node_y.len()
            && self.node_x.len() == self.node_z.len()
            && self.tri0.len() == self.tri1.len()
            && self.tri0.len() == self.tri2.len()
            && self.indices_in_bounds()
    }

    #[inline]
    fn aabb(self, index: usize) -> Aabb<T> {
        self.source_value(index).aabb()
    }

    #[inline]
    fn representative_point(self, index: usize) -> [T; 3] {
        self.source_value(index).representative_point()
    }
}

impl<'a, K, T> SourceCollection<K> for BoundaryElementTriangles<'a, T>
where
    K: HierarchicalKernel<Scalar = T, SourceGeometry = BoundaryElementTriangle<T>>,
    T: Scalar,
{
    #[inline]
    fn source(self, index: usize) -> BoundaryElementTriangle<T> {
        self.source_value(index)
    }
}

/// Borrowed nodal stream-function values for boundary-element source moments.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryElementNodalValues<'a, T: Scalar> {
    pub triangles: BoundaryElementTriangles<'a, T>,
    pub s: &'a [T],
}

impl<'a, T: Scalar> BoundaryElementNodalValues<'a, T> {
    /// Create borrowed nodal stream-function values for triangle moments.
    #[inline]
    pub fn new(triangles: BoundaryElementTriangles<'a, T>, s: &'a [T]) -> Self {
        Self { triangles, s }
    }
}

impl<'a, K, T> SourceMomentCollection<K> for BoundaryElementNodalValues<'a, T>
where
    K: HierarchicalKernel<Scalar = T, SourceMoment = [T; 3]>,
    T: Scalar,
{
    #[inline]
    fn geometry_len(self) -> usize {
        self.triangles.geometry_len()
    }

    #[inline]
    fn has_consistent_geometry_lengths(self) -> bool {
        self.s.len() == self.triangles.node_x.len()
            && self.triangles.has_consistent_geometry_lengths()
    }

    #[inline]
    fn moment(self, index: usize) -> [T; 3] {
        [
            self.s[self.triangles.tri0[index]],
            self.s[self.triangles.tri1[index]],
            self.s[self.triangles.tri2[index]],
        ]
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

#[inline]
fn half<T: Scalar>() -> T {
    T::from_f64(0.5)
}
