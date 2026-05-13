use core::cmp::Ordering;

use super::{Aabb, BoundedGeometry, DualTreeError, DualTreeScalar};

const INVALID_INDEX: u32 = u32::MAX;
/// Adjacent spatial gap must exceed this multiple of the mean sorted gap before
/// the recursive builder treats it as a cluster boundary.
const SPATIAL_GAP_DOMINANCE_FACTOR: f64 = 4.0;
/// Adjacent spatial gap must cover at least this fraction of the node span
/// before the recursive builder treats it as a cluster boundary.
const SPATIAL_GAP_MIN_SPAN_FRACTION: f64 = 0.05;
/// Adjacent Morton-code gap must exceed this multiple of the mean sorted code
/// gap before the LBVH builder treats it as a cluster boundary.
const MORTON_GAP_DOMINANCE_FACTOR: f64 = 4.0;
/// Adjacent Morton-code gap must cover at least this fraction of the node's code
/// span before the LBVH builder treats it as a cluster boundary.
const MORTON_GAP_MIN_SPAN_FRACTION: f64 = 0.05;
/// Candidate gap splits must leave at least this fraction of the range on each
/// side. This keeps ordinary curve sampling gaps from creating skinny trees.
const GAP_SPLIT_MIN_SIDE_FRACTION: usize = 8;
/// Number of quantization bits per coordinate used by the 3D Morton code.
///
/// Three axes at 21 bits each fill 63 bits, keeping the code within a signed
/// integer's nonnegative range and matching the common LBVH convention.
const MORTON_BITS_PER_AXIS: u32 = 21;
/// Largest quantized coordinate value representable by the Morton grid.
const MORTON_MAX_COORD: u64 = (1_u64 << MORTON_BITS_PER_AXIS) - 1;

/// CPU tree construction strategy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClusterTreeBuildMethod {
    /// Recursively sort each node range by the longest AABB axis, then split
    /// at a dominant adjacent spatial gap on that axis or the median otherwise.
    LongestAxisMedian,
    /// Sort once by Morton code, then split contiguous ranges at a dominant
    /// adjacent Morton-code gap or the median otherwise.
    MortonLbvh,
}

#[derive(Clone, Copy, Debug)]
struct MortonItem {
    /// Interleaved-bit spatial key computed from a representative point.
    code: u64,
    /// Original input geometry index, used as a deterministic tie-breaker.
    input_id: u32,
}

/// CPU-owned finalized binary cluster tree.
///
/// The runtime representation is deliberately a flat set of vectors rather than
/// pointer-linked nodes. Construction is CPU-owned and may allocate, but the
/// resulting slices are suitable for the allocation-free update/evaluation
/// routines.
#[derive(Clone, Debug)]
pub struct ClusterTree<T: DualTreeScalar> {
    /// Axis-aligned bounds for each node.
    pub node_aabb: Vec<Aabb<T>>,
    /// Left child node ID, or `INVALID_INDEX` for leaves.
    pub node_left_child: Vec<u32>,
    /// Right child node ID, or `INVALID_INDEX` for leaves.
    pub node_right_child: Vec<u32>,
    /// Start of this node's contiguous item range in `sorted_indices`.
    pub node_range_start: Vec<u32>,
    /// Number of items covered by this node's subtree.
    pub node_range_count: Vec<u32>,
    /// Start of this leaf's item range, or `INVALID_INDEX` for internal nodes.
    pub leaf_start: Vec<u32>,
    /// Leaf item count, or zero for internal nodes.
    pub leaf_count: Vec<u32>,
    /// Input geometry IDs in tree order. Every node covers a contiguous range.
    pub sorted_indices: Vec<u32>,
    /// Morton code for each sorted input, populated only by Morton/LBVH construction.
    pub sorted_morton_codes: Vec<u64>,
    /// Node IDs for leaves, used to update leaf summaries without scanning all nodes.
    pub leaf_node_ids: Vec<u32>,
    /// Internal node IDs grouped by tree depth from root to leaves.
    pub internal_level_ids: Vec<u32>,
    /// CSR-style offsets into `internal_level_ids` for each internal depth.
    pub internal_level_offsets: Vec<u32>,
    /// Maximum root-to-leaf depth.
    pub max_depth: u32,
}

/// Borrowed view over a finalized cluster tree.
#[derive(Clone, Copy)]
pub struct ClusterTreeView<'a, T: DualTreeScalar> {
    /// Axis-aligned bounds for each node.
    pub node_aabb: &'a [Aabb<T>],
    /// Left child node ID, or `INVALID_INDEX` for leaves.
    pub node_left_child: &'a [u32],
    /// Right child node ID, or `INVALID_INDEX` for leaves.
    pub node_right_child: &'a [u32],
    /// Start of this node's contiguous item range in `sorted_indices`.
    pub node_range_start: &'a [u32],
    /// Number of items covered by this node's subtree.
    pub node_range_count: &'a [u32],
    /// Start of this leaf's item range, or `INVALID_INDEX` for internal nodes.
    pub leaf_start: &'a [u32],
    /// Leaf item count, or zero for internal nodes.
    pub leaf_count: &'a [u32],
    /// Input geometry IDs in tree order. Every node covers a contiguous range.
    pub sorted_indices: &'a [u32],
    /// Morton code for each sorted input, populated only by Morton/LBVH construction.
    pub sorted_morton_codes: &'a [u64],
    /// Node IDs for leaves, used to update leaf summaries without scanning all nodes.
    pub leaf_node_ids: &'a [u32],
    /// Internal node IDs grouped by tree depth from root to leaves.
    pub internal_level_ids: &'a [u32],
    /// CSR-style offsets into `internal_level_ids` for each internal depth.
    pub internal_level_offsets: &'a [u32],
    /// Maximum root-to-leaf depth.
    pub max_depth: u32,
}

impl<T: DualTreeScalar> ClusterTree<T> {
    /// Build a CPU-owned finalized tree using longest-axis hybrid splitting.
    pub fn build<G>(geometry: &[G], leaf_size: usize) -> Result<Self, DualTreeError>
    where
        G: BoundedGeometry<Scalar = T>,
    {
        Self::build_with_method(
            geometry,
            leaf_size,
            ClusterTreeBuildMethod::LongestAxisMedian,
        )
    }

    /// Build a CPU-owned finalized tree using Morton-code LBVH ordering.
    ///
    /// This path computes one Morton code per representative point, sorts the
    /// whole input once, and then builds a binary tree by splitting contiguous
    /// Morton-sorted ranges at dominant adjacent code gaps or the median when
    /// no dominant cluster gap is present. Node AABBs are still computed from
    /// the full bounded geometry, so finite-size sources remain covered.
    pub fn build_morton_lbvh<G>(geometry: &[G], leaf_size: usize) -> Result<Self, DualTreeError>
    where
        G: BoundedGeometry<Scalar = T>,
    {
        Self::build_with_method(geometry, leaf_size, ClusterTreeBuildMethod::MortonLbvh)
    }

    /// Build a CPU-owned finalized tree with the selected construction strategy.
    ///
    /// Both strategies produce the same flat runtime layout and maintain the
    /// invariant that every node owns a contiguous range in `sorted_indices`.
    pub fn build_with_method<G>(
        geometry: &[G],
        leaf_size: usize,
        method: ClusterTreeBuildMethod,
    ) -> Result<Self, DualTreeError>
    where
        G: BoundedGeometry<Scalar = T>,
    {
        if geometry.is_empty() {
            return Err(DualTreeError::EmptyInput);
        }
        if leaf_size == 0 {
            return Err(DualTreeError::InvalidLeafSize);
        }
        if geometry.len() > u32::MAX as usize {
            return Err(DualTreeError::CapacityExceeded);
        }

        let mut tree = Self {
            node_aabb: Vec::new(),
            node_left_child: Vec::new(),
            node_right_child: Vec::new(),
            node_range_start: Vec::new(),
            node_range_count: Vec::new(),
            leaf_start: Vec::new(),
            leaf_count: Vec::new(),
            sorted_indices: Vec::with_capacity(geometry.len()),
            sorted_morton_codes: Vec::new(),
            leaf_node_ids: Vec::new(),
            internal_level_ids: Vec::new(),
            internal_level_offsets: vec![0],
            max_depth: 0,
        };

        for i in 0..geometry.len() {
            tree.sorted_indices.push(usize_to_u32(i)?);
        }

        // LBVH pays one global sort up front. The recursive builder can then
        // split ranges at code gaps or medians without reordering within subtrees.
        if method == ClusterTreeBuildMethod::MortonLbvh {
            tree.sorted_morton_codes = sort_indices_by_morton(&mut tree.sorted_indices, geometry);
        }

        let mut internal_by_depth: Vec<Vec<u32>> = Vec::new();
        match method {
            ClusterTreeBuildMethod::LongestAxisMedian => {
                build_range_longest_axis(
                    &mut tree,
                    geometry,
                    leaf_size,
                    0,
                    geometry.len(),
                    0,
                    &mut internal_by_depth,
                )?;
            }
            ClusterTreeBuildMethod::MortonLbvh => {
                build_range_morton(
                    &mut tree,
                    geometry,
                    leaf_size,
                    0,
                    geometry.len(),
                    0,
                    &mut internal_by_depth,
                )?;
            }
        }

        for depth in 0..internal_by_depth.len() {
            for i in 0..internal_by_depth[depth].len() {
                tree.internal_level_ids.push(internal_by_depth[depth][i]);
            }
            tree.internal_level_offsets
                .push(usize_to_u32(tree.internal_level_ids.len())?);
        }

        Ok(tree)
    }

    /// Borrow this finalized tree as runtime-compatible slices.
    #[inline]
    pub fn as_view(&self) -> ClusterTreeView<'_, T> {
        ClusterTreeView {
            node_aabb: &self.node_aabb,
            node_left_child: &self.node_left_child,
            node_right_child: &self.node_right_child,
            node_range_start: &self.node_range_start,
            node_range_count: &self.node_range_count,
            leaf_start: &self.leaf_start,
            leaf_count: &self.leaf_count,
            sorted_indices: &self.sorted_indices,
            sorted_morton_codes: &self.sorted_morton_codes,
            leaf_node_ids: &self.leaf_node_ids,
            internal_level_ids: &self.internal_level_ids,
            internal_level_offsets: &self.internal_level_offsets,
            max_depth: self.max_depth,
        }
    }
}

impl<T: DualTreeScalar> ClusterTreeView<'_, T> {
    /// Number of nodes in the tree.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.node_aabb.len()
    }

    /// Number of input geometry items covered by the tree.
    #[inline]
    pub fn n_items(&self) -> usize {
        self.sorted_indices.len()
    }

    /// Whether `node_id` refers to a leaf node.
    #[inline]
    pub fn is_leaf(&self, node_id: u32) -> bool {
        self.leaf_count[node_id as usize] > 0
    }

    /// Sentinel child/leaf-start value used in the flat node arrays.
    #[inline]
    pub fn invalid_index() -> u32 {
        INVALID_INDEX
    }
}

/// Recursively build a tree by sorting each range along its longest AABB axis.
///
/// This is the original builder. It tends to produce good axis-aligned spatial
/// splits but performs a sort at every internal node.
fn build_range_longest_axis<T, G>(
    tree: &mut ClusterTree<T>,
    geometry: &[G],
    leaf_size: usize,
    start: usize,
    end: usize,
    depth: usize,
    internal_by_depth: &mut Vec<Vec<u32>>,
) -> Result<u32, DualTreeError>
where
    T: DualTreeScalar,
    G: BoundedGeometry<Scalar = T>,
{
    let node_id = usize_to_u32(tree.node_aabb.len())?;
    let count = end - start;
    let aabb = range_aabb(&tree.sorted_indices, geometry, start, end);

    // Allocate and initialize the node before recursing so child indices can
    // refer back to stable node IDs.
    tree.node_aabb.push(aabb);
    tree.node_left_child.push(INVALID_INDEX);
    tree.node_right_child.push(INVALID_INDEX);
    tree.node_range_start.push(usize_to_u32(start)?);
    tree.node_range_count.push(usize_to_u32(count)?);
    tree.leaf_start.push(INVALID_INDEX);
    tree.leaf_count.push(0);

    if depth > tree.max_depth as usize {
        tree.max_depth = usize_to_u32(depth)?;
    }

    if count <= leaf_size {
        let node = node_id as usize;
        tree.leaf_start[node] = usize_to_u32(start)?;
        tree.leaf_count[node] = usize_to_u32(count)?;
        tree.leaf_node_ids.push(node_id);
        return Ok(node_id);
    }

    // Reorder only this node's contiguous range. Children inherit contiguous
    // subranges, which is required by summary updates and far broadcasts.
    let axis = longest_axis(aabb);
    tree.sorted_indices[start..end].sort_by(|a, b| {
        let pa = geometry[*a as usize].representative_point()[axis];
        let pb = geometry[*b as usize].representative_point()[axis];
        scalar_cmp(pa, pb)
    });

    let mid = hybrid_axis_gap_split(&tree.sorted_indices, geometry, start, end, axis);
    if internal_by_depth.len() <= depth {
        internal_by_depth.resize_with(depth + 1, Vec::new);
    }
    internal_by_depth[depth].push(node_id);

    let left = build_range_longest_axis(
        tree,
        geometry,
        leaf_size,
        start,
        mid,
        depth + 1,
        internal_by_depth,
    )?;
    let right = build_range_longest_axis(
        tree,
        geometry,
        leaf_size,
        mid,
        end,
        depth + 1,
        internal_by_depth,
    )?;

    let node = node_id as usize;
    tree.node_left_child[node] = left;
    tree.node_right_child[node] = right;

    Ok(node_id)
}

/// Recursively build a tree over an already Morton-sorted item range.
///
/// The Morton sort supplies spatial locality. Each internal node splits its
/// range at a dominant adjacent Morton-code gap or the median when no dominant
/// gap exists. This avoids additional per-node sorting while separating obvious
/// code-space clusters without letting ordinary sampling gaps create skinny
/// trees. Internal AABBs are propagated from children after both child subtrees
/// are built.
fn build_range_morton<T, G>(
    tree: &mut ClusterTree<T>,
    geometry: &[G],
    leaf_size: usize,
    start: usize,
    end: usize,
    depth: usize,
    internal_by_depth: &mut Vec<Vec<u32>>,
) -> Result<u32, DualTreeError>
where
    T: DualTreeScalar,
    G: BoundedGeometry<Scalar = T>,
{
    let node_id = usize_to_u32(tree.node_aabb.len())?;
    let count = end - start;

    // Internal Morton nodes get their AABB after child construction. Leaves
    // compute directly from the bounded geometry in their sorted range.
    tree.node_aabb.push(Aabb::empty());
    tree.node_left_child.push(INVALID_INDEX);
    tree.node_right_child.push(INVALID_INDEX);
    tree.node_range_start.push(usize_to_u32(start)?);
    tree.node_range_count.push(usize_to_u32(count)?);
    tree.leaf_start.push(INVALID_INDEX);
    tree.leaf_count.push(0);

    if depth > tree.max_depth as usize {
        tree.max_depth = usize_to_u32(depth)?;
    }

    if count <= leaf_size {
        let node = node_id as usize;
        tree.node_aabb[node] = range_aabb(&tree.sorted_indices, geometry, start, end);
        tree.leaf_start[node] = usize_to_u32(start)?;
        tree.leaf_count[node] = usize_to_u32(count)?;
        tree.leaf_node_ids.push(node_id);
        return Ok(node_id);
    }

    // Split at a dominant adjacent Morton-code gap when one exists; otherwise
    // use the median to keep continuous or uniformly sampled geometry balanced.
    let mid = hybrid_morton_gap_split(tree.sorted_morton_codes.as_slice(), start, end);
    if internal_by_depth.len() <= depth {
        internal_by_depth.resize_with(depth + 1, Vec::new);
    }
    internal_by_depth[depth].push(node_id);

    let left = build_range_morton(
        tree,
        geometry,
        leaf_size,
        start,
        mid,
        depth + 1,
        internal_by_depth,
    )?;
    let right = build_range_morton(
        tree,
        geometry,
        leaf_size,
        mid,
        end,
        depth + 1,
        internal_by_depth,
    )?;

    let node = node_id as usize;
    tree.node_left_child[node] = left;
    tree.node_right_child[node] = right;
    tree.node_aabb[node] = tree.node_aabb[left as usize].union(tree.node_aabb[right as usize]);

    Ok(node_id)
}

/// Sort input IDs by Morton code computed from each item's representative point.
///
/// Ties are resolved by the original input ID so duplicate Morton codes produce
/// deterministic trees. The actual node bounds are still based on `aabb()`, not
/// on representative points.
fn sort_indices_by_morton<T, G>(indices: &mut [u32], geometry: &[G]) -> Vec<u64>
where
    T: DualTreeScalar,
    G: BoundedGeometry<Scalar = T>,
{
    let bounds = representative_point_bounds(indices, geometry);
    let mut items = Vec::with_capacity(indices.len());
    for i in 0..indices.len() {
        let input_id = indices[i];
        let point = geometry[input_id as usize].representative_point();
        items.push(MortonItem {
            code: morton_code(point, bounds),
            input_id,
        });
    }

    items.sort_by(|a, b| match a.code.cmp(&b.code) {
        Ordering::Equal => a.input_id.cmp(&b.input_id),
        order => order,
    });

    for i in 0..items.len() {
        indices[i] = items[i].input_id;
    }
    let mut codes = Vec::with_capacity(items.len());
    for item in items {
        codes.push(item.code);
    }
    codes
}

/// Compute global representative-point bounds for Morton quantization.
///
/// The bounds are intentionally based on representative points because Morton
/// codes order items by a single point key. Full geometry extents are handled
/// separately by leaf and internal AABBs.
fn representative_point_bounds<T, G>(indices: &[u32], geometry: &[G]) -> ([f64; 3], [f64; 3])
where
    T: DualTreeScalar,
    G: BoundedGeometry<Scalar = T>,
{
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for i in 0..indices.len() {
        let point = geometry[indices[i] as usize].representative_point();
        for axis in 0..3 {
            let value = point[axis].to_f64();
            if value < min[axis] {
                min[axis] = value;
            }
            if value > max[axis] {
                max[axis] = value;
            }
        }
    }
    (min, max)
}

/// Compute the 3D Morton code for one representative point.
///
/// Each coordinate is quantized to `MORTON_BITS_PER_AXIS` bits and then the
/// coordinate bits are interleaved as x0, y0, z0, x1, y1, z1, ...
fn morton_code<T: DualTreeScalar>(point: [T; 3], bounds: ([f64; 3], [f64; 3])) -> u64 {
    let mut coords = [0_u64; 3];
    for axis in 0..3 {
        coords[axis] = quantize_morton_coord(point[axis].to_f64(), bounds.0[axis], bounds.1[axis]);
    }

    let mut code = 0_u64;
    for bit in 0..MORTON_BITS_PER_AXIS {
        code |= ((coords[0] >> bit) & 1) << (3 * bit);
        code |= ((coords[1] >> bit) & 1) << (3 * bit + 1);
        code |= ((coords[2] >> bit) & 1) << (3 * bit + 2);
    }
    code
}

/// Map one floating-point coordinate into the Morton integer grid.
///
/// Degenerate or non-finite bounds collapse to zero. That keeps construction
/// deterministic for duplicate representative points and lets the `(code,
/// input_id)` sort tie-breaker preserve total ordering.
fn quantize_morton_coord(value: f64, min: f64, max: f64) -> u64 {
    let extent = max - min;
    if !value.is_finite() || !min.is_finite() || !max.is_finite() || extent <= 0.0 {
        return 0;
    }

    let normalized = ((value - min) / extent).clamp(0.0, 1.0);
    (normalized * MORTON_MAX_COORD as f64).round() as u64
}

/// Compute the AABB covering `sorted_indices[start..end]`.
fn range_aabb<T, G>(indices: &[u32], geometry: &[G], start: usize, end: usize) -> Aabb<T>
where
    T: DualTreeScalar,
    G: BoundedGeometry<Scalar = T>,
{
    let mut out = Aabb::empty();
    for i in start..end {
        out = out.union(geometry[indices[i] as usize].aabb());
    }
    out
}

/// Split a sorted range at a dominant representative-point gap on `axis`.
///
/// The caller has already sorted `indices[start..end]` along the chosen axis.
/// A dominant gap is used only when it is large compared with the range span
/// and the mean adjacent spacing, and when it does not create an extreme
/// imbalance. Otherwise the median split gives smoother behavior for loops,
/// helices, and other continuous source distributions.
fn hybrid_axis_gap_split<T, G>(
    indices: &[u32],
    geometry: &[G],
    start: usize,
    end: usize,
    axis: usize,
) -> usize
where
    T: DualTreeScalar,
    G: BoundedGeometry<Scalar = T>,
{
    let count = end - start;
    let median = start + count / 2;
    let mut split = median;
    let mut best_gap = T::ZERO;
    for i in start + 1..end {
        let left = geometry[indices[i - 1] as usize].representative_point()[axis];
        let right = geometry[indices[i] as usize].representative_point()[axis];
        let gap = right - left;
        if gap > best_gap {
            best_gap = gap;
            split = i;
        }
    }
    if accepts_hybrid_gap_split(
        split,
        start,
        end,
        best_gap.to_f64(),
        spatial_axis_span(indices, geometry, start, end, axis).to_f64(),
        SPATIAL_GAP_DOMINANCE_FACTOR,
        SPATIAL_GAP_MIN_SPAN_FRACTION,
    ) {
        split
    } else {
        median
    }
}

/// Split a Morton-sorted range at a dominant adjacent code gap.
///
/// Equal-code ranges and smoothly sampled ranges fall back to a median split,
/// which keeps degenerate representative points and continuous curves from
/// creating empty, repeatedly identical, or badly imbalanced partitions.
fn hybrid_morton_gap_split(codes: &[u64], start: usize, end: usize) -> usize {
    let count = end - start;
    let median = start + count / 2;
    let mut split = median;
    let mut best_gap = 0_u64;
    for i in start + 1..end {
        let gap = codes[i].saturating_sub(codes[i - 1]);
        if gap > best_gap {
            best_gap = gap;
            split = i;
        }
    }
    if accepts_hybrid_gap_split(
        split,
        start,
        end,
        best_gap as f64,
        codes[end - 1].saturating_sub(codes[start]) as f64,
        MORTON_GAP_DOMINANCE_FACTOR,
        MORTON_GAP_MIN_SPAN_FRACTION,
    ) {
        split
    } else {
        median
    }
}

/// Return the representative-point span along one already-sorted axis.
fn spatial_axis_span<T, G>(
    indices: &[u32],
    geometry: &[G],
    start: usize,
    end: usize,
    axis: usize,
) -> T
where
    T: DualTreeScalar,
    G: BoundedGeometry<Scalar = T>,
{
    let lo = geometry[indices[start] as usize].representative_point()[axis];
    let hi = geometry[indices[end - 1] as usize].representative_point()[axis];
    hi - lo
}

/// Decide whether the largest adjacent gap is meaningful enough to use.
///
/// The dominance check separates true clusters from normal sampling variation,
/// while the balance check keeps one outlier from forcing a long chain of
/// nearly empty siblings.
fn accepts_hybrid_gap_split(
    split: usize,
    start: usize,
    end: usize,
    gap: f64,
    span: f64,
    dominance_factor: f64,
    min_span_fraction: f64,
) -> bool {
    let count = end - start;
    if count <= 2 || span <= 0.0 || gap <= 0.0 {
        return false;
    }

    let left_count = split - start;
    let right_count = end - split;
    let min_side = count / GAP_SPLIT_MIN_SIDE_FRACTION;
    if min_side > 0 && (left_count < min_side || right_count < min_side) {
        return false;
    }

    let mean_gap = span / (count - 1) as f64;
    gap >= dominance_factor * mean_gap && gap >= min_span_fraction * span
}

/// Return the axis with largest AABB extent.
fn longest_axis<T: DualTreeScalar>(aabb: Aabb<T>) -> usize {
    let mut axis = 0;
    let mut extent = aabb.extent(0);
    for candidate in 1..3 {
        let candidate_extent = aabb.extent(candidate);
        if candidate_extent > extent {
            axis = candidate;
            extent = candidate_extent;
        }
    }
    axis
}

/// Total ordering helper for scalar values during construction sorting.
///
/// The tree builder rejects no finite values explicitly, so this avoids relying
/// on `partial_cmp().unwrap()` for scalar types with NaN-like values. Equal and
/// unordered values are treated as equal by falling through to `Ordering::Equal`.
fn scalar_cmp<T: DualTreeScalar>(a: T, b: T) -> Ordering {
    if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// Convert `usize` into the u32 index type used by runtime tree arrays.
pub(crate) fn usize_to_u32(value: usize) -> Result<u32, DualTreeError> {
    if value > u32::MAX as usize {
        Err(DualTreeError::CapacityExceeded)
    } else {
        Ok(value as u32)
    }
}
