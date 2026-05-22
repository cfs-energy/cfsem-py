use core::cmp::Ordering;

use super::{Aabb, BoundedGeometryCollection, HierarchicalError, Scalar};

const INVALID_INDEX: u32 = u32::MAX;
/// Adjacent spatial gap must exceed this multiple of the mean sorted gap before
/// the longest-axis builder treats it as a cluster boundary.
const SPATIAL_GAP_DOMINANCE_FACTOR: f64 = 4.0;
/// Adjacent spatial gap must cover at least this fraction of the node span
/// before the longest-axis builder treats it as a cluster boundary.
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
///
/// Both builders are implemented with explicit stack buffers rather than true
/// recursion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BuildMethod {
    /// Sort each node range by the longest AABB axis, then split at a dominant
    /// adjacent spatial gap on that axis or the median otherwise.
    ///
    /// This method follows a top-down divide-and-conquer tree shape, but uses an
    /// explicit stack internally.
    LongestAxis,
    /// Sort once by Morton code, then split contiguous ranges at a dominant
    /// adjacent Morton-code gap or the median otherwise.
    ///
    /// This method also uses an explicit stack internally.
    MortonLbvh,
}

#[derive(Clone, Copy, Debug)]
struct MortonItem {
    /// Interleaved-bit spatial key computed from a representative point.
    code: u64,
    /// Original input geometry index, used as a deterministic tie-breaker.
    input_id: u32,
}

/// Parent child slot filled by a pending tree-build frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ChildSlot {
    /// Root frame has no parent.
    Root,
    /// Attach the built node as its parent's left child.
    Left,
    /// Attach the built node as its parent's right child.
    Right,
}

/// Explicit stack frame used by the iterative longest-axis builder.
#[derive(Clone, Copy, Debug)]
struct LongestAxisBuildFrame {
    /// Start of the contiguous item range in `sorted_indices`.
    start: usize,
    /// End of the contiguous item range in `sorted_indices`.
    end: usize,
    /// Depth of this node in the tree.
    depth: usize,
    /// Parent node ID, ignored for the root frame.
    parent: u32,
    /// Which child slot of `parent` this frame fills.
    slot: ChildSlot,
}

/// Explicit stack frame used by the iterative Morton builder.
#[derive(Clone, Copy, Debug)]
enum MortonBuildFrame {
    /// Allocate a node and, if internal, schedule its children and finish step.
    Enter {
        /// Start of the contiguous item range in `sorted_indices`.
        start: usize,
        /// End of the contiguous item range in `sorted_indices`.
        end: usize,
        /// Depth of this node in the tree.
        depth: usize,
        /// Parent node ID, ignored for the root frame.
        parent: u32,
        /// Which child slot of `parent` this frame fills.
        slot: ChildSlot,
    },
    /// Finalize an internal node's AABB after both children have been built.
    Finish {
        /// Internal node ID to finalize.
        node_id: u32,
    },
}

/// CPU-owned finalized binary cluster tree.
///
/// The runtime representation is deliberately a flat set of vectors rather than
/// pointer-linked nodes. Construction is CPU-owned and may allocate, but the
/// resulting slices are suitable for the allocation-free update/evaluation
/// routines.
#[derive(Clone, Debug)]
pub struct ClusterTree<T: Scalar> {
    /// Axis-aligned bounds for each node.
    pub(crate) node_aabb: Vec<Aabb<T>>,
    /// Left child node ID, or `INVALID_INDEX` for leaves.
    pub(crate) node_left_child: Vec<u32>,
    /// Right child node ID, or `INVALID_INDEX` for leaves.
    pub(crate) node_right_child: Vec<u32>,
    /// Start of this node's contiguous item range in `sorted_indices`.
    pub(crate) node_range_start: Vec<u32>,
    /// Number of items covered by this node's subtree.
    pub(crate) node_range_count: Vec<u32>,
    /// Start of this leaf's item range, or `INVALID_INDEX` for internal nodes.
    pub(crate) leaf_start: Vec<u32>,
    /// Leaf item count, or zero for internal nodes.
    pub(crate) leaf_count: Vec<u32>,
    /// Input geometry IDs in tree order. Every node covers a contiguous range.
    pub(crate) sorted_indices: Vec<u32>,
    /// Morton code for each sorted input, populated only by Morton/LBVH construction.
    pub(crate) sorted_morton_codes: Vec<u64>,
    /// Node IDs for leaves, used to update leaf summaries without scanning all nodes.
    pub(crate) leaf_node_ids: Vec<u32>,
    /// Internal node IDs grouped by tree depth from root to leaves.
    pub(crate) internal_level_ids: Vec<u32>,
    /// CSR-style offsets into `internal_level_ids` for each internal depth.
    pub(crate) internal_level_offsets: Vec<u32>,
    /// Maximum root-to-leaf depth.
    pub(crate) max_depth: u32,
}

/// Borrowed view over a finalized cluster tree.
#[derive(Clone, Copy)]
pub struct ClusterTreeView<'a, T: Scalar> {
    /// Axis-aligned bounds for each node.
    pub(crate) node_aabb: &'a [Aabb<T>],
    /// Left child node ID, or `INVALID_INDEX` for leaves.
    pub(crate) node_left_child: &'a [u32],
    /// Right child node ID, or `INVALID_INDEX` for leaves.
    pub(crate) node_right_child: &'a [u32],
    /// Start of this node's contiguous item range in `sorted_indices`.
    pub(crate) node_range_start: &'a [u32],
    /// Number of items covered by this node's subtree.
    pub(crate) node_range_count: &'a [u32],
    /// Start of this leaf's item range, or `INVALID_INDEX` for internal nodes.
    pub(crate) leaf_start: &'a [u32],
    /// Leaf item count, or zero for internal nodes.
    pub(crate) leaf_count: &'a [u32],
    /// Input geometry IDs in tree order. Every node covers a contiguous range.
    pub(crate) sorted_indices: &'a [u32],
    /// Morton code for each sorted input, populated only by Morton/LBVH construction.
    pub(crate) sorted_morton_codes: &'a [u64],
    /// Node IDs for leaves, used to update leaf summaries without scanning all nodes.
    pub(crate) leaf_node_ids: &'a [u32],
    /// Internal node IDs grouped by tree depth from root to leaves.
    pub(crate) internal_level_ids: &'a [u32],
    /// CSR-style offsets into `internal_level_ids` for each internal depth.
    pub(crate) internal_level_offsets: &'a [u32],
    /// Maximum root-to-leaf depth.
    pub(crate) max_depth: u32,
}

impl<T: Scalar> ClusterTree<T> {
    /// Build a CPU-owned finalized tree using longest-axis hybrid splitting.
    pub fn build<G>(geometry: G) -> Result<Self, HierarchicalError>
    where
        G: BoundedGeometryCollection<T>,
    {
        Self::build_with_method(geometry, BuildMethod::LongestAxis)
    }

    /// Build a CPU-owned finalized tree using Morton-code LBVH ordering.
    ///
    /// This path computes one Morton code per representative point, sorts the
    /// whole input once, and then builds a binary tree by splitting contiguous
    /// Morton-sorted ranges at dominant adjacent code gaps or the median when
    /// no dominant cluster gap is present. Node AABBs are still computed from
    /// the full bounded geometry, so finite-size sources remain covered.
    pub fn build_morton_lbvh<G>(geometry: G) -> Result<Self, HierarchicalError>
    where
        G: BoundedGeometryCollection<T>,
    {
        Self::build_with_method(geometry, BuildMethod::MortonLbvh)
    }

    /// Build a CPU-owned finalized tree with the selected construction strategy.
    ///
    /// Both strategies produce the same flat runtime layout and maintain the
    /// invariant that every node owns a contiguous range in `sorted_indices`.
    pub fn build_with_method<G>(geometry: G, method: BuildMethod) -> Result<Self, HierarchicalError>
    where
        G: BoundedGeometryCollection<T>,
    {
        if geometry.is_empty() {
            return Err(HierarchicalError::EmptyInput);
        }
        if !geometry.valid_lengths() {
            return Err(HierarchicalError::LengthMismatch);
        }
        if geometry.len() > u32::MAX as usize {
            return Err(HierarchicalError::CapacityExceeded);
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

        // LBVH pays one global sort up front. The explicit-stack builder can
        // then split ranges at code gaps or medians without reordering within
        // subtrees.
        if method == BuildMethod::MortonLbvh {
            tree.sorted_morton_codes = sort_indices_by_morton(&mut tree.sorted_indices, geometry);
        }

        let mut internal_by_depth: Vec<Vec<u32>> = Vec::new();
        match method {
            BuildMethod::LongestAxis => {
                build_range_longest_axis(
                    &mut tree,
                    geometry,
                    0,
                    geometry.len(),
                    &mut internal_by_depth,
                )?;
            }
            BuildMethod::MortonLbvh => {
                build_range_morton(
                    &mut tree,
                    geometry,
                    0,
                    geometry.len(),
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

    /// Axis-aligned bounds for every tree node.
    #[inline]
    pub fn node_aabbs(&self) -> &[Aabb<T>] {
        &self.node_aabb
    }

    /// Left child node ID for every node, or [`ClusterTreeView::invalid_index`] for leaves.
    #[inline]
    pub fn node_left_children(&self) -> &[u32] {
        &self.node_left_child
    }

    /// Right child node ID for every node, or [`ClusterTreeView::invalid_index`] for leaves.
    #[inline]
    pub fn node_right_children(&self) -> &[u32] {
        &self.node_right_child
    }

    /// Start offsets for the sorted item range owned by each node.
    #[inline]
    pub fn node_range_starts(&self) -> &[u32] {
        &self.node_range_start
    }

    /// Item counts for the sorted range owned by each node.
    #[inline]
    pub fn node_range_counts(&self) -> &[u32] {
        &self.node_range_count
    }

    /// Leaf start offsets into [`ClusterTree::sorted_indices`].
    #[inline]
    pub fn leaf_starts(&self) -> &[u32] {
        &self.leaf_start
    }

    /// Leaf item counts. Internal nodes have count zero.
    #[inline]
    pub fn leaf_counts(&self) -> &[u32] {
        &self.leaf_count
    }

    /// Input geometry IDs in tree order.
    #[inline]
    pub fn sorted_indices(&self) -> &[u32] {
        &self.sorted_indices
    }

    /// Morton codes in tree order when this tree was built by the Morton/LBVH method.
    #[inline]
    pub fn sorted_morton_codes(&self) -> &[u64] {
        &self.sorted_morton_codes
    }

    /// Node IDs for leaves.
    #[inline]
    pub fn leaf_node_ids(&self) -> &[u32] {
        &self.leaf_node_ids
    }

    /// Internal node IDs grouped by depth from root to leaves.
    #[inline]
    pub fn internal_level_ids(&self) -> &[u32] {
        &self.internal_level_ids
    }

    /// CSR-style offsets into [`ClusterTree::internal_level_ids`].
    #[inline]
    pub fn internal_level_offsets(&self) -> &[u32] {
        &self.internal_level_offsets
    }

    /// Maximum root-to-leaf depth.
    #[inline]
    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }
}

impl<T: Scalar> ClusterTreeView<'_, T> {
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

    /// Axis-aligned bounds for every tree node.
    #[inline]
    pub fn node_aabbs(&self) -> &[Aabb<T>] {
        self.node_aabb
    }

    /// Left child node ID for every node, or [`ClusterTreeView::invalid_index`] for leaves.
    #[inline]
    pub fn node_left_children(&self) -> &[u32] {
        self.node_left_child
    }

    /// Right child node ID for every node, or [`ClusterTreeView::invalid_index`] for leaves.
    #[inline]
    pub fn node_right_children(&self) -> &[u32] {
        self.node_right_child
    }

    /// Start offsets for the sorted item range owned by each node.
    #[inline]
    pub fn node_range_starts(&self) -> &[u32] {
        self.node_range_start
    }

    /// Item counts for the sorted range owned by each node.
    #[inline]
    pub fn node_range_counts(&self) -> &[u32] {
        self.node_range_count
    }

    /// Leaf start offsets into [`ClusterTreeView::sorted_indices`].
    #[inline]
    pub fn leaf_starts(&self) -> &[u32] {
        self.leaf_start
    }

    /// Leaf item counts. Internal nodes have count zero.
    #[inline]
    pub fn leaf_counts(&self) -> &[u32] {
        self.leaf_count
    }

    /// Input geometry IDs in tree order.
    #[inline]
    pub fn sorted_indices(&self) -> &[u32] {
        self.sorted_indices
    }

    /// Morton codes in tree order when this tree was built by the Morton/LBVH method.
    #[inline]
    pub fn sorted_morton_codes(&self) -> &[u64] {
        self.sorted_morton_codes
    }

    /// Node IDs for leaves.
    #[inline]
    pub fn leaf_node_ids(&self) -> &[u32] {
        self.leaf_node_ids
    }

    /// Internal node IDs grouped by depth from root to leaves.
    #[inline]
    pub fn internal_level_ids(&self) -> &[u32] {
        self.internal_level_ids
    }

    /// CSR-style offsets into [`ClusterTreeView::internal_level_ids`].
    #[inline]
    pub fn internal_level_offsets(&self) -> &[u32] {
        self.internal_level_offsets
    }

    /// Maximum root-to-leaf depth.
    #[inline]
    pub fn max_depth(&self) -> u32 {
        self.max_depth
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

/// Iteratively build a tree by sorting each range along its longest AABB axis.
///
/// This builder tends to produce good axis-aligned spatial splits but performs
/// a sort at every internal node. It uses an explicit stack instead of
/// recursion because each node's AABB can be computed from its full source
/// range before child nodes are constructed.
fn build_range_longest_axis<T, G>(
    tree: &mut ClusterTree<T>,
    geometry: G,
    start: usize,
    end: usize,
    internal_by_depth: &mut Vec<Vec<u32>>,
) -> Result<(), HierarchicalError>
where
    T: Scalar,
    G: BoundedGeometryCollection<T>,
{
    let mut stack = Vec::new();
    stack.push(LongestAxisBuildFrame {
        start,
        end,
        depth: 0,
        parent: INVALID_INDEX,
        slot: ChildSlot::Root,
    });

    while let Some(frame) = stack.pop() {
        let node_id = usize_to_u32(tree.node_aabb.len())?;
        let count = frame.end - frame.start;
        let aabb = range_aabb(&tree.sorted_indices, geometry, frame.start, frame.end);

        // Allocate and initialize the node before pushing child frames so child
        // frames can refer back to this stable node ID.
        tree.node_aabb.push(aabb);
        tree.node_left_child.push(INVALID_INDEX);
        tree.node_right_child.push(INVALID_INDEX);
        tree.node_range_start.push(usize_to_u32(frame.start)?);
        tree.node_range_count.push(usize_to_u32(count)?);
        tree.leaf_start.push(INVALID_INDEX);
        tree.leaf_count.push(0);

        match frame.slot {
            ChildSlot::Root => {}
            ChildSlot::Left => {
                tree.node_left_child[frame.parent as usize] = node_id;
            }
            ChildSlot::Right => {
                tree.node_right_child[frame.parent as usize] = node_id;
            }
        }

        if frame.depth > tree.max_depth as usize {
            tree.max_depth = usize_to_u32(frame.depth)?;
        }

        if count <= 1 {
            let node = node_id as usize;
            tree.leaf_start[node] = usize_to_u32(frame.start)?;
            tree.leaf_count[node] = usize_to_u32(count)?;
            tree.leaf_node_ids.push(node_id);
            continue;
        }

        // Reorder only this node's contiguous range. Children inherit
        // contiguous subranges, which is required by summary updates and far
        // broadcasts.
        let axis = longest_axis(aabb);
        tree.sorted_indices[frame.start..frame.end].sort_by(|a, b| {
            let pa = geometry.representative_point(*a as usize)[axis];
            let pb = geometry.representative_point(*b as usize)[axis];
            scalar_cmp(pa, pb)
        });

        let mid =
            hybrid_axis_gap_split(&tree.sorted_indices, geometry, frame.start, frame.end, axis);
        if internal_by_depth.len() <= frame.depth {
            internal_by_depth.resize_with(frame.depth + 1, Vec::new);
        }
        internal_by_depth[frame.depth].push(node_id);

        let child_depth = frame.depth + 1;
        // Push right first so the LIFO stack visits the left subtree before the
        // right subtree, preserving deterministic left-first node ordering.
        stack.push(LongestAxisBuildFrame {
            start: mid,
            end: frame.end,
            depth: child_depth,
            parent: node_id,
            slot: ChildSlot::Right,
        });
        stack.push(LongestAxisBuildFrame {
            start: frame.start,
            end: mid,
            depth: child_depth,
            parent: node_id,
            slot: ChildSlot::Left,
        });
    }

    Ok(())
}

/// Iteratively build a tree over an already Morton-sorted item range.
///
/// The Morton sort supplies spatial locality. Each internal node splits its
/// range at a dominant adjacent Morton-code gap or the median when no dominant
/// gap exists. This avoids additional per-node sorting while separating obvious
/// code-space clusters without letting ordinary sampling gaps create skinny
/// trees. Internal AABBs are propagated from children in explicit finish
/// frames after both child subtrees are built.
fn build_range_morton<T, G>(
    tree: &mut ClusterTree<T>,
    geometry: G,
    start: usize,
    end: usize,
    internal_by_depth: &mut Vec<Vec<u32>>,
) -> Result<(), HierarchicalError>
where
    T: Scalar,
    G: BoundedGeometryCollection<T>,
{
    let mut stack = Vec::new();
    stack.push(MortonBuildFrame::Enter {
        start,
        end,
        depth: 0,
        parent: INVALID_INDEX,
        slot: ChildSlot::Root,
    });

    while let Some(frame) = stack.pop() {
        match frame {
            MortonBuildFrame::Enter {
                start,
                end,
                depth,
                parent,
                slot,
            } => {
                let node_id = usize_to_u32(tree.node_aabb.len())?;
                let count = end - start;

                // Internal Morton nodes get their AABB in a finish frame after
                // child construction. Leaves compute directly from geometry.
                tree.node_aabb.push(Aabb::empty());
                tree.node_left_child.push(INVALID_INDEX);
                tree.node_right_child.push(INVALID_INDEX);
                tree.node_range_start.push(usize_to_u32(start)?);
                tree.node_range_count.push(usize_to_u32(count)?);
                tree.leaf_start.push(INVALID_INDEX);
                tree.leaf_count.push(0);

                match slot {
                    ChildSlot::Root => {}
                    ChildSlot::Left => {
                        tree.node_left_child[parent as usize] = node_id;
                    }
                    ChildSlot::Right => {
                        tree.node_right_child[parent as usize] = node_id;
                    }
                }

                if depth > tree.max_depth as usize {
                    tree.max_depth = usize_to_u32(depth)?;
                }

                if count <= 1 {
                    let node = node_id as usize;
                    tree.node_aabb[node] = range_aabb(&tree.sorted_indices, geometry, start, end);
                    tree.leaf_start[node] = usize_to_u32(start)?;
                    tree.leaf_count[node] = usize_to_u32(count)?;
                    tree.leaf_node_ids.push(node_id);
                    continue;
                }

                // Split at a dominant adjacent Morton-code gap when one
                // exists; otherwise use the median to keep continuous or
                // uniformly sampled geometry balanced.
                let mid = hybrid_morton_gap_split(tree.sorted_morton_codes.as_slice(), start, end);
                if internal_by_depth.len() <= depth {
                    internal_by_depth.resize_with(depth + 1, Vec::new);
                }
                internal_by_depth[depth].push(node_id);

                let child_depth = depth + 1;
                // Finish must run after both child enter frames, so it is
                // pushed before the children on this LIFO stack. Right is
                // pushed before left to preserve deterministic left-first node
                // ordering.
                stack.push(MortonBuildFrame::Finish { node_id });
                stack.push(MortonBuildFrame::Enter {
                    start: mid,
                    end,
                    depth: child_depth,
                    parent: node_id,
                    slot: ChildSlot::Right,
                });
                stack.push(MortonBuildFrame::Enter {
                    start,
                    end: mid,
                    depth: child_depth,
                    parent: node_id,
                    slot: ChildSlot::Left,
                });
            }
            MortonBuildFrame::Finish { node_id } => {
                let node = node_id as usize;
                let left = tree.node_left_child[node] as usize;
                let right = tree.node_right_child[node] as usize;
                tree.node_aabb[node] = tree.node_aabb[left].union(tree.node_aabb[right]);
            }
        }
    }

    Ok(())
}

/// Sort input IDs by Morton code computed from each item's representative point.
///
/// Ties are resolved by the original input ID so duplicate Morton codes produce
/// deterministic trees. The actual node bounds are still based on `aabb()`, not
/// on representative points.
fn sort_indices_by_morton<T, G>(indices: &mut [u32], geometry: G) -> Vec<u64>
where
    T: Scalar,
    G: BoundedGeometryCollection<T>,
{
    let bounds = representative_point_bounds(indices, geometry);
    let mut items = Vec::with_capacity(indices.len());
    for i in 0..indices.len() {
        let input_id = indices[i];
        let point = geometry.representative_point(input_id as usize);
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
fn representative_point_bounds<T, G>(indices: &[u32], geometry: G) -> ([f64; 3], [f64; 3])
where
    T: Scalar,
    G: BoundedGeometryCollection<T>,
{
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for i in 0..indices.len() {
        let point = geometry.representative_point(indices[i] as usize);
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
fn morton_code<T: Scalar>(point: [T; 3], bounds: ([f64; 3], [f64; 3])) -> u64 {
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
fn range_aabb<T, G>(indices: &[u32], geometry: G, start: usize, end: usize) -> Aabb<T>
where
    T: Scalar,
    G: BoundedGeometryCollection<T>,
{
    let mut out = Aabb::empty();
    for i in start..end {
        out = out.union(geometry.aabb(indices[i] as usize));
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
    geometry: G,
    start: usize,
    end: usize,
    axis: usize,
) -> usize
where
    T: Scalar,
    G: BoundedGeometryCollection<T>,
{
    let count = end - start;
    let median = start + count / 2;
    let mut split = median;
    let mut best_gap = T::ZERO;
    for i in start + 1..end {
        let left = geometry.representative_point(indices[i - 1] as usize)[axis];
        let right = geometry.representative_point(indices[i] as usize)[axis];
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
fn spatial_axis_span<T, G>(indices: &[u32], geometry: G, start: usize, end: usize, axis: usize) -> T
where
    T: Scalar,
    G: BoundedGeometryCollection<T>,
{
    let lo = geometry.representative_point(indices[start] as usize)[axis];
    let hi = geometry.representative_point(indices[end - 1] as usize)[axis];
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
fn longest_axis<T: Scalar>(aabb: Aabb<T>) -> usize {
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
fn scalar_cmp<T: Scalar>(a: T, b: T) -> Ordering {
    if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// Convert `usize` into the u32 index type used by runtime tree arrays.
pub(crate) fn usize_to_u32(value: usize) -> Result<u32, HierarchicalError> {
    if value > u32::MAX as usize {
        Err(HierarchicalError::CapacityExceeded)
    } else {
        Ok(value as u32)
    }
}
