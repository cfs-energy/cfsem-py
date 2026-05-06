use core::cmp::Ordering;

use super::{Aabb, BoundedGeometry, DualTreeError, DualTreeScalar};

const INVALID_INDEX: u32 = u32::MAX;

/// CPU-owned finalized binary cluster tree.
#[derive(Clone, Debug)]
pub struct ClusterTree<T: DualTreeScalar> {
    pub node_aabb: Vec<Aabb<T>>,
    pub node_left_child: Vec<u32>,
    pub node_right_child: Vec<u32>,
    pub node_range_start: Vec<u32>,
    pub node_range_count: Vec<u32>,
    pub leaf_start: Vec<u32>,
    pub leaf_count: Vec<u32>,
    pub sorted_indices: Vec<u32>,
    pub leaf_node_ids: Vec<u32>,
    pub internal_level_ids: Vec<u32>,
    pub internal_level_offsets: Vec<u32>,
    pub max_depth: u32,
}

/// Borrowed view over a finalized cluster tree.
#[derive(Clone, Copy)]
pub struct ClusterTreeView<'a, T: DualTreeScalar> {
    pub node_aabb: &'a [Aabb<T>],
    pub node_left_child: &'a [u32],
    pub node_right_child: &'a [u32],
    pub node_range_start: &'a [u32],
    pub node_range_count: &'a [u32],
    pub leaf_start: &'a [u32],
    pub leaf_count: &'a [u32],
    pub sorted_indices: &'a [u32],
    pub leaf_node_ids: &'a [u32],
    pub internal_level_ids: &'a [u32],
    pub internal_level_offsets: &'a [u32],
    pub max_depth: u32,
}

impl<T: DualTreeScalar> ClusterTree<T> {
    /// Build a CPU-owned finalized tree using longest-axis median splitting.
    pub fn build<G>(geometry: &[G], leaf_size: usize) -> Result<Self, DualTreeError>
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
            leaf_node_ids: Vec::new(),
            internal_level_ids: Vec::new(),
            internal_level_offsets: vec![0],
            max_depth: 0,
        };

        for i in 0..geometry.len() {
            tree.sorted_indices.push(usize_to_u32(i)?);
        }

        let mut internal_by_depth: Vec<Vec<u32>> = Vec::new();
        build_range(
            &mut tree,
            geometry,
            leaf_size,
            0,
            geometry.len(),
            0,
            &mut internal_by_depth,
        )?;

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
            leaf_node_ids: &self.leaf_node_ids,
            internal_level_ids: &self.internal_level_ids,
            internal_level_offsets: &self.internal_level_offsets,
            max_depth: self.max_depth,
        }
    }
}

impl<T: DualTreeScalar> ClusterTreeView<'_, T> {
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.node_aabb.len()
    }

    #[inline]
    pub fn n_items(&self) -> usize {
        self.sorted_indices.len()
    }

    #[inline]
    pub fn is_leaf(&self, node_id: u32) -> bool {
        self.leaf_count[node_id as usize] > 0
    }

    #[inline]
    pub fn invalid_index() -> u32 {
        INVALID_INDEX
    }
}

fn build_range<T, G>(
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

    let axis = longest_axis(aabb);
    tree.sorted_indices[start..end].sort_by(|a, b| {
        let pa = geometry[*a as usize].representative_point()[axis];
        let pb = geometry[*b as usize].representative_point()[axis];
        scalar_cmp(pa, pb)
    });

    let mid = start + count / 2;
    if internal_by_depth.len() <= depth {
        internal_by_depth.resize_with(depth + 1, Vec::new);
    }
    internal_by_depth[depth].push(node_id);

    let left = build_range(
        tree,
        geometry,
        leaf_size,
        start,
        mid,
        depth + 1,
        internal_by_depth,
    )?;
    let right = build_range(
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

fn scalar_cmp<T: DualTreeScalar>(a: T, b: T) -> Ordering {
    if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

pub(crate) fn usize_to_u32(value: usize) -> Result<u32, DualTreeError> {
    if value > u32::MAX as usize {
        Err(DualTreeError::CapacityExceeded)
    } else {
        Ok(value as u32)
    }
}
