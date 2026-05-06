use super::{ClusterTreeView, DualTreeError, DualTreeScalar};

/// CPU-owned finalized dual interaction plan.
#[derive(Clone, Debug, Default)]
pub struct DualInteractionPlan {
    pub near_target_ids: Vec<u32>,
    pub near_source_ids: Vec<u32>,
    pub far_target_node_ids: Vec<u32>,
    pub far_source_node_ids: Vec<u32>,
}

/// Borrowed view over a finalized interaction plan.
#[derive(Clone, Copy)]
pub struct DualInteractionPlanView<'a> {
    pub near_target_ids: &'a [u32],
    pub near_source_ids: &'a [u32],
    pub far_target_node_ids: &'a [u32],
    pub far_source_node_ids: &'a [u32],
}

impl DualInteractionPlan {
    /// Build a dual-tree interaction plan for fixed source/target geometry.
    pub fn build<T: DualTreeScalar>(
        source_tree: ClusterTreeView<'_, T>,
        target_tree: ClusterTreeView<'_, T>,
        theta: T,
    ) -> Result<Self, DualTreeError> {
        if theta < T::ZERO {
            return Err(DualTreeError::InvalidTheta);
        }
        if source_tree.n_nodes() == 0 || target_tree.n_nodes() == 0 {
            return Err(DualTreeError::EmptyInput);
        }

        let mut plan = Self::default();
        let mut active = Vec::new();
        active.push((0_u32, 0_u32));

        while let Some((target_node, source_node)) = active.pop() {
            if is_far(target_tree, source_tree, target_node, source_node, theta) {
                plan.far_target_node_ids.push(target_node);
                plan.far_source_node_ids.push(source_node);
                continue;
            }

            let target_leaf = target_tree.is_leaf(target_node);
            let source_leaf = source_tree.is_leaf(source_node);

            if target_leaf && source_leaf {
                expand_leaf_pair(
                    &mut plan,
                    target_tree,
                    source_tree,
                    target_node,
                    source_node,
                );
                continue;
            }

            let target_diam_sq = target_tree.node_aabb[target_node as usize].diameter_sq();
            let source_diam_sq = source_tree.node_aabb[source_node as usize].diameter_sq();

            let split_target = !target_leaf && (source_leaf || target_diam_sq >= source_diam_sq);
            let split_source = !source_leaf && (target_leaf || source_diam_sq >= target_diam_sq);

            if split_target && split_source {
                push_split_both(
                    &mut active,
                    target_tree,
                    source_tree,
                    target_node,
                    source_node,
                );
            } else if split_target {
                push_split_target(&mut active, target_tree, target_node, source_node);
            } else if split_source {
                push_split_source(&mut active, source_tree, target_node, source_node);
            }
        }

        plan.sort_pairs();

        Ok(plan)
    }

    #[inline]
    pub fn as_view(&self) -> DualInteractionPlanView<'_> {
        DualInteractionPlanView {
            near_target_ids: &self.near_target_ids,
            near_source_ids: &self.near_source_ids,
            far_target_node_ids: &self.far_target_node_ids,
            far_source_node_ids: &self.far_source_node_ids,
        }
    }

    fn sort_pairs(&mut self) {
        let mut near_pairs = Vec::with_capacity(self.near_source_ids.len());
        for i in 0..self.near_source_ids.len() {
            near_pairs.push((self.near_source_ids[i], self.near_target_ids[i]));
        }
        near_pairs.sort_by_key(|pair| pair.0);
        for i in 0..near_pairs.len() {
            self.near_source_ids[i] = near_pairs[i].0;
            self.near_target_ids[i] = near_pairs[i].1;
        }

        let mut far_pairs = Vec::with_capacity(self.far_source_node_ids.len());
        for i in 0..self.far_source_node_ids.len() {
            far_pairs.push((self.far_source_node_ids[i], self.far_target_node_ids[i]));
        }
        far_pairs.sort_by_key(|pair| pair.0);
        for i in 0..far_pairs.len() {
            self.far_source_node_ids[i] = far_pairs[i].0;
            self.far_target_node_ids[i] = far_pairs[i].1;
        }
    }
}

fn is_far<T: DualTreeScalar>(
    target_tree: ClusterTreeView<'_, T>,
    source_tree: ClusterTreeView<'_, T>,
    target_node: u32,
    source_node: u32,
    theta: T,
) -> bool {
    if theta <= T::ZERO {
        return false;
    }

    let target_aabb = target_tree.node_aabb[target_node as usize];
    let source_aabb = source_tree.node_aabb[source_node as usize];
    let gap_sq = target_aabb.gap_distance_sq(&source_aabb);
    if gap_sq <= T::ZERO {
        return false;
    }

    let target_diam = target_aabb.diameter_sq().sqrt();
    let source_diam = source_aabb.diameter_sq().sqrt();
    let combined = target_diam + source_diam;
    gap_sq * theta * theta > combined * combined
}

fn expand_leaf_pair<T: DualTreeScalar>(
    plan: &mut DualInteractionPlan,
    target_tree: ClusterTreeView<'_, T>,
    source_tree: ClusterTreeView<'_, T>,
    target_node: u32,
    source_node: u32,
) {
    let target_start = target_tree.leaf_start[target_node as usize] as usize;
    let target_count = target_tree.leaf_count[target_node as usize] as usize;
    let source_start = source_tree.leaf_start[source_node as usize] as usize;
    let source_count = source_tree.leaf_count[source_node as usize] as usize;

    for ti in 0..target_count {
        let target_id = target_tree.sorted_indices[target_start + ti];
        for si in 0..source_count {
            let source_id = source_tree.sorted_indices[source_start + si];
            plan.near_target_ids.push(target_id);
            plan.near_source_ids.push(source_id);
        }
    }
}

fn push_split_target<T: DualTreeScalar>(
    active: &mut Vec<(u32, u32)>,
    target_tree: ClusterTreeView<'_, T>,
    target_node: u32,
    source_node: u32,
) {
    let left = target_tree.node_left_child[target_node as usize];
    let right = target_tree.node_right_child[target_node as usize];
    if left != ClusterTreeView::<T>::invalid_index() {
        active.push((left, source_node));
    }
    if right != ClusterTreeView::<T>::invalid_index() {
        active.push((right, source_node));
    }
}

fn push_split_source<T: DualTreeScalar>(
    active: &mut Vec<(u32, u32)>,
    source_tree: ClusterTreeView<'_, T>,
    target_node: u32,
    source_node: u32,
) {
    let left = source_tree.node_left_child[source_node as usize];
    let right = source_tree.node_right_child[source_node as usize];
    if left != ClusterTreeView::<T>::invalid_index() {
        active.push((target_node, left));
    }
    if right != ClusterTreeView::<T>::invalid_index() {
        active.push((target_node, right));
    }
}

fn push_split_both<T: DualTreeScalar>(
    active: &mut Vec<(u32, u32)>,
    target_tree: ClusterTreeView<'_, T>,
    source_tree: ClusterTreeView<'_, T>,
    target_node: u32,
    source_node: u32,
) {
    let target_left = target_tree.node_left_child[target_node as usize];
    let target_right = target_tree.node_right_child[target_node as usize];
    let source_left = source_tree.node_left_child[source_node as usize];
    let source_right = source_tree.node_right_child[source_node as usize];
    let targets = [target_left, target_right];
    let sources = [source_left, source_right];

    for ti in 0..2 {
        if targets[ti] == ClusterTreeView::<T>::invalid_index() {
            continue;
        }
        for si in 0..2 {
            if sources[si] == ClusterTreeView::<T>::invalid_index() {
                continue;
            }
            active.push((targets[ti], sources[si]));
        }
    }
}
