use super::{ClusterTreeView, DualInteractionPlanView, DualTreeError, DualTreeKernel};

/// CPU-owned source summary storage.
pub struct SourceNodeSummaries<K: DualTreeKernel> {
    pub node_summaries: Vec<K::SourceSummary>,
}

impl<K: DualTreeKernel> SourceNodeSummaries<K> {
    #[inline]
    pub fn new(tree: ClusterTreeView<'_, K::Scalar>) -> Self {
        Self {
            node_summaries: vec![K::SourceSummary::default(); tree.n_nodes()],
        }
    }
}

/// CPU-owned target summary storage.
pub struct TargetNodeSummaries<K: DualTreeKernel> {
    pub node_summaries: Vec<K::TargetSummary>,
}

impl<K: DualTreeKernel> TargetNodeSummaries<K> {
    #[inline]
    pub fn new(tree: ClusterTreeView<'_, K::Scalar>) -> Self {
        Self {
            node_summaries: vec![K::TargetSummary::default(); tree.n_nodes()],
        }
    }
}

/// Scratch storage for exact and far contribution evaluation.
pub struct EvaluationScratch<'a, O> {
    pub contribution: &'a mut [O],
}

/// Update source summaries for a fixed source tree and changed source moments.
#[inline]
pub fn update_source_summaries_into<K: DualTreeKernel>(
    kernel: &K,
    tree: ClusterTreeView<'_, K::Scalar>,
    sources: &[K::SourceGeometry],
    moments: &[K::SourceMoment],
    summaries: &mut [K::SourceSummary],
) -> DualTreeError {
    if sources.len() != tree.n_items() || moments.len() != tree.n_items() {
        return DualTreeError::LengthMismatch;
    }
    if summaries.len() < tree.n_nodes() {
        return DualTreeError::ScratchTooSmall;
    }

    for i in 0..tree.leaf_node_ids.len() {
        let node_id = tree.leaf_node_ids[i];
        let start = tree.leaf_start[node_id as usize] as usize;
        let count = tree.leaf_count[node_id as usize] as usize;
        let source_ids = &tree.sorted_indices[start..start + count];
        let err = kernel.summarize_leaf_sources(
            source_ids,
            sources,
            moments,
            &mut summaries[node_id as usize],
        );
        if err != DualTreeError::Ok {
            return err;
        }
    }

    propagate_source_summaries(kernel, tree, summaries)
}

/// Update target summaries for fixed target geometry.
#[inline]
pub fn update_target_summaries_into<K: DualTreeKernel>(
    kernel: &K,
    tree: ClusterTreeView<'_, K::Scalar>,
    targets: &[K::TargetGeometry],
    summaries: &mut [K::TargetSummary],
) -> DualTreeError {
    if targets.len() != tree.n_items() {
        return DualTreeError::LengthMismatch;
    }
    if summaries.len() < tree.n_nodes() {
        return DualTreeError::ScratchTooSmall;
    }

    for i in 0..tree.leaf_node_ids.len() {
        let node_id = tree.leaf_node_ids[i];
        let start = tree.leaf_start[node_id as usize] as usize;
        let count = tree.leaf_count[node_id as usize] as usize;
        let target_ids = &tree.sorted_indices[start..start + count];
        let err =
            kernel.summarize_leaf_targets(target_ids, targets, &mut summaries[node_id as usize]);
        if err != DualTreeError::Ok {
            return err;
        }
    }

    propagate_target_summaries(kernel, tree, summaries)
}

/// Evaluate the Barnes-Hut plan into `out`.
#[inline]
pub fn evaluate_into<K: DualTreeKernel>(
    kernel: &K,
    plan: DualInteractionPlanView<'_>,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    target_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    target_summaries: &[K::TargetSummary],
    sources: &[K::SourceGeometry],
    targets: &[K::TargetGeometry],
    moments: &[K::SourceMoment],
    out: &mut [K::Output],
    scratch: &mut EvaluationScratch<'_, K::Output>,
) -> DualTreeError {
    if sources.len() != source_tree.n_items()
        || targets.len() != target_tree.n_items()
        || moments.len() != source_tree.n_items()
        || out.len() != target_tree.n_items()
        || plan.near_target_ids.len() != plan.near_source_ids.len()
        || plan.far_target_node_ids.len() != plan.far_source_node_ids.len()
    {
        return DualTreeError::LengthMismatch;
    }
    if source_summaries.len() < source_tree.n_nodes()
        || target_summaries.len() < target_tree.n_nodes()
    {
        return DualTreeError::ScratchTooSmall;
    }
    if scratch.contribution.is_empty() {
        return DualTreeError::ScratchTooSmall;
    }

    for i in 0..out.len() {
        kernel.zero_output(&mut out[i]);
    }

    for pair in 0..plan.near_target_ids.len() {
        let target_id = plan.near_target_ids[pair] as usize;
        let source_id = plan.near_source_ids[pair] as usize;
        let err = kernel.eval_exact(
            &targets[target_id],
            &sources[source_id],
            &moments[source_id],
            &mut scratch.contribution[0],
        );
        if err != DualTreeError::Ok {
            return err;
        }
        kernel.accumulate(&mut out[target_id], &scratch.contribution[0]);
    }

    for pair in 0..plan.far_target_node_ids.len() {
        let target_node = plan.far_target_node_ids[pair] as usize;
        let source_node = plan.far_source_node_ids[pair] as usize;
        let err = kernel.eval_far(
            &target_summaries[target_node],
            &source_summaries[source_node],
            &mut scratch.contribution[0],
        );
        if err != DualTreeError::Ok {
            return err;
        }

        let start = target_tree.node_range_start[target_node] as usize;
        let count = target_tree.node_range_count[target_node] as usize;
        for i in 0..count {
            let target_id = target_tree.sorted_indices[start + i] as usize;
            kernel.accumulate(&mut out[target_id], &scratch.contribution[0]);
        }
    }

    DualTreeError::Ok
}

/// Dense exact fallback using nested range loops.
#[inline]
pub fn dense_direct_evaluate_into<K: DualTreeKernel>(
    kernel: &K,
    sources: &[K::SourceGeometry],
    targets: &[K::TargetGeometry],
    moments: &[K::SourceMoment],
    out: &mut [K::Output],
    scratch: &mut EvaluationScratch<'_, K::Output>,
) -> DualTreeError {
    if sources.len() != moments.len() || targets.len() != out.len() {
        return DualTreeError::LengthMismatch;
    }
    if scratch.contribution.is_empty() {
        return DualTreeError::ScratchTooSmall;
    }

    for i in 0..out.len() {
        kernel.zero_output(&mut out[i]);
    }

    for target_id in 0..targets.len() {
        for source_id in 0..sources.len() {
            let err = kernel.eval_exact(
                &targets[target_id],
                &sources[source_id],
                &moments[source_id],
                &mut scratch.contribution[0],
            );
            if err != DualTreeError::Ok {
                return err;
            }
            kernel.accumulate(&mut out[target_id], &scratch.contribution[0]);
        }
    }

    DualTreeError::Ok
}

#[inline]
fn propagate_source_summaries<K: DualTreeKernel>(
    kernel: &K,
    tree: ClusterTreeView<'_, K::Scalar>,
    summaries: &mut [K::SourceSummary],
) -> DualTreeError {
    if tree.internal_level_offsets.is_empty() {
        return DualTreeError::Ok;
    }

    let n_levels = tree.internal_level_offsets.len() - 1;
    for level_rev in 0..n_levels {
        let level = n_levels - 1 - level_rev;
        let start = tree.internal_level_offsets[level] as usize;
        let end = tree.internal_level_offsets[level + 1] as usize;
        for i in start..end {
            let node_id = tree.internal_level_ids[i];
            let left = tree.node_left_child[node_id as usize];
            let right = tree.node_right_child[node_id as usize];
            let child_ids = [left, right];
            let children = [summaries[left as usize], summaries[right as usize]];
            let err = kernel.combine_source_summaries(
                &children,
                &child_ids,
                &mut summaries[node_id as usize],
            );
            if err != DualTreeError::Ok {
                return err;
            }
        }
    }

    DualTreeError::Ok
}

#[inline]
fn propagate_target_summaries<K: DualTreeKernel>(
    kernel: &K,
    tree: ClusterTreeView<'_, K::Scalar>,
    summaries: &mut [K::TargetSummary],
) -> DualTreeError {
    if tree.internal_level_offsets.is_empty() {
        return DualTreeError::Ok;
    }

    let n_levels = tree.internal_level_offsets.len() - 1;
    for level_rev in 0..n_levels {
        let level = n_levels - 1 - level_rev;
        let start = tree.internal_level_offsets[level] as usize;
        let end = tree.internal_level_offsets[level + 1] as usize;
        for i in start..end {
            let node_id = tree.internal_level_ids[i];
            let left = tree.node_left_child[node_id as usize];
            let right = tree.node_right_child[node_id as usize];
            let child_ids = [left, right];
            let children = [summaries[left as usize], summaries[right as usize]];
            let err = kernel.combine_target_summaries(
                &children,
                &child_ids,
                &mut summaries[node_id as usize],
            );
            if err != DualTreeError::Ok {
                return err;
            }
        }
    }

    DualTreeError::Ok
}
