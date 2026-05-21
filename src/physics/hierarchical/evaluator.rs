use super::{
    BoundedGeometry, ClusterTreeView, HierarchicalError, HierarchicalKernel, TargetCollection,
};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

/// CPU-owned source summary storage.
pub struct SourceNodeSummaries<K: HierarchicalKernel> {
    pub node_summaries: Vec<K::SourceSummary>,
}

impl<K: HierarchicalKernel> SourceNodeSummaries<K> {
    #[inline]
    pub fn new(tree: ClusterTreeView<'_, K::Scalar>) -> Self {
        Self {
            node_summaries: vec![K::SourceSummary::default(); tree.n_nodes()],
        }
    }
}

/// Scratch storage for exact and far contribution evaluation.
pub struct EvaluationScratch<'a, O> {
    pub contribution: &'a mut [O],
}

/// Number of contribution scratch entries required by source-tree-only evaluation.
#[inline]
pub fn source_tree_evaluation_scratch_len() -> usize {
    1
}

/// Number of contribution scratch entries required by parallel source-tree evaluation.
#[inline]
pub fn parallel_source_tree_evaluation_scratch_len(target_count: usize) -> usize {
    let chunk_size = crate::chunksize(target_count);
    target_count.div_ceil(chunk_size).max(1)
}

/// Update source summaries for a fixed source tree and changed source moments.
#[inline]
pub fn update_source_summaries_into<K: HierarchicalKernel>(
    kernel: &K,
    tree: ClusterTreeView<'_, K::Scalar>,
    sources: &[K::SourceGeometry],
    moments: &[K::SourceMoment],
    summaries: &mut [K::SourceSummary],
) -> HierarchicalError {
    let err = validate_source_tree_layout(tree);
    if err != HierarchicalError::Ok {
        return err;
    }
    if sources.len() != tree.n_items() || moments.len() != tree.n_items() {
        return HierarchicalError::LengthMismatch;
    }
    if summaries.len() < tree.n_nodes() {
        return HierarchicalError::ScratchTooSmall;
    }

    for i in 0..tree.leaf_node_ids.len() {
        let node_id = tree.leaf_node_ids[i];
        let start = tree.leaf_start[node_id as usize] as usize;
        let count = tree.leaf_count[node_id as usize] as usize;
        let end = start + count;
        let source_ids = &tree.sorted_indices[start..end];
        let err = kernel.summarize_leaf_sources(
            source_ids,
            sources,
            moments,
            &mut summaries[node_id as usize],
        );
        if err != HierarchicalError::Ok {
            return err;
        }
    }

    propagate_source_summaries(kernel, tree, summaries)
}

/// Evaluate targets independently against the source tree.
///
/// This is the public hierarchical evaluation path. Each target is summarized
/// as a single target leaf, then walked against the source tree using the same
/// source-side acceptance criterion as the lower-level interaction-plan
/// evaluator.
#[inline]
pub fn evaluate_source_tree_into<K, C>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    sources: &[K::SourceGeometry],
    targets: C,
    moments: &[K::SourceMoment],
    theta: K::Scalar,
    out: &mut [K::Output],
    scratch: &mut EvaluationScratch<'_, K::Output>,
) -> HierarchicalError
where
    K: HierarchicalKernel,
    K::TargetGeometry: Copy,
    C: TargetCollection<K>,
{
    let err = validate_source_tree_layout(source_tree);
    if err != HierarchicalError::Ok {
        return err;
    }
    evaluate_source_tree_into_validated(
        kernel,
        source_tree,
        source_summaries,
        sources,
        targets,
        moments,
        theta,
        out,
        scratch,
    )
}

#[inline]
fn evaluate_source_tree_into_validated<K, C>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    sources: &[K::SourceGeometry],
    targets: C,
    moments: &[K::SourceMoment],
    theta: K::Scalar,
    out: &mut [K::Output],
    scratch: &mut EvaluationScratch<'_, K::Output>,
) -> HierarchicalError
where
    K: HierarchicalKernel,
    K::TargetGeometry: Copy,
    C: TargetCollection<K>,
{
    if sources.len() != source_tree.n_items()
        || moments.len() != source_tree.n_items()
        || targets.len() != out.len()
        || !targets.has_consistent_lengths()
    {
        return HierarchicalError::LengthMismatch;
    }
    if source_summaries.len() < source_tree.n_nodes() || scratch.contribution.is_empty() {
        return HierarchicalError::ScratchTooSmall;
    }

    let mut target_summary = K::TargetSummary::default();
    let mut active = Vec::new();
    let target_ids = [0_u32];

    for target_id in 0..targets.len() {
        let target = targets.target(target_id);
        let err = evaluate_source_tree_scalar(
            kernel,
            source_tree,
            source_summaries,
            sources,
            target,
            moments,
            theta,
            &mut out[target_id],
            &mut scratch.contribution[0],
            &mut target_summary,
            &mut active,
            &target_ids,
        );
        if err != HierarchicalError::Ok {
            return err;
        }
    }

    HierarchicalError::Ok
}

/// Evaluate one scalar target against the source tree.
///
/// Serial and parallel vector evaluators both call this helper so the source
/// traversal and acceptance behavior cannot diverge between evaluation modes.
#[inline]
fn evaluate_source_tree_scalar<K>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    sources: &[K::SourceGeometry],
    target: K::TargetGeometry,
    moments: &[K::SourceMoment],
    theta: K::Scalar,
    out: &mut K::Output,
    contribution: &mut K::Output,
    target_summary: &mut K::TargetSummary,
    active: &mut Vec<u32>,
    target_ids: &[u32],
) -> HierarchicalError
where
    K: HierarchicalKernel,
    K::TargetGeometry: Copy,
{
    kernel.zero_output(out);
    let err =
        kernel.summarize_leaf_targets(target_ids, core::slice::from_ref(&target), target_summary);
    if err != HierarchicalError::Ok {
        return err;
    }

    active.clear();
    active.push(0_u32);
    while let Some(source_node) = active.pop() {
        let source_node_index = source_node as usize;
        let source_summary = &source_summaries[source_node_index];
        let source_aabb = source_tree.node_aabb[source_node_index];
        if kernel.accept_far(target.aabb(), source_aabb, source_summary, theta) {
            kernel.eval_far(target_summary, source_summary, contribution);
            kernel.accumulate(out, contribution);
            continue;
        }

        let leaf_count = source_tree.leaf_count[source_node_index];
        if leaf_count > 0 {
            let start = source_tree.leaf_start[source_node_index] as usize;
            let count = leaf_count as usize;
            let end = start + count;
            let source_ids = &source_tree.sorted_indices[start..end];
            for i in 0..source_ids.len() {
                let source_id = source_ids[i] as usize;
                kernel.eval_exact(
                    &target,
                    &sources[source_id],
                    &moments[source_id],
                    contribution,
                );
                kernel.accumulate(out, contribution);
            }
        } else {
            active.push(source_tree.node_left_child[source_node_index]);
            active.push(source_tree.node_right_child[source_node_index]);
        }
    }

    HierarchicalError::Ok
}

/// Evaluate targets independently against the source tree in parallel over target chunks.
///
/// This is intentionally the simplest parallelization of the single-tree
/// solver: each worker owns a disjoint target/output slice and runs the serial
/// source-tree evaluator on that slice. It shares the source tree and source
/// summaries between workers, and avoids any cross-thread output accumulation.
#[inline]
pub fn evaluate_source_tree_into_par<K, C>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    sources: &[K::SourceGeometry],
    targets: C,
    moments: &[K::SourceMoment],
    theta: K::Scalar,
    out: &mut [K::Output],
    scratch: &mut EvaluationScratch<'_, K::Output>,
) -> HierarchicalError
where
    K: HierarchicalKernel + Sync,
    K::TargetGeometry: Copy,
    C: TargetCollection<K>,
{
    let err = validate_source_tree_layout(source_tree);
    if err != HierarchicalError::Ok {
        return err;
    }
    if sources.len() != source_tree.n_items()
        || moments.len() != source_tree.n_items()
        || targets.len() != out.len()
        || !targets.has_consistent_lengths()
    {
        return HierarchicalError::LengthMismatch;
    }
    if source_summaries.len() < source_tree.n_nodes() {
        return HierarchicalError::ScratchTooSmall;
    }
    if targets.is_empty() {
        return HierarchicalError::Ok;
    }

    let chunk_size = crate::chunksize(targets.len());
    let chunk_count = targets.len().div_ceil(chunk_size);
    if scratch.contribution.len() < chunk_count {
        return HierarchicalError::ScratchTooSmall;
    }

    let error_code = AtomicU32::new(HierarchicalError::Ok as u32);

    (
        (0..chunk_count).into_par_iter(),
        out.par_chunks_mut(chunk_size),
        scratch.contribution[..chunk_count].par_iter_mut(),
    )
        .into_par_iter()
        .for_each(|(chunk_id, out_chunk, contribution)| {
            if error_code.load(Ordering::Relaxed) != HierarchicalError::Ok as u32 {
                return;
            }
            let start = chunk_id * chunk_size;
            let end = start + out_chunk.len();
            let target_chunk = targets.slice(start, end);
            let mut chunk_scratch = EvaluationScratch {
                contribution: core::slice::from_mut(contribution),
            };
            let err = evaluate_source_tree_into_validated(
                kernel,
                source_tree,
                source_summaries,
                sources,
                target_chunk,
                moments,
                theta,
                out_chunk,
                &mut chunk_scratch,
            );
            if err != HierarchicalError::Ok {
                let _ = error_code.compare_exchange(
                    HierarchicalError::Ok as u32,
                    err as u32,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
            }
        });

    HierarchicalError::from_u32(error_code.load(Ordering::Relaxed))
}

/// Compute the source-tree level represented at each target by the terminal traversal nodes.
///
/// This is a diagnostic companion to [`evaluate_source_tree_into`]. It mirrors
/// the same source-tree walk but does not evaluate field values. Far-accepted
/// nodes contribute their traversal depth, while direct leaf fallbacks
/// contribute the leaf depth. Each contribution is weighted by the number of
/// original source items represented by that terminal node, giving a per-target
/// mean accepted source level.
#[inline]
pub fn accepted_source_level_diagnostic_into<K, C>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    targets: C,
    theta: K::Scalar,
    out: &mut [f64],
) -> HierarchicalError
where
    K: HierarchicalKernel,
    K::TargetGeometry: Copy,
    C: TargetCollection<K>,
{
    let err = validate_source_tree_layout(source_tree);
    if err != HierarchicalError::Ok {
        return err;
    }
    if targets.len() != out.len() || !targets.has_consistent_lengths() {
        return HierarchicalError::LengthMismatch;
    }
    if source_summaries.len() < source_tree.n_nodes() {
        return HierarchicalError::ScratchTooSmall;
    }

    let mut active = Vec::new();
    for target_id in 0..targets.len() {
        let target = targets.target(target_id);
        let mut weighted_level = 0.0_f64;
        let mut represented_sources = 0.0_f64;

        active.clear();
        active.push((0_u32, 0_u32));
        while let Some((source_node, source_level)) = active.pop() {
            let source_node_index = source_node as usize;
            let source_count = source_tree.node_range_count[source_node_index] as f64;
            let source_summary = &source_summaries[source_node_index];
            let source_aabb = source_tree.node_aabb[source_node_index];
            if kernel.accept_far(target.aabb(), source_aabb, source_summary, theta) {
                weighted_level += f64::from(source_level) * source_count;
                represented_sources += source_count;
                continue;
            }

            let leaf_count = source_tree.leaf_count[source_node_index];
            if leaf_count > 0 {
                weighted_level += f64::from(source_level) * source_count;
                represented_sources += source_count;
            } else {
                let next_level = source_level + 1;
                active.push((source_tree.node_left_child[source_node_index], next_level));
                active.push((source_tree.node_right_child[source_node_index], next_level));
            }
        }

        out[target_id] = if represented_sources > 0.0 {
            weighted_level / represented_sources
        } else {
            f64::NAN
        };
    }

    HierarchicalError::Ok
}

/// Dense exact fallback using nested range loops.
#[inline]
pub fn dense_direct_evaluate_into<K: HierarchicalKernel>(
    kernel: &K,
    sources: &[K::SourceGeometry],
    targets: &[K::TargetGeometry],
    moments: &[K::SourceMoment],
    out: &mut [K::Output],
    scratch: &mut EvaluationScratch<'_, K::Output>,
) -> HierarchicalError {
    if sources.len() != moments.len() || targets.len() != out.len() {
        return HierarchicalError::LengthMismatch;
    }
    if scratch.contribution.is_empty() {
        return HierarchicalError::ScratchTooSmall;
    }

    for i in 0..out.len() {
        kernel.zero_output(&mut out[i]);
    }

    for target_id in 0..targets.len() {
        let target = &targets[target_id];
        let target_out = &mut out[target_id];
        for source_id in 0..sources.len() {
            kernel.eval_exact(
                target,
                &sources[source_id],
                &moments[source_id],
                &mut scratch.contribution[0],
            );
            kernel.accumulate(target_out, &scratch.contribution[0]);
        }
    }

    HierarchicalError::Ok
}

/// Validate the flat source-tree layout before entering hot traversal loops.
///
/// This converts malformed tree views into error codes up front instead of
/// relying on slice indexing panics in the evaluator. `ClusterTree::as_view`
/// already satisfies these invariants; this guard mainly protects borrowed
/// views passed across API boundaries or future GPU-compatible wrappers.
#[inline]
fn validate_source_tree_layout<T: super::Scalar>(
    tree: ClusterTreeView<'_, T>,
) -> HierarchicalError {
    let n_nodes = tree.n_nodes();
    if n_nodes == 0 {
        return HierarchicalError::EmptyInput;
    }
    if tree.node_left_child.len() != n_nodes
        || tree.node_right_child.len() != n_nodes
        || tree.node_range_start.len() != n_nodes
        || tree.node_range_count.len() != n_nodes
        || tree.leaf_start.len() != n_nodes
        || tree.leaf_count.len() != n_nodes
    {
        return HierarchicalError::LengthMismatch;
    }

    for i in 0..tree.sorted_indices.len() {
        if tree.sorted_indices[i] as usize >= tree.sorted_indices.len() {
            return HierarchicalError::LengthMismatch;
        }
    }

    for node_id in 0..n_nodes {
        let start = tree.node_range_start[node_id] as usize;
        let count = tree.node_range_count[node_id] as usize;
        if count == 0
            || start > tree.sorted_indices.len()
            || count > tree.sorted_indices.len() - start
        {
            return HierarchicalError::LengthMismatch;
        }
        if tree.leaf_count[node_id] == 0 {
            let left = tree.node_left_child[node_id] as usize;
            let right = tree.node_right_child[node_id] as usize;
            if left >= n_nodes || right >= n_nodes {
                return HierarchicalError::LengthMismatch;
            }
        }
    }

    for i in 0..tree.leaf_node_ids.len() {
        let node_id = tree.leaf_node_ids[i] as usize;
        if node_id >= n_nodes {
            return HierarchicalError::LengthMismatch;
        }
        let start = tree.leaf_start[node_id] as usize;
        let count = tree.leaf_count[node_id] as usize;
        if count == 0
            || start > tree.sorted_indices.len()
            || count > tree.sorted_indices.len() - start
        {
            return HierarchicalError::LengthMismatch;
        }
    }

    if !tree.internal_level_offsets.is_empty() {
        let mut previous = 0_usize;
        for i in 0..tree.internal_level_offsets.len() {
            let offset = tree.internal_level_offsets[i] as usize;
            if offset < previous || offset > tree.internal_level_ids.len() {
                return HierarchicalError::LengthMismatch;
            }
            previous = offset;
        }
    }

    for i in 0..tree.internal_level_ids.len() {
        let node_id = tree.internal_level_ids[i] as usize;
        if node_id >= n_nodes {
            return HierarchicalError::LengthMismatch;
        }
        let left = tree.node_left_child[node_id] as usize;
        let right = tree.node_right_child[node_id] as usize;
        if left >= n_nodes || right >= n_nodes {
            return HierarchicalError::LengthMismatch;
        }
    }

    HierarchicalError::Ok
}

#[inline]
fn propagate_source_summaries<K: HierarchicalKernel>(
    kernel: &K,
    tree: ClusterTreeView<'_, K::Scalar>,
    summaries: &mut [K::SourceSummary],
) -> HierarchicalError {
    if tree.internal_level_offsets.is_empty() {
        return HierarchicalError::Ok;
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
            if err != HierarchicalError::Ok {
                return err;
            }
        }
    }

    HierarchicalError::Ok
}
