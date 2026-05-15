#[cfg(test)]
use super::plan::{DualInteractionPlanChunk, DualInteractionPlanView};
use super::{BoundedGeometry, ClusterTreeView, DualTreeError, DualTreeKernel};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

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
#[cfg(test)]
pub struct TargetNodeSummaries<K: DualTreeKernel> {
    pub node_summaries: Vec<K::TargetSummary>,
    pub chunk_offsets: Vec<u32>,
}

#[cfg(test)]
impl<K: DualTreeKernel> TargetNodeSummaries<K> {
    #[inline]
    pub fn new_for_plan(plan: DualInteractionPlanView<'_, K::Scalar>) -> Self {
        let mut chunk_offsets = Vec::with_capacity(plan.chunks.len() + 1);
        chunk_offsets.push(0);
        let mut total = 0;
        for chunk_id in 0..plan.chunks.len() {
            total += plan.chunks[chunk_id].target_tree.as_view().n_nodes();
            chunk_offsets.push(total as u32);
        }
        Self {
            node_summaries: vec![K::TargetSummary::default(); total],
            chunk_offsets,
        }
    }

    #[inline]
    fn chunk_slice(&self, chunk_id: usize) -> &[K::TargetSummary] {
        let start = self.chunk_offsets[chunk_id] as usize;
        let end = self.chunk_offsets[chunk_id + 1] as usize;
        &self.node_summaries[start..end]
    }

    #[inline]
    fn chunk_slice_mut(&mut self, chunk_id: usize) -> &mut [K::TargetSummary] {
        let start = self.chunk_offsets[chunk_id] as usize;
        let end = self.chunk_offsets[chunk_id + 1] as usize;
        &mut self.node_summaries[start..end]
    }
}

/// Scratch storage for exact and far contribution evaluation.
pub struct EvaluationScratch<'a, O> {
    pub contribution: &'a mut [O],
}

/// Number of output entries required by a plan.
#[cfg(test)]
#[inline]
pub fn output_len<T: super::DualTreeScalar>(plan: DualInteractionPlanView<'_, T>) -> usize {
    plan.target_count
}

/// Number of contribution scratch entries required by serial evaluation.
#[cfg(test)]
#[inline]
pub fn serial_evaluation_scratch_len<T: super::DualTreeScalar>(
    _plan: DualInteractionPlanView<'_, T>,
) -> usize {
    1
}

/// Number of contribution scratch entries required by parallel evaluation.
#[cfg(test)]
#[inline]
pub fn parallel_evaluation_scratch_len<T: super::DualTreeScalar>(
    plan: DualInteractionPlanView<'_, T>,
) -> usize {
    plan.chunks.len()
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
pub fn update_source_summaries_into<K: DualTreeKernel>(
    kernel: &K,
    tree: ClusterTreeView<'_, K::Scalar>,
    sources: &[K::SourceGeometry],
    moments: &[K::SourceMoment],
    summaries: &mut [K::SourceSummary],
) -> DualTreeError {
    let err = validate_source_tree_layout(tree);
    if err != DualTreeError::Ok {
        return err;
    }
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
        let end = start + count;
        let source_ids = &tree.sorted_indices[start..end];
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
#[cfg(test)]
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

/// Update all target summaries owned by a chunked interaction plan.
#[cfg(test)]
#[inline]
pub fn update_plan_target_summaries_into<K: DualTreeKernel>(
    kernel: &K,
    plan: DualInteractionPlanView<'_, K::Scalar>,
    targets: &[K::TargetGeometry],
    summaries: &mut TargetNodeSummaries<K>,
) -> DualTreeError {
    if targets.len() != plan.target_count || summaries.chunk_offsets.len() != plan.chunks.len() + 1
    {
        return DualTreeError::LengthMismatch;
    }
    for chunk_id in 0..plan.chunks.len() {
        let chunk = &plan.chunks[chunk_id];
        let target_start = chunk.target_start;
        let target_end = target_start + chunk.target_count;
        let err = update_target_summaries_into(
            kernel,
            chunk.target_tree.as_view(),
            &targets[target_start..target_end],
            summaries.chunk_slice_mut(chunk_id),
        );
        if err != DualTreeError::Ok {
            return err;
        }
    }
    DualTreeError::Ok
}

/// Evaluate the Barnes-Hut plan into `out`.
#[cfg(test)]
#[inline]
pub fn evaluate_into<K: DualTreeKernel>(
    kernel: &K,
    plan: DualInteractionPlanView<'_, K::Scalar>,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    target_summaries: &TargetNodeSummaries<K>,
    sources: &[K::SourceGeometry],
    targets: &[K::TargetGeometry],
    moments: &[K::SourceMoment],
    out: &mut [K::Output],
    scratch: &mut EvaluationScratch<'_, K::Output>,
) -> DualTreeError {
    if sources.len() != source_tree.n_items()
        || targets.len() != plan.target_count
        || moments.len() != source_tree.n_items()
        || out.len() != plan.target_count
    {
        return DualTreeError::LengthMismatch;
    }
    if source_summaries.len() < source_tree.n_nodes() {
        return DualTreeError::ScratchTooSmall;
    }
    let err = validate_target_summaries(plan, target_summaries);
    if err != DualTreeError::Ok {
        return err;
    }
    if scratch.contribution.is_empty() {
        return DualTreeError::ScratchTooSmall;
    }

    for i in 0..out.len() {
        kernel.zero_output(&mut out[i]);
    }

    for chunk_id in 0..plan.chunks.len() {
        let chunk = &plan.chunks[chunk_id];
        let target_start = chunk.target_start;
        let target_end = target_start + chunk.target_count;
        let err = evaluate_chunk_into(
            kernel,
            chunk,
            source_tree,
            source_summaries,
            target_summaries.chunk_slice(chunk_id),
            sources,
            &targets[target_start..target_end],
            moments,
            &mut out[target_start..target_end],
            &mut scratch.contribution[0],
        );
        if err != DualTreeError::Ok {
            return err;
        }
    }

    DualTreeError::Ok
}

/// Evaluate the Barnes-Hut plan into `out` in parallel over target chunks.
#[cfg(test)]
#[inline]
pub fn evaluate_into_par<K: DualTreeKernel + Sync>(
    kernel: &K,
    plan: DualInteractionPlanView<'_, K::Scalar>,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    target_summaries: &TargetNodeSummaries<K>,
    sources: &[K::SourceGeometry],
    targets: &[K::TargetGeometry],
    moments: &[K::SourceMoment],
    out: &mut [K::Output],
    scratch: &mut EvaluationScratch<'_, K::Output>,
) -> DualTreeError {
    if sources.len() != source_tree.n_items()
        || targets.len() != plan.target_count
        || moments.len() != source_tree.n_items()
        || out.len() != plan.target_count
    {
        return DualTreeError::LengthMismatch;
    }
    if source_summaries.len() < source_tree.n_nodes() {
        return DualTreeError::ScratchTooSmall;
    }
    let err = validate_target_summaries(plan, target_summaries);
    if err != DualTreeError::Ok {
        return err;
    }
    if scratch.contribution.len() < plan.chunks.len() {
        return DualTreeError::ScratchTooSmall;
    }

    let error_code = AtomicU32::new(DualTreeError::Ok as u32);

    (
        plan.chunks.par_iter(),
        out.par_chunks_mut(plan.target_chunk_size),
        scratch.contribution[..plan.chunks.len()].par_iter_mut(),
    )
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_id, (chunk, out_chunk, contribution))| {
            if error_code.load(Ordering::Relaxed) != DualTreeError::Ok as u32 {
                return;
            }
            let target_start = chunk.target_start;
            let target_end = target_start + chunk.target_count;
            let err = evaluate_chunk_into(
                kernel,
                chunk,
                source_tree,
                source_summaries,
                target_summaries.chunk_slice(chunk_id),
                sources,
                &targets[target_start..target_end],
                moments,
                &mut out_chunk[..chunk.target_count],
                contribution,
            );
            if err != DualTreeError::Ok {
                let _ = error_code.compare_exchange(
                    DualTreeError::Ok as u32,
                    err as u32,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
            }
        });

    DualTreeError::from_u32(error_code.load(Ordering::Relaxed))
}

/// Evaluate targets independently against the source tree.
///
/// This is the public hierarchical evaluation path. Each target is summarized
/// as a single target leaf, then walked against the source tree using the same
/// source-side acceptance criterion as the lower-level interaction-plan
/// evaluator.
#[inline]
pub fn evaluate_source_tree_into<K: DualTreeKernel>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    sources: &[K::SourceGeometry],
    targets: &[K::TargetGeometry],
    moments: &[K::SourceMoment],
    theta: K::Scalar,
    out: &mut [K::Output],
    scratch: &mut EvaluationScratch<'_, K::Output>,
) -> DualTreeError {
    let err = validate_source_tree_layout(source_tree);
    if err != DualTreeError::Ok {
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
fn evaluate_source_tree_into_validated<K: DualTreeKernel>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    sources: &[K::SourceGeometry],
    targets: &[K::TargetGeometry],
    moments: &[K::SourceMoment],
    theta: K::Scalar,
    out: &mut [K::Output],
    scratch: &mut EvaluationScratch<'_, K::Output>,
) -> DualTreeError {
    if sources.len() != source_tree.n_items()
        || moments.len() != source_tree.n_items()
        || targets.len() != out.len()
    {
        return DualTreeError::LengthMismatch;
    }
    if source_summaries.len() < source_tree.n_nodes() || scratch.contribution.is_empty() {
        return DualTreeError::ScratchTooSmall;
    }

    let mut target_summary = K::TargetSummary::default();
    let mut active = Vec::new();
    let target_ids = [0_u32];

    for target_id in 0..targets.len() {
        // SAFETY: `targets.len() == out.len()` is checked above, and this loop
        // only visits indices in `0..targets.len()`.
        let target = unsafe { targets.get_unchecked(target_id) };
        // SAFETY: same bound as `target`; each target index is visited once.
        let target_out = unsafe { out.get_unchecked_mut(target_id) };
        kernel.zero_output(target_out);
        let err = kernel.summarize_leaf_targets(
            &target_ids,
            core::slice::from_ref(target),
            &mut target_summary,
        );
        if err != DualTreeError::Ok {
            return err;
        }

        active.clear();
        active.push(0_u32);
        while let Some(source_node) = active.pop() {
            let source_node_index = source_node as usize;
            // SAFETY: `validate_source_tree_layout` checks the root, children,
            // and leaf node ids pushed into `active`; `source_summaries` length
            // is checked against `source_tree.n_nodes()` above.
            let source_summary = unsafe { source_summaries.get_unchecked(source_node_index) };
            // SAFETY: same validated node bound as `source_summary`.
            let source_aabb = unsafe { *source_tree.node_aabb.get_unchecked(source_node_index) };
            if kernel.accept_far(target.aabb(), source_aabb, source_summary, theta) {
                let err = kernel.eval_far(
                    &target_summary,
                    source_summary,
                    &mut scratch.contribution[0],
                );
                if err != DualTreeError::Ok {
                    return err;
                }
                kernel.accumulate(target_out, &scratch.contribution[0]);
                continue;
            }

            // SAFETY: same validated node bound as `source_summary`.
            let leaf_count = unsafe { *source_tree.leaf_count.get_unchecked(source_node_index) };
            if leaf_count > 0 {
                // SAFETY: same validated node bound as `source_summary`.
                let start =
                    unsafe { *source_tree.leaf_start.get_unchecked(source_node_index) } as usize;
                let count = leaf_count as usize;
                let end = start + count;
                // SAFETY: `validate_source_tree_layout` checks every leaf
                // range against `sorted_indices.len()`.
                let source_ids = unsafe { source_tree.sorted_indices.get_unchecked(start..end) };
                for i in 0..source_ids.len() {
                    // SAFETY: this loop is bounded by `source_ids.len()`.
                    let source_id = unsafe { *source_ids.get_unchecked(i) } as usize;
                    // SAFETY: sorted source ids are checked to be less than
                    // `source_tree.n_items()`, and `sources`/`moments` lengths
                    // are checked against `source_tree.n_items()` above.
                    let source = unsafe { sources.get_unchecked(source_id) };
                    // SAFETY: same source-id bound as `source`.
                    let moment = unsafe { moments.get_unchecked(source_id) };
                    let err =
                        kernel.eval_exact(target, source, moment, &mut scratch.contribution[0]);
                    if err != DualTreeError::Ok {
                        return err;
                    }
                    kernel.accumulate(target_out, &scratch.contribution[0]);
                }
            } else {
                // SAFETY: same validated node bound as `source_summary`.
                active
                    .push(unsafe { *source_tree.node_left_child.get_unchecked(source_node_index) });
                // SAFETY: same validated node bound as `source_summary`.
                active.push(unsafe {
                    *source_tree
                        .node_right_child
                        .get_unchecked(source_node_index)
                });
            }
        }
    }

    DualTreeError::Ok
}

/// Evaluate targets independently against the source tree in parallel over target chunks.
///
/// This is intentionally the simplest parallelization of the single-tree
/// solver: each worker owns a disjoint target/output slice and runs the serial
/// source-tree evaluator on that slice. It shares the source tree and source
/// summaries between workers, and avoids any cross-thread output accumulation.
#[inline]
pub fn evaluate_source_tree_into_par<K: DualTreeKernel + Sync>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    sources: &[K::SourceGeometry],
    targets: &[K::TargetGeometry],
    moments: &[K::SourceMoment],
    theta: K::Scalar,
    out: &mut [K::Output],
    scratch: &mut EvaluationScratch<'_, K::Output>,
) -> DualTreeError {
    let err = validate_source_tree_layout(source_tree);
    if err != DualTreeError::Ok {
        return err;
    }
    if sources.len() != source_tree.n_items()
        || moments.len() != source_tree.n_items()
        || targets.len() != out.len()
    {
        return DualTreeError::LengthMismatch;
    }
    if source_summaries.len() < source_tree.n_nodes() {
        return DualTreeError::ScratchTooSmall;
    }
    if targets.is_empty() {
        return DualTreeError::Ok;
    }

    let chunk_size = crate::chunksize(targets.len());
    let chunk_count = targets.len().div_ceil(chunk_size);
    if scratch.contribution.len() < chunk_count {
        return DualTreeError::ScratchTooSmall;
    }

    let error_code = AtomicU32::new(DualTreeError::Ok as u32);

    (
        targets.par_chunks(chunk_size),
        out.par_chunks_mut(chunk_size),
        scratch.contribution[..chunk_count].par_iter_mut(),
    )
        .into_par_iter()
        .for_each(|(target_chunk, out_chunk, contribution)| {
            if error_code.load(Ordering::Relaxed) != DualTreeError::Ok as u32 {
                return;
            }
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
            if err != DualTreeError::Ok {
                let _ = error_code.compare_exchange(
                    DualTreeError::Ok as u32,
                    err as u32,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
            }
        });

    DualTreeError::from_u32(error_code.load(Ordering::Relaxed))
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
pub fn accepted_source_level_diagnostic_into<K: DualTreeKernel>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    targets: &[K::TargetGeometry],
    theta: K::Scalar,
    out: &mut [f64],
) -> DualTreeError {
    let err = validate_source_tree_layout(source_tree);
    if err != DualTreeError::Ok {
        return err;
    }
    if targets.len() != out.len() {
        return DualTreeError::LengthMismatch;
    }
    if source_summaries.len() < source_tree.n_nodes() {
        return DualTreeError::ScratchTooSmall;
    }

    let mut active = Vec::new();
    for target_id in 0..targets.len() {
        // SAFETY: this loop only visits indices in `0..targets.len()`.
        let target = unsafe { targets.get_unchecked(target_id) };
        let mut weighted_level = 0.0_f64;
        let mut represented_sources = 0.0_f64;

        active.clear();
        active.push((0_u32, 0_u32));
        while let Some((source_node, source_level)) = active.pop() {
            let source_node_index = source_node as usize;
            // SAFETY: `validate_source_tree_layout` checks every node pushed
            // into the traversal and every node-range array.
            let source_count = unsafe {
                *source_tree
                    .node_range_count
                    .get_unchecked(source_node_index)
            } as f64;
            // SAFETY: `source_summaries` length is checked against the tree node count above.
            let source_summary = unsafe { source_summaries.get_unchecked(source_node_index) };
            // SAFETY: same validated node bound as `source_count`.
            let source_aabb = unsafe { *source_tree.node_aabb.get_unchecked(source_node_index) };
            if kernel.accept_far(target.aabb(), source_aabb, source_summary, theta) {
                weighted_level += f64::from(source_level) * source_count;
                represented_sources += source_count;
                continue;
            }

            // SAFETY: same validated node bound as `source_count`.
            let leaf_count = unsafe { *source_tree.leaf_count.get_unchecked(source_node_index) };
            if leaf_count > 0 {
                weighted_level += f64::from(source_level) * source_count;
                represented_sources += source_count;
            } else {
                let next_level = source_level + 1;
                // SAFETY: same validated node bound as `source_count`.
                active.push((
                    unsafe { *source_tree.node_left_child.get_unchecked(source_node_index) },
                    next_level,
                ));
                // SAFETY: same validated node bound as `source_count`.
                active.push((
                    unsafe {
                        *source_tree
                            .node_right_child
                            .get_unchecked(source_node_index)
                    },
                    next_level,
                ));
            }
        }

        out[target_id] = if represented_sources > 0.0 {
            weighted_level / represented_sources
        } else {
            f64::NAN
        };
    }

    DualTreeError::Ok
}

#[cfg(test)]
#[inline]
fn validate_target_summaries<K: DualTreeKernel>(
    plan: DualInteractionPlanView<'_, K::Scalar>,
    summaries: &TargetNodeSummaries<K>,
) -> DualTreeError {
    if summaries.chunk_offsets.len() != plan.chunks.len() + 1 {
        return DualTreeError::LengthMismatch;
    }
    if summaries.chunk_offsets.is_empty() {
        return DualTreeError::LengthMismatch;
    }
    let last = summaries.chunk_offsets[summaries.chunk_offsets.len() - 1] as usize;
    if last > summaries.node_summaries.len() {
        return DualTreeError::ScratchTooSmall;
    }
    DualTreeError::Ok
}

#[cfg(test)]
#[inline]
fn evaluate_chunk_into<K: DualTreeKernel>(
    kernel: &K,
    chunk: &DualInteractionPlanChunk<K::Scalar>,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    target_summaries: &[K::TargetSummary],
    sources: &[K::SourceGeometry],
    targets: &[K::TargetGeometry],
    moments: &[K::SourceMoment],
    out: &mut [K::Output],
    contribution: &mut K::Output,
) -> DualTreeError {
    let target_tree = chunk.target_tree.as_view();
    if targets.len() != target_tree.n_items()
        || out.len() != target_tree.n_items()
        || chunk.near_target_ids.len() != chunk.near_source_ids.len()
        || chunk.far_target_node_ids.len() != chunk.far_source_node_ids.len()
        || target_summaries.len() < target_tree.n_nodes()
        || source_summaries.len() < source_tree.n_nodes()
    {
        return DualTreeError::LengthMismatch;
    }

    for i in 0..out.len() {
        kernel.zero_output(&mut out[i]);
    }

    for pair in 0..chunk.near_target_ids.len() {
        let target_id = chunk.near_target_ids[pair] as usize;
        let source_id = chunk.near_source_ids[pair] as usize;
        let err = kernel.eval_exact(
            &targets[target_id],
            &sources[source_id],
            &moments[source_id],
            contribution,
        );
        if err != DualTreeError::Ok {
            return err;
        }
        kernel.accumulate(&mut out[target_id], contribution);
    }

    for pair in 0..chunk.far_target_node_ids.len() {
        let target_node = chunk.far_target_node_ids[pair] as usize;
        let source_node = chunk.far_source_node_ids[pair] as usize;
        let err = kernel.eval_far(
            &target_summaries[target_node],
            &source_summaries[source_node],
            contribution,
        );
        if err != DualTreeError::Ok {
            return err;
        }

        let start = target_tree.node_range_start[target_node] as usize;
        let count = target_tree.node_range_count[target_node] as usize;
        for i in 0..count {
            let target_id = target_tree.sorted_indices[start + i] as usize;
            kernel.accumulate(&mut out[target_id], contribution);
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
        let target = &targets[target_id];
        let target_out = &mut out[target_id];
        for source_id in 0..sources.len() {
            let err = kernel.eval_exact(
                target,
                &sources[source_id],
                &moments[source_id],
                &mut scratch.contribution[0],
            );
            if err != DualTreeError::Ok {
                return err;
            }
            kernel.accumulate(target_out, &scratch.contribution[0]);
        }
    }

    DualTreeError::Ok
}

/// Validate the flat source-tree layout before entering hot traversal loops.
///
/// This converts malformed tree views into error codes up front instead of
/// relying on slice indexing panics in the evaluator. `ClusterTree::as_view`
/// already satisfies these invariants; this guard mainly protects borrowed
/// views passed across API boundaries or future GPU-compatible wrappers.
#[inline]
fn validate_source_tree_layout<T: super::DualTreeScalar>(
    tree: ClusterTreeView<'_, T>,
) -> DualTreeError {
    let n_nodes = tree.n_nodes();
    if n_nodes == 0 {
        return DualTreeError::EmptyInput;
    }
    if tree.node_left_child.len() != n_nodes
        || tree.node_right_child.len() != n_nodes
        || tree.node_range_start.len() != n_nodes
        || tree.node_range_count.len() != n_nodes
        || tree.leaf_start.len() != n_nodes
        || tree.leaf_count.len() != n_nodes
    {
        return DualTreeError::LengthMismatch;
    }

    for i in 0..tree.sorted_indices.len() {
        if tree.sorted_indices[i] as usize >= tree.sorted_indices.len() {
            return DualTreeError::LengthMismatch;
        }
    }

    for node_id in 0..n_nodes {
        let start = tree.node_range_start[node_id] as usize;
        let count = tree.node_range_count[node_id] as usize;
        if count == 0
            || start > tree.sorted_indices.len()
            || count > tree.sorted_indices.len() - start
        {
            return DualTreeError::LengthMismatch;
        }
    }

    for i in 0..tree.leaf_node_ids.len() {
        let node_id = tree.leaf_node_ids[i] as usize;
        if node_id >= n_nodes {
            return DualTreeError::LengthMismatch;
        }
        let start = tree.leaf_start[node_id] as usize;
        let count = tree.leaf_count[node_id] as usize;
        if count == 0
            || start > tree.sorted_indices.len()
            || count > tree.sorted_indices.len() - start
        {
            return DualTreeError::LengthMismatch;
        }
    }

    if !tree.internal_level_offsets.is_empty() {
        let mut previous = 0_usize;
        for i in 0..tree.internal_level_offsets.len() {
            let offset = tree.internal_level_offsets[i] as usize;
            if offset < previous || offset > tree.internal_level_ids.len() {
                return DualTreeError::LengthMismatch;
            }
            previous = offset;
        }
    }

    for i in 0..tree.internal_level_ids.len() {
        let node_id = tree.internal_level_ids[i] as usize;
        if node_id >= n_nodes {
            return DualTreeError::LengthMismatch;
        }
        let left = tree.node_left_child[node_id] as usize;
        let right = tree.node_right_child[node_id] as usize;
        if left >= n_nodes || right >= n_nodes {
            return DualTreeError::LengthMismatch;
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

#[cfg(test)]
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
