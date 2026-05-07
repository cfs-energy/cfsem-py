use super::{
    ClusterTreeView, DualInteractionPlanChunk, DualInteractionPlanView, DualTreeError,
    DualTreeKernel,
};
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
pub struct TargetNodeSummaries<K: DualTreeKernel> {
    pub node_summaries: Vec<K::TargetSummary>,
    pub chunk_offsets: Vec<u32>,
}

impl<K: DualTreeKernel> TargetNodeSummaries<K> {
    #[inline]
    pub fn new(tree: ClusterTreeView<'_, K::Scalar>) -> Self {
        Self {
            node_summaries: vec![K::TargetSummary::default(); tree.n_nodes()],
            chunk_offsets: vec![0, tree.n_nodes() as u32],
        }
    }

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
#[inline]
pub fn output_len<T: super::DualTreeScalar>(plan: DualInteractionPlanView<'_, T>) -> usize {
    plan.target_count
}

/// Number of contribution scratch entries required by serial evaluation.
#[inline]
pub fn serial_evaluation_scratch_len<T: super::DualTreeScalar>(
    _plan: DualInteractionPlanView<'_, T>,
) -> usize {
    1
}

/// Number of contribution scratch entries required by parallel evaluation.
#[inline]
pub fn parallel_evaluation_scratch_len<T: super::DualTreeScalar>(
    plan: DualInteractionPlanView<'_, T>,
) -> usize {
    plan.chunks.len()
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

/// Update all target summaries owned by a chunked interaction plan.
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
