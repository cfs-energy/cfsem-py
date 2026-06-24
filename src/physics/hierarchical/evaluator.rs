use super::{
    BoundedGeometry, ClusterTreeView, HierarchicalError, HierarchicalKernel, Scalar,
    SourceCollection, SourceMomentCollection, TargetCollection,
};
use std::sync::atomic::{AtomicU32, Ordering};

/// CPU-owned source summary storage.
pub struct SourceNodeSummaries<K: HierarchicalKernel> {
    pub node_summaries: Vec<K::SourceSummary>,
}

impl<K: HierarchicalKernel> SourceNodeSummaries<K> {
    #[inline]
    /// Construct the default kernel value.
    pub fn new(tree: ClusterTreeView<'_, K::Scalar>) -> Self {
        Self {
            node_summaries: vec![K::SourceSummary::default(); tree.n_nodes()],
        }
    }
}

/// Scratch storage for exact and far contribution evaluation.
pub struct EvaluationScratch<'a, Output> {
    pub contribution: &'a mut [Output],
}

/// Number of contribution scratch entries required by source-tree-only evaluation.
#[inline]
pub fn scratch_len() -> usize {
    1
}

/// Number of contribution scratch entries required by parallel source-tree evaluation.
#[inline]
pub fn scratch_len_par(target_count: usize) -> usize {
    let chunk_size = crate::chunksize(target_count);
    target_count.div_ceil(chunk_size).max(1)
}

/// Update source summaries for a fixed source tree and changed source moments.
#[inline]
pub fn update_summaries<K, S, M>(
    kernel: &K,
    tree: ClusterTreeView<'_, K::Scalar>,
    sources: S,
    moments: M,
    summaries: &mut [K::SourceSummary],
) -> HierarchicalError
where
    K: HierarchicalKernel,
    S: SourceCollection<K>,
    M: SourceMomentCollection<K>,
{
    let err = validate_source_tree_layout(tree);
    if err != HierarchicalError::Ok {
        return err;
    }
    if sources.len() != tree.n_items()
        || !sources.valid_lengths()
        || moments.len() != tree.n_items()
        || !moments.valid_lengths()
    {
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

/// Evaluate vector-valued targets independently against the source tree.
///
/// This is the public hierarchical evaluation path. Each target is summarized
/// as a single target leaf, walked against the source tree, and written directly
/// into caller-provided component slices. The output slice count must match the
/// kernel output dimension `D`.
#[inline]
pub fn eval<K, T, S, M, C, const D: usize>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, T>,
    source_summaries: &[K::SourceSummary],
    sources: S,
    targets: C,
    moments: M,
    theta: T,
    out: [&mut [T]; D],
    scratch: &mut EvaluationScratch<'_, [T; D]>,
) -> HierarchicalError
where
    K: HierarchicalKernel<Scalar = T, Output = [T; D]>,
    T: Scalar,
    K::TargetGeometry: Copy,
    S: SourceCollection<K>,
    M: SourceMomentCollection<K>,
    C: TargetCollection<K>,
{
    let err = validate_source_tree_layout(source_tree);
    if err != HierarchicalError::Ok {
        return err;
    }
    eval_validated(
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
/// Evaluate validated source-target rows with the hierarchical tree walk.
fn eval_validated<K, T, S, M, C, const D: usize>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, T>,
    source_summaries: &[K::SourceSummary],
    sources: S,
    targets: C,
    moments: M,
    theta: T,
    out: [&mut [T]; D],
    scratch: &mut EvaluationScratch<'_, [T; D]>,
) -> HierarchicalError
where
    K: HierarchicalKernel<Scalar = T, Output = [T; D]>,
    T: Scalar,
    K::TargetGeometry: Copy,
    S: SourceCollection<K>,
    M: SourceMomentCollection<K>,
    C: TargetCollection<K>,
{
    if D == 0
        || sources.len() != source_tree.n_items()
        || !sources.valid_lengths()
        || moments.len() != source_tree.n_items()
        || !moments.valid_lengths()
        || !targets.valid_lengths()
    {
        return HierarchicalError::LengthMismatch;
    }
    for component in 0..D {
        if out[component].len() != targets.len() {
            return HierarchicalError::LengthMismatch;
        }
    }
    if source_summaries.len() < source_tree.n_nodes() || scratch.contribution.is_empty() {
        return HierarchicalError::ScratchTooSmall;
    }

    let mut target_summary = K::TargetSummary::default();
    let mut active = Vec::new();
    let target_ids = [0_u32];
    let mut target_out = [T::ZERO; D];

    for target_id in 0..targets.len() {
        let target = targets.target(target_id);
        let err = eval_scalar(
            kernel,
            source_tree,
            source_summaries,
            sources,
            target,
            moments,
            theta,
            &mut target_out,
            &mut scratch.contribution[0],
            &mut target_summary,
            &mut active,
            &target_ids,
        );
        if err != HierarchicalError::Ok {
            return err;
        }
        for component in 0..D {
            out[component][target_id] = target_out[component];
        }
    }

    HierarchicalError::Ok
}

/// Evaluate one scalar target against the source tree.
///
/// Serial and parallel vector evaluators both call this helper so the source
/// traversal and acceptance behavior cannot diverge between evaluation modes.
#[inline]
fn eval_scalar<K, S, M>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    sources: S,
    target: K::TargetGeometry,
    moments: M,
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
    S: SourceCollection<K>,
    M: SourceMomentCollection<K>,
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
                let source = sources.source(source_id);
                let moment = moments.moment(source_id);
                kernel.eval_near(&target, &source, &moment, contribution);
                kernel.accumulate(out, contribution);
            }
        } else {
            active.push(source_tree.node_left_child[source_node_index]);
            active.push(source_tree.node_right_child[source_node_index]);
        }
    }

    HierarchicalError::Ok
}

/// Evaluate vector-valued targets against the source tree in parallel over target chunks.
///
/// This is intentionally the simplest parallelization of the single-tree
/// solver: each worker owns disjoint target and component output slices and
/// runs the serial source-tree evaluator on that slice. It shares the source
/// tree and source summaries between workers, and avoids any cross-thread
/// output accumulation.
#[inline]
pub fn eval_par<K, T, S, M, C, const D: usize>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, T>,
    source_summaries: &[K::SourceSummary],
    sources: S,
    targets: C,
    moments: M,
    theta: T,
    out: [&mut [T]; D],
    scratch: &mut EvaluationScratch<'_, [T; D]>,
) -> HierarchicalError
where
    K: HierarchicalKernel<Scalar = T, Output = [T; D]> + Sync,
    T: Scalar,
    K::TargetGeometry: Copy,
    S: SourceCollection<K>,
    M: SourceMomentCollection<K>,
    C: TargetCollection<K>,
{
    let err = validate_source_tree_layout(source_tree);
    if err != HierarchicalError::Ok {
        return err;
    }
    if D == 0
        || sources.len() != source_tree.n_items()
        || !sources.valid_lengths()
        || moments.len() != source_tree.n_items()
        || !moments.valid_lengths()
        || !targets.valid_lengths()
    {
        return HierarchicalError::LengthMismatch;
    }
    for component in 0..D {
        if out[component].len() != targets.len() {
            return HierarchicalError::LengthMismatch;
        }
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
    eval_par_chunks(
        kernel,
        source_tree,
        source_summaries,
        sources,
        targets,
        moments,
        theta,
        out,
        &mut scratch.contribution[..chunk_count],
        chunk_size,
        &error_code,
    );

    HierarchicalError::from_u32(error_code.load(Ordering::Relaxed))
}

#[inline]
/// Evaluate validated output chunks in parallel and preserve the first error code.
fn eval_par_chunks<K, T, S, M, C, const D: usize>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, T>,
    source_summaries: &[K::SourceSummary],
    sources: S,
    targets: C,
    moments: M,
    theta: T,
    out: [&mut [T]; D],
    scratch_contributions: &mut [[T; D]],
    chunk_size: usize,
    error_code: &AtomicU32,
) where
    K: HierarchicalKernel<Scalar = T, Output = [T; D]> + Sync,
    T: Scalar,
    K::TargetGeometry: Copy,
    S: SourceCollection<K>,
    M: SourceMomentCollection<K>,
    C: TargetCollection<K>,
{
    if error_code.load(Ordering::Relaxed) != HierarchicalError::Ok as u32 {
        return;
    }

    let target_count = targets.len();
    if target_count <= chunk_size {
        let mut chunk_scratch = EvaluationScratch {
            contribution: &mut scratch_contributions[..1],
        };
        let err = eval_validated(
            kernel,
            source_tree,
            source_summaries,
            sources,
            targets,
            moments,
            theta,
            out,
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
        return;
    }

    let chunk_count = target_count.div_ceil(chunk_size);
    let left_chunk_count = chunk_count / 2;
    let left_target_count = left_chunk_count * chunk_size;
    let (left_out, right_out) = split_output_components(out, left_target_count);
    let (left_scratch, right_scratch) = scratch_contributions.split_at_mut(left_chunk_count);
    let left_targets = targets.slice(0, left_target_count);
    let right_targets = targets.slice(left_target_count, target_count);

    rayon::join(
        || {
            eval_par_chunks(
                kernel,
                source_tree,
                source_summaries,
                sources,
                left_targets,
                moments,
                theta,
                left_out,
                left_scratch,
                chunk_size,
                error_code,
            );
        },
        || {
            eval_par_chunks(
                kernel,
                source_tree,
                source_summaries,
                sources,
                right_targets,
                moments,
                theta,
                right_out,
                right_scratch,
                chunk_size,
                error_code,
            );
        },
    );
}

#[inline]
/// Split component-major mutable output slices into disjoint chunks.
fn split_output_components<T, const D: usize>(
    mut out: [&mut [T]; D],
    mid: usize,
) -> ([&mut [T]; D], [&mut [T]; D]) {
    let mut left: [&mut [T]; D] = std::array::from_fn(|_| &mut [] as &mut [T]);
    let mut right: [&mut [T]; D] = std::array::from_fn(|_| &mut [] as &mut [T]);
    for component in 0..D {
        let full = std::mem::take(&mut out[component]);
        let (left_component, right_component) = full.split_at_mut(mid);
        left[component] = left_component;
        right[component] = right_component;
    }
    (left, right)
}

/// Compute the source-tree level represented at each target by the terminal traversal nodes.
///
/// This is a diagnostic companion to [`eval`]. It mirrors
/// the same source-tree walk but does not evaluate field values. Far-accepted
/// nodes contribute their traversal depth, while direct leaf fallbacks
/// contribute the leaf depth. Each contribution is weighted by the number of
/// original source items represented by each terminal node, giving per-target
/// accepted levels.
#[inline]
pub fn accepted_levels<K, C>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    targets: C,
    theta: K::Scalar,
    out: &mut [K::Scalar],
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
    if targets.len() != out.len() || !targets.valid_lengths() {
        return HierarchicalError::LengthMismatch;
    }
    if source_summaries.len() < source_tree.n_nodes() {
        return HierarchicalError::ScratchTooSmall;
    }

    let mut active = Vec::new();
    for target_id in 0..targets.len() {
        let target = targets.target(target_id);
        let mut weighted_level = K::Scalar::ZERO;
        let mut represented_sources = K::Scalar::ZERO;

        active.clear();
        active.push((0_u32, 0_u32));
        while let Some((source_node, source_level)) = active.pop() {
            let source_node_index = source_node as usize;
            let source_count = crate::math::cast::<K::Scalar>(
                source_tree.node_range_count[source_node_index] as f64,
            );
            let source_summary = &source_summaries[source_node_index];
            let source_aabb = source_tree.node_aabb[source_node_index];
            if kernel.accept_far(target.aabb(), source_aabb, source_summary, theta) {
                weighted_level = weighted_level
                    + crate::math::cast::<K::Scalar>(f64::from(source_level)) * source_count;
                represented_sources = represented_sources + source_count;
                continue;
            }

            let leaf_count = source_tree.leaf_count[source_node_index];
            if leaf_count > 0 {
                weighted_level = weighted_level
                    + crate::math::cast::<K::Scalar>(f64::from(source_level)) * source_count;
                represented_sources = represented_sources + source_count;
            } else {
                let next_level = source_level + 1;
                active.push((source_tree.node_left_child[source_node_index], next_level));
                active.push((source_tree.node_right_child[source_node_index], next_level));
            }
        }

        out[target_id] = if represented_sources > K::Scalar::ZERO {
            weighted_level / represented_sources
        } else {
            crate::math::cast::<K::Scalar>(f64::NAN)
        };
    }

    HierarchicalError::Ok
}

/// Dense exact fallback using nested range loops.
#[inline]
pub fn eval_dense<K, S, C, M>(
    kernel: &K,
    sources: S,
    targets: C,
    moments: M,
    out: &mut [K::Output],
    scratch: &mut EvaluationScratch<'_, K::Output>,
) -> HierarchicalError
where
    K: HierarchicalKernel,
    S: SourceCollection<K>,
    C: TargetCollection<K>,
    M: SourceMomentCollection<K>,
{
    if sources.len() != moments.len()
        || !sources.valid_lengths()
        || !moments.valid_lengths()
        || targets.len() != out.len()
        || !targets.valid_lengths()
    {
        return HierarchicalError::LengthMismatch;
    }
    if scratch.contribution.is_empty() {
        return HierarchicalError::ScratchTooSmall;
    }

    for i in 0..out.len() {
        kernel.zero_output(&mut out[i]);
    }

    for target_id in 0..targets.len() {
        let target = targets.target(target_id);
        let target_out = &mut out[target_id];
        for source_id in 0..sources.len() {
            let source = sources.source(source_id);
            let moment = moments.moment(source_id);
            kernel.eval_near(&target, &source, &moment, &mut scratch.contribution[0]);
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
/// Propagate leaf source summaries upward through the internal tree levels.
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
            let children = [summaries[left as usize], summaries[right as usize]];
            let err = kernel.combine_source_summaries(&children, &mut summaries[node_id as usize]);
            if err != HierarchicalError::Ok {
                return err;
            }
        }
    }

    HierarchicalError::Ok
}
