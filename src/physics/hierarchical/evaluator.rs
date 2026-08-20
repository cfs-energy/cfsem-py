use super::{
    BoundedGeometry, ClusterTreeView, HierarchicalError, HierarchicalKernel, Scalar, Skip,
    SourceCollection, SourceMomentCollection, TargetCollection,
};
use rayon::prelude::*;
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
/// into caller-provided component slices. [`Skip::Near`] retains accepted far-summary
/// contributions only, [`Skip::Far`] retains direct leaf contributions only, and `None`
/// evaluates both interaction classes. [`Skip::Both`] zeroes the output without target
/// summarization, source-tree traversal, or contribution scratch. The output slice count must
/// match the kernel output dimension `D`.
#[inline]
pub fn eval<K, T, S, M, C, const D: usize>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, T>,
    source_summaries: &[K::SourceSummary],
    sources: S,
    targets: C,
    moments: M,
    theta: T,
    skip: Option<Skip>,
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
    match skip {
        None => eval_validated::<K, T, S, M, C, D, true, true>(
            kernel,
            source_tree,
            source_summaries,
            sources,
            targets,
            moments,
            theta,
            out,
            scratch,
        ),
        Some(Skip::Near) => eval_validated::<K, T, S, M, C, D, false, true>(
            kernel,
            source_tree,
            source_summaries,
            sources,
            targets,
            moments,
            theta,
            out,
            scratch,
        ),
        Some(Skip::Far) => eval_validated::<K, T, S, M, C, D, true, false>(
            kernel,
            source_tree,
            source_summaries,
            sources,
            targets,
            moments,
            theta,
            out,
            scratch,
        ),
        Some(Skip::Both) => eval_validated::<K, T, S, M, C, D, false, false>(
            kernel,
            source_tree,
            source_summaries,
            sources,
            targets,
            moments,
            theta,
            out,
            scratch,
        ),
    }
}

#[inline]
/// Evaluate validated source-target rows with the hierarchical tree walk.
fn eval_validated<
    K,
    T,
    S,
    M,
    C,
    const D: usize,
    const EVALUATE_NEAR: bool,
    const EVALUATE_FAR: bool,
>(
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
    if !EVALUATE_NEAR && !EVALUATE_FAR {
        for component in out {
            component.fill(T::ZERO);
        }
        return HierarchicalError::Ok;
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
        let err = eval_scalar::<K, S, M, EVALUATE_NEAR, EVALUATE_FAR>(
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

/// Handle terminal nodes selected by the shared source-tree traversal.
trait TraversalVisitor<K: HierarchicalKernel> {
    /// Handle a source node accepted through the kernel's far criterion.
    fn on_far_accept(
        &mut self,
        source_node_index: usize,
        source_level: u32,
        source_summary: &K::SourceSummary,
    );

    /// Handle a rejected source leaf through direct source interactions.
    fn on_near_leaf(&mut self, source_node_index: usize, source_level: u32, source_ids: &[u32]);
}

/// Traverse one target against the source tree and report each terminal node.
///
/// Field evaluation and traversal diagnostics both use this function, keeping
/// kernel-specific acceptance and leaf fallback behavior identical.
#[inline]
fn traverse_source_tree<K, V>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    target: &K::TargetGeometry,
    theta: K::Scalar,
    active: &mut Vec<(u32, u32)>,
    visitor: &mut V,
) where
    K: HierarchicalKernel,
    V: TraversalVisitor<K>,
{
    active.clear();
    active.push((0_u32, 0_u32));
    let target_aabb = target.aabb();
    while let Some((source_node, source_level)) = active.pop() {
        let source_node_index = source_node as usize;
        let source_summary = &source_summaries[source_node_index];
        let source_aabb = source_tree.node_aabb[source_node_index];
        if kernel.accept_far(target_aabb, source_aabb, source_summary, theta) {
            visitor.on_far_accept(source_node_index, source_level, source_summary);
            continue;
        }

        let leaf_count = source_tree.leaf_count[source_node_index];
        if leaf_count > 0 {
            let start = source_tree.leaf_start[source_node_index] as usize;
            let count = leaf_count as usize;
            let end = start + count;
            let source_ids = &source_tree.sorted_indices[start..end];
            visitor.on_near_leaf(source_node_index, source_level, source_ids);
        } else {
            let next_level = source_level + 1;
            active.push((source_tree.node_left_child[source_node_index], next_level));
            active.push((source_tree.node_right_child[source_node_index], next_level));
        }
    }
}

/// Terminal-node visitor that evaluates far summaries and direct leaf sources.
struct EvaluationTraversalVisitor<'a, K, S, M, const EVALUATE_NEAR: bool, const EVALUATE_FAR: bool>
where
    K: HierarchicalKernel,
{
    kernel: &'a K,
    sources: S,
    target: K::TargetGeometry,
    moments: M,
    out: &'a mut K::Output,
    contribution: &'a mut K::Output,
    target_summary: &'a K::TargetSummary,
}

impl<K, S, M, const EVALUATE_NEAR: bool, const EVALUATE_FAR: bool> TraversalVisitor<K>
    for EvaluationTraversalVisitor<'_, K, S, M, EVALUATE_NEAR, EVALUATE_FAR>
where
    K: HierarchicalKernel,
    S: SourceCollection<K>,
    M: SourceMomentCollection<K>,
{
    #[inline]
    fn on_far_accept(
        &mut self,
        _source_node_index: usize,
        _source_level: u32,
        source_summary: &K::SourceSummary,
    ) {
        if EVALUATE_FAR {
            self.kernel
                .eval_far(self.target_summary, source_summary, self.contribution);
            self.kernel.accumulate(self.out, self.contribution);
        }
    }

    #[inline]
    fn on_near_leaf(&mut self, _source_node_index: usize, _source_level: u32, source_ids: &[u32]) {
        if !EVALUATE_NEAR {
            return;
        }
        for &source_id in source_ids {
            let source_id = source_id as usize;
            let source = self.sources.source(source_id);
            let moment = self.moments.moment(source_id);
            self.kernel
                .eval_near(&self.target, &source, &moment, self.contribution);
            self.kernel.accumulate(self.out, self.contribution);
        }
    }
}

/// Evaluate one scalar target against the source tree.
///
/// Serial and parallel vector evaluators both call this helper, and its
/// terminal-node actions use the same traversal as diagnostics.
#[inline]
fn eval_scalar<K, S, M, const EVALUATE_NEAR: bool, const EVALUATE_FAR: bool>(
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
    active: &mut Vec<(u32, u32)>,
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

    let mut visitor = EvaluationTraversalVisitor::<K, S, M, EVALUATE_NEAR, EVALUATE_FAR> {
        kernel,
        sources,
        target,
        moments,
        out,
        contribution,
        target_summary,
    };
    traverse_source_tree(
        kernel,
        source_tree,
        source_summaries,
        &target,
        theta,
        active,
        &mut visitor,
    );

    HierarchicalError::Ok
}

/// Evaluate vector-valued targets against the source tree in parallel over target chunks.
///
/// This is intentionally the simplest parallelization of the single-tree
/// solver: each worker owns disjoint target and component output slices and
/// runs the serial source-tree evaluator on that slice. It shares the source
/// tree and source summaries between workers, and avoids any cross-thread output accumulation.
/// [`Skip::Near`] retains accepted far-summary contributions only, [`Skip::Far`] retains direct
/// leaf contributions only, and `None` evaluates both interaction classes. [`Skip::Both`] zeroes
/// the output without target summarization, source-tree traversal, or parallel scratch use.
#[inline]
pub fn eval_par<K, T, S, M, C, const D: usize>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, T>,
    source_summaries: &[K::SourceSummary],
    sources: S,
    targets: C,
    moments: M,
    theta: T,
    skip: Option<Skip>,
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
    if skip == Some(Skip::Both) {
        for component in out {
            component.fill(T::ZERO);
        }
        return HierarchicalError::Ok;
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
    match skip {
        None => eval_par_chunks::<K, T, S, M, C, D, true, true>(
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
        ),
        Some(Skip::Near) => eval_par_chunks::<K, T, S, M, C, D, false, true>(
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
        ),
        Some(Skip::Far) => eval_par_chunks::<K, T, S, M, C, D, true, false>(
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
        ),
        Some(Skip::Both) => unreachable!("Skip::Both returns before parallel evaluation"),
    }

    HierarchicalError::from_u32(error_code.load(Ordering::Relaxed))
}

#[inline]
/// Evaluate validated output chunks in parallel and preserve the first error code.
fn eval_par_chunks<
    K,
    T,
    S,
    M,
    C,
    const D: usize,
    const EVALUATE_NEAR: bool,
    const EVALUATE_FAR: bool,
>(
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
        let err = eval_validated::<K, T, S, M, C, D, EVALUATE_NEAR, EVALUATE_FAR>(
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
            eval_par_chunks::<K, T, S, M, C, D, EVALUATE_NEAR, EVALUATE_FAR>(
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
            eval_par_chunks::<K, T, S, M, C, D, EVALUATE_NEAR, EVALUATE_FAR>(
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

/// Canonical CSC sparsity for direct source-target interactions selected by a tree walk.
///
/// Rows are original source indices and columns are target indices, so the shape is
/// `(source_count, target_count)`. Row indices are sorted within each column and are unique.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NearFieldInteractionMap {
    /// Original source indices for stored direct interactions.
    pub row_indices: Vec<u32>,
    /// CSC column offsets, with length `target_count + 1`.
    pub column_pointers: Vec<usize>,
    /// Number of source rows in the sparse pattern.
    pub source_count: usize,
    /// Number of target columns in the sparse pattern.
    pub target_count: usize,
}

/// Diagnostic data collected in one traversal of the source tree per target.
#[derive(Clone, Debug, PartialEq)]
pub struct TraversalDiagnostics<T: Scalar> {
    /// Source-tree level represented at each target.
    pub accepted_levels: Vec<T>,
    /// Direct near-field source-target interaction pattern.
    pub near_field_interaction_map: NearFieldInteractionMap,
}

/// Terminal-node visitor that collects one target's traversal diagnostics.
struct DiagnosticTraversalVisitor<'a, K>
where
    K: HierarchicalKernel,
{
    source_tree: ClusterTreeView<'a, K::Scalar>,
    weighted_level: K::Scalar,
    represented_sources: K::Scalar,
    row_indices: &'a mut Vec<u32>,
}

impl<K> DiagnosticTraversalVisitor<'_, K>
where
    K: HierarchicalKernel,
{
    #[inline]
    fn record_terminal_node(&mut self, source_node_index: usize, source_level: u32) {
        let source_count = crate::math::cast::<K::Scalar>(
            self.source_tree.node_range_count[source_node_index] as f64,
        );
        self.weighted_level = self.weighted_level
            + crate::math::cast::<K::Scalar>(f64::from(source_level)) * source_count;
        self.represented_sources = self.represented_sources + source_count;
    }

    #[inline]
    fn accepted_level(&self) -> K::Scalar {
        if self.represented_sources > K::Scalar::ZERO {
            self.weighted_level / self.represented_sources
        } else {
            crate::math::cast::<K::Scalar>(f64::NAN)
        }
    }
}

impl<K> TraversalVisitor<K> for DiagnosticTraversalVisitor<'_, K>
where
    K: HierarchicalKernel,
{
    #[inline]
    fn on_far_accept(
        &mut self,
        source_node_index: usize,
        source_level: u32,
        _source_summary: &K::SourceSummary,
    ) {
        self.record_terminal_node(source_node_index, source_level);
    }

    #[inline]
    fn on_near_leaf(&mut self, source_node_index: usize, source_level: u32, source_ids: &[u32]) {
        self.record_terminal_node(source_node_index, source_level);
        self.row_indices.extend_from_slice(source_ids);
    }
}

/// Traversal diagnostics collected for a contiguous target chunk.
struct TraversalDiagnosticsChunk<T: Scalar> {
    accepted_levels: Vec<T>,
    row_indices: Vec<u32>,
    column_lengths: Vec<usize>,
}

/// Collect traversal diagnostics for a validated contiguous target chunk.
fn traversal_diagnostics_chunk<K, C>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    targets: C,
    theta: K::Scalar,
) -> TraversalDiagnosticsChunk<K::Scalar>
where
    K: HierarchicalKernel,
    K::TargetGeometry: Copy,
    C: TargetCollection<K>,
{
    let mut accepted_levels = Vec::with_capacity(targets.len());
    let mut row_indices = Vec::new();
    let mut column_lengths = Vec::with_capacity(targets.len());
    let mut active = Vec::new();

    for target_id in 0..targets.len() {
        let target = targets.target(target_id);
        let column_start = row_indices.len();
        let mut visitor = DiagnosticTraversalVisitor::<K> {
            source_tree,
            weighted_level: K::Scalar::ZERO,
            represented_sources: K::Scalar::ZERO,
            row_indices: &mut row_indices,
        };
        traverse_source_tree(
            kernel,
            source_tree,
            source_summaries,
            &target,
            theta,
            &mut active,
            &mut visitor,
        );
        accepted_levels.push(visitor.accepted_level());
        row_indices[column_start..].sort_unstable();
        column_lengths.push(row_indices.len() - column_start);
    }

    TraversalDiagnosticsChunk {
        accepted_levels,
        row_indices,
        column_lengths,
    }
}

/// Validate inputs shared by serial and parallel traversal diagnostics.
fn validate_traversal_diagnostics_inputs<K, C>(
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    targets: C,
) -> Result<(), HierarchicalError>
where
    K: HierarchicalKernel,
    C: TargetCollection<K>,
{
    let err = validate_source_tree_layout(source_tree);
    if err != HierarchicalError::Ok {
        return Err(err);
    }
    if !targets.valid_lengths() {
        return Err(HierarchicalError::LengthMismatch);
    }
    if source_summaries.len() < source_tree.n_nodes() {
        return Err(HierarchicalError::ScratchTooSmall);
    }
    Ok(())
}

/// Merge target chunks into canonical CSC traversal diagnostics.
fn merge_traversal_diagnostics_chunks<T: Scalar>(
    chunks: Vec<TraversalDiagnosticsChunk<T>>,
    source_count: usize,
    target_count: usize,
) -> TraversalDiagnostics<T> {
    let entry_count = chunks.iter().map(|chunk| chunk.row_indices.len()).sum();
    let mut accepted_levels = Vec::with_capacity(target_count);
    let mut row_indices = Vec::with_capacity(entry_count);
    let mut column_pointers = Vec::with_capacity(target_count + 1);
    column_pointers.push(0);

    for chunk in chunks {
        accepted_levels.extend(chunk.accepted_levels);
        row_indices.extend(chunk.row_indices);
        for column_length in chunk.column_lengths {
            column_pointers.push(column_pointers.last().copied().unwrap() + column_length);
        }
    }

    debug_assert_eq!(accepted_levels.len(), target_count);
    debug_assert_eq!(column_pointers.len(), target_count + 1);
    TraversalDiagnostics {
        accepted_levels,
        near_field_interaction_map: NearFieldInteractionMap {
            row_indices,
            column_pointers,
            source_count,
            target_count,
        },
    }
}

/// Collect accepted levels and the direct near-field CSC pattern.
///
/// This uses the same terminal-node traversal as field evaluation. The interaction map records
/// every original source owned by a rejected terminal leaf; far-accepted nodes are omitted. The
/// resulting pattern has shape `(source_count, target_count)`.
pub fn traversal_diagnostics<K, C>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    targets: C,
    theta: K::Scalar,
) -> Result<TraversalDiagnostics<K::Scalar>, HierarchicalError>
where
    K: HierarchicalKernel,
    K::TargetGeometry: Copy,
    C: TargetCollection<K>,
{
    validate_traversal_diagnostics_inputs::<K, C>(source_tree, source_summaries, targets)?;
    let source_count = source_tree.node_range_count[0] as usize;
    let target_count = targets.len();
    let chunk = traversal_diagnostics_chunk(kernel, source_tree, source_summaries, targets, theta);
    Ok(merge_traversal_diagnostics_chunks(
        vec![chunk],
        source_count,
        target_count,
    ))
}

/// Collect accepted levels and the direct near-field CSC pattern in parallel over targets.
///
/// Target chunks are traversed independently, then merged in target order to preserve canonical
/// CSC columns and byte-for-byte agreement with [`traversal_diagnostics`].
pub fn traversal_diagnostics_par<K, C>(
    kernel: &K,
    source_tree: ClusterTreeView<'_, K::Scalar>,
    source_summaries: &[K::SourceSummary],
    targets: C,
    theta: K::Scalar,
) -> Result<TraversalDiagnostics<K::Scalar>, HierarchicalError>
where
    K: HierarchicalKernel + Sync,
    K::TargetGeometry: Copy,
    C: TargetCollection<K>,
{
    validate_traversal_diagnostics_inputs::<K, C>(source_tree, source_summaries, targets)?;
    let source_count = source_tree.node_range_count[0] as usize;
    let target_count = targets.len();
    if target_count == 0 {
        return Ok(merge_traversal_diagnostics_chunks(
            Vec::new(),
            source_count,
            target_count,
        ));
    }

    let chunk_size = crate::chunksize(target_count);
    let chunk_count = target_count.div_ceil(chunk_size);
    let chunks = (0..chunk_count)
        .into_par_iter()
        .map(|chunk_id| {
            let start = chunk_id * chunk_size;
            let end = (start + chunk_size).min(target_count);
            traversal_diagnostics_chunk(
                kernel,
                source_tree,
                source_summaries,
                targets.slice(start, end),
                theta,
            )
        })
        .collect();
    Ok(merge_traversal_diagnostics_chunks(
        chunks,
        source_count,
        target_count,
    ))
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
