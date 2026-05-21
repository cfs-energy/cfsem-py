use super::{Aabb, Scalar};

/// Runtime error code for hierarchical tree operations.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HierarchicalError {
    Ok = 0,
    EmptyInput = 1,
    LengthMismatch = 2,
    ScratchTooSmall = 3,
    InvalidLeafSize = 4,
    InvalidTheta = 5,
    CapacityExceeded = 6,
    KernelError0 = 7,
    KernelError1 = 8,
}

impl HierarchicalError {
    /// Convert a raw GPU-compatible error code into a named error.
    ///
    /// Args:
    ///     value: Integer error code produced by hierarchical tree operations.
    ///
    /// Returns:
    ///     Matching error value, or [`HierarchicalError::KernelError1`] for unknown codes.
    #[inline]
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::Ok,
            1 => Self::EmptyInput,
            2 => Self::LengthMismatch,
            3 => Self::ScratchTooSmall,
            4 => Self::InvalidLeafSize,
            5 => Self::InvalidTheta,
            6 => Self::CapacityExceeded,
            7 => Self::KernelError0,
            8 => Self::KernelError1,
            _ => Self::KernelError1,
        }
    }
}

/// Geometry that can be inserted into a cluster tree.
pub trait BoundedGeometry {
    type Scalar: Scalar;

    /// Return the axis-aligned bounds of this geometry.
    ///
    /// Returns:
    ///     Bounds that fully contain the geometry.
    fn aabb(&self) -> Aabb<Self::Scalar>;

    /// Return a representative point used for source-tree ordering.
    ///
    /// Returns:
    ///     Point used by spatial sorting and Morton-code construction.
    fn representative_point(&self) -> [Self::Scalar; 3];
}

/// Borrowed target storage that can produce scalar target geometry values.
///
/// This keeps the evaluator generic over physical target point layouts. Plain
/// slices work for Rust callers, while Python bindings can pass borrowed
/// component columns without first allocating interleaved target structs.
pub trait TargetCollection<K: HierarchicalKernel>: Copy + Sync {
    /// Number of target points in the collection.
    fn len(self) -> usize;

    /// Return whether all column-like target storage has the same length.
    ///
    /// Evaluators call this immediately before looping over target data so
    /// mismatched columns return [`HierarchicalError::LengthMismatch`] instead of
    /// panicking from inside the hot target loop.
    fn has_consistent_lengths(self) -> bool;

    /// Return one scalar target geometry value.
    fn target(self, index: usize) -> K::TargetGeometry;

    /// Return a borrowed sub-collection over `start..end`.
    fn slice(self, start: usize, end: usize) -> Self;

    /// Return whether the collection contains no targets.
    #[inline]
    fn is_empty(self) -> bool {
        self.len() == 0
    }
}

impl<K> TargetCollection<K> for &[K::TargetGeometry]
where
    K: HierarchicalKernel,
    K::TargetGeometry: Copy,
{
    #[inline]
    fn len(self) -> usize {
        <[K::TargetGeometry]>::len(self)
    }

    #[inline]
    fn has_consistent_lengths(self) -> bool {
        true
    }

    #[inline]
    fn target(self, index: usize) -> K::TargetGeometry {
        self[index]
    }

    #[inline]
    fn slice(self, start: usize, end: usize) -> Self {
        &self[start..end]
    }
}

/// Trait implemented by physics kernels that can use the generic hierarchical evaluator.
pub trait HierarchicalKernel {
    type Scalar: Scalar;
    type SourceGeometry: BoundedGeometry<Scalar = Self::Scalar> + Sync;
    type TargetGeometry: BoundedGeometry<Scalar = Self::Scalar> + Sync;

    type SourceMoment: Clone + Send + Sync;
    type SourceSummary: Copy + Default + Send + Sync;
    type TargetSummary: Copy + Default + Send + Sync;
    type Output: Copy + Default + Send + Sync;

    /// Summarize a leaf node from the original source values.
    ///
    /// Args:
    ///     source_ids: Source indices owned by the leaf.
    ///     sources: Source geometry values.
    ///     moments: Source amplitudes or moments.
    ///     out: Summary value to fill.
    ///
    /// Returns:
    ///     Error code for the summary operation.
    fn summarize_leaf_sources(
        &self,
        source_ids: &[u32],
        sources: &[Self::SourceGeometry],
        moments: &[Self::SourceMoment],
        out: &mut Self::SourceSummary,
    ) -> HierarchicalError;

    /// Combine child source summaries into one parent summary.
    ///
    /// Args:
    ///     children: Source summary storage for the whole tree.
    ///     child_ids: Child node indices to combine.
    ///     out: Parent summary value to fill.
    ///
    /// Returns:
    ///     Error code for the summary operation.
    fn combine_source_summaries(
        &self,
        children: &[Self::SourceSummary],
        child_ids: &[u32],
        out: &mut Self::SourceSummary,
    ) -> HierarchicalError;

    /// Summarize a leaf node from target geometry.
    ///
    /// Args:
    ///     target_ids: Target indices owned by the leaf.
    ///     targets: Target geometry values.
    ///     out: Summary value to fill.
    ///
    /// Returns:
    ///     Error code for the summary operation.
    fn summarize_leaf_targets(
        &self,
        target_ids: &[u32],
        targets: &[Self::TargetGeometry],
        out: &mut Self::TargetSummary,
    ) -> HierarchicalError;

    /// Evaluate the exact source-target interaction.
    ///
    /// Args:
    ///     target: Target geometry value.
    ///     source: Source geometry value.
    ///     moment: Source amplitude or moment.
    ///     out: Contribution value to fill.
    ///
    fn eval_exact(
        &self,
        target: &Self::TargetGeometry,
        source: &Self::SourceGeometry,
        moment: &Self::SourceMoment,
        out: &mut Self::Output,
    );

    /// Evaluate a far-field source summary against a target summary.
    ///
    /// Args:
    ///     target: Target summary value.
    ///     source: Source summary value.
    ///     out: Contribution value to fill.
    ///
    fn eval_far(
        &self,
        target: &Self::TargetSummary,
        source: &Self::SourceSummary,
        out: &mut Self::Output,
    );

    /// Decide whether a source node is far enough from a target to use its summary.
    ///
    /// The default implementation uses the standard AABB Barnes-Hut criterion.
    /// Kernels can override this to apply source-summary-specific constraints,
    /// such as stricter acceptance for open filament arcs.
    ///
    /// Args:
    ///     target_aabb: Bounds of the target or target group.
    ///     source_aabb: Bounds of the source node.
    ///     source: Source summary for kernel-specific acceptance checks.
    ///     theta: Barnes-Hut acceptance angle.
    ///
    /// Returns:
    ///     Whether the evaluator may use the source summary.
    #[inline]
    fn accept_far(
        &self,
        target_aabb: Aabb<Self::Scalar>,
        source_aabb: Aabb<Self::Scalar>,
        _source: &Self::SourceSummary,
        theta: Self::Scalar,
    ) -> bool {
        geometric_accept_far(target_aabb, source_aabb, theta)
    }

    /// Reset an output accumulator to the kernel's zero value.
    ///
    /// Args:
    ///     out: Output accumulator to reset.
    fn zero_output(&self, out: &mut Self::Output);

    /// Add one contribution into an output accumulator.
    ///
    /// Args:
    ///     out: Output accumulator to update.
    ///     contribution: Contribution to add.
    fn accumulate(&self, out: &mut Self::Output, contribution: &Self::Output);

    /// Describe a kernel-specific error code.
    ///
    /// Args:
    ///     error: Error code returned by a kernel method.
    ///
    /// Returns:
    ///     Static description of the error code.
    #[inline]
    fn describe_error(&self, error: HierarchicalError) -> &'static str {
        match error {
            HierarchicalError::KernelError0 => "kernel error 0",
            HierarchicalError::KernelError1 => "kernel error 1",
            _ => "not a kernel error",
        }
    }
}

/// Standard geometric Barnes-Hut acceptance test for two AABBs.
///
/// This helper is shared by the default kernel implementation and kernel-specific
/// acceptance hooks that first modify the effective `theta` value.
#[inline]
pub(crate) fn geometric_accept_far<T: Scalar>(
    target_aabb: Aabb<T>,
    source_aabb: Aabb<T>,
    theta: T,
) -> bool {
    if theta <= T::ZERO {
        return false;
    }

    let gap_sq = target_aabb.gap_distance_sq(&source_aabb);
    if gap_sq <= T::ZERO {
        return false;
    }

    let target_diam = target_aabb.diameter_sq().sqrt();
    let source_diam = source_aabb.diameter_sq().sqrt();
    let combined = target_diam + source_diam;
    gap_sq * theta * theta > combined * combined
}
