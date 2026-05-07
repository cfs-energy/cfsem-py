use super::{Aabb, DualTreeScalar};

/// Runtime error code for dual-tree operations.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DualTreeError {
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

/// Geometry that can be inserted into a cluster tree.
pub trait BoundedGeometry {
    type Scalar: DualTreeScalar;

    fn aabb(&self) -> Aabb<Self::Scalar>;
    fn representative_point(&self) -> [Self::Scalar; 3];
}

/// Trait implemented by physics kernels that can use the generic dual-tree evaluator.
pub trait DualTreeKernel {
    type Scalar: DualTreeScalar;
    type SourceGeometry: BoundedGeometry<Scalar = Self::Scalar> + Sync;
    type TargetGeometry: BoundedGeometry<Scalar = Self::Scalar> + Sync;

    type SourceMoment: Clone + Send + Sync;
    type SourceSummary: Copy + Default + Send + Sync;
    type TargetSummary: Copy + Default + Send + Sync;
    type Output: Copy + Default + Send + Sync;

    fn summarize_leaf_sources(
        &self,
        source_ids: &[u32],
        sources: &[Self::SourceGeometry],
        moments: &[Self::SourceMoment],
        out: &mut Self::SourceSummary,
    ) -> DualTreeError;

    fn combine_source_summaries(
        &self,
        children: &[Self::SourceSummary],
        child_ids: &[u32],
        out: &mut Self::SourceSummary,
    ) -> DualTreeError;

    fn summarize_leaf_targets(
        &self,
        target_ids: &[u32],
        targets: &[Self::TargetGeometry],
        out: &mut Self::TargetSummary,
    ) -> DualTreeError;

    fn combine_target_summaries(
        &self,
        children: &[Self::TargetSummary],
        child_ids: &[u32],
        out: &mut Self::TargetSummary,
    ) -> DualTreeError;

    fn eval_exact(
        &self,
        target: &Self::TargetGeometry,
        source: &Self::SourceGeometry,
        moment: &Self::SourceMoment,
        out: &mut Self::Output,
    ) -> DualTreeError;

    fn eval_far(
        &self,
        target: &Self::TargetSummary,
        source: &Self::SourceSummary,
        out: &mut Self::Output,
    ) -> DualTreeError;

    fn zero_output(&self, out: &mut Self::Output);
    fn accumulate(&self, out: &mut Self::Output, contribution: &Self::Output);

    #[inline]
    fn describe_error(&self, error: DualTreeError) -> &'static str {
        match error {
            DualTreeError::KernelError0 => "kernel error 0",
            DualTreeError::KernelError1 => "kernel error 1",
            _ => "not a kernel error",
        }
    }
}
