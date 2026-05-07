//! Generic dual-tree Barnes-Hut infrastructure.
//!
//! This module provides the generic geometry, tree, plan, kernel-trait, and
//! evaluator pieces. Concrete physics kernels are intentionally implemented
//! elsewhere.

mod aabb;
mod evaluator;
mod kernel;
pub mod kernels;
mod plan;
mod scalar;
mod tree;

pub use aabb::Aabb;
pub use evaluator::{
    EvaluationScratch, SourceNodeSummaries, TargetNodeSummaries, dense_direct_evaluate_into,
    evaluate_into, update_source_summaries_into, update_target_summaries_into,
};
pub use kernel::{BoundedGeometry, DualTreeError, DualTreeKernel};
pub use plan::{DualInteractionPlan, DualInteractionPlanView};
pub use scalar::DualTreeScalar;
pub use tree::{ClusterTree, ClusterTreeView};

#[cfg(test)]
mod tests;
