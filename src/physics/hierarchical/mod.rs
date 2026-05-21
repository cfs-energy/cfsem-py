//! Generic single-tree Barnes-Hut infrastructure.
//!
//! The public hierarchical evaluation path builds one source tree over fixed
//! source geometry and evaluates target points directly against that source
//! tree. Source leaves are fixed at one primitive per leaf, which is the most
//! conservative acceptance choice and has also been the fastest configuration
//! for the current kernels. Concrete physics kernels are implemented in the
//! [`kernels`] submodule.
//!
//! # References
//!
//! * \[1\] PhysicsNeMo Contributors, "NVIDIA PhysicsNeMo: An open-source
//!     framework for physics-based deep learning in science and engineering,"
//!     Feb. 24, 2023. \[Online\]. Available: <https://github.com/NVIDIA/physicsnemo>

mod aabb;
mod convenience;
mod evaluator;
mod kernel;
pub mod kernels;
mod scalar;
mod tree;

pub use aabb::Aabb;
pub use convenience::{
    ConstructionMethod, DiagnosticInfo, flux_density_dipole_hierarchical,
    flux_density_linear_filament_hierarchical, flux_density_triangle_mesh_hierarchical,
    vector_potential_dipole_hierarchical, vector_potential_linear_filament_hierarchical,
    vector_potential_triangle_mesh_hierarchical,
};
pub use evaluator::{
    EvaluationScratch, SourceNodeSummaries, accepted_source_level_diagnostic_into,
    dense_direct_evaluate_into, eval, eval_par, parallel_source_tree_evaluation_scratch_len,
    source_tree_evaluation_scratch_len, update_source_summaries_into,
};
pub(crate) use kernel::geometric_accept_far;
pub use kernel::{
    BoundedGeometry, BoundedGeometryCollection, HierarchicalError, HierarchicalKernel,
    SourceCollection, SourceMomentCollection, TargetCollection,
};
pub use scalar::Scalar;
pub use tree::{ClusterTree, ClusterTreeBuildMethod, ClusterTreeView};

#[cfg(test)]
mod tests;
