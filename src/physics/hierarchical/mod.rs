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

pub mod aabb;
pub mod convenience;
pub mod evaluator;
pub mod kernel;
pub mod kernels;
pub mod tree;

pub use convenience::{
    flux_density_dipole_hierarchical, flux_density_linear_filament_hierarchical,
    flux_density_triangle_mesh_hierarchical, vector_potential_dipole_hierarchical,
    vector_potential_linear_filament_hierarchical, vector_potential_triangle_mesh_hierarchical,
};

pub(crate) use crate::math::Scalar;
pub(crate) use aabb::Aabb;
#[cfg(test)]
pub(crate) use evaluator::eval_dense;
pub(crate) use evaluator::{
    EvaluationScratch, SourceNodeSummaries, scratch_len, scratch_len_par, update_summaries,
};
pub use evaluator::{eval, eval_par, eval_par_with_skip, eval_with_skip};
pub use kernel::Skip;
pub(crate) use kernel::{
    BoundedGeometry, BoundedGeometryCollection, HierarchicalError, HierarchicalKernel,
    SourceCollection, SourceMomentCollection, TargetCollection, geometric_accept_far,
};
pub(crate) use tree::{BuildMethod, ClusterTree, ClusterTreeView};

#[cfg(test)]
mod tests;
