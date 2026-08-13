//! One-shot hierarchical field-solver convenience functions.
//!
//! These helpers intentionally rebuild the source tree every call. For the
//! operating range where the hierarchical solve wins over the dense direct
//! methods, tree construction is usually a small part of total runtime, so this
//! API favors direct-method ergonomics.

use crate::mesh::triangle3d::TriangleMeshView;
use std::time::Instant;

use super::kernels::{
    BoundaryElementFluxDensityKernel, BoundaryElementNodalValues, BoundaryElementTriangles,
    BoundaryElementVectorPotentialKernel, DipoleFluxDensityKernel, DipoleMoments, DipoleSources,
    DipoleTargets, DipoleVectorPotentialKernel, LinearFilamentFluxDensityKernel,
    LinearFilamentSources, LinearFilamentVectorPotentialKernel,
};
use super::{
    BuildMethod, ClusterTree, EvaluationScratch, HierarchicalError, HierarchicalKernel, Scalar,
    Skip, SourceCollection, SourceMomentCollection, SourceNodeSummaries, TargetCollection, eval,
    eval_par, eval_par_with_skip, eval_with_skip, scratch_len, scratch_len_par, update_summaries,
};

/// Diagnostic information returned by stateless hierarchical solves.
pub struct Diagnostics<K: HierarchicalKernel> {
    /// Source tree built for this solve.
    pub source_tree: ClusterTree<K::Scalar>,
    /// Wall-clock construction time in seconds.
    pub construction_seconds: f64,
    /// Wall-clock evaluation time in seconds.
    pub evaluation_seconds: f64,
    /// Number of sources in the solve.
    pub source_count: usize,
    /// Number of targets in the solve.
    pub target_count: usize,
}

impl<K: HierarchicalKernel> Diagnostics<K> {
    /// Borrow the source tree built for this solve.
    #[inline]
    pub fn source_tree(&self) -> &ClusterTree<K::Scalar> {
        &self.source_tree
    }
}

/// Hierarchical magnetic flux density of dipole sources at Cartesian targets.
///
/// This one-shot helper mirrors the direct dipole API while rebuilding the
/// source tree internally.
///
/// This is an approximate method, and no particular accuracy level is guaranteed.
/// Truncated methods like this one may average entire local loop structures out of
/// existence; as a result, maximum relative error is unbounded. This method must be
/// tuned to a given use-case in order to be useful, and should not be used to calculate
/// safety-related field limits.
///
/// Args:
///     loc: Dipole source coordinates.
///     moment: Dipole magnetic moment components.
///     obs: Observation point coordinates.
///     outer_radius: Radius for the magnetized-sphere near-field treatment.
///     construction_method: Source-tree construction method.
///     theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
///     par: Whether to evaluate target batches in parallel.
///     out: Output component slices to fill.
///
/// Returns:
///     Source-tree diagnostics and construction/evaluation timing on success.
///
/// Errors:
///     Returns [`HierarchicalError`] when input lengths are inconsistent, tree construction fails,
///     scratch storage is too small, or a kernel reports an error.
pub fn flux_density_dipole_hierarchical<T: Scalar>(
    loc: (&[T], &[T], &[T]),
    moment: (&[T], &[T], &[T]),
    obs: (&[T], &[T], &[T]),
    outer_radius: &[T],
    construction_method: BuildMethod,
    theta: T,
    par: bool,
    out: (&mut [T], &mut [T], &mut [T]),
) -> Result<Diagnostics<DipoleFluxDensityKernel<T>>, HierarchicalError> {
    let sources = DipoleSources::new(loc.0, loc.1, loc.2, outer_radius);
    let moments = DipoleMoments::new(moment.0, moment.1, moment.2);
    let targets = DipoleTargets::new(obs.0, obs.1, obs.2);
    one_shot_vec3(
        DipoleFluxDensityKernel::<T>::new(),
        sources,
        moments,
        targets,
        construction_method,
        theta,
        par,
        None,
        out,
    )
}

/// Hierarchical magnetic vector potential of dipole sources at Cartesian targets.
///
/// This one-shot helper mirrors the direct dipole API while rebuilding the
/// source tree internally.
///
/// This is an approximate method, and no particular accuracy level is guaranteed.
/// Truncated methods like this one may average entire local loop structures out of
/// existence; as a result, maximum relative error is unbounded. This method must be
/// tuned to a given use-case in order to be useful, and should not be used to calculate
/// safety-related field limits.
///
/// Args:
///     loc: Dipole source coordinates.
///     moment: Dipole magnetic moment components.
///     obs: Observation point coordinates.
///     outer_radius: Radius for the magnetized-sphere near-field treatment.
///     construction_method: Source-tree construction method.
///     theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
///     par: Whether to evaluate target batches in parallel.
///     out: Output component slices to fill.
///
/// Returns:
///     Source-tree diagnostics and construction/evaluation timing on success.
///
/// Errors:
///     Returns [`HierarchicalError`] when input lengths are inconsistent, tree construction fails,
///     scratch storage is too small, or a kernel reports an error.
pub fn vector_potential_dipole_hierarchical<T: Scalar>(
    loc: (&[T], &[T], &[T]),
    moment: (&[T], &[T], &[T]),
    obs: (&[T], &[T], &[T]),
    outer_radius: &[T],
    construction_method: BuildMethod,
    theta: T,
    par: bool,
    out: (&mut [T], &mut [T], &mut [T]),
) -> Result<Diagnostics<DipoleVectorPotentialKernel<T>>, HierarchicalError> {
    let sources = DipoleSources::new(loc.0, loc.1, loc.2, outer_radius);
    let moments = DipoleMoments::new(moment.0, moment.1, moment.2);
    let targets = DipoleTargets::new(obs.0, obs.1, obs.2);
    one_shot_vec3(
        DipoleVectorPotentialKernel::<T>::new(),
        sources,
        moments,
        targets,
        construction_method,
        theta,
        par,
        None,
        out,
    )
}

/// Hierarchical magnetic flux density of linear filament segments.
///
/// This one-shot helper mirrors [`crate::physics::linear_filament::flux_density_linear_filament`]
/// while rebuilding the source tree internally.
///
/// This is an approximate method, and no particular accuracy level is guaranteed.
/// Truncated methods like this one may average entire local loop structures out of
/// existence; as a result, maximum relative error is unbounded. This method must be
/// tuned to a given use-case in order to be useful, and should not be used to calculate
/// safety-related field limits.
///
/// Args:
///     xyzp: Observation point coordinates.
///     xyzfil: Filament segment start coordinates.
///     dlxyzfil: Filament segment start-to-end displacement components.
///     ifil: Current in each filament segment.
///     wire_radius: Wire radius for each filament segment.
///     construction_method: Source-tree construction method.
///     theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
///     par: Whether to evaluate target batches in parallel.
///     out: Output component slices to fill.
///
/// Returns:
///     Source-tree diagnostics and construction/evaluation timing on success.
///
/// Errors:
///     Returns [`HierarchicalError`] when input lengths are inconsistent, tree construction fails,
///     scratch storage is too small, or a kernel reports an error.
pub fn flux_density_linear_filament_hierarchical<T: Scalar>(
    xyzp: (&[T], &[T], &[T]),
    xyzfil: (&[T], &[T], &[T]),
    dlxyzfil: (&[T], &[T], &[T]),
    ifil: &[T],
    wire_radius: &[T],
    construction_method: BuildMethod,
    theta: T,
    par: bool,
    out: (&mut [T], &mut [T], &mut [T]),
) -> Result<Diagnostics<LinearFilamentFluxDensityKernel<T>>, HierarchicalError> {
    let sources = LinearFilamentSources::new(xyzfil, dlxyzfil, wire_radius);
    let targets = DipoleTargets::new(xyzp.0, xyzp.1, xyzp.2);
    one_shot_vec3(
        LinearFilamentFluxDensityKernel::<T>::new(),
        sources,
        ifil,
        targets,
        construction_method,
        theta,
        par,
        None,
        out,
    )
}

/// Hierarchical magnetic vector potential of linear filament segments.
///
/// This one-shot helper mirrors [`crate::physics::linear_filament::vector_potential_linear_filament`]
/// while rebuilding the source tree internally.
///
/// This is an approximate method, and no particular accuracy level is guaranteed.
/// Truncated methods like this one may average entire local loop structures out of
/// existence; as a result, maximum relative error is unbounded. This method must be
/// tuned to a given use-case in order to be useful, and should not be used to calculate
/// safety-related field limits.
///
/// Args:
///     xyzp: Observation point coordinates.
///     xyzfil: Filament segment start coordinates.
///     dlxyzfil: Filament segment start-to-end displacement components.
///     ifil: Current in each filament segment.
///     wire_radius: Wire radius for each filament segment.
///     construction_method: Source-tree construction method.
///     theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
///     par: Whether to evaluate target batches in parallel.
///     out: Output component slices to fill.
///
/// Returns:
///     Source-tree diagnostics and construction/evaluation timing on success.
///
/// Errors:
///     Returns [`HierarchicalError`] when input lengths are inconsistent, tree construction fails,
///     scratch storage is too small, or a kernel reports an error.
pub fn vector_potential_linear_filament_hierarchical<T: Scalar>(
    xyzp: (&[T], &[T], &[T]),
    xyzfil: (&[T], &[T], &[T]),
    dlxyzfil: (&[T], &[T], &[T]),
    ifil: &[T],
    wire_radius: &[T],
    construction_method: BuildMethod,
    theta: T,
    par: bool,
    out: (&mut [T], &mut [T], &mut [T]),
) -> Result<Diagnostics<LinearFilamentVectorPotentialKernel<T>>, HierarchicalError> {
    let sources = LinearFilamentSources::new(xyzfil, dlxyzfil, wire_radius);
    let targets = DipoleTargets::new(xyzp.0, xyzp.1, xyzp.2);
    one_shot_vec3(
        LinearFilamentVectorPotentialKernel::<T>::new(),
        sources,
        ifil,
        targets,
        construction_method,
        theta,
        par,
        None,
        out,
    )
}

/// Hierarchical magnetic flux density from a triangle mesh with nodal stream-function values.
///
/// This one-shot helper mirrors [`crate::physics::boundary_element::flux_density_triangle_mesh`]
/// while rebuilding the source tree internally.
///
/// This is an approximate method, and no particular accuracy level is guaranteed.
/// Truncated methods like this one may average entire local loop structures out of
/// existence; as a result, maximum relative error is unbounded. This method must be
/// tuned to a given use-case in order to be useful, and should not be used to calculate
/// safety-related field limits.
///
/// Direct triangle interactions use the exact uniform-triangle potential gradient.
/// A source triangle contributes zero at targets geometrically on that triangle.
///
/// References:
/// - D. R. Wilton, J. Rivero, W. A. Johnson, and F. Vipiana, “Evaluation of Static
///   Potential Integrals on Triangular Domains,” IEEE Access, vol. 8, pp. 99806–99819,
///   2020, doi: 10.1109/ACCESS.2020.2997287.
///
/// Args:
///     obs: Observation point coordinates.
///     mesh: Triangle mesh source geometry.
///     s: Nodal stream-function values.
///     construction_method: Source-tree construction method.
///     theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
///     par: Whether to evaluate target batches in parallel.
///     out: Output component slices to fill.
///
/// Returns:
///     Source-tree diagnostics and construction/evaluation timing on success.
///
/// Errors:
///     Returns [`HierarchicalError`] when input lengths are inconsistent, mesh conversion fails,
///     tree construction fails, scratch storage is too small, or a kernel reports an error.
pub fn flux_density_triangle_mesh_hierarchical(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    construction_method: BuildMethod,
    theta: f64,
    par: bool,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<Diagnostics<BoundaryElementFluxDensityKernel<f64>>, HierarchicalError> {
    mesh.validate_nodal_values(s)
        .map_err(|_| HierarchicalError::LengthMismatch)?;
    let sources = BoundaryElementTriangles::new(mesh);
    let moments = BoundaryElementNodalValues::new(sources, s);
    let targets = DipoleTargets::new(obs.0, obs.1, obs.2);
    one_shot_vec3(
        BoundaryElementFluxDensityKernel::<f64>::new(),
        sources,
        moments,
        targets,
        construction_method,
        theta,
        par,
        None,
        out,
    )
}

/// Hierarchical magnetic vector potential from a triangle mesh with nodal stream-function values.
///
/// This one-shot helper mirrors [`crate::physics::boundary_element::vector_potential_triangle_mesh`]
/// while rebuilding the source tree internally.
///
/// This is an approximate method, and no particular accuracy level is guaranteed.
/// Truncated methods like this one may average entire local loop structures out of
/// existence; as a result, maximum relative error is unbounded. This method must be
/// tuned to a given use-case in order to be useful, and should not be used to calculate
/// safety-related field limits.
///
/// Direct triangle interactions use the exact uniform-triangle potential, which
/// is finite and continuous on triangle interiors, edges, and vertices.
///
/// References:
/// - D. R. Wilton, J. Rivero, W. A. Johnson, and F. Vipiana, “Evaluation of Static
///   Potential Integrals on Triangular Domains,” IEEE Access, vol. 8, pp. 99806–99819,
///   2020, doi: 10.1109/ACCESS.2020.2997287.
///
/// Args:
///     obs: Observation point coordinates.
///     mesh: Triangle mesh source geometry.
///     s: Nodal stream-function values.
///     construction_method: Source-tree construction method.
///     theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
///     par: Whether to evaluate target batches in parallel.
///     out: Output component slices to fill.
///
/// Returns:
///     Source-tree diagnostics and construction/evaluation timing on success.
///
/// Errors:
///     Returns [`HierarchicalError`] when input lengths are inconsistent, mesh conversion fails,
///     tree construction fails, scratch storage is too small, or a kernel reports an error.
pub fn vector_potential_triangle_mesh_hierarchical(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    construction_method: BuildMethod,
    theta: f64,
    par: bool,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<Diagnostics<BoundaryElementVectorPotentialKernel<f64>>, HierarchicalError> {
    mesh.validate_nodal_values(s)
        .map_err(|_| HierarchicalError::LengthMismatch)?;
    let sources = BoundaryElementTriangles::new(mesh);
    let moments = BoundaryElementNodalValues::new(sources, s);
    let targets = DipoleTargets::new(obs.0, obs.1, obs.2);
    one_shot_vec3(
        BoundaryElementVectorPotentialKernel::<f64>::new(),
        sources,
        moments,
        targets,
        construction_method,
        theta,
        par,
        None,
        out,
    )
}

/// Filtered variant of [`flux_density_dipole_hierarchical`].
pub fn flux_density_dipole_hierarchical_with_skip<T: Scalar>(
    loc: (&[T], &[T], &[T]),
    moment: (&[T], &[T], &[T]),
    obs: (&[T], &[T], &[T]),
    outer_radius: &[T],
    construction_method: BuildMethod,
    theta: T,
    par: bool,
    skip: Skip,
    out: (&mut [T], &mut [T], &mut [T]),
) -> Result<Diagnostics<DipoleFluxDensityKernel<T>>, HierarchicalError> {
    let sources = DipoleSources::new(loc.0, loc.1, loc.2, outer_radius);
    let moments = DipoleMoments::new(moment.0, moment.1, moment.2);
    let targets = DipoleTargets::new(obs.0, obs.1, obs.2);
    one_shot_vec3(
        DipoleFluxDensityKernel::<T>::new(),
        sources,
        moments,
        targets,
        construction_method,
        theta,
        par,
        Some(skip),
        out,
    )
}

/// Filtered variant of [`vector_potential_dipole_hierarchical`].
pub fn vector_potential_dipole_hierarchical_with_skip<T: Scalar>(
    loc: (&[T], &[T], &[T]),
    moment: (&[T], &[T], &[T]),
    obs: (&[T], &[T], &[T]),
    outer_radius: &[T],
    construction_method: BuildMethod,
    theta: T,
    par: bool,
    skip: Skip,
    out: (&mut [T], &mut [T], &mut [T]),
) -> Result<Diagnostics<DipoleVectorPotentialKernel<T>>, HierarchicalError> {
    let sources = DipoleSources::new(loc.0, loc.1, loc.2, outer_radius);
    let moments = DipoleMoments::new(moment.0, moment.1, moment.2);
    let targets = DipoleTargets::new(obs.0, obs.1, obs.2);
    one_shot_vec3(
        DipoleVectorPotentialKernel::<T>::new(),
        sources,
        moments,
        targets,
        construction_method,
        theta,
        par,
        Some(skip),
        out,
    )
}

/// Filtered variant of [`flux_density_linear_filament_hierarchical`].
pub fn flux_density_linear_filament_hierarchical_with_skip<T: Scalar>(
    xyzp: (&[T], &[T], &[T]),
    xyzfil: (&[T], &[T], &[T]),
    dlxyzfil: (&[T], &[T], &[T]),
    ifil: &[T],
    wire_radius: &[T],
    construction_method: BuildMethod,
    theta: T,
    par: bool,
    skip: Skip,
    out: (&mut [T], &mut [T], &mut [T]),
) -> Result<Diagnostics<LinearFilamentFluxDensityKernel<T>>, HierarchicalError> {
    let sources = LinearFilamentSources::new(xyzfil, dlxyzfil, wire_radius);
    let targets = DipoleTargets::new(xyzp.0, xyzp.1, xyzp.2);
    one_shot_vec3(
        LinearFilamentFluxDensityKernel::<T>::new(),
        sources,
        ifil,
        targets,
        construction_method,
        theta,
        par,
        Some(skip),
        out,
    )
}

/// Filtered variant of [`vector_potential_linear_filament_hierarchical`].
pub fn vector_potential_linear_filament_hierarchical_with_skip<T: Scalar>(
    xyzp: (&[T], &[T], &[T]),
    xyzfil: (&[T], &[T], &[T]),
    dlxyzfil: (&[T], &[T], &[T]),
    ifil: &[T],
    wire_radius: &[T],
    construction_method: BuildMethod,
    theta: T,
    par: bool,
    skip: Skip,
    out: (&mut [T], &mut [T], &mut [T]),
) -> Result<Diagnostics<LinearFilamentVectorPotentialKernel<T>>, HierarchicalError> {
    let sources = LinearFilamentSources::new(xyzfil, dlxyzfil, wire_radius);
    let targets = DipoleTargets::new(xyzp.0, xyzp.1, xyzp.2);
    one_shot_vec3(
        LinearFilamentVectorPotentialKernel::<T>::new(),
        sources,
        ifil,
        targets,
        construction_method,
        theta,
        par,
        Some(skip),
        out,
    )
}

/// Filtered variant of [`flux_density_triangle_mesh_hierarchical`].
pub fn flux_density_triangle_mesh_hierarchical_with_skip(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    construction_method: BuildMethod,
    theta: f64,
    par: bool,
    skip: Skip,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<Diagnostics<BoundaryElementFluxDensityKernel<f64>>, HierarchicalError> {
    mesh.validate_nodal_values(s)
        .map_err(|_| HierarchicalError::LengthMismatch)?;
    let sources = BoundaryElementTriangles::new(mesh);
    let moments = BoundaryElementNodalValues::new(sources, s);
    let targets = DipoleTargets::new(obs.0, obs.1, obs.2);
    one_shot_vec3(
        BoundaryElementFluxDensityKernel::<f64>::new(),
        sources,
        moments,
        targets,
        construction_method,
        theta,
        par,
        Some(skip),
        out,
    )
}

/// Filtered variant of [`vector_potential_triangle_mesh_hierarchical`].
pub fn vector_potential_triangle_mesh_hierarchical_with_skip(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    construction_method: BuildMethod,
    theta: f64,
    par: bool,
    skip: Skip,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<Diagnostics<BoundaryElementVectorPotentialKernel<f64>>, HierarchicalError> {
    mesh.validate_nodal_values(s)
        .map_err(|_| HierarchicalError::LengthMismatch)?;
    let sources = BoundaryElementTriangles::new(mesh);
    let moments = BoundaryElementNodalValues::new(sources, s);
    let targets = DipoleTargets::new(obs.0, obs.1, obs.2);
    one_shot_vec3(
        BoundaryElementVectorPotentialKernel::<f64>::new(),
        sources,
        moments,
        targets,
        construction_method,
        theta,
        par,
        Some(skip),
        out,
    )
}

/// Build trees, evaluate a vector-valued hierarchical solve, and return diagnostics.
pub(crate) fn one_shot_vec3<K, T, S, M, C>(
    kernel: K,
    sources: S,
    moments: M,
    targets: C,
    construction_method: BuildMethod,
    theta: T,
    par: bool,
    skip: Option<Skip>,
    out: (&mut [T], &mut [T], &mut [T]),
) -> Result<Diagnostics<K>, HierarchicalError>
where
    K: HierarchicalKernel<Scalar = T, Output = [T; 3]> + Sync,
    T: Scalar,
    S: SourceCollection<K>,
    M: SourceMomentCollection<K>,
    K::TargetGeometry: Copy,
    C: TargetCollection<K>,
{
    if out.0.len() != targets.len()
        || out.1.len() != targets.len()
        || out.2.len() != targets.len()
        || !targets.valid_lengths()
        || sources.len() != moments.len()
        || !sources.valid_lengths()
        || !moments.valid_lengths()
    {
        return Err(HierarchicalError::LengthMismatch);
    }

    let construction_start = Instant::now();
    let source_tree = match construction_method {
        BuildMethod::LongestAxis => ClusterTree::build(sources)?,
        BuildMethod::MortonLbvh => ClusterTree::build_morton_lbvh(sources)?,
    };
    let source_summaries = if skip == Some(Skip::Both) {
        None
    } else {
        let mut source_summaries = SourceNodeSummaries::<K>::new(source_tree.as_view());
        let err = update_summaries(
            &kernel,
            source_tree.as_view(),
            sources,
            moments,
            &mut source_summaries.node_summaries,
        );
        if err != HierarchicalError::Ok {
            return Err(err);
        }
        Some(source_summaries)
    };
    let construction_seconds = construction_start.elapsed().as_secs_f64();

    let evaluation_start = Instant::now();
    if let Some(source_summaries) = source_summaries {
        let scratch_len = match par {
            true => scratch_len_par(targets.len()),
            false => scratch_len(),
        };
        let mut scratch_values = vec![[T::ZERO; 3]; scratch_len];
        let mut scratch = EvaluationScratch {
            contribution: &mut scratch_values,
        };
        let out_components = [out.0, out.1, out.2];
        let err = match (par, skip) {
            (true, Some(skip)) => eval_par_with_skip(
                &kernel,
                source_tree.as_view(),
                &source_summaries.node_summaries,
                sources,
                targets,
                moments,
                theta,
                skip,
                out_components,
                &mut scratch,
            ),
            (true, None) => eval_par(
                &kernel,
                source_tree.as_view(),
                &source_summaries.node_summaries,
                sources,
                targets,
                moments,
                theta,
                out_components,
                &mut scratch,
            ),
            (false, Some(skip)) => eval_with_skip(
                &kernel,
                source_tree.as_view(),
                &source_summaries.node_summaries,
                sources,
                targets,
                moments,
                theta,
                skip,
                out_components,
                &mut scratch,
            ),
            (false, None) => eval(
                &kernel,
                source_tree.as_view(),
                &source_summaries.node_summaries,
                sources,
                targets,
                moments,
                theta,
                out_components,
                &mut scratch,
            ),
        };
        if err != HierarchicalError::Ok {
            return Err(err);
        }
    } else {
        out.0.fill(T::ZERO);
        out.1.fill(T::ZERO);
        out.2.fill(T::ZERO);
    }
    let evaluation_seconds = evaluation_start.elapsed().as_secs_f64();

    Ok(Diagnostics {
        source_tree,
        construction_seconds,
        evaluation_seconds,
        source_count: sources.len(),
        target_count: targets.len(),
    })
}
