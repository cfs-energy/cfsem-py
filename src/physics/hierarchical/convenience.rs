//! One-shot hierarchical field-solver convenience functions.
//!
//! These helpers intentionally rebuild the source tree every call. For the
//! operating range where the hierarchical solve wins over the dense direct
//! methods, tree construction is usually a small part of total runtime, so this
//! API favors direct-method ergonomics over solver object reuse.

use crate::mesh::triangle3d::TriangleMeshView;
use crate::physics::boundary_element::QuadratureKind;

use super::kernels::{
    BoundaryElementFluxDensityKernel, BoundaryElementTriangle,
    BoundaryElementVectorPotentialKernel, DipoleFluxDensityKernel, DipoleSource, DipoleTarget,
    DipoleVectorPotentialKernel, LinearFilamentFluxDensityKernel, LinearFilamentSource,
    LinearFilamentVectorPotentialKernel,
};
use super::{
    ClusterTree, EvaluationScratch, HierarchicalError, HierarchicalKernel, SourceNodeSummaries,
    evaluate_source_tree_into, evaluate_source_tree_into_par,
    parallel_source_tree_evaluation_scratch_len, source_tree_evaluation_scratch_len,
    update_source_summaries_into,
};

/// Hierarchical magnetic flux density of dipole sources at Cartesian targets.
///
/// This one-shot helper mirrors the direct dipole API while rebuilding the
/// source tree internally. Use the reusable solver pieces directly when many
/// source-magnitude updates share fixed geometry.
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
///     outer_radius: Radius for the magnetized-sphere near-field treatment.
///     obs: Observation point coordinates.
///     theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
///     par: Whether to evaluate target batches in parallel.
///     out: Output component slices to fill.
///
/// Returns:
///     Nothing on success.
///
/// Errors:
///     Returns [`HierarchicalError`] when input lengths are inconsistent, tree construction fails,
///     scratch storage is too small, or a kernel reports an error.
pub fn flux_density_dipole_hierarchical(
    loc: (&[f64], &[f64], &[f64]),
    moment: (&[f64], &[f64], &[f64]),
    outer_radius: &[f64],
    obs: (&[f64], &[f64], &[f64]),
    theta: f64,
    par: bool,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), HierarchicalError> {
    let sources = dipole_sources_from_slices(loc, outer_radius)?;
    let moments = vec3_from_slices(moment, sources.len())?;
    let targets = dipole_targets_from_slices(obs)?;
    one_shot_vec3(
        DipoleFluxDensityKernel::<f64>::new(),
        &sources,
        &moments,
        &targets,
        theta,
        par,
        out,
    )
}

/// Hierarchical magnetic vector potential of dipole sources at Cartesian targets.
///
/// This one-shot helper mirrors the direct dipole API while rebuilding the
/// source tree internally. Use the reusable solver pieces directly when many
/// source-magnitude updates share fixed geometry.
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
///     outer_radius: Radius for the magnetized-sphere near-field treatment.
///     obs: Observation point coordinates.
///     theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
///     par: Whether to evaluate target batches in parallel.
///     out: Output component slices to fill.
///
/// Returns:
///     Nothing on success.
///
/// Errors:
///     Returns [`HierarchicalError`] when input lengths are inconsistent, tree construction fails,
///     scratch storage is too small, or a kernel reports an error.
pub fn vector_potential_dipole_hierarchical(
    loc: (&[f64], &[f64], &[f64]),
    moment: (&[f64], &[f64], &[f64]),
    outer_radius: &[f64],
    obs: (&[f64], &[f64], &[f64]),
    theta: f64,
    par: bool,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), HierarchicalError> {
    let sources = dipole_sources_from_slices(loc, outer_radius)?;
    let moments = vec3_from_slices(moment, sources.len())?;
    let targets = dipole_targets_from_slices(obs)?;
    one_shot_vec3(
        DipoleVectorPotentialKernel::<f64>::new(),
        &sources,
        &moments,
        &targets,
        theta,
        par,
        out,
    )
}

/// Hierarchical magnetic flux density of linear filament segments.
///
/// This one-shot helper mirrors [`crate::physics::linear_filament::flux_density_linear_filament`]
/// while rebuilding the source tree internally. Use the reusable solver pieces
/// directly when many current updates share fixed geometry.
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
///     theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
///     par: Whether to evaluate target batches in parallel.
///     out: Output component slices to fill.
///
/// Returns:
///     Nothing on success.
///
/// Errors:
///     Returns [`HierarchicalError`] when input lengths are inconsistent, tree construction fails,
///     scratch storage is too small, or a kernel reports an error.
pub fn flux_density_linear_filament_hierarchical(
    xyzp: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
    wire_radius: &[f64],
    theta: f64,
    par: bool,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), HierarchicalError> {
    let sources = linear_filament_sources_from_slices(xyzfil, dlxyzfil, wire_radius)?;
    if ifil.len() != sources.len() {
        return Err(HierarchicalError::LengthMismatch);
    }
    let targets = dipole_targets_from_slices(xyzp)?;
    one_shot_vec3(
        LinearFilamentFluxDensityKernel::<f64>::new(),
        &sources,
        ifil,
        &targets,
        theta,
        par,
        out,
    )
}

/// Hierarchical magnetic vector potential of linear filament segments.
///
/// This one-shot helper mirrors [`crate::physics::linear_filament::vector_potential_linear_filament`]
/// while rebuilding the source tree internally. Use the reusable solver pieces
/// directly when many current updates share fixed geometry.
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
///     theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
///     par: Whether to evaluate target batches in parallel.
///     out: Output component slices to fill.
///
/// Returns:
///     Nothing on success.
///
/// Errors:
///     Returns [`HierarchicalError`] when input lengths are inconsistent, tree construction fails,
///     scratch storage is too small, or a kernel reports an error.
pub fn vector_potential_linear_filament_hierarchical(
    xyzp: (&[f64], &[f64], &[f64]),
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    ifil: &[f64],
    wire_radius: &[f64],
    theta: f64,
    par: bool,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), HierarchicalError> {
    let sources = linear_filament_sources_from_slices(xyzfil, dlxyzfil, wire_radius)?;
    if ifil.len() != sources.len() {
        return Err(HierarchicalError::LengthMismatch);
    }
    let targets = dipole_targets_from_slices(xyzp)?;
    one_shot_vec3(
        LinearFilamentVectorPotentialKernel::<f64>::new(),
        &sources,
        ifil,
        &targets,
        theta,
        par,
        out,
    )
}

/// Hierarchical magnetic flux density from a triangle mesh with nodal stream-function values.
///
/// This one-shot helper mirrors [`crate::physics::boundary_element::flux_density_triangle_mesh`]
/// while rebuilding the source tree internally. Use the reusable solver pieces
/// directly when many source-value updates share fixed geometry.
///
/// This is an approximate method, and no particular accuracy level is guaranteed.
/// Truncated methods like this one may average entire local loop structures out of
/// existence; as a result, maximum relative error is unbounded. This method must be
/// tuned to a given use-case in order to be useful, and should not be used to calculate
/// safety-related field limits.
///
/// Args:
///     obs: Observation point coordinates.
///     mesh: Triangle mesh source geometry.
///     s: Nodal stream-function values.
///     quad_kind: Triangle quadrature rule.
///     theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
///     par: Whether to evaluate target batches in parallel.
///     out: Output component slices to fill.
///
/// Returns:
///     Nothing on success.
///
/// Errors:
///     Returns [`HierarchicalError`] when input lengths are inconsistent, mesh conversion fails,
///     tree construction fails, scratch storage is too small, or a kernel reports an error.
pub fn flux_density_triangle_mesh_hierarchical(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    quad_kind: QuadratureKind,
    theta: f64,
    par: bool,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), HierarchicalError> {
    let (sources, moments) = boundary_element_sources_from_mesh(mesh, s)?;
    let targets = dipole_targets_from_slices(obs)?;
    one_shot_vec3(
        BoundaryElementFluxDensityKernel::<f64>::new(quad_kind),
        &sources,
        &moments,
        &targets,
        theta,
        par,
        out,
    )
}

/// Hierarchical magnetic vector potential from a triangle mesh with nodal stream-function values.
///
/// This one-shot helper mirrors [`crate::physics::boundary_element::vector_potential_triangle_mesh`]
/// while rebuilding the source tree internally. Use the reusable solver pieces
/// directly when many source-value updates share fixed geometry.
///
/// This is an approximate method, and no particular accuracy level is guaranteed.
/// Truncated methods like this one may average entire local loop structures out of
/// existence; as a result, maximum relative error is unbounded. This method must be
/// tuned to a given use-case in order to be useful, and should not be used to calculate
/// safety-related field limits.
///
/// Args:
///     obs: Observation point coordinates.
///     mesh: Triangle mesh source geometry.
///     s: Nodal stream-function values.
///     quad_kind: Triangle quadrature rule.
///     theta: Barnes-Hut acceptance angle. Smaller values are more accurate and slower.
///     par: Whether to evaluate target batches in parallel.
///     out: Output component slices to fill.
///
/// Returns:
///     Nothing on success.
///
/// Errors:
///     Returns [`HierarchicalError`] when input lengths are inconsistent, mesh conversion fails,
///     tree construction fails, scratch storage is too small, or a kernel reports an error.
pub fn vector_potential_triangle_mesh_hierarchical(
    obs: (&[f64], &[f64], &[f64]),
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    quad_kind: QuadratureKind,
    theta: f64,
    par: bool,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), HierarchicalError> {
    let (sources, moments) = boundary_element_sources_from_mesh(mesh, s)?;
    let targets = dipole_targets_from_slices(obs)?;
    one_shot_vec3(
        BoundaryElementVectorPotentialKernel::<f64>::new(quad_kind),
        &sources,
        &moments,
        &targets,
        theta,
        par,
        out,
    )
}

fn one_shot_vec3<K>(
    kernel: K,
    sources: &[K::SourceGeometry],
    moments: &[K::SourceMoment],
    targets: &[K::TargetGeometry],
    theta: f64,
    par: bool,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), HierarchicalError>
where
    K: HierarchicalKernel<Scalar = f64, Output = [f64; 3]> + Sync,
    K::TargetGeometry: Copy,
{
    if out.0.len() != targets.len() || out.1.len() != targets.len() || out.2.len() != targets.len()
    {
        return Err(HierarchicalError::LengthMismatch);
    }

    let source_tree = ClusterTree::build(sources)?;
    let mut source_summaries = SourceNodeSummaries::<K>::new(source_tree.as_view());
    let mut err = update_source_summaries_into(
        &kernel,
        source_tree.as_view(),
        sources,
        moments,
        &mut source_summaries.node_summaries,
    );
    if err != HierarchicalError::Ok {
        return Err(err);
    }

    let mut values = vec![[0.0; 3]; targets.len()];
    let scratch_len = match par {
        true => parallel_source_tree_evaluation_scratch_len(targets.len()),
        false => source_tree_evaluation_scratch_len(),
    };
    let mut scratch_values = vec![[0.0; 3]; scratch_len];
    let mut scratch = EvaluationScratch {
        contribution: &mut scratch_values,
    };
    err = match par {
        true => evaluate_source_tree_into_par(
            &kernel,
            source_tree.as_view(),
            &source_summaries.node_summaries,
            sources,
            targets,
            moments,
            theta,
            &mut values,
            &mut scratch,
        ),
        false => evaluate_source_tree_into(
            &kernel,
            source_tree.as_view(),
            &source_summaries.node_summaries,
            sources,
            targets,
            moments,
            theta,
            &mut values,
            &mut scratch,
        ),
    };
    if err != HierarchicalError::Ok {
        return Err(err);
    }

    for i in 0..values.len() {
        out.0[i] = values[i][0];
        out.1[i] = values[i][1];
        out.2[i] = values[i][2];
    }
    Ok(())
}

fn dipole_targets_from_slices(
    points: (&[f64], &[f64], &[f64]),
) -> Result<Vec<DipoleTarget<f64>>, HierarchicalError> {
    if points.0.len() != points.1.len() || points.0.len() != points.2.len() {
        return Err(HierarchicalError::LengthMismatch);
    }
    let mut targets = Vec::with_capacity(points.0.len());
    for i in 0..points.0.len() {
        targets.push(DipoleTarget {
            position: [points.0[i], points.1[i], points.2[i]],
        });
    }
    Ok(targets)
}

fn dipole_sources_from_slices(
    loc: (&[f64], &[f64], &[f64]),
    outer_radius: &[f64],
) -> Result<Vec<DipoleSource<f64>>, HierarchicalError> {
    if loc.0.len() != loc.1.len() || loc.0.len() != loc.2.len() || loc.0.len() != outer_radius.len()
    {
        return Err(HierarchicalError::LengthMismatch);
    }
    let mut sources = Vec::with_capacity(loc.0.len());
    for i in 0..loc.0.len() {
        sources.push(DipoleSource {
            position: [loc.0[i], loc.1[i], loc.2[i]],
            outer_radius: outer_radius[i],
        });
    }
    Ok(sources)
}

fn linear_filament_sources_from_slices(
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
) -> Result<Vec<LinearFilamentSource<f64>>, HierarchicalError> {
    let n = xyzfil.0.len();
    if xyzfil.1.len() != n
        || xyzfil.2.len() != n
        || dlxyzfil.0.len() != n
        || dlxyzfil.1.len() != n
        || dlxyzfil.2.len() != n
        || wire_radius.len() != n
    {
        return Err(HierarchicalError::LengthMismatch);
    }
    let mut sources = Vec::with_capacity(n);
    for i in 0..n {
        sources.push(LinearFilamentSource {
            start: [xyzfil.0[i], xyzfil.1[i], xyzfil.2[i]],
            end: [
                xyzfil.0[i] + dlxyzfil.0[i],
                xyzfil.1[i] + dlxyzfil.1[i],
                xyzfil.2[i] + dlxyzfil.2[i],
            ],
            wire_radius: wire_radius[i],
        });
    }
    Ok(sources)
}

fn boundary_element_sources_from_mesh(
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
) -> Result<(Vec<BoundaryElementTriangle<f64>>, Vec<[f64; 3]>), HierarchicalError> {
    mesh.validate_nodal_values(s)
        .map_err(|_| HierarchicalError::LengthMismatch)?;
    let mut sources = Vec::with_capacity(mesh.len());
    let mut moments = Vec::with_capacity(mesh.len());
    for i in 0..mesh.len() {
        let nodes = mesh.triangle_nodes(i);
        sources.push(BoundaryElementTriangle {
            n0: nodes[0],
            n1: nodes[1],
            n2: nodes[2],
        });
        moments.push(mesh.triangle_scalars(i, s));
    }
    Ok((sources, moments))
}

fn vec3_from_slices(
    values: (&[f64], &[f64], &[f64]),
    n: usize,
) -> Result<Vec<[f64; 3]>, HierarchicalError> {
    if values.0.len() != n || values.1.len() != n || values.2.len() != n {
        return Err(HierarchicalError::LengthMismatch);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push([values.0[i], values.1[i], values.2[i]]);
    }
    Ok(out)
}
