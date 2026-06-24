use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use super::{
    QuadratureKind, TRIANGLE_MAX_QUADRATURE_POINTS, calc_tri_area, map_tri_uv,
    triangle_basis_current_densities, triangle_current_density, triangle_flux_density_basis,
    triangle_quadrature_points,
};
use crate::math::cross3;
use crate::mesh::TriangleMeshView;
use crate::physics::circular_filament::flux_density_circular_filament_cartesian_scalar;
use crate::physics::linear_filament::flux_density_linear_filament_scalar;
use crate::physics::point_source::dipole::flux_density_dipole_scalar;

#[inline]
fn validate_force_mapping_inputs(
    outx: &[f64],
    outy: &[f64],
    outz: &[f64],
    ntri_tgt: usize,
    nsrc: usize,
) -> Result<(), &'static str> {
    let expected = ntri_tgt
        .checked_mul(nsrc)
        .ok_or("Force mapping size overflow")?;
    if outx.len() != expected || outy.len() != expected || outz.len() != expected {
        return Err("Output dimension mismatch");
    }
    Ok(())
}

#[inline]
fn validate_force_mapping_vector_inputs(
    fx: &[f64],
    fy: &[f64],
    fz: &[f64],
    s_src: &[f64],
) -> Result<usize, &'static str> {
    let nsrc = s_src.len(); // [-]
    if fy.len() != fx.len() || fz.len() != fx.len() {
        return Err("Force mapping dimension mismatch");
    }
    if nsrc == 0 {
        if fx.is_empty() {
            return Ok(0);
        }
        return Err("Force mapping dimension mismatch");
    }
    if !fx.len().is_multiple_of(nsrc) {
        return Err("Force mapping dimension mismatch");
    }
    Ok(fx.len() / nsrc)
}

#[inline]
fn triangle_flux_density_basis_integral_on_triangle(
    src0: [f64; 3],
    src1: [f64; 3],
    src2: [f64; 3],
    tgt0: [f64; 3],
    tgt1: [f64; 3],
    tgt2: [f64; 3],
    quad_kind: QuadratureKind,
) -> [f64; 3] {
    let tri_area_tgt = calc_tri_area(tgt0, tgt1, tgt2); // [m^2]
    let quad_points_tgt = triangle_quadrature_points(quad_kind);

    let mut out = [0.0; 3]; // [T*m^2/A]
    for qp in quad_points_tgt {
        let obs = map_tri_uv(tgt0, tgt1, tgt2, [qp[1], qp[2]]); // [m]
        let b = triangle_flux_density_basis(src0, src1, src2, obs, quad_kind); // [T/A]
        let w = qp[0] * tri_area_tgt; // [m^2]
        out[0] += b[0] * w; // [T*m^2/A]
        out[1] += b[1] * w; // [T*m^2/A]
        out[2] += b[2] * w; // [T*m^2/A]
    }

    out
}

#[inline]
fn integrate_target_triangle_force_from_bfield_fn<F>(
    tri_nodes: [[f64; 3]; 3],
    tri_s: [f64; 3],
    tri_owner: usize,
    quad_kind: QuadratureKind,
    eval_b: F,
) -> Result<[f64; 3], &'static str>
where
    F: Fn(
        usize,
        (&[f64], &[f64], &[f64]),
        (&mut [f64], &mut [f64], &mut [f64]),
    ) -> Result<(), &'static str>,
{
    let tri_area = calc_tri_area(tri_nodes[0], tri_nodes[1], tri_nodes[2]); // [m^2]
    let quad_points = triangle_quadrature_points(quad_kind);
    let nqp = quad_points.len(); // [-]
    let k_tgt = triangle_current_density(tri_nodes[0], tri_nodes[1], tri_nodes[2], tri_s); // [A/m]

    let mut xq = [0.0; TRIANGLE_MAX_QUADRATURE_POINTS];
    let mut yq = [0.0; TRIANGLE_MAX_QUADRATURE_POINTS];
    let mut zq = [0.0; TRIANGLE_MAX_QUADRATURE_POINTS];
    let mut wq = [0.0; TRIANGLE_MAX_QUADRATURE_POINTS];
    for (iqp, qp) in quad_points.iter().enumerate() {
        let obs = map_tri_uv(tri_nodes[0], tri_nodes[1], tri_nodes[2], [qp[1], qp[2]]); // [m]
        xq[iqp] = obs[0]; // [m]
        yq[iqp] = obs[1]; // [m]
        zq[iqp] = obs[2]; // [m]
        wq[iqp] = qp[0] * tri_area; // [m^2]
    }

    let mut bx = [0.0; TRIANGLE_MAX_QUADRATURE_POINTS];
    let mut by = [0.0; TRIANGLE_MAX_QUADRATURE_POINTS];
    let mut bz = [0.0; TRIANGLE_MAX_QUADRATURE_POINTS];
    eval_b(
        tri_owner,
        (&xq[..nqp], &yq[..nqp], &zq[..nqp]),
        (&mut bx[..nqp], &mut by[..nqp], &mut bz[..nqp]),
    )?;

    let mut out = [0.0; 3]; // [N / source-unit]
    for iqp in 0..nqp {
        let f = cross3(k_tgt, [bx[iqp], by[iqp], bz[iqp]]); // [N/m^3]
        out[0] += f[0] * wq[iqp]; // [N / source-unit]
        out[1] += f[1] * wq[iqp]; // [N / source-unit]
        out[2] += f[2] * wq[iqp]; // [N / source-unit]
    }

    Ok(out)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn triangle_mesh_force_mapping_row(
    mesh_src: &TriangleMeshView<'_>,
    tgt_nodes: [[f64; 3]; 3],
    tgt_s: [f64; 3],
    skip_identical_src: Option<usize>,
    quad_kind: QuadratureKind,
    outx: &mut [f64],
    outy: &mut [f64],
    outz: &mut [f64],
) {
    outx.fill(0.0); // [N/A]
    outy.fill(0.0); // [N/A]
    outz.fill(0.0); // [N/A]

    for isrc in 0..mesh_src.len() {
        if skip_identical_src == Some(isrc) {
            continue;
        }

        let (src_nodes, src_idx) = mesh_src.triangle_nodes_and_indices(isrc);
        let block = triangle_basis_force_block(
            src_nodes[0],
            src_nodes[1],
            src_nodes[2],
            tgt_nodes[0],
            tgt_nodes[1],
            tgt_nodes[2],
            quad_kind,
        );

        for isrc_basis in 0..3 {
            let col = src_idx[isrc_basis]; // [-]
            for itgt_basis in 0..3 {
                outx[col] += tgt_s[itgt_basis] * block[itgt_basis][isrc_basis][0]; // [N/A]
                outy[col] += tgt_s[itgt_basis] * block[itgt_basis][isrc_basis][1]; // [N/A]
                outz[col] += tgt_s[itgt_basis] * block[itgt_basis][isrc_basis][2]; // [N/A]
            }
        }
    }
}

/// Force block for the three target-triangle basis functions due to the three source-triangle
/// basis functions.
///
/// Args:
///     src0: Source triangle vertex 0 `[x, y, z]` (m).
///     src1: Source triangle vertex 1 `[x, y, z]` (m).
///     src2: Source triangle vertex 2 `[x, y, z]` (m).
///     tgt0: Target triangle vertex 0 `[x, y, z]` (m).
///     tgt1: Target triangle vertex 1 `[x, y, z]` (m).
///     tgt2: Target triangle vertex 2 `[x, y, z]` (m).
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///
/// Returns:
///     Elemental force block indexed as `[target_basis][source_basis][component]` (N/A^2).
#[inline]
pub fn triangle_basis_force_block(
    src0: [f64; 3],
    src1: [f64; 3],
    src2: [f64; 3],
    tgt0: [f64; 3],
    tgt1: [f64; 3],
    tgt2: [f64; 3],
    quad_kind: QuadratureKind,
) -> [[[f64; 3]; 3]; 3] {
    let bint = [
        triangle_flux_density_basis_integral_on_triangle(
            src0, src1, src2, tgt0, tgt1, tgt2, quad_kind,
        ),
        triangle_flux_density_basis_integral_on_triangle(
            src1, src2, src0, tgt0, tgt1, tgt2, quad_kind,
        ),
        triangle_flux_density_basis_integral_on_triangle(
            src2, src0, src1, tgt0, tgt1, tgt2, quad_kind,
        ),
    ];
    let ktgt = triangle_basis_current_densities(tgt0, tgt1, tgt2); // [1/m]

    let mut out = [[[0.0; 3]; 3]; 3];
    for itgt_basis in 0..3 {
        for isrc_basis in 0..3 {
            out[itgt_basis][isrc_basis] = cross3(ktgt[itgt_basis], bint[isrc_basis]); // [N/A^2]
        }
    }

    out
}

/// Contract one elemental force block with source and target nodal current-potential values.
///
/// Args:
///     block: Elemental force block `[target_basis][source_basis][component]` (N/A^2).
///     s_src: Source nodal current-potential values `[s0, s1, s2]` (A).
///     s_tgt: Target nodal current-potential values `[s0, s1, s2]` (A).
///
/// Returns:
///     Integrated triangle-to-triangle force `[fx, fy, fz]` (N).
#[inline]
pub fn triangle_force_from_potential_vectors(
    block: [[[f64; 3]; 3]; 3],
    s_src: [f64; 3],
    s_tgt: [f64; 3],
) -> [f64; 3] {
    let mut out = [0.0; 3]; // [N]
    for itgt_basis in 0..3 {
        for isrc_basis in 0..3 {
            out[0] += s_tgt[itgt_basis] * block[itgt_basis][isrc_basis][0] * s_src[isrc_basis]; // [N]
            out[1] += s_tgt[itgt_basis] * block[itgt_basis][isrc_basis][1] * s_src[isrc_basis]; // [N]
            out[2] += s_tgt[itgt_basis] * block[itgt_basis][isrc_basis][2] * s_src[isrc_basis]; // [N]
        }
    }
    out
}

/// Assemble the frozen-target source-node to target-triangle force mapping between two
/// triangle meshes.
///
/// Args:
///     mesh_src: Borrowed source triangle-mesh geometry view.
///     mesh_tgt: Borrowed target triangle-mesh geometry view.
///     s_tgt: Fixed target nodal current-potential values (A).
///     quad_kind: Triangle quadrature rule selector (dimensionless).
///     out: Output force mapping buffers `(fx_map, fy_map, fz_map)` (N/A), each row-major
///         in `(target triangle, source node)` order.
///
/// Returns:
///     `Ok(())` after writing the frozen-target force mapping.
#[inline]
pub fn triangle_mesh_force_mapping(
    mesh_src: &TriangleMeshView<'_>,
    mesh_tgt: &TriangleMeshView<'_>,
    s_tgt: &[f64],
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    mesh_tgt.validate_nodal_values(s_tgt)?;
    validate_force_mapping_inputs(out.0, out.1, out.2, mesh_tgt.len(), mesh_src.nnode())?;

    for itgt in 0..mesh_tgt.len() {
        let tgt_nodes = mesh_tgt.triangle_nodes(itgt);
        let tgt_s = mesh_tgt.triangle_scalars(itgt, s_tgt);
        let rowx = &mut out.0[itgt * mesh_src.nnode()..(itgt + 1) * mesh_src.nnode()];
        let rowy = &mut out.1[itgt * mesh_src.nnode()..(itgt + 1) * mesh_src.nnode()];
        let rowz = &mut out.2[itgt * mesh_src.nnode()..(itgt + 1) * mesh_src.nnode()];
        triangle_mesh_force_mapping_row(
            mesh_src, tgt_nodes, tgt_s, None, quad_kind, rowx, rowy, rowz,
        );
    }

    Ok(())
}

/// Parallel variant of [`triangle_mesh_force_mapping`].
#[inline]
pub fn triangle_mesh_force_mapping_par(
    mesh_src: &TriangleMeshView<'_>,
    mesh_tgt: &TriangleMeshView<'_>,
    s_tgt: &[f64],
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    mesh_tgt.validate_nodal_values(s_tgt)?;
    validate_force_mapping_inputs(out.0, out.1, out.2, mesh_tgt.len(), mesh_src.nnode())?;

    out.0
        .par_chunks_mut(mesh_src.nnode())
        .zip_eq(out.1.par_chunks_mut(mesh_src.nnode()))
        .zip_eq(out.2.par_chunks_mut(mesh_src.nnode()))
        .enumerate()
        .try_for_each(|(itgt, ((rowx, rowy), rowz))| {
            let tgt_nodes = mesh_tgt.triangle_nodes(itgt);
            let tgt_s = mesh_tgt.triangle_scalars(itgt, s_tgt);
            triangle_mesh_force_mapping_row(
                mesh_src, tgt_nodes, tgt_s, None, quad_kind, rowx, rowy, rowz,
            );
            Ok::<(), &'static str>(())
        })?;

    Ok(())
}

/// Assemble the self-excluded frozen-target source-node to target-triangle force mapping for
/// one mesh.
///
/// The identical source-triangle contribution is omitted for each target-triangle row.
#[inline]
pub fn triangle_mesh_self_force_mapping(
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    mesh.validate_nodal_values(s)?;
    validate_force_mapping_inputs(out.0, out.1, out.2, mesh.len(), mesh.nnode())?;

    for itgt in 0..mesh.len() {
        let tgt_nodes = mesh.triangle_nodes(itgt);
        let tgt_s = mesh.triangle_scalars(itgt, s);
        let rowx = &mut out.0[itgt * mesh.nnode()..(itgt + 1) * mesh.nnode()];
        let rowy = &mut out.1[itgt * mesh.nnode()..(itgt + 1) * mesh.nnode()];
        let rowz = &mut out.2[itgt * mesh.nnode()..(itgt + 1) * mesh.nnode()];
        triangle_mesh_force_mapping_row(
            mesh,
            tgt_nodes,
            tgt_s,
            Some(itgt),
            quad_kind,
            rowx,
            rowy,
            rowz,
        );
    }

    Ok(())
}

/// Parallel variant of [`triangle_mesh_self_force_mapping`].
#[inline]
pub fn triangle_mesh_self_force_mapping_par(
    mesh: &TriangleMeshView<'_>,
    s: &[f64],
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    mesh.validate_nodal_values(s)?;
    validate_force_mapping_inputs(out.0, out.1, out.2, mesh.len(), mesh.nnode())?;

    out.0
        .par_chunks_mut(mesh.nnode())
        .zip_eq(out.1.par_chunks_mut(mesh.nnode()))
        .zip_eq(out.2.par_chunks_mut(mesh.nnode()))
        .enumerate()
        .try_for_each(|(itgt, ((rowx, rowy), rowz))| {
            let tgt_nodes = mesh.triangle_nodes(itgt);
            let tgt_s = mesh.triangle_scalars(itgt, s);
            triangle_mesh_force_mapping_row(
                mesh,
                tgt_nodes,
                tgt_s,
                Some(itgt),
                quad_kind,
                rowx,
                rowy,
                rowz,
            );
            Ok::<(), &'static str>(())
        })?;

    Ok(())
}

/// Assemble the frozen-target source-current to target-triangle force mapping for linear
/// filament sources.
#[inline]
pub fn triangle_mesh_force_mapping_from_linear_filaments(
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    mesh_tgt: &TriangleMeshView<'_>,
    s_tgt: &[f64],
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let nfil = xyzfil.0.len(); // [-]
    if xyzfil.1.len() != nfil
        || xyzfil.2.len() != nfil
        || dlxyzfil.0.len() != nfil
        || dlxyzfil.1.len() != nfil
        || dlxyzfil.2.len() != nfil
        || wire_radius.len() != nfil
    {
        return Err("Length mismatch");
    }

    mesh_tgt.validate_nodal_values(s_tgt)?;
    validate_force_mapping_inputs(out.0, out.1, out.2, mesh_tgt.len(), nfil)?;

    for itgt in 0..mesh_tgt.len() {
        let tgt_nodes = mesh_tgt.triangle_nodes(itgt);
        let tgt_s = mesh_tgt.triangle_scalars(itgt, s_tgt);
        let rowx = &mut out.0[itgt * nfil..(itgt + 1) * nfil];
        let rowy = &mut out.1[itgt * nfil..(itgt + 1) * nfil];
        let rowz = &mut out.2[itgt * nfil..(itgt + 1) * nfil];
        rowx.fill(0.0); // [N/A]
        rowy.fill(0.0); // [N/A]
        rowz.fill(0.0); // [N/A]

        for ifil in 0..nfil {
            let fil0 = (xyzfil.0[ifil], xyzfil.1[ifil], xyzfil.2[ifil]); // [m]
            let fil1 = (
                fil0.0 + dlxyzfil.0[ifil],
                fil0.1 + dlxyzfil.1[ifil],
                fil0.2 + dlxyzfil.2[ifil],
            ); // [m]
            let force = integrate_target_triangle_force_from_bfield_fn(
                tgt_nodes,
                tgt_s,
                itgt,
                quad_kind,
                |_, obs, bout| {
                    for i in 0..obs.0.len() {
                        let b = flux_density_linear_filament_scalar(
                            (fil0, fil1, 1.0),
                            wire_radius[ifil],
                            (obs.0[i], obs.1[i], obs.2[i]),
                        );
                        bout.0[i] = b.0; // [T/A]
                        bout.1[i] = b.1; // [T/A]
                        bout.2[i] = b.2; // [T/A]
                    }
                    Ok(())
                },
            )?;
            rowx[ifil] = force[0]; // [N/A]
            rowy[ifil] = force[1]; // [N/A]
            rowz[ifil] = force[2]; // [N/A]
        }
    }

    Ok(())
}

/// Parallel variant of [`triangle_mesh_force_mapping_from_linear_filaments`].
#[inline]
pub fn triangle_mesh_force_mapping_from_linear_filaments_par(
    xyzfil: (&[f64], &[f64], &[f64]),
    dlxyzfil: (&[f64], &[f64], &[f64]),
    wire_radius: &[f64],
    mesh_tgt: &TriangleMeshView<'_>,
    s_tgt: &[f64],
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let nfil = xyzfil.0.len(); // [-]
    if xyzfil.1.len() != nfil
        || xyzfil.2.len() != nfil
        || dlxyzfil.0.len() != nfil
        || dlxyzfil.1.len() != nfil
        || dlxyzfil.2.len() != nfil
        || wire_radius.len() != nfil
    {
        return Err("Length mismatch");
    }

    mesh_tgt.validate_nodal_values(s_tgt)?;
    validate_force_mapping_inputs(out.0, out.1, out.2, mesh_tgt.len(), nfil)?;

    out.0
        .par_chunks_mut(nfil)
        .zip_eq(out.1.par_chunks_mut(nfil))
        .zip_eq(out.2.par_chunks_mut(nfil))
        .enumerate()
        .try_for_each(|(itgt, ((rowx, rowy), rowz))| {
            let tgt_nodes = mesh_tgt.triangle_nodes(itgt);
            let tgt_s = mesh_tgt.triangle_scalars(itgt, s_tgt);
            rowx.fill(0.0);
            rowy.fill(0.0);
            rowz.fill(0.0);

            for ifil in 0..nfil {
                let fil0 = (xyzfil.0[ifil], xyzfil.1[ifil], xyzfil.2[ifil]);
                let fil1 = (
                    fil0.0 + dlxyzfil.0[ifil],
                    fil0.1 + dlxyzfil.1[ifil],
                    fil0.2 + dlxyzfil.2[ifil],
                );
                let force = integrate_target_triangle_force_from_bfield_fn(
                    tgt_nodes,
                    tgt_s,
                    itgt,
                    quad_kind,
                    |_, obs, bout| {
                        for i in 0..obs.0.len() {
                            let b = flux_density_linear_filament_scalar(
                                (fil0, fil1, 1.0),
                                wire_radius[ifil],
                                (obs.0[i], obs.1[i], obs.2[i]),
                            );
                            bout.0[i] = b.0;
                            bout.1[i] = b.1;
                            bout.2[i] = b.2;
                        }
                        Ok::<(), &'static str>(())
                    },
                )?;
                rowx[ifil] = force[0];
                rowy[ifil] = force[1];
                rowz[ifil] = force[2];
            }

            Ok::<(), &'static str>(())
        })?;

    Ok(())
}

/// Assemble the frozen-target source-current to target-triangle force mapping for circular
/// filament sources.
#[inline]
pub fn triangle_mesh_force_mapping_from_circular_filaments(
    rfil: &[f64],
    zfil: &[f64],
    mesh_tgt: &TriangleMeshView<'_>,
    s_tgt: &[f64],
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let nfil = rfil.len(); // [-]
    if zfil.len() != nfil {
        return Err("Length mismatch");
    }

    mesh_tgt.validate_nodal_values(s_tgt)?;
    validate_force_mapping_inputs(out.0, out.1, out.2, mesh_tgt.len(), nfil)?;

    for itgt in 0..mesh_tgt.len() {
        let tgt_nodes = mesh_tgt.triangle_nodes(itgt);
        let tgt_s = mesh_tgt.triangle_scalars(itgt, s_tgt);
        let rowx = &mut out.0[itgt * nfil..(itgt + 1) * nfil];
        let rowy = &mut out.1[itgt * nfil..(itgt + 1) * nfil];
        let rowz = &mut out.2[itgt * nfil..(itgt + 1) * nfil];
        rowx.fill(0.0);
        rowy.fill(0.0);
        rowz.fill(0.0);

        for ifil in 0..nfil {
            let force = integrate_target_triangle_force_from_bfield_fn(
                tgt_nodes,
                tgt_s,
                itgt,
                quad_kind,
                |_, obs, bout| {
                    for i in 0..obs.0.len() {
                        let b = flux_density_circular_filament_cartesian_scalar(
                            (rfil[ifil], zfil[ifil], 1.0),
                            (obs.0[i], obs.1[i], obs.2[i]),
                        );
                        bout.0[i] = b.0; // [T/A]
                        bout.1[i] = b.1; // [T/A]
                        bout.2[i] = b.2; // [T/A]
                    }
                    Ok(())
                },
            )?;
            rowx[ifil] = force[0];
            rowy[ifil] = force[1];
            rowz[ifil] = force[2];
        }
    }

    Ok(())
}

/// Parallel variant of [`triangle_mesh_force_mapping_from_circular_filaments`].
#[inline]
pub fn triangle_mesh_force_mapping_from_circular_filaments_par(
    rfil: &[f64],
    zfil: &[f64],
    mesh_tgt: &TriangleMeshView<'_>,
    s_tgt: &[f64],
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let nfil = rfil.len(); // [-]
    if zfil.len() != nfil {
        return Err("Length mismatch");
    }

    mesh_tgt.validate_nodal_values(s_tgt)?;
    validate_force_mapping_inputs(out.0, out.1, out.2, mesh_tgt.len(), nfil)?;

    out.0
        .par_chunks_mut(nfil)
        .zip_eq(out.1.par_chunks_mut(nfil))
        .zip_eq(out.2.par_chunks_mut(nfil))
        .enumerate()
        .try_for_each(|(itgt, ((rowx, rowy), rowz))| {
            let tgt_nodes = mesh_tgt.triangle_nodes(itgt);
            let tgt_s = mesh_tgt.triangle_scalars(itgt, s_tgt);
            rowx.fill(0.0);
            rowy.fill(0.0);
            rowz.fill(0.0);

            for ifil in 0..nfil {
                let force = integrate_target_triangle_force_from_bfield_fn(
                    tgt_nodes,
                    tgt_s,
                    itgt,
                    quad_kind,
                    |_, obs, bout| {
                        for i in 0..obs.0.len() {
                            let b = flux_density_circular_filament_cartesian_scalar(
                                (rfil[ifil], zfil[ifil], 1.0),
                                (obs.0[i], obs.1[i], obs.2[i]),
                            );
                            bout.0[i] = b.0;
                            bout.1[i] = b.1;
                            bout.2[i] = b.2;
                        }
                        Ok::<(), &'static str>(())
                    },
                )?;
                rowx[ifil] = force[0];
                rowy[ifil] = force[1];
                rowz[ifil] = force[2];
            }

            Ok::<(), &'static str>(())
        })?;

    Ok(())
}

/// Assemble the frozen-target source-amplitude to target-triangle force mapping for dipole
/// sources with fixed moment directions.
#[inline]
pub fn triangle_mesh_force_mapping_from_dipoles(
    loc: (&[f64], &[f64], &[f64]),
    moment_dir: (&[f64], &[f64], &[f64]),
    outer_radius: &[f64],
    mesh_tgt: &TriangleMeshView<'_>,
    s_tgt: &[f64],
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let ndip = loc.0.len(); // [-]
    if loc.1.len() != ndip
        || loc.2.len() != ndip
        || moment_dir.0.len() != ndip
        || moment_dir.1.len() != ndip
        || moment_dir.2.len() != ndip
        || outer_radius.len() != ndip
    {
        return Err("Length mismatch");
    }

    mesh_tgt.validate_nodal_values(s_tgt)?;
    validate_force_mapping_inputs(out.0, out.1, out.2, mesh_tgt.len(), ndip)?;

    for itgt in 0..mesh_tgt.len() {
        let tgt_nodes = mesh_tgt.triangle_nodes(itgt);
        let tgt_s = mesh_tgt.triangle_scalars(itgt, s_tgt);
        let rowx = &mut out.0[itgt * ndip..(itgt + 1) * ndip];
        let rowy = &mut out.1[itgt * ndip..(itgt + 1) * ndip];
        let rowz = &mut out.2[itgt * ndip..(itgt + 1) * ndip];
        rowx.fill(0.0);
        rowy.fill(0.0);
        rowz.fill(0.0);

        for idip in 0..ndip {
            let force = integrate_target_triangle_force_from_bfield_fn(
                tgt_nodes,
                tgt_s,
                itgt,
                quad_kind,
                |_, obs, bout| {
                    for i in 0..obs.0.len() {
                        let b = flux_density_dipole_scalar(
                            (loc.0[idip], loc.1[idip], loc.2[idip]),
                            (moment_dir.0[idip], moment_dir.1[idip], moment_dir.2[idip]),
                            outer_radius[idip],
                            (obs.0[i], obs.1[i], obs.2[i]),
                        );
                        bout.0[i] = b.0; // [T/source-amplitude]
                        bout.1[i] = b.1; // [T/source-amplitude]
                        bout.2[i] = b.2; // [T/source-amplitude]
                    }
                    Ok(())
                },
            )?;
            rowx[idip] = force[0];
            rowy[idip] = force[1];
            rowz[idip] = force[2];
        }
    }

    Ok(())
}

/// Parallel variant of [`triangle_mesh_force_mapping_from_dipoles`].
#[inline]
pub fn triangle_mesh_force_mapping_from_dipoles_par(
    loc: (&[f64], &[f64], &[f64]),
    moment_dir: (&[f64], &[f64], &[f64]),
    outer_radius: &[f64],
    mesh_tgt: &TriangleMeshView<'_>,
    s_tgt: &[f64],
    quad_kind: QuadratureKind,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let ndip = loc.0.len(); // [-]
    if loc.1.len() != ndip
        || loc.2.len() != ndip
        || moment_dir.0.len() != ndip
        || moment_dir.1.len() != ndip
        || moment_dir.2.len() != ndip
        || outer_radius.len() != ndip
    {
        return Err("Length mismatch");
    }

    mesh_tgt.validate_nodal_values(s_tgt)?;
    validate_force_mapping_inputs(out.0, out.1, out.2, mesh_tgt.len(), ndip)?;

    out.0
        .par_chunks_mut(ndip)
        .zip_eq(out.1.par_chunks_mut(ndip))
        .zip_eq(out.2.par_chunks_mut(ndip))
        .enumerate()
        .try_for_each(|(itgt, ((rowx, rowy), rowz))| {
            let tgt_nodes = mesh_tgt.triangle_nodes(itgt);
            let tgt_s = mesh_tgt.triangle_scalars(itgt, s_tgt);
            rowx.fill(0.0);
            rowy.fill(0.0);
            rowz.fill(0.0);

            for idip in 0..ndip {
                let force = integrate_target_triangle_force_from_bfield_fn(
                    tgt_nodes,
                    tgt_s,
                    itgt,
                    quad_kind,
                    |_, obs, bout| {
                        for i in 0..obs.0.len() {
                            let b = flux_density_dipole_scalar(
                                (loc.0[idip], loc.1[idip], loc.2[idip]),
                                (moment_dir.0[idip], moment_dir.1[idip], moment_dir.2[idip]),
                                outer_radius[idip],
                                (obs.0[i], obs.1[i], obs.2[i]),
                            );
                            bout.0[i] = b.0;
                            bout.1[i] = b.1;
                            bout.2[i] = b.2;
                        }
                        Ok::<(), &'static str>(())
                    },
                )?;
                rowx[idip] = force[0];
                rowy[idip] = force[1];
                rowz[idip] = force[2];
            }

            Ok::<(), &'static str>(())
        })?;

    Ok(())
}

/// Contract a frozen-target source-to-target-triangle force mapping with source coefficients to
/// obtain one force vector per target triangle.
///
/// Args:
///     fx: Row-major target-triangle by source mapping x-component (N/source-unit).
///     fy: Row-major target-triangle by source mapping y-component (N/source-unit).
///     fz: Row-major target-triangle by source mapping z-component (N/source-unit).
///     s_src: Source coefficients (source units).
///     out: Output force components per target triangle (N).
///
/// Returns:
///     `Ok(())` after writing one force vector per target triangle.
#[inline]
pub fn triangle_mesh_triangle_forces_from_potential_vectors(
    fx: &[f64],
    fy: &[f64],
    fz: &[f64],
    s_src: &[f64],
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let ntri = validate_force_mapping_vector_inputs(fx, fy, fz, s_src)?;
    if out.0.len() != ntri || out.1.len() != ntri || out.2.len() != ntri {
        return Err("Output dimension mismatch");
    }

    let nsrc = s_src.len(); // [-]
    for itri in 0..ntri {
        let rowx = &fx[itri * nsrc..(itri + 1) * nsrc];
        let rowy = &fy[itri * nsrc..(itri + 1) * nsrc];
        let rowz = &fz[itri * nsrc..(itri + 1) * nsrc];
        let mut accx = 0.0; // [N]
        let mut accy = 0.0; // [N]
        let mut accz = 0.0; // [N]
        for isrc in 0..nsrc {
            accx += rowx[isrc] * s_src[isrc]; // [N]
            accy += rowy[isrc] * s_src[isrc]; // [N]
            accz += rowz[isrc] * s_src[isrc]; // [N]
        }
        out.0[itri] = accx; // [N]
        out.1[itri] = accy; // [N]
        out.2[itri] = accz; // [N]
    }

    Ok(())
}

/// Contract a frozen-target source-to-target-triangle force mapping with source coefficients to
/// obtain the total force on the target mesh.
///
/// Args:
///     fx: Row-major target-triangle by source mapping x-component (N/source-unit).
///     fy: Row-major target-triangle by source mapping y-component (N/source-unit).
///     fz: Row-major target-triangle by source mapping z-component (N/source-unit).
///     s_src: Source coefficients (source units).
///
/// Returns:
///     Total force `[fx, fy, fz]` on the target mesh (N).
#[inline]
pub fn triangle_mesh_force_from_potential_vectors(
    fx: &[f64],
    fy: &[f64],
    fz: &[f64],
    s_src: &[f64],
) -> Result<[f64; 3], &'static str> {
    let ntri = validate_force_mapping_vector_inputs(fx, fy, fz, s_src)?;
    let mut tri_fx = vec![0.0; ntri];
    let mut tri_fy = vec![0.0; ntri];
    let mut tri_fz = vec![0.0; ntri];
    triangle_mesh_triangle_forces_from_potential_vectors(
        fx,
        fy,
        fz,
        s_src,
        (&mut tri_fx, &mut tri_fy, &mut tri_fz),
    )?;

    Ok([
        tri_fx.iter().sum(),
        tri_fy.iter().sum(),
        tri_fz.iter().sum(),
    ]) // [N]
}
