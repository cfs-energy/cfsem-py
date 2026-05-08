//! Meshing, filamentization, and reference-geometry infrastructure.

use crate::math::{cross3, dot3, norm3, normalize3, scale3, sub3};
use core::f64::consts::PI;

use nalgebra::Vector3;
use nalgebra::geometry::Rotation3;
use num_traits::{Float, FromPrimitive};

pub mod elements;
pub mod quad2d;
pub mod quadrature;
pub mod sampling;
pub mod triangle3d;

pub use elements::quad2d::QuadratureRule;
pub use quad2d::QuadMeshView2d;
pub use quadrature::GaussLegendreRule;
pub use sampling::{FaceSample, VolumeSample};
pub use triangle3d::TriangleMeshView;

/// Floating-point trait bound shared by the reusable mesh/element helpers.
pub trait Scalar: Float + FromPrimitive + Copy + std::fmt::Debug + Send + Sync + 'static {}

impl<T> Scalar for T where T: Float + FromPrimitive + Copy + std::fmt::Debug + Send + Sync + 'static {}

pub(crate) fn cast<F: Scalar>(value: f64) -> F {
    F::from_f64(value).expect("finite f64 literal should cast to target float")
}

/// Filamentize a helix about an arbitrary piecewise-linear path.
///
/// # Arguments
///
/// * `path`:               (m) x,y,z centerline path segment points, each length `n`
/// * `helix_start_offset`: (m) x,y,z location of starting point relative to first path point
/// * `twist_pitch`:        (m) centerline path length per helix revolution
/// * `angle_offset`:       (rad) initial rotation of helix about the path, applied on top of the start offset
/// * `out`:                (m) x,y,z helix path outputs, each length `n`
///
/// # Commentary
///
/// Assumes angle between sequential path segments is small and will fail
/// if that angle approaches or exceeds 90 degrees.
///
/// The helix initial position vector, `helix_start_offset`, must be in a plane normal to
/// the first path segment in order to produce good results. If it is not in-plane,
/// it will be projected on to that plane and then scaled to the magnitude of its
/// original length s.t. the distance from the helix to the path center is preserved
/// but its orientation is not.
pub fn filament_helix_path(
    path: (&[f64], &[f64], &[f64]),
    helix_start_offset: [f64; 3],
    twist_pitch: f64,
    angle_offset: f64,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let (xp, yp, zp) = path;
    let (xfil, yfil, zfil) = out;
    let n = xp.len();

    if n < 2 {
        return Err("Input path must have at least 2 points");
    }
    if yp.len() != n || zp.len() != n {
        return Err("Input dimension mismatch");
    }
    if xfil.len() != n || yfil.len() != n || zfil.len() != n {
        return Err("Output dimension mismatch");
    }

    if norm3(helix_start_offset) < f64::EPSILON.sqrt() {
        xfil.copy_from_slice(xp);
        yfil.copy_from_slice(yp);
        zfil.copy_from_slice(zp);
        return Ok(());
    }

    {
        let ds = [xp[1] - xp[0], yp[1] - yp[0], zp[1] - zp[0]];
        let ds_unit = normalize3(ds);

        let parallel_mag = dot3(helix_start_offset, ds_unit);
        let parallel_component = scale3(ds_unit, parallel_mag);

        let helix_start_offset_projected = sub3(helix_start_offset, parallel_component);

        let helix_start_offset_mag = norm3(helix_start_offset);
        let projected_mag = norm3(helix_start_offset_projected);

        let c = helix_start_offset_mag / projected_mag;
        let helix_start_offset_corrected = scale3(helix_start_offset_projected, c);

        let final_mag = norm3(helix_start_offset_corrected);

        if (1.0 - helix_start_offset_mag / final_mag).abs() > 1e-4 {
            return Err(
                "Helix start offset magnitude was not preserved. Check that helix start offset is not zero or parallel to path.",
            );
        }

        let final_parallel_mag = dot3(helix_start_offset_corrected, ds_unit);

        if final_parallel_mag > 1e-4 * helix_start_offset_mag {
            return Err(
                "Projection of helix_start_offset on to plane of first path segment failed",
            );
        }

        xfil[0] = helix_start_offset_corrected[0] + xp[0];
        yfil[0] = helix_start_offset_corrected[1] + yp[0];
        zfil[0] = helix_start_offset_corrected[2] + zp[0];
    }

    let mut ds = [xp[1] - xp[0], yp[1] - yp[0], zp[1] - zp[0]];
    for i in 1..n {
        let ds_prev = ds;
        let ds_prev_unit = normalize3(ds_prev);

        if i < n - 1 {
            ds = [xp[i + 1] - xp[i], yp[i + 1] - yp[i], zp[i + 1] - zp[i]];
        } else {
            ds = ds_prev;
        }
        let ds_mag = norm3(ds);
        let ds_unit = normalize3(ds);

        let perpendicularity = norm3(cross3(ds_unit, ds_prev_unit));
        let maybe_path_rotation =
            Rotation3::rotation_between(&Vector3::from(ds_prev_unit), &Vector3::from(ds_unit))
                .ok_or(
                    "Path orientation rotation failed, likely due to path segment angle > 90 deg",
                )?;

        let rotation_almost_parallel = maybe_path_rotation.into_inner().iter().any(|x| x.is_nan());
        let path_rotation = if perpendicularity < 1e-16 || rotation_almost_parallel {
            Rotation3::identity()
        } else {
            maybe_path_rotation
        };

        let twist_angle = 2.0 * PI * ds_mag / twist_pitch;
        let twist_rotation = Rotation3::from_scaled_axis(twist_angle * Vector3::from(ds_unit));

        let r_prev = Vector3::from([
            xfil[i - 1] - xp[i - 1],
            yfil[i - 1] - yp[i - 1],
            zfil[i - 1] - zp[i - 1],
        ]);
        let r = twist_rotation * (path_rotation * r_prev);

        xfil[i] = r.x + xp[i];
        yfil[i] = r.y + yp[i];
        zfil[i] = r.z + zp[i];
    }

    rotate_filaments_about_path(path, angle_offset, (xfil, yfil, zfil))?;

    Ok(())
}

/// (In-place) rotation of an offset path about the path that it is offset from.
/// Intended to be used with helix paths generated by [`filament_helix_path`].
///
/// # Arguments
///
/// * `path`:               (m) x,y,z centerline path segment points, each length `n`
/// * `angle_offset`:       (rad) initial rotation of helix about the path, applied on top of the start offset
/// * `out`:                (m) x,y,z helix path, each length `n`, mutated in-place.
pub fn rotate_filaments_about_path(
    path: (&[f64], &[f64], &[f64]),
    angle_offset: f64,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let (xp, yp, zp) = path;
    let (xfil, yfil, zfil) = out;
    let n = xp.len();

    if n < 2 {
        return Err("Input path must have at least 2 points");
    }
    if yp.len() != n || zp.len() != n {
        return Err("Input dimension mismatch");
    }
    if xfil.len() != n || yfil.len() != n || zfil.len() != n {
        return Err("Output dimension mismatch");
    }

    for i in 0..n {
        let ds = if i < n - 1 {
            [xp[i + 1] - xp[i], yp[i + 1] - yp[i], zp[i + 1] - zp[i]]
        } else {
            [xp[i] - xp[i - 1], yp[i] - yp[i - 1], zp[i] - zp[i - 1]]
        };

        let ds_unit = normalize3(ds);
        let rotation = Rotation3::from_scaled_axis(angle_offset * Vector3::from(ds_unit));

        let mut r = Vector3::from([xfil[i] - xp[i], yfil[i] - yp[i], zfil[i] - zp[i]]);
        r = rotation * r;

        xfil[i] = r.x + xp[i];
        yfil[i] = r.y + yp[i];
        zfil[i] = r.z + zp[i];
    }

    Ok(())
}
