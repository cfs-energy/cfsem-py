//! Meshing and filamentization functions and data structures.
use crate::math::{cross3, dot3, rss3};
use core::f64::consts::PI;

use nalgebra::Vector3;
use nalgebra::geometry::Rotation3;

use num_traits::{Float, NumCast};

/// Linear segments in cartesian coordinates,
/// defined in mesh format as references to points.
#[doc(hidden)] // Might make breaking changes soon
#[non_exhaustive] // Might add more data fields later, like facets or elements
pub struct MeshEdgeList<T>
where
    T: Float + Send + Sync,
{
    /// Points in cartesian coordinates
    nodes: Vec<(T, T, T)>,
    /// Line segments defined by the indices of the start and end nodes
    edges: Vec<(usize, usize)>,
}

impl<T> MeshEdgeList<T>
where
    T: Float + Send + Sync,
{
    /// Check validity of segment indices & store
    pub fn new(nodes: Vec<(T, T, T)>, edges: Vec<(usize, usize)>) -> Result<Self, &'static str> {
        // Check if node indices are valid
        let n = nodes.len();
        if edges
            .iter()
            .any(|e| e.0 > n - 1 || e.1 > n - 1 || e.0 == e.1)
        {
            return Err("Segment refers to non-existent node or collapsed edge");
        }

        Ok(Self { nodes, edges })
    }

    /// Immutable reference to node list
    pub fn nodes(&self) -> &[(T, T, T)] {
        &self.nodes[..]
    }

    /// Immutable reference to edge indices
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges[..]
    }
}

/// Convert a point to f64 values
///
/// # Panics
///
/// * On encountering a value in the input that is not representable as f64.
pub(crate) fn convert_point<T>(p: (T, T, T)) -> (f64, f64, f64)
where
    T: NumCast,
{
    (
        NumCast::from(p.0).unwrap(),
        NumCast::from(p.1).unwrap(),
        NumCast::from(p.2).unwrap(),
    )
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
    helix_start_offset: (f64, f64, f64),
    twist_pitch: f64,
    angle_offset: f64,
    out: (&mut [f64], &mut [f64], &mut [f64]),
) -> Result<(), &'static str> {
    let (xp, yp, zp) = path;
    let (xfil, yfil, zfil) = out;
    let n = xp.len();

    // Check dimensions
    if n < 2 {
        return Err("Input path must have at least 2 points");
    }
    if yp.len() != n || zp.len() != n {
        return Err("Input dimension mismatch");
    }
    if xfil.len() != n || yfil.len() != n || zfil.len() != n {
        return Err("Output dimension mismatch");
    }

    // Do calculation

    //
    // First, project the start offset on to the plane of the starting point
    // so that repeated rotations don't eventually fold it across the centerline.
    //

    //    Use an encapsulating scope to keep a clear separation between local variables
    //    use for this part and ones used later.
    {
        // Direction of first step in path
        let ds = (xp[1] - xp[0], yp[1] - yp[0], zp[1] - zp[0]); // [m]
        let ds_unit = tuplenormalize(ds);

        // Get the component of the start offset that is parallel to the path
        let parallel_mag = tupledot(helix_start_offset, ds_unit);
        let parallel_component = (
            ds_unit.0 * parallel_mag,
            ds_unit.1 * parallel_mag,
            ds_unit.2 * parallel_mag,
        );

        // Project the helix start offset on to the plane of the first segment
        let helix_start_offset_projected = (
            helix_start_offset.0 - parallel_component.0,
            helix_start_offset.1 - parallel_component.1,
            helix_start_offset.2 - parallel_component.2,
        );

        // Rescale the helix start offset to the original magnitude
        // so that the user's choice of helix radius does not change
        let helix_start_offset_mag = tuplerss(helix_start_offset); // [m]
        let projected_mag = tuplerss(helix_start_offset_projected);

        let c = helix_start_offset_mag / projected_mag; // Correction to length
        let helix_start_offset_corrected = (
            helix_start_offset_projected.0 * c,
            helix_start_offset_projected.1 * c,
            helix_start_offset_projected.2 * c,
        );

        let final_mag = tuplerss(helix_start_offset_corrected);

        // Check that the user's choice of radius was not modified
        if (1.0 - helix_start_offset_mag / final_mag).abs() > 1e-4 {
            return Err(
                "Helix start offset magnitude was not preserved. Check that helix start offset is not zero or parallel to path.",
            );
        }

        // Check that the final projected helix start is, in fact,
        // perpendicular to the first path segment
        let final_parallel_mag = tupledot(helix_start_offset_corrected, ds_unit);

        if final_parallel_mag > 1e-4 * helix_start_offset_mag {
            return Err(
                "Projection of helix_start_offset on to plane of first path segment failed",
            );
        }

        // If everything looks ok, write the first helix segment to the output
        xfil[0] = helix_start_offset_corrected.0 + xp[0];
        yfil[0] = helix_start_offset_corrected.1 + yp[0];
        zfil[0] = helix_start_offset_corrected.2 + zp[0];
    }

    //
    // Then, apply rotations due to path orientation and twist pitch
    //
    let mut ds = (xp[1] - xp[0], yp[1] - yp[0], zp[1] - zp[0]);
    for i in 1..n {
        // Find the rotation needed for the change in path orientation
        let ds_prev = ds;
        let ds_prev_unit = tuplenormalize(ds_prev);

        if i < n - 1 {
            ds = (xp[i + 1] - xp[i], yp[i + 1] - yp[i], zp[i + 1] - zp[i]);
        } else {
            // For the last point, assume the path does not change direction
            ds = ds_prev
        }
        let ds_mag = tuplerss(ds);
        let ds_unit = tuplenormalize(ds);

        let path_rotation;
        let perpendicularity = tuplerss(tuplecross(ds_unit, ds_prev_unit));
        let maybe_path_rotation = Rotation3::rotation_between(
            &Vector3::from([ds_prev_unit.0, ds_prev_unit.1, ds_prev_unit.2]),
            &Vector3::from([ds_unit.0, ds_unit.1, ds_unit.2]),
        )
        .ok_or("Path orientation rotation failed, likely due to path segment angle > 90 deg")?;

        // Nalgebra rotation_between fails for nearly-parallel vectors,
        // but this is the easiest case because no rotation is required
        let rotation_almost_parallel = maybe_path_rotation.into_inner().iter().any(|x| x.is_nan());
        if perpendicularity < 1e-16 || rotation_almost_parallel {
            path_rotation = Rotation3::identity();
        } else {
            path_rotation = maybe_path_rotation;
        }

        // Find the rotation needed for the twist pitch
        let twist_angle = 2.0 * PI * ds_mag / twist_pitch;
        let twist_rotation = Rotation3::from_scaled_axis(
            twist_angle * Vector3::from([ds_unit.0, ds_unit.1, ds_unit.2]),
        );

        // Apply the rotation to get the new helix point,
        // starting from the offset between the previous point and the center path
        let r_prev = Vector3::from([
            xfil[i - 1] - xp[i - 1],
            yfil[i - 1] - yp[i - 1],
            zfil[i - 1] - zp[i - 1],
        ]);
        let r = twist_rotation * (path_rotation * r_prev);

        // Translate back to path & store point
        xfil[i] = r.x + xp[i];
        yfil[i] = r.y + yp[i];
        zfil[i] = r.z + zp[i];
    }

    // Finally, apply the angle offset as an additional rotation
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

    // Check dimensions
    if n < 2 {
        return Err("Input path must have at least 2 points");
    }
    if yp.len() != n || zp.len() != n {
        return Err("Input dimension mismatch");
    }
    if xfil.len() != n || yfil.len() != n || zfil.len() != n {
        return Err("Output dimension mismatch");
    }

    // Do rotations
    for i in 0..n {
        // Get the path unit vector at this location
        let ds;
        if i < n - 1 {
            ds = (xp[i + 1] - xp[i], yp[i + 1] - yp[i], zp[i + 1] - zp[i]);
        } else {
            // For the last point, assume the direction is the same as the previous point
            ds = (xp[i] - xp[i - 1], yp[i] - yp[i - 1], zp[i] - zp[i - 1]);
        }

        // Get the rotation matrix
        let ds_unit = tuplenormalize(ds);
        let rotation = Rotation3::from_scaled_axis(
            angle_offset * Vector3::from([ds_unit.0, ds_unit.1, ds_unit.2]),
        );

        // Apply the rotation
        let mut r = Vector3::from([xfil[i] - xp[i], yfil[i] - yp[i], zfil[i] - zp[i]]);
        r = rotation * r;

        // Translate back to path & store point
        xfil[i] = r.x + xp[i];
        yfil[i] = r.y + yp[i];
        zfil[i] = r.z + zp[i];
    }

    Ok(())
}

/// Convenience function for taking the dot product of two vectors stored as tuples
#[inline]
fn tupledot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    dot3(a.0, a.1, a.2, b.0, b.1, b.2)
}

/// Convenience function for taking the magnitude of a vector stored as a tuple
#[inline]
fn tuplerss(a: (f64, f64, f64)) -> f64 {
    rss3(a.0, a.1, a.2)
}

/// Convenience function for normalizing a vector stored as a tuple
#[inline]
fn tuplenormalize(a: (f64, f64, f64)) -> (f64, f64, f64) {
    let mag = tuplerss(a);
    (a.0 / mag, a.1 / mag, a.2 / mag)
}

/// Convenience function for taking the cross product of two vectors stored as tuples
#[inline]
fn tuplecross(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    cross3(a.0, a.1, a.2, b.0, b.1, b.2)
}
