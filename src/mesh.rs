//! Meshing and filamentization functions and data structures.
use crate::math::{add_scaled3, cross3, dot3, dot3_arr, rss3, sub3};
use core::f64::consts::PI;

use nalgebra::Vector3;
use nalgebra::geometry::Rotation3;

use num_traits::Float;

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

/// Borrowed view of a triangle surface mesh with one scalar value per node.
///
/// Intended as an internal lowered representation for boundary-element kernels.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct TriangleMeshView<'a> {
    nodes: (&'a [f64], &'a [f64], &'a [f64]),
    triangles: (&'a [usize], &'a [usize], &'a [usize]),
    s: &'a [f64],
}

#[inline]
pub(crate) fn triangle_max_edge_length_squared(n0: [f64; 3], n1: [f64; 3], n2: [f64; 3]) -> f64 {
    let e01 = sub3(n1, n0);
    let e12 = sub3(n2, n1);
    let e20 = sub3(n0, n2);
    dot3_arr(e01, e01)
        .max(dot3_arr(e12, e12))
        .max(dot3_arr(e20, e20))
}

#[inline]
pub(crate) fn triangle_closest_point(
    obs: [f64; 3],
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
) -> [f64; 3] {
    let ab = sub3(n1, n0);
    let ac = sub3(n2, n0);
    let ap = sub3(obs, n0);
    let d1 = dot3_arr(ab, ap);
    let d2 = dot3_arr(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return n0;
    }

    let bp = sub3(obs, n1);
    let d3 = dot3_arr(ab, bp);
    let d4 = dot3_arr(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return n1;
    }

    let vc = d1.mul_add(d4, -(d3 * d2));
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return add_scaled3(n0, ab, d1 / (d1 - d3));
    }

    let cp = sub3(obs, n2);
    let d5 = dot3_arr(ab, cp);
    let d6 = dot3_arr(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return n2;
    }

    let vb = d5.mul_add(d2, -(d1 * d6));
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return add_scaled3(n0, ac, d2 / (d2 - d6));
    }

    let bc = sub3(n2, n1);
    let va = d3.mul_add(d6, -(d5 * d4));
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return add_scaled3(n1, bc, (d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }

    let denom_inv = 1.0 / (va + vb + vc);
    let v = vb * denom_inv;
    let w = vc * denom_inv;
    add_scaled3(add_scaled3(n0, ab, v), ac, w)
}

#[inline]
pub(crate) fn triangle_subdivide_about_point(
    point: [f64; 3],
    n0: [f64; 3],
    n1: [f64; 3],
    n2: [f64; 3],
) -> [[[f64; 3]; 3]; 3] {
    [[point, n0, n1], [point, n1, n2], [point, n2, n0]]
}

pub(crate) fn validate_triangle_mesh_geometry(
    nodes: (&[f64], &[f64], &[f64]),
    triangles: (&[usize], &[usize], &[usize]),
) -> Result<(usize, usize), &'static str> {
    let nnode = nodes.0.len(); // [-]
    if nodes.1.len() != nnode || nodes.2.len() != nnode {
        return Err("Node coordinate dimension mismatch");
    }

    let ntri = triangles.0.len(); // [-]
    if triangles.1.len() != ntri || triangles.2.len() != ntri {
        return Err("Triangle index dimension mismatch");
    }

    if triangles
        .0
        .iter()
        .chain(triangles.1.iter())
        .chain(triangles.2.iter())
        .any(|&idx| idx >= nnode)
    {
        return Err("Triangle refers to non-existent node");
    }

    Ok((nnode, ntri))
}

impl<'a> TriangleMeshView<'a> {
    /// Validate dimensions and construct a borrowed mesh view.
    pub(crate) fn new(
        nodes: (&'a [f64], &'a [f64], &'a [f64]),
        triangles: (&'a [usize], &'a [usize], &'a [usize]),
        s: &'a [f64],
    ) -> Result<Self, &'static str> {
        let (nnode, _ntri) = validate_triangle_mesh_geometry(nodes, triangles)?;
        if s.len() != nnode {
            return Err("Nodal scalar dimension mismatch");
        }

        Ok(Self {
            nodes,
            triangles,
            s,
        })
    }

    /// Number of triangles in the view.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.triangles.0.len()
    }

    /// Node coordinates and nodal scalar values for one triangle.
    #[inline]
    pub(crate) fn triangle_nodes(&self, i: usize) -> ([[f64; 3]; 3], [f64; 3]) {
        let idx = [
            self.triangles.0[i],
            self.triangles.1[i],
            self.triangles.2[i],
        ];
        let nodes = idx.map(|k| [self.nodes.0[k], self.nodes.1[k], self.nodes.2[k]]); // [m]
        let s = idx.map(|k| self.s[k]); // [A]
        (nodes, s)
    }
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

#[cfg(test)]
mod tests {
    use super::TriangleMeshView;

    #[test]
    fn test_triangle_mesh_view_valid() {
        let nodes = (
            &[0.0, 1.0, 0.0][..],
            &[0.0, 0.0, 1.0][..],
            &[0.0, 0.0, 0.0][..],
        );
        let triangles = (&[0usize][..], &[1usize][..], &[2usize][..]);
        let s = &[1.0, -0.5, 0.25][..];

        let view = TriangleMeshView::new(nodes, triangles, s).unwrap();
        assert_eq!(view.len(), 1);

        let (tri_nodes, tri_s) = view.triangle_nodes(0);
        assert_eq!(
            tri_nodes,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
        assert_eq!(tri_s, [1.0, -0.5, 0.25]);
    }
}
