//! Borrowed view and geometric helpers for 3D triangle surface meshes.

use crate::mesh::elements::tri::mapping as tri3;

/// Borrowed view of triangle surface-mesh geometry.
#[derive(Clone, Copy, Debug)]
pub struct TriangleMeshView<'a> {
    nodes: (&'a [f64], &'a [f64], &'a [f64]),
    triangles: (&'a [usize], &'a [usize], &'a [usize]),
}

#[inline]
fn triangle_area_from_indices(nodes: (&[f64], &[f64], &[f64]), idx: [usize; 3]) -> f64 {
    let n0 = [nodes.0[idx[0]], nodes.1[idx[0]], nodes.2[idx[0]]];
    let n1 = [nodes.0[idx[1]], nodes.1[idx[1]], nodes.2[idx[1]]];
    let n2 = [nodes.0[idx[2]], nodes.1[idx[2]], nodes.2[idx[2]]];
    tri3::area(n0, n1, n2)
}

#[inline]
/// Return a dimensionless triangle quality metric based on area and edge lengths.
///
/// The metric is
/// `4 * sqrt(3) * area / sum(edge^2)`.
/// It equals `1` for an equilateral triangle and tends to `0` for degenerate or
/// sliver triangles, so it is a cheap way to detect extremely poor aspect ratio.
fn triangle_quality_from_indices(nodes: (&[f64], &[f64], &[f64]), idx: [usize; 3]) -> f64 {
    let n0 = [nodes.0[idx[0]], nodes.1[idx[0]], nodes.2[idx[0]]];
    let n1 = [nodes.0[idx[1]], nodes.1[idx[1]], nodes.2[idx[1]]];
    let n2 = [nodes.0[idx[2]], nodes.1[idx[2]], nodes.2[idx[2]]];

    let edge_sq = |a: [f64; 3], b: [f64; 3]| -> f64 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        dx * dx + dy * dy + dz * dz
    };

    let area = tri3::area(n0, n1, n2);
    let edge_sum = edge_sq(n0, n1) + edge_sq(n1, n2) + edge_sq(n2, n0);
    4.0 * 3.0_f64.sqrt() * area / edge_sum
}

fn validate_triangle_mesh_geometry(
    nodes: (&[f64], &[f64], &[f64]),
    triangles: (&[usize], &[usize], &[usize]),
) -> Result<(usize, usize), &'static str> {
    let nnode = nodes.0.len();
    if nodes.1.len() != nnode || nodes.2.len() != nnode {
        return Err("Node coordinate dimension mismatch");
    }

    let ntri = triangles.0.len();
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

    let mut low_quality_triangles = Vec::new();
    for i in 0..ntri {
        let idx = [triangles.0[i], triangles.1[i], triangles.2[i]];
        let area = triangle_area_from_indices(nodes, idx);
        if area < 1e-14 {
            return Err("Triangle has zero area");
        }
        let quality = triangle_quality_from_indices(nodes, idx);
        if quality < 1e-3 {
            low_quality_triangles.push((i, quality, idx));
        }
    }
    if !low_quality_triangles.is_empty() {
        let entries = low_quality_triangles
            .iter()
            .map(|(i, quality, idx)| format!("{i} (quality={quality:.3e}, indices={idx:?})"))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!(
            "warning: {} triangles have very poor aspect ratio: {entries}",
            low_quality_triangles.len()
        );
    }

    Ok((nnode, ntri))
}

impl<'a> TriangleMeshView<'a> {
    /// Validate dimensions and construct a borrowed mesh view.
    pub fn new(
        nodes: (&'a [f64], &'a [f64], &'a [f64]),
        triangles: (&'a [usize], &'a [usize], &'a [usize]),
    ) -> Result<Self, &'static str> {
        validate_triangle_mesh_geometry(nodes, triangles)?;
        Ok(Self { nodes, triangles })
    }

    /// Number of nodes in the view.
    #[inline]
    pub fn nnode(&self) -> usize {
        self.nodes.0.len()
    }

    /// Number of triangles in the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.triangles.0.len()
    }

    /// Whether the view contains no triangles.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Triangle-node indices for one triangle.
    #[inline]
    pub fn triangle_indices(&self, i: usize) -> [usize; 3] {
        [
            self.triangles.0[i],
            self.triangles.1[i],
            self.triangles.2[i],
        ]
    }

    /// Node coordinates for one triangle.
    #[inline]
    pub fn triangle_nodes(&self, i: usize) -> [[f64; 3]; 3] {
        let idx = self.triangle_indices(i);
        idx.map(|k| [self.nodes.0[k], self.nodes.1[k], self.nodes.2[k]])
    }

    /// Node coordinates and indices for one triangle.
    #[inline]
    pub fn triangle_nodes_and_indices(&self, i: usize) -> ([[f64; 3]; 3], [usize; 3]) {
        let idx = self.triangle_indices(i);
        let nodes = idx.map(|k| [self.nodes.0[k], self.nodes.1[k], self.nodes.2[k]]);
        (nodes, idx)
    }

    /// Validate that a nodal scalar slice matches the mesh node count.
    #[inline]
    pub fn validate_nodal_values(&self, s: &[f64]) -> Result<(), &'static str> {
        if s.len() != self.nnode() {
            return Err("Nodal scalar dimension mismatch");
        }
        Ok(())
    }

    /// Nodal scalar values for one triangle.
    #[inline]
    pub fn triangle_scalars(&self, i: usize, s: &[f64]) -> [f64; 3] {
        let idx = self.triangle_indices(i);
        idx.map(|k| s[k])
    }
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

        let view = TriangleMeshView::new(nodes, triangles).unwrap();
        assert_eq!(view.len(), 1);
        assert_eq!(view.nnode(), 3);
        view.validate_nodal_values(s).unwrap();

        let tri_nodes = view.triangle_nodes(0);
        let tri_s = view.triangle_scalars(0, s);
        assert_eq!(
            tri_nodes,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
        assert_eq!(tri_s, [1.0, -0.5, 0.25]);
    }

    #[test]
    fn test_triangle_mesh_view_rejects_zero_area_triangle() {
        let nodes = (
            &[0.0, 1.0, 2.0][..],
            &[0.0, 0.0, 0.0][..],
            &[0.0, 0.0, 0.0][..],
        );
        let triangles = (&[0usize][..], &[1usize][..], &[2usize][..]);

        let err = TriangleMeshView::new(nodes, triangles).unwrap_err();
        assert_eq!(err, "Triangle has zero area");
    }

    #[test]
    fn test_triangle_mesh_view_rejects_repeated_vertex_triangle() {
        let nodes = (
            &[0.0, 1.0, 0.0][..],
            &[0.0, 0.0, 1.0][..],
            &[0.0, 0.0, 0.0][..],
        );
        let triangles = (&[0usize][..], &[1usize][..], &[1usize][..]);

        let err = TriangleMeshView::new(nodes, triangles).unwrap_err();
        assert_eq!(err, "Triangle has zero area");
    }
}
