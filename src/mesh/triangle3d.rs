//! Borrowed view and geometric helpers for 3D triangle surface meshes.

use crate::mesh::elements::tri::mapping as tri3;

#[derive(Clone, Copy, Debug)]
enum NodeStorage<'a> {
    Columns((&'a [f64], &'a [f64], &'a [f64])),
    RowMajor(&'a [f64]),
}

#[derive(Clone, Copy, Debug)]
enum TriangleStorage<'a> {
    Columns((&'a [usize], &'a [usize], &'a [usize])),
    RowMajorI64(&'a [i64]),
}

/// Borrowed view of triangle surface-mesh geometry.
#[derive(Clone, Copy, Debug)]
pub struct TriangleMeshView<'a> {
    nodes: NodeStorage<'a>,
    triangles: TriangleStorage<'a>,
    nnode: usize,
    ntri: usize,
}

#[inline]
fn triangle_area_from_nodes(nodes: [[f64; 3]; 3]) -> f64 {
    let [n0, n1, n2] = nodes;
    tri3::area(n0, n1, n2)
}

#[inline]
/// Return a dimensionless triangle quality metric based on area and edge lengths.
///
/// The metric is
/// `4 * sqrt(3) * area / sum(edge^2)`.
/// It equals `1` for an equilateral triangle and tends to `0` for degenerate or
/// sliver triangles, so it is a cheap way to detect extremely poor aspect ratio.
fn triangle_quality_from_nodes(nodes: [[f64; 3]; 3]) -> f64 {
    let [n0, n1, n2] = nodes;

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
        let tri_nodes = idx.map(|k| [nodes.0[k], nodes.1[k], nodes.2[k]]);
        let area = triangle_area_from_nodes(tri_nodes);
        if area < 1e-14 {
            return Err("Triangle has zero area");
        }
        let quality = triangle_quality_from_nodes(tri_nodes);
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
        let (nnode, ntri) = validate_triangle_mesh_geometry(nodes, triangles)?;
        Ok(Self {
            nodes: NodeStorage::Columns(nodes),
            triangles: TriangleStorage::Columns(triangles),
            nnode,
            ntri,
        })
    }

    /// Validate dimensions and construct a borrowed row-major mesh view.
    pub fn from_row_major_i64(
        nodes: &'a [f64],
        triangles: &'a [i64],
    ) -> Result<Self, &'static str> {
        if !nodes.len().is_multiple_of(3) {
            return Err("nodes must have shape (nnode, 3)");
        }
        if !triangles.len().is_multiple_of(3) {
            return Err("triangles must have shape (ntri, 3)");
        }
        let nnode = nodes.len() / 3;
        let ntri = triangles.len() / 3;
        let view = Self {
            nodes: NodeStorage::RowMajor(nodes),
            triangles: TriangleStorage::RowMajorI64(triangles),
            nnode,
            ntri,
        };
        view.validate_geometry()?;
        Ok(view)
    }

    /// Number of nodes in the view.
    #[inline]
    pub fn nnode(&self) -> usize {
        self.nnode
    }

    /// Number of triangles in the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.ntri
    }

    /// Whether the view contains no triangles.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Triangle-node indices for one triangle.
    #[inline]
    pub fn triangle_indices(&self, i: usize) -> [usize; 3] {
        match self.triangles {
            TriangleStorage::Columns(triangles) => [triangles.0[i], triangles.1[i], triangles.2[i]],
            TriangleStorage::RowMajorI64(triangles) => {
                let start = 3 * i;
                [
                    triangles[start] as usize,
                    triangles[start + 1] as usize,
                    triangles[start + 2] as usize,
                ]
            }
        }
    }

    /// Node coordinates for one triangle.
    #[inline]
    pub fn triangle_nodes(&self, i: usize) -> [[f64; 3]; 3] {
        let idx = self.triangle_indices(i);
        idx.map(|k| self.node(k))
    }

    /// Node coordinates and indices for one triangle.
    #[inline]
    pub fn triangle_nodes_and_indices(&self, i: usize) -> ([[f64; 3]; 3], [usize; 3]) {
        let idx = self.triangle_indices(i);
        let nodes = idx.map(|k| self.node(k));
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

    #[inline]
    fn node(&self, i: usize) -> [f64; 3] {
        match self.nodes {
            NodeStorage::Columns(nodes) => [nodes.0[i], nodes.1[i], nodes.2[i]],
            NodeStorage::RowMajor(nodes) => {
                let start = 3 * i;
                [nodes[start], nodes[start + 1], nodes[start + 2]]
            }
        }
    }

    fn validate_geometry(&self) -> Result<(), &'static str> {
        let mut low_quality_triangles = Vec::new();
        for i in 0..self.ntri {
            let idx = match self.triangles {
                TriangleStorage::Columns(triangles) => {
                    [triangles.0[i], triangles.1[i], triangles.2[i]]
                }
                TriangleStorage::RowMajorI64(triangles) => {
                    let start = 3 * i;
                    let idx0 = triangles[start];
                    let idx1 = triangles[start + 1];
                    let idx2 = triangles[start + 2];
                    if idx0 < 0 || idx1 < 0 || idx2 < 0 {
                        return Err("Triangle refers to non-existent node");
                    }
                    [idx0 as usize, idx1 as usize, idx2 as usize]
                }
            };
            if idx[0] >= self.nnode || idx[1] >= self.nnode || idx[2] >= self.nnode {
                return Err("Triangle refers to non-existent node");
            }
            let tri_nodes = idx.map(|k| self.node(k));
            let area = triangle_area_from_nodes(tri_nodes);
            if area < 1e-14 {
                return Err("Triangle has zero area");
            }
            let quality = triangle_quality_from_nodes(tri_nodes);
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
        Ok(())
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
