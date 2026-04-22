use crate::mesh::QuadMeshView2d;

pub(crate) static SINGLE_QUAD4_NODES: [[f64; 2]; 4] =
    [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]];
pub(crate) static SINGLE_QUAD4_ELEMENTS: [[usize; 4]; 1] = [[0usize, 1, 2, 3]];

pub(crate) fn single_element_quad4_mesh() -> QuadMeshView2d<'static, f64, 4> {
    QuadMeshView2d {
        nodes_rz: &SINGLE_QUAD4_NODES,
        elements: &SINGLE_QUAD4_ELEMENTS,
    }
}
