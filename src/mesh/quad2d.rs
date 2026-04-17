//! Borrowed view of 2D quadrilateral mesh geometry.

/// Borrowed view of a 2D quadrilateral mesh with fixed nodes per element.
#[derive(Clone, Copy)]
pub struct QuadMeshView2d<'a, F: Copy, const NODES_PER_ELEMENT: usize> {
    /// Node coordinates stored as a caller-defined 2D pair.
    pub nodes_rz: &'a [[F; 2]],
    /// Element connectivity in family-specific local-node order.
    pub elements: &'a [[usize; NODES_PER_ELEMENT]],
}

impl<'a, F: Copy, const NODES_PER_ELEMENT: usize> QuadMeshView2d<'a, F, NODES_PER_ELEMENT> {
    /// Number of mesh nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes_rz.len()
    }

    /// Number of elements.
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Validate that every connectivity entry references an existing node.
    pub fn validate_connectivity(&self) -> Result<(), String> {
        let node_count = self.num_nodes();
        for (element_index, element) in self.elements.iter().enumerate() {
            for &node in element {
                if node >= node_count {
                    return Err(format!(
                        "element {element_index} references node {node}, but mesh has only {node_count} nodes"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Return the node indices of one element.
    pub fn element_nodes(
        &self,
        element_index: usize,
    ) -> Result<[usize; NODES_PER_ELEMENT], String> {
        self.elements
            .get(element_index)
            .copied()
            .ok_or_else(|| format!("element index {element_index} out of bounds"))
    }

    /// Gather the physical coordinates of one element's nodes.
    pub fn element_coords(
        &self,
        element_index: usize,
    ) -> Result<[[F; 2]; NODES_PER_ELEMENT], String> {
        let nodes = self.element_nodes(element_index)?;
        Ok(std::array::from_fn(|local_index| {
            self.nodes_rz[nodes[local_index]]
        }))
    }
}
