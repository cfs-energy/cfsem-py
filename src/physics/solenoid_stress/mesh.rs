use crate::physics::solenoid_stress::quad4::NODES_PER_ELEMENT;
use crate::physics::solenoid_stress::types::Real;

#[derive(Clone, Copy)]
pub struct MeshView<'a, F: Real> {
    pub nodes_rz: &'a [[F; 2]],
    pub elements: &'a [[usize; NODES_PER_ELEMENT]],
}

#[derive(Clone, Copy, Debug)]
pub struct PressureLoad<F: Real> {
    pub element: usize,
    pub local_face: u8,
    pub value: F,
}

#[derive(Debug, Clone)]
pub struct AssemblyResult<F: Real> {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<F>,
    pub rhs: Vec<F>,
    pub ndof: usize,
}

impl<'a, F: Real> MeshView<'a, F> {
    pub fn num_nodes(&self) -> usize {
        self.nodes_rz.len()
    }

    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    pub fn validate_nodes(&self) -> Result<(), String> {
        for (index, node) in self.nodes_rz.iter().enumerate() {
            if node[0] < F::zero() {
                return Err(format!(
                    "node {index} has negative radius {:?}; axisymmetric radius must be nonnegative",
                    node[0]
                ));
            }
        }
        Ok(())
    }

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

    pub fn element_nodes(
        &self,
        element_index: usize,
    ) -> Result<[usize; NODES_PER_ELEMENT], String> {
        self.elements
            .get(element_index)
            .copied()
            .ok_or_else(|| format!("element index {element_index} out of bounds"))
    }

    pub fn element_coords(
        &self,
        element_index: usize,
    ) -> Result<[[F; 2]; NODES_PER_ELEMENT], String> {
        let nodes = self.element_nodes(element_index)?;
        let mut coords = [[F::zero(); 2]; NODES_PER_ELEMENT];
        for (local_index, global_index) in nodes.into_iter().enumerate() {
            coords[local_index] = self.nodes_rz[global_index];
        }
        Ok(coords)
    }
}
