//! Lightweight validated views of the axisymmetric element meshes and assembled outputs.

use crate::physics::solenoid_stress::types::Real;

#[derive(Clone, Copy)]
pub struct MeshView<'a, F: Real, const NODES_PER_ELEMENT: usize> {
    /// Node coordinates stored as `(r, z)`.
    pub nodes_rz: &'a [[F; 2]],
    /// Element connectivity in family-specific local-node order.
    pub elements: &'a [[usize; NODES_PER_ELEMENT]],
}

#[derive(Clone, Copy, Debug)]
pub struct PressureLoad<F: Real> {
    /// Element index receiving the load.
    pub element: usize,
    /// Local face index in the element-family numbering used by `face_reference`.
    pub local_face: u8,
    /// Pressure magnitude, taken positive in the inward normal direction.
    pub value: F,
}

#[derive(Clone, Copy, Debug)]
pub struct TractionLoad<F: Real> {
    /// Element index receiving the load.
    pub element: usize,
    /// Local face index in the element-family numbering used by `face_reference`.
    pub local_face: u8,
    /// Constant traction vector in global meridian coordinates `[t_r, t_z]`.
    pub value: [F; 2],
}

#[derive(Clone, Copy, Debug)]
pub struct ThermalMaterial<F: Real> {
    /// Thermal strain coefficients in axisymmetric strain order `[rr, zz, tt, rz]`.
    pub alpha: [F; 4],
    /// Stress-free reference temperature for this material.
    pub reference_temperature: F,
}

#[derive(Debug, Clone)]
pub struct AssemblyResult<F: Real> {
    /// Sparse row indices for the assembled stiffness matrix triplets.
    pub rows: Vec<usize>,
    /// Sparse column indices for the assembled stiffness matrix triplets.
    pub cols: Vec<usize>,
    /// Sparse values for the assembled stiffness matrix triplets.
    pub vals: Vec<F>,
    /// Global right-hand side vector.
    pub rhs: Vec<F>,
    /// Total number of displacement unknowns in the global system.
    pub ndof: usize,
}

impl<'a, F: Real, const NODES_PER_ELEMENT: usize> MeshView<'a, F, NODES_PER_ELEMENT> {
    /// Number of mesh nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes_rz.len()
    }

    /// Number of elements.
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Validate node coordinates that are specific to the axisymmetric setting.
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

    /// Gather the physical `(r, z)` coordinates of one element's nodes.
    pub fn element_coords(
        &self,
        element_index: usize,
    ) -> Result<[[F; 2]; NODES_PER_ELEMENT], String> {
        let nodes = self.element_nodes(element_index)?;
        let mut coords = [[F::zero(); 2]; NODES_PER_ELEMENT];
        for (local_index, global_index) in nodes.iter().copied().enumerate() {
            coords[local_index] = self.nodes_rz[global_index];
        }
        Ok(coords)
    }
}
