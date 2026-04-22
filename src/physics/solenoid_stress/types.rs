//! Shared numeric traits and constants for the solenoid-stress backend.

use faer_traits::RealField;
use num_traits::{Float, FromPrimitive};

/// Floating-point trait bound used throughout the solenoid-stress backend.
///
/// Keeping the bound in one place makes it easier to support both `f32` and `f64` entry points
/// without duplicating generic constraints everywhere else.
pub trait Real:
    Float + FromPrimitive + RealField + Copy + std::fmt::Debug + Send + Sync + 'static
{
}

impl<T> Real for T where
    T: Float + FromPrimitive + RealField + Copy + std::fmt::Debug + Send + Sync + 'static
{
}

/// Cast a literal `f64` constant into the active floating-point type.
pub fn cast<F: Real>(value: f64) -> F {
    F::from_f64(value).expect("finite f64 literal should cast to target float")
}

/// Return the constant `2*pi` in the active floating-point type.
pub fn two_pi<F: Real>() -> F {
    cast(2.0 * core::f64::consts::PI)
}

/// Number of displacement unknowns carried by each node in the axisymmetric structural solver.
pub const DOF_PER_NODE: usize = 2;

/// Return the number of structural displacement unknowns carried by one element with the given
/// node count.
pub const fn dof_per_element(nodes_per_element: usize) -> usize {
    DOF_PER_NODE * nodes_per_element
}

/// Map one element's node indices to the global displacement DOF indices
/// `[u_r1, u_z1, u_r2, u_z2, ...]`.
pub fn local_dofs<const NODES_PER_ELEMENT: usize, const DOF_PER_ELEMENT: usize>(
    nodes: &[usize; NODES_PER_ELEMENT],
) -> [usize; DOF_PER_ELEMENT] {
    const {
        assert!(DOF_PER_ELEMENT == DOF_PER_NODE * NODES_PER_ELEMENT);
    }
    let mut local_dofs = [0usize; DOF_PER_ELEMENT];
    for (local_node, global_node) in nodes.iter().copied().enumerate() {
        local_dofs[2 * local_node] = 2 * global_node;
        local_dofs[2 * local_node + 1] = 2 * global_node + 1;
    }
    local_dofs
}

/// One scalar normal-pressure load topology entry for one element face.
///
/// The pressure amplitude is supplied later at `build_rhs(...)` time, not stored here.
#[derive(Clone, Copy, Debug)]
pub struct PressureLoad {
    /// Element index receiving the load.
    pub element: usize,
    /// Local face index in the element-family numbering used by `face_reference`:
    /// `0 = bottom`, `1 = right`, `2 = top`, `3 = left`.
    pub local_face: u8,
}

/// One traction-vector load topology entry for one element face.
///
/// The traction amplitudes are supplied later at `build_rhs(...)` time, not stored here.
#[derive(Clone, Copy, Debug)]
pub struct TractionLoad {
    /// Element index receiving the load.
    pub element: usize,
    /// Local face index in the element-family numbering used by `face_reference`:
    /// `0 = bottom`, `1 = right`, `2 = top`, `3 = left`.
    pub local_face: u8,
}

/// Per-material thermal-expansion data for the axisymmetric thermoelastic model.
#[derive(Clone, Copy, Debug)]
pub struct ThermalMaterial<F: Real> {
    /// Thermal strain coefficients in axisymmetric strain order `[rr, zz, tt, rz]`.
    ///
    /// Units: `[strain / temperature]`.
    pub alpha: [F; 4],
    /// Stress-free reference temperature for this material.
    ///
    /// Units: `[temperature]`.
    pub reference_temperature: F,
}

/// Sparse triplets for the assembled structural stiffness matrix before CSC compression.
///
/// Each entry has units
/// `[generalized nodal force / displacement] = [energy / distance^2]`,
/// which is the axisymmetric analogue of stiffness.
#[derive(Debug, Clone)]
pub struct StiffnessTriplets<F: Real> {
    /// Sparse row indices for the assembled stiffness-operator triplets.
    pub rows: Vec<usize>,
    /// Sparse column indices for the assembled stiffness-operator triplets.
    pub cols: Vec<usize>,
    /// Sparse values for the assembled stiffness-operator triplets.
    pub vals: Vec<F>,
}
