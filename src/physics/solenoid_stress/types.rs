//! Shared numeric traits and constants for the solenoid-stress backend.

use num_traits::{Float, FromPrimitive};

/// Floating-point trait bound used throughout the solenoid-stress backend.
///
/// Keeping the bound in one place makes it easier to support both `f32` and `f64` entry points
/// without duplicating generic constraints everywhere else.
pub trait Real: Float + FromPrimitive + Copy + std::fmt::Debug + Send + Sync + 'static {}

impl<T> Real for T where T: Float + FromPrimitive + Copy + std::fmt::Debug + Send + Sync + 'static {}

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
