//! Load-vector construction and repeated-load operators.
//!
//! Each load type lives in its own submodule.  The same submodule owns both the direct
//! element-load construction used during assembly and the sparse operator construction used for
//! repeated-load solves.

use crate::physics::solenoid_stress::types::Real;

mod body_force;
mod pressure;
mod thermal;
mod traction;

#[derive(Debug, Clone)]
pub struct SparseOperator<F: Real> {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<F>,
    pub nrow: usize,
    pub ncol: usize,
}

#[derive(Debug, Clone)]
pub struct ThermalLoadOperator<F: Real> {
    pub temperature_to_rhs: SparseOperator<F>,
    pub reference_rhs: Vec<F>,
}

pub use body_force::{body_force_operator_quad4, body_force_operator_quad9};
pub use pressure::{pressure_operator_quad4, pressure_operator_quad9};
pub use thermal::{temperature_operator_quad4, temperature_operator_quad9};
pub use traction::{traction_operator_quad4, traction_operator_quad9};

pub(crate) use body_force::accumulate_body_force;
pub(crate) use pressure::pressure_element_load;
pub(crate) use thermal::accumulate_thermal_load;
pub(crate) use traction::traction_element_load;
