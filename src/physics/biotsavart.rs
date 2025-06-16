//! Biot-Savart calculations for B-field from current filaments.

// Preserve references for backwards-compatibility
pub use crate::physics::linear_filament::{
    flux_density_linear_filament as flux_density_biot_savart,
    flux_density_linear_filament_par as flux_density_biot_savart_par,
};
