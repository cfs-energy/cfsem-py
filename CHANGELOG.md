# Changelog

## 3.0.0 2025-10-29

### Added

* Rust
    * Add methods for vector potential of a dipole in `physics::point_source` and `python.rs` bindings
    * Add `math::{clip_nan, switch_float}` functions for branchless-in-assembly float selection operations
* Python
    * Add `vector_potential_dipole` function
    * Add optional sphere radius input for dipole flux density

### Changed

* Rust
    * !Require sphere radius input for dipole flux density
    * Update dependencies
    * Use more codegen units and don't do LTO for debug builds
    * Use more mul_add in flux_circular_filament
    * Use multiplication instead of pow in dipole calcs
* Python
    * !Enable more instruction sets for x86 processors

## 2.7.0 2025-10-15

### Added
* Add method for calculating self-inductance for coaxial collection of ideal circular filaments.

## 2.6.0 2025-10-15

Substantial performance improvement for `flux_density_linear_filament` Biot-Savart methods.
This also improves performance in calculations that use these methods, such as linear filament
body force density calcs.

### Added

* Rust
    * Add `dot3f` and `cross3f` 32-bit float variants

### Changed

* Rust
    * Use mixed-precision method for `flux_density_linear_filament_scalar`
    * High-dynamic-range part of the calc is still done using 64-bit floats
    * Low-dynamic-range part of the calc is now done using 32-bit floats
        * _All_ addition operations in 32-bit section are done using 
        fused multiply-add operations, usually chained to defer
        roundoff to final operation. As a result, total roundoff error
        accumulated in this section is minimal.
    * Return is upcast back to 64-bit float to support precise summation downstream
    * 1.4-2x speedup without any meaningful loss of precision
        * No change to unit test tolerances needed; unlike an all-32-bit implementation,
        this mixed-precision method passes all the same tests as the 64-bit-only method
* Python
    * Update dep versions
    * Use latest rust backend version, which includes 1.4-2x speedup for flux_density_linear_filament Biot-Savart calcs

## Earlier Versions

See archived changelogs for versions prior to 2.6.0.
