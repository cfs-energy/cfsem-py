# Changelog

See archived changelogs for versions prior to 2.6.0.

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
