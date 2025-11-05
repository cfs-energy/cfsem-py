# Changelog

## 3.0.1 2025-11-05

### Changed

* Rust
    * Improve performance of dipole calcs
        * Now >1Gelem/s throughput including magnetized sphere fallback and nan clipping
    * Update benchmarks to use latest version of criterion
* Python
    * Use latest rust backend with improved dipole calc perf
    * Eliminate duplicate wheel builds during deployment
        * Maturin now builds for all supported python versions automatically in the same job;
          matrix on python versions is no longer necessary
        * Later, this can be further reduced to single wheels by building for a stable abi3 target
    * Remove support for python 3.9 (leaving long term support)

## 3.0.0 2025-10-29

### Added

* Rust
    * Add methods for vector potential of a dipole in `physics::point_source` and `python.rs` bindings
    * Add `physics::volumetric` module with methods for fields inside a uniformly magnetized sphere
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
    * Replace flatten() with ravel() everywhere to reduce copies

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
