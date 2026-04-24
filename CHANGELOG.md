# Changelog

## 6.0.0 2026-04-24

### Changed

* Rust
    * !Generalize axisymmetric stress solver to cover both axisymmetric and plane-strain
    * !Rename items from "axisymmetric" to indicate more general usage
* Python
    * !Rename axisymmetric_fem module to fem2d
    * !Update FEM bindings to match changes to Rust backend
    * Add handling for explicit 9-point quad mesh input


## 5.4.0 2026-04-24

### Changed

* Rust
    * Update BEM current density basis sign and scale convention to be more standard

## 5.3.1 2026-04-23

### Changed

* Rust
    * Make `triangle_basis_current_densities` pub

## 5.3.0 2026-04-10

### Added

* Rust
    * Solenoid stress module with axisymmetric finite element system assembly
    * `mesh` submodules for quadrature rules and for each supported mesh and element type
    * Build python bindings with abi3-py310
    * Update rust dep versions
* Python
    * Bindings and examples for new axisymmetric FEM stress solver

### Changed

* Rust
    * Factor out quadrature rules and triangle mesh functions from BEM into mesh module
        * This is technically a breaking change, but documented here as a minor change because there are no direct users of the affected items yet
* Python
    * Fix docs build script

## 5.2.0 2026-04-01

### Added

* Rust
    * Add `point_source::current_element` module with element kernels shared between point-segment and boundary-element methods
    * Add `boundary_element` module with B-field, A-field, inductance, and traction force
    * Add minimal triangle mesh view interface in `mesh.rs` using nodes-and-indices storage
* Python
    * Add bindings for new BEM field methods as well as helpers for visualizing quadrature points and triangle surface current density
    * Add boundary element method to field_explorer example

### Changed

* Refactor point-segment calculations to extract B-field and A-field kernels to `current_element.rs`
    * These kernels are shared between `point_segment` and `boundary_element` modules, and will likely be used by a future finite element module as well.
    * These kernels include singularity-clamping that robustifies the point-segment B-field and A-field.
* Removed experimental mesh_filament module in favor of more rigorous boundary element method
    * Minor change because this was not part of the public API

## 5.1.0 2026-03-24

### Added

* Add `inductance_linear_filaments,_par` and `inductance_linear_filaments_matrix,_par` inductance functions with disjoint-filament APIs matching the ones used for B-field and A-field. 

## 5.0.0 2026-03-24

### Added

* Add interaction matrix output options for linear filament A-field and B-field
* Add discretization sensitivity check to linear filament self inductance test against Lyle's calc
* Add `loop_inductance.py` example comparing different methods of calculating the self-inductance of a loop

### Changed

* !Remove `point_segment::self_inductance_piecewise_linear_filaments`
    * This formula diverges slowly under increasing discretization due to 1/r singularity
* !Update `linear_filament::inductance_piecewise_linear_filaments` to use vector potential integral method
    * !Remove `self_inductance` flag which is no longer needed
    * !Add `wire_radius` input
    * Use finite-length, finite-radius vector potential method, which allows direct evaluation of self-field
    * Use 3-point quadrature for target integration to reduce error in coupling with long segments
* Update python API for linear filament A and B with defaults that set output shape to vector by default
    * Preserves default behavior -> not breaking for python API

## 4.0.1 2026-03-12

### Changed

* Update readme
* Update python docs
* Unmask saved figure during tests so that the figure is built for docs
* Update build_docs script
* Add lengthwise discretization slider to field explorer example

## 4.0.0 2026-03-03

### Added

* !Remove `flux_density_biot_savart` backwards-compatibility alias
* !Upgrade `flux_density_linear_filament`, `vector_potential_linear_filament`, and `body_force_density_linear_filament` functions
  * Now handle finite wire length and finite wire thickness analytically
  * New `wire_radius` input; A-field blends quandratically to a nonzero value at wire center, B-field blends linearly to zero
    * Both match ideal behavior of uniform current density cylinder
  * Old point-source segment formulations moved to `point_source::segment` module and available as `flux_density_point_segment` and `vector_potential_point_segment` functions
* Add `field_explorer.py` example with plots comparing linear filament and point-segment calcs
* Improve parallelism heuristics to use half of available parallelism as heuristic for physical cores
    * Prevents oversubscription on systems with hyperthreading
* Use x86-64-v3 reference CPU instead of manually listing instruction sets

## 3.1.0 2025-12-19

### Added

* Add method inductance_matrix_axisymmetric_coaxial_rectangular_coils to calculate inductance matrix for a set of coaxial coils with rectangular cross-section and prescribed current density per coil section

## 3.0.3 2025-11-06

### Changed

* Use ternary instead of or-defaulting for array defaults in dipole functions
    * Eliminates issue with ambiguous truthiness of arrays under some circumstances

## 3.0.2 2025-11-05

### Changed

* Use numpy borrow interface instead of manually borrowchecking numpy arrays

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
