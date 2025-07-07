# Changelog

## 2.4.3 2025-07-07

### Changed

* No changes to rust side; version roll for lock-step versioning with python

## 2.4.2 2025-06-16

### Changed

* Include pyo3 extension-module feature to resolve build issues on linux

## 2.4.1 2025-06-16

### Changed

* Update release workflow

## 2.4.0 2025-06-13

### Changed

* Merge library with python bindings by adding `python` feature gate in front of bindings module
* Synchronize rust and python library versions
* Update license to MIT only for compatibility with combined Python library
* Update release workflow to use cargo-semver-checks action directly

## 2.0.0 2025-02-03

### Added

* Add libm dep for reproducible trig functions
    * Rust std/core defers to libc or other platform-dependent math libraries for trig functions, which can cause platform-dependent results
    * libm is a pure rust implementation of most functions from MUSL libm, and is platform-independent to the extent that the processor's implementation of floating point math conforms to IEEE-754
* Add scalar calculations of vector potential, magnetic field, and poloidal flux of a circular filament extracted from vector loop
    * Scalar calculations now used as the inner function in the vector loops
    * This produces no performance regression, and some improvements in a few cases, while improving readability by separating physics from array handling
* Add `mutual_inductance_circular_to_linear` family of functions for calculating mutual inductance between circular filaments and piecewise-linear paths
* Add `flux_density_circular_filament_cartesian` family of functions for calculating B-field from circular filaments to points in cartesian coordinates
* Add `cartesian_to_cylindrical` and `cylindrical_to_cartesian` conversion functions
* Add `decompose_filament` function for converting the start and end points of a filament to the midpoint and length vector
* Add `body_force_density_linear_filament` and `body_force_density_circular_filament_cartesian` families of functions for calculating JxB force density
* Add `mesh_filament` module and `mesh_edge_inductance` function for calculating full inductance matrix over a collection of disjoint segments
    * Same throughput perf as linear filament inductance for a given input geometry, but allocates for the full NxM output matrix
    * doc(hidden) for now, as the API is likely to change in the near future to avoid reallocating input data
* Add `point_source` module with dipole field
* Add (internal) `testing` module with array-handling utilities for tests
* Add (internal) macros for checking slice lengths

### Changed

* Use `Slice::fill(0.0)` instead of manually zeroing output arrays
* Use macro for checking array lengths to reduce repeated code
* !Consolidate function signatures of circular filament calcs to reduce number of args and group args by physical association

# Changelog

## 1.1.0 2024-08-20

### Added

* Add vector potential calcs for linear and circular filaments w/ parallel variants
* Add parallel variants of circular filament flux and flux density calcs
* Add tests of serial and parallel variants to make sure they produce the same result
* Add tests of equivalence between flux/inductance, flux density, and vector potential calcs

### Changed

* Move Biot-Savart calcs to linear filament module and rename appropriately
  * Leave use-as references to prevent breaking change to API
* Eliminate small allocations from parallel variant of Biot-Savart to reduce overhead when running with a large number of cores
  * 40%-100% speedup for small numbers of observation points
* Defensively zero-out output slices
* Convert some `#[inline(always)]` directives to plain `#[inline]`

## 1.0.0 2024-07-09

### Added

* Initial release
