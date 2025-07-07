# Changelog

## 2.4.3 2025-07-07

### Changed

* Configure linter to be slightly more strict
* Format and resolve new lints

## 2.4.2 2025-06-16

### Changed

* Include pyo3 extension-module feature to resolve build issues on linux

## 2.4.1 2025-06-16

### Changed

* Update release workflow

## 2.4.0 2025-06-13

### Changed

* Merge with rust library
* Update type hints for symmetric bindings
* Synchronize rust and python library versions

## 2.3.1 2025-03-19

### Changed

* Update rust to 2024 edition
* Update pyo3 rust dep version to resolve conditional compilation issue in build for linux arm/aarch64

## 2.3.0 2025-03-19

### Added

* Add support for python 3.13

## 2.2.0 2025-02-03

### Added

* Add circular-to-linear filament bridge functions
    * B-field and mutual inductance
    * Much faster and more accurate than discretizing loops and using linear filament functions, and more cross-platform consistent than doing coordinate conversions on circular filament methods at the python level
* Add dipole flux density function
* Add JxB body force density functions
* Add run-time assertions to check domain of validity of Lyle's method for self-inductance

### Changed

* Roll rust bindings forward to latest numpy and pyo3
* Update interfaces with rust functions that have changed their function signature
* Update python bindings with convenience functions for making inputs contiguous and flat, to reduce repetition

## 2.1.0 2024-08-27

### Added

* Add vector potential calcs for linear and circular filaments
* Add parallel options for circular-filament calcs
* Add parametrization of unit tests over parallel and serial variants
* Add optional parallel flags to functions that use functions with new parallel options

### Changed

## 2.0.4 2024-07-10

### Changed

* Use `cfsem` rust dep from crates.io
* Update docs and license for open-souce release

## 2.0.3 2024-07-09

### Changed

* Support python 3.12

## 2.0.2 2024-07-02

### Changed

* Update filament_coil function to use np.meshgrid instead of nested comprehension
* Update test of filament_coil to check against previous version of calc
* Move filamentization tests to their own test file
* Update nalgebra dep
* Update ruff dep

## 2.0.1 2024-06-18

### Changed

* Improve docstrings
* Test building docs in CI
* Update length checks in `inductance_piecewise_linear_filaments` to allow use when first and second path do not have the same number of segments

## 2.0.0 2024-05-10

### Changed

* (!) Transpose filament_helix_path() array inputs and outputs to eliminate some unnecessary copies
* (!) Transpose array inputs to piecewise linear inductance methods to eliminate some unnecessary copies
* Remove dep on scipy for release (now dev only)
* Add more parametrized cases to tests for filament_helix_path
* Use cubic interpolation method for boundary flux in distributed inductance calc
* Update flux_density_biot_savart rust function to mutate an input slice instead of allocating locally
* Parametrize tests of flux_density_biot_savart over parallel and serial implementations
* Add optional parallel flag to flux_density_biot_savart python bindings
* Roll forward rust deps on numpy and pyo3
* Run linting, tests, and coverage directly and remove dep on pre-commit

### Added

* Add `mesh` module in rust library
* Add rust implementation of filament_helix_path()
* Add rotate_filaments_about_path() via rust bindings
* Add CHANGELOG.md
* Add MU_0 from handcalc to eliminate scipy dep
* Add _filament_helix_path_py to test functions
  * Integrated error over the length of a filament makes this not very useful for testing against new calc
* Add flux_density_biot_savart_par in rust library
* Add nalgebra and rayon to rust deps
