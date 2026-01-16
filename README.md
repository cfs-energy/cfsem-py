# cfsem

[Docs - Rust](https://docs.rs/cfsem) | [Docs - Python](https://cfsem-py.readthedocs.io/)

Quasi-steady electromagnetics including filamentized approximations, Biot-Savart, and Grad-Shafranov.

## Installation - System Dependencies

When building the `mlfmm` feature, some system dependencies are needed.

* cmake
* C/C++ toolchain (clang/gcc + make or ninja)
* git (for submodules)
* BLAS/LAPACK 
  * Linux & Windows: openblas
  * Mac: already included by the OS (Accelerate)
* zlib (for Boost iostreams; typically provided by the OS)

as well as some run-time dependencies:

* zlib
* System C++ runtime
* BLAS/LAPACK (MacOS only, nominally provided by OS)

## Installation - Python

Requirements

* Python 3.9-3.13 and pip
* If on an x86 processor, you will need a CPU from roughly 2013 or later.

```bash
pip install cfsem
```

## Installation - Rust

To include this library in a Rust project, add an entry to your Cargo.toml's `[dependencies]` section:

```toml
cfsem = "*"
```

If building with the `rat-mlfmm` feature, the library must be included as a git dependency

```toml
cfsem = { git = "https://github.com/cfs-energy/cfsem-py.git", tag = "4.0.0" }
```

## Benchmarking - Rust

Benchmarks are configured in Cargo.toml, and can be run via cargo:

```bash
cargo bench
```

## Development - Python

Requirements

* [Rust](https://www.rust-lang.org/tools/install)

To install in the active python environment, do

```bash
uv pip install -e . --group dev
```

To build the Rust bindings only, do

```bash
maturin develop --release --features=python
```

No part of installation requires root. If access issues are encountered, this can likely be resolved by using a virtual environment.

Some computationally-expensive calculations are written in Rust. These calculations and their python bindings are installed from pre-built binaries when installing from pypi or compiled during local development installation, with no intervention from the user in either case. Symmetric bindings with docstrings are available in the `bindings.py` module and re-exported at the library level.

To build with all of the optimizations available on your local machine, you can do:

```bash
RUSTCFLAGS="-Ctarget-cpu=native" pip install -e . --group dev --reinstall
```

## Contributing

Contributions consistent with the goals and anti-goals of the package are welcome.

Please make an issue ticket to discuss changes before investing significant time into a branch.

Goals

* Library-level functions and formulas
* Comprehensive documentation including literature references, assumptions, and units-of-measure
* Quantitative unit-testing of formulas
* Performance (both speed and memory-efficiency)
  * Guide development of performance-sensitive functions with structured benchmarking
* Cross-platform compatibility
* Minimization of long-term maintenance overhead (both for the library, and for users of the library)
  * Semantic versioning
  * Automated linting and formatting tools
  * Centralized CI and toolchain configuration in as few files as possible

Anti-Goals

* Fanciness that increases environment complexity, obfuscates reasoning, or introduces platform restrictions
* Brittle CI or toolchain processes that drive increased maintenance overhead
* Application-level functionality (graphical interfaces, simulation frameworks, etc)

## License

Licensed under the MIT license ([LICENSE](LICENSE) or http://opensource.org/licenses/MIT) .
