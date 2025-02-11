# cfsem

[Docs - Rust](https://docs.rs/cfsem) | [Docs - Python](https://cfsem-py.readthedocs.io/)

Quasi-steady electromagnetics including filamentized approximations, Biot-Savart, and Grad-Shafranov.

## Installation

Requirements

* Python 3.9-3.12 and pip
* Don't worry about this:
  * This info provided for troubleshooting purposes:
  * If on an x86 processor, you will need a CPU that supports SSE through 4.1, AVX, and FMA.
  * This should be true on any modern machine.

```bash
pip install cfsem
```

## Development

Requirements

* [Rust](https://www.rust-lang.org/tools/install)

To install in the active python environment, do

```bash
pip install -e .[dev]
```

To build the Rust bindings only, do

```bash
maturin develop --release
```

No part of installation requires root. If access issues are encountered, this can likely be resolved by using a virtual environment.

Some computationally-expensive calculations are written in Rust. These calculations and their python bindings are installed from pre-built binaries when installing from pypi or compiled during local development installation, with no intervention from the user in either case. Symmetric bindings with docstrings are available in the `bindings.py` module and re-exported at the library level.

To build with all of the optimizations available on your local machine, you can do:

```bash
RUSTCFLAGS="-Ctarget-cpu=native" maturin develop --release
pip install -e .[dev]
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
