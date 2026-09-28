# cfsem

[Docs - Rust](https://docs.rs/cfsem) | [Docs - Python](https://cfsem-py.readthedocs.io/)

Quasi-steady electromagnetics including filamentized approximations, Biot-Savart, and Grad-Shafranov.

## Installation - Python

Requirements

* Python 3.10-3.13 and pip
* If on an x86 processor, you will need a CPU from roughly 2013 or later.

```bash
pip install cfsem
```

## Installation - Rust

To include this library in a Rust project, add an entry to your Cargo.toml's `[dependencies]` section:

```toml
cfsem = "*"
```

## Benchmarking - Rust

Benchmarks are configured in Cargo.toml, and can be run via cargo:

```bash
cargo bench
```

To build the docs with katex math rendering:

```bash
RUSTDOCFLAGS="--html-in-header=katex-header.html" cargo rustdoc --open
```

## Development - Python

Requirements

* [Rust](https://www.rust-lang.org/tools/install)

To install in the active python environment, do

```bash
uv pip install -e . --group dev
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

## References

The table summarizes sources cited in the implementation, tests, examples, and documentation, along with publications retained for future work. Reference numbers are local to this bibliography; individual API docstrings use their own numbering.

| Reference | Short publication name | Use in this repository |
| --- | --- | --- |
| [1](#ref-1) | Griffiths — Introduction to Electrodynamics | Linear-filament fields and finite-radius magnetized-sphere fields and vector potentials. |
| [2](#ref-2) | Purcell and Morin — Electricity and Magnetism | Vector-potential formulation of mutual inductance between circular and linear filaments. |
| [3](#ref-3) | Montgomery and Terrell — Aircore Solenoids | Elliptic-integral expressions for circular-filament magnetic fields. |
| [4](#ref-4) | MIT — Sources of Magnetic Fields | On-axis circular-loop field, off-axis field expressions, and analytic validation formulas. |
| [5](#ref-5) | Dennison — Magnet Formulas | Practical off-axis circular-loop field formulas. |
| [6](#ref-6) | Simpson et al. — Circular Current Loop | Circular-filament magnetic field and vector potential. |
| [7](#ref-7) | Kaltsas et al. — Analytic Tokamak Equilibrium | Background for the circular-filament poloidal-flux formulation. |
| [8](#ref-8) | Jardin — Computational Methods in Plasma Physics | Poloidal flux and finite-difference Grad–Shafranov operators. |
| [9](#ref-9) | Huang and Menard — Free-Boundary Equilibrium Solver | Circular-filament poloidal-flux Green's function in axisymmetric equilibrium calculations. |
| [10](#ref-10) | Lyle — Circular Coils of Rectangular Section | Sixth-order self-inductance approximation for rectangular-section circular coils. |
| [11](#ref-11) | Rosa and Cohen — Self-Inductance of Circles | Wien's self-inductance formula for a circular loop with circular conductor section. |
| [12](#ref-12) | Rosa and Grover — Inductance Formulas and Tables | Wien's self-inductance formula for a circular loop with annular conductor section. |
| [13](#ref-13) | Ejima et al. — Volt-Second Analysis | Flux and energy relationships used for distributed axisymmetric-conductor inductance. |
| [14](#ref-14) | Romero and JET-EFDA Contributors — Internal Inductance Dynamics | Internal and external inductance definitions for distributed axisymmetric conductors. |
| [15](#ref-15) | Wai and Kolemen — GSPD | Equilibrium-design context for distributed axisymmetric-conductor inductance. |
| [16](#ref-16) | Wesson — Tokamaks | Recovering magnetic-field components from poloidal-flux derivatives. |
| [17](#ref-17) | van Nugteren and Deelen — rat-mlfmm | Inspiration for the linear-filament kernel's geometric evaluation and finite-radius handling. |
| [18](#ref-18) | Zahn — The Vector Potential | Finite straight-wire field and vector-potential relationships. |
| [19](#ref-19) | Hurwitz et al. — Coil Self-Field, Circular Section | Finite-thickness circular-section circular conductor interior field; thin-conductor approximation; not yet implemented. |
| [20](#ref-20) | Landreman et al. — Coil Self-Field, Rectangular Section | Finite-thickness rectangular-section circular conductor interior field; thin-conductor approximation; not yet implemented. |
| [21](#ref-21) | Mousavi and Sukumar — Generalized Duffy Transformation | Background for integrating singular boundary-element kernels. |
| [22](#ref-22) | Graglia — Triangle Green's-Function Integrals | Background for triangle potential and potential-gradient integrals. |
| [23](#ref-23) | Wilton et al. (1984) — Polygonal Potential Integrals | Analytic potential-integral background for surface-current boundary elements. |
| [24](#ref-24) | Duffy — Singular Vertex Quadrature | Singularity-removing transformations used in boundary-element quadrature development. |
| [25](#ref-25) | Peeren — Stream Function Approach | Representing surface currents using nodal stream functions. |
| [26](#ref-26) | Hussain et al. — Gaussian Quadrature for Triangles | Background for numerical integration over triangular elements. |
| [27](#ref-27) | Dunavant — Symmetrical Triangle Quadrature | Triangle quadrature rules for boundary-element integration. |
| [28](#ref-28) | Wilton et al. (2020) — Static Potential Integrals | Exact triangle potential and gradient kernels for fields, vector potentials, and inductance. |
| [29](#ref-29) | Gumerov et al. — Analytical Galerkin Boundary Integrals | Analytic double-integral formulation cited as an independent validation reference for triangle inductance. |
| [30](#ref-30) | Abramowitz and Stegun — Handbook of Mathematical Functions | Polynomial approximations for complete elliptic integrals. |
| [31](#ref-31) | Michel and Stoitsov — Gauss Hypergeometric Function | Stable complex hypergeometric-function evaluation and analytic continuation. |
| [32](#ref-32) | NIST — Digital Library of Mathematical Functions | Hypergeometric definitions and branches; Gauss–Legendre quadrature and Legendre polynomials. |
| [33](#ref-33) | JuliaMath — HypergeometricFunctions.jl | Adapted hypergeometric implementation; attribution is recorded in THIRD_PARTY_NOTICES.md. |
| [34](#ref-34) | SciPy — hyp2f1 API Reference | Hypergeometric-function conventions and comparison implementation. |
| [35](#ref-35) | NVIDIA — PhysicsNeMo | Software reference cited by the hierarchical field-evaluation module. |
| [36](#ref-36) | Bower — Applied Mechanics of Solids | Displacement finite elements, quadrilateral interpolation, geometry mapping, and quadrature. |
| [37](#ref-37) | Wilson — Structural Analysis of Axisymmetric Solids | Axisymmetric finite-element strain and stiffness formulation. |
| [38](#ref-38) | Mitchell et al. — Axisymmetric Finite-Element Analysis | Formulation and experimental validation background for axisymmetric structural analysis. |
| [39](#ref-39) | Fried — Axisymmetric Elastic Solid | Axisymmetric elasticity finite-element formulation. |
| [40](#ref-40) | Lee — Thermal Stresses in a Hollow Cylinder | Analytic thermal-stress calculation for a radial temperature gradient. |
| [41](#ref-41) | Engineering ToolBox — Thick-Walled Cylinders | Lamé hoop and radial stress formulas under internal and external pressure. |
| [42](#ref-42) | Doane — Pressure Vessels | Thick-walled-cylinder pressure-stress hand calculations. |
| [43](#ref-43) | NIST — CODATA Recommended Values | Vacuum-permeability constants cited by the Python and Rust implementations. |
| [44](#ref-44) | Wikipedia — Finite Difference Coefficient | Central and one-sided finite-difference stencil coefficients. |
| [45](#ref-45) | Wikipedia — Grad–Shafranov Equation | Background linked from the Grad–Shafranov module. |
| [46](#ref-46) | Wikipedia — Magnetic Vector Potential | Background for the vector-potential line-integral formulation of mutual inductance. |
| [47](#ref-47) | Wikipedia — Helmholtz Coil | Analytic comparison and background for the Helmholtz-coil example. |
| [48](#ref-48) | HyperPhysics — Current Loop | Analytic circular-loop field used in electromagnetic tests. |
| [49](#ref-49) | Kernfeld — rustdoc-katex-demo | Source of the KaTeX header used to render mathematics in Rust documentation. |

### Bibliography

1. <a id="ref-1"></a> D. J. Griffiths, *Introduction to Electrodynamics*, 4th ed., Pearson, 2014; 5th ed., Cambridge University Press, 2024. The linear-filament module cites the fourth edition and the volumetric module cites the fifth edition.

2. <a id="ref-2"></a> E. M. Purcell and D. J. Morin, [*Electricity and Magnetism*](https://www.cambridge.org/highereducation/books/electricity-and-magnetism/0F97BB6C5D3A56F19B9835EDBEAB087C), Cambridge University Press.

3. <a id="ref-3"></a> D. B. Montgomery and J. Terrell, [“Some Useful Information for the Design of Aircore Solenoids. Part I. Relationships Between Magnetic Field, Power, Ampere-Turns and Current Density. Part II. Homogeneous Magnetic Fields”](https://apps.dtic.mil/sti/citations/tr/AD0269073), MIT Francis Bitter National Magnet Laboratory, 1961.

4. <a id="ref-4"></a> MIT, [“Sources of Magnetic Fields,” *8.02 Course Notes*, Chapter 9](https://web.mit.edu/8.02t/www/802TEAL3D/visualizations/coursenotes/modules/guide09.pdf). See Example 9.2, equations 9.1.13–9.1.15, and Appendix 1, equation 9.8.7.

5. <a id="ref-5"></a> E. Dennison, [“Off-Axis Field of a Current Loop,” *Magnet Formulas*](https://tiggerntatie.github.io/emagnet-py/offaxis/off_axis_loop.html).

6. <a id="ref-6"></a> J. C. Simpson, J. E. Lane, C. D. Immer, R. C. Youngquist, and T. Steinrock, [“Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop”](https://ntrs.nasa.gov/citations/20010038494), NASA, 2001.

7. <a id="ref-7"></a> D. Kaltsas, A. Kuiroukidis, and G. Throumoulopoulos, [“A tokamak pertinent analytic equilibrium with plasma flow of arbitrary direction”](https://doi.org/10.1063/1.5120341), *Physics of Plasmas*, vol. 26, article 124501, 2019.

8. <a id="ref-8"></a> S. Jardin, *Computational Methods in Plasma Physics*, 1st ed., CRC Press, 2010.

9. <a id="ref-9"></a> J. Huang and J. Menard, [“Development of an Auto-Convergent Free-Boundary Axisymmetric Equilibrium Solver”](https://www.osti.gov/biblio/1051805-development-auto-convergent-free-boundary-axisymmetric-equilibrium-solver), *Journal of Undergraduate Research*, vol. 6, 2006.

10. <a id="ref-10"></a> T. R. Lyle, [“IX. On the self-inductance of circular coils of rectangular section”](https://doi.org/10.1098/rsta.1914.0009), *Philosophical Transactions of the Royal Society of London, Series A*, vol. 213, pp. 421–435, 1914.

11. <a id="ref-11"></a> E. Rosa and L. Cohen, [“On the Self-Inductance of Circles”](https://nvlpubs.nist.gov/nistpubs/bulletin/04/nbsbulletinv4n1p149_A2b.pdf), *Bulletin of the Bureau of Standards*, 1908.

12. <a id="ref-12"></a> E. B. Rosa and F. W. Grover, [“Formulas and tables for the calculation of mutual and self-inductance (Revised)”](https://doi.org/10.6028/bulletin.185), *Bulletin of the Bureau of Standards*, vol. 8, no. 1, 1912. See equation 64, p. 112, for the annular-section formula.

13. <a id="ref-13"></a> S. Ejima, R. W. Callis, J. L. Luxon, R. D. Stambaugh, T. S. Taylor, and J. C. Wesley, [“Volt-second analysis and consumption in Doublet III plasmas”](https://doi.org/10.1088/0029-5515/22/10/006), *Nuclear Fusion*, vol. 22, no. 10, pp. 1313–1319, 1982.

14. <a id="ref-14"></a> J. A. Romero and JET-EFDA Contributors, [“Plasma internal inductance dynamics in a tokamak”](https://doi.org/10.1088/0029-5515/50/11/115002), *Nuclear Fusion*, vol. 50, article 115002, 2010. [Preprint](https://arxiv.org/abs/1009.1984v1).

15. <a id="ref-15"></a> J. T. Wai and E. Kolemen, [“GSPD: An algorithm for time-dependent tokamak equilibria design”](https://arxiv.org/abs/2306.13163), arXiv:2306.13163, 2023.

16. <a id="ref-16"></a> J. Wesson, *Tokamaks*, Clarendon Press, 1987. See equation 3.2.2 for the magnetic field in terms of poloidal flux.

17. <a id="ref-17"></a> J. van Nugteren and N. Deelen, [*rat-mlfmm*](https://gitlab.com/Project-Rat/rat-mlfmm/-/tree/1e1d387522fafac50c0540af1ebb15d1d506d33d), software repository, revision `1e1d387522fafac50c0540af1ebb15d1d506d33d`.

18. <a id="ref-18"></a> M. Zahn, [“5.4: The Vector Potential,” *Electromagnetic Field Theory: A Problem Solving Approach*](https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Electro-Optics/Electromagnetic_Field_Theory%3A_A_Problem_Solving_Approach_(Zahn)/05%3A_The_Magnetic_Field/5.04%3A_The_Vector_Potential), Engineering LibreTexts.

19. <a id="ref-19"></a> S. Hurwitz, M. Landreman, and T. M. Antonsen Jr., [“Efficient calculation of the self magnetic field, self-force, and self-inductance for electromagnetic coils”](https://arxiv.org/abs/2310.09313), arXiv:2310.09313, 2023. Equations 16–19 give the field inside and near a conductor with circular cross-section; retained for future implementation.

20. <a id="ref-20"></a> M. Landreman, S. Hurwitz, and T. M. Antonsen Jr., [“Efficient calculation of self magnetic field, self-force, and self-inductance for electromagnetic coils. II. Rectangular cross-section”](https://arxiv.org/abs/2310.12087), arXiv:2310.12087, 2023. Equations 15–21 give the interior field for rectangular cross-section; retained for future implementation.

21. <a id="ref-21"></a> S. E. Mousavi and N. Sukumar, [“Generalized Duffy transformation for integrating vertex singularities”](https://doi.org/10.1007/s00466-009-0424-1), *Computational Mechanics*, vol. 45, nos. 2–3, pp. 127–140, 2010.

22. <a id="ref-22"></a> R. D. Graglia, [“On the numerical integration of the linear shape functions times the 3-D Green's function or its gradient on a plane triangle”](https://doi.org/10.1109/8.247786), *IEEE Transactions on Antennas and Propagation*, vol. 41, no. 10, pp. 1448–1455, 1993.

23. <a id="ref-23"></a> D. Wilton, S. Rao, A. Glisson, D. Schaubert, O. Al-Bundak, and C. Butler, [“Potential integrals for uniform and linear source distributions on polygonal and polyhedral domains”](https://doi.org/10.1109/TAP.1984.1143304), *IEEE Transactions on Antennas and Propagation*, vol. 32, no. 3, pp. 276–281, 1984.

24. <a id="ref-24"></a> M. G. Duffy, “Quadrature Over a Pyramid or Cube of Integrands with a Singularity at a Vertex,” *SIAM Journal on Numerical Analysis*, vol. 19, no. 6, pp. 1260–1262, 1982.

25. <a id="ref-25"></a> G. N. Peeren, [“Stream function approach for determining optimal surface currents”](https://doi.org/10.6100/IR570424), Ph.D. thesis, Technische Universiteit Eindhoven, 2003.

26. <a id="ref-26"></a> F. Hussain, M. S. Karim, and R. Ahamad, [“Appropriate Gaussian quadrature formulae for triangles”](https://zhilin.math.ncsu.edu/TEACHING/MA587/Gaussian_Quadrature2D.pdf), *International Journal of Applied Mathematics and Computation*, vol. 4, no. 1, pp. 24–38, 2012.

27. <a id="ref-27"></a> D. A. Dunavant, [“High Degree Efficient Symmetrical Gaussian Quadrature Rules for the Triangle”](https://doi.org/10.1002/nme.1620210612), *International Journal for Numerical Methods in Engineering*, vol. 21, no. 6, pp. 1129–1148, 1985.

28. <a id="ref-28"></a> D. R. Wilton, J. Rivero, W. A. Johnson, and F. Vipiana, [“Evaluation of Static Potential Integrals on Triangular Domains”](https://doi.org/10.1109/ACCESS.2020.2997287), *IEEE Access*, vol. 8, pp. 99806–99819, 2020.

29. <a id="ref-29"></a> N. A. Gumerov, S. Kaneko, and R. Duraiswami, [“Analytical Galerkin Boundary Integrals of Laplace Kernel Layer Potentials in R^3”](https://doi.org/10.1137/23M1547688), *SIAM Journal on Scientific Computing*, vol. 46, no. 2, pp. A974–A997, 2024.

30. <a id="ref-30"></a> M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions: With Formulas, Graphs, and Mathematical Tables*, 1970. See sections 17.3.34 and 17.3.36 for the elliptic-integral approximations.

31. <a id="ref-31"></a> N. Michel and M. V. Stoitsov, [“Fast computation of the Gauss hypergeometric function with all its parameters complex with application to the Pöschl–Teller–Ginocchio potential wave functions”](https://doi.org/10.1016/j.cpc.2007.11.007), *Computer Physics Communications*, vol. 178, no. 7, pp. 535–551, 2008.

32. <a id="ref-32"></a> NIST, *Digital Library of Mathematical Functions*: [§15.2, “Definitions and Analytical Properties”](https://dlmf.nist.gov/15.2); [§3.5(v), “Gauss Quadrature”](https://dlmf.nist.gov/3.5#v), especially equations 3.5.18–3.5.21; and [§18.3, “Definitions”](https://dlmf.nist.gov/18.3), for Legendre polynomials.

33. <a id="ref-33"></a> JuliaMath contributors, [*HypergeometricFunctions.jl*](https://github.com/JuliaMath/HypergeometricFunctions.jl), version 0.3.30. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and license information.

34. <a id="ref-34"></a> SciPy Developers, [“scipy.special.hyp2f1,” *SciPy API Reference*](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp2f1.html).

35. <a id="ref-35"></a> PhysicsNeMo Contributors, [“NVIDIA PhysicsNeMo: An open-source framework for physics-based deep learning in science and engineering”](https://github.com/NVIDIA/physicsnemo), software repository.

36. <a id="ref-36"></a> A. F. Bower, *Applied Mechanics of Solids*, CRC Press, 2009. See Section 8.1 and Table 8.3 for displacement finite elements and interpolation; Sections 8.1.11–8.1.13 cover mapping and quadrature.

37. <a id="ref-37"></a> E. L. Wilson, [“Structural Analysis of Axisymmetric Solids”](https://doi.org/10.2514/3.3356), *AIAA Journal*, vol. 3, no. 12, pp. 2269–2274, 1965.

38. <a id="ref-38"></a> R. A. Mitchell, R. M. Woolley, and C. R. Fisher, [“Formulation and experimental verification of an axisymmetric finite-element structural analysis”](https://nvlpubs.nist.gov/nistpubs/jres/75C/jresv75Cn3-4p155_A1b.pdf), *Journal of Research of the National Bureau of Standards, Section C*, vol. 75C, pp. 155–163, 1971.

39. <a id="ref-39"></a> I. Fried, “Notes on the finite element analysis of the axisymmetric elastic solid,” *International Journal of Solids and Structures*, vol. 10, no. 3, 1974.

40. <a id="ref-40"></a> C. C. Lee, [*A Note on Thermal Stresses in a Hollow Cylinder of Linearly Varying Temperature*](https://preserve.lehigh.edu/system/files/derivatives/coverpage/427261.pdf), Lehigh University, 1961. See equation 1.1.

41. <a id="ref-41"></a> Engineering ToolBox, [“Stress in Thick-Walled Cylinders or Tubes”](https://www.engineeringtoolbox.com/stress-thick-walled-tube-d_949.html).

42. <a id="ref-42"></a> J. Doane, [*Pressure Vessels — Thin and Thick-Walled Stress Analysis*](https://www.suncam.com/miva/downloads/docs/303.pdf), SunCam course 303, 2018.

43. <a id="ref-43"></a> NIST, *CODATA Recommended Values of the Fundamental Physical Constants*: [2018 values](https://www.physics.nist.gov/cuu/pdf/wall_2018.pdf), cited by Python; [2022 values](https://physics.nist.gov/cuu/pdf/wall_2022.pdf), cited by Rust.

44. <a id="ref-44"></a> Wikipedia contributors, [“Finite difference coefficient”](https://en.wikipedia.org/w/index.php?title=Finite_difference_coefficient).

45. <a id="ref-45"></a> Wikipedia contributors, [“Grad–Shafranov equation”](https://en.wikipedia.org/wiki/Grad%E2%80%93Shafranov_equation).

46. <a id="ref-46"></a> Wikipedia contributors, [“Magnetic vector potential”](https://en.wikipedia.org/w/index.php?title=Magnetic_vector_potential&oldid=1259654939#Magnetic_vector_potential), revision of November 26, 2024.

47. <a id="ref-47"></a> Wikipedia contributors, [“Helmholtz coil”](https://en.wikipedia.org/wiki/Helmholtz_coil#Derivation), derivation of the axial field used in the example.

48. <a id="ref-48"></a> HyperPhysics, Georgia State University, [“Magnetic Field of Current Loop”](http://hyperphysics.phy-astr.gsu.edu/hbase/magnetic/curloo.html).

49. <a id="ref-49"></a> P. Kernfeld, [*rustdoc-katex-demo*, `katex-header.html`](https://github.com/paulkernfeld/rustdoc-katex-demo/blob/master/katex-header.html), software example.

## License

Licensed under the MIT license (see LICENSE or <http://opensource.org/licenses/MIT>) .
