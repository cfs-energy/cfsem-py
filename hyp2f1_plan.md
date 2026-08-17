# Plan: Complex Gauss Hypergeometric Function

**Date:** 2026-08-17

**Repo:** cfsem-py

**Branch:** jlogan/hyp2f1

**Status:** COMPLETE

**Completed:** 2026-08-17

## Feature Request

Implement the Gauss hypergeometric function

\[
{}_2F_1(a,b;c;z)
\]

for `Complex<f64>` values of all four arguments. The Rust backend will provide
an allocation-free scalar kernel, a serial elementwise dense-vector kernel, and
a Rayon-parallel elementwise dense-vector kernel. The Python API will accept
only one-dimensional, C-contiguous NumPy `complex128` arrays of equal length;
it will not broadcast, implicitly cast, or accept real and `complex64` inputs.

The implementation must evaluate the defining series for `|z| < 1`, perform
analytic continuation for `|z| > 1`, handle the principal branch around the
standard cut `[1, +infinity)`, and remain efficient enough that the vector
wrappers add little overhead beyond repeated scalar evaluations.

### Success criteria

- `hyp2f1_scalar` handles finite complex parameters throughout the complex
  `z` plane except mathematical singularities and documented numerical limits.
- Values above and below the branch cut are distinct and respect the sign of
  zero in `Im(z)`; `x + 0j`, for `x > 1`, selects the upper lip and
  `x - 0j` selects the lower lip.
- The serial and parallel vector functions overwrite an output slice with one
  independent result per input index and reject unequal lengths before writing.
- The Python binding accepts only contiguous `complex128` arrays, defaults to
  parallel execution, and returns a newly allocated `complex128` array.
- The real-parameter subset agrees with SciPy within region-appropriate
  tolerances, and fully complex fixtures agree with high-precision `mpmath`
  reference values.
- The implementation has finite iteration caps, contains no scalar-path heap
  allocation, and includes Criterion measurements for every numerical region.

### Deliberate non-goals

- NumPy broadcasting, scalar/array mixing, strided input, implicit dtype
  conversion, multidimensional arrays, `complex64`, and real-only overloads.
- Arbitrary precision or a guarantee of near-machine precision for unbounded
  `|a|`, `|b|`, and `|c|`. Large parameters are tested and documented, but the
  initial implementation remains an `f64` algorithm.
- Reproducing SciPy's exact internal algorithm or porting the restrictively
  licensed Computer Physics Communications reference program.
- Exposing algorithm selection, tolerances, iteration caps, or convergence
  diagnostics as public API parameters in the first version.

## Reconnaissance Summary

### Relevant modules

+--------------------+----------------------------------------------+------------------------------------------------+
| File               | Existing responsibility                      | Planned relationship                           |
+====================+==============================================+================================================+
| src/math.rs        | Public pure-math functions and shared scalar | Declare a private hyp2f1 submodule and         |
|                    | helpers.                                     | re-export the three public Rust functions.     |
+--------------------+----------------------------------------------+------------------------------------------------+
| src/math/hyp2f1.rs | New file.                                    | Own the polyalgorithm, private numerical       |
|                    |                                              | helpers, vector wrappers, and Rust unit tests. |
+--------------------+----------------------------------------------+------------------------------------------------+
| src/lib.rs         | Crate modules, chunk-size heuristic, and     | Reuse chunksize; no new top-level module is    |
|                    | common array macros.                         | required.                                      |
+--------------------+----------------------------------------------+------------------------------------------------+
| src/python.rs      | PyO3/NumPy bindings and extension-module     | Add the strict complex-array binding and       |
|                    | registration.                                | register it in the Pure math group.            |
+--------------------+----------------------------------------------+------------------------------------------------+
| Cargo.toml         | Rust dependencies, features, and Criterion   | Add direct num-complex and the hyp2f1          |
|                    | benchmark targets.                           | benchmark target; optionally add csv as a      |
|                    |                                              | test-only dependency if fixture parsing grows. |
+--------------------+----------------------------------------------+------------------------------------------------+
| Cargo.lock         | Reproducible Rust dependency resolution.     | Record the direct num-complex dependency       |
|                    |                                              | without changing its resolved version.         |
+--------------------+----------------------------------------------+------------------------------------------------+
| CHANGELOG.md       | Release history.                             | Retain and refine the existing 12.1.0 hyp2f1   |
|                    |                                              | entry as the public contract is finalized.     |
+--------------------+----------------------------------------------+------------------------------------------------+
| cfsem/cfsem.pyi    | Extension-module type declarations.          | Add ComplexArray and the strict hyp2f1         |
|                    |                                              | signature.                                     |
+--------------------+----------------------------------------------+------------------------------------------------+
| cfsem/__init__.py  | Public package exports, including ellipe and | Import and export hyp2f1 directly from the     |
|                    | ellipk from the extension.                   | extension.                                     |
+--------------------+----------------------------------------------+------------------------------------------------+
| test/test_math.py  | SciPy comparisons for pure math bindings.    | Add binding behavior, SciPy-subset, branch,    |
|                    |                                              | and shared-fixture tests.                      |
+--------------------+----------------------------------------------+------------------------------------------------+
| test/data/         | No current hyp2f1 or complex-gamma fixtures. | Add documented high-precision complex          |
|                    |                                              | reference CSVs.                                |
+--------------------+----------------------------------------------+------------------------------------------------+
| tools/             | New generator directory entry.               | Add                                            |
|                    |                                              | generate_hyp2f1_reference.py to produce both   |
|                    |                                              | mpmath datasets outside normal CI.             |
+--------------------+----------------------------------------------+------------------------------------------------+
| benches/hyp2f1.rs  | New file.                                    | Measure scalar, serial, and parallel paths by  |
|                    |                                              | region and vector length.                      |
+--------------------+----------------------------------------------+------------------------------------------------+

### Existing patterns to follow

- Rust scalar kernels use an `_scalar` suffix, are `#[inline]`, and return the
  numerical value directly.
- Dense Rust functions accept borrowed slices and caller-owned mutable output
  slices. They return `Result<(), &'static str>` for length mismatch.
- Parallel functions chunk contiguous inputs using `chunksize`, use Rayon, and
  delegate each chunk to the serial function.
- Python array bindings use `PyReadonlyArray1<T>::as_slice()` to require a
  contiguous, aligned array and use `PyArray1::from_vec` for owned output.
- The existing pure-math Python bindings are registered in the `Pure math`
  section and re-exported directly by `cfsem/__init__.py`.
- Rust tests are colocated with the implementation. Python tests live under
  `test/`; CI also requires `cargo fmt`, Clippy, `ty`, Ruff, pytest, and the
  documentation build to pass.
- During final plan review, the worktree already contained user-owned 12.1.0
  version/changelog preparation in `Cargo.toml`, `Cargo.lock`, and
  `CHANGELOG.md`. Implementation must preserve and build on those edits rather
  than overwrite or duplicate them.

### Mathematical domain context

All inputs and outputs are dimensionless. DLMF 15.2 defines the principal
Gauss function by the series

\[
{}_2F_1(a,b;c;z) = \sum_{n=0}^{\infty}
\frac{(a)_n(b)_n}{(c)_n n!}z^n
\]

inside the unit disk and by analytic continuation elsewhere. In general, the
function has poles at `c = 0, -1, -2, ...`, although a terminating numerator
can end the series before the denominator pole. As a function of `z`, the
principal branch has a conventional cut from `1` to positive infinity.

SciPy exposes real `a`, `b`, and `c` with real or complex `z`; it is therefore
an oracle only for a subset of this feature. Its documented complex algorithm
also confirms that a single power series plus elementary transformations is
not enough around the annular region near `exp(+-i*pi/3)`.

SciPy enhancement issue #23450 records the same missing complex-parameter use
case, points directly to Michel--Stoitsov and `mpmath`, and links the older
complex-`z` accuracy issue #1561. As reviewed on 2026-08-17, the enhancement
remains open with no assignee, implementation branch, or pull request. This
supports an independent implementation rather than waiting for or attempting
to generalize SciPy's real-parameter kernel. The present feature intentionally
goes beyond the ticket title: `c`, as well as `a`, `b`, and `z`, is complex.

The numerical design follows the mathematical structure in Michel and
Stoitsov's all-complex algorithm: transformed power series, stabilized
connection formulas when `b-a` or `c-a-b` is near an integer, and short Taylor
continuation across the remaining transformation gaps. The implementation is
to be derived independently from the paper and DLMF identities. The CPC source
code must not be copied because its license prohibits redistribution and
commercial use. The MIT-licensed HypergeometricFunctions.jl implementation may
be used as a behavioral cross-check; any direct code adaptation would require
retaining its MIT notice and attribution.

## Architecture Plan

### Public API

The Rust API will use `num_complex::Complex64`, which is the same
`Complex<f64>` representation exposed by the NumPy crate as `Complex64`.

```rust
/// Evaluate Gauss's hypergeometric function on the principal branch.
///
/// The parameters `a`, `b`, `c`, and argument `z` may all be complex.
/// Values on the branch cut distinguish `+0.0` and `-0.0` in `z.im`.
/// Mathematical singularities, non-finite inputs, or failure to converge
/// return a complex NaN unless a limiting value is unambiguous.
#[inline]
pub fn hyp2f1_scalar(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    z: Complex64,
) -> Complex64;

/// Evaluate Gauss's hypergeometric function elementwise on contiguous slices.
///
/// All input and output slices must have equal length. The output is
/// overwritten rather than accumulated. Empty slices are valid.
pub fn hyp2f1(
    a: &[Complex64],
    b: &[Complex64],
    c: &[Complex64],
    z: &[Complex64],
    out: &mut [Complex64],
) -> Result<(), &'static str>;

/// Parallel elementwise evaluation of Gauss's hypergeometric function.
///
/// Uses Rayon over contiguous chunks and otherwise has the same contract as
/// `hyp2f1`.
pub fn hyp2f1_par(
    a: &[Complex64],
    b: &[Complex64],
    c: &[Complex64],
    z: &[Complex64],
    out: &mut [Complex64],
) -> Result<(), &'static str>;
```

The extension-module binding, as represented in `cfsem/cfsem.pyi`, will be:

```python
from numpy import complex128

ComplexArray: TypeAlias = NDArray[complex128]

def hyp2f1(
    a: ComplexArray,
    b: ComplexArray,
    c: ComplexArray,
    z: ComplexArray,
    par: bool = True,
) -> ComplexArray:
    """Evaluate 2F1 elementwise for equal-length contiguous complex128 arrays."""
```

The binding intentionally accepts arrays only. A caller wanting one scalar
evaluation passes length-one `complex128` arrays. This keeps the Python surface
narrow and avoids a second scalar extraction/conversion policy.

### Module layout

`src/math.rs` will gain the submodule declaration and public re-exports while
retaining its existing functions:

```rust
mod hyp2f1;
pub use hyp2f1::{hyp2f1, hyp2f1_par, hyp2f1_scalar};
```

`src/math/hyp2f1.rs` will contain these private layers:

1. Constants, complex classification helpers, and branch-aware elementary
   functions.
2. Complex reciprocal-gamma/log-gamma helpers required by connection formulas.
3. Direct and terminating series evaluators.
4. Stabilized expansions about `z = 1` and `z = infinity`.
5. Taylor continuation through the transformation gaps.
6. The scalar region-selection dispatcher.
7. Serial and parallel slice wrappers.
8. Colocated Rust tests.

This single numerical module is intentionally deeper than a collection of
public helper modules. Gamma approximations, region thresholds, recurrence
coefficients, and convergence bookkeeping remain private and can be replaced
without changing callers.

### Information hiding

- Callers see one mathematical operation, not the selected continuation
  formula. Region thresholds and the choice of Taylor anchor are private.
- The scalar kernel owns all singular-case and branch-cut policy. Vector and
  Python layers do not duplicate mathematical dispatch.
- Complex gamma support is scoped privately to this module. This avoids
  accidentally establishing an unsupported general-purpose gamma API.
- A private evaluation status distinguishes convergence from iteration failure
  during development and tests. The public scalar maps unsuccessful numerical
  evaluation to the documented IEEE NaN result.
- Python dtype and contiguity enforcement stays in `src/python.rs`; the Rust
  math module operates only on ordinary slices.

### Numerical algorithm

#### 1. Branch-aware elementary helpers

Implement small, tested helpers instead of repeatedly spelling fragile complex
expressions:

- `complex_log1p(w)` using
  `0.5*ln1p(2*Re(w) + |w|^2) + i*atan2(Im(w), 1+Re(w))`, preserving signed
  zero and avoiding cancellation near zero. Handle `w == -1` explicitly, and
  fall back to the ordinary principal complex logarithm when rounding makes
  the `ln1p` argument invalid outside the helper's cancellation-sensitive
  neighborhood.
- `complex_expm1(w)` using real `expm1`, half-angle trigonometric terms, and a
  separately computed imaginary part.
- `complex_pow(base, exponent) = exp(exponent*log(base))` with the principal
  logarithm and signed-zero tests on the negative real axis.
- `complex_abs(w) = hypot(Re(w), Im(w))` plus scaled complex division/inversion
  (`fdiv`/`finv` or equivalent) for selector ratios. Do not form `norm_sqr`
  where finite components could overflow before a mathematically finite ratio
  is obtained.
- Exact predicates for zero, finite components, nonpositive integral complex
  values, and distance from the nearest real integer. "Integral" requires an
  exactly zero imaginary part; proximity logic uses
  `epsilon = value - round(Re(value))` and its complex norm.

Do not normalize away `-0.0`. In particular, computing `1-z` and `-z` must
retain the side from which `z` approaches the branch cut.

#### 2. Complex gamma support

Implement private `log_gamma`, reciprocal-gamma, Pochhammer, and combined gamma
ratio operations with a Lanczos approximation and the reflection identity.
Compute connection coefficients as combined log-gamma sums before exponentiating
so numerator and denominator overflow do not occur independently.

Do not add a general special-functions dependency solely for gamma. The current
Rust candidates either describe their complex implementation as developmental
(`spfunc`) or pull in a much broader special-functions/tensor surface
(`torsh-special`), and neither exposes the reciprocal-gamma and correlated
log-ratio operations the stabilized recurrences require. Reconsider this
decision if a small, well-validated crate with those primitives is identified
during implementation; the fixture contract remains the acceptance gate.

Use a direct product recurrence for small integral Pochhammer orders and a
combined log-gamma ratio only for larger safe orders. This avoids introducing
gamma poles into a finite product merely as an implementation artifact.

Apply Lanczos on the conditioning-favorable half-plane and reflection on the
other side. Reflection computes `log(sin(pi*z))` with a scaled large-imaginary
formula instead of evaluating an overflowing raw complex sine. Record the
coefficient set's source and license beside the constants; high-precision
fixture acceptance, rather than the nominal approximation order, decides
whether the set is adequate.

Near nonpositive integers, use reciprocal gamma and explicit pole detection
rather than exponentiating a divergent `log_gamma`. Implement the stabilized
epsilon-difference helpers needed by Michel--Stoitsov using `complex_expm1` of
log-gamma differences; never subtract two nearly equal gamma values directly.

Validate gamma helpers independently against high-precision fixtures before
using them in `hyp2f1`. They stay private because only the combinations exercised
by the hypergeometric algorithm are in scope.

#### 3. Exceptional and terminating cases

Dispatch cheap cases before any transformation:

- Any NaN component, or unsupported infinite component: complex NaN.
- `z == 0`: one, including signed-zero variants.
- `a == 0` or `b == 0`: one unless an earlier non-finite-input rule applies.
- Negative integral `a` or `b`: evaluate the shorter terminating polynomial.
  For `c = -n`, a numerator polynomial of degree `m` is accepted exactly when
  `m <= n`; stop at the known degree without attempting the formally `0/0`
  next-term recurrence. Otherwise return complex NaN as an undefined pole.
- Nonpositive integral `c` in a nonterminating case: complex NaN.
- `z == 1`: use Gauss's gamma-ratio value when `Re(c-a-b) > 0`; otherwise return
  complex NaN unless the terminating-polynomial path already supplied a value.
- Exact identities `c == a` and `c == b` may use `(1-z)^(-b)` and
  `(1-z)^(-a)` after pole/termination checks. These are both faster and more
  accurate than general continuation.

The terminating evaluator uses the standard term recurrence and an exact term
count. It may choose between `z` and the Pfaff-transformed polynomial according
to the smaller argument modulus, as described by Michel--Stoitsov, but it must
not recurse through the general dispatcher.

#### 4. Normalization and region selection

Use the symmetry in `a` and `b` to arrange `Re(b-a) >= 0`. If
`Re(c-a-b) < 0`, apply Euler's transformation once,

\[
{}_2F_1(a,b;c;z) = (1-z)^{c-a-b}
{}_2F_1(c-a,c-b;c;z),
\]

and carry the branch-aware prefactor outside the normalized evaluation. This
arrangement makes the stable infinity and one-centered expansions use a
nonnegative nearest integer `m`.

Use a private initial radius `R = 0.9`, matching the studied algorithm, then
benchmark and accuracy-test it before freezing it. Form fixed transformation
descriptors that carry both the transformed parameters and the exact
branch-aware prefactor; evaluators consume a descriptor without recursively
redispatching. The viable candidates are:

1. Direct series when `|z| <= R`.
2. Pfaff transformation when `|z/(z-1)| <= R`.
3. Stabilized infinity expansion when `|1/z| <= R`.
4. Pfaff plus the infinity expansion when `|(z-1)/z| <= R`.
5. Stabilized one-centered expansion when `|1-z| <= R`.
6. Pfaff plus the one-centered expansion when `|1/(1-z)| <= R`.
7. Taylor continuation when none of the six transformed arguments is small
   enough.

Paths 2, 4, and 6 use the same explicit Pfaff descriptor

\[
w=\frac{z}{z-1},\quad
(a',b',c')=(a,c-b,c),\quad
P=(1-z)^{-a},
\]

so `F(a,b;c;z) = P*F(a',b';c';w)`. Path 2 evaluates the primed direct
series, path 4 evaluates its infinity expansion in `1/w = (z-1)/z`, and path
6 evaluates its one-centered expansion in `1-w = 1/(1-z)`. Paths 1, 3, and 5
use the normalized unprimed parameters. Any outer Euler prefactor is restored
last. Construct `1-z`, `-z`, `w`, and every prefactor through the branch-aware
helpers so signed-zero cut information survives the descriptor.

When more than one candidate qualifies, select the smallest transformed
modulus. Treat moduli within
`32*f64::EPSILON*max(1, r1, r2)` as tied and resolve them in the listed order,
so direct and Pfaff series win over coefficient-heavy connection formulas.
This rule is private, deterministic, and simple enough to validate against
benchmarks before any more elaborate cost model is justified.

#### 5. Direct series

Generate successive terms with

\[
t_{n+1}=t_n\frac{(a+n)(b+n)}{(c+n)(n+1)}z.
\]

Use a two-component compensated complex sum and track the largest term to
detect severe cancellation. Start with `REL_TOL = 8*f64::EPSILON`; require at
least two terms and stop only when the new term is small relative to a nonzero
compensated sum for two consecutive iterations, or is exactly zero because the
recurrence has terminated/underflowed. A zero sum alone does not indicate
convergence. Keep the tolerance private and change it only with corresponding
fixture evidence.

Use a finite `MAX_SERIES_ITERATIONS` constant, initially 10,000. Reaching the
cap, encountering a denominator zero not covered by termination, or producing
an unexpected non-finite intermediate marks evaluation unsuccessful.

#### 6. Stable expansions about one and infinity

A naive DLMF connection formula subtracts individually divergent terms when
`c-a-b` or `b-a` is equal or close to an integer. Do not implement that formula
as the near-integer path.

For the one-centered expansion, write

`c-a-b = m + epsilon`, with `m = round(Re(c-a-b))` after normalization.
For the infinity expansion, write

`b-a = m + epsilon`, with `m = round(Re(b-a))` after swapping `a` and `b` if
needed.

Implement the finite `A` contribution, infinite `B` recurrence, and stabilized
starting coefficients from Michel--Stoitsov sections 4, 5.1, and 5.2. Express
the coefficients through private reciprocal-gamma, Pochhammer, complex sinc,
`expm1`, and gamma-difference helpers so the limit remains finite as
`epsilon -> 0`. Use the ordinary two-term connection formula only when
`|epsilon|` exceeds an initial private `NEAR_INTEGER_RADIUS = 0.1` and the
condition estimate `(abs(part1)+abs(part2))/abs(part1+part2)` predicts enough
remaining digits for the requested tolerance; otherwise use the stabilized
recurrence. Treat a zero denominator in that estimate as maximally ill
conditioned. Boundary sweeps determine whether the initial radius can safely
be reduced for performance.

Each expansion must have its own iteration cap, compensated summation, and
two-consecutive-term convergence check. Unit tests must sweep epsilon across
zero in real and imaginary directions, rather than testing only the exact
integer case.

#### 7. Taylor continuation for uncovered regions

For a target in the remaining gaps near `exp(+-i*pi/3)`, choose

\[
z_0 = r_0 z/|z|,\qquad
r_0 = 0.9\text{ if }|z|<1,\quad r_0 = 1.1\text{ otherwise}.
\]

Evaluate

\[
q_0=F(a,b;c;z_0),\qquad
q_1=\frac{ab}{c}F(a+1,b+1;c+1;z_0)
\]

using forced direct or infinity paths that cannot re-enter Taylor continuation.
The inner anchor uses the direct series at radius `0.9`; the reciprocal
argument for the outer radius-`1.1` anchor is `1/1.1 < 1`, so the forced infinity
series converges even though it lies just outside the preferred selector radius
`R`. The forced evaluator therefore bypasses only the cost threshold, not its
iteration cap or convergence checks.

Generate subsequent coefficients from the hypergeometric differential equation:

\[
q_{n+2}=\frac{
[n(2z_0-1)-c+(a+b+1)z_0]q_{n+1}
+\frac{(a+n)(b+n)}{n+1}q_n
}{z_0(1-z_0)(n+2)}.
\]

Accumulate `q_n*(z-z0)^n` with compensated summation. Test the combined norm of
two consecutive terms to prevent a single anomalously small coefficient from
causing false convergence. Since `|z-z0| <= 0.1`, this should normally take
roughly 10--20 terms; still enforce a conservative finite cap.

#### 8. Result and failure policy

Use a private result such as `EvalOutcome { value, converged }` internally.
The public scalar returns `value` on success and `Complex64::new(NaN, NaN)` on
nonconvergence, a path-dependent singularity, or an undefined pole. Return
infinity only for a tested special case whose direction is unambiguous; avoid
manufacturing a misleading phase from `infinity * complex` arithmetic.

The vector kernels do not abort because one element is singular; they store
that element's NaN and continue. Their `Result` is reserved for structural
length errors.

### Data flow

```text
Complex64 a,b,c,z
        |
        v
special cases / terminating polynomial
        |
        v
symmetry + optional Euler normalization
        |
        v
deterministic region selector
   |        |          |             |
direct    Pfaff   one/infinity    Taylor gap
   |        |          |             |
   +--------+----------+-------------+
                    |
                    v
       restore accumulated prefactor
                    |
                    v
             Complex64 result

equal-length slices -> scalar loop -> output slice
equal-length slices -> Rayon chunks -> serial loops -> output slice
complex128 NumPy arrays -> borrowed slices -> Rust vector API -> owned NumPy output
```

### Mathematical constraints

+-----------------------------------+-----------------------------------+----------------------------------------------+
| Constraint                        | Enforcement location              | Mechanism                                    |
+===================================+===================================+==============================================+
| All array lengths are equal.      | Rust serial and parallel entry    | Check before any output write; return        |
|                                   | points.                           | Err("Length mismatch").                      |
+-----------------------------------+-----------------------------------+----------------------------------------------+
| Python inputs are one-dimensional | PyO3 function signature and       | PyReadonlyArray1<Complex64>::as_slice();     |
| contiguous complex128 arrays.     | NumPy slice extraction.           | propagate dtype/contiguity exceptions.       |
+-----------------------------------+-----------------------------------+----------------------------------------------+
| Integer parameter classification  | Scalar special-case dispatcher.   | Require exactly zero imaginary part and an   |
| is not applied to general complex |                                   | exactly integral finite real component.      |
| values.                           |                                   |                                              |
+-----------------------------------+-----------------------------------+----------------------------------------------+
| The principal branch and branch   | Complex log/power helpers and all | Preserve signed zero; test both lips of the  |
| cut side are consistent.          | connection prefactors.            | cut and conjugate approaches off the cut.    |
+-----------------------------------+-----------------------------------+----------------------------------------------+
| A denominator pole is not hidden  | Terminating and direct series.    | Check c+n == 0 before division and permit    |
| by floating-point division.       |                                   | a known polynomial degree m <= n.            |
+-----------------------------------+-----------------------------------+----------------------------------------------+
| Every iterative path terminates.  | All series/recurrence evaluators. | Fixed iteration cap plus explicit success or |
|                                   |                                   | failure outcome.                             |
+-----------------------------------+-----------------------------------+----------------------------------------------+
| No scalar-path allocation occurs. | Entire numerical module.          | Stack-only scalar state; benchmark and       |
|                                   |                                   | inspect implementation for Vec/boxing.       |
+-----------------------------------+-----------------------------------+----------------------------------------------+
| Parallel writes are disjoint.     | hyp2f1_par.                       | Zip equally sized immutable chunks with one  |
|                                   |                                   | mutable output chunk per Rayon task.         |
+-----------------------------------+-----------------------------------+----------------------------------------------+

### Approaches considered

#### Approach A: Port SciPy/xsf

SciPy has mature region tests, real-axis special cases, and extensive fixtures.
However, its public and internal complex-`z` implementation accepts real
`a`, `b`, and `c`; its gamma ratios and degeneracy logic rely on that fact.
Generalizing the port would amount to designing a new algorithm inside a large
real-parameter codebase. This approach is rejected, though SciPy remains an
important oracle for the common subset and a source of adversarial test regions.

#### Approach B: Independent all-complex transformed-series polyalgorithm

Implement the Michel--Stoitsov mathematical scheme in native Rust: direct and
Pfaff series, stable one/infinity recurrences, then Taylor continuation for the
small uncovered regions. This has an allocation-free hot path, handles all four
complex arguments, exposes no algorithmic complexity to callers, and maps well
to independent dense-array evaluation. The main cost is implementing and
testing private complex gamma machinery and near-integer cancellation control.

This is the recommended approach.

The implementation must be independently derived from the paper/DLMF. The CPC
program is reference material only and cannot be copied. The MIT Julia package
can serve as a differential oracle during development; if implementation text
or substantial structure is translated from it, add its copyright and license
to a new `THIRD_PARTY_NOTICES.md` in the same change.

#### Approach C: Integrate the hypergeometric differential equation

Start in the unit disk and numerically continue along a branch-aware path with
an adaptive complex ODE solver. This naturally handles general complex
parameters and could serve as a slow validation oracle. It is substantially
slower, requires dynamic stepping and path planning, complicates branch-cut
semantics, and is a poor fit for millions of independent dense-array elements.
It is rejected for the production kernel.

## Implementation Steps

1. **Establish complex types, module boundary, and reference data.**

   Dependencies: none.

   - Add `num-complex = "0.4.6"` as a direct dependency in `Cargo.toml` and
     retain the already resolved version in `Cargo.lock`; the lockfile's root
     package dependency list will gain `num-complex` alongside the preexisting
     user-owned 12.1.0 version change.
   - Add `src/math/hyp2f1.rs`, declare it from `src/math.rs`, and initially
     expose the three signatures with a minimal `z == 0` implementation.
   - Add `test/data/hyp2f1_reference.csv`. Record `a`, `b`, `c`, `z`, expected
     real/imaginary parts, the `mpmath` version, precision, and how upper/lower
     cut limits were generated. Include direct, Pfaff, one, infinity, Taylor,
     polynomial, near-integer, branch-cut, and moderate-parameter rows.
   - Add `test/data/complex_gamma_reference.csv` for the private gamma and
     reciprocal-gamma combinations, including conjugates and points adjacent
     to poles. Keeping this separate prevents the hypergeometric fixture from
     becoming a polymorphic test-data format.
   - Add `tools/generate_hyp2f1_reference.py` to regenerate both CSVs with at
     least 100 decimal digits. It is not part of normal CI and declares its
     pinned `mpmath` version in its module docstring and emitted metadata.
   - Load the fixture text in Rust tests with `include_str!`. Prefer a small
     test-only parser while the format remains a header plus comment metadata
     and fixed-count, unquoted numeric fields parsed only as `f64`. The parser
     must reject the wrong field count or an invalid float with the fixture
     filename and row number.
   - If quoting, optional columns, multiple record shapes, or useful diagnostics
     make that parser more than a straightforward split-and-parse helper, add
     the maintained `csv` crate under `[dev-dependencies]` instead. Either
     choice stays out of production dependencies and has no scalar-path cost.

   Verification: `cargo check`; independently spot-check fixture rows in an
   interactive high-precision session; confirm no repository file contains CPC
   source text.

2. **Build and validate private complex elementary and gamma helpers.**

   Dependencies: step 1.

   - Implement signed-zero-aware `log1p`, `expm1`, power, complex sinc,
     integer classification, Lanczos `log_gamma`, reciprocal gamma, Pochhammer,
     combined gamma ratios, and epsilon-difference helpers.
   - Use reflection only on the side of the complex plane where it improves
     conditioning; explicitly handle poles rather than relying on `sin(pi*z)`
     overflow behavior.
   - Add fixture-based and identity-based Rust tests, including values adjacent
     to gamma poles and conjugate pairs.

   Verification: focused Rust tests pass with region-appropriate tolerances;
   no helper is made public; code review confirms exact floating-point equality
   is limited to documented mathematical predicates, and Clippy is clean.

3. **Implement special cases and the direct/terminating series.**

   Dependencies: steps 1--2.

   - Implement non-finite input policy, zero argument/parameter identities,
     `c == a`, `c == b`, terminating numerator parameters, denominator poles,
     and `z == 1` handling.
   - Implement allocation-free direct and terminating recurrence evaluators,
     compensated summation, cancellation tracking, and finite iteration caps.
   - Add the public scalar dispatcher for the cases now supported.

   Verification: polynomial fixtures, DLMF identities, pole cases, and direct
   `|z| <= 0.9` fixtures pass; intentionally tiny iteration caps in private test
   hooks prove that nonconvergence becomes NaN rather than an infinite loop.

4. **Implement Pfaff/Euler normalization and stable connection expansions.**

   Dependencies: steps 2--3.

   - Implement symmetry and one-time Euler normalization without recursive
     oscillation.
   - Implement Pfaff transformation with branch-aware `log1p`.
   - Implement stabilized one-centered and infinity expansions from the
     published recurrence formulas, including finite and infinite parts for
     `m + epsilon` parameter differences.
   - Add the fixed-descriptor, smallest-modulus region selector for all six
     transformed arguments, with the documented deterministic tie order.
   - Keep forced internal evaluators separate from the public dispatcher so
     transformations cannot recurse cyclically.

   Verification: fixture sweeps on both sides of `|z| = 1`, near `z = 1`, and
   near every sampled integer difference pass; selector-specific unit tests
   demonstrate each path is reachable; symmetry and Euler/Pfaff identities
   agree independently.

5. **Close the transformation gaps with Taylor continuation.**

   Dependencies: step 4.

   - Implement the `r0 = 0.9/1.1` anchor selection, forced anchor and derivative
     evaluations, three-term coefficient recurrence, compensated accumulation,
     and two-term convergence criterion.
   - Add the Taylor path to the region selector only when no cheaper transformed
     series qualifies.
   - Add test-only path diagnostics or counters behind `cfg(test)` so tests can
     assert that gap fixtures really exercise Taylor continuation.

   Verification: fixtures around both `exp(+-i*pi/3)` gaps, just inside and
   outside the unit circle, pass; the differential-equation residual is small;
   the normal fixture set never reaches the Taylor iteration cap.

6. **Add serial and Rayon-parallel dense-vector APIs.**

   Dependencies: completed scalar kernel in step 5.

   - Validate all five lengths before writing output.
   - Implement a simple indexed serial loop that overwrites every output.
   - Implement deterministic, disjoint Rayon chunks using `chunksize`; each
     task delegates to `hyp2f1` on its chunk.
   - Cover empty, singleton, sub-slice, mismatch, NaN element, and mixed-region
     arrays.

   Verification: every finite result is bitwise equal to the corresponding
   scalar call and serial/parallel NaNs have matching classification; mismatch
   leaves a sentinel-filled output unchanged. Miri is not required because the
   crate forbids unsafe code and uses safe Rayon slice splitting.

7. **Expose the strict Python binding.**

   Dependencies: step 6.

   - Import `numpy::Complex64` in `src/python.rs` and add a `#[pyfunction]` with
     `par=true`.
   - Convert all four read-only arrays with `as_slice`, allocate one Rust output
     vector, dispatch to serial or parallel Rust, and construct a NumPy array.
   - Keep the GIL while borrowing NumPy slices. Releasing it would allow Python
     code in another thread to mutate nominally read-only input memory and is
     outside the safety contract of this implementation.
   - Register the function, add `ComplexArray` and the signature to the stub,
     and import/export `hyp2f1` from `cfsem/__init__.py`.
   - Do not add a coercing wrapper in `cfsem/bindings.py`.
   - Map the Rust length error to `PyValueError`; let PyO3/NumPy report dtype,
     dimensionality, and non-contiguity errors without rewriting them.

   Verification: build the extension and run focused Python tests. Confirm
   complex128 contiguous input succeeds, while float64, complex64,
   multidimensional, non-contiguous, and unequal-length inputs raise before a
   numerical call.

8. **Add end-to-end accuracy, regression, and benchmark coverage.**

   Dependencies: steps 1--7.

   - Expand `test/test_math.py` with SciPy comparisons for real `a,b,c`, shared
     full-complex CSV fixtures, strict binding tests, and branch-cut signed-zero
     tests. Include the historical SciPy #1561 case
     `F(1,1;4;3+4i)` as a named regression in the supported SciPy subset.
   - Add `benches/hyp2f1.rs` and its `Cargo.toml` target. Benchmark scalar calls
     and serial/parallel vectors across direct, Pfaff, one, infinity, Taylor,
     near-degenerate, and polynomial cases at several lengths.
   - Record the parallel crossover and check that serial vector throughput is
     close to a hand-written scalar loop. Criterion observations inform the
     private radius and selector order; benchmark numbers are not hard-coded as
     CI pass/fail thresholds.
   - Document the all-complex contract, branch convention, singular behavior,
     no-broadcasting rule, and large-parameter accuracy limitation in Rust docs
     and the Python docstring.
   - Refine the existing user-authored 12.1.0 `CHANGELOG.md` entry if needed to
     match the final exported names; do not add another release heading.

   Verification: run the full Rust and Python CI command set; run Criterion in
   release mode; inspect that the public API addition is accepted by
   `cargo-semver-checks`.

### Final verification commands

Run the repository's checked-in CI commands rather than inventing a separate
local gate:

```bash
cargo fmt --check --verbose
cargo clippy
cargo test --verbose
cargo bench --bench hyp2f1

uv pip install . --group dev
uv run --locked ruff check ./cfsem
uv run --locked ruff format ./cfsem --check
uv run --locked ty check ./cfsem
uv run --locked pytest ./test/
sh build_docs.sh
```

The GitHub `cargo-semver-checks-action` remains the authoritative semver gate.
Run a local `cargo semver-checks` too when that optional tool is installed.

## Test Plan

### Unit tests

+--------------------------------------+-------------------------------------------+-----------------------------------------------+
| Test                                 | Inputs                                    | Expected behavior                             |
+======================================+===========================================+===============================================+
| complex elementary helpers near zero | Small complex values and signed zeros.    | Agreement with high-precision formulas; no    |
|                                      |                                           | loss from 1+w or exp(w)-1.                    |
+--------------------------------------+-------------------------------------------+-----------------------------------------------+
| gamma helper fixtures and identities | General complex values, conjugates, and   | Fixture agreement; recurrence and reflection  |
|                                      | points adjacent to poles.                 | identities within stated tolerances.          |
+--------------------------------------+-------------------------------------------+-----------------------------------------------+
| direct series fixtures               | abs(z) from zero through 0.9 with complex | Agreement with mpmath and clean convergence.  |
|                                      | parameters.                               |                                               |
+--------------------------------------+-------------------------------------------+-----------------------------------------------+
| terminating polynomial               | Negative integral a or b, including       | Exact finite term count and fixture agreement |
|                                      | safe and unsafe integral c.               | or documented NaN at a pole.                  |
+--------------------------------------+-------------------------------------------+-----------------------------------------------+
| Pfaff transformed region             | Negative-real and complex z for which     | Correct value and test-only selector reports  |
|                                      | abs(z/(z-1)) is smallest.                 | the Pfaff path.                               |
+--------------------------------------+-------------------------------------------+-----------------------------------------------+
| one-centered expansion               | z near one; c-a-b generic, integral,      | Stable fixture agreement through epsilon=0.   |
|                                      | and complex-near-integral.                |                                               |
+--------------------------------------+-------------------------------------------+-----------------------------------------------+
| infinity expansion                   | abs(z) > 1; b-a generic, integral, and    | Stable fixture agreement above and below the  |
|                                      | complex-near-integral.                    | branch cut.                                   |
+--------------------------------------+-------------------------------------------+-----------------------------------------------+
| Taylor gap continuation              | Targets around exp(+-i*pi/3) with         | Correct fixtures and the Taylor selector path |
|                                      | radii just below and above one.           | is exercised.                                 |
+--------------------------------------+-------------------------------------------+-----------------------------------------------+
| vector scalar equivalence            | Mixed-region arrays, empty and singleton  | Every finite entry is bitwise equal to its    |
|                                      | arrays.                                   | scalar evaluation; NaN classification agrees. |
+--------------------------------------+-------------------------------------------+-----------------------------------------------+
| serial parallel equivalence          | Multiple lengths around the chunking      | Bitwise-identical finite ordered outputs and  |
|                                      | threshold.                                | matching NaN classification.                  |
+--------------------------------------+-------------------------------------------+-----------------------------------------------+

### Edge-case tests

+----------------------------------+-------------------------------------------+--------------------------------------------------+
| Test                             | Inputs                                    | Expected behavior                                |
+==================================+===========================================+==================================================+
| non-finite inputs                | NaN or infinity in every component and    | Complex NaN without panic or unbounded loop.     |
|                                  | argument position.                        |                                                  |
+----------------------------------+-------------------------------------------+--------------------------------------------------+
| denominator poles                | c = 0,-1,-2,..., with terminating         | Finite only when the polynomial degree is at     |
|                                  | degrees before, at, and after the pole.   | or before the pole index; otherwise complex NaN. |
+----------------------------------+-------------------------------------------+--------------------------------------------------+
| exact special identities         | z=0, a=0, b=0, c=a, c=b, and              | Documented closed-form value.                    |
|                                  | combinations thereof.                     |                                                  |
+----------------------------------+-------------------------------------------+--------------------------------------------------+
| argument unity                   | z=1 with positive, zero, and negative     | Gamma-ratio value only in the convergent case;   |
|                                  | Re(c-a-b).                                | otherwise documented NaN unless polynomial.      |
+----------------------------------+-------------------------------------------+--------------------------------------------------+
| signed-zero branch cut           | x+0j and x-0j, x>1, plus tiny             | Exact cut values approach their corresponding    |
|                                  | positive/negative imaginary perturbations | lip; signs are not collapsed.                    |
+----------------------------------+-------------------------------------------+--------------------------------------------------+
| transformation boundaries        | Values immediately around every selector  | No discontinuity beyond tolerance and no         |
|                                  | radius and abs(z)=1.                      | region left unsupported.                         |
+----------------------------------+-------------------------------------------+--------------------------------------------------+
| forced iteration exhaustion      | Test-only low caps for each iterative     | Failure maps to complex NaN deterministically.   |
|                                  | evaluator.                                |                                                  |
+----------------------------------+-------------------------------------------+--------------------------------------------------+
| length mismatch is transactional | Each input/output shorter in turn, output | Error returned and sentinel output untouched.    |
|                                  | prefilled with sentinels.                 |                                                  |
+----------------------------------+-------------------------------------------+--------------------------------------------------+
| strict Python dtype and layout   | float64, complex64, 2-D, transposed/      | Python exception before backend evaluation.      |
|                                  | strided, and valid complex128 arrays.     | Valid input alone succeeds.                      |
+----------------------------------+-------------------------------------------+--------------------------------------------------+

### Mathematical validation tests

- **Symmetry:** verify `F(a,b;c;z) == F(b,a;c;z)` away from singularities.
- **Conjugation:** off the branch cut, conjugating all four arguments conjugates
  the result.
- **Euler and Pfaff transformations:** compare independently evaluated sides
  across all applicable regions, including complex exponents.
- **Derivative identity:** compare a centered complex finite-difference local
  derivative with `(a*b/c)F(a+1,b+1;c+1;z)` where well conditioned.
- **Hypergeometric differential equation:** form `y`, `y'`, and `y''` from
  shifted functions and verify

  \[
  z(1-z)y''+[c-(a+b+1)z]y'-aby=0.
  \]

- **Branch approach:** values at `x+-i*delta` converge toward the corresponding
  signed-zero branch-cut value as `delta` decreases.
- **Parameter continuity:** sweep `b-a` and `c-a-b` through nearby integers in
  real and imaginary directions and check continuity against mpmath fixtures.

### Regression tests

- Existing `ellipe` and `ellipk` tests continue to pass after extending the
  math module.
- The extension remains importable on Python 3.10--3.13 and all existing public
  symbols remain present.
- Existing Rust tests, Python coverage, Ruff, `ty`, Clippy, formatting,
  documentation, and semver checks remain green.
- Both high-precision reference CSVs are deterministic, identify the pinned
  generator version and precision, and contain no values generated by SciPy
  outside its real-parameter contract.

### Accuracy bands

Set tolerances by conditioning rather than claiming one global ULP target:

- Well-conditioned direct, Pfaff, and special-identity cases: target relative
  error near `1e-13`, with a small absolute floor near zero.
- Stable one/infinity and Taylor continuation cases: target `1e-12` initially.
- Near-pole, severe-cancellation, and moderate-parameter stress cases: retain
  explicit per-fixture tolerances justified by high-precision conditioning;
  do not loosen an entire region because of one difficult point.
- Values that the algorithm cannot support reliably must return NaN or be
  documented as a limitation, not silently pass a very loose tolerance.

## Risk Assessment

### Numerical cancellation near integer parameter differences

This is the dominant correctness risk. Generic connection formulas can produce
two enormous values whose finite sum loses all precision. The mitigation is to
make the stabilized `m + epsilon` recurrences a first-class implementation,
test continuity through epsilon zero, and keep generic formulas behind a
cancellation estimate.

### Complex gamma accuracy and overflow

No existing direct dependency supplies the exact private operations required.
An incorrect reflection branch or separately overflowing gamma factors would
contaminate every continuation path. Gamma helpers are therefore implemented
and validated as an explicit prerequisite, and coefficients use combined
log-gamma expressions and reciprocal gamma at poles.

### Branch-cut inconsistency

Equivalent mathematical transformations can select different branches if
complex logarithms or signed zeros are handled inconsistently. All powers are
routed through a small set of principal-log helpers. Upper/lower lip tests are
required for direct special cases, Pfaff, one-centered, infinity, and Taylor
paths.

### Transformation recursion or uncovered regions

A naive dispatcher can bounce between Euler/Pfaff forms or reach Taylor again
while evaluating its anchors. Forced internal evaluators and a one-time
normalization record prevent cycles. Selector path tests cover all six mapped
regions and both Taylor gaps.

### Large complex parameters

Published transformed-series algorithms can become inaccurate as parameter
magnitudes, particularly imaginary parts, grow. The first release will not
promise uniform accuracy for unbounded parameters. Stress fixtures and
differential-equation residuals characterize the useful range; failure should
be NaN rather than a plausible but unchecked value when convergence tests
detect breakdown.

### Parallel overhead

Scalar calls can be cheap in polynomial and short direct-series cases, so Rayon
may lose for small vectors. The public `par` choice remains explicit, and
Criterion determines the practical crossover. The parallel implementation
uses coarse contiguous chunks rather than one Rayon task per element.

### Python memory safety and GIL behavior

Borrowed NumPy input slices cannot safely outlive GIL protection if arbitrary
Python code may mutate the arrays concurrently. The first binding retains the
GIL, following existing repository behavior, while Rayon performs native work.
Changing this later requires an independently justified ownership or
immutability strategy.

### Licensing

The CPC reference implementation is incompatible with this MIT project and
must not be copied or mechanically translated. Development should cite the
paper's equations and DLMF. If the MIT Julia implementation contributes
substantial translated code, preserve its license and attribution explicitly.

## References

The SciPy manual reviewed for this plan reported version 1.18.0 on 2026-08-17;
the DLMF and xsf links are living references and should be rechecked when the
plan is implemented.

1. [SciPy `scipy.special.hyp2f1` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp2f1.html)
2. [NIST DLMF 15.2, Definitions and Analytical Properties](https://dlmf.nist.gov/15.2)
3. [NIST DLMF 15.8, Transformations of Variable](https://dlmf.nist.gov/15.8)
4. [N. Michel and M. V. Stoitsov, all-complex `2F1` algorithm](https://arxiv.org/abs/0708.0116)
5. [J. W. Pearson, S. Olver, and M. A. Porter, numerical methods review](https://doi.org/10.1007/s11075-016-0173-0)
6. [SciPy/xsf complex-argument implementation](https://github.com/scipy/xsf/blob/main/include/xsf/hyp2f1.h)
7. [MIT-licensed HypergeometricFunctions.jl](https://github.com/JuliaMath/HypergeometricFunctions.jl)
8. [`num-complex` 0.4.6 API and branch conventions](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html)
9. [`spfunc` complex gamma API and development-status note](https://docs.rs/spfunc/0.1.3/spfunc/)
10. [`torsh-special` complex special-functions surface](https://docs.rs/torsh-special/0.1.2/torsh_special/complex/index.html)
11. [SciPy issue #23450: complex-parameter `hyp2f1` enhancement](https://github.com/scipy/scipy/issues/23450)
12. [SciPy issue #1561: historical complex-`z` accuracy defect](https://github.com/scipy/scipy/issues/1561)
