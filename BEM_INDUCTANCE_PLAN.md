# Plan: Exact Triangle Potential for BEM Inductance and B-Field

**Date:** 2026-08-06  
**Repo:** cfsem-py  
**Branch:** jlogan/bem-analytic-inductance  
**Status:** Complete

## Feature Request

Replace the source-side numerical triangle integration used by the BEM inductance
kernel with the exact potential of a uniformly loaded flat triangle. Use the exact
formula as the primary path for every self, near, touching, and far pairing. Only
restore the existing quadrature kernel for demonstrably well-separated pairs if
accuracy sweeps or end-to-end benchmarks show a significant far-field regression.

Use the same analytic formulation to rewrite the direct triangle B-field kernel.
At points near but not geometrically on the finite source triangle, use the analytic
observation-point gradient. At points geometrically on that triangle, define that
triangle's B contribution to be exactly zero. This is an explicit convention at the
sheet discontinuity, not a claim that zero is the unique one-sided or principal-value
field.

Add absolute regressions that fail on the current large-aspect-ratio behavior:
the isolated edge-sharing sliver pair through real matrix assembly and the reported
annular-pancake stored energy. Add near-surface, on-surface, jump-condition, mapping,
force, and hierarchical tests for B.

The implementation source is D. R. Wilton et al., “Evaluation of Static Potential
Integrals on Triangular Domains,” IEEE Access, 2020,
<https://doi.org/10.1109/ACCESS.2020.2997287>: stable auxiliaries and limits in
Eqs. (5)-(9), exact gradient in Eq. (15), and parent/plane limits in Eqs. (19)-(21).

### Scope boundaries

- Allow a deliberate breaking change in the B-field APIs: remove the triangle
  quadrature argument from direct, mesh, mapping, and hierarchical B evaluation,
  because analytic evaluation no longer uses it. Retain quadrature arguments where
  they still control real target integration (inductance, vector potential, force).
- Leave the triangle vector-potential kernel unchanged. The shared internal module
  should permit a later migration without redesign.
- Leave hierarchical B far-cluster multipoles unchanged; near leaves inherit the
  direct analytic kernel.
- Do not add a mesh-aspect warning as a substitute for the fix.
- Do not alter filament, dipole, or circular-loop kernels.
- Preserve the current precondition that low-level triangle primitives receive
  nondegenerate triangles; existing mesh validation remains the public guard.

## Reconnaissance Summary

### Relevant Modules

+------------------------------------------------------------+-------------------------------+
| File                                                       | Responsibility and impact     |
+============================================================+===============================+
| src/physics/boundary_element/inductance.rs                  | Current nested Dunavant and    |
|                                                            | self-only Duffy coupling,      |
|                                                            | elemental blocks, and dense   |
|                                                            | assembly. Main consumer of    |
|                                                            | exact scalar potential.       |
+------------------------------------------------------------+-------------------------------+
| src/physics/boundary_element/flux_density.rs                | Current regular, subdivided,   |
|                                                            | and surface-Duffy B paths.     |
|                                                            | Replace with exact gradient    |
|                                                            | and on-triangle zero rule.     |
+------------------------------------------------------------+-------------------------------+
| src/physics/boundary_element/triangle_potential.rs          | New private cached geometry,   |
|                                                            | exact potential/gradient,      |
|                                                            | stable limits, and surface     |
|                                                            | predicate.                    |
+------------------------------------------------------------+-------------------------------+
| src/physics/boundary_element/mod.rs                         | Register module/reference;     |
|                                                            | retain public re-exports.      |
+------------------------------------------------------------+-------------------------------+
| src/mesh/elements/tri/tri3.rs                               | Add longest-edge bisection and |
|                                                            | triangle-pair distance for     |
|                                                            | adaptive target integration.  |
+------------------------------------------------------------+-------------------------------+
| src/physics/boundary_element/body_force_density.rs          | Integrates triangle B over     |
|                                                            | targets; audit zero convention |
|                                                            | and repeated gradient work.    |
+------------------------------------------------------------+-------------------------------+
| src/physics/hierarchical/kernels/                           | Near leaves call direct B; far |
| boundary_element_flux_density.rs                            | summaries must not change.     |
+------------------------------------------------------------+-------------------------------+
| src/physics/boundary_element/test.rs and                    | Rust numerical, physics,       |
| src/physics/hierarchical/tests.rs                           | assembly, force, and hierarchy |
|                                                            | regressions.                  |
+------------------------------------------------------------+-------------------------------+
| test/test_boundary_element.py and test/test_hierarchical.py | Python absolute energy, B,     |
|                                                            | mapping, parallel, hierarchy.  |
+------------------------------------------------------------+-------------------------------+
| cfsem/bindings.py and docs/python/boundary_element.md       | Document exact behavior, zero  |
|                                                            | convention, and quad meaning.  |
+------------------------------------------------------------+-------------------------------+
| scripts/cfsem_inductance_aspect_repro.py                    | Add supplied diagnostic script |
|                                                            | for non-CI reproduction.       |
+------------------------------------------------------------+-------------------------------+
| benches/boundary_element.rs and Cargo.toml                  | Criterion baselines and final  |
|                                                            | performance gate.             |
+------------------------------------------------------------+-------------------------------+
| CHANGELOG.md                                                | Record energy fix and changed  |
|                                                            | on-triangle B convention.      |
+------------------------------------------------------------+-------------------------------+

### Existing Patterns to Follow

- Field primitives are generic over `Scalar`; inductance is `f64`. `Scalar` inherits
  required floating-point operations from `num_traits::Float`.
- Keep unit comments: potential metres, gradient dimensionless, surface current A/m,
  B tesla, coupling m³, inductance henries.
- Mesh APIs validate dimensions and return `Result<(), &'static str>`; low-level
  primitives assume valid geometry.
- Serial and Rayon paths share inner implementations and are tested for identity.
- Python wrappers normalize arrays; physics remains in Rust.
- Inductance averages directed couplings to preserve reciprocity.
- Refactor B to one geometric gradient per triangle/observation, not three.

### Physics Domain Context

For flat triangle \(S\) and constant surface current \(\mathbf K\), define

\[
V_S(\mathbf x)=\int_S\frac{1}{|\mathbf x-\mathbf x'|}\,dS',
\qquad
\mathbf B(\mathbf x)=\frac{\mu_0}{4\pi}\nabla V_S(\mathbf x)\times\mathbf K.
\]

The gradient is with respect to the observation point. For oriented normal
\(\mathbf n\), one-sided limits satisfy

\[
\mathbf B^+-\mathbf B^-=\mu_0\mathbf K\times\mathbf n,
\qquad
\mathbf n\times(\mathbf B^+-\mathbf B^-)=\mu_0\mathbf K.
\]

The potential is finite on triangle interiors, edges, and vertices. Its gradient has
a normal jump and edge singular behavior. Per request, a source triangle returns
`[0, 0, 0]` on its finite surface; a coplanar point outside is not clipped.

Inductance uses \(G(S,T)=\int_T V_S(\mathbf x)\,dS\) in m³. Exact source integration
removes unresolved inner `1/R` quadrature, but a fixed outer rule can still miss
logarithmic variation across a touching sliver. Near-target adaptive integration
therefore remains required. The matrix must be symmetric and stored energy
non-negative to floating-point tolerance.

## Architecture Plan

### Recommended Architecture

Add one private exact-potential module. Inductance consumes its scalar value; B
consumes its gradient. It alone owns edge-sum algebra, normalization, coplanar limits,
and the finite-triangle surface predicate.

Inductance first uses exact source potential for all pairs and adaptive target
integration for self/touching/near pairs, averaging both directions. If the measured
4x gate is crossed, well-separated pairs revert to the legacy nested rule. B removes
surface quadrature: compute one gradient and contract with current density, except for
the explicit on-triangle zero.

### Public API and Internal Interface

The inductance signature stays unchanged because `quad` still controls target
integration and the evidence-driven far fallback. The B signature intentionally loses
`quad`:

```python
def triangle_mesh_inductance_matrix(
    nodes: NDArray[float64],
    triangles: NDArray[int64],
    par: bool = True,
    quad: str = "dunavant3",
) -> NDArray[float64]:
    """Assemble the symmetric nodal BEM inductance matrix.

    The source 1/R integral is analytic. quad controls target integration and
    any evidence-driven well-separated fallback retained by the implementation.
    """


def flux_density_triangle_mesh(
    obs: NDArray[float64],
    nodes: NDArray[float64],
    triangles: NDArray[int64],
    s: NDArray[float64],
    par: bool = True,
) -> Array3xN:
    """Evaluate B from a triangle surface-current mesh.

    Direct interactions use the exact potential gradient. A source triangle
    contributes zero at observations geometrically on that triangle; all other
    triangles are still accumulated.
    """
```

The private Rust interface is deliberately small:

```rust
pub(crate) struct UniformTriangle<T: Scalar> {
    // Vertices and cached local edge/normal/scale geometry are private.
}

impl<T: Scalar> UniformTriangle<T> {
    pub(crate) fn new(n0: [T; 3], n1: [T; 3], n2: [T; 3]) -> Self;
    pub(crate) fn scalar_potential(&self, obs: [T; 3]) -> T;
    pub(crate) fn scalar_potential_gradient(&self, obs: [T; 3]) -> [T; 3];
    pub(crate) fn contains_on_surface(&self, obs: [T; 3]) -> bool;
}
```

`new` caches normal, edge tangents, in-plane edge normals, area, maximum edge, and
normalization scale. Fields stay private. A debug assertion preserves the existing
nondegenerate-triangle precondition.

Keep public `triangle_geometric_coupling` and
`triangle_geometric_coupling_regular` signatures. Remove `quad_kind` from public
`triangle_flux_density_basis`, `flux_density_triangle`, mesh/mapping B, and
hierarchical-B constructors/helpers. The regular coupling remains a legacy baseline
and evidence-driven far fallback.

### Exact Evaluator

1. Translate to an observation-projection-centered local frame.
2. Scale lengths by maximum edge. Rescale \(V\) by length; \(\nabla V\) is invariant.
3. Form signed quantities from Eqs. (5)-(8), using `asinh` and `atan2` rather than
   subtractive logs or one-argument arctangent.
4. Sum Eq. (9) for V and Eqs. (15)/(19) for observation gradient.
5. Implement Eqs. (20)-(21) for coplanar/edge/vertex limits; avoid `0 * ln(0)` and NaN.
6. Use magnitude-ordered or compensated edge summation if far sweeps expose cancellation.
7. Prove gradient sign with finite differences and sheet jump before connecting B.

`contains_on_surface` tests the finite triangle using closest-point distance:

\[
|\mathbf x-\mathrm{closest}_S(\mathbf x)|\le64\,\epsilon_TL_{\max}.
\]

This roundoff-scale threshold means representable \(10^{-12}L\) offsets are analytic.
Coplanar outside points remain analytic.

### Inductance Integration

For directed \(G(S,T)\):

1. Retrieve cached exact source geometry.
2. Evaluate exact V at selected target Dunavant points.
3. Accept one target rule for well-separated pairs; source V is still exact.
4. For self/touching/near, compare parent with two longest-edge children and recurse.
   Cache each child estimate so accepted work is not recomputed.
5. Use relative tolerance \(10^{-5}\) and root absolute tolerance
   \(10^{-14}L_{\mathrm{ref}}^3\), where \(L_{\mathrm{ref}}\) is the larger maximum
   edge. Split the absolute budget by child target area. Scale the error test with an
   absolute-integral estimate so negative Dunavant weights cannot hide cancellation.
6. Before trusting the error estimator for touching/intersecting pairs, force at least
   \(\lceil\log_2(\max(1,a_S,a_T))\rceil\) levels, capped at 10, where each \(a\) is
   maximum edge divided by minimum altitude. This prevents Dunavant-1 aliasing from
   accepting an unresolved sliver.
7. Cap total depth at 14. At the cap, return the finer child sum rather than panic.
   Test-only statistics report capped leaves; all acceptance fixtures must report zero.
8. Define near by true triangle distance no greater than the larger maximum edge.
   Include vertex-face and edge-edge candidates so shared edges report zero.
9. Average \(G(S,T)\) and \(G(T,S)\); avoid duplicate identical-triangle work.

Longest-edge bisection improves sliver shape. Precompute `UniformTriangle<f64>` once
per mesh triangle. Adaptivity and tolerances remain private.

### B-Field Evaluation

For one constant \(\mathbf K\):

1. Retrieve cached geometry.
2. Return exact zero before gradient if `contains_on_surface(obs)`.
3. Otherwise evaluate exact gradient, including close signed offsets/coplanar outside.
4. Return \(\mu_0/(4\pi)\nabla V\times\mathbf K\).

Compute physical `triangle_current_density` and one gradient in direct B. Mappings
compute one gradient and combine it with three basis currents. Keep
`triangle_flux_density_basis` as a convenience wrapper, with no quadrature argument.

After tests, remove B-only quadrature, near subdivision, and surface Duffy. Keep the
subdivision constant used by vector potential.

The zero rule is per source triangle. Other triangles still contribute. Self-force
already skips its identical source; retain/test that explicit exclusion.

### Far-Field Decision Gate

First implement exact potential/gradient for every direct source. A regression is
significant if either:

- release single-thread aspect-one dense assembly or representative direct B is more
  than 4× slower over at least five Criterion samples; or
- exact evaluation fails an independent \(1L\) to \(10^8L\) far sweep where legacy
  meets tolerance because of closed-form cancellation.

If neither occurs, retain exact-all and remove unused fallback helpers. If one occurs,
use legacy only where actual distance exceeds one maximum edge. Self/near remain
analytic. Decide independently for inductance and B, rerun all tests, and document
evidence. Do not add a public tuning option. Current measurements put all-analytic
far inductance at about 35x the legacy pair cost, so it uses the fallback; analytic B
is about 2.1x the legacy Dunavant-3 path and remains analytic for every direct pair.

### Information Hiding

- `triangle_potential.rs` hides frames, signed edge terms, stable limits, and scaling.
- `inductance.rs` hides pair distance, target recursion, budgets, and symmetry.
- `flux_density.rs` hides geometric reuse across physical/basis currents.
- `tri3.rs` hides segment/face distance and bisection details.
- Benchmarks/repro are validation tools, not runtime configuration.

### Data Flow

```text
Inductance:
mesh -> cached geometry -> target quadrature/adaptive bisection -> exact V
     -> directed G -> reciprocal average -> mu0/(4 pi) G (K_i dot K_j)
     -> dense matrix -> 0.5 phi^T M phi

B:
triangle + stream + observation -> cached geometry + constant K -> surface test
     -> on: zero / off: exact grad(V) -> mu0/(4 pi) grad(V) cross K
     -> direct sum / mapping / force / hierarchical near leaf
```

### Physics Constraints

+--------------------------+----------------------------+-----------------------------+
| Constraint               | Enforcement location       | Mechanism                   |
+==========================+============================+=============================+
| V has units of metres    | triangle_potential.rs      | Unit comments and scale/    |
| and scales with length.  | tests                      | translation tests.          |
+--------------------------+----------------------------+-----------------------------+
| Gradient is observation- | evaluator tests            | Finite differences and      |
| point gradient.          |                            | signed sheet jump.          |
+--------------------------+----------------------------+-----------------------------+
| B has correct cross/sign.| flux_density.rs            | One contraction helper and  |
|                          |                            | independent references.     |
+--------------------------+----------------------------+-----------------------------+
| Source B is exact zero   | evaluator and B wrapper    | Closest-point predicate and |
| on finite triangle.      |                            | early return.               |
+--------------------------+----------------------------+-----------------------------+
| Near offsets not clipped.| predicate tests            | 64 epsilon L and 1e-12 L    |
|                          |                            | signed tests.               |
+--------------------------+----------------------------+-----------------------------+
| Coupling is reciprocal.  | inductance.rs              | Directed average and matrix |
|                          |                            | symmetry assertions.        |
+--------------------------+----------------------------+-----------------------------+
| Energy is non-negative   | Rust/Python regressions    | Random admissible quadratic |
| to roundoff.             |                            | forms, scale-aware bound.   |
+--------------------------+----------------------------+-----------------------------+
| On-triangle V is finite. | exact evaluator            | Explicit interior/edge/     |
|                          |                            | vertex limits.              |
+--------------------------+----------------------------+-----------------------------+
| Far behavior is stable.  | evaluator/B tests          | Sweep to 1e8 L versus       |
|                          |                            | asymptotic/high precision.  |
+--------------------------+----------------------------+-----------------------------+
| Direct B is divergence-  | docs and physics test      | Curl construction and a     |
| free away from source.   |                            | finite-difference check.    |
+--------------------------+----------------------------+-----------------------------+

### Approaches Considered

+--------------------------+---------------------------+-------------------------------+
| Approach                 | Advantages                | Decision                      |
+==========================+===========================+===============================+
| Exact V/gradient all     | Removes inner singular    | Chosen; direct fix with an    |
| direct pairs; adaptive   | rule; one deep module.    | evidence-only far fallback.  |
| target near              |                           |                               |
+--------------------------+---------------------------+-------------------------------+
| Exact only self/near     | Minimizes far-path change.| Reserve only as fallback; it  |
|                          |                           | retains two kernels/threshold.|
+--------------------------+---------------------------+-------------------------------+
| Port Duffy subdivision   | Small initial diff.       | Rejected: remains rule/aspect |
| from vector potential    |                           | dependent, no exact gradient.|
+--------------------------+---------------------------+-------------------------------+

## Implementation Steps

### 1. Freeze failures and baselines

**Dependencies:** none.

- Add supplied `scripts/cfsem_inductance_aspect_repro.py` with only import/format changes.
- Confirm ~2.26× aspect-67 annulus and pair error growth from aspect 1 through 67.
- Add/register `benches/boundary_element.rs`: self/near/far, aspects 1/67, aspect-one/
  sliver matrices, direct B, mapping, and force.
- Record output in PR notes, not generated repository files.

**Verify:** repro fails current kernel as reported; Criterion runs stably.

### 2. Build/test exact triangle potential

**Dependencies:** step 1.

- Create/register `triangle_potential.rs` and `UniformTriangle<T>`.
- Implement normalized 2020 potential, gradient, limits, and predicate with citations.
- Keep disconnected from production until independent tests pass.

**Verify:** adaptive/high-precision references, finite differences, transformations,
near/on/far, and meaningful `f32`/`f64` checks pass without production helpers in refs.

### 3. Build geometry helpers/adaptive directed coupling

**Dependencies:** step 2.

- Add/test pair distance and longest-edge bisection in `tri3.rs`.
- Add exact directed target integrator in `inductance.rs`.
- Implement cached child estimates, dimensional/relative error budgets, shape-based
  minimum refinement, near classifier, finite cap behavior, and test-only statistics.
- Leave public `triangle_geometric_coupling_regular` unchanged.

**Verify:** self, edge-sharing, near, far converge; aspect 67 stays below depth 14.

### 4. Migrate production inductance

**Dependencies:** steps 2-3.

- Rewrite main coupling around exact directed integration and reciprocal average.
- Remove production self-only Duffy/obsolete constant after tests.
- Precompute geometry per dense assembly; preserve output, scatter, APIs, Rayon.

**Verify:** pair extraction, annular energy, existing absolute tests, parallel identity,
reciprocity, symmetry, and energy-sign pass.

### 5. Migrate direct B/mappings

**Dependencies:** step 2.

- Add cached geometry/current contraction with early surface zero.
- Refactor direct B to one K/gradient and mappings to one gradient/three bases.
- Remove the now-unused B-field `quad_kind` arguments across Rust and Python APIs.
- Remove B-only quadrature/subdivision/Duffy after tests; retain vector-A dependencies.

**Verify:** zero, signed-near, jump, mapping, parallel, scalar precision, filament refs.

### 6. Audit force/hierarchical consumers

**Dependencies:** step 5.

- Reuse optimized geometry in force code where practical.
- Preserve explicit self-force identical-source exclusion.
- Add independent/symmetry force reference, not only same-helper comparison.
- Confirm hierarchical near inherits exact B; leave far unchanged.

**Verify:** force and hierarchy tests, self exclusion, theta-zero on/near cases pass.

### 7. Execute far-field gate

**Dependencies:** steps 4-6.

- Sweep V/gradient \(1L\) to \(10^8L\); run five release benchmark samples.
- Retain exact-all by default. Activate legacy only for distance \(>1L\), separately per
  kernel, if documented threshold crosses. Record the measured ratio.
- Rerun all tests; expose no public threshold.

**Verify:** PR records evidence and final path.

### 8. Documentation/changelog

**Dependencies:** final step-7 decision.

- Update Rust docs, Python direct/mapping/hierarchical B and inductance docstrings,
  `docs/python/boundary_element.md`, and `CHANGELOG.md`.
- State final `quad` behavior and zero convention; recommend signed offset for one side.

**Verify:** docs build and agree.

### 9. Full verification

**Dependencies:** steps 1-8.

- Run fmt, repository Clippy flags, Rust tests, Python format/lint/type/full tests.
- Run repro; inspect symmetry, energy bound, jump sign, and parallel differences.

**Verify:** all CI-equivalent commands and acceptance criteria pass.

## Test Plan

### Unit

- **`triangle_potential_matches_independent_reference`:** far, ordinary, near, coplanar
  inside/outside, edge, vertex. Target \(5\times10^{-12}\) off limits and
  \(5\times10^{-10}\) at explicit limits.
- **`triangle_potential_gradient_matches_finite_difference`:** both sides/outside,
  converged central-difference window; \(10^{-8}\) relative and correct sign.
- **`triangle_potential_is_rigid_motion_and_scale_covariant`:** translate, rotate,
  permute, scale \(10^{-9}\) to \(10^9\). V scales; gradient rotates/scale invariant.
- **`triangle_on_surface_predicate_is_finite_triangle_only`:** interior/edges/vertices,
  coplanar outside, signed \(10^{-12}L\), \(10^{-10}L\), \(10^{-6}L\).
- **`triangle_pair_distance_and_bisection_are_robust`:** disjoint, touching,
  intersecting, skew, aspect 67/permutations; distance, area, longest edge.
- **`directed_exact_coupling_matches_independent_reference`:** self, edge-sharing, near,
  far against reference sharing no production helper.
- **`triangle_flux_density_uses_gradient_cross_current`:** ordinary/near versus
  independent high-order Biot-Savart.

### Edge cases

- **`triangle_potential_limits_are_finite`:** interior/edges/vertices for equilateral,
  scalene, aspect 67; no NaN/Inf.
- **`triangle_b_is_exactly_zero_on_source_triangle`:** interior/edges/vertices; bitwise
  zero for basis and physical-current paths.
- **`triangle_b_coplanar_outside_is_not_clipped`:** just beyond edge/far outside;
  finite nonzero reference match.
- **`triangle_b_near_surface_is_not_clipped`:** signed offsets through \(10^{-12}L\);
  finite one-sided values/nonzero jump.
- **`exact_triangle_far_field_remains_stable`:** \(R/L=1\) through \(10^8\);
  correct asymptote, no sign flip/NaN.
- **`adaptive_target_respects_depth_and_tolerance`:** self/shared-edge aspect 67 gets
  the shape-derived minimum refinement and stays below the total depth cap. A strict
  test-only tolerance exercises deterministic finer-sum cap behavior and statistics.

### Physics validation

- **`triangle_b_satisfies_sheet_jump_condition`:** signed interior offsets; jump tends
  to \(\mu_0\mathbf K\times\mathbf n\) within \(10^{-5}\).
- **`triangle_b_is_divergence_free_off_surface`:** finite-difference divergence small
  relative to \(B/L\).
- **`triangle_coupling_is_reciprocal`:** ordinary, skew, touching, self, sliver.
- **`triangle_inductance_matrix_is_symmetric_and_energy_nonnegative`:** aspect-one/
  sliver matrices and admissible random quadratic forms.
- **`flux_density_mapping_matches_direct_contraction_independently`:** several streams
  plus independent single-triangle value to avoid self-comparison.
- **`self_force_mapping_still_excludes_identical_triangle`:** explicit sum over
  nonidentical sources; serial/parallel equality.

### Regression

- **`edge_sharing_sliver_pair_coupling_regression`:** aspects 1, 4, 16, 67 at fixed area.
  Extract G via two-triangle matrix minus singles. Independent converged constants,
  <0.5% each. Current code fails by aspect 4.
- **`annular_pancake_energy_matches_filament_reference`:** 0.5-0.50883 m annulus,
  9×48 cells, 16 kA radial-linear stream; \(0.5\phi^TM\phi\) within 1% of converged
  coaxial filaments. Current ~2.26 ratio fails.
- **`annular_energy_error_not_aspect_dependent`:** aspects 1, 4, 16, 67 and wide-band
  high-aspect controls; <1% retained fixtures.
- **`inductance_quadrature_orders_agree_on_sliver`:** Dunavant 1/3/5 pair and compact
  annulus meet absolute tolerance, removing 1.80/2.26/1.11 pattern.
- **`well_shaped_mesh_accuracy_does_not_regress`:** preserve aspect-one and strip/
  circular-loop comparisons.
- **`triangle_b_matches_existing_far_and_filament_references`:** preserve circular
  filament, axis, mapping, parallel tests.
- **`hierarchical_theta_zero_matches_exact_direct_b`:** ordinary, near, on-triangle;
  theta zero equals direct, nonzero theta retains bounds.

### Performance

Criterion review gates, not CI assertions:

- exact potential/gradient versus legacy Dunavant 1/3/5 far/near;
- self, shared-edge aspects 1/67, well-separated coupling;
- aspect-one/sliver matrices serial/parallel;
- direct B, mapping, force blocks;
- exact-all versus any proposed far fallback. Current measurements: inductance
  exact-all ~35x and reciprocal production fallback ~2x legacy for far pairs; B analytic
  ~2.1x legacy Dunavant-3.

## Acceptance Criteria

1. Pair assembly within 0.5% at aspects 1, 4, 16, 67.
2. Narrow annulus within 1%; well-shaped controls do not regress.
3. Reciprocity/symmetry remain at roundoff.
4. Single-source on-triangle B exact zero; \(10^{-12}L\) offsets not clipped.
5. One-sided B jump has correct sign/magnitude.
6. Direct, mapping, force, parallel, hierarchical-near pass independent tests.
7. Exact-all remains production unless documented far gate crosses; the current
   inductance measurement justifies far-only fallback, while B remains exact-all.
8. All CI-equivalent checks pass.

## Risk Assessment

+---------------------------+----------+-----------------------------------------+
| Risk                      | Severity | Mitigation                              |
+===========================+==========+=========================================+
| Far cancellation          | High     | Normalize, asinh/atan2, ordered sums,   |
|                           |          | sweep to 1e8 L, evidence-only fallback. |
+---------------------------+----------+-----------------------------------------+
| Gradient sign/orientation | High     | Finite difference, Biot-Savart, vertex  |
| error                     |          | permutations, sheet jump.               |
+---------------------------+----------+-----------------------------------------+
| Exact inner leaves sliver | High     | Near adaptive longest-edge target and   |
| target unresolved         |          | pair/annulus acceptance.                |
+---------------------------+----------+-----------------------------------------+
| Surface clips real offset | High     | 64 epsilon L; 1e-12 L/outside tests.    |
+---------------------------+----------+-----------------------------------------+
| Zero mistaken for PV      | Medium   | Document per-source convention and test |
|                           |          | one-sided/other contributions.          |
+---------------------------+----------+-----------------------------------------+
| Performance regression    | Medium   | Cache geometry, one gradient, 4x gate.  |
+---------------------------+----------+-----------------------------------------+
| Recursion cost/cap        | Medium   | True distance, near-only two-child      |
|                           |          | refinement, convergence/depth tests.    |
+---------------------------+----------+-----------------------------------------+
| Force behavior changes    | Medium   | Audit call graph, explicit self exclude,|
|                           |          | independent force tests.                |
+---------------------------+----------+-----------------------------------------+
| Self-referential tests    | High     | High-precision constants, matrix        |
|                           |          | extraction, filament energy.            |
+---------------------------+----------+-----------------------------------------+
| Degenerate triangles      | Low      | Existing precondition/validation and    |
|                           |          | debug assertion.                        |
+---------------------------+----------+-----------------------------------------+

## Known Limitations

- On-source zero is intentionally discontinuous. Use signed normal offset for a side.
- Vector potential retains numerical quadrature and closest-point subdivision.
- Hierarchical far B retains its opening-controlled approximation.
- Performance gates are reviewed on hardware, not asserted in CI.

## Implementation Outcome

- Aspect-67 annulus energy is 455.27/455.81/455.82 J for Dunavant 1/3/5 versus
  the independent 456.29 J filament reference; the previous default result was
  about 1029 J.
- The isolated shared-edge pair meets the independent reference within 0.05% at
  aspects 1, 4, 16, and 67 for Dunavant 1/3/5.
- Analytic direct B costs about 2.1x the legacy far-field rule, below the 4x gate,
  and remains analytic everywhere off the source triangle.
- All-analytic far inductance cost about 35x legacy, so well-separated pairs use
  a reciprocal nested-quadrature fallback. Its measured pair cost is about 2x
  legacy; self and near interactions remain analytic/adaptive.
- Final verification: 130 Rust tests and 727 Python tests pass, with 100% Python
  coverage; formatting, lint, type checks, and docs build also pass.

## References

1. D. R. Wilton et al., “Evaluation of Static Potential Integrals on Triangular
   Domains,” IEEE Access, 2020.
   <https://doi.org/10.1109/ACCESS.2020.2997287>.
2. D. Wilton et al., “Potential integrals for uniform and linear source distributions
   on polygonal and polyhedral domains,” IEEE TAP, 1984.
   <https://doi.org/10.1109/TAP.1984.1143304>.
3. G. N. Peeren, “Stream function approach for determining optimal surface currents,”
   Eindhoven University of Technology, 2003.
   <https://doi.org/10.6100/IR570424>.
