# cfsem python examples

## Field Explorer

In this interactive example, examine the B-field and A-field of cfsem's finite-length-finite-thickness filament calcs
and compare to point-source calculations, section discretizations, and $B = \nabla \times A$ equivalence tests.

The full example can be run like `uv run --group dev examples/field_explorer.py`; the plot shown below is only an excerpt.

<figure markdown="span">
  <iframe
    class="plotly-embed"
    loading="lazy"
    src="../example_outputs/field_explorer.html"
    title="Field explorer example"
  ></iframe>
  <figcaption>
    B-field of an arrangement of finite-length, finite-thickness filaments.
  </figcaption>
</figure>

## Helmholtz Coil Pair

This example uses [cfsem.flux_density_circular_filament][] to map the B-field of
a [Helmholtz coil pair](https://en.wikipedia.org/wiki/Helmholtz_coil),
an arrangement of two circular coils which produces a region of
nearly uniform magnetic field.

<figure markdown="span">
  ![Helmholtz coil example](example_outputs/helmholtz.png)
  <figcaption>
    B-field of a Helmholtz coil, calculated with cfsem.
    On the left, the red dashes outline where the B-field magnitude is within 1% of its value at (r=0, z=0), and the black dots show where the coils intersect the r-z plane.
  </figcaption>
</figure>

``` py title="examples/helmholtz.py"
--8<-- "examples/helmholtz.py"
```

## High-aspect-ratio Coil Inductance

Estimate the (low-frequency) self- and mutual- inductance of a pair of air-core solenoids,
comparing results from modeling as either collections of axisymmetric loops or
as thin helical filaments.

``` py title="examples/inductance.py"
--8<-- "examples/inductance.py"
```

## Axisymmetric FEM Solenoid Stress Explorer

Explore the axisymmetric FEM stress solver in a Plotly Dash app. The example
varies the solenoid cross-section, current density, and loop-source position,
assembles the sparse FEM system, solves it with `scipy.sparse.linalg.factorized`,
and compares radial sections against the 1D finite-difference reference model.

## Axisymmetric FEM Solenoid Stress Convergence Study

Run an explicit radial-refinement study that configures the 2D FEM model to
match the 1D solver assumptions as closely as possible, then compares both
against a fine 1D reference on the midplane.

## Axisymmetric FEM Surface Traction Example

Run a small non-GUI example that applies constant surface traction vectors in
global `(r, z)` components. The script demonstrates:

- axial traction on the top surface
- radial traction on the outer wall
- combined pressure plus shear-like top traction

## Loop Inductance

Estimate the (low-frequency) self-inductance of a finite-radius wire loop by different methods.

<figure markdown="span">
  ![Helmholtz coil example](example_outputs/loop_inductance.png)
  <figcaption>
    Inductance of a wire loop by different integration methods.
  </figcaption>
</figure>
