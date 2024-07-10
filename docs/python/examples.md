# cfsem python examples

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
