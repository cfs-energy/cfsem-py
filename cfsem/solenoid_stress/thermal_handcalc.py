from __future__ import annotations

import numpy as np
import numpy.typing as npt

def s_thermal_long_cylinder_linear_temperature(
    radius: npt.ArrayLike,
    ri: float,
    ro: float,
    elasticity_modulus: float,
    poisson_ratio: float,
    alpha: float,
    temperature_inner: float,
    temperature_outer: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return analytic thermal stress for a long isotropic hollow cylinder.

    The temperature field is assumed to vary linearly with radius from
    `temperature_inner` at `ri` to `temperature_outer` at `ro`. The returned
    stress components correspond to the classical infinitely long-cylinder
    solution with traction-free inner and outer radii.

    Args:
        radius: Sample radii with shape `(n,)` or any array-like shape broadcastable
            to a one-dimensional radius vector. Units are `[length]`.
        ri: Inner radius. Units are `[length]`.
        ro: Outer radius. Units are `[length]`.
        elasticity_modulus: Isotropic Young's modulus. Units are `[pressure]`.
        poisson_ratio: Isotropic Poisson ratio. Units are `[dimensionless]`.
        alpha: Isotropic thermal expansion coefficient. Units are
            `[strain / temperature]`.
        temperature_inner: Temperature at `ri`. Units are `[temperature]`.
        temperature_outer: Temperature at `ro`. Units are `[temperature]`.

    Returns:
        tuple: Analytic stress arrays `(sigma_rr, sigma_tt, sigma_zz)`, each with
        shape matching `np.asarray(radius).shape` and units `[stress] = [pressure]`.

    References:
        Lee, C. C., *A Note on Thermal Stresses in a Hollow Cylinder of Linearly
        Varying Temperature*, Lehigh University, 1961, Eq. 1.1.
        https://preserve.lehigh.edu/system/files/derivatives/coverpage/427261.pdf
    """

    radius_arr = np.asarray(radius, dtype=np.float64)
    slope = (temperature_outer - temperature_inner) / (ro - ri)
    intercept = temperature_inner - slope * ri

    def temperature(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return intercept + slope * x

    def temperature_moment(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 0.5 * intercept * (x * x - ri * ri) + (slope / 3.0) * (x**3 - ri**3)

    integral_outer = float(temperature_moment(np.asarray(ro, dtype=np.float64)))
    integral_radius = temperature_moment(radius_arr)
    prefactor = alpha * elasticity_modulus / (1.0 - poisson_ratio)
    annulus_area_factor = ro * ro - ri * ri

    sigma_rr = prefactor * (
        ((radius_arr * radius_arr - ri * ri) / (radius_arr * radius_arr * annulus_area_factor))
        * integral_outer
        - integral_radius / (radius_arr * radius_arr)
    )
    sigma_tt = prefactor * (
        ((radius_arr * radius_arr + ri * ri) / (radius_arr * radius_arr * annulus_area_factor))
        * integral_outer
        + integral_radius / (radius_arr * radius_arr)
        - temperature(radius_arr)
    )
    sigma_zz = prefactor * ((2.0 * integral_outer / annulus_area_factor) - temperature(radius_arr))
    return (
        np.asarray(sigma_rr, dtype=np.float64),
        np.asarray(sigma_tt, dtype=np.float64),
        np.asarray(sigma_zz, dtype=np.float64),
    )
