"""
Radial and hoop stress in an infinitely long solenoid under linearly-varying self field.
The "infinite length" assumption is equivalent to assuming zero R-Z shear ("deck of cards")
and assuming no B-field in the R-direction (no Z-load or stress).

The linearly varying B-field from inside to outside allows slightly extending this
to partially account for the finite length of a real solenoid, which produces
a region of negative Bz near the outer radius (as opposed to the true infinite solenoid,
for which Bz trends to exactly zero at the outer radius).

Iwasa 2e pg 101 eqns 3.77a,b .
"""

from numpy.typing import NDArray


def s_long_solenoid(
    r: NDArray,
    ri: float,
    ro: float,
    j: float,
    bzi: float,
    bzo: float,
    poisson_ratio: float,
) -> tuple[NDArray, NDArray]:
    """
    Radial and hoop stress in an infinitely long solenoid under linearly-varying self field.
    The "infinite length" assumption is equivalent to assuming zero R-Z shear ("deck of cards")
    and assuming no B-field in the R-direction (no Z-load or stress).

    The linearly varying B-field from inside to outside allows slightly extending this
    to partially account for the finite length of a real solenoid, which produces
    a region of negative Bz near the outer radius (as opposed to the true infinite solenoid,
    for which Bz trends to exactly zero at the outer radius).

    Iwasa 2e pg 101 eqns 3.77a,b .

    Assumes
    * Infinitely long solenoid (no R-field or Z-stress).
    * Linear B-field fall-off between inner and outer radius
      * "Very long" solenoid - allows some negative field at the OD, but always linearly varying
    * Uniform current density; no bulk regions of non-conducting structure
    * Radial stress at inner and outer radius is zero (BC due to no support)
    * Isotropic material
    * No thermal stress

    Can acommodate a uniform or linearly-varying background field, but not general fields.

    Args:
        r: [m] (n x 1) array of radius points at which to evaluate the stress
        ri: [m] inner radius
        ro: [m] outer radius
        j: [A/m^2] current density
        bzi: [T] axial B-field at inner radius
        bzo: [T] axial B-field at outer radius
        poisson_ratio: [dimensionless] Material property; off-axis stress coupling term

    Returns:
        s_radial, s_hoop - each (n x 1) with units of [Pa]
    """

    # Terms shared between s_radial and s_hoop
    nu = poisson_ratio
    rho = r / ri
    alpha = ro / ri
    kappa = bzo / bzi

    jbr = j * bzi * ri  # [Pa]
    term1 = jbr / (alpha - 1.0)  # [Pa]
    term2 = (2.0 + nu) / 3.0
    term3 = (3.0 + nu) / 8.0
    term4 = alpha - kappa
    term5 = 1.0 - kappa

    # s_radial
    term6 = ((alpha**2 + alpha + 1.0 - alpha**2 / rho**2) / (alpha + 1.0)) - rho
    term7 = term3 * term5 * (alpha**2 + 1.0 - alpha**2 / rho**2 - rho**2)
    s_radial = term1 * (term2 * term4 * term6 - term7)  # [Pa]

    # s_hoop
    term8 = term2 * (alpha**2 + alpha + 1.0 + alpha**2 / rho**2) / (alpha + 1.0)
    term9 = rho * (1.0 + 2.0 * nu) / 3.0
    term10 = term4 * (term8 - term9)

    term11 = term3 * (alpha**2 + 1.0 + alpha**2 / rho**2)
    term12 = rho**2 * (1.0 + 3.0 * nu) / 8.0
    term13 = term5 * (term11 - term12)

    s_hoop = term1 * (term10 - term13)  # [Pa]

    return s_radial, s_hoop  # [Pa]
