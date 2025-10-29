"""
Lamé's Equations - solutions to thick walled cylinder stress distribution for a configuration
of infinite length, with a fixed pressure boundary condition on the inner and outer radii,
with constant and isotropic material properties and small displacements.
"""

from numpy.typing import NDArray


def s_hoop_thick_wall_cylinder(r: NDArray, ri: float, ro: float, pin: float, pout: float) -> NDArray:
    """
    Hoop stress at a location in a thick walled cylinder under pressure load
    with ends "capped", although the capped constraint does not affect the hoop or radial stress
    compared to an infinite-length constraint.

    https://www.engineeringtoolbox.com/stress-thick-walled-tube-d_949.html
    https://www.suncam.com/miva/downloads/docs/303.pdf

    Args:
        r: [m] radius at which to evaluate
        ri: [m] inner radius
        ro: [m] outer radius
        pin: [Pa] inside pressure
        pout: [Pa] outside pressure

    Returns:
        [Pa] hoop stress
    """
    # Factors of pi cancel out
    # fmt: off
    s_hoop = (
        (pin * ri ** 2 - pout * ro ** 2) / (ro ** 2 - ri ** 2) -
        (ri ** 2 * ro ** 2 * (pout - pin) / (r ** 2 * (ro ** 2 - ri ** 2)))
    )
    # fmt: on

    return s_hoop  # [Pa] hoop stress


def s_radial_thick_wall_cylinder(r: NDArray, ri: float, ro: float, pin: float, pout: float) -> NDArray:
    """
    Radial stress at a location in a thick walled cylinder under pressure load
    with ends "capped", although the capped constraint does not affect the hoop or radial stress
    compared to an infinite-length constraint.

    https://www.engineeringtoolbox.com/stress-thick-walled-tube-d_949.html
    https://www.suncam.com/miva/downloads/docs/303.pdf

    Args:
        r: [m] radius at which to evaluate
        ri: [m] inner radius
        ro: [m] outer radius
        pin: [Pa] inside pressure
        pout: [Pa] outside pressure

    Returns:
        [Pa] radial stress
    """
    # Factors of pi cancel out
    # fmt: off
    s_radial = (
        ((pin * ri**2 - pout * ro**2) / (ro**2 - ri**2)) + \
        (ri**2 * ro**2 * (pout - pin) / (r**2 * (ro**2 - ri**2)))
    )
    # fmt: on

    return s_radial  # [Pa] radial stress
