/// Formulas for volumetric sources.
/// These are context-heavy and require some care to apply
/// in a way that is consistent with assumptions,
/// so they are kept out of the public API.
use crate::MU0_OVER_4PI;

/// Magnetic flux density inside a uniformly magnetized sphere
/// with some radius and total magnetic moment.
///
/// The field outside the sphere is consistent with an ideal dipole, and
/// there is a discontinuity in the field at the surface of the sphere
/// consistent with the apparent discontinuity in material properties.
///
/// The dipole moment of the magnetized sphere can be calculated as
/// m = 4/3 pi R^3 M, where R is the sphere's outer radius and M is
/// the (vector) uniform magnetization.
///
/// Based on Griffith's 5e eqn 6.16 with some manipulation to phrase
/// in terms of dipole moment and to reduce float roundoff and number of operations.
///
/// The formula implemented here is
/// B = 2/3 mu_0 M
/// where M is the vector magnetization and r is the vector
/// from the center of the sphere to the observation point.
/// The magnetization M is then replaced with the formula above
/// in terms of the total magnetic moment.
///
/// References
///
/// * \[1\] “Griffith, D.J. (2024) Introduction to Electrodynamics. 5th Edition”
///
/// Arguments
///
/// * moment: (A-m^2) magnetic moment vector of the sphere
/// * outer_radius: (m) the radius of the sphere
///
/// Returns
///
/// * (bx, by, bz) [T] magnetic field components anywhere inside the sphere
#[inline]
pub(crate) fn flux_density_inside_magnetized_sphere(
    moment: (f64, f64, f64),
    outer_radius: f64,
) -> (f64, f64, f64) {
    let r3 = outer_radius * outer_radius * outer_radius;
    let c = 2.0 * MU0_OVER_4PI / r3;
    (moment.0 * c, moment.1 * c, moment.2 * c)
}

/// Magnetic vector potential inside a uniformly magnetized sphere
/// with some radius and total magnetic moment.
///
/// The field outside the sphere is consistent with an ideal dipole, and
/// there is a discontinuity in the field at the surface of the sphere
/// consistent with the apparent discontinuity in material properties.
///
/// The dipole moment of the magnetized sphere can be calculated as
/// m = 4/3 pi R^3 M, where R is the sphere's outer radius and M is
/// the (vector) uniform magnetization.
///
/// Based on Griffith's 5e eqn 6.16 with some manipulation to phrase
/// in terms of dipole moment, reduce float roundoff and number of operations,
/// and to extract vector potential.
///
/// The formula implemented here is
/// A = 1/3 mu_0 cross(M, r)
/// where M is the vector magnetization and r is the vector
/// from the center of the sphere to the observation point.
/// The magnetization M is then replaced with the formula above
/// in terms of the total magnetic moment.
///
/// References
///
/// * \[1\] “Griffith, D.J. (2024) Introduction to Electrodynamics. 5th Edition”
///
/// Arguments
///
/// * mhat_cross_rhat: (dimensionless) precalculated direction of A-field
/// * mmag: (A-m^2) magnitude of magnetic moment of the sphere
/// * rmag: (m) magnitude of radius vector from dipole origin to observation point
/// * outer_radius: (m) the radius of the sphere
///
/// Returns
///
/// * (ax, ay, az) [V-s/m] magnetic vector potential components at the target location
#[inline]
pub(crate) fn vector_potential_inside_magnetized_sphere(
    mhat_cross_rhat: (f64, f64, f64),
    mmag: f64,
    rmag: f64,
    outer_radius: f64,
) -> (f64, f64, f64) {
    let r3 = outer_radius * outer_radius * outer_radius;
    let c = MU0_OVER_4PI * mmag * rmag / r3;
    (
        mhat_cross_rhat.0 * c,
        mhat_cross_rhat.1 * c,
        mhat_cross_rhat.2 * c,
    )
}
