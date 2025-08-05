import os
from pathlib import Path
from time import monotonic_ns

import numpy as np
import pytest
from pytest import approx
from scipy.sparse.linalg import factorized

from cfsem.solenoid_stress.solenoid_1d import (
    SolenoidStress1D,
    solenoid_1d_structural_factor,
    solenoid_1d_structural_rhs,
)
from cfsem.solenoid_stress.solenoid_handcalc import s_long_solenoid
from cfsem.solenoid_stress.thick_wall_cylinder_handcalc import (
    s_hoop_thick_wall_cylinder,
    s_radial_thick_wall_cylinder,
)


@pytest.mark.parametrize("r0", [0.5, 0.7])
@pytest.mark.parametrize("r1", [1.0, 1.8])
@pytest.mark.parametrize("order", [2, 4])
def test_solenoid_stress(r0, r1, order):
    """Make sure the analytic and finite-difference calcs match for a geometry
    that they can both represent"""

    #
    # Grid
    #

    # Section geometry
    # For testing against the analytic calc, the winding pack is the whole section,
    # but the FD calc can have rwp<r1 creating a structural case in the region [rwp, r1]
    rwp = r1

    # Material properties, roughly SS316
    poisson_ratio = 0.27  # [dimensionless]
    elasticity_modulus = 200 * 1e9  # [Pa]

    # Settings
    dx_reqd = 0.001  # [m] target resolution

    nr = int(np.ceil((r1 - r0) / dx_reqd)) + 1
    rgrid = np.linspace(r0, r1, nr)
    nudge = 1e-6  # Pad with points just outside to avoid nulling any real current density
    rgrid = np.array([r0 - nudge] + rgrid.tolist() + [r1 + nudge])
    r0, r1 = rgrid[0], rgrid[-1]  # Account for nudge in later calcs
    nr = nr + 2

    # Linearly varying B-field, constant current density
    j = 0.2 * 390 * 1e6  # [A/m^2] from A/mm^2
    bz_guess = 27.0  # [T]
    rlin = np.zeros_like(rgrid)
    wpinds = np.where(rgrid <= rwp + nudge)
    rlin[wpinds] = 1.0 - (rgrid[wpinds] - r0) / (rwp - r0)
    # Linear B-field profile is pretty reasonable, reflects infinite solenoid
    bz = bz_guess * rlin

    #
    # Build operators and rhs
    #

    c = solenoid_1d_structural_factor(elasticity_modulus, poisson_ratio)
    rhs = solenoid_1d_structural_rhs(c, j, bz)
    operators = SolenoidStress1D(
        rgrid=rgrid,
        elasticity_modulus=elasticity_modulus,
        poisson_ratio=poisson_ratio,
        order=order,
        direct_inverse=True,
    ).operators

    a_bu = operators.a_bu
    a_ub = operators.a_ub
    a_eu = operators.a_eu
    a_eu_radial = operators.a_eu_radial
    a_eu_hoop = operators.a_eu_hoop
    a_se = operators.a_se

    #
    # Solve for displacement
    #

    print("Solving for displacement")

    start_time = monotonic_ns()
    # Direct inverse method
    # About 30us to do the matrix multiplication
    # np.squeeze and np.ravel fail here for some reason, but we need u_r to be flat for plotting
    u_r = np.squeeze(np.asarray(a_ub @ rhs))

    end_time = monotonic_ns()
    print(f"    Solved using pre-calculated direct inverse in {(end_time - start_time) / 1e9:.6f} s")

    start_time = monotonic_ns()
    # The better solve method that isn't an operator
    # About 100us solve
    a_ub_solver = factorized(a_bu)
    u_r_direct_solver = a_ub_solver(rhs)

    end_time = monotonic_ns()
    print(f"    Solved using umfpack solver in {(end_time - start_time) / 1e9:.6f} s")

    assert np.allclose(u_r, u_r_direct_solver, rtol=1e-6, atol=1e-5)

    #
    # Extract strain
    #

    e_phi = a_eu_hoop @ u_r  # [dimensionless] hoop strain is easy
    e_r = a_eu_radial @ u_r  # [dimensionless] radial strain = d/dr u_r
    e = a_eu @ u_r  # [dimensionless] full strain vector
    assert np.all(e[:nr] == e_r)
    assert np.all(e[nr:] == e_phi)

    # Check method using combined operator
    a_be_hoop = a_eu_hoop @ a_ub
    e_phi_combined = a_be_hoop @ rhs
    assert np.allclose(e_phi, e_phi_combined, rtol=1e-6)

    #
    # Extract stress
    #

    s = a_se @ e
    s_r, s_phi = s[:nr], s[nr:]

    #
    # Compare to handcalc
    #
    rgrid_wp = rgrid[wpinds]
    s_r_ideal, s_hoop_ideal = s_long_solenoid(rgrid_wp, r0, rwp, j, 27.0, 0.0, poisson_ratio)

    # Error in R-stress is slightly difficult to evaluate near the ends,
    # where it goes to zero, but the absolute error is very small,
    # and relative error is very small near the maximum value
    atol = np.max(np.abs(s_r_ideal)) * 1e-3
    assert np.allclose(s_r, s_r_ideal, rtol=1e-2, atol=atol)
    # Near the maximum, we can use a tighter tolerance
    i_srmax = np.argmax(np.abs(s_r_ideal))
    assert s_r[i_srmax] == approx(s_r_ideal[i_srmax], rel=1e-3)

    # Hoop stress is the most important, and maintains a very tight match
    assert np.allclose(s_phi, s_hoop_ideal, rtol=1e-3)


@pytest.mark.parametrize("pi", [1e5, 1e6])
@pytest.mark.parametrize("po", [1e5, 1e6])
@pytest.mark.parametrize("r0", [0.5, 0.7])
@pytest.mark.parametrize("r1", [1.0, 1.8])
def test_solenoid_against_thick_wall_cylinder(pi, po, r0, r1):
    #
    # Grid
    #

    # Material properties, roughly SS316
    poisson_ratio = 0.27  # [dimensionless]
    elasticity_modulus = 200 * 1e9  # [Pa]

    # Settings
    dx_reqd = 0.001  # [m] target resolution

    nr = int(np.ceil((r1 - r0) / dx_reqd)) + 1
    rgrid = np.linspace(r0, r1, nr)
    nudge = 1e-6  # Pad with points just outside to avoid nulling any real current density
    rgrid = np.array([r0 - nudge] + rgrid.tolist() + [r1 + nudge])

    r0, r1 = rgrid[0], rgrid[-1]  # Account for nudge in later calcs
    nr = nr + 2

    # No current density
    j = np.zeros_like(rgrid)
    bz = np.zeros_like(rgrid)

    #
    # Build operators and rhs
    #

    c = solenoid_1d_structural_factor(elasticity_modulus, poisson_ratio)
    rhs = solenoid_1d_structural_rhs(c, j, bz, pi, po)

    solenoid_stress = SolenoidStress1D(
        rgrid=rgrid, elasticity_modulus=elasticity_modulus, poisson_ratio=poisson_ratio, direct_inverse=True
    )
    operators = solenoid_stress.operators

    # a_bu = operators.a_bu
    a_ub = operators.a_ub
    a_eu = operators.a_eu
    a_eu_radial = operators.a_eu_radial
    a_eu_hoop = operators.a_eu_hoop
    a_se = operators.a_se

    #
    # Solve for displacement
    #

    print("Solving for displacement")

    start_time = monotonic_ns()
    # Direct inverse method
    # About 30us to do the matrix multiplication
    # np.squeeze and np.ravel fail here for some reason, but we need u_r to be flat for plotting
    u_r = np.squeeze(np.asarray(a_ub @ rhs))
    end_time = monotonic_ns()
    print(f"    Solved using pre-calculated direct inverse in {(end_time - start_time) / 1e9:.6f} s")

    start_time = monotonic_ns()
    # The better solve method that isn't an operator
    # About 100us solve
    a_ub_solver = solenoid_stress.displacement_solver
    u_r_direct_solver = a_ub_solver(rhs)

    end_time = monotonic_ns()

    assert np.allclose(u_r, u_r_direct_solver, rtol=1e-6, atol=1e-5)

    #
    # Extract strain
    #

    e_phi = a_eu_hoop @ u_r  # [dimensionless] hoop strain is easy
    e_r = a_eu_radial @ u_r  # [dimensionless] radial strain = d/dr u_r
    e = a_eu @ u_r  # [dimensionless] full strain vector
    assert np.all(e[:nr] == e_r)
    assert np.all(e[nr:] == e_phi)

    # Check method using combined operator
    a_be_hoop = a_eu_hoop @ a_ub
    e_phi_combined = a_be_hoop @ rhs
    assert np.allclose(e_phi, e_phi_combined, rtol=1e-6)

    #
    # Extract stress
    #

    s = a_se @ e
    s_r, s_phi = s[:nr], s[nr:]

    #
    # Compare to handcalc
    #
    s_r_ideal, s_hoop_ideal = (
        s_radial_thick_wall_cylinder(rgrid, r0, r1, pi, po),
        s_hoop_thick_wall_cylinder(rgrid, r0, r1, pi, po),
    )

    # Error in R-stress is slightly difficult to evaluate near the ends,
    # where it goes to zero, but the absolute error is very small,
    # and relative error is very small near the maximum value
    assert np.allclose(s_r, s_r_ideal, rtol=1e-3)
    # Near the maximum, we can use a tighter tolerance
    i_srmax = np.argmax(np.abs(s_r_ideal))
    assert s_r[i_srmax] == approx(s_r_ideal[i_srmax], rel=1e-4)

    # Hoop stress is the most important, and maintains a very tight match
    assert np.allclose(s_phi, s_hoop_ideal, rtol=1e-4)


def test_write_mat_and_json():
    r0, r1 = 0.1, 0.2
    dx_reqd = 0.01  # [m] target resolution
    nr = int(np.ceil((r1 - r0) / dx_reqd)) + 1
    rgrid = np.linspace(r0, r1, nr)

    # Material properties, roughly SS316
    poisson_ratio = 0.27  # [dimensionless]
    elasticity_modulus = 200 * 1e9  # [Pa]

    here = Path(__file__).parent

    solenoid_stress = SolenoidStress1D(
        rgrid=rgrid, elasticity_modulus=elasticity_modulus, poisson_ratio=poisson_ratio, direct_inverse=False
    )
    operators = solenoid_stress.operators

    fpath = None
    try:
        fpath = operators.write_mat(here)
    finally:
        fpath = fpath or here / "stress_operators.mat"
        if os.path.isfile(fpath):
            os.remove(fpath)

    # Roundtrip json ser/de
    json_string = solenoid_stress.model_dump_json()
    reloaded = SolenoidStress1D.model_validate_json(json_string)
    assert reloaded == solenoid_stress
