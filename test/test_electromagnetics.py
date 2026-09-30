"""Tests of standalone electromagnetics calcs"""

import numpy as np
import scipy.sparse as sparse
from pytest import approx, mark, raises

import cfsem

from test import test_funcs as _test


def _circular_loop_xyz(major_radius: float, ndiscr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.linspace(0.0, 2.0 * np.pi, ndiscr, endpoint=True)
    x = major_radius * np.cos(phi)
    y = major_radius * np.sin(phi)
    z = np.zeros_like(x)
    return x, y, z


def _linear_filament_self_inductance_from_vector_potential(
    major_radius: float,
    minor_radius: float,
    ndiscr: int,
    par: bool,
) -> float:
    x, y, z = _circular_loop_xyz(major_radius, ndiscr)
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dz = z[1:] - z[:-1]
    xyzfil = (x[:-1], y[:-1], z[:-1])
    dlxyzfil = (dx, dy, dz)
    ifil = np.ones_like(dx)
    xyzp = (x[:-1] + 0.5 * dx, y[:-1] + 0.5 * dy, z[:-1] + 0.5 * dz)
    ax, ay, az = cfsem.vector_potential_linear_filament(
        xyzp=xyzp,
        xyzfil=xyzfil,
        dlxyzfil=dlxyzfil,
        ifil=ifil,
        wire_radius=minor_radius,
        par=par,
    )
    return float(np.sum(ax * dx + ay * dy + az * dz))


@mark.parametrize("r", [0.775, np.pi])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("ndiscr", [1000, 2000])
@mark.parametrize("par", [True, False])
def test_body_force_density(r, z, ndiscr, par):
    """Spot check bindings; more complete tests are run in Rust"""
    xp = np.linspace(0.1, 0.8, 5)
    yp = np.zeros(5)
    zp = np.linspace(-1.0, 1.0, 5)
    xmesh, ymesh, zmesh = np.meshgrid(xp, yp, zp, indexing="ij")
    xmesh = xmesh.flatten()
    ymesh = ymesh.flatten()
    zmesh = zmesh.flatten()
    obs = (xmesh, ymesh, zmesh)

    rng = np.random.default_rng(1234098)
    j = [rng.uniform(-1e6, 1e6, len(xmesh)) for _ in range(3)]

    fil, dlxyzfil = _test._filament_loop(r, z, ndiscr)
    xyzfil = (fil[0][:-1], fil[1][:-1], fil[2][:-1])
    ifil = np.ones_like(xyzfil[0])

    jxbx, jxby, jxbz = cfsem.body_force_density_circular_filament_cartesian(
        [1.0],
        [r],
        (np.zeros_like([z]), np.zeros_like([z]), [z]),
        (np.zeros_like([z]), np.zeros_like([z]), np.ones_like([z])),
        obs,
        j,
        par,
    )
    wire_radius = np.zeros_like(ifil)
    jxbx1, jxby1, jxbz1 = cfsem.body_force_density_linear_filament(
        xyzfil, dlxyzfil, ifil, obs, j, wire_radius, par=par
    )
    jxbx2, jxby2, jxbz2 = cfsem.body_force_density_linear_filament(
        xyzfil, dlxyzfil, ifil, obs, j, 0.0, par=par
    )

    assert np.allclose(jxbx, jxbx1, rtol=1e-2, atol=1e-9)
    assert np.allclose(jxby, jxby1, rtol=1e-2, atol=1e-9)
    assert np.allclose(jxbz, jxbz1, rtol=1e-2, atol=1e-9)
    assert np.allclose(jxbx1, jxbx2, rtol=1e-12, atol=1e-12)
    assert np.allclose(jxby1, jxby2, rtol=1e-12, atol=1e-12)
    assert np.allclose(jxbz1, jxbz2, rtol=1e-12, atol=1e-12)


@mark.parametrize("r", [0.775 * 2, np.pi])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("par", [True, False])
def test_flux_density_dipole(r, z, par):
    """Spot check bindings; more complete tests are run in Rust"""
    r = r / 300.0  # Make a very small filament
    xp = np.linspace(0.1, 0.8, 5)
    yp = np.zeros(5)
    zp = np.linspace(-1.0, 1.0, 5)
    area = np.pi * r**2  # m^2
    xmesh, ymesh, zmesh = np.meshgrid(xp, yp, zp, indexing="ij")
    xmesh = xmesh.flatten()
    ymesh = ymesh.flatten()
    zmesh = zmesh.flatten()

    bx, by, bz = cfsem.flux_density_circular_filament_cartesian(
        [1.0],
        [r],
        (np.zeros_like([z]), np.zeros_like([z]), [z]),
        (np.zeros_like([z]), np.zeros_like([z]), np.ones_like([z])),
        (xmesh, ymesh, zmesh),
        par,
    )

    bxd, byd, bzd = cfsem.flux_density_dipole(
        loc=([0.0], [0.0], [z]),
        moment=([0.0], [0.0], [area]),
        xyzp=(xmesh, ymesh, zmesh),
        par=par,
    )

    assert np.allclose(bx, bxd, rtol=5e-2, atol=1e-12)
    assert np.allclose(by, byd, rtol=5e-2, atol=1e-12)
    assert np.allclose(bz, bzd, rtol=5e-2, atol=1e-12)

    # Make sure we're not just comparing numbers that are too small to examine properly
    assert not np.allclose(by, bzd, rtol=5e-2, atol=1e-12)


@mark.parametrize("par", [True, False])
def test_vector_potential_dipole(par):
    """Check that B=curl(A)"""
    xgrid = np.linspace(-2.0, 2.0, 7)
    ygrid = np.linspace(-1.0, 3.0, 9)
    zgrid = np.linspace(-3.0, 1.0, 11)

    xmesh, ymesh, zmesh = np.meshgrid(xgrid, ygrid, zgrid, indexing="ij")
    obs = (xmesh.flatten(), ymesh.flatten(), zmesh.flatten())
    points = np.column_stack(obs)

    rng = np.random.RandomState(1235897)
    n = 5
    loc = (rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n))
    moment = (rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n))

    bx, by, bz = cfsem.flux_density_dipole(loc, moment, obs, par=par)

    bx = np.asarray(bx)
    by = np.asarray(by)
    bz = np.asarray(bz)

    def vector_potential_at(xp, yp, zp):
        axp, ayp, azp = cfsem.vector_potential_dipole(loc, moment, ([xp], [yp], [zp]), par=par)
        return float(axp[0]), float(ayp[0]), float(azp[0])

    eps = 1e-6
    curl = np.zeros_like(points)

    for i, (xp, yp, zp) in enumerate(points):
        da = np.zeros((3, 3))

        ap = np.array(vector_potential_at(xp + eps, yp, zp))
        am = np.array(vector_potential_at(xp - eps, yp, zp))
        da[0, :] = (ap - am) / (2.0 * eps)

        ap = np.array(vector_potential_at(xp, yp + eps, zp))
        am = np.array(vector_potential_at(xp, yp - eps, zp))
        da[1, :] = (ap - am) / (2.0 * eps)

        ap = np.array(vector_potential_at(xp, yp, zp + eps))
        am = np.array(vector_potential_at(xp, yp, zp - eps))
        da[2, :] = (ap - am) / (2.0 * eps)

        curl[i, 0] = da[1, 2] - da[2, 1]
        curl[i, 1] = da[2, 0] - da[0, 2]
        curl[i, 2] = da[0, 1] - da[1, 0]

    # Make sure we have enough to compare
    magnitudes = np.linalg.norm(np.column_stack((bx, by, bz)), axis=1)
    assert np.all(magnitudes > 1e-10)

    # Check curl
    assert np.allclose(curl[:, 0], bx, rtol=1e-3, atol=1e-14)
    assert np.allclose(curl[:, 1], by, rtol=1e-3, atol=1e-14)
    assert np.allclose(curl[:, 2], bz, rtol=1e-3, atol=1e-14)

    # Make sure the test _wouldn't_ pass if values were swapped
    assert not np.allclose(curl[:, 0], by, rtol=1e-3, atol=1e-14)


@mark.parametrize("r", [0.775, np.pi])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("ndiscr", [200, 400])
@mark.parametrize("par", [True, False])
def test_mutual_inductance_circular_to_linear(r, z, ndiscr, par):
    """Spot check bindings; more complete tests are run in Rust"""
    r1, z1 = (r + 0.1, abs(z) ** 0.5)
    fil, _dl = _test._filament_loop(r, z, ndiscr=ndiscr)
    fil1, dl1 = _test._filament_loop(r1, z1, ndiscr=ndiscr)
    (x1, y1, z1_points) = fil1

    m_linear = cfsem.mutual_inductance_piecewise_linear_filaments(fil, fil1)
    m_circular = cfsem.flux_circular_filament([1.0], [r], [z], [r1], [z1], par)
    m_circular_to_linear = cfsem.mutual_inductance_circular_to_linear(
        [r], [z], [1.0], (x1[:-1], y1[:-1], z1_points[:-1]), dl1, par
    )

    # Linear discretization really is not very good unless we use a number of discretizations that is
    # not reasonable for testing, so the tolerances are pretty loose
    assert m_circular_to_linear == approx(m_circular, rel=0.2)
    assert m_circular_to_linear == approx(m_linear, rel=0.2)


@mark.parametrize("r", [0.775, np.pi])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("par", [True, False])
def test_flux_density_circular_filament_cartesian(r, z, par):
    """Spot check bindings; more complete tests are run in Rust"""
    xp = np.linspace(0.1, 0.8, 5)
    yp = np.zeros(5)
    zp = np.linspace(-1.0, 1.0, 5)
    xmesh, ymesh, zmesh = np.meshgrid(xp, yp, zp, indexing="ij")
    xmesh = xmesh.flatten()
    ymesh = ymesh.flatten()
    zmesh = zmesh.flatten()

    bx, by, bz = cfsem.flux_density_circular_filament_cartesian(
        [1.0],
        [r],
        (np.zeros_like([z]), np.zeros_like([z]), [z]),
        (np.zeros_like([z]), np.zeros_like([z]), np.ones_like([z])),
        (xmesh, ymesh, zmesh),
        par,
    )
    br, bz_circ = cfsem.flux_density_circular_filament([1.0], [r], [z], xmesh, zmesh, par)

    assert np.allclose(bx, br, rtol=1e-6, atol=1e-10)
    assert np.allclose(bz, bz_circ, rtol=1e-6, atol=1e-10)
    assert np.allclose(by, np.zeros_like(by), atol=1e-10)


@mark.parametrize("r", [7.7, np.pi])  # Needs to be large for Lyle with very small width
@mark.parametrize("z", [0.0, np.e / 2])
@mark.parametrize("h_over_r", [5e-2, 1e-2, 1e-3])
@mark.parametrize("n", [int(1e3), int(1e4)])
def test_self_inductance_piecewise_linear_filaments(r, z, h_over_r, n):
    # Test self inductance via the finite-radius A·dl line integral
    # against Lyle's calc for finite-thickness coils.
    # [m] can't be infinitesimally thin for Lyle's calc, but can be very thin compared to height and radius
    w = 0.001
    h = h_over_r * r  # [m]

    nt = 13  # number of turns

    thetas = np.linspace(0.0, 2.0 * np.pi * nt, n, endpoint=True)

    x1 = np.cos(thetas) * r
    y1 = np.sin(thetas) * r
    z1 = np.linspace(z - h / 2, z + h / 2, n)

    xyz1 = np.vstack((x1, y1, z1))

    self_inductance_piecewise_linear = cfsem.self_inductance_piecewise_linear_filaments(
        xyz1,
        wire_radius=0.5 * w,
    )  # [H]

    self_inductance_lyle6 = cfsem.self_inductance_lyle6(r, w, h, nt)  # [H]

    rel = 5e-2
    assert self_inductance_piecewise_linear == approx(self_inductance_lyle6, rel=rel)


@mark.parametrize("r1", [0.5, np.pi])
@mark.parametrize("r2", [0.1, np.pi / 10.0])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("ndiscr", [100, 200])
@mark.parametrize("par", [True, False])
def test_mutual_inductance_piecewise_linear_filaments(r1, r2, z, ndiscr, par):
    # Test against calc for mutual inductance of circular filaments
    rzn1 = np.array([[r1], [z], [1.0]])
    rzn2 = np.array([[r2], [-z / np.e], [1.0]])

    m_circular = cfsem.mutual_inductance_of_circular_filaments(rzn1, rzn2, par)

    thetas = np.linspace(0.0, 2.0 * np.pi, ndiscr, endpoint=True)

    x1 = np.cos(thetas) * rzn1[0]
    y1 = np.sin(thetas) * rzn1[0]
    z1 = np.ones_like(thetas) * rzn1[1]

    x2 = np.cos(thetas) * rzn2[0]
    y2 = np.sin(thetas) * rzn2[0]
    z2 = np.ones_like(thetas) * rzn2[1]

    xyz1 = np.vstack((x1, y1, z1))
    xyz2 = np.vstack((x2, y2, z2))

    m_piecewise_linear = cfsem.mutual_inductance_piecewise_linear_filaments(xyz1, xyz2)

    assert np.allclose([m_circular], [m_piecewise_linear], rtol=1e-4)


@mark.parametrize("r", [0.1, np.pi / 10.0])
@mark.parametrize("n_filaments", [int(1e4), int(2e4)])
@mark.parametrize("par", [True, False])
def test_biot_savart_against_flux_density_ideal_solenoid(r, n_filaments, par):
    # Check Biot-Savart calc against ideal solenoid calc
    length = 20.0 * r  # [m]
    num_turns = 7  # [#]
    current = np.e  # [A]

    # Ideal calc
    b_ideal = cfsem.flux_density_ideal_solenoid(current, num_turns, length)  # [T]

    # Biot-Savart calc should produce the same magnitude
    #   Build a spiral coil
    x1 = np.linspace(-length / 2, length / 2, n_filaments + 1)
    y1 = r * np.cos(num_turns * 2.0 * np.pi * x1 / length)
    z1 = r * np.sin(num_turns * 2.0 * np.pi * x1 / length)
    xyz1 = np.stack((x1, y1, z1), 1).T
    dl1 = xyz1[:, 1:] - xyz1[:, 0:-1]
    dlxyzfil = (
        np.ascontiguousarray(dl1[0, :]),
        np.ascontiguousarray(dl1[1, :]),
        np.ascontiguousarray(dl1[2, :]),
    )
    ifil = current * np.ones(n_filaments)
    xyzfil = (x1[:-1], y1[:-1], z1[:-1])
    #   Get B-field at the origin
    zero = np.array([0.0])
    wire_radius = np.zeros_like(ifil)
    bx, _by, _bz = cfsem.flux_density_linear_filament(
        xyzp=(zero, zero, zero),
        xyzfil=xyzfil,
        dlxyzfil=dlxyzfil,
        ifil=ifil,
        wire_radius=wire_radius,
        par=par,
    )
    b_bs = bx[0]  # [T] First and only element on the axis of the solenoid

    assert b_bs == approx(b_ideal, rel=1e-2)


@mark.parametrize("r", [-np.pi, -0.775, 0.775, np.pi])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("n_filaments", [int(1e4), int(2e4)])
@mark.parametrize("par", [True, False])
def test_biot_savart_against_flux_density_circular_filament(r, z, n_filaments, par):
    # Note we are mapping between (x, y, z) and (r, phi, z) coordinates here

    # Biot-Savart filaments in cartesian coords
    phi = np.linspace(0.0, 2.0 * np.pi, n_filaments)
    xfils = r * np.cos(phi)
    yfils = r * np.sin(phi)
    zfils = np.ones_like(xfils) * z

    # Observation grid
    rs = np.linspace(-abs(r) + 0.1, abs(r) - 0.1, 20)
    zs = np.linspace(-1.0, 1.0, 10)

    R, Z = np.meshgrid(rs, zs, indexing="ij")
    rprime = R.flatten()
    zprime = Z.flatten()

    # Circular filament calc
    # [T]
    Br_circular, Bz_circular = cfsem.flux_density_circular_filament(
        np.ones(1), np.array([r]), np.array([z]), rprime, zprime, par
    )

    # Biot-Savart calc
    xyzp = (rprime, np.zeros_like(zprime), zprime)
    xyzfil = (xfils[1:], yfils[1:], zfils[1:])
    dlxyzfil = (xfils[1:] - xfils[:-1], yfils[1:] - yfils[:-1], zfils[1:] - zfils[:-1])
    ifil = np.ones_like(xfils[1:])
    wire_radius = np.zeros_like(ifil)
    Br_bs, By_bs, Bz_bs = cfsem.flux_density_linear_filament(
        xyzp, xyzfil, dlxyzfil, ifil, wire_radius, par
    )  # [T]

    assert np.allclose(Br_circular, Br_bs, rtol=1e-6, atol=1e-7)  # Should match circular calc
    assert np.allclose(Bz_circular, Bz_bs, rtol=1e-6, atol=1e-7)  # ...
    assert np.allclose(By_bs, np.zeros_like(By_bs), atol=1e-7)  # Should sum to zero everywhere


@mark.parametrize("r", [0.775, np.pi])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("par", [True, False])
def test_flux_circular_filament_against_mutual_inductance_of_cylindrical_coils(r, z, par):
    # Two single-turn coils with irrelevant cross-section,
    # each discretized into a single filament
    rc1 = r  # Coil center radii
    rc2 = 10.0 * r  # Large enough to be much larger than 1
    rzn1 = cfsem.filament_coil(rc1, z, 0.05, 0.05, 1.5, 2, 2)
    rzn2 = cfsem.filament_coil(rc2, -z, 0.05, 0.05, 1.5, 2, 2)

    # Unpack and copy to make contiguous in memory
    r1, z1, n1 = rzn1.T
    r2, z2, n2 = rzn2.T
    r1, z1, n1, r2, z2, n2 = [x.copy() for x in [r1, z1, n1, r2, z2, n2]]

    # Calculate mutual inductance between these two filaments
    f1 = np.array((r1, z1, n1))
    f2 = np.array((r2, z2, n2))
    m_filaments = cfsem.mutual_inductance_of_cylindrical_coils(f1, f2, par)

    # Calculate mutual inductance via python test calc
    # and test the mutual inductance of coils calc.
    # This also tests the mutual_inductance_of_circular_filaments calc
    # against the python version at the same time.
    m_filaments_test = _test._mutual_inductance_of_cylindrical_coils(f1.T, f2.T)
    assert abs(1 - m_filaments / m_filaments_test) < 1e-6

    # Do flux calcs
    psi_2to1 = np.sum(n1 * cfsem.flux_circular_filament(n2, r2, z2, r1, z1, par))
    psi_1to2 = np.sum(n2 * cfsem.flux_circular_filament(n1, r1, z1, r2, z2, par))

    # Because the integrated poloidal flux at a given location is the same as mutual inductance,
    # we should get the same number using our mutual inductance calc
    current = 1.0  # 1A reference current just for clarity
    m_from_psi = psi_2to1 / current
    assert abs(1 - m_from_psi / m_filaments) < 1e-6

    # Because mutual inductance is reflexive, reversing the direction of the check should give the same result
    # so we can check to make sure the psi calc gives the same result in both directions
    assert psi_2to1 == approx(psi_1to2, rel=1e-6)


@mark.parametrize("r", [0.775, np.pi])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("par", [True, False])
def test_flux_density_circular_filament_against_flux_circular_filament(r, z, par):
    rzn1 = cfsem.filament_coil(r, z, 0.05, 0.05, 1.0, 4, 4)
    rfil, zfil, _ = rzn1.T
    ifil = np.ones_like(rfil)

    rs = np.linspace(0.01, min(rfil) - 0.1, 10)
    zs = np.linspace(-1.0, 1.0, 10)

    R, Z = np.meshgrid(rs, zs, indexing="ij")
    rprime = R.flatten()
    zprime = Z.flatten()

    Br, Bz = cfsem.flux_density_circular_filament(ifil, rfil, zfil, rprime, zprime, par)  # [T]

    # We can also get B from the derivative of the flux function (Wesson eqn 3.2.2),
    # so we'll use that to check that we get the same result.
    # Wesson uses flux per radian (as opposed to our total flux), so we have to adjust out a factor
    # of 2*pi in the conversion from flux to B-field. This makes sense because we are converting
    # between the _integral_ of B (psi) and B itself, so we should see a factor related to the
    # space we integrated over to get psi.

    dr = 1e-4
    dz = 1e-4
    psi = cfsem.flux_circular_filament(ifil, rfil, zfil, rprime, zprime, par)
    dpsidz = (cfsem.flux_circular_filament(ifil, rfil, zfil, rprime, zprime + dz, par) - psi) / dz
    dpsidr = (cfsem.flux_circular_filament(ifil, rfil, zfil, rprime + dr, zprime, par) - psi) / dr

    Br_from_psi = -dpsidz / rprime / (2.0 * np.pi)  # [T]
    Bz_from_psi = dpsidr / rprime / (2.0 * np.pi)  # [T]

    assert np.allclose(Br, Br_from_psi, rtol=1e-2)
    assert np.allclose(Bz, Bz_from_psi, rtol=1e-2)


@mark.parametrize("r", [np.e / 100, 0.775, np.pi])
@mark.parametrize("par", [True, False])
def test_flux_density_circular_filament_against_ideal_solenoid(r, par):
    # We can also check against the ideal solenoid calc to make sure we don't have a systematic
    # offset or scaling error

    length = 20.0 * r  # [m]
    rzn1 = cfsem.filament_coil(r, 0.0, 0.05, length, 1.0, 1, 40)
    rfil, zfil, _ = rzn1.T
    ifil = np.ones_like(rfil)

    b_ideal = cfsem.flux_density_ideal_solenoid(
        current=1.0, num_turns=ifil.size, length=length
    )  # [T] ideal solenoid Bz at origin
    _, bz_origin = cfsem.flux_density_circular_filament(ifil, rfil, zfil, np.zeros(1), np.zeros(1), par)

    assert np.allclose(np.array([b_ideal]), bz_origin, rtol=1e-2)


@mark.parametrize("r", [np.e / 100, 0.775, np.pi])
@mark.parametrize("par", [True, False])
def test_flux_density_circular_filament_against_ideal_loop(r, par):
    # We can also check against an ideal current loop calc
    # http://hyperphysics.phy-astr.gsu.edu/hbase/magnetic/curloo.html

    current = 1.0  # [A]
    ifil = np.array([current])
    rfil = np.array([r])
    zfil = np.array([0.0])

    b_ideal = cfsem.MU_0 * current / (2.0 * r)  # [T] ideal loop Bz at origin
    _, bz_origin = cfsem.flux_density_circular_filament(ifil, rfil, zfil, np.zeros(1), np.zeros(1), par)

    assert np.allclose(np.array([b_ideal]), bz_origin, rtol=1e-6)


@mark.parametrize("par", [True, False])
@mark.parametrize(
    "name", ["flux_circular_filament", "vector_potential_circular_filament", "flux_density_circular_filament"]
)
def test_circular_optional_wire_radius(name, par):
    import cfsem.cfsem as raw

    calc, raw_calc = getattr(cfsem, name), getattr(raw, name)
    current, rfil, zfil, robs, zobs = (
        np.asarray(x, dtype=float) for x in ([2, -3], [0.5, 1.0], [0, 0.3], [0.6, 0.7], [0.2, 0.4])
    )
    args = (current, rfil, zfil, robs, zobs)
    legacy = calc(*args, par)  # Sixth positional argument still means `par`.
    for radius in [None, np.zeros(2), [0.0, 0.0]]:
        np.testing.assert_array_equal(calc(*args, par, wire_radius=radius), legacy)
    np.testing.assert_array_equal(raw_calc(*args, par), legacy)
    np.testing.assert_array_equal(raw_calc(*args, par, wire_radius=None), legacy)
    np.testing.assert_array_equal(raw_calc(*args, par, wire_radius=np.zeros(2)), legacy)


@mark.parametrize("par", [True, False])
@mark.parametrize("nobs", [0, 1, 3])
@mark.parametrize("sf", [-1.0, 1.0])
@mark.parametrize("so", [-1.0, 1.0])
def test_circular_potential_and_flux_per_source_wire_radius(par, nobs, sf, so):
    from scipy.special import ellipe, ellipk

    import cfsem.cfsem as raw

    current = np.array([1.0, -2.0, 0.5])
    rfil = np.array([1.0, 1.0, 0.4])
    zfil = np.array([0.0, 0.0, -0.3])
    radii = np.array([0.01, 99.0, 0.02, 99.0, 0.0, 99.0])[::2]
    robs, zobs = np.ones(nobs), np.zeros(nobs)
    args = (current, sf * rfil, zfil, so * robs, zobs)
    # Finite loops at their centerline, plus a remote ideal loop.
    finite = np.sum(cfsem.MU_0 * current[:2] / (4 * np.pi) * (2 * np.log(8 / radii[:2]) - 3))
    q = (rfil[2] + robs) ** 2 + (zfil[2] - zobs) ** 2
    m = 4 * rfil[2] * robs / q
    thin = cfsem.MU_0 / np.pi * current[2] * rfil[2] / np.sqrt(q) * ((2 - m) * ellipk(m) - 2 * ellipe(m)) / m
    a_phi = cfsem.vector_potential_circular_filament(*args, par, wire_radius=radii)
    flux = cfsem.flux_circular_filament(*args, par, wire_radius=radii)
    assert np.all(np.isfinite(a_phi))
    np.testing.assert_allclose(a_phi, so * (finite + thin), rtol=2e-8)
    np.testing.assert_array_equal(flux, 2 * np.pi * so * robs * a_phi)
    np.testing.assert_array_equal(
        raw.vector_potential_circular_filament(*args, par, np.ascontiguousarray(radii)), a_phi
    )
    np.testing.assert_array_equal(raw.flux_circular_filament(*args, par, np.ascontiguousarray(radii)), flux)


@mark.parametrize("par", [True, False])
@mark.parametrize(
    "name", ["flux_circular_filament", "vector_potential_circular_filament", "flux_density_circular_filament"]
)
@mark.parametrize("nobs", [0, 2])
@mark.parametrize("radii", [[], [0.01], [0.01, 0.02, 0.03]])
def test_circular_wire_radius_length_error(name, par, nobs, radii):
    import cfsem.cfsem as raw

    args = tuple(
        np.asarray(x, dtype=float)
        for x in ([1.0, 2.0], [1.0, 1.1], [0.0, 0.0], np.zeros(nobs), np.zeros(nobs))
    )
    for calc in [getattr(cfsem, name), getattr(raw, name)]:
        with raises(ValueError, match="Length mismatch"):
            calc(*args, par, wire_radius=np.asarray(radii))


@mark.parametrize("par", [True, False])
@mark.parametrize("force", [False, True])
def test_circular_cartesian_optional_wire_radius(par, force):
    import cfsem.cfsem as raw

    name = (
        "body_force_density_circular_filament_cartesian"
        if force
        else "flux_density_circular_filament_cartesian"
    )
    current, rfil, zfil = np.array([2.0, -3.0]), np.array([0.5, 1.0]), np.array([0.0, 0.3])
    obs = (np.array([0.0, 0.3, -0.4]), np.array([0.0, -0.4, 0.3]), np.array([0.2, 0.4, 0.5]))
    j = (np.array([1.0, -2.0, 3.0]), np.array([0.5, 1.0, -0.3]), np.array([-2.0, 0.0, 0.7]))
    source = (
        current,
        rfil,
        (np.zeros_like(zfil), np.zeros_like(zfil), zfil),
        (np.zeros_like(zfil), np.zeros_like(zfil), np.ones_like(zfil)),
    )
    args = (*source, obs, j) if force else (*source, obs)
    calc, raw_calc = getattr(cfsem, name), getattr(raw, name)
    legacy = calc(*args, par)
    for radii in [None, np.zeros(2), [0.0, 0.0]]:
        np.testing.assert_array_equal(calc(*args, par, wire_radius=radii), legacy)
    np.testing.assert_array_equal(raw_calc(*args, par), legacy)
    np.testing.assert_array_equal(raw_calc(*args, par, wire_radius=None), legacy)
    np.testing.assert_array_equal(raw_calc(*args, par, wire_radius=np.zeros(2)), legacy)


@mark.parametrize("par", [True, False])
@mark.parametrize("force", [False, True])
@mark.parametrize("nobs", [0, 2])
@mark.parametrize("radii", [[], [0.01], [0.01, 0.02, 0.03]])
def test_circular_cartesian_wire_radius_length_error(par, force, nobs, radii):
    import cfsem.cfsem as raw

    name = (
        "body_force_density_circular_filament_cartesian"
        if force
        else "flux_density_circular_filament_cartesian"
    )
    obs = (np.ones(nobs), np.zeros(nobs), np.zeros(nobs))
    args = (
        np.ones(2),
        np.ones(2),
        (np.zeros(2), np.zeros(2), np.zeros(2)),
        (np.zeros(2), np.zeros(2), np.ones(2)),
        obs,
    )
    if force:
        args += (obs,)
    for calc in [getattr(cfsem, name), getattr(raw, name)]:
        with raises(ValueError, match="Length mismatch"):
            calc(*args, par, wire_radius=np.asarray(radii))


@mark.parametrize("par", [True, False])
def test_circular_cartesian_independent_source_poses(par):
    rotations = np.array(
        [
            np.eye(3),
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        ],
        dtype=float,
    )
    rfil, current = np.array([1.0, 0.8, 0.5]), np.array([1.0, -2.0, 0.5])
    radii = np.array([0.01, 0.02, 0.0])
    target = np.array([1.0, 0.0, 0.0])
    centers = target - np.array([1.0, 0.8, 0.6])[:, None] * rotations[:, :, 0]
    normals = rotations[:, :, 2] * np.array([1.0, 2.0, 3.0])[:, None]
    obs = target + np.array([[0, 0, 0], [0.003, 0.002, 0.004], [-0.002, -0.003, -0.002]])
    jobs = np.array([[1.0, -2.0, 0.3], [0.7, 0.0, -0.4], [-0.3, 0.5, 0.0]])
    expected = np.zeros_like(obs)
    for i in range(3):
        local = (obs - centers[i]) @ rotations[i]
        r = np.hypot(local[:, 0], local[:, 1])
        br, bz = cfsem.flux_density_circular_filament(
            current[i : i + 1],
            rfil[i : i + 1],
            [0.0],
            r,
            local[:, 2],
            par,
            wire_radius=radii[i : i + 1],
        )
        local_b = np.column_stack((br * local[:, 0] / r, br * local[:, 1] / r, bz))
        expected += local_b @ rotations[i].T
    # Component arrays are deliberately strided views. Normals differ per source.
    args = (current, rfil, tuple(centers.T), tuple(normals.T), tuple(obs.T))
    actual = cfsem.flux_density_circular_filament_cartesian(*args, par, wire_radius=radii)
    force = cfsem.body_force_density_circular_filament_cartesian(*args, tuple(jobs.T), par, wire_radius=radii)
    np.testing.assert_allclose(np.array(actual).T, expected, rtol=2e-10, atol=1e-16)
    np.testing.assert_allclose(np.array(force).T, np.cross(jobs, expected), rtol=2e-10, atol=1e-16)
    # A common translation changes points and centers, never field/current vectors.
    shift = np.array([0.7, -0.4, 1.3])
    shifted = cfsem.flux_density_circular_filament_cartesian(
        current,
        rfil,
        tuple((centers + shift).T),
        tuple(normals.T),
        tuple((obs + shift).T),
        par,
        wire_radius=radii,
    )
    np.testing.assert_allclose(shifted, actual, rtol=2e-10, atol=1e-16)


@mark.parametrize("par", [True, False])
@mark.parametrize("force", [False, True])
@mark.parametrize("wire_radius", [0.0, 0.01])
@mark.parametrize(
    "bad, value",
    [("normal", 0.0)] + [(bad, value) for bad in ["loc", "normal", "obs"] for value in [np.nan, np.inf]],
)
def test_circular_cartesian_invalid_geometry(par, force, wire_radius, bad, value):
    nobs = 2
    name = (
        "body_force_density_circular_filament_cartesian"
        if force
        else "flux_density_circular_filament_cartesian"
    )
    loc = ([0.0], [0.0], [0.0])
    normal = ([0.0], [0.0], [1.0])
    obs = (np.zeros(nobs), np.zeros(nobs), np.ones(nobs))
    if bad == "loc":
        loc = ([0.0], [0.0], [value])
    elif bad == "normal":
        normal = ([0.0], [0.0], [value])
    else:
        obs = (np.zeros(nobs), np.zeros(nobs), np.full(nobs, value))
    args = ([1.0], [1.0], loc, normal, obs)
    if force:
        args += ((np.ones(nobs), np.zeros(nobs), np.zeros(nobs)),)
    result = np.asarray(getattr(cfsem, name)(*args, par, wire_radius=[wire_radius]))
    assert result.shape == (3, nobs)
    assert np.all(np.isnan(result))


@mark.parametrize("par", [True, False])
@mark.parametrize(
    "force,bad",
    [
        (force, bad)
        for force in [False, True]
        for bad in ["radius", "center", "normal", "observation", "jobs"]
        if force or bad != "jobs"
    ],
)
def test_circular_cartesian_pose_shapes(par, force, bad):
    name = (
        "body_force_density_circular_filament_cartesian"
        if force
        else "flux_density_circular_filament_cartesian"
    )
    rfil = [] if bad == "radius" else [1.0]
    center = ([], [0.0], [0.0]) if bad == "center" else ([0.0], [0.0], [0.0])
    normal = ([0.0], [], [1.0]) if bad == "normal" else ([0.0], [0.0], [1.0])
    obs = ([0.0], [0.0], []) if bad == "observation" else ([0.0], [0.0], [1.0])
    args = ([1.0], rfil, center, normal, obs)
    if force:
        args += (([1.0], [], [0.0]) if bad == "jobs" else ([1.0], [0.0], [0.0]),)
    with raises(ValueError, match="Length mismatch"):
        getattr(cfsem, name)(*args, par)


@mark.parametrize("par", [True, False])
@mark.parametrize("nobs", [1, 3])
def test_flux_density_circular_filament_per_source_wire_radius(par, nobs):
    from scipy.special import ellipe, ellipkm1

    current = np.array([1.0, -2.0, 0.5])
    rfil = np.array([1.0, 1.0, 0.4])
    zfil = np.array([0.0, 0.0, -0.3])
    # Exercise the wrapper's contiguous conversion and different source radii.
    radii = np.array([0.01, 99.0, 0.02, 99.0, 0.0, 99.0])[::2]
    robs, zobs = np.ones(nobs), np.zeros(nobs)
    actual = cfsem.flux_density_circular_filament(current, rfil, zfil, robs, zobs, par, wire_radius=radii)
    assert np.all(np.isfinite(actual))

    # Two finite-radius loops evaluated at their shared conductor centerline,
    # plus a separate thin filament. Reference: Hurwitz 17 and 19a, using SciPy
    # elliptic integrals independently of the Rust approximations.
    q = 4 * rfil[:2] ** 2 + radii[:2] ** 2 / np.sqrt(np.e)
    complement = radii[:2] ** 2 / (np.sqrt(np.e) * q)
    bz_reg = (
        cfsem.MU_0 * current[:2] / (2 * np.pi * np.sqrt(q)) * (ellipkm1(complement) - ellipe(1 - complement))
    )
    bz_finite = np.sum(bz_reg + 3 * cfsem.MU_0 * current[:2] / (16 * np.pi * rfil[:2]))
    expected_br, expected_bz = cfsem.flux_density_circular_filament(
        current[2:], rfil[2:], zfil[2:], robs, zobs, par
    )
    np.testing.assert_allclose(actual[0], expected_br, rtol=1e-12)
    np.testing.assert_allclose(actual[1], expected_bz + bz_finite, rtol=2e-8)


@mark.parametrize("par", [True, False])
@mark.parametrize("nobs", [1, 4])
@mark.parametrize("radius", [-0.01, np.nan])
def test_circular_negative_wire_radius(par, nobs, radius):
    # All combined circular kernels use wire-radius magnitude.
    current = np.ones(3)
    rfil = np.ones(3)
    zfil = np.zeros(3)
    wire_radius = np.array([0.0, radius, 0.01])
    robs = np.array([1.003, 0.0, 1.0, 1.02])[:nobs]
    zobs = np.array([0.004, 1.0, 0.0, 0.01])[:nobs]
    loc = (zfil, zfil, zfil)
    normal = (zfil, zfil, current)
    obs = (robs, np.zeros(nobs), zobs)
    jobs = (np.ones(nobs), np.ones(nobs), np.ones(nobs))
    for calc, args in (
        (cfsem.vector_potential_circular_filament, (current, rfil, zfil, robs, zobs)),
        (cfsem.flux_circular_filament, (current, rfil, zfil, robs, zobs)),
        (cfsem.flux_density_circular_filament, (current, rfil, zfil, robs, zobs)),
        (cfsem.flux_density_circular_filament_cartesian, (current, rfil, loc, normal, obs)),
        (cfsem.body_force_density_circular_filament_cartesian, (current, rfil, loc, normal, obs, jobs)),
    ):
        np.testing.assert_array_equal(calc(*args, par, wire_radius), calc(*args, par, np.abs(wire_radius)))


@mark.parametrize("par", [True, False])
@mark.parametrize("sf", [-1.0, 1.0])
@mark.parametrize("so", [-1.0, 1.0])
def test_circular_finite_radius_blend_and_far_field(par, sf, so):
    # Unit loop, b/a=.01: independent uniform-J cross-section quadrature.
    # Gauss-Legendre radial / midpoint angular orders 64/256 and 96/384 agree to 1e-9.
    r = so * np.array([1.0, 0.988, 1.02, 1.0, 0.982, 1.04, 1.0, 0.0])
    z = np.array([0.015, 0.016, 0.0, 0.025, 0.024, 0.0, 10.0, 10.0])
    expected = (
        cfsem.MU_0
        / (4 * np.pi)
        * np.array(
            [
                [133.272965286997, 80.3615478215542, 0.0, 79.9079909884819, 53.7056093252171, 0.0],
                [
                    5.39003743720978,
                    65.4098491497257,
                    -94.1467219114031,
                    4.80779121261902,
                    45.0095655376613,
                    -44.847395671673,
                ],
            ]
        )
    )
    expected[0] *= so
    actual = np.asarray(
        cfsem.flux_density_circular_filament([1.0], [sf], [0.0], r, z, par, wire_radius=[0.01])
    )
    error = np.linalg.norm(actual[:, :6] - expected, axis=0) / np.linalg.norm(expected, axis=0)
    assert np.all(error < 1e-3)
    thin = cfsem.flux_density_circular_filament([1.0], [sf], [0.0], r[-2:], z[-2:], par)
    np.testing.assert_allclose(actual[:, -2:], thin, rtol=1e-9, atol=1e-20)
    assert actual[0, -1] == 0.0

    # Same disk quadrature for A_phi; both quadrature orders agree to 1e-14.
    expected_a = (
        so
        * cfsem.MU_0
        / (4 * np.pi)
        * np.array(
            [
                8.558861359200655,
                8.019288525062006,
                7.926008834048046,
                7.53797552697675,
                7.220474354152276,
                6.510475334691865,
            ]
        )
    )
    a = cfsem.vector_potential_circular_filament([1.0], [sf], [0.0], r, z, par, wire_radius=[0.01])
    # Check the 0.1% target; finite-section/model truncation dominates fit error.
    np.testing.assert_allclose(a[:6], expected_a, rtol=1e-3)
    thin_a = cfsem.vector_potential_circular_filament([1.0], [sf], [0.0], r[-2:-1], z[-2:-1], par)
    np.testing.assert_allclose(a[-2:-1], thin_a, rtol=1e-8)
    assert a[-1] == 0.0
    flux = cfsem.flux_circular_filament([1.0], [sf], [0.0], r, z, par, wire_radius=[0.01])
    np.testing.assert_array_equal(flux, 2 * np.pi * r * a)


@mark.parametrize("par", [True, False])
def test_flux_density_circular_filament_on_axis(par):
    # Superpose translated loops with both current signs. Include signed zero,
    # loop centers, and distant axial points where elliptic terms can cancel.
    rfil = np.array([0.3, 1.7, 2.1])
    zfil = np.array([-0.5, 0.0, 1.25])
    ifil = np.array([2.0, -3.0, 0.75])
    zobs = np.array([-1e4, -2.0, -0.5, 0.0, 1.25, 3.0, 1e4])
    robs = np.zeros_like(zobs)
    robs[::2] = -0.0
    expected_bz = np.sum(
        cfsem.MU_0 * ifil * rfil**2 / (2.0 * (rfil**2 + (zobs[:, None] - zfil) ** 2) ** 1.5),
        axis=1,
    )
    # Python and Rust currently use permeability constants differing by 6.8e-10.
    # The Rust scalar test checks accuracy near machine precision.
    rtol = 1e-9

    br, bz = cfsem.flux_density_circular_filament(ifil, rfil, zfil, robs, zobs, par)
    np.testing.assert_array_equal(br, 0.0)
    np.testing.assert_allclose(bz, expected_bz, rtol=rtol, atol=0.0)

    for calc in (cfsem.vector_potential_circular_filament, cfsem.flux_circular_filament):
        for wire in (None, np.zeros_like(ifil), np.full_like(ifil, 0.001)):
            np.testing.assert_array_equal(calc(ifil, rfil, zfil, robs, zobs, par, wire), 0.0)

    bx, by, bz = cfsem.flux_density_circular_filament_cartesian(
        ifil,
        rfil,
        (np.zeros_like(zfil), np.zeros_like(zfil), zfil),
        (np.zeros_like(zfil), np.zeros_like(zfil), np.ones_like(zfil)),
        (robs, robs, zobs),
        par,
    )
    np.testing.assert_array_equal(bx, 0.0)
    np.testing.assert_array_equal(by, 0.0)
    np.testing.assert_allclose(bz, expected_bz, rtol=rtol, atol=0.0)


@mark.parametrize("a", [0.775, np.pi])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("par", [True, False])
def test_flux_density_circular_filament_against_numerical(a, z, par):
    # Test the elliptic-integral calc for B-field of a loop against numerical integration

    n = 10
    rs = np.linspace(0.1, 10.0, n)
    zs = np.linspace(-5.0, 5.0, n)

    R, Z = np.meshgrid(rs, zs, indexing="ij")
    rprime = R.flatten()
    zprime = Z.flatten()

    current = 1.0  # 1A reference current

    # Calc using elliptic integral fits
    Br, Bz = cfsem.flux_density_circular_filament(
        np.array([current]), np.array([a]), np.array([z]), rprime, zprime, par
    )  # [T]

    # Calc using numerical integration around the loop
    Br_num = np.zeros_like(Br)
    Bz_num = np.zeros_like(Br)
    for i, x in enumerate(zip(rprime, zprime, strict=True)):
        robs, zobs = x
        Br_num[i], Bz_num[i] = _test._flux_density_circular_filament_numerical(
            current, a, robs, zobs - z, n=100
        )

    assert np.allclose(Br, Br_num)
    assert np.allclose(Bz, Bz_num)


@mark.parametrize("par", [True, False])
@mark.parametrize("nr", [20])
@mark.parametrize("nz", [20])
def test_self_inductance_lyle6_against_filamentization_and_distributed_and_axisymmetric(par, nr, nz):
    # Test that the Lyle approximation gives a similar result to
    # a case done by brute-force filamentization w/ a heuristic for self-inductance of a loop
    r, z, dr, dz, nt = (0.8, 0.0, 0.5, 2.0, 3.0)
    L_Lyle = cfsem.self_inductance_lyle6(
        r, dr, dz, nt
    )  # Estimate self-inductance via closed-form approximation
    L_fil = _test._self_inductance_filamentized(
        r, z, dr, dz, nt, nr, nz
    )  # Estimate self-inductance via discretization

    # Approximate conductor cross-section for axisymmetric calc
    cnd_w, cnd_h = (dr / nr, dz / nz)  # Approximate conductor width and height

    # Set up distributed-conductor solve
    fils = cfsem.filament_coil(r, z, dr, dz, nt, nr, nz)
    rfil, zfil, _ = fils.T
    current = np.ones_like(rfil) / rfil.size  # [A] 1A total reference current
    rgrid = np.arange(0.5, 2.0, 0.05)
    zgrid = np.arange(-3.0, 3.0, 0.05)
    rmesh, zmesh = np.meshgrid(rgrid, zgrid, indexing="ij")
    #  Do filamentized psi and B calcs for convenience,
    #  although ideally we'd do a grad-shafranov solve here for a smoother field
    psi = cfsem.flux_circular_filament(current, rfil, zfil, rmesh.flatten(), zmesh.flatten(), par)
    psi = psi.reshape(rmesh.shape)
    br, bz = cfsem.flux_density_circular_filament(current, rfil, zfil, rmesh.flatten(), zmesh.flatten(), par)
    br = br.reshape(rmesh.shape)
    bz = bz.reshape(rmesh.shape)
    #  Build up the mask of the conductor region
    rmin = r - dr / 2
    rmax = r + dr / 2
    zmin = z - dz / 2
    zmax = z + dz / 2
    mask = np.where(rmesh > rmin, True, False)
    mask *= np.where(rmesh < rmax, True, False)
    mask *= np.where(zmesh > zmin, True, False)
    mask *= np.where(zmesh < zmax, True, False)
    #  Build a rough approximation of the conductor bounding contour
    rleft = (rmin - 0.05) * np.ones(10)
    rtop = np.linspace(rmin - 0.05, rmax + 0.05, 10)
    rright = (rmax + 0.05) * np.ones(10)
    rbot = rtop[::-1]
    rpath = np.concatenate((rleft, rtop, rright, rbot))
    zleft = np.linspace(zmin - 0.05, zmax + 0.05, 10)
    ztop = (zmax + 0.05) * np.ones(10)
    zright = zleft[::-1]
    zbot = (zmin - 0.05) * np.ones(10)
    zpath = np.concatenate((zleft, ztop, zright, zbot))
    #  Do the distributed conductor calc
    L_distributed, _, _ = cfsem.self_inductance_distributed_axisymmetric_conductor(
        current=1.0,
        grid=(rgrid, zgrid),
        mesh=(rmesh, zmesh),
        b_part=(br, bz),
        psi_part=psi,
        mask=mask,
        edge_path=(rpath, zpath),
    )

    # Do the axisymmetric run
    L_axisymmetric = cfsem.self_inductance_axisymmetric_coil(
        f=fils.T,
        section_kind="rectangular",
        section_size=(cnd_w, cnd_h),
    )

    # Require 5% accuracy (seat of the pants, since we're comparing approximations)
    assert L_Lyle == approx(L_fil, 0.05)
    assert (nt**2 * L_distributed) == approx(L_fil, 0.05)
    assert L_Lyle == approx(L_axisymmetric, 0.05)


@mark.parametrize("par", [True, False])
def test_self_inductance_axisymmetric_across_section_types(par):
    """Test that the different conductor cross-section types give similar results"""
    r, z, dr, dz, nt, nr, nz = (0.8, 0.0, 0.5, 2.0, 3.0, 20, 20)
    # Approximate conductor cross-section for axisymmetric calc
    cnd_w, cnd_h = (dr / nr, dz / nz)  # Approximate conductor width and height

    # Set up distributed-conductor solve
    fils = cfsem.filament_coil(r, z, dr, dz, nt, nr, nz)

    cnd_w, cnd_h = (dr / 20, dz / 20)  # Approximate conductor width and height
    cnd_r = (cnd_w * cnd_h / np.pi) ** 0.5  # Equivalent-area radius

    # Use base height/width for rectangular
    L_rect = cfsem.self_inductance_axisymmetric_coil(
        f=fils.T,
        section_kind="rectangular",
        section_size=(cnd_w, cnd_h),
    )
    # Use equivalent-area radius for circular and annular
    L_circle = cfsem.self_inductance_axisymmetric_coil(
        f=fils.T,
        section_kind="circular",
        section_size=cnd_r,
    )
    # Use equivalent-area radius for major radius, and outer radius = 2*inner radius
    L_annulus = cfsem.self_inductance_axisymmetric_coil(
        f=fils.T,
        section_kind="annular",
        section_size=(cnd_r / 2, cnd_r),
    )

    assert L_rect == approx(L_circle, rel=1e-2)
    assert L_rect == approx(L_annulus, rel=1e-2)


@mark.parametrize("r", [0.775, np.pi])
@mark.parametrize("dr", [0.001, 0.02])
@mark.parametrize("nt", [1.0, 7.7])
def test_self_inductance_lyle6_against_wien(r, dr, nt):
    """Test that the Lyle approximation gives a similar result to
    Wien's formula for self-inductance of a thin circular loop."""
    r, dr, dz, nt = (r, dr, dr, nt)
    L_Lyle = cfsem.self_inductance_lyle6(
        r, dr, dz, nt
    )  # [H] Estimate self-inductance via closed-form approximation
    L_wien = nt**2 * cfsem.self_inductance_circular_ring_wien(
        major_radius=r, minor_radius=(0.5 * (dr**2 + dz**2) ** 0.5)
    )  # [H]  Estimate self-inductance via Wien's formula
    assert L_Lyle == approx(L_wien, rel=0.05)  # Require 5% accuracy (seat of the pants)


def test_wien_against_paper_examples():
    """
    Test self_inductance_circular_ring_wien againts the examples in the paper it is taken from.
    This is indirectly tested against a parametrized filamentization in test_self_inductance_annular_ring .
    """
    major_radius_1 = 25e-2
    minor_radius_1 = 0.05e-2
    L_ref_1 = 654.40537 * np.pi * 1e-7 * 1e-2  # units: henry
    L_1 = cfsem.self_inductance_circular_ring_wien(major_radius_1, minor_radius_1)
    assert approx(L_ref_1) == L_1

    major_radius_2 = 25e-2
    minor_radius_2 = 0.5e-2
    L_ref_2 = 424.1761 * np.pi * 1e-7 * 1e-2  # units: henry
    L_2 = cfsem.self_inductance_circular_ring_wien(major_radius_2, minor_radius_2)
    assert approx(L_ref_2) == L_2


@mark.parametrize("r", [0.775, 1.5])
@mark.parametrize("z", [0.0, np.pi])
@mark.parametrize("dr_over_r", [0.1, 0.2])
@mark.parametrize("dz_over_r", [0.1, 3.5])
@mark.parametrize("nt", [3.0, 400.0])
@mark.parametrize("nr", [5, 7])
@mark.parametrize("nz", [50, 100])
def test_self_inductance_lyle6_against_filamentized(r, z, dr_over_r, dz_over_r, nt, nr, nz):
    # Test that the Lyle approximation gives a similar result to
    # a case done by brute-force filamentization w/ a heuristic for self-inductance of a loop
    r, z, dr, dz, nt = (
        r,
        z,
        r * dr_over_r,
        r * dz_over_r,
        nt,
    )
    L_Lyle = cfsem.self_inductance_lyle6(
        r, dr, dz, nt
    )  # Estimate self-inductance via closed-form approximation
    L_fil = _test._self_inductance_filamentized(
        r, z, dr, dz, nt, nr, nz
    )  # Estimate self-inductance via discretization
    assert float(L_Lyle) == approx(L_fil, 0.05)  # Require 5% accuracy (seat of the pants)


@mark.parametrize("major_radius", np.linspace(0.35, 1.25, 3, endpoint=True))
@mark.parametrize("a", np.linspace(0.01, 0.04, 3, endpoint=True))
@mark.parametrize("b", np.linspace(0.05, 0.1, 3, endpoint=True))
@mark.parametrize("n_grid", [100])
@mark.parametrize("nr_fil", [10])
@mark.parametrize("nz_fil", [10])
def test_self_inductance_annular_ring(major_radius, a, b, n_grid, nr_fil, nz_fil):
    # First, test a near-solid version against Wien for a solid loop
    major_radius_1 = major_radius
    minor_radius_1 = b
    inner_minor_radius_1 = 1e-4

    L_wien_1 = cfsem.self_inductance_circular_ring_wien(major_radius_1, minor_radius_1)
    L_annular_1 = cfsem.self_inductance_annular_ring(major_radius_1, inner_minor_radius_1, minor_radius_1)

    assert L_annular_1 == approx(L_wien_1, rel=1e-2)

    # Then, test thick hollow version against filamentization
    major_radius_2 = major_radius
    minor_radius_2 = b
    inner_minor_radius_2 = a

    L_annular_2 = cfsem.self_inductance_annular_ring(major_radius_2, inner_minor_radius_2, minor_radius_2)

    rs = np.linspace(
        major_radius_2 - minor_radius_2,
        major_radius_2 + minor_radius_2,
        n_grid,
        endpoint=True,
    )

    zs = np.linspace(
        -minor_radius_2,
        minor_radius_2,
        n_grid,
        endpoint=True,
    )

    rmesh, zmesh = np.meshgrid(rs, zs, indexing="ij")
    mask = np.ones_like(rmesh)
    mask *= np.where(np.sqrt(zmesh**2 + (rmesh - major_radius_2) ** 2) <= minor_radius_2, True, False)
    mask *= np.where(
        np.sqrt(zmesh**2 + (rmesh - major_radius_2) ** 2) >= inner_minor_radius_2,
        True,
        False,
    )

    L_fil = _test._self_inductance_filamentized(
        major_radius_2,
        0.0,
        minor_radius_2 * 2,
        minor_radius_2 * 2,
        nt=1.0,
        nr=nr_fil,
        nz=nz_fil,
        mask=(rs, zs, mask),
    )  # Estimate self-inductance via discretization

    assert L_annular_2 == approx(L_fil, rel=2e-2)

    # Exercise error handling
    with raises(ValueError):
        # Zero radius
        cfsem.self_inductance_annular_ring(0.1, 0.0, 0.01)

    with raises(ValueError):
        # Larger inner than outer
        cfsem.self_inductance_annular_ring(0.1, 0.02, 0.01)

    with raises(ValueError):
        # Larger outer than major
        cfsem.self_inductance_annular_ring(0.1, 0.01, 0.11)


@mark.parametrize("r", [0.775, 1.51])
@mark.parametrize("z", [0.0, np.pi])
@mark.parametrize("par", [True, False])
def test_vector_potential_axisymmetric(r, z, par):
    # Spot-check vector potential against inductance calcs.
    # More detailed three-way tests against both B-field and inductance
    # are done in the Rust library.

    # Filament
    ifil = np.atleast_1d([1.0])  # [A] 1A total reference current
    rfil = np.atleast_1d([r])
    zfil = np.atleast_1d([z])

    # Observation points
    rgrid = np.arange(0.5, 2.0, 0.05)
    zgrid = np.arange(-3.0, 3.0, 0.05)
    rmesh, zmesh = np.meshgrid(rgrid, zgrid, indexing="ij")

    psi = cfsem.flux_circular_filament(ifil, rfil, zfil, rmesh.flatten(), zmesh.flatten(), par)
    a_phi = cfsem.vector_potential_circular_filament(ifil, rfil, zfil, rmesh.flatten(), zmesh.flatten(), par)

    # Integrate vector potential around a loop to get the flux
    psi_from_a = 2.0 * np.pi * rmesh.flatten() * a_phi  # [Wb]

    # We should not be able to tell the difference above a single roundoff
    assert np.allclose(psi, psi_from_a, rtol=1e-16, atol=1e-16)


@mark.parametrize("r", [-np.pi, -0.775, 0.775, np.pi])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("n_filaments", [int(1e4)])
@mark.parametrize("par", [True, False])
def test_vector_potential_linear_against_circular_filament(r, z, n_filaments, par):
    # Note we are mapping between (x, y, z) and (r, phi, z) coordinates here

    # Biot-Savart filaments in cartesian coords
    phi = np.linspace(0.0, 2.0 * np.pi, n_filaments)
    xfils = r * np.cos(phi)
    yfils = r * np.sin(phi)
    zfils = np.ones_like(xfils) * z

    # Observation grid
    rs = np.linspace(-abs(r) + 0.1, abs(r) - 0.1, 20)
    zs = np.linspace(-1.0, 1.0, 10)

    rmesh, zmesh = np.meshgrid(rs, zs, indexing="ij")
    rprime = rmesh.flatten()
    zprime = zmesh.flatten()

    # Circular filament calc
    a_phi = cfsem.vector_potential_circular_filament(
        np.ones(1), np.array([r]), np.array([z]), rprime, zprime, par
    )  # [V-s/m]

    # Biot-Savart calc
    xyzp = (rprime, np.zeros_like(zprime), zprime)
    xyzfil = (xfils[1:], yfils[1:], zfils[1:])
    dlxyzfil = (xfils[1:] - xfils[:-1], yfils[1:] - yfils[:-1], zfils[1:] - zfils[:-1])
    ifil = np.ones_like(xfils[1:])
    wire_radius = np.zeros_like(ifil)
    ax, ay, az = cfsem.vector_potential_linear_filament(
        xyzp, xyzfil, dlxyzfil, ifil, wire_radius, par
    )  # [V-s/m]

    assert np.allclose(a_phi, ay, rtol=1e-12, atol=1e-12)  # Should match circular calc
    assert np.allclose(az, np.zeros_like(az), atol=1e-9)  # Should sum to zero everywhere
    assert np.allclose(ax, np.zeros_like(ax), atol=1e-9)  # ...


@mark.parametrize("ndiscr", [100, 200, 400, 800])
@mark.parametrize("par", [True, False])
def test_vector_potential_linear_self_inductance_against_wien(ndiscr, par):
    """Test integration of dot(A, dL) against Wien's formula for self-inductance."""
    major_radius = 0.5  # [m]
    minor_radius = 5e-3  # [m] finite conductor radius
    l_from_a = _linear_filament_self_inductance_from_vector_potential(
        major_radius=major_radius,
        minor_radius=minor_radius,
        ndiscr=ndiscr,
        par=par,
    )
    l_wien = float(cfsem.self_inductance_circular_ring_wien(major_radius, minor_radius))  # [H]
    assert l_from_a == approx(
        l_wien, rel=8e-2
    ), f"ndiscr={ndiscr}, L_from_A={l_from_a:.6e}, L_wien={l_wien:.6e}"


@mark.parametrize("ndiscr_coarse", [100, 200, 400])
@mark.parametrize("par", [True, False])
def test_vector_potential_linear_self_inductance_discretization_stability(ndiscr_coarse, par):
    major_radius = 0.5  # [m]
    minor_radius = 5e-3  # [m] finite conductor radius
    l_coarse = _linear_filament_self_inductance_from_vector_potential(
        major_radius=major_radius,
        minor_radius=minor_radius,
        ndiscr=ndiscr_coarse,
        par=par,
    )
    l_fine = _linear_filament_self_inductance_from_vector_potential(
        major_radius=major_radius,
        minor_radius=minor_radius,
        ndiscr=2 * ndiscr_coarse,
        par=par,
    )
    l_wien = float(cfsem.self_inductance_circular_ring_wien(major_radius, minor_radius))  # [H]
    rel_delta = abs(l_fine - l_coarse) / max(abs(l_wien), 1e-30)
    assert rel_delta < 2e-2, f"ndiscr_coarse={ndiscr_coarse}, rel_delta={rel_delta:.6e}"


@mark.parametrize("r", [0.5, np.pi])
@mark.parametrize("a", [1e-3, 1e-2, 2e-2])
@mark.parametrize("n", [400, int(1e3)])
def test_linear_filament_self_inductance_against_wien(r, a, n):
    major_radius = r  # [m]
    minor_radius = a  # [m]
    ndiscr = n

    phi = np.linspace(0.0, 2.0 * np.pi, ndiscr, endpoint=True)
    x = major_radius * np.cos(phi)
    y = major_radius * np.sin(phi)
    z = np.zeros_like(x)

    l_self = float(cfsem.self_inductance_piecewise_linear_filaments((x, y, z), wire_radius=minor_radius))
    l_wien = float(cfsem.self_inductance_circular_ring_wien(major_radius, minor_radius))

    assert l_self == approx(l_wien, rel=8e-2)


@mark.parametrize("ndiscr", [128, 200])
@mark.parametrize("par", [True, False])
def test_vector_potential_linear_matrix_contracts_to_vector_and_inductance(ndiscr, par):
    major_radius = 0.5  # [m]
    minor_radius = 5e-3  # [m]
    x, y, z = _circular_loop_xyz(major_radius, ndiscr)

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dz = z[1:] - z[:-1]
    xyzfil = (x[:-1], y[:-1], z[:-1])
    dlxyzfil = (dx, dy, dz)
    tq = np.array(
        [
            0.11270166537925831,
            0.5,
            0.8872983346207417,
        ]
    )
    wq = np.array(
        [
            0.2777777777777778,
            0.4444444444444444,
            0.2777777777777778,
        ]
    )
    xq = x[:-1, None] + tq[None, :] * dx[:, None]
    yq = y[:-1, None] + tq[None, :] * dy[:, None]
    zq = z[:-1, None] + tq[None, :] * dz[:, None]
    xyzquad = (xq.reshape(-1), yq.reshape(-1), zq.reshape(-1))
    ifil = np.ones_like(dx)

    ax, ay, az = cfsem.vector_potential_linear_filament(
        xyzquad, xyzfil, dlxyzfil, ifil, wire_radius=minor_radius, par=par, output="vector"
    )
    axm, aym, azm = cfsem.vector_potential_linear_filament(
        xyzquad, xyzfil, dlxyzfil, ifil, wire_radius=minor_radius, par=par, output="matrix"
    )

    assert axm.shape == (3 * dx.size, dx.size)
    assert aym.shape == (3 * dx.size, dx.size)
    assert azm.shape == (3 * dx.size, dx.size)
    assert np.allclose(ax, np.sum(axm, axis=1), rtol=1e-12, atol=1e-12)
    assert np.allclose(ay, np.sum(aym, axis=1), rtol=1e-12, atol=1e-12)
    assert np.allclose(az, np.sum(azm, axis=1), rtol=1e-12, atol=1e-12)

    aq_dot_dl = (
        ax.reshape((-1, 3)) * dx[:, None]
        + ay.reshape((-1, 3)) * dy[:, None]
        + az.reshape((-1, 3)) * dz[:, None]
    )
    l_from_a = float(np.sum(wq[None, :] * aq_dot_dl))
    l_direct = float(
        cfsem.inductance_piecewise_linear_filaments(
            xyzfil0=xyzfil,
            dlxyzfil0=dlxyzfil,
            xyzfil1=xyzfil,
            dlxyzfil1=dlxyzfil,
            wire_radius=minor_radius,
        )
    )
    l_self = float(cfsem.self_inductance_piecewise_linear_filaments((x, y, z), wire_radius=minor_radius))
    l_wien = float(cfsem.self_inductance_circular_ring_wien(major_radius, minor_radius))

    assert l_direct == approx(l_from_a, rel=1e-12)
    assert l_self == approx(l_direct, rel=1e-12)
    assert l_direct == approx(l_wien, rel=8e-2)


def test_vector_potential_linear_invalid_output_mode():
    xyzp = (np.array([0.1]), np.array([0.2]), np.array([0.3]))
    xyzfil = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
    dlxyzfil = (np.array([1.0]), np.array([0.0]), np.array([0.0]))
    ifil = np.array([1.0])

    with raises(ValueError, match="output must be 'vector' or 'matrix'"):
        cfsem.vector_potential_linear_filament(xyzp, xyzfil, dlxyzfil, ifil, wire_radius=0.0, output="bad")


@mark.parametrize("par", [True, False])
def test_inductance_linear_filaments_matrix_contracts_to_vector(par):
    xyzfil_src = (
        np.array([-0.4, -0.1, 0.2, 0.5]),
        np.array([0.0, 0.1, -0.1, 0.05]),
        np.array([-0.2, -0.1, 0.0, 0.1]),
    )
    dlxyzfil_src = (
        np.array([0.06, 0.05, 0.04, 0.03]),
        np.array([0.01, -0.02, 0.03, -0.01]),
        np.array([0.02, 0.01, -0.01, 0.0]),
    )
    wire_radius_src = np.array([2e-3, 3e-3, 4e-3, 5e-3])

    xyzfil_tgt = (
        np.array([0.6, 0.9, 1.2]),
        np.array([-0.1, -0.05, 0.0]),
        np.array([0.3, 0.2, 0.1]),
    )
    dlxyzfil_tgt = (
        np.array([-0.03, -0.02, -0.01]),
        np.array([0.02, 0.025, 0.03]),
        np.array([0.01, 0.008, 0.006]),
    )

    m_vec = cfsem.inductance_linear_filaments(
        xyzfil_tgt,
        dlxyzfil_tgt,
        xyzfil_src,
        dlxyzfil_src,
        wire_radius_src=wire_radius_src,
        output="vector",
    )
    m_mat = cfsem.inductance_linear_filaments(
        xyzfil_tgt,
        dlxyzfil_tgt,
        xyzfil_src,
        dlxyzfil_src,
        wire_radius_src=wire_radius_src,
        par=par,
        output="matrix",
    )

    assert m_mat.shape == (xyzfil_src[0].size, xyzfil_tgt[0].size)
    assert np.allclose(m_vec, np.sum(m_mat, axis=0), rtol=1e-12, atol=1e-15)

    m_piecewise = cfsem.inductance_piecewise_linear_filaments(
        xyzfil0=xyzfil_src,
        dlxyzfil0=dlxyzfil_src,
        xyzfil1=xyzfil_tgt,
        dlxyzfil1=dlxyzfil_tgt,
        wire_radius=wire_radius_src,
    )
    assert float(np.sum(m_vec)) == approx(m_piecewise, rel=1e-12)
    # Smoke-test scalar wire-radius broadcasting on both output paths.
    m_vec_scalar = cfsem.inductance_linear_filaments(
        xyzfil_tgt,
        dlxyzfil_tgt,
        xyzfil_src,
        dlxyzfil_src,
        wire_radius_src=2e-3,
        output="vector",
    )
    m_mat_scalar = cfsem.inductance_linear_filaments(
        xyzfil_tgt,
        dlxyzfil_tgt,
        xyzfil_src,
        dlxyzfil_src,
        wire_radius_src=2e-3,
        par=par,
        output="matrix",
    )
    assert m_vec_scalar.shape == (xyzfil_tgt[0].size,)
    assert m_mat_scalar.shape == (xyzfil_src[0].size, xyzfil_tgt[0].size)
    assert np.all(np.isfinite(m_vec_scalar))
    assert np.all(np.isfinite(m_mat_scalar))

    with raises(ValueError, match="output must be 'vector' or 'matrix'"):
        cfsem.inductance_linear_filaments(
            xyzfil_tgt,
            dlxyzfil_tgt,
            xyzfil_src,
            dlxyzfil_src,
            wire_radius_src=2e-3,
            output="bad",
        )


@mark.parametrize("par", [True, False])
@mark.parametrize("csc_type", [sparse.csc_matrix, sparse.csc_array])
def test_inductance_linear_filaments_sparse_preserves_csc_pattern(par, csc_type):
    xyzfil_src = (
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.1, -0.1]),
        np.zeros(3),
    )
    dlxyzfil_src = (np.zeros(3), np.zeros(3), np.full(3, 0.8))
    xyzfil_tgt = (
        np.array([0.2, 1.2, 2.2, 3.2]),
        np.array([0.3, -0.2, 0.1, 0.0]),
        np.array([0.1, 0.2, -0.1, 0.3]),
    )
    dlxyzfil_tgt = (
        np.full(4, 0.1),
        np.full(4, 0.05),
        np.array([0.4, 0.4, 0.0, 0.4]),
    )
    row_indices = np.array([0, 2, 1, 0, 1, 2], dtype=np.int32)
    column_pointers = np.array([0, 2, 2, 3, 6], dtype=np.int32)
    interaction_map = csc_type(
        (np.full(row_indices.size, np.nan), row_indices, column_pointers),
        shape=(3, 4),
    )

    actual = cfsem.inductance_linear_filaments_sparse(
        xyzfil_tgt,
        dlxyzfil_tgt,
        xyzfil_src,
        dlxyzfil_src,
        interaction_map,
        wire_radius_src=np.array([0.01, 0.02, 0.03]),
        par=par,
    )
    dense = cfsem.inductance_linear_filaments(
        xyzfil_tgt,
        dlxyzfil_tgt,
        xyzfil_src,
        dlxyzfil_src,
        wire_radius_src=np.array([0.01, 0.02, 0.03]),
        par=par,
        output="matrix",
    )

    assert sparse.isspmatrix_csc(actual)
    assert actual.shape == interaction_map.shape
    np.testing.assert_array_equal(actual.indices, interaction_map.indices)
    np.testing.assert_array_equal(actual.indptr, interaction_map.indptr)
    for target in range(actual.shape[1]):
        start, end = actual.indptr[target : target + 2]
        np.testing.assert_allclose(
            actual.data[start:end],
            dense[actual.indices[start:end], target],
            rtol=1e-14,
            atol=1e-18,
        )
    assert actual.nnz == interaction_map.nnz
    assert actual.data[2] == 0.0

    scalar_radius = cfsem.inductance_linear_filaments_sparse(
        xyzfil_tgt,
        dlxyzfil_tgt,
        xyzfil_src,
        dlxyzfil_src,
        interaction_map,
        wire_radius_src=0.01,
        par=par,
    )
    assert scalar_radius.shape == interaction_map.shape


def test_inductance_linear_filaments_sparse_validates_map():
    xyz = (np.array([0.0, 1.0]), np.zeros(2), np.zeros(2))
    dlxyz = (np.zeros(2), np.zeros(2), np.ones(2))
    csr_map = sparse.eye(2, format="csr")
    with raises(TypeError, match="must be a scipy.sparse.csc_matrix or csc_array"):
        cfsem.inductance_linear_filaments_sparse(xyz, dlxyz, xyz, dlxyz, csr_map)

    wrong_shape = sparse.eye(3, format="csc")
    with raises(ValueError, match="must have shape"):
        cfsem.inductance_linear_filaments_sparse(xyz, dlxyz, xyz, dlxyz, wrong_shape)

    duplicate_rows = sparse.csc_matrix(
        (np.ones(2), np.array([0, 0]), np.array([0, 2, 2])),
        shape=(2, 2),
    )
    assert not duplicate_rows.has_canonical_format
    with raises(ValueError, match="sorted, unique row indices"):
        cfsem.inductance_linear_filaments_sparse(xyz, dlxyz, xyz, dlxyz, duplicate_rows)


@mark.parametrize("ndiscr", [128, 200])
@mark.parametrize("par", [True, False])
def test_flux_density_linear_matrix_contracts_to_vector(ndiscr, par):
    major_radius = 0.5  # [m]
    minor_radius = 5e-3  # [m]
    x, y, z = _circular_loop_xyz(major_radius, ndiscr)

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dz = z[1:] - z[:-1]
    xyzfil = (x[:-1], y[:-1], z[:-1])
    dlxyzfil = (dx, dy, dz)
    xyzobs = (
        np.array([0.1, 0.2, -0.3]),
        np.array([0.2, -0.1, 0.4]),
        np.array([0.0, 0.3, -0.2]),
    )
    ifil = np.ones_like(dx)

    bx, by, bz = cfsem.flux_density_linear_filament(
        xyzobs, xyzfil, dlxyzfil, ifil, wire_radius=minor_radius, par=par, output="vector"
    )
    bxm, bym, bzm = cfsem.flux_density_linear_filament(
        xyzobs, xyzfil, dlxyzfil, ifil, wire_radius=minor_radius, par=par, output="matrix"
    )

    assert bxm.shape == (xyzobs[0].size, dx.size)
    assert bym.shape == (xyzobs[0].size, dx.size)
    assert bzm.shape == (xyzobs[0].size, dx.size)
    assert np.allclose(bx, np.sum(bxm, axis=1), rtol=1e-12, atol=1e-12)
    assert np.allclose(by, np.sum(bym, axis=1), rtol=1e-12, atol=1e-12)
    assert np.allclose(bz, np.sum(bzm, axis=1), rtol=1e-12, atol=1e-12)


def test_flux_density_linear_invalid_output_mode():
    xyzp = (np.array([0.1]), np.array([0.2]), np.array([0.3]))
    xyzfil = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
    dlxyzfil = (np.array([1.0]), np.array([0.0]), np.array([0.0]))
    ifil = np.array([1.0])

    with raises(ValueError, match="output must be 'vector' or 'matrix'"):
        cfsem.flux_density_linear_filament(xyzp, xyzfil, dlxyzfil, ifil, wire_radius=0.0, output="bad")


def test_inductance_matrix_axisymmetric_coaxial_rectangular_coils():
    # Create set of four non-overlapping coaxial rectangular coils and prescribed turn density
    r = [0.5, 1.0, 1.5, 2.0]
    z = [0.0, -0.4, +0.2, 0.6]
    dr = [0.1, 0.2, 0.3, 0.2]
    dz = [0.2, 0.3, 0.3, 0.2]
    td = [10.0, 10.0, 5.0, 5.0]
    nr = [10, 10, 10, 10]
    nz = [10, 10, 10, 10]

    # Calculate inductance matrix using rectangular coil calc
    L_rectangular_coils = cfsem.inductance_matrix_axisymmetric_coaxial_rectangular_coils(
        r=r,
        z=z,
        dr=dr,
        dz=dz,
        td=td,
        nr=nr,
        nz=nz,
    )

    # One run with refined filamentization to make sure this does not significantly change results
    L_rectangular_coils_fine = cfsem.inductance_matrix_axisymmetric_coaxial_rectangular_coils(
        r=r,
        z=z,
        dr=dr,
        dz=dz,
        td=td,
        nr=[n * 2 for n in nr],
        nz=[n * 2 for n in nz],
    )
    assert np.allclose(L_rectangular_coils, L_rectangular_coils_fine, rtol=1e-6)

    # Compare total inductance against fully filamentized calculation
    nt = [td[c] * dr[c] * dz[c] for c in range(4)]
    filaments = np.vstack(
        [cfsem.filament_coil(r[i], z[i], dr[i], dz[i], nt[i], nr[i], nz[i]) for i in range(4)]
    )
    L_fully_filamentized = cfsem.self_inductance_axisymmetric_coil(
        f=filaments.T,
        section_kind="rectangular",
        section_size=(2e-3, 2e-3),
    )
    assert L_rectangular_coils.sum() == approx(L_fully_filamentized, rel=1e-2)

    # Compare individual self-inductances against Lyle approx for first coil
    for i in range(4):
        L_lyle = cfsem.self_inductance_lyle6(
            r=r[i],
            dr=dr[i],
            dz=dz[i],
            n=nt[i],
        )
        # Should be identical because self_inductance_axisymmetric_coil uses the same underlying
        # calculation, here we're also testing that self_inductance_axisymmetric_coil is using
        # the correct indexing
        assert L_rectangular_coils[i, i] == L_lyle

    # Overlap check 1:
    # "Corner" + "Corner" overlap
    with raises(AssertionError):
        # Overlapping coils in r
        cfsem.inductance_matrix_axisymmetric_coaxial_rectangular_coils(
            r=[+1.0, +1.8],
            z=[-0.5, +0.3],
            dr=[1.0, 1.0],
            dz=[1.0, 1.0],
            td=[1.0, 1.0],
            nr=[10, 10],
            nz=[10, 10],
        )

    # Overlap check 2:
    # "Corner" + "Full R size" overlap
    with raises(AssertionError):
        # Overlapping coils in r
        cfsem.inductance_matrix_axisymmetric_coaxial_rectangular_coils(
            r=[+1.0, +1.8],
            z=[-0.5, +0.3],
            dr=[1.0, 4.0],
            dz=[1.0, 1.0],
            td=[1.0, 1.0],
            nr=[10, 10],
            nz=[10, 10],
        )

    # Overlap check 3:
    # "Corner" + "Full Z size" overlap
    with raises(AssertionError):
        # Overlapping coils in z
        cfsem.inductance_matrix_axisymmetric_coaxial_rectangular_coils(
            r=[+1.0, +1.8],
            z=[-0.5, +0.3],
            dr=[1.0, 1.0],
            dz=[1.0, 4.0],
            td=[1.0, 1.0],
            nr=[10, 10],
            nz=[10, 10],
        )

    # Overlap check 4:
    # Coil 1 fully inside Coil 2
    with raises(AssertionError):
        # Overlapping coils in both r and z
        cfsem.inductance_matrix_axisymmetric_coaxial_rectangular_coils(
            r=[+1.5, +1.6],
            z=[-0.1, +0.1],
            dr=[1.0, 2.0],
            dz=[1.0, 2.0],
            td=[1.0, 1.0],
            nr=[10, 10],
            nz=[10, 10],
        )
