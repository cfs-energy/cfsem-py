"""Tests of standalone electromagnetics calcs"""

import numpy as np
from pytest import approx, mark, raises

import cfsem

from . import test_funcs as _test


@mark.parametrize("r", [0.775, np.pi])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("par", [True, False])
def test_body_force_density(r, z, par):
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

    fil, dlxyzfil = _test._filament_loop(r, z, 1000)
    xyzfil = (fil[0][:-1], fil[1][:-1], fil[2][:-1])
    ifil = np.ones_like(xyzfil[0])

    jxbx, jxby, jxbz = cfsem.body_force_density_circular_filament_cartesian(
        [1.0], [r], [z], obs, j, par
    )
    jxbx1, jxby1, jxbz1 = cfsem.body_force_density_linear_filament(
        xyzfil, dlxyzfil, ifil, obs, j, par
    )

    assert np.allclose(jxbx, jxbx1, rtol=1e-2, atol=1e-9)
    assert np.allclose(jxby, jxby1, rtol=1e-2, atol=1e-9)
    assert np.allclose(jxbz, jxbz1, rtol=1e-2, atol=1e-9)


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
        [1.0], [r], [z], (xmesh, ymesh, zmesh), par
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


@mark.parametrize("r", [0.775, np.pi])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("par", [True, False])
def test_mutual_inductance_circular_to_linear(r, z, par):
    """Spot check bindings; more complete tests are run in Rust"""
    r1, z1 = (r + 0.1, abs(z) ** 0.5)
    fil, _dl = _test._filament_loop(r, z, ndiscr=200)
    fil1, dl1 = _test._filament_loop(r1, z1, ndiscr=200)
    (x1, y1, z1) = fil1

    m_linear = cfsem.mutual_inductance_piecewise_linear_filaments(fil, fil1)
    m_circular = cfsem.flux_circular_filament([1.0], [r], [z], [r1], [z1])
    m_circular_to_linear = cfsem.mutual_inductance_circular_to_linear(
        [r], [z], [1.0], (x1[:-1], y1[:-1], z1[:-1]), dl1, par
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
        [1.0], [r], [z], (xmesh, ymesh, zmesh), par
    )
    br, bz_circ = cfsem.flux_density_circular_filament(
        [1.0], [r], [z], xmesh, zmesh, par
    )

    assert np.allclose(bx, br, rtol=1e-6, atol=1e-10)
    assert np.allclose(bz, bz_circ, rtol=1e-6, atol=1e-10)
    assert np.allclose(by, np.zeros_like(by), atol=1e-10)


@mark.parametrize("r", [7.7, np.pi])  # Needs to be large for Lyle with very small width
@mark.parametrize("z", [0.0, np.e / 2])
@mark.parametrize("h_over_r", [5e-2, 0.25, 1.0])
def test_self_inductance_piecewise_linear_filaments(r, z, h_over_r):
    # Test self inductance via neumann's formula
    # against Lyle's calc for finite-thickness coils
    w = 0.001  # [m] can't be infinitesimally thin for Lyle's calc, but can be very thin compared to height and radius
    h = h_over_r * r  # [m]

    nt = 13  # number of turns
    n = int(1e4)

    thetas = np.linspace(0.0, 2.0 * np.pi * nt, n, endpoint=True)

    x1 = np.cos(thetas) * r
    y1 = np.sin(thetas) * r
    z1 = np.linspace(z - h / 2, z + h / 2, n)

    xyz1 = np.vstack((x1, y1, z1))

    self_inductance_piecewise_linear = cfsem.self_inductance_piecewise_linear_filaments(
        xyz1
    )  # [H]

    self_inductance_lyle6 = cfsem.self_inductance_lyle6(r, w, h, nt)  # [H]

    assert self_inductance_piecewise_linear == approx(self_inductance_lyle6, rel=5e-2)


@mark.parametrize("r1", [0.5, np.pi])
@mark.parametrize("r2", [0.1, np.pi / 10.0])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("par", [True, False])
def test_mutual_inductance_piecewise_linear_filaments(r1, r2, z, par):
    # Test against calc for mutual inductance of circular filaments
    rzn1 = np.array([[r1], [z], [1.0]])
    rzn2 = np.array([[r2], [-z / np.e], [1.0]])

    m_circular = cfsem.mutual_inductance_of_circular_filaments(rzn1, rzn2, par)

    n = 100

    thetas = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)

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
@mark.parametrize("par", [True, False])
def test_biot_savart_against_flux_density_ideal_solenoid(r, par):
    # Check Biot-Savart calc against ideal solenoid calc
    length = 20.0 * r  # [m]
    num_turns = 7  # [#]
    current = np.e  # [A]

    # Ideal calc
    b_ideal = cfsem.flux_density_ideal_solenoid(current, num_turns, length)  # [T]

    # Biot-Savart calc should produce the same magnitude
    #   Build a spiral coil
    n_filaments = int(1e4)
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
    bx, _by, _bz = cfsem.flux_density_biot_savart(
        xyzp=(zero, zero, zero), xyzfil=xyzfil, dlxyzfil=dlxyzfil, ifil=ifil, par=par
    )
    b_bs = bx[0]  # [T] First and only element on the axis of the solenoid

    assert b_bs == approx(b_ideal, rel=1e-2)


@mark.parametrize("r", [0.775, np.pi])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("par", [True, False])
def test_biot_savart_against_flux_density_circular_filament(r, z, par):
    # Note we are mapping between (x, y, z) and (r, phi, z) coordinates here

    # Biot-Savart filaments in cartesian coords
    n_filaments = int(1e4)
    phi = np.linspace(0.0, 2.0 * np.pi, n_filaments)
    xfils = r * np.cos(phi)
    yfils = r * np.sin(phi)
    zfils = np.ones_like(xfils) * z

    # Observation grid
    rs = np.linspace(0.01, r - 0.1, 10)
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
    Br_bs, By_bs, Bz_bs = cfsem.flux_density_biot_savart(
        xyzp, xyzfil, dlxyzfil, ifil, par
    )  # [T]

    assert np.allclose(
        Br_circular, Br_bs, rtol=1e-6, atol=1e-7
    )  # Should match circular calc
    assert np.allclose(Bz_circular, Bz_bs, rtol=1e-6, atol=1e-7)  # ...
    assert np.allclose(
        By_bs, np.zeros_like(By_bs), atol=1e-7
    )  # Should sum to zero everywhere


@mark.parametrize("r", [0.775, np.pi])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("par", [True, False])
def test_flux_circular_filament_against_mutual_inductance_of_cylindrical_coils(
    r, z, par
):
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

    Br, Bz = cfsem.flux_density_circular_filament(
        ifil, rfil, zfil, rprime, zprime, par
    )  # [T]

    # We can also get B from the derivative of the flux function (Wesson eqn 3.2.2),
    # so we'll use that to check that we get the same result.
    # Wesson uses flux per radian (as opposed to our total flux), so we have to adjust out a factor
    # of 2*pi in the conversion from flux to B-field. This makes sense because we are converting
    # between the _integral_ of B (psi) and B itself, so we should see a factor related to the
    # space we integrated over to get psi.

    dr = 1e-4
    dz = 1e-4
    psi = cfsem.flux_circular_filament(ifil, rfil, zfil, rprime, zprime, par)
    dpsidz = (
        cfsem.flux_circular_filament(ifil, rfil, zfil, rprime, zprime + dz, par) - psi
    ) / dz
    dpsidr = (
        cfsem.flux_circular_filament(ifil, rfil, zfil, rprime + dr, zprime, par) - psi
    ) / dr

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
    _, bz_origin = cfsem.flux_density_circular_filament(
        ifil, rfil, zfil, np.zeros(1), np.zeros(1), par
    )

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
    _, bz_origin = cfsem.flux_density_circular_filament(
        ifil, rfil, zfil, np.zeros(1), np.zeros(1), par
    )

    assert np.allclose(np.array([b_ideal]), bz_origin, rtol=1e-6)


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
    for i, x in enumerate(zip(rprime, zprime)):
        robs, zobs = x
        Br_num[i], Bz_num[i] = _test._flux_density_circular_filament_numerical(
            current, a, robs, zobs - z, n=100
        )

    assert np.allclose(Br, Br_num)
    assert np.allclose(Bz, Bz_num)


@mark.parametrize("par", [True, False])
def test_self_inductance_lyle6_against_filamentization_and_distributed(par):
    # Test that the Lyle approximation gives a similar result to
    # a case done by brute-force filamentization w/ a heuristic for self-inductance of a loop
    r, z, dr, dz, nt, nr, nz = (0.8, 0.0, 0.5, 2.0, 3.0, 20, 20)
    L_Lyle = cfsem.self_inductance_lyle6(
        r, dr, dz, nt
    )  # Estimate self-inductance via closed-form approximation
    L_fil = _test._self_inductance_filamentized(
        r, z, dr, dz, nt, nr, nz
    )  # Estimate self-inductance via discretization

    # Set up distributed-conductor solve
    fils = cfsem.filament_coil(r, z, dr, dz, nt, nr, nz)
    rfil, zfil, _ = fils.T
    current = np.ones_like(rfil) / rfil.size  # [A] 1A total reference current
    rgrid = np.arange(0.5, 2.0, 0.05)
    zgrid = np.arange(-3.0, 3.0, 0.05)
    rmesh, zmesh = np.meshgrid(rgrid, zgrid, indexing="ij")
    #  Do filamentized psi and B calcs for convenience,
    #  although ideally we'd do a grad-shafranov solve here for a smoother field
    psi = cfsem.flux_circular_filament(
        current, rfil, zfil, rmesh.flatten(), zmesh.flatten(), par
    )
    psi = psi.reshape(rmesh.shape)
    br, bz = cfsem.flux_density_circular_filament(
        current, rfil, zfil, rmesh.flatten(), zmesh.flatten(), par
    )
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

    # Require 5% accuracy (seat of the pants, since we're comparing approximations)
    assert L_Lyle == approx(L_fil, 0.05)
    assert (nt**2 * L_distributed) == approx(L_fil, 0.05)


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
    assert L_1 == approx(L_ref_1)

    major_radius_2 = 25e-2
    minor_radius_2 = 0.5e-2
    L_ref_2 = 424.1761 * np.pi * 1e-7 * 1e-2  # units: henry
    L_2 = cfsem.self_inductance_circular_ring_wien(major_radius_2, minor_radius_2)
    assert L_2 == approx(L_ref_2)


@mark.parametrize("r", [0.775, 1.5])
@mark.parametrize("z", [0.0, np.pi])
@mark.parametrize("dr_over_r", [0.1, 0.2])
@mark.parametrize("dz_over_r", [0.1, 3.5])
@mark.parametrize("nt", [3.0, 400.0])
def test_self_inductance_lyle6_against_filamentized(
    r, z, dr_over_r, dz_over_r, nt
):
    # Test that the Lyle approximation gives a similar result to
    # a case done by brute-force filamentization w/ a heuristic for self-inductance of a loop
    r, z, dr, dz, nt, nr, nz = (
        r,
        z,
        r * dr_over_r,
        r * dz_over_r,
        nt,
        5,
        100,
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
def test_self_inductance_annular_ring(major_radius, a, b):
    # First, test a near-solid version against Wien for a solid loop
    major_radius_1 = major_radius
    minor_radius_1 = b
    inner_minor_radius_1 = 1e-4

    L_wien_1 = cfsem.self_inductance_circular_ring_wien(major_radius_1, minor_radius_1)
    L_annular_1 = cfsem.self_inductance_annular_ring(
        major_radius_1, inner_minor_radius_1, minor_radius_1
    )

    assert L_annular_1 == approx(L_wien_1, rel=1e-2)

    # Then, test thick hollow version against filamentization
    major_radius_2 = major_radius
    minor_radius_2 = b
    inner_minor_radius_2 = a

    L_annular_2 = cfsem.self_inductance_annular_ring(
        major_radius_2, inner_minor_radius_2, minor_radius_2
    )

    n = 100
    rs = np.linspace(
        major_radius_2 - minor_radius_2,
        major_radius_2 + minor_radius_2,
        n,
        endpoint=True,
    )

    zs = np.linspace(
        -minor_radius_2,
        minor_radius_2,
        n,
        endpoint=True,
    )

    rmesh, zmesh = np.meshgrid(rs, zs, indexing="ij")
    mask = np.ones_like(rmesh)
    mask *= np.where(
        np.sqrt(zmesh**2 + (rmesh - major_radius_2) ** 2) <= minor_radius_2, True, False
    )
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
        nr=10,
        nz=10,
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

    psi = cfsem.flux_circular_filament(
        ifil, rfil, zfil, rmesh.flatten(), zmesh.flatten(), par
    )
    a_phi = cfsem.vector_potential_circular_filament(
        ifil, rfil, zfil, rmesh.flatten(), zmesh.flatten(), par
    )

    # Integrate vector potential around a loop to get the flux
    psi_from_a = 2.0 * np.pi * rmesh.flatten() * a_phi  # [Wb]

    # We should not be able to tell the difference above a single roundoff
    assert np.allclose(psi, psi_from_a, rtol=1e-16, atol=1e-16)


@mark.parametrize("r", [0.775, np.pi])
@mark.parametrize("z", [0.0, np.e / 2, -np.e / 2])
@mark.parametrize("par", [True, False])
def test_vector_potential_linear_against_circular_filament(r, z, par):
    # Note we are mapping between (x, y, z) and (r, phi, z) coordinates here

    # Biot-Savart filaments in cartesian coords
    n_filaments = int(1e4)
    phi = np.linspace(0.0, 2.0 * np.pi, n_filaments)
    xfils = r * np.cos(phi)
    yfils = r * np.sin(phi)
    zfils = np.ones_like(xfils) * z

    # Observation grid
    rs = np.linspace(0.01, r - 0.1, 10)
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
    ax, ay, az = cfsem.vector_potential_linear_filament(
        xyzp, xyzfil, dlxyzfil, ifil, par
    )  # [V-s/m]

    assert np.allclose(a_phi, ay, rtol=1e-12, atol=1e-12)  # Should match circular calc
    assert np.allclose(
        az, np.zeros_like(az), atol=1e-9
    )  # Should sum to zero everywhere
    assert np.allclose(ax, np.zeros_like(ax), atol=1e-9)  # ...
