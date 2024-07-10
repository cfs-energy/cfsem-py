import numpy as np
from pytest import mark

import cfsem

from . import test_funcs as _test


@mark.parametrize("r", [0.1, np.pi / 6.0])
@mark.parametrize("nt", [0.5, 13.0])
@mark.parametrize("twist_pitch", [0.1, np.pi / 10.0, 1.0, float("inf")])
@mark.parametrize("angle_offset", [0.0, 0.01, np.pi])
def test_filament_helix_path(r, nt, twist_pitch, angle_offset):
    # Make sure it doesn't produce any obvious error on a fairly complicated 3d path case
    n = 50
    x = np.linspace(0, 2 * np.pi * nt, n)
    y = 0.5**0.5 * np.cos(x)
    z = 0.5**0.5 * np.sin(2 * x)
    path = (x, y, z)
    helix = cfsem.filament_helix_path(
        path=path,
        helix_start_offset=(0.0, r, 0.0),
        twist_pitch=twist_pitch,
        angle_offset=angle_offset,
    )

    helix = np.array(helix)
    path = np.array(path)

    #    Make sure they are all the right distance from the path centerline
    distances = helix - path
    distance_err = np.sqrt(np.sum(distances * distances, axis=0)) - r
    assert np.allclose(distance_err, np.zeros_like(distance_err), atol=1e-3)
    #    Make sure they are all perpendicular to the path centerline
    ds = path[:, 1:] - path[:, :-1]
    for i in range(n - 1):
        assert (
            np.dot(ds[:, i], distances[:, i]) < 1e-3
        ), f"perpendicularity error at index {i}, helix {helix[i]}, path {path[i]}, ds {ds[i]}, radius {distances[i]}"

    # Make sure it produces the correct result for some simple analytic cases
    x = np.linspace(0.0, 1.0, 50)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    path = (x, y, z)

    first_radius = (
        0.01,
        r,
        0.0,
    )  # [m] with a small out-of-plane component to test projection
    helix = cfsem.filament_helix_path(
        path=path,
        helix_start_offset=first_radius,
        twist_pitch=twist_pitch,
        angle_offset=angle_offset,
    )

    yhelix = r * np.cos(2.0 * np.pi * x / twist_pitch + angle_offset)
    zhelix = r * np.sin(2.0 * np.pi * x / twist_pitch + angle_offset)

    helix_handcalc = (x, yhelix, zhelix)

    assert np.allclose(helix, helix_handcalc, rtol=5e-3)

    # Make sure rotate_filaments_about_path produces the same result
    # as building a helix with an angle offset
    helix_without_rotation = cfsem.filament_helix_path(
        path=path,
        helix_start_offset=first_radius,
        twist_pitch=twist_pitch,
        angle_offset=0.0,
    )

    helix_post_rotated = cfsem.rotate_filaments_about_path(
        path=path, angle_offset=angle_offset, fils=helix_without_rotation
    )

    assert np.allclose(helix, helix_post_rotated)


def test_filament_coil():
    r, z, dr, dz, nt, nr, nz = (1.0, 0.0, 0.1, 0.1, 7.0, 2, 2)
    f = cfsem.filament_coil(r, z, dr, dz, nt, nr, nz)
    assert f.shape[0] == 4  # Right number of filaments generated?
    assert f.shape[1] == 3  # Right number of components?
    assert np.sum(f[:, 2]) == nt  # Turns sum to the right number?
    assert all(
        [rzn[0] < r + dr / 2 for rzn in f]
    )  # Are all the filaments inside the coil pack?
    assert all([rzn[0] > r - dr / 2 for rzn in f])
    assert all([rzn[1] < z + dz / 2 for rzn in f])
    assert all([rzn[1] > z - dz / 2 for rzn in f])

    # Compare to earlier version of the function
    f_comprehension = _test._filament_coil_comprehension(r, z, dr, dz, nt, nr, nz)
    assert np.allclose(f, f_comprehension)
