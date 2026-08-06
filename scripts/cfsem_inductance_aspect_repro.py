"""Reproduce the triangle-aspect-ratio dependence of BEM stored energy.

The narrow 9-by-48 annulus has aspect-ratio-67 triangles and produced about
1029 J before the analytic near-pair fix. The independent filament reference
is about 456 J. Run this script before and after kernel changes to exercise the
full public matrix-assembly path.
"""

from __future__ import annotations

from pathlib import Path

import cfsem
import numpy as np
from scipy import constants
from scipy.special import ellipe, ellipk

MU0 = constants.mu_0
STRIP_GMR = float(np.exp(-1.5))
I_TOT = 16_000.0
NARROW_BAND = (0.5, 0.50883)
WIDE_BAND = (0.4, 0.6)


def mutual_coaxial_coplanar(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Maxwell mutual inductance of coplanar coaxial filament loops [H]."""
    parameter = 4.0 * r1 * r2 / (r1 + r2) ** 2
    modulus = np.sqrt(parameter)
    return MU0 * np.sqrt(r1 * r2) * (
        (2.0 / modulus - modulus) * ellipk(parameter) - (2.0 / modulus) * ellipe(parameter)
    )


def sheet_energy_reference(r_in: float, r_out: float, n_sub: int = 400) -> float:
    """Uniform azimuthal sheet energy from concentric filament loops [J]."""
    dr = (r_out - r_in) / n_sub
    radii = r_in + (np.arange(n_sub) + 0.5) * dr
    current = I_TOT / n_sub
    i, j = np.triu_indices(n_sub, k=1)
    mutual = mutual_coaxial_coplanar(radii[i], radii[j])
    self_inductance = MU0 * radii * (np.log(8.0 * radii / (STRIP_GMR * dr)) - 2.0)
    return float(
        0.5 * current**2 * self_inductance.sum() + current**2 * mutual.sum()
    )


def build_annulus(
    r_in: float, r_out: float, n_rad: int, n_ang: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nodes, triangular faces, and node radii for a flat annulus."""
    radii = np.linspace(r_in, r_out, n_rad + 1)
    angles = np.arange(n_ang) * (2.0 * np.pi / n_ang)
    nodes = np.stack(
        [
            np.outer(radii, np.cos(angles)),
            np.outer(radii, np.sin(angles)),
            np.zeros((n_rad + 1, n_ang)),
        ],
        axis=-1,
    ).reshape(-1, 3)

    def node_id(i: int, j: int) -> int:
        return i * n_ang + (j % n_ang)

    faces = []
    for i in range(n_rad):
        for j in range(n_ang):
            faces.append([node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1)])
            faces.append([node_id(i, j), node_id(i + 1, j + 1), node_id(i, j + 1)])
    return (
        np.ascontiguousarray(nodes, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int64),
        np.repeat(radii, n_ang),
    )


def aspect(r_in: float, r_out: float, n_rad: int, n_ang: int) -> float:
    """Azimuthal arc length divided by radial cell thickness."""
    radius = 0.5 * (r_in + r_out)
    return (2.0 * np.pi * radius / n_ang) / ((r_out - r_in) / n_rad)


def bem_energy(
    band: tuple[float, float], n_rad: int, n_ang: int, quad: str = "dunavant3"
) -> float:
    """Assemble the public BEM operator and contract its stored energy [J]."""
    nodes, faces, node_radius = build_annulus(*band, n_rad, n_ang)
    matrix = cfsem.triangle_mesh_inductance_matrix(nodes, faces, par=True, quad=quad)
    stream = I_TOT * (node_radius - band[0]) / (band[1] - band[0])
    return 0.5 * float(stream @ matrix @ stream)


def sweep(
    band: tuple[float, float], grids: list[tuple[int, int]], reference: float
) -> list[tuple[float, float]]:
    """Print and return aspect/energy-ratio rows."""
    rows = []
    for n_rad, n_ang in grids:
        mesh_aspect = aspect(*band, n_rad, n_ang)
        energy = bem_energy(band, n_rad, n_ang)
        ratio = energy / reference
        rows.append((mesh_aspect, ratio))
        print(
            f"n_rad={n_rad:3d} n_ang={n_ang:4d} aspect={mesh_aspect:7.2f} "
            f"W={energy:9.3f} J ratio={ratio:7.4f}"
        )
    return rows


def main() -> None:
    refs = {
        NARROW_BAND: sheet_energy_reference(*NARROW_BAND),
        WIDE_BAND: sheet_energy_reference(*WIDE_BAND),
    }
    print(f"narrow independent reference: {refs[NARROW_BAND]:.6f} J")
    print(f"wide independent reference:   {refs[WIDE_BAND]:.6f} J")

    print("\nNarrow annulus aspect sweep")
    narrow = sweep(NARROW_BAND, [(9, 48), (9, 96), (9, 192), (9, 384)], refs[NARROW_BAND])
    print("\nWide annulus controls")
    wide = sweep(WIDE_BAND, [(16, 256), (16, 64), (16, 32)], refs[WIDE_BAND])

    print("\nQuadrature-order check at the reported mesh")
    for quadrature in ("dunavant1", "dunavant3", "dunavant5"):
        ratio = bem_energy(NARROW_BAND, 9, 48, quadrature) / refs[NARROW_BAND]
        print(f"{quadrature:10s} ratio={ratio:7.4f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for rows, label in ((narrow, "narrow band"), (wide, "wide band")):
        x, y = zip(*sorted(rows), strict=False)
        axis.plot(x, y, "o-", label=label)
    axis.axhline(1.0, color="0.5", linestyle="--")
    axis.set_xscale("log")
    axis.set_xlabel("triangle aspect ratio")
    axis.set_ylabel("BEM energy / filament reference")
    axis.grid(True, which="both", color="0.9")
    axis.legend(frameon=False)
    output = Path(__file__).with_suffix(".png")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    print(f"\nfigure written to {output}")


if __name__ == "__main__":
    main()
