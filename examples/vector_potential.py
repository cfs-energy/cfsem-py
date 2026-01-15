from __future__ import annotations

import os
import time

import numpy as np

if os.getenv("CFSEM_TESTING"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt

import cfsem


def main() -> None:
    # Equilateral triangle loop in the x-z plane centered at the origin.
    side_length = 1.0
    height = np.sqrt(3.0) * 0.5 * side_length
    v0 = np.array([0.0, 0.0, 2.0 * height / 3.0])
    v1 = np.array([-0.5 * side_length, 0.0, -height / 3.0])
    v2 = np.array([0.5 * side_length, 0.0, -height / 3.0])
    vertices = (v0, v1, v2)

    xyzfil = (
        np.array([v0[0], v1[0], v2[0]]),
        np.array([v0[1], v1[1], v2[1]]),
        np.array([v0[2], v1[2], v2[2]]),
    )
    dlxyzfil = (
        np.array([v1[0] - v0[0], v2[0] - v1[0], v0[0] - v2[0]]),
        np.array([v1[1] - v0[1], v2[1] - v1[1], v0[1] - v2[1]]),
        np.array([v1[2] - v0[2], v2[2] - v1[2], v0[2] - v2[2]]),
    )
    ifil = np.full(3, 1.0)

    # Sample plane: x-z plane at y=0 to show end effects.
    n = 21 if os.getenv("CFSEM_TESTING") else 2001
    x = np.linspace(-1.0, 1.0, n)
    z = np.linspace(-1.0, 1.0, n)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    yy = np.zeros_like(xx)
    xyzp = (xx.ravel(), yy.ravel(), zz.ravel())

    wire_radius = 0.01
    t0 = time.perf_counter()
    ax, ay, az = cfsem.vector_potential_linear_filament(
        xyzp, xyzfil, dlxyzfil, ifil, wire_radius=wire_radius, par=True
    )
    t_linear = time.perf_counter() - t0
    amag = np.sqrt(ax * ax + ay * ay + az * az).reshape(xx.shape)
    amag_log10 = np.log10(amag + 1e-30)

    # Discretize into point-segment sources for comparison.
    nseg = 600
    xfil_ps = []
    yfil_ps = []
    zfil_ps = []
    dlx_ps = []
    dly_ps = []
    dlz_ps = []
    ifil_ps = []

    for start, dl in zip(vertices, zip(*dlxyzfil)):
        dl = np.array(dl)
        dseg = dl / nseg
        seg_starts = start + dseg * np.arange(nseg)[:, None]
        xfil_ps.append(seg_starts[:, 0])
        yfil_ps.append(seg_starts[:, 1])
        zfil_ps.append(seg_starts[:, 2])
        dlx_ps.append(np.full(nseg, dseg[0]))
        dly_ps.append(np.full(nseg, dseg[1]))
        dlz_ps.append(np.full(nseg, dseg[2]))
        ifil_ps.append(np.full(nseg, ifil[0]))

    xyzfil_ps = (
        np.concatenate(xfil_ps),
        np.concatenate(yfil_ps),
        np.concatenate(zfil_ps),
    )
    dlxyzfil_ps = (
        np.concatenate(dlx_ps),
        np.concatenate(dly_ps),
        np.concatenate(dlz_ps),
    )
    ifil_ps = np.concatenate(ifil_ps)

    t0 = time.perf_counter()
    ax_ps, ay_ps, az_ps = cfsem.vector_potential_point_segment(
        xyzp, xyzfil_ps, dlxyzfil_ps, ifil_ps, par=True
    )
    t_point = time.perf_counter() - t0
    amag_ps = np.sqrt(ax_ps * ax_ps + ay_ps * ay_ps + az_ps * az_ps).reshape(xx.shape)

    fig, axs = plt.subplots(
        2,
        3,
        figsize=(12, 6),
        dpi=120,
        gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]},
    )
    ax_map, ax_line_x, ax_line_z = axs[0]
    ax_err_map, ax_err_x, ax_err_z = axs[1]

    im = ax_map.imshow(
        amag_log10,
        extent=(x.min(), x.max(), z.min(), z.max()),
        origin="lower",
        cmap="magma",
        aspect="equal",
    )
    ax_map.contour(
        xx,
        zz,
        amag_log10,
        levels=50,
        colors="k",
        linewidths=0.6,
        alpha=0.6,
    )
    tri_x = [v0[0], v1[0], v2[0], v0[0]]
    tri_z = [v0[2], v1[2], v2[2], v0[2]]
    ax_map.plot(tri_x, tri_z, color="white", linewidth=2.0)
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("z [m]")
    ax_map.set_title("Triangle Loop (linear filaments)\nA-field magnitude (log10)")
    cbar = fig.colorbar(im, ax=ax_map)
    cbar.set_label("log10(|A|) [T m]")

    mid_idx = n // 2
    ax_line_x.plot(x, amag[mid_idx, :], color="black", label="linear")
    ax_line_x.plot(
        x, amag_ps[mid_idx, :], color="cyan", linestyle="--", label="point segment"
    )
    ax_line_x.set_xlabel("x [m]")
    ax_line_x.set_ylabel("|A| [T m]")
    ax_line_x.set_title("Slice along x (z = 0)")
    ax_line_x.set_ylim(0.0, np.max(amag[mid_idx, :]))
    ax_line_x.grid(True, alpha=0.3)
    ax_line_x.legend(frameon=False)

    ax_line_z.plot(z, amag[:, mid_idx], color="black", label="linear")
    ax_line_z.plot(
        z, amag_ps[:, mid_idx], color="cyan", linestyle="--", label="point segment"
    )
    ax_line_z.set_xlabel("z [m]")
    ax_line_z.set_ylabel("|A| [T m]")
    ax_line_z.set_title("Slice along z (x = 0)")
    ax_line_z.grid(True, alpha=0.3)
    ax_line_z.legend(frameon=False)

    err = np.abs(amag - amag_ps)

    def segment_distance(x0: float, z0: float, x1: float, z1: float) -> np.ndarray:
        vx = x1 - x0
        vz = z1 - z0
        denom = vx * vx + vz * vz
        t = ((xx - x0) * vx + (zz - z0) * vz) / denom
        t = np.clip(t, 0.0, 1.0)
        projx = x0 + t * vx
        projz = z0 + t * vz
        return np.sqrt((xx - projx) ** 2 + (zz - projz) ** 2)

    d0 = segment_distance(v0[0], v0[2], v1[0], v1[2])
    d1 = segment_distance(v1[0], v1[2], v2[0], v2[2])
    d2 = segment_distance(v2[0], v2[2], v0[0], v0[2])
    mask = (d0 < wire_radius) | (d1 < wire_radius) | (d2 < wire_radius)
    err = np.where(mask, np.nan, err)
    err_log10 = np.log10(err)

    im_err = ax_err_map.imshow(
        err_log10,
        extent=(x.min(), x.max(), z.min(), z.max()),
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    ax_err_map.set_xlabel("x [m]")
    ax_err_map.set_ylabel("z [m]")
    ax_err_map.set_title("Point-segment discretization\nerror magnitude (log10)")
    cbar_err = fig.colorbar(im_err, ax=ax_err_map)
    cbar_err.set_label("log10(|ΔA|) [T m]")

    rel_err_x = err[mid_idx, :] / (amag[mid_idx, :] + 1e-30)
    ax_err_x.plot(x, rel_err_x, color="black")
    ax_err_x.set_xlabel("x [m]")
    ax_err_x.set_ylabel("relative error")
    ax_err_x.set_title("Point-segment discretization\nrelative error slice along x (z = 0)")
    ax_err_x.grid(True, alpha=0.3)

    ax_err_z.set_axis_off()

    n_targets = xyzp[0].size
    n_linear = ifil.size * n_targets
    n_point = ifil_ps.size * n_targets
    print(f"Linear filament (cfsem direct): {t_linear:.3f} s ({n_linear:.1e} interactions)")
    print(f"Point segment (cfsem direct):   {t_point:.3f} s ({n_point:.1e} interactions)")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
