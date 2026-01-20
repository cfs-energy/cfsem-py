from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import cfsem


def main() -> None:
    # Single filament from z=-0.5 to z=0.5 along the z-axis.
    xyzfil = (np.array([0.0]), np.array([0.0]), np.array([-0.5]))
    dlxyzfil = (np.array([0.0]), np.array([0.0]), np.array([1.0]))
    ifil = np.array([1.0])

    # Sample plane: x-z plane at y=0 to show end effects.
    n = 2001
    x = np.linspace(-1.0, 1.0, n)
    z = np.linspace(-1.0, 1.0, n)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    yy = np.zeros_like(xx)

    xyzp = (xx.ravel(), yy.ravel(), zz.ravel())

    # Use a small wire radius to avoid singularities on-axis.
    wire_radius = 0.01
    bx, by, bz = cfsem.flux_density_linear_filament(
        xyzp, xyzfil, dlxyzfil, ifil, wire_radius=wire_radius, par=True
    )

    bmag = np.sqrt(bx * bx + by * by + bz * bz).reshape(xx.shape)
    bmag_log10 = np.log10(bmag + 1e-30)

    # Discretize into point-segment sources for comparison.
    nseg = 200
    dz = 1.0 / nseg
    xfil_ps = np.zeros(nseg)
    yfil_ps = np.zeros(nseg)
    zfil_ps = np.linspace(-0.5, 0.5 - dz, nseg)
    dlx_ps = np.zeros(nseg)
    dly_ps = np.zeros(nseg)
    dlz_ps = np.full(nseg, dz)
    ifil_ps = np.full(nseg, ifil[0])
    xyzfil_ps = (xfil_ps, yfil_ps, zfil_ps)
    dlxyzfil_ps = (dlx_ps, dly_ps, dlz_ps)

    bx_ps, by_ps, bz_ps = cfsem.flux_density_point_segment(
        xyzp, xyzfil_ps, dlxyzfil_ps, ifil_ps, par=True
    )
    bmag_ps = np.sqrt(bx_ps * bx_ps + by_ps * by_ps + bz_ps * bz_ps).reshape(xx.shape)

    fig, axs = plt.subplots(
        2,
        3,
        figsize=(14, 9),
        dpi=120,
        gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]},
    )
    ax_map, ax_line_x, ax_line_z = axs[0]
    ax_err_map, ax_err_x, ax_err_z = axs[1]
    im = ax_map.imshow(
        bmag_log10,
        extent=(x.min(), x.max(), z.min(), z.max()),
        origin="lower",
        cmap="magma",
        aspect="equal",
    )
    ax_map.contour(
        xx,
        zz,
        bmag_log10,
        levels=50,
        colors="k",
        linewidths=0.6,
        alpha=0.6,
    )

    ax_map.plot([0.0, 0.0], [-0.5, 0.5], color="white", linewidth=2.0)
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("z [m]")
    ax_map.set_title("B-field magnitude (log10)")
    cbar = fig.colorbar(im, ax=ax_map)
    cbar.set_label("log10(|B|) [T]")

    mid_idx = n // 2
    ax_line_x.plot(x, bmag[mid_idx, :], color="black", label="linear")
    ax_line_x.plot(
        x, bmag_ps[mid_idx, :], color="cyan", linestyle="--", label="point segment"
    )
    ax_line_x.set_xlabel("x [m]")
    ax_line_x.set_ylabel("|B| [T]")
    ax_line_x.set_title("Slice along x (z = 0)")
    ax_line_x.grid(True, alpha=0.3)
    ax_line_x.legend(frameon=False)

    ax_line_z.plot(z, bmag[:, mid_idx], color="black", label="linear")
    ax_line_z.plot(
        z, bmag_ps[:, mid_idx], color="cyan", linestyle="--", label="point segment"
    )
    ax_line_z.set_xlabel("z [m]")
    ax_line_z.set_ylabel("|B| [T]")
    ax_line_z.set_title("Slice along z (x = 0)")
    ax_line_z.grid(True, alpha=0.3)
    ax_line_z.legend(frameon=False)

    err = np.abs(bmag - bmag_ps)
    err_log10 = np.log10(err + 1e-30)

    im_err = ax_err_map.imshow(
        err_log10,
        extent=(x.min(), x.max(), z.min(), z.max()),
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    ax_err_map.set_xlabel("x [m]")
    ax_err_map.set_ylabel("z [m]")
    ax_err_map.set_title("Error magnitude (log10)")
    cbar_err = fig.colorbar(im_err, ax=ax_err_map)
    cbar_err.set_label("log10(|ΔB|) [T]")

    ax_err_x.plot(x, err[mid_idx, :], color="black")
    ax_err_x.set_xlabel("x [m]")
    ax_err_x.set_ylabel("|ΔB| [T]")
    ax_err_x.set_title("Error slice along x (z = 0)")
    ax_err_x.grid(True, alpha=0.3)

    ax_err_z.plot(z, err[:, mid_idx], color="black")
    ax_err_z.set_xlabel("z [m]")
    ax_err_z.set_ylabel("|ΔB| [T]")
    ax_err_z.set_title("Error slice along z (x = 0)")
    ax_err_z.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
