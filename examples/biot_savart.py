from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import time

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
    t0 = time.perf_counter()
    bx, by, bz = cfsem.flux_density_linear_filament(
        xyzp, xyzfil, dlxyzfil, ifil, wire_radius=wire_radius, par=True
    )
    t_linear = time.perf_counter() - t0

    bmag = np.sqrt(bx * bx + by * by + bz * bz).reshape(xx.shape)
    bmag_log10 = np.log10(bmag + 1e-30)

    # Discretize into point-segment sources for comparison.
    nseg = 1000
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

    t0 = time.perf_counter()
    bx_ps, by_ps, bz_ps = cfsem.flux_density_point_segment(
        xyzp, xyzfil_ps, dlxyzfil_ps, ifil_ps, par=True
    )
    t_point = time.perf_counter() - t0
    bmag_ps = np.sqrt(bx_ps * bx_ps + by_ps * by_ps + bz_ps * bz_ps).reshape(xx.shape)

    fig, axs = plt.subplots(
        4,
        3,
        figsize=(12, 8),
        dpi=120,
        gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]},
    )
    ax_map, ax_line_x, ax_line_z = axs[0]
    ax_err_map, ax_err_x, ax_err_z = axs[1]
    ax_mlfmm_map, ax_mlfmm_x, ax_mlfmm_z = axs[2]
    ax_vl_map, ax_vl_x, ax_vl_z = axs[3]
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
    ax_line_x.set_ylim(0.0, np.max(bmag[mid_idx, :]))
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
    mask = np.abs(xx) < wire_radius
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
    ax_err_map.set_title("Point-segment error magnitude (log10)")
    cbar_err = fig.colorbar(im_err, ax=ax_err_map)
    cbar_err.set_label("log10(|ΔB|) [T]")

    ax_err_x.plot(x, err[mid_idx, :], color="black")
    ax_err_x.set_xlabel("x [m]")
    ax_err_x.set_ylabel("|ΔB| [T]")
    ax_err_x.set_title("Point-segment error slice along x (z = 0)")
    ax_err_x.grid(True, alpha=0.3)

    ax_err_z.set_axis_off()

    try:
        t0 = time.perf_counter()
        b_mlfmm, _ = cfsem.fields_linear_filament_mlfmm(
            xyzp,
            xyzfil_ps,
            dlxyzfil_ps,
            ifil_ps,
            np.full(nseg, 1e-6),
            use_van_lanen=False,
            direct_threshold=1,  # Always MLFMM to test multipole expansion
            order=None,
        )
        t_mlfmm = time.perf_counter() - t0
        bx_m, by_m, bz_m = b_mlfmm
        bmag_m = np.sqrt(bx_m * bx_m + by_m * by_m + bz_m * bz_m).reshape(xx.shape)
        err_m = np.abs(bmag_m - bmag_ps)
        err_m = np.where(mask, np.nan, err_m)
        err_m_log10 = np.log10(err_m)

        im_m = ax_mlfmm_map.imshow(
            err_m_log10,
            extent=(x.min(), x.max(), z.min(), z.max()),
            origin="lower",
            cmap="viridis",
            aspect="equal",
        )
        ax_mlfmm_map.set_xlabel("x [m]")
        ax_mlfmm_map.set_ylabel("z [m]")
        ax_mlfmm_map.set_title("MLFMM error vs point segment (log10)")
        cbar_m = fig.colorbar(im_m, ax=ax_mlfmm_map)
        cbar_m.set_label("log10(|ΔB|) [T]")

        ax_mlfmm_x.plot(x, err_m[mid_idx, :], color="black")
        ax_mlfmm_x.set_xlabel("x [m]")
        ax_mlfmm_x.set_ylabel("|ΔB| [T]")
        ax_mlfmm_x.set_title("MLFMM error slice along x (z = 0)")
        ax_mlfmm_x.grid(True, alpha=0.3)

        ax_line_z.plot(
            z,
            bmag_m[:, mid_idx],
            color="lime",
            linestyle=":",
            label="mlfmm",
        )
        ax_line_z.legend(frameon=False)

        ax_mlfmm_z.set_axis_off()

        t0 = time.perf_counter()
        b_mlfmm_vl, _ = cfsem.fields_linear_filament_mlfmm(
            xyzp,
            xyzfil,
            dlxyzfil,
            ifil,
            np.full(ifil.size, 1e-6),
            use_van_lanen=True,
            direct_threshold=1,  # Still direct method in this case
            order=None,
        )
        t_mlfmm_vl = time.perf_counter() - t0
        bx_vl, by_vl, bz_vl = b_mlfmm_vl
        bmag_vl = np.sqrt(bx_vl * bx_vl + by_vl * by_vl + bz_vl * bz_vl).reshape(xx.shape)
        err_vl = np.abs(bmag_vl - bmag)
        # err_vl = np.where(mask, np.nan, err_vl)
        err_vl_log10 = np.log10(err_vl)

        im_vl = ax_vl_map.imshow(
            err_vl_log10,
            extent=(x.min(), x.max(), z.min(), z.max()),
            origin="lower",
            cmap="viridis",
            aspect="equal",
        )
        ax_vl_map.set_xlabel("x [m]")
        ax_vl_map.set_ylabel("z [m]")
        ax_vl_map.set_title("MLFMM (van Lanen direct) error vs linear (log10)")
        cbar_vl = fig.colorbar(im_vl, ax=ax_vl_map)
        cbar_vl.set_label("log10(|ΔB|) [T]")

        ax_vl_x.plot(x, err_vl[mid_idx, :], color="black")
        ax_vl_x.set_xlabel("x [m]")
        ax_vl_x.set_ylabel("|ΔB| [T]")
        ax_vl_x.set_title("Van Lanen error slice along x (z = 0)")
        ax_vl_x.grid(True, alpha=0.3)

        ax_vl_z.plot(z, err_vl[:, mid_idx], color="black")
        ax_vl_z.set_xlabel("z [m]")
        ax_vl_z.set_ylabel("|ΔB| [T]")
        ax_vl_z.set_title("Van Lanen error slice along z (x = 0)")
        ax_vl_z.grid(True, alpha=0.3)

        n_targets = xyzp[0].size
        n_linear = ifil.size * n_targets
        n_point = ifil_ps.size * n_targets
        n_mlfmm = ifil_ps.size * n_targets
        n_mlfmm_vl = ifil.size * n_targets
        print(f"Linear filament:            {t_linear:.3f} s ({n_linear:1e} interactions)")
        print(f"Point segment:              {t_point:.3f} s ({n_point:1e} interactions)")
        print(f"MLFMM (point-segment exp.): {t_mlfmm:.3f} s ({n_mlfmm:1e} interactions)")
        print(f"MLFMM (VL direct):          {t_mlfmm_vl:.3f} s ({n_mlfmm_vl:1e} interactions)")
    except RuntimeError:
        n_targets = xyzp[0].size
        n_linear = ifil.size * n_targets
        n_point = ifil_ps.size * n_targets
        print(f"Linear filament: {t_linear:.3f} s ({n_linear:1e} interactions)")
        print(f"Point segment:   {t_point:.3f} s ({n_point:1e} interactions)")
        for ax in (ax_mlfmm_map, ax_mlfmm_x, ax_mlfmm_z, ax_vl_map, ax_vl_x, ax_vl_z):
            ax.text(0.5, 0.5, "MLFMM not available", ha="center", va="center")
            ax.set_axis_off()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
