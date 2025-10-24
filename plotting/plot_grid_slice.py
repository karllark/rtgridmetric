import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("basename", help="base name for the grid")
    parser.add_argument("--xval", default=0.0, type=float, help="x slice value")
    parser.add_argument("--zval", default=-2.5, type=float, help="z slice value")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    xval = args.xval
    zval = args.zval

    posfile = fits.open(f"{args.basename}_pos.fits")
    taufile = fits.open(f"{args.basename}_tau_ref_per_pc.fits")

    fontsize = 14
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=2)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    figsize = (14, 7)
    fig, axs = plt.subplots(ncols=2, figsize=figsize)

    parent_grid = True
    cur_ptype = "k-"
    cur_alpha = 1.0
    for hpos, htau in zip(posfile, taufile):
        pos = hpos.data
        # transpose to translate between C++ and python
        tau = np.transpose(htau.data)

        xsize = tau.shape[0]
        ysize = tau.shape[1]
        zsize = tau.shape[2]

        # xy slice
        if (zval >= np.min(pos[2, 0 : zsize + 1])) & (zval < np.max(pos[2, 0 : zsize + 1])):
            ax = axs[0]
            yvals = [np.min(pos[1, 0 : ysize + 1]), np.max(pos[1, 0 : ysize + 1])]
            for i in range(xsize + 1):
                xvals = pos[0, i] * np.array([1.0, 1.0])
                ax.plot(xvals, yvals, cur_ptype, alpha=cur_alpha)

            xvals = [np.min(pos[0, 0 : xsize + 1]), np.max(pos[0, 0 : xsize + 1])]
            for j in range(ysize + 1):
                yvals = pos[1, j] * np.array([1.0, 1.0])
                ax.plot(xvals, yvals, cur_ptype, alpha=cur_alpha)

        if parent_grid:
            ax.set_xlim([np.min(pos[0, 0 : xsize + 1]), np.max(pos[0, 0 : xsize + 1])])
            ax.set_ylim([np.min(pos[1, 0 : ysize + 1]), np.max(pos[1, 0 : ysize + 1])])
            ax.set_xlabel("x [pc]")
            ax.set_ylabel("y [pc]")
            ax.set_title(f"z = {zval:.2f} pc")

        # yz slice
        if (xval >= np.min(pos[0, 0 : xsize + 1])) & (xval < np.max(pos[0, 0 : xsize + 1])):
            ax = axs[1]
            yvals = [np.min(pos[1, 0 : ysize + 1]), np.max(pos[1, 0 : ysize + 1])]
            for i in range(zsize + 1):
                zvals = pos[2, i] * np.array([1.0, 1.0])
                ax.plot(zvals, yvals, cur_ptype, alpha=cur_alpha)

            zvals = [np.min(pos[2, 0 : zsize + 1]), np.max(pos[2, 0 : zsize + 1])]
            for j in range(ysize + 1):
                yvals = pos[1, j] * np.array([1.0, 1.0])
                ax.plot(zvals, yvals, cur_ptype, alpha=cur_alpha)

        if parent_grid:
            ax.set_xlim([np.min(pos[2, 0 : zsize + 1]), np.max(pos[2, 0 : zsize + 1])])
            ax.set_ylim([np.min(pos[1, 0 : ysize + 1]), np.max(pos[1, 0 : ysize + 1])])
            ax.set_xlabel("z [pc]")
            ax.set_title(f"x = {xval:.2f} pc")
            parent_grid = False
            cur_ptype = "k-"
            cur_alpha = 0.5

    fig.tight_layout()

    posfile.close()
    taufile.close()

    save_str = f"{args.basename}_slices_x{xval:.2f}_z{zval:.2f}"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
