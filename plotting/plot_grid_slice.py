import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from astropy.io import fits


def plot_grid_slice(ax, basename, stype, sval, lines=False):
    """
    Plot a grid slice that handles hierarchical grids

    Parameters
    ----------
    ax : matplotlib axis object
        where shall the plot go

    basename : str
        base filename of run

    stype : str
        type of plot, choices are "xy", or "yz"

    sval : float
        slice value, either z for "xy" or x for "yz"
    """
    posfile = fits.open(f"{basename}_pos.fits")
    rdfile = fits.open(f"{basename}_rad_field.fits")

    # get the min/max of the radiation field
    parent_grid = True
    rdmin = 1e20
    rdmax = 0.0
    for hpos, hrd in zip(posfile, rdfile):
        rd = hrd.data
        rd = np.transpose(hrd.data)
        pos = hpos.data
        if stype == "xy":
            zsize = rd.shape[2]
            if (sval >= np.min(pos[2, 0 : zsize + 1])) & (
                sval < np.max(pos[2, 0 : zsize + 1])
            ):
                k = np.round(
                    zsize * (sval - pos[2, 0]) / (pos[2, -1] - pos[2, 0])
                ).astype(int)
                if k >= zsize:
                    k -= 1
                rd_tmp = rd[:, :, k, 0]
                gvals = rd_tmp > 0.0
                if np.sum(gvals) > 0:
                    rdmin = np.min([np.min(rd_tmp[gvals]), rdmin])
                    rdmax = np.max([np.max(rd_tmp[gvals]), rdmax])
        elif stype == "yz":
            xsize = rd.shape[0]
            if (sval >= np.min(pos[0, 0 : xsize + 1])) & (
                sval < np.max(pos[0, 0 : xsize + 1])
            ):
                if xsize > 1:
                    i = np.round(
                        xsize * (sval - pos[0, 0]) / (pos[0, -1] - pos[0, 0])).astype(int)
                else:
                    i = 0
                if i >= xsize:
                    i -= 1
                rd_tmp = rd[i, :, :, 0]
                gvals = rd_tmp > 0.0
                if np.sum(gvals) > 0:
                    rdmin = np.min([np.min(rd_tmp[gvals]), rdmin])
                    rdmax = np.max([np.max(rd_tmp[gvals]), rdmax])
        else:
            print("stype = {stype} not supported")
            exit()
        parent_grid = False

    cmap = plt.get_cmap("afmhot")

    parent_grid = True
    cur_ptype = "k-"
    cur_alpha = 1.0
    cur_zorder = 2000
    for hpos, hrd in zip(posfile, rdfile):
        pos = hpos.data
        # transpose to translate between C++ and python
        rd = np.transpose(hrd.data)

        xsize = rd.shape[0]
        ysize = rd.shape[1]
        zsize = rd.shape[2]

        # xy slice
        if stype == "xy":
            if (sval >= np.min(pos[2, 0 : zsize + 1])) & (
                sval < np.max(pos[2, 0 : zsize + 1])
            ):

                # color in cell
                k = np.round(
                    zsize * (sval - pos[2, 0]) / (pos[2, -1] - pos[2, 0])
                ).astype(int)
                if k >= zsize:
                    k -= 1

                for i in range(xsize):
                    x1 = pos[0, i]
                    dx = pos[0, i + 1] - pos[0, i]
                    for j in range(ysize):
                        y1 = pos[1, j]
                        dy = pos[1, j + 1] - pos[1, j]
                        if rd[i, j, k, 0] > 0.0:
                            ccol = (rd[i, j, k, 0] - rdmin) / (rdmax - rdmin)
                            rect = Rectangle(
                                (x1, y1),
                                dx,
                                dy,
                                facecolor=cmap(ccol),
                                edgecolor="none",
                                linewidth=2,
                            )
                            ax.add_patch(rect)

                if lines:
                    yvals = [np.min(pos[1, 0 : ysize + 1]), np.max(pos[1, 0 : ysize + 1])]
                    for i in range(xsize + 1):
                        xvals = pos[0, i] * np.array([1.0, 1.0])
                        ax.plot(xvals, yvals, cur_ptype, alpha=cur_alpha, zorder=cur_zorder)

                    xvals = [np.min(pos[0, 0 : xsize + 1]), np.max(pos[0, 0 : xsize + 1])]
                    for j in range(ysize + 1):
                        yvals = pos[1, j] * np.array([1.0, 1.0])
                        ax.plot(xvals, yvals, cur_ptype, alpha=cur_alpha, zorder=cur_zorder)

            if parent_grid:
                ax.set_xlim(
                    [np.min(pos[0, 0 : xsize + 1]), np.max(pos[0, 0 : xsize + 1])]
                )
                ax.set_ylim(
                    [np.min(pos[1, 0 : ysize + 1]), np.max(pos[1, 0 : ysize + 1])]
                )
                ax.set_xlabel("x [pc]")
                ax.set_ylabel("y [pc]")
                ax.set_title(f"z = {sval:.2f} pc")
                cur_ptype = "b-"
                cur_alpha = 0.4
                cur_zorder = 1

        # yz slice
        elif stype == "yz":
            if (sval >= np.min(pos[0, 0 : xsize + 1])) & (
                sval < np.max(pos[0, 0 : xsize + 1])
            ):
                if parent_grid:
                    rd = rd[:, :, 0:-1, :]
                    zsize -= 1

                if xsize > 1:
                    i = np.round(
                        xsize * (sval - pos[0, 0]) / (pos[0, -1] - pos[0, 0])).astype(int)
                else:
                    i = 0
                if i >= xsize:
                    i -= 1

                # color in cell
                for k in range(zsize):
                    z1 = pos[2, k]
                    dz = pos[2, k + 1] - pos[2, k]
                    for j in range(ysize):
                        y1 = pos[1, j]
                        dy = pos[1, j + 1] - pos[1, j]
                        if rd[i, j, k, 0] > 0.0:
                            ccol = (rd[i, j, k, 0] - rdmin) / (rdmax - rdmin)
                            rect = Rectangle(
                                (z1, y1),
                                dz,
                                dy,
                                facecolor=cmap(ccol),
                                edgecolor="none",
                                linewidth=2,
                            )
                            ax.add_patch(rect)

                if lines:
                    yvals = [np.min(pos[1, 0 : ysize + 1]), np.max(pos[1, 0 : ysize + 1])]
                    for i in range(zsize + 1):
                        zvals = pos[2, i] * np.array([1.0, 1.0])
                        ax.plot(zvals, yvals, cur_ptype, alpha=cur_alpha, zorder=cur_zorder)

                    zvals = [np.min(pos[2, 0 : zsize + 1]), np.max(pos[2, 0 : zsize + 1])]
                    for j in range(ysize + 1):
                        yvals = pos[1, j] * np.array([1.0, 1.0])
                        ax.plot(zvals, yvals, cur_ptype, alpha=cur_alpha, zorder=cur_zorder)

            if parent_grid:
                ax.set_xlim(
                    [np.min(pos[2, 0 : zsize + 1]), np.max(pos[2, 0 : zsize + 1])]
                )
                ax.set_ylim(
                    [np.min(pos[1, 0 : ysize + 1]), np.max(pos[1, 0 : ysize + 1])]
                )
                ax.set_xlabel("z [pc]")
                ktitle = rf"x = {sval:.2f} pc"
                ax.set_title(ktitle)
                cur_ptype = "b-"
                cur_alpha = 0.4
                cur_zorder = 1

        parent_grid = False

    posfile.close()
    rdfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("basename", help="base name for the grid")
    parser.add_argument("--xval", default=0.0, type=float, help="x slice value")
    parser.add_argument("--zval", default=-3.5, type=float, help="z slice value")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument(
        "--values", help="show radfield values instead of grid", action="store_true"
    )
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    posfile = fits.open(f"{args.basename}_pos.fits")
    rdfile = fits.open(f"{args.basename}_rad_field.fits")

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

    plot_grid_slice(axs[0], args.basename, "xy", args.zval)
    plot_grid_slice(axs[1], args.basename, "yz", args.xval)

    fig.tight_layout()

    posfile.close()
    rdfile.close()

    save_str = f"{args.basename}_slices_x{args.xval:.2f}_z{args.zval:.2f}"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
