import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

from astropy.io import fits

from plot_grid_slice import plot_grid_slice

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("basename", help="base file name with radiation field")
    parser.add_argument(
        "--targ_threshold", default=0.95, type=float, help="target fractional threshold"
    )
    parser.add_argument("--xval", default=0.0, type=float, help="x slice value")
    parser.add_argument("--zval", default=-3.5, type=float, help="z slice value")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    rd = fits.getdata(f"{args.basename}_rad_field.fits")
    pos = fits.getdata(f"{args.basename}_pos.fits")
    # remove the last big x cell that has no dust, just the star
    rd = rd[0, 0:-1, :, :]

    fname_unc = f"{args.basename}_rad_field_unc.fits"
    if os.path.isfile(fname_unc):
        unc = fits.getdata(fname_unc)
        unc = unc[0, 0:-1, :, :]
    else:
        unc = None

    fontsize = 14
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=2)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    figsize = (7, 8)

    fig = plt.figure(layout="constrained", figsize=figsize)

    gs = GridSpec(2, 2, figure=fig, width_ratios=[10, 3], height_ratios=[1.5, 1])
    ax1 = fig.add_subplot(gs[1, :])
    ax2 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    # ax4 = fig.add_subplot(gs[1, 2])

    plot_grid_slice(ax2, args.basename, "xy", args.zval, lines=True)

    plot_grid_slice(ax3, args.basename, "yz", args.xval, lines=True)

    ax = ax1

    # use gradient function
    grad = np.gradient(rd)

    klabel = ["Z", "Y", "X"]
    colors = ["tab:purple", "b", "g"]
    lines = ["-.", ":", "--"]
    for i in range(3):
        gvals = grad[i] != 0.0
        histo = np.histogram(np.absolute(grad[i][gvals]) / rd[gvals], 100)
        midvals = 0.5 * (histo[1][1:] + histo[1][0:-1])
        histvals = np.array(histo[0], dtype=float) / np.sum(histo[0])

        # determine cumulative sum
        csum = np.cumulative_sum(histvals, dtype=float)
        csum /= csum[-1]
        thresval = np.interp(args.targ_threshold, csum, midvals)
        ax1.plot(
            [thresval, thresval],
            [0.0, 0.05],
            color="k",
            alpha=0.5,
            linestyle=lines[i],
        )

        ax1.plot(
            midvals,
            histvals,
            label=rf"{klabel[i]}; $f_t = {thresval:.3f}$",
            color=colors[i],
            alpha=0.5,
            linestyle=lines[i],
        )

    # compute D_A as defined in the paper
    # direction independent measure
    DA = np.zeros(rd.shape)
    for i in range(rd.shape[0]):
        for j in range(rd.shape[1]):
            for k in range(rd.shape[2]):
                nDA = 0
                if i > 0:
                    DA[i, j, k] += abs(rd[i, j, k] - rd[i - 1, j, k]) / rd[i, j, k]
                    nDA += 1
                if i < rd.shape[0] - 1:
                    DA[i, j, k] += abs(rd[i, j, k] - rd[i + 1, j, k]) / rd[i, j, k]
                    nDA += 1
                if j > 0:
                    DA[i, j, k] += abs(rd[i, j, k] - rd[i, j - 1, k]) / rd[i, j, k]
                    nDA += 1
                if j < rd.shape[1] - 1:
                    DA[i, j, k] += abs(rd[i, j, k] - rd[i, j + 1, k]) / rd[i, j, k]
                    nDA += 1
                if k > 0:
                    DA[i, j, k] += abs(rd[i, j, k] - rd[i, j, k - 1]) / rd[i, j, k]
                    nDA += 1
                if k < rd.shape[2] - 1:
                    DA[i, j, k] += abs(rd[i, j, k] - rd[i, j, k + 1]) / rd[i, j, k]
                    nDA += 1
                DA[i, j, k] /= nDA

    histo = np.histogram(DA, 100)
    midvals = 0.5 * (histo[1][1:] + histo[1][0:-1])
    histvals = np.array(histo[0], dtype=float) / np.sum(histo[0])

    # determine cumulative sum
    csum = np.cumulative_sum(histvals, dtype=float)
    csum /= csum[-1]
    thresval = np.interp(args.targ_threshold, csum, midvals)
    ax1.plot(
        [thresval, thresval], [0.0, 0.05], color="k", alpha=0.5, linestyle="-"
    )

    ax1.plot(
        midvals,
        histvals,
        label=rf"DA; $f_t = {thresval:.3f}$",
        color="tab:olive",
        alpha=0.5,
        linestyle="-",
    )

    unc = None
    if unc is not None:
        gvals = unc != 0.0
        histo = np.histogram(unc[gvals] / rd[gvals], 100)
        ax1.plot(
            0.5 * (histo[1][1:] + histo[1][0:-1]),
            histo[0],
            label=f"Uncs",
            color=colors[3],
            linestyle="--",
            alpha=0.7,
        )

    ax1.set_xlabel("fractional change between cells")
    ax1.set_ylabel("cell fraction")

    ax1.legend(fontsize=0.8 * fontsize)

    save_str = f"{args.basename}_radfield_diffs_x{args.xval:.2f}_z{args.zval:.2f}"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
