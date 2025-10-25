import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

from astropy.io import fits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="file name with radiation field")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fname = args.filename
    rd = fits.getdata(fname)
    # remove the last big x cell that has no dust, just the star
    rd = rd[0, 0:-1, :, :]

    fname_unc = fname.replace(".fits", "_unc.fits")
    if os.path.isfile(fname_unc):
        unc = fits.getdata(fname.replace(".fits", "_unc.fits"))
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

    gs = GridSpec(2, 2, figure=fig, width_ratios=[10, 3], height_ratios=[1, 1.5])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    # ax4 = fig.add_subplot(gs[1, 2])

    vmin = np.min(rd)
    vmax = np.max(rd)

    zslice = rd[6, :, :]
    ax2.set_xlim(-5.0, 5.0)
    ax2.set_xlabel("x [pc]")
    ax2.set_ylim(-5.0, 5.0)
    ax2.set_ylabel("y [pc]")
    zpos = ax2.imshow(
        zslice, norm=LogNorm(vmin=vmin, vmax=vmax), aspect="auto", extent=ax2.axis()
    )

    xslice = np.transpose(rd[:, 5, :])
    print(xslice.shape)
    # xpos = ax3.imshow(xslice) #, interpolation='none')
    ax3.set_xlim(-5.0, -2.0)
    ax3.set_xlabel("z [pc]")
    ax3.set_ylim(-5.0, 5.0)
    # ax3.set_ylabel("y [pc]")
    xpos = ax3.imshow(
        xslice, norm=LogNorm(vmin=vmin, vmax=vmax), aspect="auto", extent=ax3.axis()
    )

    # cb = plt.colorbar(xpos)
    # fig.colorbar(xpos, ax=ax3)

    ax = ax1

    # use gradient function instead
    grad = np.gradient(rd)

    klabel = ["Z", "Y", "X"]
    colors = ["r", "b", "g", "tab:purple"]
    for i in range(3):
        gvals = grad[i] != 0.0
        histo = np.histogram(grad[i][gvals] / rd[gvals], 100)
        ax1.plot(
            0.5 * (histo[1][1:] + histo[1][0:-1]),
            histo[0],
            label=klabel[i],
            color=colors[i],
            alpha=0.7,
        )

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
    ax1.set_ylabel("# cells")
    # ax1.set_title(fname)

    ax1.legend()

    save_str = fname.replace(".fits", "_radfield_diffs")
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
