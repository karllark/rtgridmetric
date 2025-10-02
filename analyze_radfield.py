import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt

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
    if os.path.isfile(fname):
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

    figsize = (10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    # use gradient function instead
    grad = np.gradient(rd)

    klabel = ["Z", "Y", "X"]
    colors = ["r", "b", "g", "tab:purple"]
    for i in range(3):
        gvals = grad[i] != 0.0
        histo = np.histogram(grad[i][gvals]/rd[gvals], 100)
        plt.plot(0.5*(histo[1][1:] + histo[1][0:-1]), histo[0], label=klabel[i],
                 color=colors[i], alpha=0.7)

    if unc is not None:
        gvals = unc != 0.0
        histo = np.histogram(unc[gvals]/rd[gvals], 100)
        plt.plot(0.5*(histo[1][1:] + histo[1][0:-1]), histo[0], label=f"Uncs",
                    color=colors[3], linestyle="--", alpha=0.7)

    ax.set_xlabel("fractional change between cells")
    ax.set_ylabel("# cells")
    ax.set_title(fname)

    plt.legend()

    fig.tight_layout()

    save_str = fname.replace(".fits", "_radfield_diffs")
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()