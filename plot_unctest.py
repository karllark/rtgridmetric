import argparse
import matplotlib.pyplot as plt
from astropy.io import fits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bumps", help="save figure as a png file", action="store_true"
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()


    basefile = "output_unctest_CB18/dirty_gridmetric_slab_tau1_10_10_10"

    fontsize = 16
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    figsize = (8, 6)
    fig, ax = plt.subplots(figsize=figsize)

    nphot = ["_1e4", "_1e5", ""]
    for cphot in nphot:
        eunc = fits.getdata(f"{basefile}{cphot}_rad_field_empirunc.fits")
        aunc = fits.getdata(f"{basefile}{cphot}_rad_field_aveunc.fits")

        for k in range(eunc.shape[2] - 1):
            ax.plot(eunc[:, :, k], aunc[:, :, k], "ko", alpha=0.2)

    ax.plot([0.0, 1e-6], [0.0, 1e-6], "k-")

    ax.set_xlabel("Empirical Unc")
    ax.set_ylabel("Camps & Baes (2018) Unc")

    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.tight_layout()

    save_str = "unctest_cb18"
    if args.bumps:
        save_str = f"{save_str}_bumps"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
