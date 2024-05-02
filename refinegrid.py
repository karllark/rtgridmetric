import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("basename", help="file name with radiation field")
    parser.add_argument(
        "--fthres",
        help="fractional threshold above which to subdivide",
        default=0.1,
        type=float,
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    rd = fits.getdata(f"{args.basename}_rad_field.fits")
    pos = fits.getdata(f"{args.basename}_pos.fits")
    tau = fits.getdata(f"{args.basename}_tau_ref_per_pc.fits")
    tau_header = fits.getheader(f"{args.basename}_tau_ref_per_pc.fits")
    # focus on a single wavelength and remove z cells w/o any dust
    # latter is specific to slab benchmark case
    rd = rd[0, 0:-1, :, :]
    # transpose to get the axes to orient as they do in C++
    # difference in how FITS stores versus C++
    rd = np.transpose(rd)
    tau = np.transpose(tau)

    # fractional gradient
    grad = np.gradient(rd) / rd

    totcells = 0
    ptau = copy.copy(tau)
    sub_tau = []
    sub_pos = []

    subvals = np.zeros((rd.shape[0], rd.shape[1], rd.shape[2], 3))

    subgrid_index = 1
    for i in range(rd.shape[0]):
        for j in range(rd.shape[1]):
            for k in range(rd.shape[2]):
                subn = np.full(3, 1)
                for l in range(3):
                    if np.absolute(grad[l][i, j, k]) > args.fthres:
                        subn[l] = (np.absolute(grad[l][i, j, k]) // args.fthres) + 1
                    else:
                        subn[l] = 1
                subvals[i, j, k, :] = subn
                # print(i, j, k, subn)
                # setup subdivided cell
                if np.prod(subn) > 1:
                    # positions
                    npos = np.zeros((np.max(subn) + 1, 3))
                    indxs = np.array([i, j, k], dtype=int)
                    for l in range(3):
                        delt = (pos[l, indxs[l] + 1] - pos[l, indxs[l]]) / (subn[l])
                        npos[0:subn[l] + 1, l] = np.linspace(
                            pos[l, indxs[l]], pos[l, indxs[l] + 1], subn[l] + 1
                        )
                    sub_pos.append(npos)
                    # tau_per_pc
                    ntau = np.full(subn, tau[i, j, k])
                    sub_tau.append(ntau)
                    # update primary grid to point to the new subgrid
                    ptau[i, j, k] = -1 * subgrid_index
                    subgrid_index += 1

                totcells += np.prod(subn)

    print(len(sub_pos))
    print(totcells, 80 * 80 * 40, totcells / (80 * 80 * 40))

    # write the new grid files
    obase = args.basename.replace("output", "input")

    # tau per pc
    hdul = fits.HDUList([fits.PrimaryHDU(np.transpose(ptau))])
    hdul[0].header["RAD_TAU"] = tau_header["RAD_TAU"]
    hdul[0].header["GRDDEPTH"] = tau_header["GRDDEPTH"] + 1
    hdul[0].header.comments["GRDDEPTH"] = "maximum depth of the grid"
    hdul[0].header["FTHRES"] = args.fthres
    hdul[0].header.comments["FTHRES"] = "threshold above which to subdivide a cell"
    hdul[0].header["COMMENT"] = "DIRTY model grid definition file (tau)"
    hdul[0].header["COMMENT"] = "created with refinegrid.py"
    hdul[0].header["COMMENT"] = "Grid refined based on the fractional radiation field gradient"
    for ctau in sub_tau:
        hdul.append(fits.ImageHDU(np.transpose(ctau)))
        hdul[-1].header["PAR_GRID"] = 0
    hdul.writeto(f"{obase}_fthres{args.fthres}_tau_ref_per_pc.fits", overwrite=True)

    # positions
    hdul = fits.HDUList([fits.PrimaryHDU(pos)])
    hdul[0].header["FTHRES"] = args.fthres
    hdul[0].header.comments["FTHRES"] = "threshold above which to subdivide a cell"
    hdul[0].header["COMMENT"] = "DIRTY model grid definition file (pos)"
    hdul[0].header["COMMENT"] = "created with refinegrid.py"
    hdul[0].header["COMMENT"] = "Grid refined based on the fractional radiation field gradient"
    for cpos in sub_pos:
        hdul.append(fits.ImageHDU(np.transpose(cpos)))
        hdul[-1].header["PAR_GRID"] = 0
    hdul.writeto(f"{obase}_fthres{args.fthres}_pos.fits", overwrite=True)

    fontsize = 16
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    figsize = (10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(subvals[:, :, :, 0].flatten(), label="X")
    ax.hist(subvals[:, :, :, 1].flatten(), label="Y")
    ax.hist(subvals[:, :, :, 2].flatten(), label="Z")

    plt.legend()

    fig.tight_layout()

    save_str = f"{args.basename}_subdivs"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()