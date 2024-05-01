import copy
import argparse
import numpy as np

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

    subgrid_index = 1
    for i in range(rd.shape[0]):
        for j in range(rd.shape[1]):
            for k in range(rd.shape[2]):
                subn = np.full(3, 1)
                for l in range(3):
                    if grad[l][i, j, k] > args.fthres:
                        subn[l] = (grad[l][i, j, k] // args.fthres) + 1
                    else:
                        subn[l] = 1
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