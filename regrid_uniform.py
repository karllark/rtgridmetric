import argparse
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from astropy.io import fits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="file name with radiation field")
    args = parser.parse_args()

    fname = args.filename
    hdul = fits.open(fname)

    subsize = np.full(3, 1)
    for chdu in hdul[1:]:
        for l in range(3):
            if chdu.data.shape[l+1] > subsize[l]:
                subsize[l] = chdu.data.shape[l+1]

    # create a new 3D grid that has enough pixels in all dim
    mdata = hdul[0].data[:, 0:-1, :, :]
    nrd = np.zeros((1,
                    mdata.shape[1]*subsize[0],
                    mdata.shape[2]*subsize[1],
                    mdata.shape[3]*subsize[2]))

    # get the subgrids and insert them into new grid
    for i in range(mdata.shape[1]):
        print("nz:", i)
        nx = np.linspace(i*subsize[0], (i+1)*subsize[0]-1, subsize[0], dtype=int)
        for j in range(mdata.shape[2]):
            ny = np.linspace(j*subsize[1], (j+1)*subsize[1]-1, subsize[1], dtype=int)
            for k in range(mdata.shape[3]):
                nz = np.linspace(k*subsize[2], (k+1)*subsize[2]-1, subsize[2], dtype=int)
                if mdata[0, i, j, k] < 0:
                    chdu = hdul[int(-1*mdata[0, i, j, k])]
                    ishape = np.array(chdu.shape[1:])
                    cdata = chdu.data[0, :, :, :]
                    for l in range(3):
                        if ishape[l] <= 1:
                            ishape[l] += 1
                            cdata = np.repeat(cdata, 2, axis=l)
                    cx = np.linspace(i*subsize[0], ((i+1)*subsize[0])-1, ishape[0])
                    cy = np.linspace(j*subsize[1], ((j+1)*subsize[1])-1, ishape[1])
                    cz = np.linspace(k*subsize[2], ((k+1)*subsize[2])-1, ishape[2])
                    interp = RegularGridInterpolator((cx, cy, cz), cdata)
                    for cnx in nx:
                        for cny in ny:
                            for cnz in nz:
                                nrd[0, cnx, cny, cnz] = interp([cnx, cny, cnz])

    hdul.close()

    fits.writeto(args.filename.replace(".fits", "_uniform.fits"), nrd, overwrite=True)
    fits.writeto(args.filename.replace(".fits", "_uniform_transpose.fits"), np.transpose(nrd[0, :, : :]), overwrite=True)