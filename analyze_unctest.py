import numpy as np
from astropy.io import fits

if __name__ == "__main__":
    basefile = "output_unctest/dirty_gridmetric_slab_tau1_10_10_10"
    # get the radiation fields from 100 runs and compute an empirical unc
    nfiles = 100
    for i in range(nfiles):
        cfile = f"{basefile}_n{i+1}_rad_field.fits"
        cdata = fits.getdata(cfile)
        if i == 0:
            adata = np.zeros((cdata.shape[1], cdata.shape[2], cdata.shape[3], nfiles))
        adata[:, :, :, i] = cdata[0, :, :, :]

    # compute the std in each cell
    dataunc = np.std(adata, axis=3)

    fits.writeto(f"{basefile}_rad_field_empirunc.fits", dataunc)