import numpy as np
from astropy.io import fits

if __name__ == "__main__":
    basefile = "output_unctest/dirty_gridmetric_slab_tau1_10_10_10"

    # empirical unc
    adata = fits.getdata(f"{basefile}_rad_field.fits")
    dataunc = np.std(adata, axis=0, ddof=1)
    fits.writeto(f"{basefile}_rad_field_empirunc.fits", dataunc, overwrite=True)

    # average calcuated unc
    adata = fits.getdata(f"{basefile}_rad_field_unc.fits")
    dataunc = np.average(adata, axis=0)
    fits.writeto(f"{basefile}_rad_field_aveunc.fits", dataunc, overwrite=True)