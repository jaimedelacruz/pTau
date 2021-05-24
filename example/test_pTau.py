import numpy as np
import time
import pyTau
from astropy.io import fits
import matplotlib.pyplot as plt

# *********************************************************

def readfits(fname, ext = 0):
    """
    Reads a fits file
    """
    return np.ascontiguousarray(fits.open(fname, 'readonly')[ext].data[:], dtype='float32')

# *********************************************************

if __name__ == "__main__":

    #
    # Let's load a FALC model, which we have repeated in a X,Y grid of 4x4 pixels in CGS units
    #
    m = readfits('modelin.fits')
    
    z = m[0]*1
    Tg = m[1]*1
    vz = m[2]*1
    rho = m[3]*1
    pgas = m[4]*1

    ny, nx, ndep = z.shape
    
    #
    # Let's assume that we have a set of variables Temp, Pgas, z in CGS units
    # The axis ordering is (ny, nx, ndep) where ndep is the fast axis (c-ordering).
    #
    # Note that the upper boundary of the atmosphere must be located at index=0.

    t0 = time.time()
    nthreads = 8 # Use as many as your computer has!
    
    ltau500 = pyTau.getTau(Tg, z, Pg=pgas, nthreads=nthreads, wav = [5000.0])
    
    t1 = time.time()
    print("Pgas case dT = {0}s".format(t1-t0))
    
    #
    # Alternatively we can use rho instead of Pgas but it is almost a factor x2 slower due to the internals of the EOS
    #
    t0 = time.time()
    
    ltau500 = pyTau.getTau(Tg, z, rho=rho, nthreads=nthreads, wav = [5000.0])
    
    t1 = time.time()
    print("Rho case dT = {0}s".format(t1-t0))
    
    
    #
    # Imagine that we also want to get LTE electron densities, we can compute them in parallel
    #

    Ne = pyTau.getNe(Tg, Pg=pgas, nthreads=nthreads)
    


    #
    # We can also compute an optimal depth grid for radiative transfer calculations.
    # Part of this recipe is taken from the old Multi3D (Carlsson 1986), which
    # looks for gradients in Tg, rho and ltau500. Additionally, I have added the possibility
    # of also including gradients along Vz, which are a big source of trouble when
    # solving the NLTE in coarse grids. The output is given in index number of the original grid.
    # Using these tools, we can obtain a very high convergence rate with RH1.5D / STiC.

    nDep2 = ndep    # Number of points in the new interpolated grid
    smooth= 1       # smooth gradients with a top-hat PSF (should be smaller than 5)
    velscal=0.4     # scaling of velocity gradients in km/s (larger number gives less weight)
    Tcut = 25000.   # Temperature cut off in the corona
    ltau_cut = 1.5  # Maximum ltau500 in the lowe boundary
    
    idx = pyTau.getOptimizedScale(Tg, rho, vz, ltau500, nDep2=nDep2, Tcut=Tcut,
                                  ltau_cut=ltau_cut, smooth_window=smooth,
                                  vel_scal=velscal, nthreads=nthreads)


    #
    # Interpolate variables to the new optimized grid
    #
    z_new    = pyTau.OptimizeVariable(idx, z, nthreads=nthreads)
    Tg_new   = pyTau.OptimizeVariable(idx, Tg, nthreads=nthreads)
    vz_new   = pyTau.OptimizeVariable(idx, vz, nthreads=nthreads)
    rho_new  = pyTau.OptimizeVariable(idx, rho, nthreads=nthreads)
    pgas_new = pyTau.OptimizeVariable(idx, pgas, nthreads=nthreads)


    #
    # plot variables in the new grid
    #
    
    plt.ion(); plt.close("all")
    
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,3))
    
    ax[0].plot(z[0,0]*1.e-5, Tg[0,0]*1.e-3,'k.', markersize=4, label='Original')
    ax[0].plot(z_new[0,0]*1.e-5, Tg_new[0,0]*1.e-3,'.', color='orangered', markersize=4, label='Optimized')
    ax[0].set_ylim(4,13)
    ax[0].set_ylabel('T [kK]')
    ax[0].set_xlabel('z [km]')
    ax[0].legend(frameon=True, loc='upper left', fontsize=6)
    
    ax[1].plot(z[0,0]*1.e-5, vz[0,0]*1.e-5,'k.', markersize=4)
    ax[1].plot(z_new[0,0]*1.e-5, vz_new[0,0]*1.e-5,'.', color='orangered', markersize=4)
    ax[1].set_ylabel('vz [km/s]')
    ax[1].set_xlabel('z [km]')

    f.set_tight_layout(True)
    f.show()
