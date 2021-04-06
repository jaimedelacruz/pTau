import numpy as np
import time
import pyTau

if __name__ == "__main__":

    #
    # Let's assume that we have a set of variables Temp, Pgas, z in CGS units
    # The axis ordering is (ny, nx, ndep) where ndep is the fast axis (c-ordering).
    #
    # Note that the upper boundary of the atmosphere must be located at index=0.

    t0 = time.time()
    nthreads = 8 # Use as many as your computer has!
    ltau500 = pyTau.getTau(temp, z3d, Pg=pgas, nthreads=nthreads, wav = [5000.0])
    t1 = time.time()
    print("Pgas case dT = {0}s".format(t1-t0))
    
    #
    # Alternatively we can use rho instead of Pgas but it is almost a factor x2 slower due to the internals of the EOS
    #
    t0 = time.time()
    ltau500 = pyTau.getTau(temp, z3d, rho=rho, nthreads=nthreads, wav = [5000.0])
    t1 = time.time()
    print("Rho case dT = {0}s".format(t1-t0))
    
    
    
    
    
