import pTau
import numpy as np
from ipdb import set_trace

"""
Wrapper routines for C++ cython module
Coded by J. de la Cruz Rodriguez (ISP-SU, 2021)
"""

# *******************************************************************************

def getContinuumOpacity(Tg, Pg = None, rho = None, nthreads=4, wav=[5000.0]):
    """
    getContinuumOpacity computes the background opacity due to H (H, H-, etc.. and Thompson)
    Input: 
     Tg: Gas temperature in Kelvin, 3D numpy array [ny, nx, ndep] (float32 or float64)
     Pg: Gas pressure in Barye, 3D numpy array [ny, nx, ndep] (float32 or float64)
    rho: Mass density in g/cm**3,  3D numpy array [ny, nx, ndep] (float32 or float64)

    Note: Pg or rho must be provided even if they are defined as keywords.
    
    Optional:
    nthreads: number of parallel threads to use in the calculations
         wav: Wavelenghts at which the continuum opacity must be computed [in Angstroms]
    
    Coded by J. de la Cruz Rodriguez (ISP-SU, 2021)
    """
    wav1 = np.float64(wav)
    dtype = Tg.dtype
    ny, nx, nDep = Tg.shape

    print("getContinuumOpacity: ny={0}, nx={1}, nDep={2}, dtype={3}, nthreads={4}".format(ny, nx, nDep, dtype.name, nthreads))
    
    
    tcont =  Tg.flags['C_CONTIGUOUS']
    if(not tcont):
        temp = np.ascontiguousarray(tcont, order='c')
    else:
        temp = Tg

    
    if((Pg is None) and (rho is None)):
        print("ERROR, you must provide at least Pgas or Rho, exiting...")
        return None

    if(Pg is not None):
        pcont = Pg.flags['C_CONTIGUOUS']
        if(not pcont):
            pgas = np.ascontiguousarray(Pg, order='c')
        else:
            pgas = Pg


        if(dtype == 'float32'):
            return pTau.getBackgroundOpacityPgas_float(temp, pgas, wav, nthreads=nthreads)
        elif(dtype == 'float64'):
            return pTau.getBackgroundOpacityPgas_double(temp, pgas, wav, nthreads=nthreads)
        else:
            print("Unknown data type, use dtype='float64' or dtype='float32', exiting...")
            return None

    elif(rho is not None):
        
        rcont = rho.flags['C_CONTIGUOUS']
        if(not rcont):
            r = np.ascontiguousarray(rho, order='c')
        else:
            r = rho


        if(dtype == 'float32'):
            return pTau.getBackgroundOpacityRho_float(temp, r, wav, nthreads=nthreads)
        elif(dtype == 'float64'):
            return pTau.getBackgroundOpacityRho_double(temp, r, wav, nthreads=nthreads)
        else:
            print("Unknown data type, use dtype='float64' or dtype='float32', exiting...")
            return None

    return None

# *******************************************************************************

def getTau(Tg, z, Pg = None, rho = None, nthreads=4, wav=[5000.0]):
    """
    getTau computes the optical-depth scale given a set of temperature, Z-scale and [Pgas or Rho]
    Input: 
     Tg: Gas temperature in Kelvin, 3D numpy array [ny, nx, ndep] (float32 or float64)
      z: z-scale, 3D numpy array [ny, nx, ndep] (float32 or float64). Note that the top of the box must be located at index 0.
     Pg: Gas pressure in Barye, 3D numpy array [ny, nx, ndep] (float32 or float64)
    rho: Mass density in g/cm**3,  3D numpy array [ny, nx, ndep] (float32 or float64)
    
    Note: Pg or rho must be provided even if they are defined as keywords.
    
    Optional:
    nthreads: number of parallel threads to use in the calculations
         wav: Wavelenghts at which the continuum opacity must be computed [in Angstroms]
    
    Coded by J. de la Cruz Rodriguez (ISP-SU, 2021)
    """
    wav1 = np.asarray(wav, dtype='float64', order='c')
    
    alpha = getContinuumOpacity(Tg, Pg = Pg, rho=rho, nthreads=nthreads, wav = wav1)
    nWav = wav1.size

    ny, nx, nDep = Tg.shape
    tau = np.zeros((nWav, ny, nx, nDep), dtype=Tg.dtype, order='c')
    
    for ww in range(nWav):
        alpha1 = np.ascontiguousarray(alpha[:,:,:,ww])
        
        if(z.dtype == 'float64'):
            tau[ww] = pTau.getTau_double(z, alpha1, nthreads=nthreads)
        elif(z.dtype == 'float32'):
            tau[ww] = pTau.getTau_float(z, alpha1, nthreads=nthreads)
        else:
            print("getTau: Unknown data type for z, use dtype='float32' or dtype='float64', exiting...")
            return None

    return tau.squeeze()

# *******************************************************************************
