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

def _checkArray(var, dtype = None):

    if(dtype is None):
        dtype = var.dtype
    
    if((var.flags['C_CONTIGUOUS'] == True) and (var.dtype == dtype)):
        return var
    else:
        print("_checkArray: making array contiguous in memory")
        return np.ascontiguousarray(var, dtype=dtype)
    
# *******************************************************************************

def _checkDims(arr1, arr2, Name = ""):

    nDim = len(arr1.shape)
    if(len(arr2.shape) != nDim):
        print("Error, Array [{0}] has different dimensions than reference array, exiting".format(Name))
        return False
    
    allFine = True

    for ii in range(nDim):
        if(arr1.shape[ii] != arr2.shape[ii]):
            allFine = False
            print("Error, Array [{0}] has different dimensions than reference array: dim[{1}] = {2} != {3}]".format(Name, ii, arr2.shape[ii], arr1.shape[ii]))

    return allFine

# *******************************************************************************

def getOptimizedScale(temp, rho, vlos, ltau, nDep2 = None, nthreads = 4, Tcut = 50000.0, ltau_cut = 2.0, smooth_window = 1, dtype = None):
    
    #
    # Let's decide dtype based on temp
    #
    if(dtype is None):
        dtype = temp.dtype
        if((dtype != 'float32') and (dtype != 'float64')):
            dtype = 'float64'

    print("getOptimizedScale: dtype = {0}".format(dtype))
    #
    # Check that arrays are contiguous in memory and
    # all have the same dtype
    #
    temp1 = _checkArray(temp, dtype=dtype)
    rho1  = _checkArray(rho, dtype=dtype)
    vlos1  = _checkArray(vlos, dtype=dtype)
    ltau1  = _checkArray(ltau, dtype=dtype)

    # Check dimensions
    if(not _checkDims(temp1, rho1, Name = "rho")): return None
    if(not _checkDims(temp1, vlos1, Name = "vlos")): return None
    if(not _checkDims(temp1, ltau1, Name = "ltau")): return None

    #set_trace()
    
    # Call C++ wrapper based on the type
    if(dtype == 'float64'):
        return pTau.OptimizeGradients_double(temp1, rho1, vlos1, ltau1, nthreads = int(nthreads), nDep2 = nDep2, Tcut = Tcut, ltau_cut = ltau_cut, smooth_window = int(smooth_window))
    else:
        return pTau.OptimizeGradients_float(temp1, rho1, vlos1, ltau1, nthreads = int(nthreads), nDep2 = nDep2, Tcut = Tcut, ltau_cut = ltau_cut, smooth_window = int(smooth_window))

    
# *******************************************************************************

def OptimizeVariable(index, var, nthreads = 4, log = False):
    #
    # Let's decide dtype based on temp
    #
    dtype = var.dtype
    if((dtype != 'float32') and (dtype != 'float64')):
        dtype = 'float64'

    #
    # Check that arrays are contiguous in memory and
    # all have the same dtype
    #
    var1  = _checkArray(var, dtype=dtype)
    index1 = _checkArray(index, dtype=dtype)
    
    ## Check dimensions
    #if(not _checkDims(var1, index1, Name = "index")): return None

    # take the log?
    if(log):
        var1 = np.log(var1)
    
    # Call C++ wrapper based on the type
    if(dtype == 'float64'):
        res =  pTau.interpolate_gradient_double(index1, var1, nthreads = int(nthreads))
    else:
        res =  pTau.interpolate_gradient_float(index1, var1, nthreads = int(nthreads))

    if(log):
        res = np.exp(res)

    return res
        
        
# *******************************************************************************
