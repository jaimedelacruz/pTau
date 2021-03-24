"""
CYTHON interface for C++.
Author: J. de la Cruz Rodriguez (ISP-SU, 2021)
"""
import cython
cimport numpy as np
from numpy cimport ndarray as ar
from numpy import empty, ascontiguousarray, zeros, ones, outer, int32, arange
from libcpp cimport bool
import sys
from scipy.interpolate import interp2d

__author__="Jaime de la Cruz Rodriguez (ISP-SU 2021)"
__status__="Developing"
__email__="jaime@astro.su.se"

# ***********************************************************************************************
#
# C++ prototypes
#
# ***********************************************************************************************

cdef extern from "pTau.hpp":
    cdef void integrate_alpha_double "integrate_alpha<double>"(int nPix, int nDep, const double* const z,
		     const double* const  alpha, double* const ltau, int nthreads)
    cdef void integrate_alpha_float "integrate_alpha<float>"(int nPix, int nDep, const float* const z,
		     const double* const  alpha, float* const ltau, int nthreads)
    
    cdef void getAlpha_T_rho_double "getAlpha_T_rho<double>"(long ntot, const double* const Tg, const double* const rho,
		    int  nLambda, const double* const  lambd, double* const alpha, int  nthreads)
    cdef void getAlpha_T_rho_float "getAlpha_T_rho<float>"(long ntot, const float* const Tg, const float* const rho,
		    int  nLambda, const double* const  lambd, double* const alpha, int  nthreads)

    cdef void getAlpha_T_Pg_double "getAlpha_T_Pg<double>"(long ntot, const double* const Tg, const double* const Pg,
		    int  nLambda, const double* const  lambd, double* const alpha, int  nthreads)
    cdef void getAlpha_T_Pg_float "getAlpha_T_Pg<float>"(long ntot, const float* const Tg, const float* const Pg,
		    int  nLambda, const double* const  lambd, double* const alpha, int  nthreads)


# ***********************************************************************************************
#
# Python interface
#
# ***********************************************************************************************

def getBackgroundOpacityPgas_float(ar[float,ndim=3] Tg, ar[float,ndim=3] Pg, ar[double,ndim=1] wav, int nthreads = 4):

    # Dimensions
    cdef int nwav = wav.size
    cdef int ny = Tg.shape[0]
    cdef int nx = Tg.shape[1]
    cdef int nDep = Tg.shape[2]
    cdef long ntot = (<long>(nx*ny)) * nDep;
    
    # Allocate result
    cdef ar[double, ndim=3] alpha = zeros((ny,nx,nDep), dtype='float64')

    # call C++ tools
    getAlpha_T_Pg_float(ntot, <float*>Tg.data, <float*>Pg.data, nwav, <double*>wav.data, <double*>alpha.data, <int>nthreads)
    

    return alpha

# ***********************************************************************************************

def getBackgroundOpacityPgas_double(ar[double,ndim=3] Tg, ar[double,ndim=3] Pg, ar[double,ndim=1] wav, int nthreads = 4):

    # Dimensions
    cdef int nwav = wav.size
    cdef int ny = Tg.shape[0]
    cdef int nx = Tg.shape[1]
    cdef int nDep = Tg.shape[2]
    cdef long ntot = (<long>(nx*ny)) * nDep;
    
    # Allocate result
    cdef ar[double, ndim=3] alpha = zeros((ny,nx,nDep), dtype='float64')

    # call C++ tools
    getAlpha_T_Pg_double(ntot, <double*>Tg.data, <double*>Pg.data, nwav, <double*>wav.data, <double*>alpha.data, <int>nthreads)
    

    return alpha

# ***********************************************************************************************


def getBackgroundOpacityRho_float(ar[float,ndim=3] Tg, ar[float,ndim=3] rho, ar[double,ndim=1] wav, int nthreads = 4):

    # Dimensions
    cdef int nwav = wav.size
    cdef int ny = Tg.shape[0]
    cdef int nx = Tg.shape[1]
    cdef int nDep = Tg.shape[2]
    cdef long ntot = (<long>(nx*ny)) * nDep;
    
    # Allocate result
    cdef ar[double, ndim=3] alpha = zeros((ny,nx,nDep), dtype='float64')

    # call C++ tools
    getAlpha_T_rho_float(ntot, <float*>Tg.data, <float*>rho.data, nwav, <double*>wav.data, <double*>alpha.data, <int>nthreads)
    

    return alpha

# ***********************************************************************************************

def getBackgroundOpacityRho_double(ar[double,ndim=3] Tg, ar[double,ndim=3] rho, ar[double,ndim=1] wav, int nthreads = 4):

    # Dimensions
    cdef int nwav = wav.size
    cdef int ny = Tg.shape[0]
    cdef int nx = Tg.shape[1]
    cdef int nDep = Tg.shape[2]
    cdef long ntot = (<long>(nx*ny)) * nDep;
    
    # Allocate result
    cdef ar[double, ndim=3] alpha = zeros((ny,nx,nDep), dtype='float64')

    # call C++ tools
    getAlpha_T_rho_double(ntot, <double*>Tg.data, <double*>rho.data, nwav, <double*>wav.data, <double*>alpha.data, <int>nthreads)
    

    return alpha

# ***********************************************************************************************

def getTau_float(ar[float,ndim=3] z, ar[double, ndim=3] alpha, int nthreads=4):

    # Dimensions
    cdef int ny = z.shape[0]
    cdef int nx = z.shape[1]
    cdef int nDep = z.shape[2]
    cdef int nPix = nx*ny

    
    # Allocate result
    cdef ar[float, ndim=3] tau = zeros((ny,nx,nDep), dtype='float32')


    # call C++ tools
    integrate_alpha_float(nPix, nDep, <float*>z.data, <double*>alpha.data, <float*>tau.data, <int>nthreads)

    return tau

# ***********************************************************************************************

def getTau_double(ar[double,ndim=3] z, ar[double, ndim=3] alpha, int nthreads=4):

    # Dimensions
    cdef int ny = z.shape[0]
    cdef int nx = z.shape[1]
    cdef int nDep = z.shape[2]
    cdef int nPix = nx*ny

    
    # Allocate result
    cdef ar[double, ndim=3] tau = zeros((ny,nx,nDep), dtype='float64')


    # call C++ tools
    integrate_alpha_double(nPix, nDep, <double*>z.data, <double*>alpha.data, <double*>tau.data, <int>nthreads)

    return tau

# ***********************************************************************************************

