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

cdef extern from "gradients.hpp":
    cdef void optimize_gradients_float "gr::optimizeGradients<float>"(int nPix, int nDep, const float* const temp, const float* const ltau,
                                                                      const float* const rho, const float* const vlos, int smooth_window,
                                                                      float Tcut, float tau_cut, int nthreads, int nDep2, float* const res)

    cdef void optimize_gradients_double "gr::optimizeGradients<double>"(int nPix, int nDep, const double* const temp, const double* const ltau,
                                                                        const double* const rho, const double* const vlos, int smooth_window,
                                                                        double Tcut, double tau_cut, int nthreads, int nDep2, double* const res)
    
    cdef void interpolateGradient_double "gr::interpolateGradient<double>"(int  nPix, int nDep, const double* const  var, int  nDep2,
                                                                           const double* const  index, const double* const index_new, double* const res,
                                                                           int  nthreads)
    
    cdef void interpolateGradient_float "gr::interpolateGradient<float>"(int  nPix, int  nDep, const float* const  var, int  nDep2,
                                                                         const float* const  index, const float* const index_new, float* const res,
                                                                         int  nthreads)
    
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

    cdef ar[double, ndim=1] wav1 = wav * 1.e-8 # to cm

    
    # Allocate result
    cdef ar[double, ndim=4] alpha = zeros((ny,nx,nDep,nwav), dtype='float64', order='c')

    # call C++ tools
    getAlpha_T_Pg_float(ntot, <float*>Tg.data, <float*>Pg.data, nwav, <double*>wav1.data, <double*>alpha.data, <int>nthreads)
    

    return alpha

# ***********************************************************************************************

def getBackgroundOpacityPgas_double(ar[double,ndim=3] Tg, ar[double,ndim=3] Pg, ar[double,ndim=1] wav, int nthreads = 4):

    # Dimensions
    cdef int nwav = wav.size
    cdef int ny = Tg.shape[0]
    cdef int nx = Tg.shape[1]
    cdef int nDep = Tg.shape[2]
    cdef long ntot = (<long>(nx*ny)) * nDep;

    cdef ar[double, ndim=1] wav1 = wav * 1.e-8 # to cm

    
    # Allocate result
    cdef ar[double, ndim=4] alpha = zeros((ny,nx,nDep,nwav), dtype='float64', order='c')

    # call C++ tools
    getAlpha_T_Pg_double(ntot, <double*>Tg.data, <double*>Pg.data, nwav, <double*>wav1.data, <double*>alpha.data, <int>nthreads)
    

    return alpha

# ***********************************************************************************************


def getBackgroundOpacityRho_float(ar[float,ndim=3] Tg, ar[float,ndim=3] rho, ar[double,ndim=1] wav, int nthreads = 4):

    # Dimensions
    cdef int nwav = wav.size
    cdef int ny = Tg.shape[0]
    cdef int nx = Tg.shape[1]
    cdef int nDep = Tg.shape[2]
    cdef long ntot = (<long>(nx*ny)) * nDep;

    cdef ar[double, ndim=1] wav1 = wav * 1.e-8 # to cm
    
    # Allocate result
    cdef ar[double, ndim=4] alpha = zeros((ny,nx,nDep,nwav), dtype='float64', order='c')

    # call C++ tools
    getAlpha_T_rho_float(ntot, <float*>Tg.data, <float*>rho.data, nwav, <double*>wav1.data, <double*>alpha.data, <int>nthreads)
    

    return alpha

# ***********************************************************************************************

def getBackgroundOpacityRho_double(ar[double,ndim=3] Tg, ar[double,ndim=3] rho, ar[double,ndim=1] wav, int nthreads = 4):

    # Dimensions
    cdef int nwav = wav.size
    cdef int ny = Tg.shape[0]
    cdef int nx = Tg.shape[1]
    cdef int nDep = Tg.shape[2]
    cdef long ntot = (<long>(nx*ny)) * nDep;

    cdef ar[double, ndim=1] wav1 = wav * 1.e-8 # to cm
    
    # Allocate result
    cdef ar[double, ndim=4] alpha = zeros((ny,nx,nDep,nwav), dtype='float64', order = 'c')

    # call C++ tools
    getAlpha_T_rho_double(ntot, <double*>Tg.data, <double*>rho.data, nwav, <double*>wav1.data, <double*>alpha.data, <int>nthreads)
    

    return alpha

# ***********************************************************************************************

def getTau_float(ar[float,ndim=3] z, ar[double, ndim=3] alpha, int nthreads=4):

    # Dimensions
    cdef int ny = z.shape[0]
    cdef int nx = z.shape[1]
    cdef int nDep = z.shape[2]
    cdef int nPix = nx*ny

    
    # Allocate result
    cdef ar[float, ndim=3] tau = zeros((ny,nx,nDep), dtype='float32', order='c')


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
    cdef ar[double, ndim=3] tau = zeros((ny,nx,nDep), dtype='float64', order='c')


    # call C++ tools
    integrate_alpha_double(nPix, nDep, <double*>z.data, <double*>alpha.data, <double*>tau.data, <int>nthreads)

    return tau

# ***********************************************************************************************

def OptimizeGradients_double(ar[double,ndim=3] temp, ar[double,ndim=3] rho, ar[double,ndim=3] vlos, ar[double, ndim=3] ltau, int nthreads=4, nDep2 = None, int smooth_window = 1, double Tcut = 50000.0, double ltau_cut = 2.0):

    cdef int ny = temp.shape[0]
    cdef int nx = temp.shape[1]
    cdef int nDep = temp.shape[2]
    cdef int nDep_new = nDep
    cdef int nPix = ny*nx
    
    if(nDep2 is not None):
        nDep_new = int(nDep2)
        
    print("OptimizeGradients_double: ny={0}, nx={1}, nDep={2}, nDep2={3}".format(ny, nx, nDep, nDep_new))
    cdef ar[double,ndim = 3] res = zeros((ny, nx, nDep_new), dtype='float64', order='c')

    optimize_gradients_double(nPix, nDep, <double*>temp.data, <double*>ltau.data, <double*>rho.data, <double*>vlos.data, <int>smooth_window,
                              <double>Tcut, <double>ltau_cut, <int>nthreads, <int>nDep_new, <double*>res.data)

    return res

# ***********************************************************************************************

def OptimizeGradients_float(ar[float,ndim=3] temp, ar[float,ndim=3] rho, ar[float,ndim=3] vlos, ar[float, ndim=3] ltau, int nthreads=4, nDep2 = None, int smooth_window = 1, float Tcut = 50000.0, float ltau_cut = 2.0):

    cdef int ny = temp.shape[0]
    cdef int nx = temp.shape[1]
    cdef int nDep = temp.shape[2]
    cdef int nDep_new = nDep
    cdef int nPix = ny*nx

    
    if(nDep2 is not None):
        nDep_new = int(nDep2)
        
    print("OptimizeGradients_float: ny={0}, nx={1}, nDep={2}, nDep2={3}".format(ny, nx, nDep, nDep_new))
    cdef ar[float,ndim = 3] res = zeros((ny, nx, nDep_new), dtype='float32', order='c')

    optimize_gradients_float(nPix, nDep, <float*>temp.data, <float*>ltau.data, <float*>rho.data, <float*>vlos.data, <int>smooth_window,
                              <float>Tcut, <float>ltau_cut, <int>nthreads, <int>nDep_new, <float*>res.data)

    return res

# ***********************************************************************************************

def interpolate_gradient_double(ar[double, ndim=3] index_new, ar[double, ndim=3] var, int nthreads=4):

    cdef int ny = var.shape[0]
    cdef int nx = var.shape[1]
    cdef int nDep = var.shape[2]
    cdef int nDep_new = index_new.shape[2]
    cdef int nPix = ny*nx

    cdef ar[double, ndim=3] res = zeros((ny, nx, nDep_new), dtype='float64', order='c')
    cdef ar[double, ndim=1] index = arange(nDep, dtype='float64')
    
    interpolateGradient_double(nPix, nDep, <double*> var.data, nDep_new, <double*>index.data, <double*>index_new.data, <double*>res.data, <int>nthreads)

    return res

# ***********************************************************************************************

def interpolate_gradient_float(ar[float, ndim=3] index_new, ar[float, ndim=3] var, int nthreads=4):

    cdef int ny = var.shape[0]
    cdef int nx = var.shape[1]
    cdef int nDep = var.shape[2]
    cdef int nDep_new = index_new.shape[2]
    cdef int nPix = ny*nx
    

    cdef ar[float, ndim=3] res = zeros((ny, nx, nDep_new), dtype='float32', order='c')
    cdef ar[float, ndim=1] index = arange(nDep, dtype='float32')
    
    interpolateGradient_float(nPix, nDep, <float*> var.data, nDep_new, <float*>index.data, <float*>index_new.data, <float*>res.data, <int>nthreads)

    return res

# ***********************************************************************************************

