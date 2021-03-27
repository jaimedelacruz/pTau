#include <vector>
#include <cstring>
#include <cstdio>
#include <omp.h>

namespace gr{


  // ********************************************************************************* //

  template<class T> inline T signFortran(T const &val)
  {return ((val < static_cast<T>(0)) ? static_cast<T>(-1) : static_cast<T>(1));}
  

  // ********************************************************************************* //

  template<typename T> inline
  T getDerSteffen(T const &xu, T const &xc, T const &xd, T const &yu, T const &yc, T const &yd)
  {
    //
    // High order harmonic derivatives
    // Ref: Steffen (1990), A&A..239..443S
    //
    
    T const odx = xc - xu;
    T const dx  = xd - xc;
        
    T const S0 = (yd - yc) / dx;
    T const Su = (yc - yu) / odx;
    
    T const P0 = std::abs((Su*dx + S0*odx) / (odx+dx)) / 2;
    
    return (signFortran(S0) + signFortran(Su)) * std::min<T>(std::abs(Su),std::min<T>(std::abs(S0), P0));

  }
  
  // ********************************************************************************* //

  template<typename T>
  void getDerivatives(int const n, const T* const __restrict__ x, const T* const __restrict__ y, T* __restrict__ yp)
  {
    yp[0] = 0;
    yp[n-1] = 0;

    int const n1 = n-1;
    
    for(int ii=1; ii<n1; ++ii)
      yp[ii] = getDerSteffen(x[ii-1], x[ii], x[ii+1],y[ii-1], y[ii], y[ii+1]);
  }
  
  // ************************************************************************************************************** //

  template<typename T> inline
  void hermitian(int const n, const T* const __restrict__ x, const T* const __restrict__ y,
	       int const nn, const T* const __restrict__ xx,  T* const  __restrict__ yy)
  {

    // --- Get derivatives --- //
    
    T* __restrict__ yp = new T [n]();
    getDerivatives<T>(n, x, y, yp);

    
    // --- limits and order --- //
    
    int k0=0, k1=0, dk=0;
    if((x[1]-x[0]) > 0){
      dk = 1, k0=-1, k1=n;
    }else{
      dk = -1, k0=n, k1=-1;
    }

    int j0=0, j1=0, dj=0;
    if((xx[1]-xx[0]) > 0){
      j0 = -1, j1=nn; dj=1;
    }else{
      j1 = -1, j0=nn; dj=-1;
    }
			 
    
    int const kk0 = k0;
    int const kk1 = k1;
    int const dkk = dk;
    
    int const jj1 = j1;
    int const jj0 = j0;
    int const djj = dj;
    
    int off = jj0+djj;

    for(int kk=kk0+2*dkk; kk != kk1; kk+=dkk){
      T const dx = x[kk] - x[kk-dkk];

      int const off1 = off;
      
      for(int jj=off1; jj != jj1; jj += djj){
	
	// --- clip value to the existing data range, it will repeat edge values automatically --- //
	T const ixx = std::max<T>(std::min<T>(xx[jj], x[kk1-dkk]), x[kk0+dkk]);
	  
	if((ixx >= x[kk-dkk]) && (ixx <= x[kk])){
	  T const u = (ixx-x[kk-dkk])/dx;
	  T const uu = u*u;
	  T const uuu = u*uu;
	  
	  yy[jj] = y[kk-dkk] * (1 - 3*uu + 2*uuu) + (3*uu - 2*uuu) * y[kk]
	    + (uuu - 2*uu + u) * dx * yp[kk-dkk] + (uuu - uu) * dx * yp[kk];
	    
	  off += djj;
	} // if
      } // jj
    } // kk

    
    delete [] yp;
  }
  
  // ************************************************************************************************************** //

  template<typename T> inline
  void linear(int const n, const T* const __restrict__ x, const T* const __restrict__ y,
	      int const nn, const T* const __restrict__ xx,  T* const __restrict__ yy)
  {

    // --- limits and order --- //
    
    int k0=0, k1=0, dk=0;
    if((x[1]-x[0]) > 0){
      dk = 1, k0=-1, k1=n;
    }else{
      dk = -1, k0=n, k1=-1;
    }

    int j0=0, j1=0, dj=0;
    if((xx[1]-xx[0]) > 0){
      j0 = -1, j1=nn; dj=1;
    }else{
      j1 = -1, j0=nn; dj=-1;
    }
			 
    
    int const kk0 = k0;
    int const kk1 = k1;
    int const dkk = dk;
    
    int const jj1 = j1;
    int const jj0 = j0;
    int const djj = dj;
    
    int off = jj0+djj;

    for(int kk=kk0+2*dkk; kk != kk1; kk+=dkk){
      T const dx = x[kk] - x[kk-dkk];
      
      T const a = (y[kk] - y[kk-dkk]) / dx;
      T const b = y[kk-dkk] - a*x[kk-dkk];

      int const off1 = off;
      
      for(int jj=off1; jj != jj1; jj += djj){
	
	// --- clip value to the existing data range, it will repeat edge values automatically --- //
	T const ixx = std::max<T>(std::min<T>(xx[jj], x[kk1-dkk]), x[kk0+dkk]);

	
	if((ixx >= x[kk-dkk]) && (ixx <= x[kk])){	  
	  yy[jj] = a*ixx + b;
	    
	  off += djj;
	} // if
      } // jj
    } // kk
  }

  // ********************************************************************************* //

  template<typename T> inline
  void smooth_and_scale_gradients(int const N, T* const __restrict__ d, int const wsize, T const scaling, int const k0)
  {
    int const N1 = N - 1;

    
    // --- standard smoothing with top-hat PSF --- //
    
    if(wsize > 1){

      int const w2 = wsize / 2;
      int const Nw2 = N+w2;
      
      T* __restrict__ d_orig = new T [N+2*w2];
      
      std::memcpy(&d_orig[w2], d, sizeof(T)*N);
      
      for(int ii=0; ii<w2; ++ii){
	d_orig[ii]      = d[0];
	d_orig[ii+Nw2]  = d[N1]; 
      }
      
      
      for(int kk = w2; kk < Nw2; ++kk){
	int const j0 = kk-w2;
	int const j1 = kk+w2;
	
	T sum = 0;
	for(int jj=j0; jj<=j1; ++jj)
	  sum += d_orig[jj];
	
	d[kk-w2] = sum / (j1-j0+1); 
      }
      
      delete [] d_orig;   
    }

    
    // --- make sure that the scaling of the array is correct --- //
    
    T const off = d[0];
    T const range = scaling / (d[N1] - d[0]);
    
    for(int ii=0; ii<N; ++ii){
      d[ii] = (d[ii]-off) * range + k0;
    }   
  }
  
  // ************************************************************************************************************** //

  template<typename T> inline T* arange(int const N, T const v0, T const v1)
  {

    T const range = (v1-v0) / T(N-1);
    T* __restrict__ res = new T [N]();
    for(int ii=0; ii<N; ++ii) res[ii] = T(ii)*range + v0;
    return res;
  }
  

  // ************************************************************************************************************** //
  
  template<typename T> inline
  void optimizeGradients_one(int const nDep, const T* const __restrict__ temp, const T* const __restrict__ ltau,
			     const T* const __restrict__ rho, const T* const __restrict__ vlos, int const smooth_window,
			     T const Tcut, T const tau_cut, int const nDep2, T* const __restrict__ res)

  {

    // --- scaling rules --- //

    static const double log11 = 1 / log10(1.1);
    constexpr static const double vscal = 1.0E-5 / 2.0;


    // --- detect limits --- //
    
    int k0 = 0, k1 = nDep-1;
    for(int ii=0; ii<nDep; ++ii){
      if((temp[ii] >= Tcut)) k0 = ii;
      else break;
    }
    for(int ii=0; ii<nDep; ++ii){
      if(ltau[ii] <= tau_cut) k1 = ii;
      else break;
    }
    
    k1 = std::min(k1 + 1, nDep-1);
    int const nIndex = k1 - k0 + 1;

    
    // --- Measure gradients --- //
    
    int const kk1 = k1;
    int const kk0 = k0;
    
    T* const __restrict__ aind = new T [nIndex]();
    
    for(int kk=k0+1; kk<=k1; ++kk){
      T grad = std::abs(log10(temp[kk]) - log10(temp[kk-1])) * log11;
      grad = std::max<T>(grad, std::abs(log10(rho[kk]) - log10(rho[kk-1])) * log11 );
      grad = std::max<T>(grad, std::abs(vlos[kk]  -  vlos[kk-1]) * vscal);
      grad = std::max<T>(grad, std::abs(ltau[kk] - ltau[kk-1]) * 10);

      aind[kk-k0] = aind[kk-1] +  grad;
    }
    
    // --- smooth gradients and interpolate to new grid --- //
    
    smooth_and_scale_gradients<T>(nIndex, aind, smooth_window, nIndex, k0);
    
    const T* const __restrict__ index_new = arange<T>(nDep2, k0, k1);
    const T* const __restrict__ index = arange<T>(nIndex, k0, k1);

    linear<T>(nIndex, index, aind, nDep2, index_new, res);
    
    
    delete [] aind;
    delete [] index_new;
    delete [] index;
  }
  
  // ************************************************************************************************************** //

  template<typename T> inline
  void optimizeGradients(int const nPix, int const nDep, const T* const __restrict__ temp, const T* const __restrict__ ltau,
			 const T* const __restrict__ rho, const T* const __restrict__ vlos, int const smooth_window,
			 T const Tcut, T const tau_cut, int const nthreads, int const nDep2, T* const __restrict__ res)
    
  {
    
    int ipix = 0;
    
#pragma omp parallel default(shared) firstprivate(ipix) num_threads(nthreads)
    {
#pragma omp for schedule(static)
      for(ipix=0; ipix<nPix; ++ipix){
	
	optimizeGradients_one<T>(nDep, &temp[nDep*ipix], &ltau[nDep*ipix], &rho[nDep*ipix], &vlos[nDep*ipix],
				 smooth_window, Tcut, tau_cut, nDep2, &res[nDep2*ipix]);
	
      } // ipix
    } // parallel block
  }
  
  // ************************************************************************************************************** //

  template<typename T> inline
  void interpolateGradient_one(int const nDep, const T* const __restrict__ var, int const nDep2, const T* const __restrict__ index,
				const T* const __restrict__ index_new, T* const __restrict__ res)
  {
    //linear<T>(nDep, index, var, nDep2, index_new, res);
    hermitian<T>(nDep, index, var, nDep2, index_new, res); 

  }
  
  // ************************************************************************************************************** //

  template<typename T> inline
  void interpolateGradient(int const nPix, int const nDep, const T* const __restrict__ var, int const nDep2, const T* const __restrict__ index,
			   const T* const __restrict__ index_new, T* const __restrict__ res, int const nthreads)
  {

    int ipix = 0;
    
#pragma omp parallel default(shared) firstprivate(ipix) num_threads(nthreads)
    {
#pragma omp for schedule(static)
      for(ipix=0; ipix<nPix; ++ipix){
	interpolateGradient_one<T>(nDep, &var[nDep*ipix], nDep2, index, &index_new[nDep2*ipix], &res[nDep2*ipix]);
      } // ipix
    } // parallel block

  }
  
  // ************************************************************************************************************** //

}
