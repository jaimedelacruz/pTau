#include <vector>
#include <omp.h>
#include <algorithm>

/* ---
  
  Python tools for the computation of a Tau scale.
  Coded by J. de la Cruz Rodriguez (ISP-SU, 2021)
  
  --- */

#include "physical_constants.hpp"
#include "cSIR/cSIR_eos.hpp"
#include "cSIR/cSIR_cont.hpp"

// ********************************************************* //

template<typename T> inline
void checkNaNs(int const n, T* const __restrict__ var)
{
  for(int ii=0; ii<n; ++ii)
    var[ii] = (isnan(var[ii]) ? 0 : var[ii]);
}


// ********************************************************* //


template<typename T> inline
T getMu(int const nAbund, const T* const __restrict__ abund)
{
  T mu = 0;
  for(int ii=0; ii<nAbund; ++ii)
    mu += abund[ii];
  
  mu /= abund[0];
  
  return mu;
}

// ********************************************************* //

template<typename T> inline
T getAvweight(int const N, const T* const __restrict__ abund)
{
  T sum = 0, tabund = 0;
  
  for(int ii=0; ii<N; ++ii){
    sum += sr::AMASS<T>[ii] * abund[ii];
    tabund += abund[ii];
  }
  
  sum /= tabund;
  return sum * phyc::AMU<T>;
}

// ********************************************************* //

template<typename T> inline
void getAlpha_one(T const Tg, T const Ne, int const nLambda,
		  const double* const __restrict__ lambda, T* const __restrict__ alpha,
		   T* const __restrict__ Hpop)
{
  // alpha has dimensions (nLambda) //
  
  //T Hpop[sr::nHv] = {}; // H populations
  //sr::compute_Ntot(Tg, Ne, abund, Hpop); // Solve EOS to get H partial densities
  
  for(int ww=0; ww<nLambda; ++ww){
    T const tmp =  sr::continuum_absorption<double>(double(Tg), double(Ne), lambda[ww], Hpop);
    alpha[ww] = std::max(((isnan(tmp)) ? T(1.e-31) : tmp), T(1.e-31));
  }
}
// ********************************************************* //

template<typename T> inline
void getAlpha_one_bif(T const Tg, T const Ne, int const nLambda,
		  const double* const __restrict__ lambda, T* const __restrict__ alpha,
		   T* const __restrict__ Hpop)
{
  // alpha has dimensions (nLambda) //
  
  //T Hpop[sr::nHv] = {}; // H populations
  //sr::compute_Ntot(Tg, Ne, abund, Hpop); // Solve EOS to get H partial densities
  
  for(int ww=0; ww<nLambda; ++ww){
    T const tmp =  sr::continuum_absorption_bif<double>(double(Tg), double(Ne), lambda[ww], Hpop);
    alpha[ww] = std::max(((isnan(tmp)) ? T(1.e-31) : tmp), T(1.e-31));
  }
}
// ********************************************************* //

template<typename T> inline
T Ne_from_Rho(T const &Tg, T const& rho,  T Hpop[sr::nHv], std::vector<T> const& abund, T const avweight, T const mu)
{
  
  T const BKT = Tg * phyc::BK<T>;
  
  T Pg = (rho /avweight) * BKT;
  T Pe = sr::Pe_from_Pg(Tg, Pg, abund, Hpop);
  T irho = sr::computeRho(Hpop[0], mu);
  
  double diff = 1.0;
  int it = 0;
  
  while((diff>=1.e-5) && (it++ < 500)){
    Pe *= (1.0 + rho / irho) / 2;
    Pg = sr::compute_Pg(Tg, Pe, abund, Hpop);
    
    irho = sr::computeRho(Hpop[0], mu);
    
    diff = std::abs((irho - rho) / rho);
  }
  
  return Pe / BKT;
}



// ********************************************************* //

template<typename T> inline
void getAlpha_T_Pg(long const ntot, const T* const __restrict__ Tg, const T* const __restrict__ Pg,
		   int const nLambda, const double* const __restrict__ lambda, double* const __restrict__ alpha,
		   int const nthreads)
{
  
  static std::vector<double> const abund =
    {9.13226338e-01, 8.53214895e-02, 1.12475628e-11, 1.29139300e-11,
     3.63963999e-10, 3.31939113e-04, 1.02578992e-04, 7.78141233e-04,
     3.31939113e-08, 1.12475628e-04, 1.95460240e-06, 3.47582921e-05,
     2.69810240e-06, 3.24383256e-05, 2.57666779e-07, 1.48271755e-05,
     2.89106881e-07, 3.31939113e-06, 1.51725448e-07, 2.09439421e-06,
     1.15095522e-09, 8.93425674e-08, 9.14236232e-09, 4.27620413e-07,
     2.24418383e-07, 3.16999391e-05, 7.60428576e-08, 1.62576747e-06,
     1.48271755e-08, 3.63963999e-08, 6.93519103e-10, 2.34994895e-09,
     2.14317892e-10, 2.04671998e-09, 3.89994452e-10, 1.55259588e-09,
     3.63963999e-10, 7.26203652e-10, 1.58876048e-10, 2.89106881e-10,
     2.40468630e-11, 7.60428576e-11, 1.00244009e-20, 6.32496940e-11,
     1.20519807e-11, 4.47773542e-11, 7.96266470e-12, 6.62305602e-11,
     2.63668609e-11, 9.14236232e-11, 9.14236232e-12, 1.58876048e-10,
     2.95841045e-11, 1.55259588e-10, 1.20519807e-11, 1.23327074e-10,
     1.51725448e-11, 3.24383256e-11, 4.68876459e-12, 2.89106881e-11,
     1.00244009e-20, 9.14236232e-12, 2.95841045e-12, 1.20519807e-11,
     1.15095522e-12, 1.15095522e-11, 1.66363646e-12, 7.78141233e-12,
     9.14236232e-13, 1.09915371e-11, 5.26088040e-12, 6.93519103e-12,
     1.23327074e-12, 1.17776442e-11, 1.70238753e-12, 2.57666779e-11,
     2.04671998e-11, 5.76844065e-11, 9.35531529e-12, 1.12475628e-11,
     7.26203652e-12, 6.47229686e-11, 4.68876459e-12, 1.00244009e-20,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20, 1.00244009e-20,
     1.00244009e-20, 1.20519807e-12, 1.00244009e-20, 3.09783604e-13,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20, 1.00244009e-20,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20};
  
  

  // --- prepare parallel loop and get alpha --- //

  long ii = 0;
  int tid = 0, per=0, oper=-1;
  float pscl = 100.0 / (ntot-1.0);
  double Ne = 0, tg = 0, pg = 0;

  double* const __restrict__ Hpop_all = new double [sr::nHv*nthreads];
  double* __restrict__ Hpop = NULL;
  
#pragma omp parallel default(shared) firstprivate(ii, Ne, Hpop, tg, pg, tid) num_threads(nthreads)
  {

    tid = omp_get_thread_num();
    Hpop = &Hpop_all[tid*sr::nHv];
    
#pragma omp for schedule(dynamic,8)
    for(ii=0; ii<ntot; ++ii){
      tg = Tg[ii], pg = Pg[ii];
      
      Ne = sr::Pe_from_Pg<double>(tg, pg, abund, Hpop) / (tg*phyc::BK<double>);
      
      if(isnan(Ne))
	Ne = sr::init_Pe_from_Pg<double>(tg, pg, 0.913)/ (tg*phyc::BK<double>);
      
      
      getAlpha_one<double>(tg, Ne, nLambda, lambda, &alpha[ii*nLambda], Hpop);

      
      if(tid == 0){
	per = int(ii*pscl+0.5);
	if(per != oper){
	  oper = per;
	  fprintf(stderr,"\rprocessing -> %4d%s", per,"%");
	}
      }
    } // ii
  } // parallel
  fprintf(stderr,"\rprocessing -> %4d%s\n", int(100),"%");

  delete [] Hpop_all;
}
  // ********************************************************* //

template<typename T> inline
void getAlpha_T_Ne(long const ntot, const T* const __restrict__ Tg, const T* const __restrict__ Ne_in, const T* const __restrict__ nHin,
		    int const nLambda, const double* const __restrict__ lambda, double* const __restrict__ alpha,
		    int const nthreads)
{
  
  static std::vector<double> const abund =
    {9.13226338e-01, 8.53214895e-02, 1.12475628e-11, 1.29139300e-11,
     3.63963999e-10, 3.31939113e-04, 1.02578992e-04, 7.78141233e-04,
     3.31939113e-08, 1.12475628e-04, 1.95460240e-06, 3.47582921e-05,
     2.69810240e-06, 3.24383256e-05, 2.57666779e-07, 1.48271755e-05,
     2.89106881e-07, 3.31939113e-06, 1.51725448e-07, 2.09439421e-06,
     1.15095522e-09, 8.93425674e-08, 9.14236232e-09, 4.27620413e-07,
     2.24418383e-07, 3.16999391e-05, 7.60428576e-08, 1.62576747e-06,
     1.48271755e-08, 3.63963999e-08, 6.93519103e-10, 2.34994895e-09,
     2.14317892e-10, 2.04671998e-09, 3.89994452e-10, 1.55259588e-09,
     3.63963999e-10, 7.26203652e-10, 1.58876048e-10, 2.89106881e-10,
     2.40468630e-11, 7.60428576e-11, 1.00244009e-20, 6.32496940e-11,
     1.20519807e-11, 4.47773542e-11, 7.96266470e-12, 6.62305602e-11,
     2.63668609e-11, 9.14236232e-11, 9.14236232e-12, 1.58876048e-10,
     2.95841045e-11, 1.55259588e-10, 1.20519807e-11, 1.23327074e-10,
     1.51725448e-11, 3.24383256e-11, 4.68876459e-12, 2.89106881e-11,
     1.00244009e-20, 9.14236232e-12, 2.95841045e-12, 1.20519807e-11,
     1.15095522e-12, 1.15095522e-11, 1.66363646e-12, 7.78141233e-12,
     9.14236232e-13, 1.09915371e-11, 5.26088040e-12, 6.93519103e-12,
     1.23327074e-12, 1.17776442e-11, 1.70238753e-12, 2.57666779e-11,
     2.04671998e-11, 5.76844065e-11, 9.35531529e-12, 1.12475628e-11,
     7.26203652e-12, 6.47229686e-11, 4.68876459e-12, 1.00244009e-20,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20, 1.00244009e-20,
     1.00244009e-20, 1.20519807e-12, 1.00244009e-20, 3.09783604e-13,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20, 1.00244009e-20,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20};


  
  // --- get Mu and avweight--- //

  static double const mu       = getMu(      int(abund.size()), &abund[0]);
  static double const avweight = getAvweight(int(abund.size()), &abund[0]);

  
  // --- prepare parallel loop and get alpha --- //

  long ii = 0; 
  int k = 0;
  
  int tid = 0, per=0, oper=-1;
  float pscl = 100.0 / (ntot-1.0);
  double Ne = 0, tg=0, r = 0;

  double* const __restrict__ Hpop_all = new double [sr::nHv*nthreads];
  double* __restrict__ Hpop = NULL;
  
  
#pragma omp parallel default(shared) firstprivate(ii, Ne, tid, Hpop, tg, r, k) num_threads(nthreads)
  {

    tid = omp_get_thread_num();
    Hpop = &Hpop_all[tid*sr::nHv];
    
#pragma omp for schedule(dynamic,8)
    for(ii=0; ii<ntot; ++ii){
      tg = Tg[ii];//, r = rho[ii];
      Ne = Ne_in[ii];
      
      Hpop[0] = nHin[ii];
      
      for(k=1; k<(sr::nHv-1); ++k){
	Hpop[0] += nHin[k*ntot + ii];
      }
      Hpop[1] = nHin[5*ntot + ii];
      
      getAlpha_one_bif(tg, Ne, nLambda, lambda, &alpha[ii*nLambda], Hpop);

      if(tid == 0){
	per = int(ii*pscl+0.5);
	if(per != oper){
	  oper = per;
	  fprintf(stderr,"\rprocessing -> %4d%s", per,"%");
	}
      }
    } // ii
  } // parallel

  fprintf(stderr,"\rprocessing -> %4d%s\n", int(100), "%");

  delete [] Hpop_all;
}
// ********************************************************* //

template<typename T> inline
void getAlpha_T_rho(long const ntot, const T* const __restrict__ Tg, const T* const __restrict__ rho,
		    int const nLambda, const double* const __restrict__ lambda, double* const __restrict__ alpha,
		    int const nthreads)
{
  
  static std::vector<double> const abund =
    {9.13226338e-01, 8.53214895e-02, 1.12475628e-11, 1.29139300e-11,
     3.63963999e-10, 3.31939113e-04, 1.02578992e-04, 7.78141233e-04,
     3.31939113e-08, 1.12475628e-04, 1.95460240e-06, 3.47582921e-05,
     2.69810240e-06, 3.24383256e-05, 2.57666779e-07, 1.48271755e-05,
     2.89106881e-07, 3.31939113e-06, 1.51725448e-07, 2.09439421e-06,
     1.15095522e-09, 8.93425674e-08, 9.14236232e-09, 4.27620413e-07,
     2.24418383e-07, 3.16999391e-05, 7.60428576e-08, 1.62576747e-06,
     1.48271755e-08, 3.63963999e-08, 6.93519103e-10, 2.34994895e-09,
     2.14317892e-10, 2.04671998e-09, 3.89994452e-10, 1.55259588e-09,
     3.63963999e-10, 7.26203652e-10, 1.58876048e-10, 2.89106881e-10,
     2.40468630e-11, 7.60428576e-11, 1.00244009e-20, 6.32496940e-11,
     1.20519807e-11, 4.47773542e-11, 7.96266470e-12, 6.62305602e-11,
     2.63668609e-11, 9.14236232e-11, 9.14236232e-12, 1.58876048e-10,
     2.95841045e-11, 1.55259588e-10, 1.20519807e-11, 1.23327074e-10,
     1.51725448e-11, 3.24383256e-11, 4.68876459e-12, 2.89106881e-11,
     1.00244009e-20, 9.14236232e-12, 2.95841045e-12, 1.20519807e-11,
     1.15095522e-12, 1.15095522e-11, 1.66363646e-12, 7.78141233e-12,
     9.14236232e-13, 1.09915371e-11, 5.26088040e-12, 6.93519103e-12,
     1.23327074e-12, 1.17776442e-11, 1.70238753e-12, 2.57666779e-11,
     2.04671998e-11, 5.76844065e-11, 9.35531529e-12, 1.12475628e-11,
     7.26203652e-12, 6.47229686e-11, 4.68876459e-12, 1.00244009e-20,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20, 1.00244009e-20,
     1.00244009e-20, 1.20519807e-12, 1.00244009e-20, 3.09783604e-13,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20, 1.00244009e-20,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20};


  
  // --- get Mu and avweight--- //

  static double const mu       = getMu(      int(abund.size()), &abund[0]);
  static double const avweight = getAvweight(int(abund.size()), &abund[0]);

  
  // --- prepare parallel loop and get alpha --- //

  long ii = 0;
  int tid = 0, per=0, oper=-1;
  float pscl = 100.0 / (ntot-1.0);
  double Ne = 0, tg=0, r = 0;

  double* const __restrict__ Hpop_all = new double [sr::nHv*nthreads];
  double* __restrict__ Hpop = NULL;
  
#pragma omp parallel default(shared) firstprivate(ii, Ne, tid, Hpop, tg, r) num_threads(nthreads)
  {

    tid = omp_get_thread_num();
    Hpop = &Hpop_all[tid*sr::nHv];
    
#pragma omp for schedule(dynamic,8)
    for(ii=0; ii<ntot; ++ii){
      tg = Tg[ii], r = rho[ii];
      Ne = Ne_from_Rho(tg, r, Hpop, abund, avweight, mu);
      getAlpha_one(tg, Ne, nLambda, lambda, &alpha[ii*nLambda], Hpop);

      if(tid == 0){
	per = int(ii*pscl+0.5);
	if(per != oper){
	  oper = per;
	  fprintf(stderr,"\rprocessing -> %4d%s", per,"%");
	}
      }
    } // ii
  } // parallel

  fprintf(stderr,"\rprocessing -> %4d%s\n", int(100), "%");

  delete [] Hpop_all;
}

// ********************************************************* //

template<typename T> inline
void integrate_one(int const nDep, const T* const __restrict__ z,
		   const double* const __restrict__ alpha, T* const __restrict__ ltau)
{
  
  // --- get tau scale using trapezoidal rule --- //

  double itau = 0;
  ltau[0] = 0;
  
  for(int ii=1; ii<nDep; ++ii){
    itau += (alpha[ii-1] + alpha[ii])*std::abs(z[ii-1]-z[ii])/2;
    ltau[ii] = itau;
  }
  
  // --- fix upper boundary and take the log10 --- //
  itau = exp(2*log(ltau[1]) - log(ltau[2]));
  
  for(int ii=0; ii<nDep; ++ii)
    ltau[ii] = log10(ltau[ii] + itau);
  
}

// ********************************************************* //

template<typename T> inline
void integrate_alpha(long const nPix, long const nDep, const T* const __restrict__ z,
		     const double* const __restrict__ alpha, T* const __restrict__ ltau, int const nthreads)
{
  long ii = 0;
  
#pragma omp parallel default(shared) firstprivate(ii) num_threads(nthreads)
  {
#pragma omp for schedule(static)
    for(ii=0; ii<nPix; ++ii){
      integrate_one(nDep, &z[ii*nDep], &alpha[ii*nDep], &ltau[ii*nDep]);
    }
  }
}

// ********************************************************* //

template<typename T> inline
void get_Ne_T_Pg(long const ntot, const T* const __restrict__ Tg, const T* const __restrict__ Pg,
		 T* const __restrict__ Ne, int const nthreads)
{
  
  static std::vector<long double> const abund =
    {9.13226338e-01, 8.53214895e-02, 1.12475628e-11, 1.29139300e-11,
     3.63963999e-10, 3.31939113e-04, 1.02578992e-04, 7.78141233e-04,
     3.31939113e-08, 1.12475628e-04, 1.95460240e-06, 3.47582921e-05,
     2.69810240e-06, 3.24383256e-05, 2.57666779e-07, 1.48271755e-05,
     2.89106881e-07, 3.31939113e-06, 1.51725448e-07, 2.09439421e-06,
     1.15095522e-09, 8.93425674e-08, 9.14236232e-09, 4.27620413e-07,
     2.24418383e-07, 3.16999391e-05, 7.60428576e-08, 1.62576747e-06,
     1.48271755e-08, 3.63963999e-08, 6.93519103e-10, 2.34994895e-09,
     2.14317892e-10, 2.04671998e-09, 3.89994452e-10, 1.55259588e-09,
     3.63963999e-10, 7.26203652e-10, 1.58876048e-10, 2.89106881e-10,
     2.40468630e-11, 7.60428576e-11, 1.00244009e-20, 6.32496940e-11,
     1.20519807e-11, 4.47773542e-11, 7.96266470e-12, 6.62305602e-11,
     2.63668609e-11, 9.14236232e-11, 9.14236232e-12, 1.58876048e-10,
     2.95841045e-11, 1.55259588e-10, 1.20519807e-11, 1.23327074e-10,
     1.51725448e-11, 3.24383256e-11, 4.68876459e-12, 2.89106881e-11,
     1.00244009e-20, 9.14236232e-12, 2.95841045e-12, 1.20519807e-11,
     1.15095522e-12, 1.15095522e-11, 1.66363646e-12, 7.78141233e-12,
     9.14236232e-13, 1.09915371e-11, 5.26088040e-12, 6.93519103e-12,
     1.23327074e-12, 1.17776442e-11, 1.70238753e-12, 2.57666779e-11,
     2.04671998e-11, 5.76844065e-11, 9.35531529e-12, 1.12475628e-11,
     7.26203652e-12, 6.47229686e-11, 4.68876459e-12, 1.00244009e-20,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20, 1.00244009e-20,
     1.00244009e-20, 1.20519807e-12, 1.00244009e-20, 3.09783604e-13,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20, 1.00244009e-20,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20};


  
  long double* const __restrict__ Hpop_all = new long double [sr::nHv*nthreads];
  long double* __restrict__ Hpop = NULL;
  long double tg=0, pg = 0;
  long ii = 0;
  int tid = 0;
  
  
#pragma omp parallel default(shared) firstprivate(ii, Ne, Hpop, tid, tg, pg) num_threads(nthreads)
  {

    tid = omp_get_thread_num();
    Hpop = &Hpop_all[tid*sr::nHv];

#pragma omp for schedule(dynamic,8)
    for(ii=0; ii<ntot; ++ii){
      tg = Tg[ii], pg = Pg[ii];
      Ne[ii] = sr::Pe_from_Pg<long double>(tg, pg, abund, Hpop) / (tg*phyc::BK<long double>);
    }
  }

  
  delete [] Hpop_all;
}


// ********************************************************* //

template<typename T> inline
void get_Ne_T_Rho(long const ntot, const T* const __restrict__ Tg, const T* const __restrict__ Rho,
		 T* const __restrict__ Ne, int const nthreads)
{
  
  static std::vector<long double> const abund =
    {9.13226338e-01, 8.53214895e-02, 1.12475628e-11, 1.29139300e-11,
     3.63963999e-10, 3.31939113e-04, 1.02578992e-04, 7.78141233e-04,
     3.31939113e-08, 1.12475628e-04, 1.95460240e-06, 3.47582921e-05,
     2.69810240e-06, 3.24383256e-05, 2.57666779e-07, 1.48271755e-05,
     2.89106881e-07, 3.31939113e-06, 1.51725448e-07, 2.09439421e-06,
     1.15095522e-09, 8.93425674e-08, 9.14236232e-09, 4.27620413e-07,
     2.24418383e-07, 3.16999391e-05, 7.60428576e-08, 1.62576747e-06,
     1.48271755e-08, 3.63963999e-08, 6.93519103e-10, 2.34994895e-09,
     2.14317892e-10, 2.04671998e-09, 3.89994452e-10, 1.55259588e-09,
     3.63963999e-10, 7.26203652e-10, 1.58876048e-10, 2.89106881e-10,
     2.40468630e-11, 7.60428576e-11, 1.00244009e-20, 6.32496940e-11,
     1.20519807e-11, 4.47773542e-11, 7.96266470e-12, 6.62305602e-11,
     2.63668609e-11, 9.14236232e-11, 9.14236232e-12, 1.58876048e-10,
     2.95841045e-11, 1.55259588e-10, 1.20519807e-11, 1.23327074e-10,
     1.51725448e-11, 3.24383256e-11, 4.68876459e-12, 2.89106881e-11,
     1.00244009e-20, 9.14236232e-12, 2.95841045e-12, 1.20519807e-11,
     1.15095522e-12, 1.15095522e-11, 1.66363646e-12, 7.78141233e-12,
     9.14236232e-13, 1.09915371e-11, 5.26088040e-12, 6.93519103e-12,
     1.23327074e-12, 1.17776442e-11, 1.70238753e-12, 2.57666779e-11,
     2.04671998e-11, 5.76844065e-11, 9.35531529e-12, 1.12475628e-11,
     7.26203652e-12, 6.47229686e-11, 4.68876459e-12, 1.00244009e-20,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20, 1.00244009e-20,
     1.00244009e-20, 1.20519807e-12, 1.00244009e-20, 3.09783604e-13,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20, 1.00244009e-20,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20};


  
  
  
  // --- get Mu and avweight--- //

  static long double const mu       = getMu(      int(abund.size()), &abund[0]);
  static long double const avweight = getAvweight(int(abund.size()), &abund[0]);

  
  long double* const __restrict__ Hpop_all = new long double [sr::nHv*nthreads];
  long double* __restrict__ Hpop = NULL;
  long double tg=0, r = 0;
  long ii = 0;
  int tid = 0;
  
  
#pragma omp parallel default(shared) firstprivate(ii, Ne, Hpop, tid, tg, r) num_threads(nthreads)
  {

    tid = omp_get_thread_num();
    Hpop = &Hpop_all[tid*sr::nHv];

#pragma omp for schedule(dynamic,8)
    for(ii=0; ii<ntot; ++ii){
      tg = Tg[ii], r = Rho[ii];
      Ne[ii] = Ne_from_Rho(tg, r, Hpop, abund, avweight, mu);
    }
  }

  
  delete [] Hpop_all;
}

// ********************************************************* //

template<typename U, typename T>
static void _fill_nH_LTE_6(long const nLev, U const& temp,  U const& Pg,
			   U const& Pe, double* const __restrict__ H, std::vector<T> const& abund) 
{
  constexpr static const U BK = 1.3806488E-16;
  
  if(nLev > 6){
    fprintf(stderr,"fill_nH_LTE_6: nLev cannot be larger than 6!");
    return;
  }
  
  // --- Energy [ergs]: {0.000, 82258.211, 97491.219, 102822.766, 105290.508, 109677.617} * H * C
  constexpr static const U ELEV[6] = {0.0, 1.63401468e-11, 1.93661011e-11, 2.04251840e-11,
				      2.09153875e-11, 2.17868629e-11};
  constexpr static const U G[6]    = {2,8,18,32,50,1};
  
  double const BKT = phyc::BK<double> * temp;
  
  
  // --- solve EOS for H --- //
  
  double iPe = Pe, iPg = Pg,  itemp = temp;
  
  //eos.gasc<U>(itemp, iPe, iPg, pp);
  double pp[sr::nHv]  = {};
  sr::compute_Pe<double>(itemp, iPg, iPe, abund, pp);
  
  // --- copy total H I and H II --- //
  
  double const nH1 = pp[1];
  double const nH2 = pp[2];
  
  
  // --- Solve Boltzmann eq. for bound levels --- //
  
  double nH1tot = 0;
  int const nLev1 = nLev-1;
  pp[0]     = 1.0;
  pp[nLev1] = nH2;
  
  for(int ii = 1; ii<nLev1; ++ii){
    pp[ii] = (G[ii] / G[0]) * exp(-ELEV[ii]/BKT); // nH_i / nH_0
    nH1tot += pp[ii]; // nHI / nHI_0
  }
  nH1tot += 1.0; // nH1_0 / nH1_0
  
  // --- Calculate (nHI_i / nHI_0) x nHI_0 to get the actual level populations --- //
  
  nH1tot = nH1 / nH1tot;
 
  for(int ii=0; ii<nLev1; ++ii) pp[ii] *= nH1tot;
  for(int ii=0; ii<nLev; ++ii) H[ii] = pp[ii];
  
}

// ********************************************************* //


template<typename T> inline
void getH(long const nTotal, long const nH, const T* const __restrict__ Tg, const T* const __restrict__ Pg,
	  const T* const __restrict__ Ne, double* const __restrict__ Hpop, int const nthreads)
{
  static std::vector< double> const abund =
    {9.13226338e-01, 8.53214895e-02, 1.12475628e-11, 1.29139300e-11,
     3.63963999e-10, 3.31939113e-04, 1.02578992e-04, 7.78141233e-04,
     3.31939113e-08, 1.12475628e-04, 1.95460240e-06, 3.47582921e-05,
     2.69810240e-06, 3.24383256e-05, 2.57666779e-07, 1.48271755e-05,
     2.89106881e-07, 3.31939113e-06, 1.51725448e-07, 2.09439421e-06,
     1.15095522e-09, 8.93425674e-08, 9.14236232e-09, 4.27620413e-07,
     2.24418383e-07, 3.16999391e-05, 7.60428576e-08, 1.62576747e-06,
     1.48271755e-08, 3.63963999e-08, 6.93519103e-10, 2.34994895e-09,
     2.14317892e-10, 2.04671998e-09, 3.89994452e-10, 1.55259588e-09,
     3.63963999e-10, 7.26203652e-10, 1.58876048e-10, 2.89106881e-10,
     2.40468630e-11, 7.60428576e-11, 1.00244009e-20, 6.32496940e-11,
     1.20519807e-11, 4.47773542e-11, 7.96266470e-12, 6.62305602e-11,
     2.63668609e-11, 9.14236232e-11, 9.14236232e-12, 1.58876048e-10,
     2.95841045e-11, 1.55259588e-10, 1.20519807e-11, 1.23327074e-10,
     1.51725448e-11, 3.24383256e-11, 4.68876459e-12, 2.89106881e-11,
     1.00244009e-20, 9.14236232e-12, 2.95841045e-12, 1.20519807e-11,
     1.15095522e-12, 1.15095522e-11, 1.66363646e-12, 7.78141233e-12,
     9.14236232e-13, 1.09915371e-11, 5.26088040e-12, 6.93519103e-12,
     1.23327074e-12, 1.17776442e-11, 1.70238753e-12, 2.57666779e-11,
     2.04671998e-11, 5.76844065e-11, 9.35531529e-12, 1.12475628e-11,
     7.26203652e-12, 6.47229686e-11, 4.68876459e-12, 1.00244009e-20,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20, 1.00244009e-20,
     1.00244009e-20, 1.20519807e-12, 1.00244009e-20, 3.09783604e-13,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20, 1.00244009e-20,
     1.00244009e-20, 1.00244009e-20, 1.00244009e-20};


  
  long ii = 0;
  int tid = 0;
  double iTg=0, iPe=0, iPg=0;
  
#pragma omp parallel default(shared) firstprivate(ii, tid, iTg, iPe, iPg) num_threads(nthreads)
  {
    tid = omp_get_thread_num();
    
#pragma omp for schedule(dynamic, 8)
    for(ii=0; ii<nTotal; ++ii){

      iTg = Tg[ii];
      iPg = Pg[ii];
      iPe = Ne[ii]*phyc::BK<double>*iTg;
      
      _fill_nH_LTE_6<double>(nH, iTg, iPg, iPe, &Hpop[nH*ii], abund);
      
    }// ii
  } // pragma block
}
