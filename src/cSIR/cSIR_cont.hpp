#ifndef CSIRCONT
#define CSIRCONT
/* ---
   Continuum opacities, most of the opacity sources are extracted from 
   the opacity package written by A. Asensio Ramos and ported to C++
   by myself.

   Coded by J. de la Cruz Rodriguez (ISP-SU, 2021)
  
  --- */
#include <array>
#include "physical_constants.hpp"
#include "cSIR_eos.hpp"

namespace sr{
  
  template<typename T>
  inline T Hminus_FF(T const Tg, T const Nein, T const lam0, T &dTg, T &dNe){
    
    //-----------------------------------------------------------------
    // Calculates the negative hydrogen (H-) free-free continuum absorption coefficient per
    // hydrogen atom (cm^2/m^2)
    // REF: John 1989 A&A 193, 189
    // INPUT:
    // T :   temperature in K (1400 < T < 100080)
    // Ne:   electron density
    // lam0: lambda_0 in [cm]
    //
    // Modifications:
    //              JdlCR: adapted to use SI units if needed (if BA_TO_PA != 1 and CM_TO_M != 1)
    // 
    // -----------------------------------------------------------------

    constexpr static const T a1[6] = {0.,2483.346,-3449.889,2200.04,-696.271,88.283};
    constexpr static const T b1[6] = {0.,285.827,-1158.382,2427.719,-1841.4,444.517};
    constexpr static const T c1[6] = {0.,-2054.291,8746.523,-13651.105,8624.97,-1863.864};
    constexpr static const T d1[6] = {0.,2827.776,-11485.632,16755.524,-10051.53,2095.288};
    constexpr static const T e1[6] = {0.,-1341.537,5303.609,-7510.494,4400.067,-901.788};
    constexpr static const T f1[6] = {0.,208.952,-812.939,1132.738,-655.02,132.985};
    constexpr static const T a2[4] = {518.1021,473.2636,-482.2089,115.5291};
    constexpr static const T b2[4] = {-734.8666,1443.4137,-737.1616,169.6374};
    constexpr static const T c2[4] = {1021.1775,-1977.3395,1096.8827,-245.649};
    constexpr static const T d2[4] = {-479.0721,922.3575,-521.1341,114.243};
    constexpr static const T e2[4] = {93.1373,-178.9275,101.7963,-21.9972};
    constexpr static const T f2[4] = {-6.4285,12.36,-7.0571,1.5097};
    
    constexpr static const T C =  1.e-29 *phyc::CM_TO_M<T>*phyc::CM_TO_M<T>;
    constexpr static const T BK = phyc::BK<T> / phyc::ERG_TO_J<T>; // Must be in ERG if CGS
    
    T const Ne = Nein * phyc::CM3_TO_M3<T>; // convert to CGS
    constexpr static T const dNein_dNE =  1/phyc::CM3_TO_M3<T>;
    
    T const KT = Tg * BK; // in erg
    T const lambda = lam0 * 1.e4f / phyc::CM_TO_M<T>; // to microns

    if(lambda < 0.18){
      dTg = 0; dNe = 0;
      return static_cast<T>(0);
    }
    
    T const lambda2 = lambda*lambda;

    T const theta  = 5040 / Tg;
    T const stheta = sqrt(theta);
    T dtheta = -theta/Tg;
    
    T const t15 = theta*stheta;
    T const t20 = theta*theta;
    T const t25 = t20*stheta;

    T res = 0, dres = 0;

    // --- wavelength dependence --- //
    
    if(lambda < 0.3645){
      std::array<T,4> com2;
      
      for(int ii=0; ii<4; ++ii)
	com2[ii]  = a2[ii] * lambda2 + b2[ii] + c2[ii]/lambda + d2[ii]/lambda2 + e2[ii]/(lambda2*lambda) + f2[ii]/(lambda2*lambda2);
      
      
      res =   com2[0]*theta  +      com2[1]*t15           +   com2[2]*t20    +      com2[3]*t25;
      dres = (com2[0]        + 1.5f*com2[1]*stheta        + 2*com2[2]*theta  + 2.5f*com2[3]*t15) * dtheta;
      
    }else{
      std::array<T,6> com1;
      T const t30 = theta*t20;
      T const t35 = t30*stheta;
    
      for(int ii=0; ii<6; ++ii)
	com1[ii] = a1[ii] * lambda2 + b1[ii] + c1[ii]/lambda + d1[ii]/lambda2 + e1[ii]/(lambda2*lambda) + f1[ii]/(lambda2*lambda2);
      
      res =   com1[0]*theta +      com1[1]*t15     +   com1[2]*t20    +      com1[3]*t25 +   com1[4]*t30 +      com1[5]*t35;
      dres = (com1[0]       + 1.5f*com1[1]*stheta  + 2*com1[2]*theta  + 2.5f*com1[3]*t15 + 3*com1[4]*t20 + 3.5f*com1[5]*t25)*dtheta;
    }

    // --- derivatives and final result --- //
    
    T const res1 = res * Ne * KT * C;
    
    dNe = (res1 / Ne) * dNein_dNE;
    dTg = (dres*Tg + res)*C * Ne * BK;
    
    return  res1;
  }
  
  // ****************************************************************************** //
    
  template<typename T>
  inline T Hminus_BF(T const Tg, T const Nein, T const lam_in, T &dTg, T &dNe){
    // -----------------------------------------------------------------
    // Calculates the negative hydrogen (H-) bound-free continuum absorption coefficient per
    // hydrogen atom (cm^2/m^2)
    // REF: John 1989 A&A 193, 189 (with a couple of mistakes)
    // INPUT:
    //      T : temperature in K
    //	    Pe: electron density
    //	    lambda: wavelength in cm 
    // -----------------------------------------------------------------
    using namespace phyc;
    constexpr static const T lambda0 = 1.6419; // [microns]
    constexpr static const T   alpha = HH<double>*CC<double>/BK<double> * 1.e4/phyc::CM_TO_M<double>; // [K * micron]
    constexpr static const T     cte = 0.75e-18;
    constexpr static const T C = phyc::CM_TO_M<T>*phyc::CM_TO_M<T>;
    constexpr static const T cc[6] = {152.519,49.534,-118.858,92.536,-34.194,4.982};
    constexpr static const T BK = phyc::BK<T> / phyc::ERG_TO_J<T>; // Must be in ERG if CGS
    constexpr static T const dNein_dNE =  1/phyc::CM3_TO_M3<T>;

    T const Ne = Nein*phyc::CM3_TO_M3<T>;
    T const KT =  Tg * BK;
    T const Pe = KT * Ne;
    
    T const lambda = lam_in * 1.e4f / phyc::CM_TO_M<T>;
    T res = 0, dres = 0;
    
    if(lambda < lambda0){
      T const com = 1/lambda - 1/lambda0;
      T const scom = sqrt(com);
      T const com2 = com*com;
      
      T const eps1 = exp(alpha/(Tg*lambda0));
      T const eps2 = exp(-alpha/(Tg*lambda));
      
      T const sigma = (cc[0] + cc[1]*scom + cc[2]*com + cc[3]*com*scom + cc[4]*com2 + cc[5]*com2*scom)*(lambda*lambda*lambda) * com * scom;
      res = (eps1 * (1-eps2) * cte * sigma) / (Tg*Tg*sqrt(Tg));
      dres = res * (-2.5f/Tg - alpha/(lambda0*Tg*Tg) - (eps2 * alpha/(Tg*Tg*lambda)) / (1-eps2) );
    }

    dTg = C*(dres * Pe + res* Ne * BK);
    dNe = ( res * C * KT) * dNein_dNE ;
    
    return res * Pe * C;
  }
 
  // ****************************************************************************** //

  template<typename T>
  inline T Thompson(T const Ne, T &dNe)
  {
    //-----------------------------------------------------------------
    // hydrogen atom (cm^-1/m^-1) due to Thomson scattering (scattering with free electrons)
    //  INPUT:
    //		Pe: electron pressure
    //		NH: neutral H atoms partial density
    //-----------------------------------------------------------------

    using namespace phyc;
    constexpr static const double C0 = (QE<double>*QE<double>) / (4*PI<double>*ME<double>*CC<double>*CC<double>);
    constexpr static const T C = (8*PI<double>/3) * C0*C0 / (CM_TO_M<double>*CM_TO_M<double>);
    
    dNe = C;
    
    return C * Ne;
  }
  
  // ****************************************************************************** //

  template<typename T, size_t N> inline
  T const interpol_one(T const (&x)[N], T const (&y)[N], T const &xxi)
  {
    size_t idx = 1;
    T const xx = std::min<T>(std::max<T>(x[0], xxi), x[N-1]);
    
    for(size_t ii=1; ii<N; ++ii){
      if(xx<=x[ii]) idx = ii;
      else break;
    }

    T const u = (x[idx]-xx)/(x[idx]-x[idx-1]);
    
    return y[idx]*(1-u) + y[idx]*u;
  }

  // ****************************************************************************** //

  template<typename T>
  inline T Rayleigh_H2(T const nH2, double const lambda_in, T &dnH2)
  {
    static constexpr const T a[3] =  {8.779E+01, 1.323E+06, 2.245E+10};
    static constexpr const size_t Nlam = 21;
      T lambdaRH2[Nlam] = {121.57, 130.00, 140.00, 150.00, 160.00, 170.00, 185.46,
			 186.27, 193.58, 199.05, 230.29, 237.91, 253.56, 275.36,
			 296.81, 334.24, 404.77, 407.90, 435.96, 546.23, 632.80};
    static constexpr const
      T sigma[Nlam] = {2.35E-06, 1.22E-06, 6.80E-07, 4.24E-07, 2.84E-07, 2.00E-07, 1.25E-07,
		     1.22E-07, 1.00E-07, 8.70E-08, 4.29E-08, 3.68E-08, 2.75E-08, 1.89E-08,
		     1.36E-08, 8.11E-09, 3.60E-09, 3.48E-09, 2.64E-09, 1.04E-09, 5.69E-10 };

    static constexpr const T C = phyc::MEGABARN_TO_CM2<T> * phyc::CM_TO_M<T>;

    
    double const lambda = lambda_in * (1.e7 / phyc::CM_TO_M<double>);; // to nm
    
    T sigma_RH2 = 0;
    dnH2 = 0;
    
    if(lambda >= lambdaRH2[0]){
      if(lambda <=  lambdaRH2[Nlam-1]){
	sigma_RH2 =  C * interpol_one<T,Nlam>(lambdaRH2, sigma, lambda);
      }else{
	T const lambda2 = lambda*lambda;
	sigma_RH2 = C * (a[0] + (a[1] + a[2]/lambda2) / lambda2) / (lambda2*lambda2);
      }
    }

    dnH2 = sigma_RH2;
    return sigma_RH2 * nH2;
  }
   
  // ****************************************************************************** //

  template<typename T>
  inline T hydrogen(T const Tg, double lambda_in, T &dTg)
  {
    //-----------------------------------------------------------------
    // Calculates the hydrogen (H) bound-free and free-free continuum absorption coefficient per
    // hydrogen atom (cm^2/m^2)
    // REF: Landi Degl'Innocenti 1976, A&ASS, 25, 379
    // INPUT:
    // T : temperature in K (T < 12000)
    // lambda: wavelength in m/cm
    // -----------------------------------------------------------------
    static constexpr const T r   = 1.096776e-3;
    static constexpr const T c1  = 1.5777216e5;
    static constexpr const T c2  = 1.438668e8;
    static constexpr const T cte = 1.045e-26 * phyc::CM_TO_M<T>*phyc::CM_TO_M<T>;
    
    T const lambda = lambda_in * (1.e8 / phyc::CM_TO_M<double>); // to Angstroms
    T const theta1 = c1 / Tg;
    T const dtheta1 = - theta1 / Tg;

    T const theta2 = c2 / static_cast<T>(lambda * Tg);
    T const dtheta2 = - theta2 / Tg;

    T const theta3 = 2*theta1;
    T const dtheta3 = 2*dtheta1;

    // Lowest level which can be photoionized
    int const n0 = 1 + floor(sqrt(r*lambda));
    int const n02 = n0*n0;
    
    // Sum over states that can be photoionized
    T sum = 0, dsum = 0;
    if (n0 <= 8){
      sum = exp(theta1 / n02) / (n0*n02);
      dsum = sum * (dtheta1 / n02);
      
      for(int ii=n0+1; ii<=8; ++ii){
	int const ii2 = ii*ii;
	T const tmp = exp(theta1 / ii2) / (ii2*ii);
	sum += tmp;
	dsum += tmp * dtheta1 / ii2;
      }
      T const a = 0.117f / theta3;
      T const b = exp(theta1/81) / theta3;
      
      sum += a + b;
      dsum += -(a+b)*(dtheta3/theta3) + b*(dtheta1/81);
    }else{
      T const a = 0.117f / theta3;
      T const b = exp(theta1/n02) / theta3;
      
      sum = a + b;
      dsum = -(a+b)*dtheta3/theta3 + (dtheta1/n02);
    }
    
    // Approximate the value of the Gaunt factor G_ff from Mihalas Eq (80) @ theta=1, x=0.5
    T const eps1 = exp(-theta1);
    T const eps2 = exp(-theta2);

    T const lambda3 = lambda*lambda*lambda;
    
    T const gff = (1-eps2) * eps1 * lambda3;
    T const dgff = (eps1*eps2*dtheta2 - (1-eps2)*eps1*(dtheta1)) * lambda3;

    dTg = cte*(dgff*sum + gff*dsum);
    
    return gff*cte*sum;
  }
  
  // ****************************************************************************** //
  
  template<typename T>
  inline T Rayleigh_H(double const lambda_in)
  {
    // --- It is so small that I rather not use it... ---//
    // --- lambda in cm/m --- //
    constexpr static const T c1 = 5.799e-13;
    constexpr static const T c2 = 1.422e-6;
    constexpr static const T c3 = 2.784e0;
    constexpr static const T C = phyc::CM_TO_M<T>*phyc::CM_TO_M<T>;
    
    T const lambda = (lambda_in / phyc::CM_TO_M<double>) * 1.0e8; 
    T const l2 = lambda*lambda;
    
    return C*(c1+(c2+c3/l2) / l2) / (l2*l2);
  }

  // ****************************************************************************** //

  template<typename T>
  T continuum_absorption(T const &Tg, T const &Ne, double const lambda, T* const __restrict__ H, T &dTg, T &dNe)
  {
    //
    // Computes the continuum absortion using A. Asensio's routines
    // The Rayleigh H2 opacity is taken from the RH code
    // All subroutines internally should be able to deal with CGS or SI
    // units if physical_constants.hpp are defined correctly
    //
    // Hminus and hydr are computed per neutral hydrogen density (see Landi Degl'Innocenti 1976)
    //
    
    //T const& nHtot   = H[0];
    T const& nHI     = H[1];
    //T const& nHmin   = H[3];
    T const& nH2     = H[5];
    
    //T const& dnHtot  = H[sr::nHv];
    T const& dnHI    = H[sr::nHv+1];
    //T const& dnHmin  = H[sr::nHv+3];
    T const& dnH2    = H[sr::nHv+5];
    
    //T const& ddnHtot = H[2*sr::nHv];
    T const& ddnHI   = H[2*sr::nHv+1];
    //T const& ddnHmin = H[2*sr::nHv+3];
    T const& ddnH2   = H[2*sr::nHv+5];

    
    T dTg_tmp=0, dNe_tmp=0, dtmp=0;
    dNe = 0, dTg = 0;

    // --- Thompson opacity --- //
    T const thom = Thompson<T>(Ne, dNe);

    
    // --- Hydrogen BB and BF ---//
    T const hydr = hydrogen<T>(Tg, lambda, dTg_tmp); 
    dTg += dTg_tmp*nHI + dnHI*hydr;
    dNe += hydr*ddnHI;


    // --- H2 Raileigh scattering --- //
    T const RayH2 = Rayleigh_H2<T>(nH2, lambda, dtmp);
    dTg += dtmp*dnH2;
    dNe += dtmp*ddnH2;

    
    // --- Hminus BF --- //
    T const HminBF = Hminus_BF<T>(Tg, Ne, lambda, dTg_tmp, dNe_tmp);
    dTg += dTg_tmp*nHI + HminBF*dnHI;
    dNe += dNe_tmp*nHI + HminBF*ddnHI;


    // --- Hminus FF --- //
    T const HminFF = Hminus_FF<T>(Tg, Ne, lambda, dTg_tmp, dNe_tmp);
    dTg += dTg_tmp*nHI + HminFF*dnHI;
    dNe += dNe_tmp*nHI + HminFF*ddnHI;


    
    return thom + (hydr + HminFF + HminBF)*nHI + RayH2;
  }
  
  // ****************************************************************************** //

  template<typename U>
  inline U H1min_alternative(U const XH1, U const XHMIN, U const lambda, U const T, U const XNE)
  {
    double const HKT = phyc::HH<double> / (T * phyc::BK<double>);
    double const FREQ = 2.997925E10 / lambda;
    double const TKEV = 8.6171E-5*T;
    double const EHVKT=exp(-FREQ*HKT);

    U H=0,HMINBF,HMINFF=0,HMIN=0;

    double const FREQ1=FREQ*1.E-10;
    U const B=(1.3727E-15+4.3748/FREQ)/FREQ1;
    U const C=-2.5993E-7/pow(FREQ1,2);

   if(FREQ <= 1.8259E14) HMINBF=0.;
   else if(FREQ >= 2.111E14) HMINBF=6.801E-10+(5.358E-3+(1.481E3+(-5.519E7+4.808E11/FREQ1)/FREQ1)/FREQ1)/FREQ1;
   else HMINBF=3.695E-6+(-1.251E-1+1.052E3/FREQ1)/FREQ1;

   HMINFF=(B+C/T)*XH1*XNE*2.E-20;

   
  /*
    We use the number density / partition function for H-.
    The partition function for H- is 1 and not 2 as was mistakenly
    used before (fixed 2007-12-15, NP)! The EOS calculations are
    good up to 7730K, we use Kurucz approximation for higher T.
  */
   if(T < 7730.) HMIN=XHMIN;
   else HMIN=exp(0.7552/TKEV)/(2.*2.4148E15*T*sqrt(T))*XH1*XNE;
   
   H=HMINBF*(1-EHVKT)*HMIN*1.E-10;
   return H+HMINFF;
  }
  
  // ****************************************************************************** //

  template<typename T>
  inline T getHminus(T const &Tg, T const &Ne, T const &nH1)
  {
    constexpr static const double CSAHA = (phyc::HH<double> / (2.0 * phyc::PI<double> * phyc::ME<double>)) *
      (phyc::HH<double> / phyc::BK<double>);
    constexpr static const double EION_HMIN_BK = 0.754 * phyc::EV<double>/phyc::BK<double>;
    
    double const dum = CSAHA / Tg;
    double const PhiHmin = 0.25 * dum*sqrt(dum)*exp(EION_HMIN_BK / Tg);

    return PhiHmin * double(Ne) * double(nH1);
  }
  
  // ****************************************************************************** //

  template<typename T>
  inline T continuum_absorption_bif(T const &Tg, T const &Ne, double const lambda, T* const __restrict__ H)
  {
    T const nHI     = H[0];
    T const nH2     = H[1];    
    T const nHmin   = getHminus(Tg, Ne, nHI);

    T dNe_tmp=0, dTg_tmp=0;
    
    T const Hmin =  H1min_alternative(nHI / 2.0, nHmin, lambda, Tg, Ne);
    
    // --- Thompson opacity --- //
    T const thom = Thompson<T>(Ne, dNe_tmp);

    
    // --- Hydrogen BB and BF ---//
    T const hydr = hydrogen<T>(Tg, lambda, dTg_tmp); 

    return thom + (hydr)*nHI + Hmin;
  }

  
  // ****************************************************************************** //

  template<typename T>
  inline T continuum_absorption(T const &Tg, T const &Ne, double const lambda, T* const __restrict__ H)
  {
    T const nHI     = H[1];
    T const nH2     = H[5];
    T const nHmin   = H[3];
    T dTg_tmp=0, dNe_tmp=0, dtmp=0;


    T const Hmin =  H1min_alternative(nHI / 2.0, nHmin, lambda, Tg, Ne);
    
    // --- Thompson opacity --- //
    T const thom = Thompson<T>(Ne, dNe_tmp);

    
    // --- Hydrogen BB and BF ---//
    T const hydr = hydrogen<T>(Tg, lambda, dTg_tmp); 


    // --- H2 Raileigh scattering --- //
    T const RayH2 = Rayleigh_H2<T>(nH2, lambda, dtmp);

    // --- Hminus BF --- //
    //T const HminBF = Hminus_BF<T>(Tg, Ne, lambda, dTg_tmp, dNe_tmp);

    // --- Hminus FF --- //
    //T const HminFF = Hminus_FF<T>(Tg, Ne, lambda, dTg_tmp, dNe_tmp);

    //return Hmin + nHI*hydr + thom;
    return thom + (hydr)*nHI + Hmin + RayH2;

    //return thom + (hydr + HminFF + HminBF)*nHI + RayH2;
  }

  // ****************************************************************************** //

  template<typename T>
  inline T continuum_absorption_nu(T const &Tg, T const &Ne, double const &nu, T* const __restrict__ H)
  {
    double const lambda = phyc::CC<double> / nu;
    
    T const nHI = H[1];
    T const nH2 = H[5];
    
    T dTg_tmp=0, dNe_tmp=0, dtmp=0;

    // --- Thompson opacity --- //
    T const thom = Thompson<T>(Ne, dNe_tmp);

    
    // --- Hydrogen BB and BF ---//
    T const hydr = hydrogen<T>(Tg, lambda, dTg_tmp); 


    // --- H2 Raileigh scattering --- //
    //T const RayH2 = Rayleigh_H2<T>(nH2, lambda, dtmp);

    
    // --- Hminus BF --- //
    T const HminBF = Hminus_BF<T>(Tg, Ne, lambda, dTg_tmp, dNe_tmp);



    // --- Hminus FF --- //
    T const HminFF = Hminus_FF<T>(Tg, Ne, lambda, dTg_tmp, dNe_tmp);

    
    return thom + (hydr + HminFF + HminBF)*nHI;// + RayH2;
  }
}

#endif

