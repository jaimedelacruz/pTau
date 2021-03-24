#ifndef SAHAHPP
#define SAHAHPP

//#include "physical_constants.hpp"

namespace sr{
  
  template<typename T>
  constexpr inline T pow10__(T const &x){return exp(static_cast<T>(2.3025850929940459)*x);}
  
  // ************************************************************************************** //

  template<typename T>
  inline T Saha(T const Tg, T const Pe, T const Eion_eV, T const u1, T const u2)
  {
    // --- This constant is calculated to operate with Pe --- //
    static const T saha_const = static_cast<T>(2*phyc::BK<double>*pow((2.0 * phyc::PI<double> * phyc::ME<double> * phyc::BK<double>) /
								      (phyc::HH<double>*phyc::HH<double>), 1.5));

    return saha_const * (u2/u1) * Tg * Tg * sqrt(Tg) * exp(-(Eion_eV*phyc::EV<T>)/(phyc::BK<T>*Tg)) / Pe;
    //T const theta=5040/Tg;
    //return u2 * pow10__<T>(9.0805126f-theta*Eion_eV)/(u1*Pe*pow(theta,2.5f));
  }
  
  // ************************************************************************************** //
  
 
}

#endif
