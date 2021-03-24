#ifndef PHYC_H
#define PHYC_H

/* --------------------------------------------------------
   Physical constants in SI units
   Source: most of them are taken from NIST
   -------------------------------------------------------- */

namespace phyc{
  template<typename T = double> static constexpr T PI      = 3.1415926535897932384;     // Pi number

  template<typename T = double> static constexpr T CC   = 2.99792458E+10;            // Speed of light [m/s]
  template<typename T = double> static constexpr T BK   = 1.3806488E-16;             // Boltzmann constant [J/K]
  template<typename T = double> static constexpr T QE   = 4.80320441E-10;           // electron charge [C]
  template<typename T = double> static constexpr T ME   = 9.10938188E-28;            // electron mass [kg]
  template<typename T = double> static constexpr T SQRTME   = 3.018175256674138e-14;        // SQRT electron mass [kg]
  template<typename T = double> static constexpr T  EV  = 1.60217733E-12;            // One electronVolt [J]
 
  template<typename T = double> static constexpr T AMU  = 1.660538921E-24;           // Atomic mass unit [kg]
  template<typename T = double> static constexpr T HH   = 6.62606957E-27;            // Planck constant [Js]
  template<typename T = double> static constexpr T EPS0 = 1/(4*PI<T>);           // Vacuum permittivity [F/m]
  template<typename T = double> static constexpr T MU0  = 1;//1.2566370614E-06;          // Magnetic induct. of vac.
  //template<typename T = double> static constexpr T RBOHR= 5.29177349E-11;            // Bohr radius [m]           

  //template<typename T = double> static constexpr T Ryd  = 10973731.568508;           // Rydberg constant [m^-1]
  template<typename T = double> static constexpr T EV_TO_J  = 1;//QE<T>;
  template<typename T = double> static constexpr T CM_TO_M  = 1;//1.0E-2;
  template<typename T = double> static constexpr T NM_TO_M  = 1.0E-9;
  template<typename T = double> static constexpr T G_TO_KG = 1;//1.E-3;
  template<typename T = double> static constexpr T GAUSS_TO_TESLA = 1;//1.0E-4;
  template<typename T = double> static constexpr T CM3_TO_M3 = 1;//CM_TO_M<T>*CM_TO_M<T>*CM_TO_M<T>;
  template<typename T = double> static constexpr T GRCM3_TO_KGM3 = G_TO_KG<T> / CM3_TO_M3<T>;
  template<typename T = double> static constexpr T BA_TO_PA = 1;//0.1;
  template<typename T = double> static constexpr T MEGABARN_TO_CM2 = 1.0E-20;
  
  template<typename T = double> static constexpr T GAMMA   = 0.5772156649; 
  template<typename T = double> static constexpr T SQRTPI  = 1.7724538509055158819;     // sqrt of pi

  template<typename T = double> static constexpr T ERG_TO_J = 1;//1.0E-7;
  template<typename T = double> static constexpr T LARMOR   =  phyc::QE<T> / (4 * phyc::PI<T> * phyc::ME<T> * phyc::CC<T>);

}


#endif
