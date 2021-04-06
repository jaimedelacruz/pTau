#ifndef CSIRATMHPP
#define CSIRATMHPP
/* ---

   Wittmann/Mihalas-like EOS helper tools and PF.
   Implementation extracted from the SIR code (Ruiz Cobo & del Toro Iniesta 1992)
   with some improvements added, like the loweing of the ionization potential.

   Coded by J. de la Cruz Rodriguez (ISP-SU, 2021)

   --- */

#include <cstdio>
#include <vector>
#include <array>

namespace sr{

  // ****************************************************************************** //

  // --- it is like a typedef, to create an array of pointers to functions --- //
  
  template<typename T>
  using pf_funct = void (*)(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3);

  // ****************************************************************************** //
  
  template<typename T>
  static constexpr const  T AMASS[99] =
    {1.008,  4.003,  6.941,  9.012, 10.811, 12.011, 14.007, 15.999,
     18.998, 20.179, 22.990, 24.305, 26.982, 28.086, 30.974, 32.060,
     35.453, 39.948, 39.102, 40.080, 44.956, 47.900, 50.941, 51.996,
     54.938, 55.847, 58.933, 58.710, 63.546, 65.370, 69.720, 72.590,
     74.922, 78.960, 79.904, 83.800, 85.468, 87.620, 88.906, 91.220,
     92.906, 95.940, 98.906,101.070,102.905,106.400,107.868,112.400,
     114.820,118.690,121.750,127.600,126.905,131.300,132.905,137.340,
     138.906,140.120,140.908,144.240,146.000,150.400,151.960,157.250,
     158.925,162.500,164.930,167.260,168.934,170.040,174.970,178.490,
     180.948,183.850,186.200,190.200,192.200,195.090,196.967,200.590,
     204.370,207.190,208.981,210.000,210.000,222.000,223.000,226.025,
     227.000,232.038,230.040,238.029,237.048,242.000,242.000,245.000,
     248.000,252.000,253.000};

  // ****************************************************************************** //

  template<typename T>
  static constexpr const T EION[2][99] =
    // ---- neutral --- //
    {{13.595,24.58,5.39,9.32,8.298,11.256,14.529,13.614,17.418, 
     21.559,5.138,7.644,5.984,8.149,10.474,10.357,13.012,15.755,4.339, 
     6.111,6.538,6.825,6.738,6.763,7.432,7.896,7.863,7.633,7.724,9.391, 
     5.997,7.88,9.81,9.75,11.840,13.996,4.176,5.692,6.377,6.838,6.881, 
     7.10,7.28,7.36,7.46,8.33,7.574,8.991,5.785,7.34,8.64,9.01,10.454, 
     12.127,3.893,5.210,5.577,5.466,5.422,5.489,5.554,5.631,5.666,6.141 
     ,5.852,5.927,6.018,6.101,6.184,6.254,5.426,6.650,7.879,7.980,7.870 
     ,8.70,9.10,9.00,9.22,10.43,6.105,7.415,7.287,8.43,9.30,10.745, 
      4.,5.276,6.9,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.},
     // --- singly ionized --- //
     {0.,54.403,75.62,18.21,25.15,24.376,29.59,35.11,34.98,41.07 
      ,47.290,15.03,18.823,16.34,19.72,23.405,23.798,27.62,31.81,11.868, 
      12.891,13.63,14.205,16.493,15.636,16.178,17.052,18.15,20.286,17.96 
      ,20.509,15.93,18.63,21.50,21.60,24.565,27.50,11.027,12.233,13.13, 
      14.316,16.15,15.26,16.76,18.07,19.42,21.48,16.904,18.86,14.63, 
      16.50,18.60,19.09,21.20,25.10,10.001,11.060,10.850,10.550,10.730, 
      10.899,11.069,11.241,12.090,11.519,11.670,11.800,11.930,12.050, 
      12.184,13.900,14.900,16.2,17.7,16.60,17.00,20.00,18.56,20.50,18.75 
      ,20.42,15.03,16.68,19.,20.,20.,22.,10.144,12.1,12.,12.,12.,12.,12.,12.,
      12.,12.,12., 12}
    };
  
  // ****************************************************************************** //

  template<typename T>
  inline T vac2air(T const& alamb){
    if(alamb < 2000.) return alamb;
    else return alamb/(1.0+2.735182e-4+131.4182/alamb/alamb+ 
		       2.76249e8/alamb/alamb/alamb/alamb);
  }

  // ****************************************************************************** //

  template<typename T>
  inline T air2vac(T const& lambda_air){

    T lambda_vacuum = ((lambda_air > 2000) ? lambda_air / 1.00029 : lambda_air);
    
    T error = 1.0;
    while(error > 1.e-7){
      error = lambda_air - vac2air<T>(lambda_vacuum);
      lambda_vacuum = lambda_vacuum + error / 1.0029;
    }
    return lambda_vacuum;
  }
  
  // ****************************************************************************** //

  // ---- Hydrogen --- //
  
  template<typename T> inline void partition_0(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    if     (Tg > 13000.f){
      u1 = 1.51f+3.8e-5f*Tg;
      du1= 3.8e-5f;
    }else if(Tg > 16200.f){
      u1 = 11.41f+Tg*(-1.1428e-3f+Tg*3.52E-8f);
      du1 = -1.1428e-3f+2.0f*Tg*3.52E-8f;
    }else{
      u1  = static_cast<T>(2);
      du1 = static_cast<T>(0);
    }
      
    u2 = static_cast<T>(1);
    u3 = static_cast<T>(1); // to avoid dividing by zero
  } 

  // ****************************************************************************** //

   // ---- Helium --- //
  
  template<typename T> inline void partition_1(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    u1  = static_cast<T>(1);
    
    if(Tg > 30000.f){
      u1  = 14.8f+Tg*(-9.4103E-4f+Tg*1.6095E-8f);
      du1 = -9.4103E-4f + 2.0f*Tg*1.6095E-8f;
    }
      
    u2 = static_cast<T>(2);
    u3 = static_cast<T>(1);
  }

  // ****************************************************************************** //

   // ---- Lithium --- //
  
  template<typename T> inline void partition_2(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const Y = 1.e-3f * Tg;
    u1=2.081f-Y*(6.8926E-2f-Y*1.4081E-2f);
    du1=1.e-3f*(-6.8926e-2f+2.0f*Y*1.4081e-2f); // includes dYdT
    if(Tg > 6000.f){
      u1=3.4864f+Tg*(-7.3292E-4f+Tg*8.5586E-8f);
      du1=(-7.3292e-4f+2.0f*Tg*8.5586e-8f);
    }
    u2 = static_cast<T>(1);
    u3 = static_cast<T>(2);

  }

  
  // ****************************************************************************** //
  // ---- Be --- //
  
  template<typename T> inline void partition_3(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const tmp = 0.631f+7.032E-5f*Tg;
    
    if(tmp > 1.0f){
      u1 = tmp;
      du1 = 7.032e-5f;
    }else{
      u1 = static_cast<T>(1);
    }
    
    u2=static_cast<T>(2);
    u3=static_cast<T>(1);
  }

  // ****************************************************************************** //
  // ---- B --- //
  
  template<typename T> inline void partition_4(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    u1=5.9351f+1.0438E-5f*Tg;
    du1=1.0438e-5f;
    u2=static_cast<T>(1);
    u3=static_cast<T>(2);
  }

  // ****************************************************************************** //
  // ---- C --- //
  
  template<typename T> inline void partition_5(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const Y = 1.e-3*Tg;
    
    if(Tg > 1.2E4f){
      u1=13.97f+Tg*(-1.3907E-3f+Tg*9.0844E-8f);
      du1=(-1.3907e-3f+2*Tg*9.0844e-8f);
    }else{
      u1=8.6985f+Y*(2.0485E-2f+Y*(1.7629E-2f-3.9091E-4f*Y));
      du1=1.e-3f*((2.0485e-2f+2.0f*Y*1.7629e-2f-3.0f*Y*Y*3.9091e-4f));
    }

    if(Tg > 2.4e4f){
      u2=10.989f+Tg*(-6.9347E-4f+Tg*2.0861E-8f);
      du2=(-6.9347e-4f+2*Tg*2.0861e-8f);
    }else{
      u2=5.838f+1.6833E-5f*Tg;
      du2=1.6833e-5;
    }
    
    if(Tg > 1.95e4f){
      u3=-0.555f+8.e-5f*Tg;
      du3=8e-5f;
    }else{
      u3=static_cast<T>(1);
    }
  }

  // ****************************************************************************** //
  // ---- N --- //
  
  template<typename T> inline void partition_6(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {

    if(Tg > 1.8E4f){
      u1=11.396f+Tg*(-1.7139E-3f+Tg*8.633E-8f);
      du1=(-1.7139e-3f+2.0f*Tg*8.633e-8f);
    }else if(Tg > 8800.f){
      u1=2.171f+2.54E-4f*Tg;
      du1=2.54e-4f;
    }else{
      T const Y = 1.e-3*Tg;
      u1=3.9914f+Y*(1.7491e-2f-Y*(1.0148E-2f-Y*1.7138E-3f));
      du1=1.e-3f*((1.7491e-2f-2.0f*Y*1.0148e-2f+3.0f*Y*Y*1.7138e-3f));
    }

    if(Tg > 3.3E4f){
      u2=26.793f+Tg*(-1.8931E-3f+Tg*4.4612E-8f);
      du2=(-1.8931e-3f+2*Tg*4.4612e-8f);
    }else{
      u2=8.060f+1.420E-4f*Tg;
      du2=1.420e-4f;
    }

    if(Tg < 7310.5f){
      u3=5.89f;
    }else{
      u3=5.9835f+Tg*(-2.6651E-5f+Tg*1.8228E-9f);
      du3=(-2.6651e-5f+2.0f*Tg*1.8228e-9f);
    }
    
  }
  
  // ****************************************************************************** //
  // ---- O --- //
  
  template<typename T> inline void partition_7(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
 
    if(Tg > 1.9E4f){
      u1=66.81f+Tg*(-6.019E-3f+Tg*1.657E-7f);
      du1=(-6.019e-3f+2.0f*Tg*1.657e-7f);
    }else{
      u1=8.29f+1.10E-4f*Tg;
      du1=1.10e-4f;
    }

    if(Tg > 3.64E4f){
      u2=68.7f+Tg*(-4.216E-3f+Tg*6.885E-8f);
      du2=(-4.216e-3f+2.0f*Tg*6.885e-8f);
    }else{
      T const tmp = 3.51f+8.E-5f*Tg;
      if(tmp > 4.0f){
	u2 = tmp;
	du2 = 8.E-5f;
      }else{
	u2 = static_cast<T>(4);
	du2 = static_cast<T>(0);
      }
    }

    u3=7.865f+1.1348E-4f*Tg;
    du3=1.1348e-4f;
  }
  
  // ****************************************************************************** //
  // ---- F --- //
  
  template<typename T> inline void partition_8(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    if(Tg > 2.0e4f){
      u1=15.16f+Tg*(-9.229E-4f+Tg*2.312E-8f);
      du1=(-9.229e-4f+2.0f*Tg*2.312e-8f);
    }else if(Tg > 8750.f){
      u1=5.9;
      du1=0.f;
    }else{
      T const Y = 1.e-3*Tg;
      u1=4.5832f+Y*(.77683f+Y*(-.20884f+Y*(2.6771E-2f-1.3035E-3f*Y)));
      du1=(.77683f-2.0f*Y*.20884+3.0f*Y*Y*2.6771e-2f-4.0f*Y*Y*Y*1.3035e-3f)*1.e-3f;
    }
    
    u2=8.15f+8.9E-5f*Tg;
    du2=8.9e-5f;
    
    u3=2.315f+1.38E-4f*Tg;
    du3=1.38e-4f;
  }
  
  // ****************************************************************************** //
  // ---- Ne --- //
  
  template<typename T> inline void partition_9(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    if(Tg > 2.69e4f){
      u1=26.3f+Tg*(-2.113E-3f+Tg*4.359E-8f);
      du1=(-2.113e-3f+2.0f*Tg*4.359e-8f);
    }else{
      u1=static_cast<T>(1);
    }
    
    u2=5.4f+4.E-5f*Tg;
    du2=4.e-5f;
    
    u3=7.973f+7.956E-5f*Tg;
    du3=7.956e-5f;
  }
  
  // ****************************************************************************** //
  // ---- Na --- //
  
  template<typename T> inline void partition_10(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    if(Tg > 8.5e3f){
      u1=4.5568f+Tg*(-1.2415E-3f+Tg*1.3861E-7f);
      du1=(-1.2415e-3f+2.0f*Tg*1.3861e-7f);
    }else if(Tg > 5400.f){
      u1=-0.83f+5.66E-4f*Tg;
      du1=5.66e-4f;
    }else{
      T const tmp = 1.72f+9.3E-5f*Tg;
      if(tmp > 2.0){
	u1 = tmp;
	du1=9.3e-5;
      }else{
	u1 = static_cast<T>(2);
      }
    }
    
    u2=static_cast<T>(1);
    
    u3=5.69f+5.69E-6f*Tg;
    du3=5.69e-6f;
  }
  
  // ****************************************************************************** //
  // ---- Mg --- //
  
  template<typename T> inline void partition_11(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const X = log(5040.f/Tg);
    T const dX = -1.0f / Tg;
    
    if(Tg > 8.0e3f){
      u1=2.757f+Tg*(-7.8909E-4f+Tg*7.4531E-8f);
      du1=(-7.8909e-4f+2.0f*Tg*7.4531e-8f);
    }else{
      u1=1.0f+exp(-4.027262f-X*(6.173172f+X*(2.889176f+X*(2.393895f+.784131f*X))));
      du1=(-dX*(6.173172f+X*(2.0f*2.889176f+X*(3.0f*2.393895f+4.0f*X*.784131f))))*(u1-1.0f);
    }
    
    if(Tg > 2e4f){
      u2=7.1041f+Tg*(-1.0817e-3f+Tg*4.7841E-8f);
      du2=(-1.0817e-3f+2.0f*Tg*4.7841e-8f);
    }else{
      u2=2.0f+exp(-7.721172f-X*(7.600678f+X*(1.966097f+.212417f*X)));
      du2=(-dX*(7.0600678f+X*(2.0f*1.966097f+3.0f*X*.212417f)))*(u2-static_cast<T>(2));
    }

    u3 = static_cast<T>(1);
  }
  
  // ****************************************************************************** //
  // ---- Al --- //
  
  template<typename T> inline void partition_12(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const Y = 1.e-3f * Tg;
    
    u1=5.2955f+Y*(.27833f-Y*(4.7529E-2f-Y*3.0199E-3f));
    du1=1.e-3f*(.27833f-Y*(2.0f*4.7529e-2f-3.0f*Y*3.0199e-3f));
    
    if(Tg > 2.24e4f){
      u2=61.06f+Tg*(-5.987E-3f+Tg*1.485E-7f);
      du2=(-5.987e-3f+2.0f*Tg*1.485e-7f);
    }else{
      T const tmp = .725f+3.245E-5f*Tg;
      if(tmp > 1.f){
	u2  = tmp;
	du2 = 3.245e-5f;
      }else{
	u2 = 1.0f;
      }
    }
    
    if(Tg > 1.814E4f){
      u3=3.522f+Tg*(-1.59E-4f+Tg*4.382E-9f);
      du3=(-1.59e-4f+2.0f*Tg*4.382e-9f);
    }else{
      T const tmp = 1.976+3.43E-6*Tg;
      if(tmp > 2.0f){
	u3  = tmp;
	du3 =3.43e-6f;
      }else{
	u3 = 2.0f;
      }
    }  
  }
    
  // ****************************************************************************** //
  // ---- Si --- //
  
  template<typename T> inline void partition_13(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const Y = 1.e-3f * Tg;

    if(Tg > 1.04E4){
      u1=86.01f+Tg*(-1.465E-2f+Tg*7.282E-7f);
      du1=(-1.465e-2f+2.0f*Tg*7.282e-7f);
    }else{
      u1=6.7868f+Y*(.86319f+Y*(-.11622f+Y*(.013109f-6.2013E-4f*Y)));
      du1=1.e-3f*(.86319+Y*(-2.0f*.11622f+Y*(3.0f*.013109-4.f*Y*6.2013e-4f)));
    }

    if(Tg > 1.8E4){
      u2=26.44f+Tg*(-2.22E-3f+Tg*6.188E-8f);
      du2=(-2.22e-3f+2.0f*Tg*6.188e-8f);
    }else{
      u2=5.470f+4.E-5f*Tg;
      du2=4.e-5f;
    }
    
    if(Tg > 3.33E4){
      u3=19.14f+Tg*(-1.408E-3f+Tg*2.617E-8f);
      du3=(-1.408e-3f+2.0f*Tg*2.617e-8f);
    }else{
      T const tmp = .911f+1.1E-5f*Tg;
      if(tmp > 1.0f){
	u3 = tmp;
	du3= 1.1e-5f;
      }else
	u3 = 1.0f;
    }

  }
  
  // ****************************************************************************** //
  // ---- P --- //
  
  template<typename T> inline void partition_14(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    
    T const Y = 1.e-3f * Tg;
    
    if(Tg > 6.E3f){
      u1=1.56f+5.2E-4f*Tg;
      du1=5.2e-4f;
    }else{
      u1=4.2251f+Y*(-.22476f+Y*(.057306f-Y*1.0381E-3f));
      du1=1.e-3f*(-.22476f+Y*(2.0f*.057306-3.f*Y*1.0381e-3f));
    }
    
    if(Tg > 7250.f){
      u2=4.62f+5.38E-4f*Tg;
      du2=5.38e-4f;
    }else{
      u2=4.4151f+Y*(2.2494f+Y*(-.55371f+Y*(.071913f-Y*3.5156E-3f)));
      du2=1.e-3f*(2.2494f+Y*(-2.f*.55371f+Y*(3.f*.071913-4.f*Y*3.5156e-3f)));
    }
    
    u3=5.595f+3.4E-5f*Tg;
    du3=3.4e-5f;
  }
  
  // ****************************************************************************** //
  // ---- S --- //
  
  template<typename T> inline void partition_15(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {

    if(Tg > 1.16e4f){
      u1=38.76f+Tg*(-4.906E-3f+Tg*2.125E-7f);
      du1=(-4.906e-3f+2.0f*Tg*2.125e-7f);
    }else{
      u1=7.5f+2.15E-4f*Tg;
      du1=2.1e-4f;
    }
    
    if(Tg > 1.05e4f){
      u2=6.406f+Tg*(-1.68E-4f+Tg*1.323E-8f);
      du2=(-1.68e-4f+Tg*2.0f*1.323e-8f);
    }else{
      u2=2.845f+2.43E-4f*Tg;
      du2=2.43e-4f;
    }
    
    u3=7.38f+1.88E-4f*Tg;
    du3=1.88e-4f;
  }
  
  // ****************************************************************************** //
  // ---- Cl --- //
  
  template<typename T> inline void partition_16(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    if(Tg > 1.84e4f){
      u1=-81.6f+4.8E-3f*Tg;
      du1=4.8e-3f;
    }else{
      u1=5.2f+6.E-5f*Tg;
      du1=6.e-5f;
    }

    u2=7.0f+2.43E-4f*Tg;
    du2=2.43e-4f;
    
    u3=2.2f+2.62E-4f*Tg;
    du3=2.62e-4f;
  }
  
  // ****************************************************************************** //
  // ---- Ar --- //
  
  template<typename T> inline void partition_17(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    u1=1.0f;
    u2=5.20f+3.8E-5f*Tg;
    du2=3.8e-5f;
    u3=7.474f+1.554E-4f*Tg;
    du3=1.554e-4f;
  }
  // ****************************************************************************** //
  // ---- K --- //
  
  template<typename T> inline void partition_18(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    if(Tg > 5800.0f){
      u1=-9.93f+2.124E-3f*Tg;
      du1=2.124e-3;
    }else{
      T const Y = 1.e-3f*Tg;
      u1=1.9909f+Y*(.023169f-Y*(.017432f-Y*4.0938E-3f));
      du1=1.e-3f*(.023169f-Y*(2.0f*.017432f-Y*3*4.0938e-3f));
    }
    u2=1.000f;
    u3=5.304f+1.93E-5f*Tg;
    du3=1.93e-5f;
  }
  
  // ****************************************************************************** //
  // ---- Ca --- //
  
  template<typename T> inline void partition_19(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const X = log(5040./Tg);
    T const dX = -1.0f/Tg;
    
    u1=1.0f+exp(-1.731273f-X*(5.004556f+X*(1.645456f+X*(1.326861f+.508553f*X))));
    du1=-dX*(5.004556f+X*(2.0f*1.645456f+X*(3.0f*1.326861f+4.0f*.508553f*X)))*(u1-1.0f);
    u2=2.0f+exp(-1.582112f-X*(3.996089f+X*(1.890737f+.539672f*X)));
    du2=-dX*(3.996089f+X*(2.0f*1.890737f+X*3.0f*.539672f))*(u2-2.0f);
    u3=static_cast<T>(1);
  }
    
  // ****************************************************************************** //
  // ---- Sc --- //
  
  template<typename T> inline void partition_20(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const X = log(5040./Tg);
    T const dX = -1.0f/Tg;
    
    u1=4.0f+exp(2.071563f+X*(-1.2392f+X*(1.173504f+.517796f*X)));
    du1=dX*(-1.2392f+X*(2.0f*1.173504f+3.0f*X*.517796f))*(u1-4.0f);
    u2=3.0f+exp(2.988362f+X*(-.596238f+.054658f*X));
    du2=dX*(-.596238+2*X*.054658)*(u2-3.0f);
    u3=10.f;
  }
  
  // ****************************************************************************** //
  // ---- Ti --- //
  
  template<typename T> inline void partition_21(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const X = log(5040./Tg);
    T const dX = -1.0f / Tg;
    
    if(Tg < 5.5E3f){
      u1=16.37f+Tg*(-2.838E-4f+Tg*5.819E-7f);
      du1=(-2.838e-4f+2*Tg*5.819e-7f);
    }else{
      u1=5.0f+exp(3.200453f+X*(-1.227798f+X*(.799613f+.278963f*X)));
      du1=(dX*(-1.227798f+X*(2*.799613f+3*X*.278963f)))*(u1-5.f);
    }

    u2=4.f+exp(3.94529f+X*(-.551431f+.115693f*X));
    du2=(dX*(-.551431f+2.f*X*.115693f))*(u2-4.0f);

    u3=16.4f+8.5E-4f*Tg;
    du3=8.5e-4f;
  }

    
  // ****************************************************************************** //
  // ---- V --- //
  
  template<typename T> inline void partition_22(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const X = log(5040./Tg);
    T const dX = -1.0f / Tg;
    
    u1=4.f+exp(3.769611f+X*(-.906352f+X*(.724694f+.1622f*X)));
    du1=(dX*(-.906352f+X*(2*.724694f+3*X*.1622f))) * (u1-4.0f);

    u2=1.f+exp(3.755917f+X*(-.757371f+.21043f*X));
    du2=(dX*(-.757371f+X*2*.21043f))*(u2-1.);

    if(Tg < 2.25E3f){
      u3=2.4E-3f*Tg;
      du3=2.4e-3f;
    }else{
      u3=-18.f+1.03E-2f*Tg;
      du3=1.03e-2f;
    }
  }
  
  // ****************************************************************************** //
  // ---- Cr --- //
  
  template<typename T> inline void partition_23(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const X = log(5040./Tg);
    T const dX = -1.0f / Tg;
    
    u1=7.0f+exp(1.225042f+X*(-2.923459f+X*(.154709f+.09527f*X)));
    du1=dX*(-2.923459f+X*(2*.154709f+X*3*.09527f))*(u1-7.f);
    
    u2=6.0f+exp(.128752f-X*(4.143973f+X*(1.096548f+.230073f*X)));
    du2=-dX*(4.143973f+X*(2*1.096548f+3.*X*.230073f)) * (u2-6.0f);

    u3=10.4f+2.1E-3f*Tg;
    du3=2.1e-3f;
  }

  // ****************************************************************************** //
  // ---- Mn --- //
  
  template<typename T> inline void partition_24(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const X = log(5040./Tg);
    T const dX = -1.0f / Tg;
    
    u1=6.0f+exp(-.86963f-X*(5.531252f+X*(2.13632f+X*(1.061055f+.265557f*X))));
    du1=-dX*(5.531252f+X*(2*2.13632f+X*(3*1.061055f+4*.265557f*X)))*(u1-6.0f);

    u2=7.0f+exp(-.282961f-X*(3.77279f+X*(.814675f+.159822f*X)));
    du2=-dX*(3.77279f+X*(2*.814675f+3*X*.159822f))*(u2-7.0f);
    
    u3=10.0f;
  }
      
  // ****************************************************************************** //
  // ---- Fe --- //
  
  template<typename T> inline void partition_25(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const X = log(5040.0f/Tg);
    T const dXdT = -1.0f / Tg;

    if(Tg < 4000.f){
      u1  = 15.85f+Tg*(1.306E-3f+Tg*2.04E-7f);
      du1 = 1.306E-3f + 2.0f * 2.04E-7f * Tg;
    }else if(Tg > 9000.f){
      u1  = 39.149f+Tg*(-9.5922E-3f+Tg*1.2477E-6f);
      du1 = -9.5922E-3f + 2.0f*Tg*1.2477E-6f;
    }else{
      u1  = 9.0f+exp(2.930047f+X*(-.979745f+X*(.76027f+.118218f*X)));
      du1 = dXdT *(-.979745f+X*(2*.76027f+3*X*.118218f)) *(u1-9.0f);
    }

    if(Tg > 18000.f){
      u2=68.356f+Tg*(-6.1104E-3f+Tg*5.1567E-7f);
      du2=(-6.1104E-3f+2.0f*Tg*5.1567E-7f);
    }else{
      T eps = exp(3.501597f+X*(-.612094f+.280982f*X));
      u2  = 10.f + eps;
      du2 = dXdT*eps*(-.612094f+2.0f*0.280982f*X);
    }
    
    u3=17.336f+Tg*(5.5048E-4f+Tg*5.7514E-8f);
    du3=(5.5048E-4f+2.0f*Tg*5.7514E-8f);
  }
  
  // ****************************************************************************** //
  // ---- Co --- //
  
  template<typename T> inline void partition_26(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    u1=8.65f+4.9E-3f*Tg;
    du1=4.9e-3f;
    
    u2=11.2f+3.58E-3f*Tg;
    du2=3.58e-3f;
    
    u3=15.0f+1.42E-3f*Tg;
    du3=1.42e-3f;
  }

  // ****************************************************************************** //
  // ---- Ni --- //
  
  template<typename T> inline void partition_27(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const X = log(5040./Tg);
    T const dX = -1.0f / Tg;
    
    u1=9.0f+exp(3.084552f+X*(-.401323f+X*(.077498f-.278468f*X)));
    du1=dX*(-.401323f+X*(2*.077498f-3.f*X*.278468f)) * (u1-9.0f);

    u2=6.0f+exp(1.593047f-X*(1.528966f+.115654f*X));
    du2=-dX*(1.528966f+2*X*.115654f)*(u2-6.0f);

    u3=13.3f+6.9E-4f*Tg;
    du3=6.9e-4f;
  }

  
  // ****************************************************************************** //
  // ---- Cu --- //
  
  template<typename T> inline void partition_28(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    if(Tg > 6250.f){
      u1=-.3f+4.58E-4f*Tg;
      du1=4.58e-4f;
    }else{
      u1=std::max<T>(2.0f,1.50f+1.51E-4f*Tg);
      if(u1 > 2.0f) du1=1.51e-4f;
    }
    
    u2=std::max<T>(1.0f,.22f+1.49E-4f*Tg);
    if(u2 > 1.) du2 = 1.49e-4f;
    
    u3=8.025f+9.4E-5f*Tg;
    du3=9.4e-5f;
    
  }
  
  // ****************************************************************************** //
  // ---- Zn --- //
  
  template<typename T> inline void partition_29(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    u1=std::max<T>(1.0f,.632f+5.11E-5f*Tg);
    if(u1 > 1.0f) du1=5.11e-5f;
    u2=2.00f;
    u3=1.00f;
  }

  // ****************************************************************************** //
  // ---- Ga --- //
  
  template<typename T> inline void partition_30(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const Y = 1.e-3*Tg;
    
    u1=1.7931f+Y*(1.9338f+Y*(-.4643f+Y*(.054876f-Y*2.5054E-3f)));
    du1=1.e-3f*(1.9338f+Y*(-2*.4643f+Y*(3*.054876f-Y*4*2.5054e-3f)));
    if(Tg > 6.E3f){
      u1=4.18f+2.03E-4f*Tg;
      du1=2.03e-4f;
    }
    u2=1.0f;
    u3=2.0f;
  }

  // ****************************************************************************** //
  // ---- Ge --- //
  
  template<typename T> inline void partition_31(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    u1=6.12f+4.08E-4f*Tg;
    du1=4.08e-4f;
    u2=3.445f+1.78E-4f*Tg;
    du2=1.78e-4f;
    u3=1.1f;// ! APPROXIMATELY
  }

  // ****************************************************************************** //
  // ---- As --- //
  
  template<typename T> inline void partition_32(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    
    u1=2.65f+3.65E-4f*Tg;
    du1=3.65e-4f;

    if(Tg > 1.2E4f){
      u2=8.0f;
    }else{
      T const Y = 1.e-3f * Tg;
      u2=-.25384f+Y*(2.284f+Y*(-.33383f+Y*(.030408f-Y*1.1609E-3f)));
      du2=1.e-3f*(2.284f+Y*(-2*.33383f+Y*(3*.030408f-4*Y*1.1609e-3f)));
    }
    u3=8.0f;
  }

  // ****************************************************************************** //
  // ---- Se --- //
  
  template<typename T> inline void partition_33(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    T const Y = 1.e-3f * Tg;

    u1=6.34f+1.71E-4f*Tg;
    du1=1.71e-4f;
    u2=4.1786f+Y*(-.15392f+3.2053E-2f*Y);
    du2=1.e-3f*(-.15392f+3.2053e-2f*Y*2);
    u3=8.0f;
  }
  
  // ****************************************************************************** //
  // ---- Br --- //
  
  template<typename T> inline void partition_34(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    u1=4.12f+1.12E-4f*Tg;
    du1=1.12e-4f;
    u2=5.22f+3.08E-4f*Tg;
    du2=3.08e-4f;
    u3=2.3f+2.86E-4f*Tg;
    du3=2.86e-4f;
  }
  
  // ****************************************************************************** //
  // ---- Kr --- //
  
  template<typename T> inline void partition_35(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    u1=1.00f;
    u2=4.11f+7.4E-5f*Tg;
    du2=7.4e-5f;
    u3=5.35f+2.23E-4f*Tg;
    du3=2.23e-4f;
  }
    
  // ****************************************************************************** //
  // ---- Rb --- //
  
  template<typename T> inline void partition_36(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {

    if(Tg > 6250.f){
      u1=-14.9f+2.79E-3f*Tg;
      du1=2.79e-3f;
    }else{
      u1=std::max<T>(2.0f,1.38f+1.94E-4f*Tg);
      if(u1 > 2.f) du1=1.94e-4f;
    }
    
    u2=1.000f;
    
    u3=4.207f+4.85E-5f*Tg;
    du3=4.85e-5f;
  }
  
  // ****************************************************************************** //
  // ---- Sr --- //
  
  template<typename T> inline void partition_37(T const& Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  {
    if(Tg > 6500.f){
      u1=-6.12f+1.224E-3f*Tg;
      du1=1.224e-3f;
    }else{
      T const Y = 1.e-3f * Tg;
      u1=.87127f+Y*(.20148f+Y*(-.10746f+Y*(.021424f-Y*1.0231E-3f)));
      du1=1.e-3f*(.20148f+Y*(-.10746f*2+Y*(.021424f*3-Y*1.0231e-3f*4)));
    }
    u2=std::max<T>(2,.84f+2.6e-4f*Tg);
    if(u2 > 2.f)du2=2.6e-4f;
    u3=1.0f;
  }

  // ****************************************************************************** //

  template<typename T>
  static constexpr const std::array<pf_funct<T>,38> partitions =
    {&partition_0<T>,  &partition_1<T>,  &partition_2<T>,  &partition_3<T>,  &partition_4<T>,
     &partition_5<T>,  &partition_6<T>,  &partition_7<T>,  &partition_8<T>,  &partition_9<T>,
     &partition_10<T>, &partition_11<T>, &partition_12<T>, &partition_13<T>, &partition_14<T>,
     &partition_15<T>, &partition_16<T>, &partition_17<T>, &partition_18<T>, &partition_19<T>,
     &partition_20<T>, &partition_21<T>, &partition_22<T>, &partition_23<T>, &partition_24<T>,
     &partition_25<T>, &partition_26<T>, &partition_27<T>, &partition_28<T>, &partition_29<T>,
     &partition_30<T>, &partition_31<T>, &partition_32<T>, &partition_33<T>, &partition_34<T>,
     &partition_35<T>, &partition_36<T>, &partition_37<T>};
  
  // ****************************************************************************** //

  template<typename T>
  void molecb(T const X, T &y0, T &y1, T &dy0, T &dy1)
  {
    T const dx=(-X*X)/5040.f;
    y0 = -11.206998f+X*(2.7942767f+X*(7.9196803E-2f-X*2.4790744E-2f)); // H2+
    y1 = -12.533505f+X*(4.9251644f+X*(-5.6191273E-2f+X*3.2687661E-3f)); // H2

    dy0 = dx*(2.7942767f+X*(2*7.9196803e-2f-X*3*2.4790744e-2f));
    dy1 = dx*(4.9251644f+X*(-2*5.6191273e-2f-X*3*3.2687661e-3f));
  }
  
  // ****************************************************************************** //

  template<typename T>
  inline void get_partition(int nEl, T const Tg, T &u1, T &u2, T &u3, T &du1, T &du2, T &du3)
  { 
    du1 = 0; du2 = 0; du3 = 0; 
    partitions<T>[nEl](Tg, u1, u2, u3, du1, du2, du3);
  }
  // ****************************************************************************** //

  template<typename T>
  inline T get_dEion(int const &istage, T const& sqrtPeTg)
  {
    /*
      According to Hubeny & Mihalas (2014), page 244:

      Delta chi = - z * (e^2/(4*pi*eps_0)) / D  [J]
      D = sqrt(eps_0 / (2*e^2)) * sqrt(BK*T/Ne) [m]
      
      Delta chi(T,NE) = -z * e^3 * sqrt(Ne / (8pi^2*eps_0^3*T*BK))  [J]

      However: 
      1) we want to give Delta chi in eV -> We divide by eV.
      2) we want to work with Pe = Ne * BK * T

      Delta chi(T,Pe) = -z * (e^3 / BKT) * sqrt(Pe / (8*pi^2*eps_0^3)) / eV [eV]
      
      If we set eps0 = 1/(4pi), we recover the CGS case if all other constants
      are in CGS system.

    */
    
    using namespace phyc;
    using dp = double;
    
    constexpr static const double C0 = QE<dp>*QE<dp>*(QE<dp> / EV<dp>)/ BK<dp>;
    constexpr static const double C1 = 8*PI<dp>*PI<dp>*EPS0<dp>*EPS0<dp>*EPS0<dp>;
    static const T C = C0 / sqrt(C1);
    
    
    return (C * sqrtPeTg * static_cast<T>(istage));
  }
  
  // ****************************************************************************** //

  template<typename T>
  inline T getEion_eV(int const &aNum, int const &istage, T const &sqrtPeTg)
  {
    // --- Returns the ionization potential corrected in eV --- //
    // --- Source: Mihalas 78, same formalism as Rh and Multi --- //
    // --- Eion = Eion(0) - dEion(T,Ne) --- //
        
    return EION<T>[istage][aNum] - get_dEion<T>(istage, sqrtPeTg);
  }

  // ****************************************************************************** //

  template<typename T> inline 
  T Planck(T const& Tg, T const nu)
  {
    using namespace phyc;
    static constexpr const T C1 = HH<T> / BK<T>;

    T const a = HH<T> * nu;
    T const b = nu / CC<T>;
        
    return 2*(a*b)*b / (exp(C1*nu/Tg) - 1);
  }
  
  // ****************************************************************************** //

}

#endif
