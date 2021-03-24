#ifndef CSIREOS
#define CSIREOS

#include <vector>
#include <algorithm>
#include <cstdio>

#include "saha.hpp"
#include "cSIR_atomic.hpp"


namespace sr{

  constexpr static const int nHv = 7;
  
  // ****************************************************************************** //

  template<typename T >
  constexpr static const T LOG10 = 2.3025850929940459;
  
  template<typename T>
  constexpr inline T pow10_(T const &x){return exp(sr::LOG10<T>*x);}
  
  // ****************************************************************************** //
  template<typename T>
  inline T clip(T const x, T const mi, T const ma){return std::max<T>(std::min<T>(x,ma), mi);}
 
  // ****************************************************************************** //

  template <class T>
  inline T sign(T const a, T const b){
    return std::abs(a) * b / std::abs(b);
  }

  // ****************************************************************************** //
  
  template<typename T>
  inline T init_Pe_from_Pg(T const &Tg, T const &Pg, T const aH)
  {

    // --- if we assume that all electrons come from H --- //
    T const saha = sr::Saha<T>(Tg, Pg, static_cast<T>(13.6), static_cast<T>(2), static_cast<T>(1));

    // --- Solve quadratic equation --- //
    T const aaa = 1.0f + saha;        
    T const bbb = -(aH-1.0f) * saha;  
    T const ccc = -saha*aH;           
    T const ybh=(-bbb+sqrt(bbb*bbb-4*aaa*ccc))/(2*aaa);
    
    return Pg*ybh/(1.f+ybh);
  }
  

  // ****************************************************************************** //
  // ----                      Eos, no need to have a class                     --- //
  // ****************************************************************************** //
  
 template<typename T>
  inline T compute_Pg(T const Tg, T Pe,  std::vector<T> const& abund, T* __restrict__ H)
  {
    // --- Taken from SIR/Wittmann --- //
    
    T const theta = static_cast<T>(5040)/Tg;

    
    // --- Disoc. Constants --- //
    
    T y0 = 0, y1=0, dy0=0, dy1=0; 
    sr::molecb<T>(theta, y0, y1, dy0, dy1);

    y0 = clip<T>(y0, -30,30);
    y1 = clip<T>(y1, -30,30);

    T g4 = Pe * pow10_(y0);// * phyc::BA_TO_PA<T>; is this needed?
    T g5 = Pe*pow10_(y1);//* phyc::BA_TO_PA<T>; is this needed?

    
    // --- Hydrogen first --- //

    T u1=0,u2=0,u3=0, du1=0, du2=0, du3=0;
    T g1 = 0, g2 = 0, g3 = 0,
      a=0, b=0, c=0, d=0, e=0, f1=0, f2=0,
      f3=0, f4=0, f5=0,
      c3 = 0, c2=0, c1=0, const6, const7;
    double alfai=0, ab_others=0;
    T fe = 0;
    
    sr::get_partition<T>(0, Tg, u1, u2, u3, du1, du2, du3);
    
    T const sqrtPeTg = sqrt(Pe)/Tg;
    
    g2 = sr::Saha<T>(Tg, Pe, sr::EION<T>[0][0], u1, u2);   // H+/H
    g3 = 1 / std::max<T>(std::min<T>(sr::Saha<T>(Tg, Pe, 0.754, static_cast<T>(1), u1), 1.e30f),1.e-30f); // H-/H
    
    
    // --- Count electrons provided by each atom --- //

    static constexpr const int loop_lim = 28;
    for(int ii = 1; ii<loop_lim; ++ii){
      
      T const Eion1 = sr::EION<T>[0][ii];//sr::getEion_eV(ii, 0, sqrtNeTg);
      T const Eion2 = sr::getEion_eV(ii, 1, sqrtPeTg); // corrected for lowering of Eion (does not affect neutral atoms)
      sr::get_partition<T>(ii, Tg, u1, u2, u3, du1, du2, du3);

      a = sr::Saha<T>(Tg, Pe, Eion1, u1, u2);
      b = sr::Saha<T>(Tg, Pe, Eion2, u2, u3);

      c = 1.0f + a * (1.0f + b);
      alfai=abund[ii]/abund[0];
      ab_others += alfai;
      g1 += (alfai / c)*a*(1.0f+2*b);
    }

    // --- Now compute terms, it gets messy --- //

    a = 1.0f + g2 + g3;
    e = (g2/g5)*g4;
    b = 2*(1+e);
    c = g5;
    d = g2-g3;

    a = std::min<T>(std::max<T>(a, 1.e-15f), 1.e+15f);
    d = std::min<T>(std::max<T>(d, 1.e-15f), 1.e+15f);

    c1=c*b*b+a*d*b-e*a*a;
    c2=2.*a*e-d*b+a*b*g1;                                                         
    c3=-(e+b*g1) ;                                                        
    f1=0.5f*c2/c1 ;            
    f1=-f1+sign<T>(1.0f,c1)*sqrt(f1*f1-c3/c1) ;
    f5=(1.f-a*f1)/b ;
    f4=e*f5 ;
    f3=g3*f1 ;
    f2=g2*f1 ;
    fe=f2-f3+f4+g1 ;
    fe = std::min<T>(std::max<T>(fe, 1.e-30f), 1.e+30f);
    T phtot=Pe/fe;

    // --- Supposedly if f5 is small it can lead to unstable results --- //
    
    //if(f5 < 1.e-4f){
      const6 = g5/Pe*f1*f1;
      const7=f2-f3+g1; 
      for(int ii =0; ii<5; ++ii){
	f5=phtot*const6;
	f4=e*f5;
	fe=const7+f4;
	phtot=Pe/fe;
      }
      //}

    // --- Evaluate Pg --- //
    
    T const Pg = std::max<T>(Pe * (1.0f+(f1+f2+f3+f4+f5+ab_others)/fe), 1.e-15f);

    
    // --- copy H populations --- //

    T const nHtot = (Pe/fe) / (phyc::BK<T> * Tg);
    H[0] = nHtot; // pHtot
    H[1] = f1 * nHtot; // nHI 
    H[2] = f2 * nHtot; // nHII
    H[3] = f4 * nHtot; // nH- 
    H[4] = f3 * nHtot; // nH2+
    H[5] = f4 * nHtot; // nH2 
    
    
    return Pg;
  }

  // ****************************************************************************** //

  template<typename T>
  inline T compute_Ntot(T const Tg, T const Ne,  std::vector<T> const& abund, T* __restrict__ H)
  {
    // --- Taken from SIR/Wittmann --- //

    T Pe = Ne * phyc::BK<T> * Tg;
    T const theta = static_cast<T>(5040)/Tg;

    
    // --- Disoc. Constants --- //
    
    T y0 = 0, y1=0, dy0=0, dy1=0; 
    sr::molecb<T>(theta, y0, y1, dy0, dy1);

    y0 = clip<T>(y0, -30,30);
    y1 = clip<T>(y1, -30,30);

    T g4 = Pe * pow10_(y0);// * phyc::BA_TO_PA<T>; is this needed?
    T g5 = Pe*pow10_(y1);//* phyc::BA_TO_PA<T>; is this needed?

    
    // --- Hydrogen first --- //

    T u1=0,u2=0,u3=0, du1=0, du2=0, du3=0;
    T g1 = 0, g2 = 0, g3 = 0,
      a=0, b=0, c=0, d=0, e=0, f1=0, f2=0,
      f3=0, f4=0, f5=0,
      c3 = 0, c2=0, c1=0, const6, const7;
    double alfai=0, ab_others=0;
    T fe = 0;
    
    sr::get_partition<T>(0, Tg, u1, u2, u3, du1, du2, du3);
    
    T const sqrtPeTg = sqrt(Pe)/Tg;
    
    g2 = sr::Saha<T>(Tg, Pe, sr::EION<T>[0][0], u1, u2);   // H+/H
    g3 = 1 / std::max<T>(std::min<T>(sr::Saha<T>(Tg, Pe, 0.754, static_cast<T>(1), u1), 1.e30f),1.e-30f); // H-/H
    
    
    // --- Count electrons provided by each atom --- //

    static constexpr const int loop_lim = 28;
    for(int ii = 1; ii<loop_lim; ++ii){
      
      T const Eion1 = sr::EION<T>[0][ii];//sr::getEion_eV(ii, 0, sqrtNeTg);
      T const Eion2 = sr::getEion_eV(ii, 1, sqrtPeTg); // corrected for lowering of Eion (does not affect neutral atoms)
      sr::get_partition<T>(ii, Tg, u1, u2, u3, du1, du2, du3);

      a = sr::Saha<T>(Tg, Pe, Eion1, u1, u2);
      b = sr::Saha<T>(Tg, Pe, Eion2, u2, u3);

      c = 1.0f + a * (1.0f + b);
      alfai=abund[ii]/abund[0];
      ab_others += alfai;
      g1 += (alfai / c)*a*(1.0f+2*b);
    }

    // --- Now compute terms, it gets messy --- //

    a = 1.0f + g2 + g3;
    e = (g2/g5)*g4;
    b = 2*(1+e);
    c = g5;
    d = g2-g3;

    a = std::min<T>(std::max<T>(a, 1.e-15f), 1.e+15f);
    d = std::min<T>(std::max<T>(d, 1.e-15f), 1.e+15f);

    c1=c*b*b+a*d*b-e*a*a;
    c2=2.*a*e-d*b+a*b*g1;                                                         
    c3=-(e+b*g1) ;                                                        
    f1=0.5f*c2/c1 ;            
    f1=-f1+sign<T>(1.0f,c1)*sqrt(f1*f1-c3/c1) ;
    f5=(1.f-a*f1)/b ;
    f4=e*f5 ;
    f3=g3*f1 ;
    f2=g2*f1 ;
    fe=f2-f3+f4+g1 ;
    fe = std::min<T>(std::max<T>(fe, 1.e-30f), 1.e+30f);
    T phtot=Pe/fe;

    // --- Supposedly if f5 is small it can lead to unstable results --- //
    
    //if(f5 < 1.e-4f){
      const6 = g5/Pe*f1*f1;
      const7=f2-f3+g1; 
      for(int ii =0; ii<5; ++ii){
	f5=phtot*const6;
	f4=e*f5;
	fe=const7+f4;
	phtot=Pe/fe;
      }
      //}

    // --- Evaluate Pg --- //
    
    T const Pg = std::max<T>(Pe * (1.0f+(f1+f2+f3+f4+f5+ab_others)/fe), 1.e-15f);
    
    
    // --- copy H populations --- //

    T const nHtot = (Pe/fe) / (phyc::BK<T> * Tg);
    H[0] = nHtot; // pHtot
    H[1] = f1 * nHtot; // nHI 
    H[2] = f2 * nHtot; // nHII
    H[3] = f4 * nHtot; // nH- 
    H[4] = f3 * nHtot; // nH2+
    H[5] = f4 * nHtot; // nH2 
    
    
    return Pg/(phyc::BK<T> * Tg);
  }

  // ****************************************************************************** //

  template<typename T>
  inline T compute_Pe(T const Tg, T const Pg, T Pe,  std::vector<T> const& abund, T* __restrict__ H)
  {
    // --- Taken from SIR/Wittmann --- //
    T const theta = static_cast<T>(5040)/Tg;

    
    // --- Disoc. Constants --- //
    
    T y0 = 0, y1=0, dy0=0, dy1=0; 
    sr::molecb<T>(theta, y0, y1, dy0, dy1);

    y0 = clip<T>(y0,-30,30);
    y1 = clip<T>(y1,-30,30);

    
    // --- init some quantities --- //

    T g4 = Pe * pow10_(y0);// * phyc::BA_TO_PA<T>; is this needed?
    T g5 = Pe * pow10_(y1);//* phyc::BA_TO_PA<T>; is this needed?

    // --- Hydrogen first --- //

    T u1=0,u2=0,u3=0, du1=0, du2=0, du3=0;
    T g1 = 0, g2 = 0, g3 = 0,
      a=0, b=0, d=0, e=0, f1=0, f2=0,
      f3=0, f4=0, f5=0, fe = 0,
      c3 = 0, c2=0, c1=0, const6, const7;
    T alfai=0, ab_others=0, c=0;

    sr::get_partition<T>(0, Tg, u1, u2, u3, du1, du2, du3);
    
    T const sqrtPeTg = sqrt(Pe)/Tg;
    
    g2 = sr::Saha<T>(Tg, Pe, sr::EION<T>[0][0], u1, u2);   // H+/H
    g3 = 1.0 / clip<T>(sr::Saha<T>(Tg, Pe, 0.754, static_cast<T>(1), u1), 1.e-30f,1.e30f); // H-/H
    
    // --- Count electrons provided by each atom --- //

    static constexpr const int loop_lim = 28;
    for(int ii = 1; ii<loop_lim; ++ii){
      
      T const Eion1 = sr::EION<T>[0][ii];//sr::getEion_eV(ii, 0, sqrtNeTg);
      T const Eion2 = sr::getEion_eV(ii, 1, sqrtPeTg); // corrected for lowering of Eion (does not affect neutral atoms)
      sr::get_partition<T>(ii, Tg, u1, u2, u3, du1, du2, du3);

      a = sr::Saha<T>(Tg, Pe, Eion1, u1, u2);
      b = sr::Saha<T>(Tg, Pe, Eion2, u2, u3);

      c = 1.0f + a * (1.0f + b);
      alfai=abund[ii]/abund[0];
      ab_others += alfai;
      g1 += (alfai / c)*a*(1.0f+2*b);
    }

    // --- Now compute terms, it gets messy --- //

    a = 1.0f + g2 + g3;
    e = (g2/g5)*g4;
    b = 2*(1+e);
    c = g5;
    d = g2-g3;

    a = std::min<T>(std::max<T>(a, 1.e-15f), 1.e+15f);
    d = std::min<T>(std::max<T>(d, 1.e-15f), 1.e+15f);

    c1=c*b*b+a*d*b-e*a*a;
    c2=2.*a*e-d*b+a*b*g1;                                                         
    c3=-(e+b*g1) ;                                                        
    f1=0.5f*c2/c1 ;            
    f1=-f1+sign<T>(1.0f,c1)*sqrt(f1*f1-c3/c1) ;
    f5=(1.f-a*f1)/b ;
    f4=e*f5 ;
    f3=g3*f1 ;
    f2=g2*f1 ;
    fe=f2-f3+f4+g1 ;
    fe = std::min<T>(std::max<T>(fe, 1.e-30f), 1.e+30f);
    T phtot=Pe/fe;

    // --- Supposedly if f5 is small it can lead to unstable results --- //
    
    // if(f5 < 1.e-4f){
      const6 = g5/Pe*f1*f1;
      const7=f2-f3+g1; 
      for(int ii =0; ii<5; ++ii){
	f5=phtot*const6;
	f4=e*f5;
	fe=const7+f4;
	phtot=Pe/fe;
      }
      // }

    // --- Re-evaluate Pe --- //
    
    Pe = std::max<T>(Pg /(1.0f+(f1+f2+f3+f4+f5+ab_others)/fe), 1.e-15f);

    
    // --- copy H populations --- //
    
    T const nHtot = (Pe/fe) / (phyc::BK<T> * Tg);
    H[0] = nHtot; // pHtot
    H[1] = f1 * nHtot; // nHI 
    H[2] = f2 * nHtot; // nHII
    H[3] = f4 * nHtot; // nH- 
    H[4] = f3 * nHtot; // nH2+
    H[5] = f4 * nHtot; // nH2 
    
    return Pe;
  }

  // ****************************************************************************** //


  template<typename T>
  inline T Pe_from_Pg(T const& Tg, T const& Pg, std::vector<T> const &abund, T* __restrict__ H)
  {
    
    // --- Init Pe taking into account only the contribution from H --- //
    T Pe = init_Pe_from_Pg<T>(Tg, Pg, 0.91);
    
    // --- We could iterate with a Newton-Raphson, but it will probably be slower
    // --- than simply iterating like a lambda iteration
    T diff  = 1.0;
    int iter = 0;
    do{
      // --- Solve partial pressures and recount electrons --- //
      T iPe = compute_Pe<T>(Tg, Pg, Pe, abund, H);
      
      // --- Damp correction by taking the average of the new and the old --- //
      iPe = iPe*0.6f + Pe*0.4f;
      diff = std::abs((iPe-Pe)/Pe);

      Pe = iPe;
    }while(diff > 1.e-5 && (iter++ < 40));
    return Pe;
  }


  // ****************************************************************************** //


  template<typename T>
  inline T Ne_from_Nt(T const& Tg, T const& Nt, std::vector<T> const &abund, T* __restrict__ H)
  {
    T const BKT =  phyc::BK<T>*Tg;
    T iPg = Nt * BKT;
    
    // --- Init Pe taking into account only the contribution from H --- //
    T Ne = init_Pe_from_Pg<T>(Tg,  Nt * BKT, 0.91) / BKT;

    // --- We could iterate with a Newton-Raphson, but it will probably be slower
    // --- than simply iterating like a lambda iteration
    T diff  = 1.0;
    int iter = 0;
    do{
      
      // --- Solve partial pressures and recount electrons --- //
      T iNe = compute_Pe<T>(Tg, iPg, Ne*BKT, abund, H) / BKT;
;
      
      // --- Damp correction by taking the average of the new and the old --- //
      iNe = iNe*0.65f + Ne*0.35f;
      diff = std::abs((iNe-Ne)/Ne);

      Ne = iNe;
    }while(diff > 1.e-5 && (iter++<40));

    return Ne;
  }

  // ****************************************************************************** //

  template<typename T>
  inline T computeRho(T const& nHtot, T const& mu)
  {
    constexpr static const T C = phyc::AMU<T> * sr::AMASS<T>[0];
    return  nHtot * mu * C;
  }
  
  // ****************************************************************************** //  
  
  template<typename T>
  inline void ComputePartialDensities(T const& Tg, T const& Ne,  std::vector<T> const& abund, T const &nHtot, std::vector<int> const& aNums, T* __restrict__ output)
  {
    int const stride = aNums.size();
    T* __restrict__ n1 = output;
    T* __restrict__ n2 = &output[stride];
    T* __restrict__ n3 = &output[2*stride];
    T* __restrict__ u1 = &output[3*stride];
    T* __restrict__ u2 = &output[4*stride];
    T* __restrict__ u3 = &output[5*stride];

    T const BKT = phyc::BK<T>*Tg;
    T const Pe = Ne*BKT;
    T const sqrtPeTg = sqrt(Pe)/Tg;
    
    // --- Loop through all required species --- //

    int const nSpec = aNums.size();
    for(int jj=0; jj<nSpec; ++jj){

      int const ii = aNums[jj];
      T du1=0, du2=0, du3=0;
      
      
      T Eion2 =  sr::getEion_eV(ii, 1, sqrtPeTg);
      sr::get_partition<T>(ii, Tg, u1[jj], u2[jj], u3[jj], du1, du2, du3);

      // --- c = ntot / nI = (nI + nII + nIII) / nI = 1 + a + ab --- //
      
      T const a = sr::Saha<T>(Tg, Pe, sr::EION<T>[0][ii], u1[jj], u2[jj]);
      T const b = sr::Saha<T>(Tg, Pe,              Eion2, u2[jj], u3[jj]);

      // --- population of each level relative to total nHtot            --- //
      // --- c = ntot/nI -> alfa / c = nI/ntot * ntot/nHtot = nI / nHtot --- //
      // --- alfai includes the multiplication by the provided nHtot --- //
      
      T const c = clip<T>(1 + a*(1+b), 1.e-30, 1.e30);
      T const alfai = abund[ii] / abund[0] * nHtot / c;

      n1[jj] = alfai        / u1[jj];
      n2[jj] = alfai * a    / u2[jj];
      n3[jj] = alfai * (a*b)/ u3[jj];
    }
  }
  
}


#endif
