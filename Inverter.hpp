#ifndef _INVERTER_H
#define _INVERTER_H
#include <LocalField.hpp>
#include <Background.h>
#include <Geometry.hpp>
#include <algorithm>
#include <newQCDpt.h>
#include <map>
#include <iostream>
#include <fstream>
#include <Kernels.hpp>
#include <uparam.hpp>
#include <stdlib.h>

#include <newMyQCD.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#include <sstream>
#endif

namespace meth {
  
  enum bdy{bulk=0,lower=0,upper=0};
  
  template < template<class,class> class K, class T, class B>
  struct FermionInfo {
    typedef K<T,B> Type;
  };

  template<class Gauge_t,class Field_t,template<class,class> class Kernel_t>
  struct Dirac {

    typedef typename FermionInfo<Kernel_t,Gauge_t,kernels::detail::LowBdy>::Type Lower;
    typedef typename FermionInfo<Kernel_t,Gauge_t,kernels::detail::NoBdy>::Type Bulk;
    typedef typename FermionInfo<Kernel_t,Gauge_t,kernels::detail::HighBdy>::Type Upper;

    Dirac(const Field_t& source, const Gauge_t& U, const double& mass) : apply_blk(source, mass ),
									 apply_low(source, mass ),
									 apply_up (source, mass ) { };

    Field_t& operator()(Field_t& dest) {
      const int T = dest.extents()[0];
      dest.apply_on_timeslice(apply_low, 0);
      //for x_0 != 0 update all directions
      for (int x0 = 1; x0 < T; ++x0)
	for (typename Bulk::Direction mu(0); mu.is_good(); ++mu)
	  dest.apply_on_timeslice(apply_blk, x0);
      dest.apply_on_timeslice(apply_up, T);
    }

  private:
    Bulk apply_blk;
    Lower apply_low;
    Upper apply_up;
  };

}


namespace inverter {


  // cgs
  template<class GluonField,class FermionField,template<class,class> class Kernel_t,int count_max>
  int cgs(const GluonField& U,const FermionField& src,FermionField& x,
	  const double& m,const double& tol) {
  
    typedef typename meth::Dirac<GluonField,FermionField,Kernel_t> DiracOp;
  
    x = src;
    FermionField r(x), Ax(x), b(src);
    DiracOp apply_from_x(x,U,m);
    apply_from_x(Ax);
    r = b - Ax;
    FermionField rstar(x), p(r), u(r), Ap(x), q(r), uq(x), Auq(x);
  
    Cplx nr = r * rstar;
    Cplx nr0(nr);
    int count = 0;

    DiracOp apply_from_p(p,U,m);
    DiracOp apply_from_uq(uq,U,m);
  
    do {
    
      apply_from_p(Ap);
      Cplx al = nr/ ( Ap * rstar);
    
      q  = u - Ap*al;
      uq = u+q;
      x += uq*al;

      apply_from_uq(Auq);

      r -= Auq*al;
      nr0 = nr;
      nr = r*rstar;
      Cplx bt = nr/nr0;
      u = r + q*bt;
      p = u + (q + p*bt)*bt;

      std::cout << "count = " << count << std::endl;
      std::cout << "#\tResidual:" << sqrt((r^r).real()) << "\n#----------------\n";
      ++count;
    } while ( sqrt((r^r).real()) > tol && (count < count_max) );

  };




  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  ///
  ///  BiCG stab inverter - first attemp. Returns 0 if inversion
  ///  tolerance is obtained in less than count_max iterations,
  ///  otherwise stops the inversion and returns 1.
  ///  1) Is the definition of tolerance correct? 
  ///  2) Make an Inverter class and this a possible choice of
  ///  algorithm?
  ///
  ///  \author Michele Brambilla <mib.mic@gmail.com>
  ///  \date Thu Jan 17 11:19:12 2013
  template<class GluonField,class FermionField,template<class,class> class Kernel_t,int count_max>
  int BiCGstab(const GluonField& U,const FermionField& src,FermionField& x,
	       const double& m, const double& tol )
  {
  
    typedef typename meth::Dirac<GluonField,FermionField,Kernel_t> DiracOp;
  
    x = src;
    Cplx alpha(0.0,0.0), beta, omega(1.0,0.0);
    Cplx rho_1(0.0,0.0), rho_2(1.0,0.0);
    double norm_r, norm_r1;
    FermionField b(x), r0(x), Ax(x);

    DiracOp apply_from_x(x,U,m);
    apply_from_x(Ax);

    r0 = b - Ax;

    FermionField p(r0), r(r0), t(r0), s(x), v(x), rtilde(r0);
    int count = 0;

    while( count < count_max )
      {
	++count;
	rho_1 = rtilde*r;
	beta = (rho_1/rho_2) * (alpha/omega);
	p = r + (p - v * omega) * beta;

	DiracOp apply_from_p(p,U,m);
	apply_from_p(v);

	alpha = rho_1 / (rtilde*v);       
	s = r - v * alpha;

	DiracOp apply_from_s(s,U,m);
	apply_from_s(t);

	omega = (t*s) / (t*t);
	x += p * alpha + s * omega;
	r  = s - t * omega;
	if ( abs(r*r) < tol) {
	  x += p * alpha;
	  
	  apply_from_x(Ax);
	  
	  r0 = b - Ax;
	  std::cout << "#\tResidual:" << sqrt(abs(r*r)) << "\n#----------------\n";
	  return 0;
	}
	rho_2 = rho_1;
	std::cout << count << "\t|r| = " << abs(r*r) << "\n\n";
      }

    return 1;
  }



}







#endif //INVERTER_H
