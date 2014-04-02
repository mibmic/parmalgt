#ifndef _PLAQUETTE_H_
#define _PLAQUETTE_H_

#include <Kernels.hpp>
#include <Background.h>

#ifdef _OPENMP
#include <omp.h>
#else
namespace plaq {
  int omp_get_max_threads() { return 1; }
  int omp_get_thread_num() { return 0; }
}
#endif

namespace plaq {

  template <class Fld_t>
  struct Spatial {
    typedef typename kernels::std_types<Fld_t>::ptSU3_t ptSU3;
    typedef typename kernels::std_types<Fld_t>::point_t Point;
    typedef typename kernels::std_types<Fld_t>::direction_t Direction;
    typedef typename kernels::std_types<Fld_t>::bgf_t BGF;
    static const int ORD = kernels::std_types<Fld_t>::order;
    static const int n_cb = 0;

    std::vector<ptSU3> val;
    std::vector<Cplx> result;
    Spatial() : val(omp_get_max_threads(), ptSU3(bgf::zero<BGF>())), result(ORD + 1) { }

    void operator()(const Fld_t& U, const Point& n) {
      ptSU3 tmp;
      for (Direction k(1); k.is_good(); ++k)
	for (Direction l(k + 1); l.is_good(); ++l)
	  val[omp_get_thread_num()] +=  U[n][k] * U[n + k][l] 
            * dag( U[n][l] * U[n + l][k] );
    }
    void reduce() {
      for (int i = 1; i < omp_get_max_threads(); ++i) val[0] += val[i];
      result[0] = -val[0].bgf().Tr() * 2;
      for (int i = 1; i <= ORD; ++i) result[i] = -val[0][i-1].tr() * 2;
    }
  };

  template <class Fld_t>
  struct Temporal {
    typedef typename kernels::std_types<Fld_t>::ptSU3_t ptSU3;
    typedef typename kernels::std_types<Fld_t>::point_t Point;
    typedef typename kernels::std_types<Fld_t>::direction_t Direction;
    typedef typename kernels::std_types<Fld_t>::bgf_t BGF;
    static const int ORD = kernels::std_types<Fld_t>::order;
    static const int n_cb = 0;

    std::vector<ptSU3> val;
    std::vector<Cplx> result;
    Temporal() : val(omp_get_max_threads(), ptSU3(bgf::zero<BGF>())), result(ORD + 1) { }

    void operator()(const Fld_t& U, const Point& n) {
      ptSU3 tmp;
      Direction t(0);
      for (Direction k(1); k.is_good(); ++k)
        val[omp_get_thread_num()] +=  U[n][t] * U[n + t][k] 
          * dag( U[n][k] * U[n + k][t] );
    }
    void reduce() {
      for (int i = 1; i < omp_get_max_threads(); ++i) val[0] += val[i];
      result[0] = -val[0].bgf().Tr() * 2;
      for (int i = 1; i <= ORD; ++i) result[i] = -val[0][i-1].tr() * 2;
    }
  };
}
#endif /* _PLAQUETTE_H_ */
