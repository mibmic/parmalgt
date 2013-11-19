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

  namespace detail {
    template <class Field_t, class Point, class Direction>
    typename kernels::std_types<Field_t>::ptSU3_t
    two_by_one(Field_t& U, const Point& n,
               const Direction& mu, const Direction& nu){
      return U[n + mu][mu] * U[n + mu + mu][nu] * 
             dag(U[n][nu] * U[n+nu][mu] * U[n+nu+mu][mu])
           + U[n+mu][mu] * dag( U[n-nu][mu] * 
             U[n+mu-nu][mu] * U[n+mu+mu-nu][nu] ) * U[n-nu][nu]
           + U[n+mu][nu] * dag( U[n-mu][nu] * 
             U[n-mu+nu][mu] * U[n+nu][mu] ) * U[n-mu][mu]
           + dag( U[n-nu-mu][mu] * U[n-nu][mu] * U[n-nu+mu][nu] ) * 
             U[n-nu-mu][nu] * U[n-mu][mu];
    }
    template <class Field_t, class Point, class Direction>
    typename kernels::std_types<Field_t>::ptSU3_t
    one_by_two(Field_t& U, const Point& n,
                     const Direction& mu,
                     const Direction& nu){
      return U[n + mu][nu] * U[n+mu+nu][nu] * 
             dag(U[n][nu] * U[n+nu][nu] * U[n+nu+nu][mu]) 
           + dag( U[n-nu-nu][mu] * U[n-nu-nu+mu][nu] * U[n-nu+mu][nu] ) * 
             U[n-nu-nu][nu] * U[n-nu][nu];
    }
    template <class Field_t, class Point, class Direction>
    typename kernels::std_types<Field_t>::ptSU3_t
    one_by_one(Field_t& U, const Point& n,
                     const Direction& mu,
                     const Direction& nu){
       return U[n + mu][nu] *  dag(U[n][nu] * U[n + nu][mu])
	    + dag(U[n-nu][mu] * U[n+mu-nu][nu]) * U[n - nu][nu];
    }
  }

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
  template <class Field_t>
  struct ImprovedSpatial {

    // collect info about the field
    typedef typename kernels::std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename kernels::std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename kernels::std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename kernels::std_types<Field_t>::bgf_t BGF;
    typedef typename kernels::std_types<Field_t>::point_t Point;
    typedef typename kernels::std_types<Field_t>::direction_t Direction;
    static const int ORD = kernels::std_types<Field_t>::order;
    static const int DIM = kernels::std_types<Field_t>::n_dim;

    typedef typename array_t<ptSU3, 2>::Type ptsu3_array_t;    
    typedef typename array_t<double, 2>::Type weight_array_t;    
    
    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 2;

    std::vector<ptSU3> val;
    std::vector<Cplx> result;
    weight_array_t weights;
    
    ImprovedSpatial(const double& c0, const double& c1) : 
      val(omp_get_max_threads(), ptSU3(bgf::zero<BGF>())), result(ORD + 1) {
      weights[0] = c0;
      weights[1] = c1;
    }

    void operator()(Field_t& U, const Point& n) {      
      for(Direction k(1); k.is_good(); ++k)
        for (Direction l(k+1); l.is_good(); ++l){
	  // 1x1 contribution
	  val[omp_get_thread_num()] += weights[0] * U[n][k] * detail::one_by_one(U, n, k, l);
	  // 2x1 contribution
          val[omp_get_thread_num()] += weights[1] * U[n][k] *
            (detail::two_by_one(U, n, k, l) + detail::one_by_two(U, n, k, l));
	}
    }
    void reduce() { 
      for (int i = 1; i < omp_get_max_threads(); ++i) val[0] += val[i];
      result[0] = -val[0].bgf().Tr() * 2;
      for (int i = 1; i <= ORD; ++i) result[i] = -val[0][i-1].tr() * 2;
    }
  };
  template <class Field_t>
  struct ImprovedTemporal {

    // collect info about the field
    typedef typename kernels::std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename kernels::std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename kernels::std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename kernels::std_types<Field_t>::bgf_t BGF;
    typedef typename kernels::std_types<Field_t>::point_t Point;
    typedef typename kernels::std_types<Field_t>::direction_t Direction;
    static const int ORD = kernels::std_types<Field_t>::order;
    static const int DIM = kernels::std_types<Field_t>::n_dim;

    typedef typename array_t<ptSU3, 2>::Type ptsu3_array_t;    
    typedef typename array_t<double, 2>::Type weight_array_t;    
    
    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 2;

    std::vector<ptSU3> val;
    std::vector<Cplx> result;
    weight_array_t weights;
    
    ImprovedTemporal(const double& c0, const double& c1) : 
      val(omp_get_max_threads(), ptSU3(bgf::zero<BGF>())), result(ORD + 1) {
      weights[0] = c0;
      weights[1] = c1;
    }

    void operator()(Field_t& U, const Point& n) {      
      Direction mu(0);
      for (Direction l(mu+1); l.is_good(); ++l){
        // 1x1 contribution
        val[omp_get_thread_num()] += weights[0] * U[n][mu] * detail::one_by_one(U, n, mu, l);
        // 2x1 contribution
        val[omp_get_thread_num()] += weights[1] * U[n][mu] *
          (detail::two_by_one(U, n, mu, l) + detail::one_by_two(U, n, mu, l));
      }
    }
    void reduce() { 
      for (int i = 1; i < omp_get_max_threads(); ++i) val[0] += val[i];
      result[0] = -val[0].bgf().Tr() * 2;
      for (int i = 1; i <= ORD; ++i) result[i] = -val[0][i-1].tr() * 2;
    }
  };

}
#endif /* _PLAQUETTE_H_ */
