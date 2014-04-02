#ifndef CLOVER_H_
#define CLOVER_H_

#include <Types.h>
#include <Kernels.hpp>
#include <Background.h>

#ifdef _OPENMP
#include <omp.h>
#else
namespace clover {
  int omp_get_max_threads() { return 1; }
  int omp_get_thread_num() { return 0; }
}
#endif


namespace clover {
  namespace detail {
    template <class F>
    struct FieldWrapper {
      typedef typename kernels::std_types<F>::ptSU3_t ptSU3;
      typedef typename kernels::std_types<F>::direction_t Direction;
      typedef typename kernels::std_types<F>::point_t Point;
      static const int DIM = kernels::std_types<F>::n_dim;
      const F& fld;
      FieldWrapper(const F& f) : fld(f) { }
      ptSU3 operator()(const Point& n, const Direction& mu) const {
        if (mu >= DIM) {
	  return dag(fld[n - (-mu)][-mu]);
	}
        else return fld[n][mu];
      }
      ptSU3 operator()(Point n, const Direction& mu, const Direction& nu) const {
	ptSU3 tmp = (*this)(n, mu);
	n += mu;
	tmp *= (*this)(n, nu);
	n += nu;
	tmp *= (*this)(n, -mu);
	n -= mu;
	tmp *= (*this)(n, -nu);
	return tmp;
      }
    };
    template <class F> const int FieldWrapper<F>::DIM;
  } // end namespace detail

  template <class Fld_t>
  struct F {
    typedef typename kernels::std_types<Fld_t>::ptSU3_t ptSU3;
    typedef typename kernels::std_types<Fld_t>::point_t Point;
    typedef typename kernels::std_types<Fld_t>::direction_t Direction;
    Direction mu_, nu_;

    F(const Direction& mu, const Direction& nu) : mu_(mu), nu_(nu) { }

    ptSU3 operator()(const Fld_t& UU, const Point& n) const {
      detail::FieldWrapper<Fld_t> U(UU);
      return 1./8 * ( U(n, mu_, nu_) - U(n, nu_, mu_)
		      + U(n, nu_, -mu_) - U(n, mu_, -nu_)
		      + U(n, -mu_, -nu_) - U(n, -nu_, -mu_)
		      + U(n, -nu_, mu_) - U(n, -mu_ ,nu_));
    }
  };

  template <class Fld_t>
  struct E0s {
    typedef typename kernels::std_types<Fld_t>::ptSU3_t ptSU3;
    typedef typename kernels::std_types<Fld_t>::point_t Point;
    typedef typename kernels::std_types<Fld_t>::direction_t Direction;
    typedef typename kernels::std_types<Fld_t>::bgf_t BGF;
    static const int ORD = kernels::std_types<Fld_t>::order;
    static const int n_cb = 0;

    std::vector<ptSU3> val;
    std::vector<Cplx> result;
    E0s() : val(omp_get_max_threads(), ptSU3(bgf::zero<BGF>())), result(ORD + 1) { }

    void operator()(const Fld_t& UU, const Point& n) {
      ptSU3 tmp;
      for (Direction k(1); k.is_good(); ++k)
	for (Direction l(k + 1); l.is_good(); ++l){
	  F<Fld_t> Fkl(k, l);
	  tmp = Fkl(UU, n);
	  val[omp_get_thread_num()] += tmp*tmp;
	}
    }
    void reduce() {
      for (int i = 1; i < omp_get_max_threads(); ++i) val[0] += val[i];
      result[0] = -val[0].bgf().Tr();
      for (int i = 1; i <= ORD; ++i) result[i] = -val[0][i-1].tr();
    }
  };

  template <class Fld_t>
  struct E0m {
    typedef typename kernels::std_types<Fld_t>::ptSU3_t ptSU3;
    typedef typename kernels::std_types<Fld_t>::point_t Point;
    typedef typename kernels::std_types<Fld_t>::direction_t Direction;
    typedef typename kernels::std_types<Fld_t>::bgf_t BGF;
    static const int ORD = kernels::std_types<Fld_t>::order;
    static const int n_cb = 0;
    
    std::vector<ptSU3> val;
    std::vector<Cplx> result;
    E0m() : val(omp_get_max_threads(), ptSU3(bgf::zero<BGF>())), result(ORD + 1) { }

    void operator()(const Fld_t& UU, const Point& n) {
      Direction t;
      ptSU3 tmp;
      for (Direction k(1); k.is_good(); ++k){
	F<Fld_t> F0k(t, k);
	tmp = F0k(UU, n);
	val[omp_get_thread_num()] += tmp*tmp;
      }
    }

    void reduce() {
      for (int i = 1; i < omp_get_max_threads(); ++i) val[0] += val[i];
      result[0] = -val[0].bgf().Tr();
      for (int i = 1; i <= ORD; ++i) result[i] = -val[0][i-1].tr();
    }
  };
}

#endif
