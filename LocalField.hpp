#ifndef LOCAL_FIELD_HPP
#define LOCAL_FIELD_HPP

#include <vector>
#include <newQCDpt.h>
#include <Geometry.hpp>
#include <Point.hpp>
#include <Types.h>
#include <iostream>
#include <numeric>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#include <Communicator.hpp>
#else
namespace comm {
  template<class C>
  struct Communicator {
    typedef typename C::extents_t extents_t;
    typedef typename C::neighbors_t nb_t;
    Communicator(const extents_t& e) : zero(0) { }
    int zero;
    nb_t n;
    const int& rank() const { return zero; }
    const nb_t& nb() const { return n; }
  };
}
#endif
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#else
namespace fields {
  namespace detail {
    int omp_get_max_threads() { return 1; }
    int omp_get_thread_num() { return 0; }
  }
}
#endif

namespace kernels {
  template <class Field_t, class OutputIterator>
  struct Buffer;

  template <class Field_t, class InputIterator>
  struct Unbuffer;
}


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///  Updated (MPI-enabled) versions of the perturbative fields.
///
///  The MPI enabled versions of the field variables have to take
///  care of communication.
///
///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
///  \date Mon Mar 26 14:49:53 2012
namespace fields {

  namespace detail {

    template <class Field_t> class inplace_add {
    public:
      static const int dim = Field_t::dim;
      inplace_add(const Field_t &F) : other(&F) { }
      void operator()(Field_t& F, const pt::Point<dim>& n) const {
	F[n] += (*other)[n];
      }
      static const int n_cb = 0;
    private:
      Field_t const * other;
    };

   template <class Field_t> class inplace_sub {
    public:
      static const int dim = Field_t::dim;
      inplace_sub(const Field_t &F) : other(&F) { }
      void operator()(Field_t& F, const pt::Point<dim>& n) const {
	F[n] -= (*other)[n];
      }
     static const int n_cb = 0;
    private:
      Field_t const * other;
    };

    template <class Field_t, class Scalar_t>
    class inplace_smul {
    public:
      static const int dim = Field_t::dim;
      inplace_smul(const Scalar_t &s) : other(s) { }
      void operator()(Field_t& F, const pt::Point<dim>& n) const {
	F[n] *= other;
      }
      static const int n_cb = 0;
    private:
      Scalar_t other;
    };


    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    ///
    ///  The name makes sense for vectors: in this case is
    ///  "inner product" in strict sense. For matrices
    ///  it means whatever is the definition of operator*
    ///
    ///  \author Michele Brambilla <mib.mic@gmail.com>
    ///  \date Thu Jan 17 10:40:05 2013
    template <class Field_t> class inner_prod {
    public:
      static const int dim = Field_t::dim;
      static const int n_cb = 0;
      inner_prod(const Field_t &F) :
      result(omp_get_max_threads(), Cplx(0,0)), other(&F) { }

      void operator()(const Field_t& F, const pt::Point<dim>& n) {
	result[omp_get_thread_num()] += F[n] * (*other)[n];
      }

      Cplx reduce() const {
	return std::accumulate(result.begin(), result.end(), Cplx(0,0));
      }
    private:
      std::vector<Cplx> result;
      Field_t const * other;
    };


    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    ///
    ///  In case of vectors this the element-by-element
    ///  product (defined according to opertator^).
    ///  For matrices it means whatever is the definition
    ///  of operator^
    ///
    ///  \author Michele Brambilla <mib.mic@gmail.com>
    ///  \date Thu Jan 17 10:40:05 2013
    template <class Field_t> class prod {
    public:
      static const int dim = Field_t::dim;
      static const int n_cb = 0;
      prod(const Field_t &F) :
      result(omp_get_max_threads(), Cplx(0,0)), other(&F) { }

      void operator()(const Field_t& F, const pt::Point<dim>& n) {
	result[omp_get_thread_num()] += F[n] ^ (*other)[n];
      }

      Cplx reduce() const {
	return std::accumulate(result.begin(), result.end(), Cplx(0,0));
      }
    private:
      std::vector<Cplx> result;
      Field_t const * other;
    };


  }

  /// \defgroup MPI

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Local gluon field.
  ///
  ///  By local we mean that it resides on one node. One
  ///  LocalGluonField instance is always associated with one MPI
  ///  process, but will generally spawn multiple threads such that
  ///  the advantages of a cluster using nodes with
  ///  modern CPUs with several cores each may be fully exploited.
  ///
  ///  \tparam F   The type of elements residing at the lattice sites.
  ///  \tparam DIM The number of space-time dimensions.
  ///
  ///  \ingroup MPI
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Mon Mar 26 14:44:03 2012

  template <class F, int DIM>
  class LocalField {
  public:
    typedef LocalField<F,DIM> self_t;
    typedef F data_t;
    static const int dim = DIM;
    typedef std::vector< data_t > rep_t;
    typedef typename rep_t::iterator iterator;
    typedef typename rep_t::const_iterator const_iterator;
    typedef pt::Point<DIM> Point;
    typedef typename array_t<std::pair<int,int>, DIM>::Type neighbors_t;
    typedef typename geometry::Geometry<DIM>::extents_t extents_t;
    typedef typename geometry::Geometry<DIM>::raw_pt_t raw_pt;
    LocalField (const extents_t& e,
		const int& number_of_threads,
		const int& mpi_process_id,
		const neighbors_t& mpi_neighbors) :
      g(e), rep(g.vol()), n_th(number_of_threads), comm(e),
      pid(comm.rank()),neighbors(comm.nb()){
    }
    int vol() const { return g.vol(); }
    int extent(const int& i) const {
      return g[i];
    }
    extents_t extents() const { return g.get_extents(); }
    data_t& operator[](const pt::Point<DIM> &n){
      return n.template deref<rep_t>(rep);
    }
    const data_t& operator[](const pt::Point<DIM> &n) const {
      return n.template deref<rep_t>(rep);
    }

    iterator begin() { return rep.begin(); };
    iterator end() { return rep.end(); };
    const_iterator begin() const { return rep.begin(); };
    const_iterator end() const { return rep.end(); };

    pt::Point<DIM> mk_point(const raw_pt& n){
      return g.mk_point(n);
    }
    raw_pt coords(const Point& n){
      return g.coords(n);
    }

    template <class M>
    M& apply_on_timeslice(M& f, const int& t){
#ifdef USE_MPI
      buffer(t);
      comm.do_it();
      unbuffer(t);
      return apply_on_timeslice_impl
	<M, geometry::BulkIterator>(f, t, ParCheck<M>());
#else
      return apply_on_timeslice_impl
	<M, geometry::TimeSliceIter>(f, t, ParCheck<M>());
#endif
    }
    template <class M>
    M& apply_on_timeslice(M& f, const int& t) const {
#ifdef USE_MPI
      buffer(t);
      comm();
      unbuffer(t);
      return apply_on_timeslice_impl
	<M, geometry::BulkIterator>(f, t, ParCheck<M>());
#else
      return apply_on_timeslice_impl
	<M, geometry::TimeSliceIter>(f, t, ParCheck<M>());
#endif
      // return apply_on_timeslice_impl
      // 	<M, geometry::TimeSliceIter>(f, t, ParCheck<M>());;
    }
    template <class M>
    M& apply_on_timeslice_bulk(M& f, const int& t){
#ifdef USE_MPI
      buffer();
      comm();
      unbuffer();
#endif
      return apply_on_timeslice_impl
	<M, geometry::BulkIterator>(f, t, ParCheck<M>());
    }
    template <class M>
    M& apply_on_timeslice_bulk(M& f, const int& t) const {
#ifdef USE_MPI
      buffer();
      comm();
      unbuffer();
#endif
      return apply_on_timeslice_impl
	<M, geometry::BulkIterator>(f, t, ParCheck<M>());;
    }
    template <class M>
    M& apply_everywhere(M& f){
      for (int t = 0; t < g[0]; ++t)
	apply_on_timeslice(f, t);
      return f;
    }
    template <class M>
    M& apply_everywhere(M& f) const {
      for (int t = 0; t < g[0]; ++t)
	apply_on_timeslice(f, t);
      return f;
    }
    template <class M>
    const M& apply_everywhere(const M& f){
      for (int t = 0; t < g[0]; ++t)
	apply_on_timeslice(f, t);
      return f;
    }
    template <class M>
    const M& apply_everywhere(const M& f) const {
      for (int t = 0; t < g[0]; ++t)
	apply_on_timeslice(f, t);
      return f;
    }
    template <class M>
    M& apply_everywhere_bulk(M& f){
      for (int t = 0; t < g[0]; ++t)
	apply_on_timeslice_bulk(f, t);
      return f;
    }
    template <class M>
    M& apply_everywhere_bulk(M& f) const {
      for (int t = 0; t < g[0]; ++t)
	apply_on_timeslice(f, t);
      return f;
    }
    template <class M>
    const M& apply_everywhere_bulk(const M& f){
      for (int t = 0; t < g[0]; ++t)
	apply_on_timeslice_bulk(f, t);
      return f;
    }
    template <class M>
    const M& apply_everywhere_bulk(const M& f) const {
      for (int t = 0; t < g[0]; ++t)
	apply_on_timeslice_bulk(f, t);
      return f;
    }

    // Added: MB on Fri 2, 2012

    LocalField& operator+=(LocalField& other) {
      apply_everywhere(detail::inplace_add<LocalField>(other));
      return *this;
    }
    LocalField& operator-=(LocalField& other) {
      apply_everywhere(detail::inplace_sub<LocalField>(other));
      return *this;
    }
    LocalField& operator+=(const LocalField& other) {
      apply_everywhere(detail::inplace_add<LocalField>(other));
      return *this;
    }
    LocalField& operator-=(const LocalField& other) {
      apply_everywhere(detail::inplace_sub<LocalField>(other));
      return *this;
    }

    LocalField operator+(const LocalField& other) {
      LocalField result(*this);
      return result+=other;
    }
    LocalField operator-(const LocalField& other) {
      LocalField result(*this);
      return result-=other;
    }

    template<class C>
    LocalField& operator*=(C& other) {
      apply_everywhere(detail::inplace_smul<LocalField, C>(other));
      return *this;
    }
    template<class C>
    LocalField operator*(const C& other) const {
      LocalField result(*this);
      return (result *= other);
    }
    // DH: maybe call this prod instead?
    Cplx operator*(const LocalField& other) const {
      detail::inner_prod<LocalField> p(other);
      return apply_everywhere(p).reduce();
    }

    ///  \author Michele Brambilla <mib.mic@gmail.com>
    ///  \date Thu Jan 10 20:47:14 2013
    Cplx operator^(const LocalField& other) const {
      detail::prod<LocalField> p(other);
      return apply_everywhere(p).reduce();
    }

    comm::Communicator<self_t>& get_communicator() { return comm; }
    // void do_buffer(const int& t) { buffer(t); }

  private:
    ////////////////////////////////////////////////////////////
    //
    //  Some magic to make the auto-parallelization (or the lack of
    //  parallelization work.
    //
    //  \date      Wed Apr 17 14:08:24 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>
    struct True {};
    struct False {};
    // Set ParCheck<T> for any type T to true.
    template <class C, class Dummy = void>
    // template <class C, class Dummy = void>
    struct ParCheck : True { };
    // Set ParCheck<T> for a type that has a
    //  typedef void NoPar;
    // to false, using partial specializaiton
    template <class C>
    struct ParCheck<C, typename C::NoPar> : False { };
    ////////////////////////////////////////////////////////////
    //
    //  Now for the actual apply on timeslice functions
    //
    //  \date      Wed Apr 17 14:11:24 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>
    template <class M, class Iter>
    M& apply_on_timeslice_impl(M& f, const int& t, True){
      // parallelize with a simple checker-board scheme ...
      typedef typename geometry::CheckerBoard<DIM, M::n_cb, Iter>::slice slice;
      typedef typename geometry::CheckerBoard<DIM, M::n_cb, Iter>::bin bin;
      static geometry::CheckerBoard<DIM, M::n_cb, Iter> cb(g);
      for (typename slice::const_iterator s = cb[t].begin();
	   s != cb[t].end(); ++s){
	int N = s->size();
#pragma omp parallel for
	for (int i = 0; i < N; ++i)
	  f(*this, (*s)[i]);
      }
      return f;
    }
    template <class M, class Iter>
    M& apply_on_timeslice_impl(M& f, const int& t, True) const {
      // parallelize with a simple checker-board scheme ...
      typedef typename geometry::CheckerBoard<DIM, M::n_cb, Iter>::slice slice;
      typedef typename geometry::CheckerBoard<DIM, M::n_cb, Iter>::bin bin;
      static geometry::CheckerBoard<DIM, M::n_cb, Iter> cb(g);
      for (typename slice::const_iterator s = cb[t].begin();
	   s != cb[t].end(); ++s){
	int N = s->size();
#pragma omp parallel for
	for (int i = 0; i < N; ++i)
	  f(*this, (*s)[i]);
      }
      return f;
    }
    template <class M, class Iter>
    M& apply_on_timeslice_impl(M& f, const int& t, False) {
      typename geometry::Geometry<DIM>::raw_pt_t n;
      n[0] = t;
      for (int i = 1; i < DIM; ++i) n[i] = Iter::get_start(i, g);
      Iter x(g.mk_point(n), g.get_extents());
      do { f(*this, *x); } while ((++x).is_good());
    }
    template <class M, class Iter>
    M& apply_on_timeslice_impl(M& f, const int& t, False) const {
      typename geometry::Geometry<DIM>::raw_pt_t n;
      n[0] = t;
      for (int i = 1; i < DIM; ++i) n[i] = Iter::get_start(i, g);
      Iter x(g.mk_point(n), g.get_extents());
      do { f(*this, *x); } while ((++x).is_good());
    }


    ///////////////////////////////
    //
    // Buffer and unbuffer data for communications. Are iterators
    // correct?
    //
    void buffer(const int& t) const {
      pt::Direction<DIM> mu(3);
      typedef typename kernels::Buffer<self_t, typename std::vector<F>::const_iterator> bk_t;
      bk_t buff_f(comm.send_buff()[int(mu)].first.begin());
      bk_t buff_s(comm.send_buff()[int(mu)].second.begin());
      // apply_on_timeslice_impl<bk_t, geometry::ZeroBndIterator>(buff_f, t, ParCheck<bk_t>());
      // apply_on_timeslice_impl<bk_t, geometry::TBndIterator>(buff_s, t, ParCheck<bk_t>());
      raw_pt n;
      n[0] = t; n[3] = 1;
      for (n[1] = 0; n[1] < g[1]; ++n[1])
	for (n[2] = 0; n[2] < g[2]; ++n[2])
	  buff_f(*this,mk_point(n));
      n[0] = t; n[3] = g[3]-1;
      for (n[1] = 0; n[1] < g[1]; ++n[1])
	for (n[2] = 0; n[2] < g[2]; ++n[2])
	  buff_s(*this,mk_point(n));
    }
    void buffer(const int& t) {
      pt::Direction<DIM> mu(3);
      typedef typename kernels::Buffer<self_t, typename std::vector<F>::iterator> bk_t;
      bk_t buff_f(comm.send_buff()[int(mu)].first.begin());
      bk_t buff_s(comm.send_buff()[int(mu)].second.begin());
      // apply_on_timeslice_impl<bk_t, geometry::ZeroBndIterator>(buff_f, t, ParCheck<bk_t>());
      // apply_on_timeslice_impl<bk_t, geometry::TBndIterator>(buff_s, t, ParCheck<bk_t>());
      raw_pt n = {t,0,0,1};
      for (n[1] = 0; n[1] < g[1]; ++n[1])
	for (n[2] = 0; n[2] < g[2]; ++n[2])
	  buff_f(*this,mk_point(n));
      n[3] = g[3]-2;
      for (n[1] = 0; n[1] < g[1]; ++n[1])
	for (n[2] = 0; n[2] < g[2]; ++n[2])
	  buff_s(*this,mk_point(n));
    }
    void unbuffer(const int& t) {
      pt::Direction<DIM> mu(3);
      typedef typename kernels::Unbuffer<self_t, typename std::vector<F>::iterator> uk_t;
      uk_t ubuff_f(comm.recv_buff()[int(mu)].first.begin());
      uk_t ubuff_s(comm.recv_buff()[int(mu)].second.begin());
      apply_on_timeslice_impl<uk_t, geometry::ZeroBndIterator>(ubuff_f, t, ParCheck<uk_t>());
      apply_on_timeslice_impl<uk_t, geometry::TBndIterator>(ubuff_s, t, ParCheck<uk_t>());
    }

    geometry::Geometry<DIM> g;
    rep_t rep;
    int n_th; // number of threads
    int pid; // MPI process id
    neighbors_t neighbors;
    comm::Communicator<self_t> comm;
  };
}

#endif
