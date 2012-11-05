#ifndef LOCAL_FIELD_HPP
#define LOCAL_FIELD_HPP

#include <vector>
#include <newQCDpt.h>
#include <Geometry.hpp>
#include <Point.hpp>
#include <Types.h>
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef MPI
#include <mpi.h>
#endif
#include <algorithm>


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
  ///  \tparam BGF The background field class to be used.
  ///  \tparam DIM The number of space-time dimensions
  ///  \tparam ORD The perturbative order.
  ///
  ///  \ingroup MPI
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Mon Mar 26 14:44:03 2012

  template <class F, int DIM>
  class LocalField {
  public:
    typedef F data_t;
    typedef std::vector< data_t > rep_t;
    typedef pt::Point<DIM> Point;
    typedef typename array_t<std::pair<int,int>, DIM>::Type neighbors_t;
    LocalField (const typename
                     geometry::Geometry<DIM>::extents_t& e,
                     const int& number_of_threads,
                     const int& mpi_process_id,
                     const neighbors_t& mpi_neighbors) : 
      g(e), rep(g.vol()), n_th(number_of_threads), pid(mpi_process_id),
      neighbors(mpi_neighbors){
      /// constuct the buffers for communication
      for (int i = 0; i < DIM; ++i){
        send_buffer.push_back
          (std::make_pair ( std::vector<double> 
                            ( g.bnd_vol(i)*data_t::storage_size ),
                            std::vector<double> 
                            ( g.bnd_vol(i)*data_t::storage_size ) ) );
        rec_buffer.push_back
          (std::make_pair ( std::vector<double> 
                            ( g.bnd_vol(i)*data_t::storage_size ),
                            std::vector<double> 
                            ( g.bnd_vol(i)*data_t::storage_size ) ) );

      }
    }
    
    // copy constructor
    LocalField ( const LocalField< F , DIM >& other ) :
      g( other.g ), rep( other.rep ), n_th(other.n_th), pid(other.pid),
      neighbors(other.neighbors) { }


    void randomize() {
      for (typename rep_t::iterator U = rep.begin(); U != rep.end(); ++U)
        U->randomize();
    }

    int extent(const int& i) const {
      return g[i];
    }
    
    data_t& operator[](const pt::Point<DIM> &n){
      return n.template deref<data_t>(rep);
    }
    const data_t& operator[](const pt::Point<DIM> &n) const {
      return n.template deref<const data_t>(rep);
    }
    
    pt::Point<DIM> mk_point(const typename geometry::Geometry<DIM>::raw_pt_t& n){
      return g.mk_point(n);
    }
    geometry::SliceIterator<DIM, 0> mk_slice_iterator 
    (const pt::Direction<DIM> mu, const int& xi){
      return g.template mk_slice_iterator<0>(mu, xi);
    }
    geometry::SliceIterator<DIM, 2> mk_bulk_slice_iterator
    (const pt::Direction<DIM> mu, const int& xi){
      return g.mk_slice_iterator<2>(mu, xi, 1);
    }
    template <class M>
    void measure_on_slice(M& f, 
                          const pt::Direction<DIM>& d,
                          const int& xi){
      geometry::SliceIterator<DIM, 1> iter =
        g.template mk_slice_iterator<1>(d, xi, 0);
      while (iter.is_good()) f(*this, iter.yield());
    }
    template <class M>
    void measure_on_slice_with_bnd(M& f, 
                          const pt::Direction<DIM>& d,
                          const int& xi){
      geometry::SliceIterator<DIM, 0> iter =
        g.template mk_slice_iterator<0>(d, xi, 0);
      while (iter.is_good()) f(*this, iter.yield());
    }
    template <class M>
    void apply_on_slice_with_bnd(M& f, 
                          const pt::Direction<DIM>& d,
                          const int& xi){
      geometry::SliceIterator<DIM, 0> iter =
        g.template mk_slice_iterator<0>(d, xi, 0);
      while (iter.is_good()) f(*this, iter.yield());
    }
    template <class M>

    void apply_on_timeslice(M& f, const int& t){
      // parallelize with a simple checker-board scheme ...
      typedef typename geometry::CheckerBoard<DIM>::slice slice;
      typedef typename geometry::CheckerBoard<DIM>::bin bin;
#ifdef IMP_ACT
      const int CHECKER_BOARD_SIZE = 2;
#else
      const int CHECKER_BOARD_SIZE = 1;
#endif
      static geometry::CheckerBoard<DIM> cb(g, CHECKER_BOARD_SIZE);
      for (typename slice::const_iterator s = cb[t].begin();
           s != cb[t].end(); ++s){
        // here, we have to do a nasty workaround
        // parallel for does not like lists, which is what I used in
        // the CheckerBoard class. This is e.g. great for the test
        // case of the CB class and foremost for the case that we have
        // some overlap at the edges (i.e. if CHECKER_BOARD_SIZE % L
        // != 0). Here, however a vector would be nice. Let's see...
        // FIXME: This is nasty!!
        std::vector<pt::Point<DIM> > v(s->begin(), s->end());
        int N = v.size();
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
          f(*this, v[i]);
   }

      //int size = g.bnd_vol(0), sizeh = size/2, i = 0;
      // use this to remember the points
//      static std::vector<std::vector<pt::Point<DIM> > > pts(g[0] + 1);
//      if (pts[t].size() == 0){
//        geometry::SliceIterator<DIM, 0> iter =
//          g.template mk_slice_iterator<0>(pt::Direction<DIM>(0), t, 0);
//
//        pts[t].resize(size, g.begin());
//        while (iter.is_good()){
//          pts[t][i] = iter.yield();
//          pts[t][sizeh + i] = iter.yield();
//          ++i;
//        }
//      }
//
//#pragma omp parallel for default(none) shared(f, t, size, pts, sizeh)
//      for (int i = 0; i < sizeh; ++i)
//        f(*this, pts[t][i]);
//#pragma omp parallel for default(none) shared(f, t, size, pts, sizeh)
//      for (int i = sizeh; i < size; ++i)
//        f(*this, pts[t][i]);
    }
    template <class M>
    void apply_everywhere(M& f){
      for (int t = 0; t < g[0]; ++t)
        apply_on_timeslice(f, t);
    }
    template <class M>
    void apply_everywhere_serial(M& f){
      // doesn't work because bind1st makes a copy
      // we don't want that because of local data members
      // such as the fstreams in the disk reader/writer
      //std::for_each(g.begin(), g.end(), std::bind1st(f, *this));
      for(pt::Point<DIM> n = g.begin(), e = g.end(); n != e; ++n)
        f(*this, n);
    }
#ifdef MPI
    MPI_Request test_send_fwd_z(){
      write_slice_to_buffer(pt::Direction<DIM>(3), 4,
                            send_buffer[3].second);
      MPI_Request r;
      MPI_Isend(&send_buffer[3].second[0],
                send_buffer[3].second.size(),
                MPI_DOUBLE, neighbors[3].second, 0, MPI_COMM_WORLD, &r);
      return r;
    }
    void test_rec_bkw_z(){
      MPI_Status status;
      MPI_Recv(&rec_buffer[3].first[0],
      rec_buffer[3].first.size(),
               MPI_DOUBLE, neighbors[3].first, 0, MPI_COMM_WORLD,
               &status);
      read_slice_from_buffer(pt::Direction<DIM>(3), 0,
                             rec_buffer[3].first);
    }
#endif

    // Added: MB on Fri 2, 2012
    pt::Point<DIM> begin() { return g.begin(); }
    pt::Point<DIM> end()   { return g.end(); }

    LocalField<F, DIM>& operator=(LocalField<F, DIM>& other) {
      std::copy( other.rep.begin(), other.rep.end(), rep.begin() );
      return *this;
    }

    LocalField<F, DIM>& operator+=(LocalField<F, DIM>& other) {
      for (typename rep_t::iterator U = rep.begin(), V = other.rep.begin(); U != rep.end(); ++U, ++V)
	(*U) += (*V);
      return *this;
    }
    LocalField<F, DIM>& operator-=(LocalField<F, DIM>& other) {
      for (typename rep_t::iterator U = rep.begin(), V = other.rep.begin(); U != rep.end(); ++U, ++V)
	(*U) -= (*V);
      return *this;
    }

    LocalField<F, DIM> operator+(const LocalField<F, DIM>& other) {
      LocalField<F, DIM> result(*this);
      return result+=other;
    }
    LocalField<F, DIM> operator-(const LocalField<F, DIM>& other) {
      LocalField<F, DIM> result(*this);
      return result-=other;
    }
    Cplx operator*(LocalField<F, DIM>& other) {
      Cplx result;

      typename rep_t::iterator first = rep.begin();
      typename rep_t::iterator last = rep.end();
      typename rep_t::iterator first2 = other.rep.begin();
      while (first != last) {
	result += (*first++)*(*first2++);
      }
      return result;
    }

    template<class C>
    LocalField<F, DIM>& operator*=(C& other) {
      for (typename rep_t::iterator U = rep.begin(); U != rep.end(); ++U)
	(*U) *= other;
      return *this;
    }
    template<class C>
    LocalField<F, DIM> operator*(const C& other) {
      LocalField<F, DIM> result(*this);
      return (result *= other);
    }
    


  private:
    geometry::Geometry<DIM> g;
    rep_t rep;
    int n_th; // number of threads
    int pid; // MPI process id
    neighbors_t neighbors;
    /// Buffer for communication
    std::vector< std::pair<std::vector<double>, 
                           std::vector<double> > > send_buffer;
    std::vector< std::pair<std::vector<double>, 
                           std::vector<double> > > rec_buffer;
    void write_slice_to_buffer(const pt::Direction<DIM>& mu, const int &xi,
                               std::vector<double>& buff){
      std::vector<double>::iterator i = buff.begin();
      geometry::SliceIterator<DIM, 0> iter = 
        g.template mk_slice_iterator<0>(mu, xi);
      while (iter.is_good()) rep[iter.yield()].buffer(i);
    }
    void read_slice_from_buffer(const pt::Direction<DIM>& mu, const int &xi,
                                std::vector<double>& buff){
      std::vector<double>::const_iterator i = buff.begin();
      geometry::SliceIterator<DIM, 0> iter = 
        g.template mk_slice_iterator<0>(mu, xi);
      while (iter.is_good()) rep[iter.yield()].unbuffer(i);
    }
  };
}

#endif
