#ifndef LOCAL_GLUON_FIELD_HPP
#define LOCAL_GLUON_FIELD_HPP

#include <vector>
#include <newQCDpt.h>
#include <Geometry.hpp>
#include <Point.hpp>
#include <Types.h>
#include <iostream>
#include <mpi.h>
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

  template <class BGF, int ORD, int DIM>
  class LocalGluonField {
  public:
    typedef  BGptSU3<BGF, ORD> ptSU3_t;
    typedef  BGptGluon<BGF, ORD, DIM> data_t;
    typedef std::vector< data_t > rep_t;

    typedef typename array_t<std::pair<int,int>, DIM>::Type neighbors_t;
    LocalGluonField (const typename
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

    void randomize() {
      for (typename rep_t::iterator U = rep.begin(); U != rep.end(); ++U)
        U->randomize();
    }
    
    typedef BGptSU3<BGF, ORD> ptSU3;
    ptSU3& operator()(const pt::Point<DIM> &n, const pt::Direction<DIM> &mu){
      return mu.template deref_bkw<ptSU3_t&, data_t&>
        (n.template deref<data_t>(rep));
    }
    const ptSU3& operator()(const pt::Point<DIM> &n, const pt::Direction<DIM> &mu) const {
      return mu.template deref_bkw<const ptSU3_t&, const data_t&>
        (n.template deref<const data_t>(rep));
    }
    pt::Point<DIM> mk_point(const typename geometry::Geometry<DIM>::raw_pt_t& n){
      g.mk_point(n);
    }
    geometry::SliceIterator<DIM> mk_slice_iterator 
    (const pt::Direction<DIM> mu, const int& xi){
      return g.mk_slice_iterator(mu, xi);
    }
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
      geometry::SliceIterator<DIM> iter = g.mk_slice_iterator(mu, xi);
      while (iter.is_good()){
        pt::Point<DIM> n = iter.yield();
        for (pt::Direction<DIM> nu; nu.is_good(); ++nu){
          rep[n][mu].buffer(i);
        }
      }
    }
    void read_slice_from_buffer(const pt::Direction<DIM>& mu, const int &xi,
                                std::vector<double>& buff){
      std::vector<double>::const_iterator i = buff.begin();
      geometry::SliceIterator<DIM> iter = g.mk_slice_iterator(mu, xi);
      while (iter.is_good()){
        pt::Point<DIM> n = iter.yield();
        for (pt::Direction<DIM> nu; nu.is_good(); ++nu){
          rep[n][mu].unbuffer(i);
        }
      }
    }
  };
}

#endif
