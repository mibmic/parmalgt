#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <MyMath.h>
#include <Types.h>
#include <Point.hpp>

namespace geometry {

  namespace detail {
    template <int N>
    inline int product (const typename array_t<int, N>::Type &v){
      int p = 1;
      for (typename array_t<int, N>::Type::const_iterator i = v.begin(); i !=
             v.end(); ++i) p *= *i;
      return p;
    }    
  }

  template<int DIM, int SKIP> class SliceIterator;

  
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Lattice geometry helper class.
  ///
  ///  This class stores
  ///     - A list of neighbors (Geometry::neighbors).
  ///     - The extents of the lattice (Geometry::extents).
  ///     - The DIM-volume Geometry::V.
  ///     - The volumes of the boundaries (Geometry::bnd_vols).
  ///
  ///  \tparam DIM Space-time dimensions.
  ///
  ///  \ingroup MPI
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Mon Mar 26 12:12:02 2012
  template <int DIM>
  class Geometry {
  public:
    /// Type for integer DIM-Vectors.
    /// Used for the construction of a point n = (n1, n2, ...).
    /// For the day-to day use, you could write a wrapper function
    /// or consider the begin() and end() methods to swipe the whole
    /// lattice .
    typedef typename array_t<int, DIM>::Type raw_pt_t;
    /// Type to pass the lattice extents/
    /// The same as raw_pt_t, an integer DIM-vector.
    typedef typename array_t<int, DIM>::Type extents_t;
    /// Type to store the volumes of the boundaries.
    typedef extents_t vols_t;
    /// Constructor.
    /// Only needs lattice extents as input. We recommend using a
    /// helper function to create the parameter.
    Geometry (const extents_t &lattice_extents)
    : extents(lattice_extents),
      V(detail::product<DIM>(lattice_extents)),
      neighbors (detail::product<DIM>(lattice_extents), 
                  typename array_t<int, 2*DIM>::Type()) { 
      // initialize neighbors
      fill_neighbors();
      // boundary volumes
      for (int i = 0; i < DIM; ++i)
        bnd_vols[i] = V / extents[i];
    }
    /// Generate an iterator over a given slice.
    /// The SliceIterator returned will ierate all lattice points
    /// \f$n\f$, keeping \f$n_\mu = x_i\f$ fixed.
    /// For all other directions, the iteration range is
    /// \f[ 0 \leq x_\nu < L_\nu, \quad \nu = 0,\ldots, DIM - 1, \quad  \nu \neq \mu\f]
    ///
    ///
    /// \param mu Called \f$\mu\f$ above.
    /// \param xi Called \f$n_i\f$ above.
    template <int N>
    SliceIterator<DIM, N> mk_slice_iterator (const pt::Direction<DIM> mu,
                                             const int& xi,
                                             const int& x_start = 0){
      raw_pt_t n;
      std::fill(n.begin(), n.end(), x_start);
      n[mu] = xi;
      return SliceIterator<DIM, N>(mk_point(n), mu, extents);
    }
    
    pt::Point<DIM> mk_point(const raw_pt_t &n) const {
      return pt::Point<DIM>(mk_label(n), neighbors.begin());
    }
    pt::Point<DIM> begin() const {
      return pt::Point<DIM>(0, neighbors.begin());
    }
    pt::Point<DIM> end() const {
      return pt::Point<DIM>(V, neighbors.begin());
    }
    int bnd_vol(const int& dim) const {
      return bnd_vols[dim];
    }
    int vol() const { return V; }
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Get the extend in a certain direction.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Mon Jun 11 18:33:23 2012
    int operator[](const int& i) const { return extents[i]; }
  private:
    /// Local lattice extends.
    extents_t extents;
    /// Stores the nearest neighbors.  Given a point \f$n\f$,
    /// neighbors[n][DIM + mu] is the coordinate of 
    /// \f$ n + \hat \mu \f$, neighbors[n][mu] the one of 
    /// \f$ n - \hat \mu \f$, with \f$ \mu = 0,\ldots, DIM\f$.

    std::vector<typename array_t<int, 2*DIM>::Type> neighbors;
    /// Voulumes of the boundaries
    vols_t bnd_vols;
    /// DIM-voulume of the local lattice
    int V;
    /// Label (integer value) belonging to a give point
    int mk_label(const raw_pt_t &n) const {
      int vol = 1, label = 0;
      for (int i = 0; i < DIM; ++i){
        label += ((n[i] + extents[i]) % extents[i]) * vol;
        vol *= extents[i];
      }
      return label;
    }
    /// helper function to recursively fill neighbors
    void fill_neighbors (const int& n = DIM - 1, raw_pt_t x = raw_pt_t()){
      if (!n)
        for (int xi = 0; xi < extents[0]; ++xi){
          x[0] = xi;
          int l = mk_label(x);
          for(pt::Direction<DIM> mu; mu.is_good(); ++mu){
            x[mu]++;
            neighbors[l][DIM + mu] = mk_label(x);
            x[mu] -= 2;
            neighbors[l][mu] = mk_label(x);
            x[mu]++;
          }
        }
      else{
        for (int xi = 0; xi < extents[n]; ++xi){
          x[n] = xi;
          fill_neighbors (n-1, x);
        }
      }
    }
  };

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Class to facilitate the iteration over a slice of a Geometry.
  ///
  ///  Given a point \f$x\f$, and a direction \f$\mu\f$, yield will
  ///  successively return a sequence of points \f$x^{(i)}\f$, with
  ///  the \f$\mu\f$ component fixed \f$x^{(i)}_\mu = x_\mu\f$, but
  ///  all the other components sweeping the whole lattice extent.
  ///
  ///  \tparam DIM Number of space-time dimensions.
  ///
  ///  \ingroup MPI
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Mon Mar 26 12:58:50 2012

  template<int DIM, int SKIP>
  class SliceIterator {
  public:
    /// Constructor.
    /// \param x The starting point, called \f$x\f$ in the class
    ///          description.
    /// \param mu The direction to keep fixed, called \f$\mu\f$ int
    ///          the class escription.
    /// \param e Extents of the underlying Geometry.
    SliceIterator (const pt::Point<DIM> &x, 
                   const pt::Direction<DIM> mu,
                   const typename Geometry<DIM>::extents_t& e,
                   const int& stepsize = 1) :
      x_current(x), mu_exclude(mu), extents(e), counters(), 
      good_flag(true) { }

    /// Return the next point.
    /// Note you should not call this if is_good() returns false.
    ///
    /// Typical use case:
    /// 
    pt::Point<DIM> yield() {
      pt::Point<DIM> result = x_current;
      int mu = 0 == mu_exclude ? 1 : 0;
      do {
          x_current += pt::Direction<DIM>(mu);
          counters[mu]++;
        if (counters[mu] == extents[mu] - SKIP){
          counters[mu] = 0;
          // this uses periodicity !!
          x_current += pt::Direction<DIM>(mu);
          for (int n = 0; n < SKIP; ++n)
            x_current += pt::Direction<DIM>(mu);
          if (++mu == mu_exclude) ++mu;
        }
        else break;
      } while (mu <= DIM);
      if (mu == DIM) good_flag = false;
      return result;
    }

    bool is_good() const {
      return good_flag;
    }
  private:
    pt::Point<DIM> x_current;
    pt::Direction<DIM> mu_exclude;
    typename Geometry<DIM>::extents_t extents;
    typename array_t<int, DIM>::Type counters;
    bool good_flag;
    int step;
  };
}

#endif
