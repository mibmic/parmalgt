#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <Types.h>
#include <Point.hpp>
#include <math.h>
#include <list>

namespace geometry {

  template <int DIM> class Geometry;

  namespace detail {
    template <int N>
    inline int product (const typename array_t<int, N>::Type &v){
      int p = 1;
      for (typename array_t<int, N>::Type::const_iterator i = v.begin(); i !=
             v.end(); ++i) p *= *i;
      return p;
    }

    ////////////////////////////////////////////////////////////
    //
    //  Increment Policies.
    //
    //  These are policies how to increment the individual
    //  dimensions when iterating. The final iterator will then be a
    //  collection of such policies, one for each direction.
    //
    //  \date      Tue Apr  9 16:27:53 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>


    ////////////////////////////////////////////////////////////
    //
    //  Constant policy. This just keeps the corresponding component
    //  constant.
    //
    //  \date      Tue Apr 16 11:16:09 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>
    struct ConstPolicy {
      explicit ConstPolicy (int) { }
      template <int DIM>
      void next(const pt::Point<DIM>&, const pt::Direction<DIM>&) const { }
      void reset(void) const { }
      bool overflow(void) const { return true; }
      bool operator == (const ConstPolicy&) const { return true; }
      static int get_start(int) { return 0; }
    };

    ////////////////////////////////////////////////////////////
    //
    //  Periodic policy. Makes use of the underlying periodicty of the
    //  geometry and keeps on increasing.
    //
    //  \date      Tue Apr 16 11:16:46 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>
    struct PeriodicPolicy {
      int curr, max;
      explicit PeriodicPolicy (int m) : curr(0), max(m) { }
      template <int DIM>
      void next(pt::Point<DIM> &p , const pt::Direction<DIM> &mu) {
        curr++;
        p += mu;
      }
      void reset(void) { curr = 0; }
      bool overflow(void) const { return curr >= max; }
      bool operator == (const PeriodicPolicy& other) const {
        return curr == other.curr && max == other.max;
      }
      static int get_start(int) { return 0; }
    };

    ////////////////////////////////////////////////////////////
    //
    //  Bulk policy. A special from of the periodic policy. Will skip
    //  N points when it wraps around.
    //
    //  \date      Tue Apr 16 11:17:23 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>
    template <int N>
    struct BulkPolicy : public PeriodicPolicy {
      explicit BulkPolicy (int m) : PeriodicPolicy(m-N) { };
      template <int DIM>
      void next(pt::Point<DIM> &p , const pt::Direction<DIM> &mu) {
        PeriodicPolicy::next(p, mu);
        if (curr >= max)
          for (int i = 0; i < N; ++i) p += mu;
      }
      static int get_start(int) { return N/2; }
    };

    struct ZeroBoundaryPolicy : public ConstPolicy {
      explicit ZeroBoundaryPolicy (int i) : ConstPolicy(i) { }
    };
    struct TBoundaryPolicy : public ConstPolicy {
      explicit TBoundaryPolicy(int i) : ConstPolicy(i) { }
      static int get_start(const int& m) { return m - 1; }
    };

    ////////////////////////////////////////////////////////////
    //
    //  A pair to be used similar to Alexandrescu's typelist to
    //  collect the policies direciton by direciton. We will make a
    //  list like so:
    //    Pair < FirstItem, Pair < SecondItem,
    //             ... Pair <LastItem , End > ... >
    //
    //  \date      Tue Apr 16 11:18:03 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>
    template <class P, class N>
    struct Pair {
      static const int DIM = N::DIM + 1;
      P policy;
      N next;
      bool good;
      template <class InputIterator>
      explicit Pair (InputIterator L) :
      policy(*(L++)), next(L), good(true) { }
      template <int D>
      static int get_start(int M, const Geometry<D>& g) {
        if (M == DIM-1)
          return P::get_start(g[M]);
        else
          return N::get_start(M, g);
      }
      bool advance(pt::Point<DIM>& p,
                   pt::Direction<DIM> mu =
                   pt::Direction<DIM>(DIM-1)){
        return adv<DIM>(p, mu);
      }
      template <int M>
      bool adv(pt::Point<M>& p,
                pt::Direction<M> mu){
                 policy.template next<M>(p, mu);
        if (policy.overflow()){
          policy.reset();
          good = next.template adv<M>(p, --mu);
        }
        return good;
      }
      bool is_good() const { return good; }
      bool operator == (const Pair& other) const {
        return policy == other.policy && good == other.good
          && next == other.next;
      }
    };

    ////////////////////////////////////////////////////////////
    //
    //  "End" flag for the lists.
    //
    //  \date      Tue Apr 16 11:22:50 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>
    struct End {
      static const int DIM = 0;
      template <class InputIterator>
      explicit End(InputIterator) { }
      template <int N>
      bool adv(pt::Point<N>&,
               pt::Direction<N>&) { return false; }
      bool is_good() { return false; }
      bool operator == (const End& other) const { return true; }
      template <int D>
      static int get_start(int N, const Geometry<D>&) { return 0; }
    };

#define TWO_POLICY_ITER(A, B)                   \
    Pair<A, Pair<B, End> >
#define THREE_POLICY_ITER(A, B, C)                \
    Pair< A, TWO_POLICY_ITER(B, C) >
#define FOUR_POLICY_ITER(A, B, C, D)              \
    Pair< A, THREE_POLICY_ITER(B, C, D) >

    typedef TWO_POLICY_ITER(PeriodicPolicy,
                            PeriodicPolicy) PeriodicTwoDimIter;
    typedef THREE_POLICY_ITER(PeriodicPolicy,
                              PeriodicPolicy,
                              ConstPolicy) PeriodicConstThreeDimIter;
    typedef FOUR_POLICY_ITER(BulkPolicy<2>, PeriodicPolicy,
                             PeriodicPolicy, ConstPolicy) RawZBulkTimeSliceIter;
    ////////////////////////////////////////////////////////////
    //
    //  Template class to make a time silce iterator with arbitrary
    //  number of dimensions.
    //
    //  \date      Tue Apr 16 11:23:10 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>
    template <int D>
    struct TimeSliceIterList {
      typedef Pair<PeriodicPolicy, typename TimeSliceIterList<D-1>::type > type;
    };

    template <>
    struct TimeSliceIterList<1> {
      typedef Pair<ConstPolicy, End> type;
    };

    // Michele:
    template <int D> struct BulkIterList {
      typedef Pair<BulkPolicy<2>, typename TimeSliceIterList<D-1>::type > type;
    };


    ////////////////////////////////////////////////////////////
    //
    //  Iterator for the boundaries in z-direction
    //
    //  \date      Fri Apr 19 11:58:21 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>

    template <int D> struct ZeroBndIterList {
      typedef Pair<ZeroBoundaryPolicy,
                   typename TimeSliceIterList<D-1>::type > type;
    };
    template <int D> struct TBndIterList {
      typedef Pair<TBoundaryPolicy,
                   typename TimeSliceIterList<D-1>::type > type;
    };

    ////////////////////////////////////////////////////////////
    //
    //  Wrapper class containing a point.
    //
    //  \date      Tue Apr 16 11:25:19 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>

    template <int D, class IteratorList>
    class Iterator {
    public:
      typedef typename pt::Point<D> Point;
      Iterator(const Iterator& other) : x(other.x), i(other.i) { }
      Iterator(const Point& n,
               const typename Geometry<D>::extents_t& e) :
        x(n), i(e.rbegin()) { }
      const Point& operator*() const { return x; }
      bool operator == (const Iterator& other) const {
        return x == other.x && i == other.i;
      }
      static int get_start(int N, const Geometry<D>& g){ 
        return IteratorList::get_start(N,g);
      }
      bool operator != (const Iterator& other) const {
        return ! (*this == other);
      }
      Iterator& operator ++() {
        i.advance(x);
        return *this;
      }
      Iterator operator ++(int) {
        Iterator tmp(*this);
        i.advance(x);
        return tmp;
      }
      bool is_good() const { return i.is_good(); }
    private:
      Point x;
      IteratorList i;
    };
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
    explicit Geometry (const extents_t &lattice_extents)
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
    extents_t get_extents() const { return extents; }

    pt::Point<DIM> mk_point(const raw_pt_t &n) const {
      return pt::Point<DIM>(mk_label(n), neighbors.begin());
    }
    raw_pt_t coords(const pt::Point<DIM>& p) const {
      return reverse_label(p);
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
    ///  Checker board scheme bin numbers.
    ///
    ///  We implement a checker board scheme with cube size N. It
    ///  consists of two steps.
    ///  1) Divde the lattice in hypercubes of size N^D. Label these
    ///     alternating (0) and (1).
    ///  2) Within the cubes, number your elements consistently, and
    ///     combine this witht he result from 1) into the final bin
    ///     number. Thus, a point in a cube labeled with (0) and with
    ///     number 8 within this hypercube will end up in bin (0)->8.
    ///  Thus, in total we have 2*N^D bins. In the case N=1 this
    ///  reduces to the usual checkerboard scheme.
    ///
    ///  added by DH, 31st Oct. 2012
    template <int N>
    int bin(const pt::Point<DIM>& p) const {
      raw_pt_t n = coords(p);
      // A) is p in a black (0) or a white (1) block?
      int A = 0;
      for (int i = 0; i < DIM; ++i)
        A = (A + (int(floor(n[i]/N)) % 2)) % 2;
      // B) which position within the block does p have?
      int vol = 1, B = 0;
      for (int i = 0; i < DIM; ++i, vol *= N)
        B += vol*(n[i] % N);
      return A*pow(N, DIM) + B;
    }


    /// get the total number of bins with a checkerboard scheme of
    /// cube size N
    template <int N>
    int n_bins() const{
      return 2*pow(N, DIM);
    }

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
      for (int i = DIM-1; i >= 0; --i){
        label += ((n[i] + extents[i]) % extents[i]) * vol;
        vol *= extents[i];
      }
      return label;
    }
    /// Get the coordinates from a given label
    raw_pt_t reverse_label(const int& n) const {
      int vol = 1;
      raw_pt_t x;
      for (int i = DIM - 1; i >= 0; --i){
        x[i] = (n / vol) % extents[i];
        vol *= extents[i];
      }
      return x;
    }
    /// helper function to recursively fill neighbors
    void fill_neighbors (const int& n = DIM - 1, raw_pt_t x = raw_pt_t()){
      if (!n)
        for (int xi = 0; xi < extents[0]; ++xi){
          x[0] = xi;
          int l = mk_label(x);
          for(pt::Direction<DIM> mu; mu.is_good(); ++mu){
            x[mu]++;
            neighbors[l][DIM + (int)mu] = mk_label(x);
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


  //  A short hand / specialization
  typedef detail::Iterator
  <4, detail::TimeSliceIterList<4>::type > TimeSliceIter;
  typedef detail::Iterator
  <4, detail::BulkIterList<4>::type > BulkIterator;
  typedef detail::Iterator
  <4, detail::ZeroBndIterList<4>::type > ZeroBndIterator;
  typedef detail::Iterator
  <4, detail::TBndIterList<4>::type > TBndIterator;
  // specializations for four dimensions ...

  template <> template <>
  int Geometry<4>::bin<0>(const pt::Point<4>&) const { return 0; }
  template <> template <>
  int Geometry<4>::n_bins<0>() const { return 1; }

  /////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////
  ///
  ///  Checkerboard scheme for safe parallel application of operators
  ///  etc. For the moment, I gave the class two copies of the lists
  ///  of bins, one using c++ list, one using vectors. The list one is
  ///  mainly for ease of construction and for use in the unit
  ///  tests. The other one is to speed up the iteration at run-time
  ///  in LocalField.
  ///
  ///  DH, 7th Nov. 2012
  template <int D, int N, class Iter>
  class CheckerBoard {
  public:
    typedef pt::Point<D> Point; // single point
    typedef std::vector<Point> bin; // one bin
    typedef std::vector<bin> slice; // bins in a time slice
    typedef std::vector<slice> lattice; // all time slices
    explicit CheckerBoard (const Geometry<D>& g) :
      lat(g[0] + 1, slice(g.template n_bins<N>())) {
      typename geometry::Geometry<D>::raw_pt_t n;
      for (int t = 0; t <= g[0]; ++t){
        n[0] = t;
        for (int i = 1; i < D; ++i) n[i] = Iter::get_start(i, g);
        Iter x(g.mk_point(n), g.get_extents());
        do {
          lat[t].at(g.template bin<N>(*x)).push_back(*x);
        } while ((++x).is_good());
      }
    }
    const slice& operator[](const int& i) const {
      return lat[i];
    }
    // as operator[] just for non-const access. this is for testing
    // purposes only and should not be used 'in the wild' use the
    // constant operator above instead.
    slice& nca(const int& i){
      return lat[i];
    }
  private:
    lattice lat;
  };
}

#endif
