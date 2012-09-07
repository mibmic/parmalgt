#ifndef POINT_H
#define POINT_H

#include <vector>
#include <Types.h>


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///  D-Dimensional lattice points and directions.
///
///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
///  \date Mon Mar 26 14:51:20 2012
namespace pt {


  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Direction class.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Mon Mar 26 14:51:36 2012
  template <int DIM>
  class Direction {
  public:
    explicit Direction(const int& m = 0) : mu(m) { }
    Direction& operator++() { ++mu; return *this; }
    bool is_good() const { return mu < DIM; }
    operator int() const { return mu; }
    template <typename A, typename B>
    A deref_fwd(B b) const { return b[DIM + mu]; }
    template <typename A, typename B>
    A deref_bkw(B b) const { return b[mu]; }
    static const Direction t;
    static const Direction x;
    static const Direction y;
    static const Direction z;
  private:
    int mu;
  };


  template <int DIM>
  inline Direction<DIM> operator+(const Direction<DIM>& d, const int& i){
    Direction<DIM> result(i + d);
    return result;
  };

  template <int DIM>
  class Point {
  public:
    typedef typename array_t<int, 2*DIM>::Type arr_t;
    typedef typename std::vector<arr_t> vec_t;
    typedef typename vec_t::const_iterator iter_t;
    Point(int nn, const iter_t& i) : n(nn), L_begin(i) { }
    Point& operator+=(const Direction<DIM>& mu){
      n = mu.template deref_fwd<const int &, 
                                const arr_t &>(*(L_begin +n));
      return *this;
    }
    Point& operator-=(const Direction<DIM>& mu){
      n = mu.template deref_bkw<const int&,
                                const arr_t&>(*(L_begin +n));
      return *this;
    }
    Point& operator++() {
      n++;
      return *this;
    }
    operator int() const {return n;}
    template <typename T>
    const T& deref(T const * const f) const {
      return f[n];
    }
    template <typename T>
    T& deref(T* const f) const {
      return f[n];
    }
    template <typename T>
    T& deref(std::vector<T>& v) const { return v[n];  }
    template <typename T>
    const T& deref(const std::vector<T>& v) const { return v[n];  }

    bool operator==(const Point& other){
      return n == other.n && L_begin == other.L_begin;
    }
    bool operator!=(const Point& other){
      return !(*this == other);
    }
  private:
    int n; // the site
    iter_t L_begin; // positions
  };
  
  template <int DIM>
  inline Point<DIM> operator+(const Point<DIM>& p, const Direction<DIM>& mu){
    return Point<DIM>(p) += mu;
  }
  
  template <int DIM>
  inline Point<DIM> operator-(const Point<DIM>& p, const Direction<DIM>& mu){
    return Point<DIM>(p) -= mu;
  }

} // namespace pt

#endif //ifndef POINT_H
