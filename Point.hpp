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
  
  template<int DIM,typename T>
  struct MultiDir;

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
    Direction& operator--() { --mu; return *this; }
    bool is_good() const { return mu < DIM; }
    operator int() const { return mu; }
    template <typename A, typename B>
    A deref_fwd(B b) const { return b[DIM + mu]; }
    template <typename A, typename B>
    A deref_bkw(B b) const { return b[mu]; }
    Direction operator -() const { return Direction( (mu + DIM) % (2*DIM)); }
    bool operator<(const int& o) const { return mu < o; }
    bool operator>(const int& o) const { return mu > o; }
    static const Direction t;
    static const Direction x;
    static const Direction y;
    static const Direction z;

    ///////////////////////////////////
    ///////////////////////////////////
    ///  This permits a sintax like "Point + Direction * n", meaning
    ///  "starting from point Point perform n step in direction
    ///  Direction" and return the arriving point
    ///
    ///  \date Fri Mar 21 16:17:14 2014
    ///  \author Michele Brambilla <mib.mic@gmail.com>
    MultiDir<DIM,Direction> operator*(int n) { return MultiDir<DIM,Direction>(n,*this); }
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
    Point(int nn, const iter_t& i) : n(nn), L_begin(i) {  }
    Point& operator+=(const Direction<DIM>& mu){
      if (mu >= DIM) return *this -= Direction<DIM>(mu % DIM);
      n = mu.template deref_fwd<const int &, 
                                const arr_t &>(*(L_begin +n));
      return *this;
    }
    Point& operator-=(const Direction<DIM>& mu){
      if (mu >= DIM) return *this += Direction<DIM>(mu % DIM);
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
    typename T::const_reference deref(const T& v) const { return v[n]; }
    template <typename T>
    typename T::reference deref(T& v) const { return v[n]; }
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

  //////////////////////////////////////
  //////////////////////////////////////
  /// Hack to enable multi-step in direction \mu
  /// 
  ///  \date Fri Mar 21 11:56:27 2014
  ///  \author Michele Brambilla <mib.mic@gmail.com>
  template<int DIM, typename T=pt::Direction<DIM> >
  struct MultiDir {
    MultiDir(int nn, const T& m) : n_(nn), mu_(m) { };
    int& n() { return n_; }
    const T& mu() { return mu_; }
  private:
    const T mu_;
    int n_;
  };
  template<typename A, typename B, int DIM>
  inline A operator+(A p, MultiDir<DIM,B> md) {
    do {
      p += md.mu();
    } while (--md.n()>0);
    return p; }
  template<typename A, typename B, int DIM>
  inline A operator-(A p, MultiDir<DIM,B> md) {
    do {
      p -= md.mu();
    } while (--md.n()>0);
    return p;
  }


} // namespace pt

#endif //ifndef POINT_H
