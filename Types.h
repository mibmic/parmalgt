#ifndef _TYPES_H
#define _TYPES_H
#include <config.h>
#include <complex>
#include <SUN.hpp>
#include <Vector.hpp>

typedef sun::SU<3> SU3;
typedef sun::SU<3>::data_t Cplx;
typedef sun::Vec<3> CVector;

#ifdef HAVE_STDCXX_0X
#include <array>
typedef std::array<Cplx,3> three_vec_t;
template <class C, int n> struct array_t 
{
  typedef std::array<C,n> Type;
};
#elif HAVE_TR1
#include <tr1/array>
typedef std::tr1::array<Cplx,3> three_vec_t;
template <class C, int n> struct array_t 
{
  typedef std::tr1::array<C,n> Type;
};
#endif

template<int N>
struct Norm : public array_t<double, N>::Type {
  typedef typename array_t<double, N>::Type::iterator iterator;
  typedef typename array_t<double, N>::Type::const_iterator const_iterator;
  iterator begin() { return rep.begin(); };
  iterator end() { return rep.end(); };
  const_iterator begin() const { return rep.begin(); };
  const_iterator end() const { return rep.end(); };

  Norm() { };
  explicit Norm(typename array_t<double, N>::Type other) : rep(other) { };
  Norm& operator+=(Norm& other){
    for (int i = 0; i < rep.size(); ++i) rep[i] += other[i];
    return *this;
  }
  Norm& operator-=(Norm& other){
    for (int i = 0; i < rep.size(); ++i) rep[i] -= other[i];
    return *this;
  }
  double& operator[](const int i) { return rep[i]; }
 private:
  typename array_t<double, N>::Type rep;
};

template <class C, int n>
std::ostream& operator<<(std::ostream& os, const array_t<C,n>& v) {
  typename array_t<C,n>::iterator it = v.begin();
  for(;it!=v.end();++it)
    os << *it << "\t";
  return os;
}
template <int n>
std::ostream& operator<<(std::ostream& os, const Norm<n>& v) {
  typename Norm<n>::iterator it = v.begin();
  for(;it!=v.end();++it)
    os << *it << "\t";
  return os;
}

#endif
