#ifndef _NEW_MY_QCD_H_
#define _NEW_MY_QCD_H_

#include <algorithm>
#include <numeric>
#include <vector>

#include <iostream>
#include <Types.h>
#include <limits>


namespace sun {
  template <int N> complex operator^( const Vec<N>& A, const Vec<N>& B){
    complex result;
    for (int i = 0; i < N; ++i) result += A[i]*conj(B[i]);
    return result;
  }
}

template <int DIM>
  class SpinColor {
 public:
  typedef CVector data_t;
  typedef typename array_t< CVector, DIM>::Type array_t;
  typedef SpinColor self_t;
  typedef typename array_t::iterator iterator;
  typedef typename array_t::const_iterator const_iterator;
  
  static const int storage_size = DIM*CVector::rep_size;
  
  SpinColor(const self_t& other) : psi_(other.psi_) { };
  SpinColor() {}

  // access
  CVector& operator[](const int& i){ return psi_[i]; }
  const CVector& operator[](const int& i) const { return psi_[i]; }
  // iterators
  iterator begin(){return psi_.begin();}
  const_iterator begin() const {return psi_.begin();}
  iterator end(){return psi_.end();}
  const_iterator end() const {return psi_.end();}

  // norm
  double Norm() const {
    double norm;
    for (const_iterator i = begin(), e = end(); i != e; ++i){
      norm += i->Norm() * i->Norm();
    }
    return std::sqrt(norm);
  }


  // buffer
  std::vector<double>::iterator&
  buffer (  std::vector<double>::iterator& j ) const {
/*     for (const_iterator i = begin(); i != end(); ++i) */
/*       i->buffer(j); */
/*     return j; */

  }

  template <class Writer_t>
  void write(Writer_t& o) const {
/*     for (const_iterator i = begin(); i != end(); ++i) */
/*       i->write(o); */
  }

  // unbuffer
  std::vector<double>::const_iterator&
  unbuffer (  std::vector<double>::const_iterator& j ) {
/*     for (iterator i = begin(); i != end(); ++i) */
/*       i->unbuffer(j); */
/*     return j; */
  }


  void randomize() {
    for (iterator i = begin(); i != end(); ++i)
      i->randomize();
  }
  
  self_t& operator+=(const self_t& other) {
    for (int i = 0; i < psi_.size(); ++i) psi_[i] += other[i];
    return *this;
  }

  self_t& operator-=(const self_t& other) {
    for (int i = 0; i < psi_.size(); ++i) psi_[i] -= other[i];
    return *this;
  }

  template <class C>
  self_t& operator*=(const C& other){
    for (iterator i = begin(), e = end(); i != e; ++i) (*i) *= other;
    return *this;
  }

  template <class C>
  self_t& operator/=(const C& other){
    for (iterator i = begin(), e = end(); i != e; ++i) (*i) /= other;
    return *this;
  }


  self_t operator+(const self_t& other){
    self_t result(*this);
    for (int i = 0; i < psi_.size(); ++i) result[i] += other[i];
    return result;
  }

  self_t operator-(const self_t& other){
    self_t result(*this);
    for (int i = 0; i < psi_.size(); ++i) result[i] -= other[i];
    return result;
  }


  /////////////////////////////////////////////////
  /////////////////////////////////////////////////
  ///  Inner product between two spin color
  ///
  ///  \author Michele Brambilla <mib.mic@gmail.com>
  ///  \date Wed Jan 09 12:56:58 2013
  complex operator*(const self_t& other) const{
    complex result;
    for (int i = 0; i < psi_.size(); ++i)
      for (int j = 0; j < 3; ++j)
        result += psi_[i][j] * other[i][j];
    return result;
  }

  /////////////////////////////////////////////////
  /////////////////////////////////////////////////
  /// Product between two spin color
  ///
  ///  \author Michele Brambilla <mib.mic@gmail.com>
  ///  \date Wed Jan 09 12:58:12 2013
  complex operator^(const self_t& other) const{
    complex result;
    /* for (int i = 0; i < psi_.size(); ++i) result += conj(psi_[i]) * other[i]; */
    for (int i = 0; i < psi_.size(); ++i) result += (psi_[i] ^ other[i]);
    return result;
  }

  template<class C>
    self_t operator*(const C& other) const{
    self_t result(*this);
    // for (iterator i = begin(), e = end(), i1 = result.begin(); i != e; ++i, ++i1) (*i1) = (*i)*other;
    return (result *= other);
  }

  template<class C>
    self_t operator/(const C& other){
    self_t result(*this);
    // for (iterator i = result.begin(), e = result.end(); i != e; ++i) i /= other;
    return (result /= other);
  }

private:
  array_t psi_;
};

// namespace helper {
//   CVector mul(const SU3& A, const CVector& v) {
//     return std::move(A*v);
//   }
// } // helper


template<int DIM>
inline SpinColor<DIM> operator*(const SU3 U, const SpinColor<DIM> psi){
  typedef typename SpinColor<DIM>::const_iterator c_it;
  typedef typename SpinColor<DIM>::iterator       v_it;
  SpinColor<DIM> result;
  v_it d = result.begin();
  for( c_it s = psi.begin(); s != psi.end(); ++s, ++d)  *d = sun::detail::mul(U,(*s));
  /* for(int i=0;i<DIM;++i) result[i] = U*psi[i]; */
  return result;
}

template<int DIM>
inline std::ostream& operator<<(std::ostream& os, const SpinColor<DIM>& f){
  // for( typename SpinColor<DIM>::const_iterator v_it = f.begin(); v_it != f.end(); ++v_it)
  //   os << (*v_it) << "\n";
  // os << std::endl;
  std::for_each(f.begin(), f.end(), [&](const sun::Vec<3>& i) { os << i; });
  os << std::endl;
  return os;
}

#endif
