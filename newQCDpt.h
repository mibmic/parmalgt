#ifndef _QCD_PT_H_
#define _QCD_PT_H_

#include <algorithm>
#include <numeric>
#include <MyRand.h>
#include <iostream>
#include <Types.h>
#include <Background.h>
#include <PtTypes.hpp>
#include <limits>

// debug flag

const bool do_debug = true;

// is thrown by log if called with a field with a non-trivial lowest
// order.

class IsNotZeroError : public std::exception { };

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///  This is a simple struct to do the check if a field has a trivial
///  lowest order.
///
///  The point is that
///
///  a) You should not use #defines
///  b) A macro would be a pain anyway, because it must know,
///     depending on the background field class we use, what exactly
///     "non trivial" means. Hence a template is the way to go.
///
///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
///  \date Fri Feb  3 12:52:30 2012


template <bool shall_i_debug, class B> struct IsZero {
  static const bool debug_on = false;
  static void check(const B&) { } // do not debug
};

// overload for shall_i_debug == true

template <class B> struct IsZero<true, B> {
  static const bool debug_on = true;
  static void check(const B& V) {
    static double eps = std::numeric_limits<double>::epsilon() * 10;
    if ( V.Norm() > eps)  {
      std::cout << "->> " << V << std::endl;
      throw IsNotZeroError();
    }
  }
};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///  This is just a stub of the original class (which is of course
///  saved in the git repo). Only the arhitmetic methods are
///  implemented, the rest will follow.
///
///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
///  \date Tue Oct 11 16:23:38 2011



//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///  BGptSU3<B, ORD> represents a series 
///
///       V + g U^(1) + g^2 U^(2) + ... + g^ORD U^(ORD),
///
///  where V is a background field of type B.
///
///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
///  \date Tue Jan 24 16:55:46 2012

template <class B, int ORD>
class BGptSU3{
public:
  // self type
  typedef BGptSU3 self_t;
  // underlying perturbative structure
  typedef typename ptt::PtMatrix<ORD> pt_matrix_t;
  // iterators
  typedef typename pt_matrix_t::iterator iterator;
  typedef typename pt_matrix_t::const_iterator const_iterator;
  // 3x3 matrix type
  typedef typename pt_matrix_t::SU3_t SU3_t;

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Number of doubles needed to store the object.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Sun Mar 25 14:52:42 2012
  static const int storage_size = 
    B::storage_size + pt_matrix_t::storage_size;

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Write everything to a double vector.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Sun Mar 25 14:52:57 2012

  std::vector<double>::iterator &
    buffer(std::vector<double>::iterator & i) const {
    bgf_.buffer(i);
    ptU_.buffer(i);
    return i;
  }

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Write to file.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Fri May 25 16:36:27 2012

  template <class Writer_t>
    void write(Writer_t& o) const {
    bgf_.write(o);
    ptU_.write(o);
  }
  template <class Reader_t>
    void read(Reader_t& i) {
    bgf_.read(i);
    ptU_.read(i);
  }

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Read from buffer
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Mon Mar 26 16:44:30 2012

  std::vector<double>::const_iterator &
  unbuffer(std::vector<double>::const_iterator & i){
    bgf_.unbuffer(i);
    ptU_.unbuffer(i);
    return i;
  }

  explicit BGptSU3(const B& bgf) : bgf_(bgf) { }
  BGptSU3(const B& bgf, const pt_matrix_t& ptU) : 
    bgf_(bgf), ptU_(ptU) { }

  // Default constructor, needed such that self_t may be used in a
  // std::array type.
  BGptSU3() : bgf_(bgf::unit<B>()) { }

  SU3_t& operator[](const int& i) { return ptU_[i]; }
  const SU3_t& operator[](const int& i) const { return ptU_[i]; }

  B& bgf() { return bgf_; }
  const B& bgf() const { return bgf_; }
  
  pt_matrix_t& ptU() { return ptU_; };
  const pt_matrix_t& ptU() const { return ptU_; };

  const_iterator begin() const { return ptU_.begin(); }
  const_iterator end() const { return ptU_.end(); }
  iterator begin() { return ptU_.begin(); }
  iterator end() { return ptU_.end(); }

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Norm
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Tue Apr 24 16:09:19 2012

  std::vector<double> Norm() const {
    std::vector<double> result(ORD + 1, bgf_.Norm());
    for (int i = 0; i < ORD; ++i)
      result[i+1] = ptU_[i].Norm();
    return result;
  }

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Multiplication-Assign.
  ///
  ///  Multiply each order by a given factor.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Wed Jan 11 18:41:26 2012

  template <class C> self_t& operator*=(const C &z) {
    return mul_assign_impl(z, typename ptt::ScalarMultiplyable<C>::type());
  };
  
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Division-Assign. 
  ///
  ///  See operator *=.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Wed Jan 11 18:42:49 2012
  template <class C> self_t& operator/=(const C &z) {
    return div_assign_impl(z, typename ptt::ScalarMultiplyable<C>::type());
  }

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Addition-(Subtraction-)Assign for BGptSU3.
  ///
  ///  Here, we have to perform the operation order by order.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Wed Jan 11 18:44:39 2012
  self_t& operator+=(const self_t &z) {
    ptU_ += z.ptU_;
    bgf_ += z.bgf();
    return *this;
  };
  self_t& operator-=(const self_t &z) {
    ptU_ -= z.ptU_;
    bgf_ -= z.bgf();
    return *this;
  };
  
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Addition-(Subtraction-)Assign for BGptSU3.
  ///
  ///  Here, we have to perform the operation order by order.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Sun Feb 5 2012

  self_t& operator+=(const pt_matrix_t& q){
    ptU_ += q;
    return *this;
  }
  self_t& operator-=(const pt_matrix_t& q){
    ptU_ -= q;
    return *this;
  }


  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Multipy-Assign.
  ///
  ///  This version is specialized for the BGptSU3 type.
  ///
  ///  We implement this by writing
  ///    (V1 + U1)(V2 + U2) = V1*V2 + V1*U2 + U1*V2 + U2*U2
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Wed Jan 11 18:45:50 2012

  self_t& operator*=(const self_t& A) {
    self_t tmp(*this * A);
    std::swap(ptU_, tmp.ptU_);
    std::swap(bgf_, tmp.bgf_);
    return *this;
  }

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Randiomize the matrices for each order, not the backgorund
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Wed Jan 11 18:46:17 2012

  void randomize() {
    static MyRand r(1235431);
    bgf_ = bgf::random();
    for (int i = 0; i < ORD; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k)
          ptU_[i](j, k) = Cplx(r.Rand(), r.Rand());
  };

  void id(){
    std::fill(begin(),end(),SU3());
    bgf_.set_to_one();
  }
  void zero(){
    std::fill(begin(),end(),SU3());
    bgf_.set_to_zero();
  }
  /// Trace
  void Tr(Cplx *tt) const {
    for(int i = 0; i < ORD; i++)
      tt[i+1] = ptU_[i].whr[0] + ptU_[i].whr[4] + ptU_[i].whr[8];
    tt[0] = bgf_.Tr();
  }
  std::vector<Cplx> trace() const {
    std::vector<Cplx> tt(ORD+1);
    for(int i = 0; i < ORD; i++)
      tt[i+1] = ptU_[i].whr[0] + ptU_[i].whr[4] + ptU_[i].whr[8];
    tt[0] = bgf_.Tr();
    return tt;
  }
  /// Make traceless
  void Trless(){
    Cplx z;
    for(int i = 0; i < ORD; i++){
      z = D3*(ptU_[i][0] + ptU_[i][4] + ptU_[i][8]);
      ptU_[i][0] -= z;
      ptU_[i][4] -= z;
      ptU_[i][8] -= z;
    }
    bgf_.Trless();
  }

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  In the old version of the code the operation
  ///
  ///  0.5 [U - U^\dagger]_tr  (the _tr stands for the traceless part)
  ///
  ///  Was assumed to generate an object that lives int the Lie
  ///  algebra. We enforce this here, but do a check to make sure.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Fri Feb  3 16:40:29 2012
  pt_matrix_t reH() const {
    B bg(bgf_);
    bg.reH();
    IsZero<do_debug, B>::check(bg);
    pt_matrix_t U(ptU_);
    return U.reH();
  }

private:
  B bgf_;
  pt_matrix_t ptU_;
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Implementations of *= and /= for scalar types
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Fri Feb 17 12:39:55 2012
  
  template <class C> self_t& div_assign_impl(const C &z, ptt::True) {
    ptU_ /= z;
    bgf_ /= z;
    return *this;
  }
  template <class C> self_t& mul_assign_impl(const C& z, ptt::True){
    ptU_ *= z;
    bgf_ *= z;
    return *this;
  }
};

template <class B, int ORD>
std::ostream& operator<<(std::ostream& os, const BGptSU3<B, ORD> &U){
  os << "Trivial Order:\n";
  os << U.bgf() << "\n";
  os << U.ptU();
  return os;
}

template <class B, int ORD>
inline BGptSU3<B, ORD> 
operator*(const BGptSU3<B, ORD>& U, const B& bg){
  return BGptSU3<B, ORD> (U.bgf() * bg, U.ptU() * bg);
}

template <class B, int ORD>
inline BGptSU3<B, ORD> 
operator*(const B& bg, const BGptSU3<B, ORD>& U){
  return BGptSU3<B, ORD> (bg * U.bgf(), bg * U.ptU());
}

template <class BG, int ORD>
inline BGptSU3<BG, ORD> 
operator*(const BGptSU3<BG, ORD>& A, const BGptSU3<BG, ORD>& B){
  return BGptSU3<BG, ORD> (A.bgf() * B.bgf(),
                          A.ptU() * B.bgf()
                          + A.bgf() * B.ptU()
                          + A.ptU() * B.ptU());
}

template <class C, class BG, int ORD>
inline BGptSU3<BG, ORD> 
operator*(const BGptSU3<BG, ORD>& A, const C& b){
  return BGptSU3<BG, ORD> (A) *= b;
}


template <class C, class BG, int ORD>
inline BGptSU3<BG, ORD> 
operator/(const BGptSU3<BG, ORD>& A, const C& b){
  return BGptSU3<BG, ORD> (A) /= b;
}

template <class C, class BG, int ORD>
inline BGptSU3<BG, ORD> 
operator*(const C& b, const BGptSU3<BG, ORD>& A ){
  return BGptSU3<BG, ORD> (A) *= b;
}

template <class C, class BG, int ORD>
inline BGptSU3<BG, ORD> 
operator+(const C& b, const BGptSU3<BG, ORD>& A ){
  return BGptSU3<BG, ORD> (A) += b;
}

template <class C, class BG, int ORD>
inline BGptSU3<BG, ORD> 
operator-(const C& b, const BGptSU3<BG, ORD>& A ){
  return BGptSU3<BG, ORD> (A) -= b;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///  Dagger (hermitian conjugate) for ptSU3
///
///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
///  \date Thu Jan 26 11:47:34 2012

template <class B, int ORD>
inline BGptSU3<B, ORD> 
dag( const BGptSU3<B, ORD>& U ){
  // The constructor call copies everyting
  BGptSU3<B, ORD> res(U);
  for (int i = 0; i < ORD; ++i){
    // transpose
    std::swap(res[i][1], res[i][3]);
    std::swap(res[i][2], res[i][6]);
    std::swap(res[i][5], res[i][7]);
    // conjugate
    for (int j = 0; j < 9; ++j) res[i][j].im = -res[i][j].im;
  }
  res.bgf() = U.bgf().dag();
  return res;
};

/// Inversion via perturbative series

template <class B, int ORD>
inline BGptSU3<B, ORD>
inverse( const BGptSU3<B, ORD>& W ){
  // assert that the series defined by W starts with one
  //IsZero<do_debug, B>::check(W.bgf() - bgf::unit<B>());
  BGptSU3<B, ORD> Z;

  for ( int r = 0; r < ORD; ++r){
    Z[r] = -W[r];
    for ( int s = 1; s < r+1; ++s)
      Z[r] -= Z[s-1] * W[r-s];
  }

  return Z;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///  Exponential
///
///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
///  \date Tue Jan 24 16:56:39 2012

//template <class B, int AL_ORD, int PT_ORD>
//BGptSU3<B, AL_ORD, PT_ORD> 
//exp( const BGptSU3<B, AL_ORD, PT_ORD>& );

template <class B, int ORD>
inline BGptSU3<B, ORD> exp(const ptt::PtMatrix<ORD>& q){
  BGptSU3<B, ORD> result(bgf::unit<B>(), q);
  // result = 1 + q 
  ptt::PtMatrix<ORD> tmp(q);
  for ( int i = 2; i <= ORD; ++i){
    tmp *= q;
    tmp /= i;
    result.ptU() += tmp;
  }
  return result;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///  Natural Logarithm  -- Using the Mercator series
///
///
///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
///  \date Tue Jan 24 16:56:55 2012

template <class B, int ORD>
inline ptt::PtMatrix<ORD> log(const BGptSU3<B, ORD>& U){
  IsZero<do_debug, B>::check(U.bgf() - bgf::unit<B>());
  ptt::PtMatrix<ORD> result(U.ptU()), tmp(U.ptU());
  double sign = 1.;
  for (int i = 2; i <= ORD; ++i){
    sign *= -1;
    tmp *= U.ptU();
    result += tmp*(sign/i); 
    // FIXME stretch_assign would be good here!
  }
  return result;
};

template <int ORD>
inline ptt::PtMatrix<ORD> log(const ptt::PtMatrix<ORD>& U){
  ptt::PtMatrix<ORD> result(U), tmp(U);
  double sign = 1.;
  for (int i = 2; i <= ORD; ++i){
    sign *= -1;
    tmp *= U;
    result += tmp*(sign/i); 
    // FIXME stretch_assign would be good here!
  }
  return result;
};

template <class B, int ORD>
inline ptt::PtMatrix<ORD> get_q(const BGptSU3<B, ORD>& U){
  ptt::PtMatrix<ORD> result;
  B Vinv = U.bgf().inverse();
  for (int i = 0; i < ORD; ++i)
    result[i] = Vinv.ApplyFromRight(U[i]);
  ptt::PtMatrix<ORD> tmp(result), Util(result);
  double sign = 1.;
  for (int i = 2; i <= ORD; ++i){
    sign *= -1;
    tmp *= Util;
    result += tmp*(sign/i); 
    // FIXME stretch_assign would be good here!
  }
  return result;
};


template <class BGF,  int ORD, int DIM>
class BGptGluon {
public:
  typedef BGptSU3<BGF, ORD> pt_su3_t;
  typedef typename array_t<pt_su3_t, DIM>::Type array_t;
  typedef BGptGluon self_t;
  typedef typename array_t::iterator iterator;
  typedef typename array_t::const_iterator const_iterator;

  static const int storage_size = DIM*pt_su3_t::storage_size + BGF::storage_size;
 
  // access
  pt_su3_t& operator[](const int& i){ return U_[i]; }
  const pt_su3_t& operator[](const int& i) const { return U_[i]; }
  // iterators
  iterator begin(){return U_.begin();}
  const_iterator begin() const {return U_.begin();}
  iterator end(){return U_.end();}
  const_iterator end() const {return U_.end();}

  // norm
  std::vector<double> Norm() const {
    std::vector<double> norm(ORD + 1); 
    for (const_iterator i = begin(), e = end(); i != e; ++i){
      for (int j = 0; j <= ORD; ++j)
        norm[j] += i->Norm()[j] * i->Norm()[j];
    }
    for (int j = 0; j <= ORD; ++j)
      norm[j] = sqrt(norm[j]);
    return norm;
  }


  // buffer
  std::vector<double>::iterator&
  buffer (  std::vector<double>::iterator& j ) const {
    for (const_iterator i = begin(); i != end(); ++i)
      i->buffer(j);
    return j;
  }

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Write to file.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Fri May 25 16:39:34 2012

  template <class Writer_t>
  void write(Writer_t& o) const {
    for (const_iterator i = begin(); i != end(); ++i)
      i->write(o);
  }

  // unbuffer
  std::vector<double>::const_iterator&
  unbuffer (  std::vector<double>::const_iterator& j ) {
    for (iterator i = begin(); i != end(); ++i)
      i->unbuffer(j);
    return j;
  }


  void randomize() {
    for (iterator i = begin(); i != end(); ++i)
      i->randomize();
  }

  template <class C>
  self_t& operator*=(const C& other){
    std::for_each(begin(), end(), pta::mul(other));
    return *this;
  }

  template <class C>
  self_t& operator/=(const C& other){
    std::for_each(begin(), end(), pta::div(other));
     return *this;
  }

  pt_su3_t operator*(const self_t& other) const{
    return std::inner_product(begin(), end(),
                              other.begin(), pt_su3_t());
  }
  
private:
  array_t U_;
};

template  <int ORD>
class BGptCVector {
public:
  typedef typename array_t<CVector, ORD + 1>::Type vec_t;
  typedef BGptCVector self_t;
  typedef typename vec_t::iterator iterator;
  typedef typename vec_t::const_iterator const_iterator;
  // template tyedef for multiplicaion wit BGptSU3
  template <class BGF> struct pt_su3_t {
    typedef BGptSU3<BGF, ORD> Type;
  };

  /// Iterators
  iterator begin() { return v_.begin(); }
  iterator end() { return v_.end(); }
  const_iterator begin() const { return v_.begin(); }
  const_iterator end() const { return v_.end(); }

  // buffer
  std::vector<double>::iterator&
  buffer (  std::vector<double>::iterator& j ) const {
    for (const_iterator i = begin(); i != end(); ++i)
      i->buffer(j);
    return j;
  }
  // unbuffer
  std::vector<double>::const_iterator&
  unbuffer (  std::vector<double>::const_iterator& j ) {
    for (iterator i = begin(); i != end(); ++i)
      i->unbuffer(j);
    return j;
  }

  /// Access operators
  CVector& operator[](const int& i){ return v_[i]; }
  const CVector& operator[](const int& i) const { return v_[i]; }

  /// Arithmetic
  self_t& operator+=(const self_t& other){
    std::for_each(begin(), end(), pta::incr(other));
    return *this;
  }
  self_t& operator-=(const self_t& other){
    std::for_each(begin(), end(), pta::decr(other));
    return *this;
  }

  self_t operator+(const self_t& other) const{
    self_t result(*this);
    return result += other;
  }
  self_t operator-(const self_t& other) const{
    self_t result(*this);
    return result -= other;
  }
  template <class C>
  self_t& operator*=(const C& alpha) {
    std::for_each(begin(), end(), pta::mul(alpha));
    return *this;
  }
  template <class C>
  self_t& operator/=(const C& alpha) {
    std::for_each(begin(), end(), pta::div(alpha));
    return *this;
  }
  template <class C>
  self_t operator*(const C& alpha) const {
    self_t result;
    return result *= alpha;
  }
  template <class C>
  self_t operator/(const C& alpha) const {
    self_t result;
    return result /= alpha;
  }  
  template <class BGF>
  self_t operator*(const BGptSU3<BGF, ORD>& other) const{
    self_t result;
    // Handle products that involve pert. orders > 1
    // note that other[i] is of perturbative order i+1!
    for (int i = 1; i < ORD + 1; ++i)
      for (int j = 0; j < i; ++j)
        result[i] += v_[j] * other[i - j - 1];
    
    // Handle the products involving other.bgf
    for (int i = 0; i < ORD + 1; ++i)
      result[i] += other.bgf().ApplyFromLeft(v_[i]);
    return result;
  }

  bool operator==(const self_t& other) const {
    for (const_iterator i = begin(), j = other.begin();
         i != end(); ++i, ++j)
      if (*i != *j)
        return false;
    return true;
  }
private:
  vec_t v_;
};

template  <int ORD, int DIM>
class BGptSpinColor {
public:
  typedef BGptCVector<ORD> pt_vec_t;
  typedef typename array_t<pt_vec_t, DIM>::Type vec_t;
  typedef typename vec_t::iterator iterator;
  typedef typename vec_t::const_iterator const_iterator;
  typedef BGptSpinColor self_t;
  
  // access
  pt_vec_t& operator[](const int& i){ return psi_[i]; }
  const pt_vec_t& operator[](const int& i) const { return psi_[i]; }
  
  // iterators
  iterator begin() { return psi_.begin(); }
  iterator end() { return psi_.end(); }
  const_iterator begin() const { return psi_.begin(); }
  const_iterator end() const { return psi_.end(); }

  // arithmetic with self
  self_t& operator+=(const self_t& other){
    std::for_each(begin(), end(), pta::incr(other));
    return *this;
  }
  self_t& operator-=(const self_t& other){
    std::for_each(begin(), end(), pta::decr(other));
    return *this;
  }
  self_t operator+(const self_t& other) const{
    self_t result(*this);
    return result += other;
  }
  self_t operator-(const self_t& other) const{
    self_t result(*this);
    return result -= other;
  }
  // scalar arithmetic
  template <class C>
  self_t& operator*=(const C& alpha) {
    std::for_each(begin(), end(), pta::mul(alpha));
    return *this;
  }
  template <class C>
  self_t& operator/=(const C& alpha) {
    std::for_each(begin(), end(), pta::div(alpha));
    return *this;
  }
  template <class C>
  self_t operator*(const C& alpha) const {
    self_t result;
    return result *= alpha;
  }
  template <class C>
  self_t operator/(const C& alpha) const {
    self_t result;
    return result /= alpha;
  }
  template <class BGF>
  self_t operator*(const BGptGluon<BGF, ORD, DIM>& U) 
    const {
    self_t result;
    // outer iteration 0 ... DIM
    const_iterator this_mu = begin();
    iterator psi_mu = result.begin();
    typename BGptGluon<BGF, ORD, DIM>
      ::const_iterator U_mu = U.begin();
    for(; this_mu != end(); ++this_mu, ++psi_mu, ++U_mu)
      *psi_mu = *this_mu* *U_mu;
    return result;
  }
  //void uno_p_gmu(SpinColor&, int, int);
  //void uno_m_gmu(SpinColor&, int, int);

private:
  vec_t psi_;
  
};

#endif
