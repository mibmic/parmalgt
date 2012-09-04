#ifndef _BACKGROUND_H_
#define _BACKGROUND_H_

#include "MyMath.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <math.h>
#include <Types.h>
#include <MyRand.h>

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/// Background field classes.
///
/// Classes to implement trivial and abelian background fields.

namespace bgf {

  template <class C> inline SU3 operator*(const C& x, const SU3& y){
    return x.ApplyFromLeft(y);
  }
  template <class C> inline SU3 operator*( const SU3& y, const C& x){
    return x.ApplyFromRight(y);
  }
  
  /// Base class to define interface.
  class BgfBase {
  public:
    /// Left multiplication.
    virtual SU3 ApplyFromLeft ( const SU3 & ) const = 0;
    virtual CVector ApplyFromLeft ( const CVector& ) const = 0;
    /// Right multiplication.
    virtual SU3 ApplyFromRight ( const SU3 & ) const = 0;
    virtual CVector ApplyFromRight ( const CVector & ) const = 0;
    // set to zero, or one
    virtual void set_to_one () = 0;
    virtual void set_to_zero () = 0;
    // Trace
    virtual Cplx Tr() const = 0;
    virtual double Norm() const  = 0;
    // Make traceless
    virtual void Trless() = 0;
    virtual void reH() = 0;
  };

  /// Trivial (unit) background field.
  class TrivialBgf : public BgfBase {
  public:
    virtual SU3 ApplyFromLeft ( const SU3 & U) const {
      return U;
    }
    virtual SU3 ApplyFromRight ( const SU3 & U) const {
      return U;
    }
    virtual CVector ApplyFromLeft ( const CVector & U) const {
      return U;
    }
    virtual CVector ApplyFromRight ( const CVector & U) const {
      return U;
    }
    template <class C> TrivialBgf & operator*= (const C&) {
      return *this;
    }
    template <class C> TrivialBgf operator* (const C&) const {
      return *this;
    }
    TrivialBgf inverse() const { return *this; }
    void set_to_one() { }
    void set_to_zero() { throw std::exception(); } // not possible ...
    TrivialBgf dag() const { return *this; }
    Cplx Tr() const { return 3; }
    double Norm() const { return 1; }
    void Trless() {}
    void reH() {}
  };

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Abelian \f$SU(3)\f$ background field.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Tue Sep 27 11:08:30 2011
  class AbelianBgf : public BgfBase {
  public:

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Vector constructor.
    ///
    ///  Construct from a three_vec_t as the diagonal elements.
    ///
    ///  \param v Vector we want to use for initialzing.
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Tue Sep 27 11:07:14 2011
    explicit AbelianBgf(const three_vec_t &v) : v_(v){ }
    AbelianBgf() : v_(){ std::fill(v_.begin(), v_.end(), 1); }
    Cplx & operator[](const short& s){
      return v_[s];
    }
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Iterators
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Fri Jan 27 13:06:31 2012

    typedef three_vec_t::iterator iterator;
    typedef three_vec_t::const_iterator const_iterator;
    iterator begin() { return v_.begin(); }
    iterator end() { return v_.end(); }
    const_iterator begin() const { return v_.begin(); }
    const_iterator end() const { return v_.end(); }

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Left multiplication with an \f$SU(3)\f$ matrix.
    ///
    ///  This returns the product \f$ V U \f$, where \f$ V \f$ is the
    ///  background field's value represented by the instance.
    ///
    ///  \param U The \$f SU3 \$f matrix the instance is to be applied
    ///  to. 
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Tue Sep 27 11:03:50 2011
    virtual SU3 ApplyFromLeft ( const SU3 & U) const {
      SU3 result;
      for (int i = 0; i < 3; ++i)
	for (int j = 0; j < 3; ++j)
	  result.whr[3*i + j] = v_[i] * U.whr[3*i + j];
      return result;
    }
    virtual CVector ApplyFromLeft ( const CVector & v) const {
      CVector result;
      for (int i = 0; i < 3; ++i)
        result.whr[i] = v_[i] * v.whr[i];
      return result;
    }

    double Norm() const {
      double result = 0;
#ifdef HAVE_CXX0X
      for (const auto& e : v_) { result += e.re*e.re + e.im*e.im; }
#else
      for (int i = 0; i < 3; ++i)
        result += v_[i].re*v_[i].re + v_[i].im*v_[i].im;
#endif
      return std::sqrt(result);
    }

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Right multiplication with an \f$SU(3)\f$ matrix.
    ///
    ///  This returns the product \f$ U V \f$, where \f$ V \f$ is the
    ///  background field's value represented by the instance.
    ///
    ///  \param U The \$f SU3 \$f matrix the instance is to be applied
    ///  to. 
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Tue Sep 27 11:03:50 2011
    virtual SU3 ApplyFromRight ( const SU3 & U) const {
      SU3 result;
      for (int i = 0; i < 3; ++i)
	for (int j = 0; j < 3; ++j)
	  result.whr[3*i + j] = v_[j] * U.whr[3*i + j];
      return result;
    }
    virtual CVector ApplyFromRight ( const CVector & v) const {
      CVector result;
      for (int i = 0; i < 3; ++i)
        result.whr[i] = v_[i] * v.whr[i];
      return result;
    }
    bool operator==(const AbelianBgf& other) const{
      return v_ == other.v_;
    }
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Generic *= operator template.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Mon Sep 26 18:28:23 2011
    template<class C>
    AbelianBgf& operator*= ( const C& alpha ) {
      std::transform ( v_.begin(), v_.end(), v_.begin(),
		       std::bind1st( std::multiplies<Cplx>(), alpha));
      return *this;
    }
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Generic /= operator template.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Mon Sep 26 18:28:23 2011
    template<class C>
    AbelianBgf& operator/= ( const C& alpha ) {
      *this *= 1./alpha;
      return *this;
    }
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Generic multiplication.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Mon Sep 26 18:28:36 2011
    template<class C>
    AbelianBgf operator* (const C& alpha ) const {
      AbelianBgf result(*this);
      return result *= alpha;
    }
    
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Addition and subtraction of a sclar.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Wed Jan 11 18:03:06 2012
    template<class C>
    AbelianBgf& operator+= (const C& alpha ){
      std::transform ( v_.begin(), v_.end(), v_.begin(),
		       std::bind1st( std::plus<C>(), alpha));
      return *this;
    }
    AbelianBgf& operator+= (const AbelianBgf& alpha ){
      std::transform (v_.begin(), v_.end(), alpha.v_.begin(),
                      v_.begin(), std::plus<Cplx>());
      return *this;
    }
    AbelianBgf operator-() const{
      AbelianBgf result;
      for (int i = 0; i < 3; ++i) result[i] = -v_[i];
      return result;
    }
    template<class C>
    AbelianBgf& operator-= (const C& alpha ){
      return *this += (-alpha);
    }
    template<class C>
    AbelianBgf operator+ (const C& alpha ) const {
      AbelianBgf result(*this);
      return result += alpha;
    }
    template<class C>
    AbelianBgf operator- (const C& alpha ) const {
      AbelianBgf result(*this);
      return result -= alpha;
    }
    AbelianBgf inverse() const {
      AbelianBgf result;
      for (int i = 0; i < 3; ++i) result[i] = 1./v_[i];
      return result;
    }
    void set_to_one() { std::fill(v_.begin(), v_.end(), 1); }
    void set_to_zero() { std::fill(v_.begin(), v_.end(), 0); }
    /// Trace
    Cplx Tr() const {
      return std::accumulate(v_.begin(), v_.end(), Cplx(0));
    }

    /// Make traceless
    virtual void Trless() {
      Cplx Tro3 = Tr()/3.;
#ifdef HAVE_CXX0X
      for ( auto& e: v_ ) e -= Tro3;
#else
      for (int i = 0; i < 3; ++i) v_[i] -= Tro3;
#endif
    }

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  reH
    ///
    ///  This takes the traceless part of 0.5*[V - V^\dagger]
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Fri Feb  3 16:23:25 2012
    
    virtual void reH() {
#ifdef HAVE_CXX0X
      for ( auto& e: v_ ) e.re = 0;
#else
      for (int i = 0; i < 3; ++i) v_[i].re = 0;
#endif
      Trless();
    }

    AbelianBgf dag() const {
      AbelianBgf result(*this);
      for (int i = 0; i < 3; ++i) result[i].im = -result[i].im;
      return result;
    }

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Number of doubles needed to store the object.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Sun Mar 25 14:50:37 2012

    static const int storage_size = 6;

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Buffer to a vector of doubles.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Sun Mar 25 14:50:53 2012

    std::vector<double>::iterator &
    buffer(std::vector<double>::iterator & i) const {
      for (const_iterator n = begin(); n!= end(); ++n){
        *i = n->re; ++i;
        *i = n->im; ++i;
      }
      return i;
    }
  
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Write to a file
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Fri May 25 16:02:33 2012

    template <class Writer_t>
    void write(Writer_t& o) const {
      for (const_iterator n = begin(); n!= end(); ++n)
        o.write(*n);
    }
    template <class Reader_t>
    void read(Reader_t& o) {
      for (iterator n = begin(); n!= end(); ++n)
        o.read(*n);
    }

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Read from a buffer
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Mon Mar 26 16:43:22 2012

    std::vector<double>::const_iterator &
    unbuffer(std::vector<double>::const_iterator & i){
      for (iterator n = begin(); n!= end(); ++n){
        n->re = *i; ++i;
        n->im = *i; ++i;
      }
      return i;
    }

  private:
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Diagonal elements of V.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Tue Sep 27 10:50:29 2011
    three_vec_t v_;
  };

  inline AbelianBgf dag(const AbelianBgf& b){
    return b.dag();
  }

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Specialization of *= and * operator for AbelianBgf type.
  ///
  ///  The Arithmetics are quite straight forward, altough the
  ///  AbelianBgf*AbelianBgf multiplication is a special case.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Mon Sep 26 18:25:58 2011

  template<> inline AbelianBgf& AbelianBgf::operator*= 
  ( const AbelianBgf& other ){
    std::transform  (v_.begin(), v_.end(), other.v_.begin(),
		     v_.begin(), std::multiplies<Cplx>() );
    return *this;
  }

  inline std::ostream& operator<<(std::ostream& os, const AbelianBgf& b){
    os << "{ ";
    for (AbelianBgf::const_iterator i = b.begin();
           i != b.end(); ++i)
      os << "(" << i->re << ", " << i->im << ") ";
    os << "}";
    return os;
  }

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Scalar \f$SU(3)\f$ background field.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Tue Sep 27 11:08:30 2011
  class ScalarBgf : public BgfBase {
  public:

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Vector constructor.
    ///
    ///  Construct from a three_vec_t as the diagonal elements.
    ///
    ///  \param v Vector we want to use for initialzing.
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Tue Sep 27 11:07:14 2011
    explicit ScalarBgf(const Cplx &a) : a_(a){ }
    ScalarBgf() : a_(1, 0) { }

    virtual SU3 ApplyFromLeft ( const SU3 & U) const {
      SU3 result;
      for (int i = 0; i < 3; ++i)
	for (int j = 0; j < 3; ++j)
	  result.whr[3*i + j] = a_ * U.whr[3*i + j];
      return result;
    }
    virtual CVector ApplyFromLeft ( const CVector & v) const {
      CVector result;
      for (int i = 0; i < 3; ++i)
        result.whr[i] = a_ * v.whr[i];
      return result;
    }

    double Norm() const {
      return std::sqrt(a_.re*a_.re + a_.im*a_.im);
    }

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Right multiplication with an \f$SU(3)\f$ matrix.
    ///
    ///  This returns the product \f$ U V \f$, where \f$ V \f$ is the
    ///  background field's value represented by the instance.
    ///
    ///  \param U The \$f SU3 \$f matrix the instance is to be applied
    ///  to. 
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Tue Sep 27 11:03:50 2011
    virtual SU3 ApplyFromRight ( const SU3 & U) const {
      SU3 result;
      for (int i = 0; i < 3; ++i)
	for (int j = 0; j < 3; ++j)
	  result.whr[3*i + j] = a_ * U.whr[3*i + j];
      return result;
    }
    virtual CVector ApplyFromRight ( const CVector & v) const {
      CVector result;
      for (int i = 0; i < 3; ++i)
        result.whr[i] = a_ * v.whr[i];
      return result;
    }
    bool operator==(const ScalarBgf& other) const{
      return a_ == other.a_;
    }
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Generic *= operator template.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Mon Sep 26 18:28:23 2011
    template<class C>
    ScalarBgf& operator*= ( const C& alpha ) {
      a_ *= alpha;
      return *this;
    }
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Generic /= operator template.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Mon Sep 26 18:28:23 2011
    template<class C>
    ScalarBgf& operator/= ( const C& alpha ) {
      a_ /= alpha;
      return *this;
    }
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Generic multiplication.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Mon Sep 26 18:28:36 2011
    template<class C>
    ScalarBgf operator* (const C& alpha ) const {
      ScalarBgf result(*this);
      return result *= alpha;
    }
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Multiplication with self type.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Tue Sep  4 15:18:20 2012

    ScalarBgf operator* (const ScalarBgf& alpha ) const {
      ScalarBgf result(*this);
      return result *= alpha.a_;
    }
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Addition and subtraction of a sclar.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Wed Jan 11 18:03:06 2012

    template<class C>
    ScalarBgf& operator+= (const C& alpha ){
      a_ += alpha;
      return *this;
    }
    ScalarBgf& operator+= (const ScalarBgf& alpha ){
      a_ += alpha.a_;
      return *this;
    }
    ScalarBgf operator-() const{
      ScalarBgf result;
      result.a_ = -a_;
      return result;
    }
    template<class C>
    ScalarBgf& operator-= (const C& alpha ){
      a_ -= alpha;
      return *this;
    }
    template<class C>
    ScalarBgf operator+ (const C& alpha ) const {
      ScalarBgf result(*this);
      return result += alpha;
    }
    template<class C>
    ScalarBgf operator- (const C& alpha ) const {
      ScalarBgf result(*this);
      return result -= alpha;
    }
    ScalarBgf inverse() const {
      ScalarBgf result;
      result.a_ = 1./a_;
      return result;
    }
    void set_to_one() { a_ = Cplx(1,0);};
    void set_to_zero() { a_ = Cplx(0,0);};
    /// Trace
    Cplx Tr() const {
      return a_ * 3.;
    }

    /// Make traceless
    virtual void Trless() {
      set_to_zero();
    }

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  reH
    ///
    ///  This takes the traceless part of 0.5*[V - V^\dagger]
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Fri Feb  3 16:23:25 2012
    
    virtual void reH() {
      Trless();
    }

    ScalarBgf dag() const {
      ScalarBgf result(*this);
      result.a_.im = -a_.im;
      return result;
    }

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Number of doubles needed to store the object.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Sun Mar 25 14:50:37 2012

    static const int storage_size = 2;

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Buffer to a vector of doubles.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Sun Mar 25 14:50:53 2012

    std::vector<double>::iterator &
    buffer(std::vector<double>::iterator & i) const {
        *i = a_.re; ++i;
        *i = a_.im; ++i;
      return i;
    }
  
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Write to a file
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Fri May 25 16:02:33 2012

    template <class Writer_t>
    void write(Writer_t& o) const {
      o.write(a_);
    }
    template <class Reader_t>
    void read(Reader_t& o) {
      o.read(a_);
    }

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Read from a buffer
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Mon Mar 26 16:43:22 2012

    std::vector<double>::const_iterator &
    unbuffer(std::vector<double>::const_iterator & i){
      a_.re = *i; ++i;
      a_.im = *i; ++i;
      return i;
    }

    const Cplx& val() const { return a_; }

  private:
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Diagonal elements of V.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Tue Sep 27 10:50:29 2011
    Cplx a_;
  };

  inline ScalarBgf dag(const ScalarBgf& b){
    return b.dag();
  }

  inline std::ostream& operator<<(std::ostream& os, const ScalarBgf& b){
    os << "{ ";
    os << "(" << b.val().re << ", " << b.val().im << ") ";
    os << "}";
    return os;
  }
 
  class AbelianBgfFactory {
  public:
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Initialization of Vt_.
    ///
    ///  Here, we construct \f$ V(t) \f$ a la Peter Weisz, viz.
    ///  \f[ V(t) = \exp i ( {\cal E} x_0 - iC) \f],
    ///  where
    ///  \f[ {\cal E} = -i (C' - C) / T, \f]
    ///  and $C',C$ are the gluon boundary fields.
    ///
    ///  \param T Temporal lattice extend.
    ///  \param L Spatial lattice extend.
    ///  \param eta Background field parameter. C.f. P. Weisz. Usually
    ///  0 is used.
    ///  \param nu Background field parameter. C.f. P. Weisz. Usually
    ///  0 is used.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Tue Sep 27 10:46:16 2011
    AbelianBgfFactory(const int &T, const int& L, const int& s = 0) 
      : Vt_(T+1){
      if (T < 0)
        throw std::exception();
      //if (!s) {
      //  double pi = std::atan(1.)*4.;
      //  double gamma = 1./L/T * (eta + pi/3.);
      //  double eps[3] = {-2.*gamma, gamma, gamma};
      //  double iC[3] = {-( eta - pi/3) / L,
      //                  -eta * (-0.5 + nu)  / L,
      //                  -( eta * (0.5 + nu) + pi/3.) / L};
      //  for (int t = 0; t <= T; ++t)
      //    for (int k = 0; k < 3; ++k)
      //      Vt_[t][k] = exp(Cplx(0,eps[k] * t - iC[k]));
      //}
      // tree level c_t
      if (T + s != L)
        std::cout 
          << "WARNING!\nYou gave strange values for T,L, and s!\n"
          << "T = " << T << "\nL = " << L << "\ns = " << s
          << "\nYou should have T + s = L!\n"
          << "You're entering a world of pain!\n"
          << "This is my last warning.\n";
          
      double ct = 2. / (2 + s);
      // eta and nu remain zero since we only plugged in f for these values!
      const double eta = 0;
      const double nu = 0;
      double pi = std::atan(1.)*4.;
      SU3 C, Cp, B;
      C(0,0) = eta - pi/3;
      C(1,1) = eta * (nu - 0.5);
      C(2,2) = - eta * ( nu + 0.5 ) + pi/3;
      Cp(0,0) = -eta - pi;
      Cp(1,1) = eta * (nu + 0.5) + pi/3;
      Cp(2,2) = -eta * ( nu - 0.5) + 2.*pi/3;
      C *= Cplx(0, 1./L);
      Cp *= Cplx(0, 1./L);
      // Values for the diagonals of V at t = 0, T
      for (int k = 0; k < 3; ++k){
        Vt_[0][k] = exp( C(k,k) ) * ct;
        Vt_[T][k] = exp( Cp(k,k) ) * ct;
      }
      Cplx i = Cplx(0,1);
      for (int t = 1; t < T; ++t){
        Vt_[t][0] = exp(-2. * f[s+1][L/2 - 2] * (t - 0.5*T) * i + 0.5 
                        * (C(0,0) + Cp(0,0)));
        for (int k = 1; k < 3; ++k)
          Vt_[t][k] = exp(f[s+1][L/2 - 2] * (t - 0.5*T) * i + 0.5 
                          * (C(k,k) + Cp(k,k)));
      }
    };
    const three_vec_t& get(int t){ return Vt_.at(t); }
  private:
    std::vector <three_vec_t> Vt_;
    static const double f[3][31];
  };

  /// Returns V_\mu(t)
  inline AbelianBgf get_abelian_bgf(const int& t,
                                    const int& mu,
                                    const int& T = -1,
                                    const int& L = 0,
                                    const int& s = 0){
    static AbelianBgfFactory factory(T, L, s);
    static three_vec_t one = {1,1,1};
    if (!mu)
      return AbelianBgf(one);
    else
      return AbelianBgf(factory.get(t));
  };
  
  template <class B> inline B unit(){ return B(); }
  
  template <> inline AbelianBgf unit<AbelianBgf>() {
    three_vec_t alpha_v = {1, 1, 1};
    return AbelianBgf(alpha_v);
  };

  template <class B> inline B zero(){ return B(); }
  
  template <> inline AbelianBgf zero<AbelianBgf>() {
    three_vec_t alpha_v = {0, 0, 0};
    return AbelianBgf(alpha_v);
  };
  
  inline AbelianBgf random(){
    static MyRand r(134);
    three_vec_t alpha_v = {r.Rand(), r.Rand(), r.Rand()};
    return AbelianBgf(alpha_v);
  };
}

#endif
