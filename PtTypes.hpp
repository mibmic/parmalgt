#ifndef _PT_TYPES_H_
#define _PT_TYPES_H_

#include <Types.h>
#include <MyRand.h>
#include <MyMath.h>
#include <Background.h>
#include <PtAlgo.hpp>

namespace ptt {

  struct True {};
  struct False {};
  template <typename T> struct ScalarMultiplyable { typedef False type; };
  template <> struct ScalarMultiplyable<int> { typedef True type; };
  template <> struct ScalarMultiplyable<double> { typedef True type; };
  template <> struct ScalarMultiplyable<Cplx> { typedef True type; };

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  PtMatrix defines a series
  ///
  ///  A = g A^(1) + g^2 A^(2) + ... + g^(ORD + 1) A^(ORD+1)
  ///
  ///  equivalently, we have in matrix notation
  /// 
  ///  A = g A[0] + g^2 A[1] + ... + g^(ORD+1) A[ORD]
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Fri Feb  3 12:39:27 2012

  template <int N> class PtMatrix {
  public:
    typedef PtMatrix self_t;
    typedef typename array_t<SU3, N>::Type su3_array_t;
    typedef typename su3_array_t::value_type SU3_t;
    typedef typename su3_array_t::iterator iterator;
    typedef typename su3_array_t::const_iterator const_iterator;



    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Access operators
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Fri Feb  3 12:41:56 2012

    //const SU3_t& operator[](const int& i) const { return array_[i]; }
    //SU3_t& operator[](const int& i) { return array_[i]; }
    const SU3_t& operator[](const int& i) const { return array_.at(i); }
    SU3_t& operator[](const int& i) { return array_.at(i); }

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Iterators
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Fri Feb  3 12:42:03 2012

    const_iterator begin() const { return array_.begin(); }
    const_iterator end() const { return array_.end(); }
    iterator begin() { return array_.begin(); }
    iterator end() { return array_.end(); }

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Number of doubles requred to store the PtMatrix in a double
    ///  buffer (needed for MPI).
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Sun Mar 25 14:46:27 2012

    static const int storage_size = N*9*2;


    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Buffering.
    ///
    ///  Write the contents to a vector of doubles.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Sun Mar 25 14:47:32 2012
    
    std::vector<double>::iterator &
    buffer(std::vector<double>::iterator & i) const {
      for (const_iterator n = begin(); n!= end(); ++n)
        for (int j = 0; j < 9; ++j){
          *i = (*n)[j].re; ++i;
          *i = (*n)[j].im; ++i;
        }
      return i;
    }

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Write to a file.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Fri May 25 16:33:38 2012

    template <class Writer_t>
    void write(Writer_t& o) const {
      for (const_iterator n = begin(); n!= end(); ++n)
        for (int j = 0; j < 9; ++j)
          o.write((*n)[j]);
    }

    template <class Reader_t>
    void read(Reader_t& o) {
      for (iterator n = begin(); n!= end(); ++n)
        for (int j = 0; j < 9; ++j)
          o.read((*n)[j]);
    }
  
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Read from buffer.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Mon Mar 26 16:45:23 2012

    std::vector<double>::const_iterator &
    unbuffer(std::vector<double>::const_iterator & i){
      for (iterator n = begin(); n!= end(); ++n)
        for (int j = 0; j < 9; ++j){
          (*n)[j].re = *i; ++i;
          (*n)[j].im = *i; ++i;
        }
      return i;
    }    

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Arithmetics
    ///
    ///  For now, we do not overload operators, but just call the 
    ///  methods e.g add_assign etc.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Fri Feb  3 12:42:12 2012

    self_t& operator+=(const self_t& other) {
      std::for_each(begin(), end(), pta::incr(other));
      return *this;
    }
  
    self_t operator+(const self_t& other) const {
      self_t result(*this);
      return result += other;
    }

    self_t& operator-=(const self_t& other) {
      std::for_each(begin(), end(), pta::decr(other));
      return *this;
    }
  
    self_t operator-(const self_t& other) const {
      self_t result(*this);
      return result -= other;
    }
  

    
    // Here, we only want to implement the *= and /= operators for
    // scalar types, where the operation is trivial. Hence, we have to
    // make a small detour through a separate implementation function,
    // whose second argument determines if we have a scalar
    // multiplication. For which types T this is the case, the
    // class template member ScalarMultiplyable<T>::type decides for us.

    template <typename T> self_t& operator*=(const T& other) {
      return mul_assign_impl(other, typename ScalarMultiplyable<T>::type());
    }

    self_t& operator*=(const self_t& other) {
      *this = *this * other;
      return *this;
    }

    template <typename T> self_t& operator/=(const T& other) {
      return div_assign_impl(other, typename ScalarMultiplyable<T>::type());
    }

    self_t& reH() {
      Cplx tr;
      for(int i = 0; i < N; i++){
        array_[i] -= dag(array_[i]);
        array_[i] *= .5;
        tr = array_[i].Tr()/3.;
        array_[i][0] -= tr;
        array_[i][4] -= tr;
        array_[i][8] -= tr;
      }
      return *this;
    }

  private:
    su3_array_t array_;

    // implementation of *= and /=. For an explaination c.f. above.

    template <typename T> self_t& mul_assign_impl(const T& other, True){
      std::for_each(begin(), end(), pta::mul(other));
      return *this;
    }
    template <typename T> self_t& div_assign_impl(const T& other, True){
      std::for_each(begin(), end(), pta::div(other));
      return *this;
    }
  };

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///  reH call, keeping the matrix constant
///
///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
///  \date Tue Apr 24 15:10:50 2012

  template <int N> PtMatrix<N> reH(const PtMatrix<N> &A){
    PtMatrix<N> result(A);
    return result.reH();
  }

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///  Random PtMatrix
///
///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
///  \date Fri Feb  3 12:43:06 2012

  template <int N> inline PtMatrix<N> get_random_pt_matrix(){
    PtMatrix<N> result;
    static MyRand r(1235431);
#ifdef HAVE_CXX0X
    for (auto& e : result) { e = SU3rand(r); }
#else
    for (int i = 0; i < N; ++i)
      result[i] = SU3rand(r);
#endif
    return result;
  }

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///  The rest of the arithmetic as templates ...
///
///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
///  \date Fri Feb  3 12:43:47 2012

  template <int N, typename T> 
  inline PtMatrix<N> operator*(const PtMatrix<N>& A, const T& alpha){
    return PtMatrix<N>(A) *= alpha;
  }

  template <int N>
  inline PtMatrix<N> operator*(const PtMatrix<N>& A, const bgf::AbelianBgf& bg){
#ifdef HAVE_CXX0X
    PtMatrix<N> result(A);
    for (auto& e : result) { e = bg.ApplyFromRight(e); }
#else
    PtMatrix<N> result;
    for (int i = 0; i < N; ++i)
      result[i] = bg.ApplyFromRight(A[i]);
#endif
    return result;
  }

  template <int N>
  inline PtMatrix<N> operator*(const PtMatrix<N>& A, const bgf::ScalarBgf& bg){
#ifdef HAVE_CXX0X
    PtMatrix<N> result(A);
    for (auto& e : result) { e = bg.ApplyFromRight(e); }
#else
    PtMatrix<N> result;
    for (int i = 0; i < N; ++i)
      result[i] = bg.ApplyFromRight(A[i]);
#endif
    return result;
  }
  template <int N>
  inline PtMatrix<N> operator*(const PtMatrix<N>& A, const PtMatrix<N>& B){
    PtMatrix<N> result;
    for (int i = 0; i < N - 1; ++i)
      for (int j = 0; j <= i; ++j)
        result[i + 1] += A[j] * B[i - j];
    return result;
  }
  
  template <int N, typename T> 
  inline PtMatrix<N> operator*(const T& alpha, const PtMatrix<N>& A){
    return PtMatrix<N>(A) *= alpha;
  }

  template <int N>
  inline PtMatrix<N> operator*(const bgf::AbelianBgf& bg, const PtMatrix<N>& A){
#ifdef HAVE_CXX0X
    PtMatrix<N> result(A);
    for (auto& e : result) { e = bg.ApplyFromLeft(e); }
#else
    PtMatrix<N> result;
    for (int i = 0; i < N; ++i)
      result[i] = bg.ApplyFromLeft(A[i]);
#endif
    return result;
  }
  template <int N>
  inline PtMatrix<N> operator*(const bgf::ScalarBgf& bg, const PtMatrix<N>& A){
#ifdef HAVE_CXX0X
    PtMatrix<N> result(A);
    for (auto& e : result) { e = bg.ApplyFromLeft(e); }
#else
    PtMatrix<N> result;
    for (int i = 0; i < N; ++i)
      result[i] = bg.ApplyFromLeft(A[i]);
#endif
    return result;
  }

}; // namespace ptt

#endif // ndef _PT_TYPES_H_
