#ifndef _HELPER_H_
#define _HELPER_H_
#include <limits>
#include <iostream>
#include <Types.h>
#include <gtest/gtest.h>

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///  Class to compare SU3 Matrix types
///
///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
///  \date Wed Sep 21 19:12:10 2011
struct SU3Cmp {
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Constructor.
  ///
  ///  \param A First of the matrices to be compared
  ///  \param B Second matrix to be compared 
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Wed Sep 21 19:12:29 2011
  SU3Cmp(const SU3 &A, const SU3 &B) : a(A), b(B) {}
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Perform the Comparison
  ///
  ///  \param eps Accuracy to which we want to the matrices' entries
  ///  to be compared. Default value is three machine accuracies.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Wed Sep 21 19:14:08 2011
  ::testing::AssertionResult operator()
  (const double& eps = std::numeric_limits<double>::epsilon()*5)
    const {
    for (int i = 0; i < 9; ++i)
      if (abs(a[i] - b[i]) > eps)        
        return (::testing::AssertionFailure()
                << "at i = " << i << " :  ("
                << a[i].real() << ", " << a[i].imag() 
                << ") !=\n  ("<< b[i].real() << ", " << b[i].imag() << ")\n"
                << "delta = " << abs(a[i] - b[i]));
    return  ::testing::AssertionSuccess();
  }
  const SU3 &a, &b;
};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///  Class to compare Complex types
///
///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
///  \date Thu Nov 6 2011
struct Cmp {
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Constructor.
  ///
  ///  \param A First of the numbers to be compared
  ///  \param B Second number
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Wed Sep 21 19:12:29 2011
  Cmp(const Cplx &A, const Cplx &B) : a(A), b(B) {}
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Perform the Comparison
  ///
  ///  \param eps Accuracy to which we want to the matrices' entries
  ///  to be compared. Default value is three machine accuracies.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Wed Sep 21 19:14:08 2011
  ::testing::AssertionResult operator()
  (const double& eps = std::numeric_limits<double>::epsilon()*3)
    const {
    if (abs(a-b)/abs(a+b) > eps) return ::testing::AssertionFailure();
    //if (fabs((a.re - b.re)/a.re) > eps ||
    //    fabs((a.im - b.im)/a.im) > eps)
    //    return ::testing::AssertionFailure() 
    //      << "  ("  << a.re << ", " << a.im << 
    //      ") !=\n  ("<< b.re << ", " << b.im << ")";
    return  ::testing::AssertionSuccess();
  }
  const Cplx &a, &b;
};


#endif
