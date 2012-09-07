#ifndef _HELPER_H_
#define _HELPER_H_
#include "MyMath.h"
#include <limits>
#include <iostream>
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
  (const double& eps = std::numeric_limits<double>::epsilon()*3)
    const {
    for (Cplx const * aptr = a.whr, * bptr = b.whr;
	 aptr != a.whr + 9; ++aptr, ++bptr)
      if (fabs(aptr->re - bptr->re) > eps ||
	  fabs(aptr->im - bptr->im) > eps)
        return ::testing::AssertionFailure() 
          << "at i = " << aptr - a.whr << std::endl << "  ("
          << aptr->re << ", " << aptr->im << 
          ") !=\n  ("<< bptr->re << ", " << bptr->im << ")";
    return  ::testing::AssertionSuccess();
  }
  const SU3 &a, &b;
};

struct CNear {
  double eps;
  explicit CNear(const double& e = 
                 std::numeric_limits<double>::epsilon()*3) :
    eps(e) { }
  ::testing::AssertionResult operator()(const Cplx& a, const Cplx& b){
    if (fabs(a.re - b.re) > eps ||
        fabs(a.im - b.im) > eps )
      return ::testing::AssertionFailure() 
        << "(" << a.re << "," << a.im <<  ") vs. (" 
        << b.re << "," << b.im <<  ")\n";
    else 
      return ::testing::AssertionSuccess();
  }
};

#endif
