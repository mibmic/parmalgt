#include "newQCDpt.h"
#include <Background.h>
#include "gtest/gtest.h"
#include "Helper.h"
#include <cstdlib> // for rand

const int MYORD = 10;

typedef BGptSU3<bgf::AbelianBgf, MYORD> ptSU3;
typedef ptt::PtMatrix<MYORD> ptsu3;
//typedef BGptCVector<MYORD> ptCVector;
typedef BGptGluon<bgf::AbelianBgf, MYORD, 4> ptGluon;
//typedef BGptSpinColor<MYORD, 4> ptSpinColor;

int L = std::rand() % 100;
int T = std::rand() % 100;
double eta = std::rand()/static_cast<double>(RAND_MAX);
double nu = std::rand()/static_cast<double>(RAND_MAX);
// initialize the background field


// Simple test base class defining two random 
// perturbative SU3s ,,,

class AbelianBgfTest: public ::testing::Test {
public:
  AbelianBgfTest() : 
    MyPtSU3A(), 
    MyPtSU3B() 
  { 
    MyPtSU3A.randomize();
    MyPtSU3B.randomize();
  }
  ptSU3 MyPtSU3A, MyPtSU3B;
};

// Multiplication of two ptSU3s of the form
// A = V(t1) + g_0^2 A1 + ... ,
// B = V(t2) + g_0^2 B1 + ... , and
// One = 1

TEST_F(AbelianBgfTest, SimpleMultiply){
  ptSU3 One(bgf::unit<bgf::AbelianBgf>()); // unit matrix
  // to be extra safe, check multiplication from left and right
  ptSU3 ACopy = One*MyPtSU3A;
  ptSU3 BCopy = MyPtSU3B*One;
  for (int i = 0; i < MYORD; ++i){
    ASSERT_TRUE( SU3Cmp(ACopy[i], MyPtSU3A[i])() );
    //ASSERT_TRUE( SU3Cmp(BCopy[i], MyPtSU3B[i])() );
  }
}

TEST(AbelianBgf, Multiply){
  ptSU3 A(bgf::unit<bgf::AbelianBgf>()), B(bgf::unit<bgf::AbelianBgf>());
  SU3 One, Zero;
  for (int i = 0; i < 3; ++i)
    One(i,i) = Cplx(1,0);
  A[2] = One*2;
  B[4] = One*3;
  ptSU3 AB = A * B;
  ASSERT_TRUE( SU3Cmp(AB[0], Zero)() );
  ASSERT_TRUE( SU3Cmp(AB[1], Zero)() );
  ASSERT_TRUE( SU3Cmp(AB[2], One*2)() );
  ASSERT_TRUE( SU3Cmp(AB[3], Zero)() );
  ASSERT_TRUE( SU3Cmp(AB[4], One*3)() );
  ASSERT_TRUE( SU3Cmp(AB[5], Zero)() );
  ASSERT_TRUE( SU3Cmp(AB[6], Zero)() );
  ASSERT_TRUE( SU3Cmp(AB[7], One*6)() );
}

// Test the addition,
// A+A == 2*A

TEST_F(AbelianBgfTest, Add){
  ptSU3 A_times_2 = MyPtSU3A*2.;
  ptSU3 A_plus_A = MyPtSU3A + MyPtSU3A;
  for (int i = 0; i < MYORD; ++i)
    ASSERT_TRUE( SU3Cmp(A_times_2[i], A_plus_A[i])() );
  ASSERT_TRUE( A_times_2.bgf() ==  A_plus_A.bgf() );
}


// Testing the scalar multiplicatiohn

TEST_F(AbelianBgfTest, ScalarMultiply){
  Cplx alpha(std::rand()/static_cast<double>(RAND_MAX),
	     std::rand()/static_cast<double>(RAND_MAX));
  Cplx beta(std::rand()/static_cast<double>(RAND_MAX),
	    std::rand()/static_cast<double>(RAND_MAX));
  ptSU3 beta_B = MyPtSU3B;
  beta_B *= beta;
  for (int i = 0; i < MYORD; ++i){
    SU3 a = MyPtSU3A[i];
    SU3 b = MyPtSU3B[i];
    // multipy 'by hand'
    for (SU3::iterator ait = a.begin(), bit = b.begin();
         ait != a.end(); ++ait, ++bit){
      *ait *= alpha;
      *bit *= beta;
    }
    EXPECT_TRUE ( SU3Cmp( (MyPtSU3A*alpha)[i], a)());
    EXPECT_TRUE ( SU3Cmp( beta_B[i], b)());
  }
}

//TEST(BGptCVector, ArithmeticPlusSelf){
//  ptCVector u, v, w;
//  for( ptCVector::iterator i = v.begin(), j = w.begin();
//         i != v.end(); ++i, ++j)
//    for (int k = 0; k < 3; ++k){
//      i->whr[k] = Cplx(r.Rand(), r.Rand());
//      j->whr[k] = i->whr[k]*3;
//    }
//  u = v+v;
//  u += v;
//  for( ptCVector::iterator i = u.begin(), j = w.begin();
//         i != u.end(); ++i, ++j)
//    ASSERT_TRUE(*i == *j);
//}
//
//TEST(BGptCVector, ProductWithSU3){
//  ptCVector v, w;
//  SU3 One;
//  CVector OneV, ZeroV;
//  ptSU3 A(bgf::unit<bgf::AbelianBgf>());
//  for (int i = 0; i < 3; ++i){
//    One(i,i) = Cplx(1,0);
//    OneV.whr[i] = Cplx(1,0);
//  }
//  A[3] = One*2;
//  for (int i = 0; i < 3; ++i)
//    v[1].whr[i] = Cplx(3,0);
//  w = v*A;
//  ASSERT_TRUE(w[0] == ZeroV);
//  ASSERT_TRUE(w[1] == OneV*Cplx(3,0));
//  ASSERT_TRUE(w[2] == ZeroV);
//  ASSERT_TRUE(w[3] == ZeroV);
//  ASSERT_TRUE(w[4] == ZeroV);
//  ASSERT_TRUE(w[5] == OneV*Cplx(6,0));
//  ASSERT_TRUE(w[6] == ZeroV);
//}

//TEST(BGptSpinColor, ProductWithGluon){
//  ptGluon U;
//  SU3 One;
//  CVector OneV, ZeroV;
//  ptSU3 A(bgf::unit<bgf::AbelianBgf>());
//  ptSpinColor psi, chi;
//  for (int i = 0; i < 3; ++i){
//    One(i,i) = Cplx(1,0);
//    OneV.whr[i] = Cplx(1,0);
//  }
//  for (int i = 0; i < 4; ++i){
//    U[i].bgf() = bgf::unit<bgf::AbelianBgf>();
//    U[i][i] = One*i;
//    psi[i][5] = OneV*i;
//  }
//  chi = psi*U; // <-- check this
//  for (int i = 0; i < 4; ++i)
//    for (int j = 0; j <= MYORD; ++j)
//      if (i && j == 5)
//        ASSERT_TRUE(chi[i][j] == OneV*Cplx(i));
//      else if (i && j == 5 + i + 1)
//        ASSERT_TRUE(chi[i][j] == OneV*Cplx(i)*Cplx(i));
//      else
//        ASSERT_TRUE(chi[i][j] == ZeroV);
//}

//TEST(BGptSU3Test, dag){
//  ptSU3 A(bgf::random()), B;
//  A.randomize();
//  B = dag(A);
//  for (int r = 0; r < MYORD; ++r)
//    for (int i = 0; i < 3; ++i)
//      for (int j = 0; j < 3; ++j){
//        ASSERT_DOUBLE_EQ( B[r](i,j).re, A[r](j,i).re );
//      }
//}

TEST(BGptSU3Test, exp){
  ptt::PtMatrix<MYORD> A = ptt::get_random_pt_matrix<MYORD>();
  ptSU3 B = exp<bgf::AbelianBgf, MYORD>(A);
  // calculate the first few terms by hand ...
  // tree level
  ASSERT_TRUE(B.bgf() == bgf::unit<bgf::AbelianBgf>());
  // one loop
  ASSERT_TRUE(SU3Cmp(A[0],B[0])());
  // two loop
  SU3 e = A[0]*A[0]*0.5 + A[1];  
  ASSERT_TRUE(SU3Cmp(e,B[1])());
  // three loop
  e = A[0]*A[0]*A[0]/6. + A[0]*A[1]*.5 + A[1]*A[0]*.5 + A[2];
  double eps = std::numeric_limits<double>::epsilon() * 10;
  ASSERT_TRUE(SU3Cmp(e,B[2])(eps));
}

// Check if exp(A) * exp(-A) = 1 holds

TEST(BGptSU3Test, expUnit){
  for (int _n = 0; _n < 1000; ++_n){
    ptt::PtMatrix<MYORD> A = ptt::get_random_pt_matrix<MYORD>();
    ptSU3 B = exp<bgf::AbelianBgf, MYORD>(A);
    ptSU3 C = exp<bgf::AbelianBgf, MYORD>(A * -1.);
    ptSU3 D = B*C;
    SU3 zero;
    // calculate the first few terms by hand ...
    // tree level
    ASSERT_TRUE(D.bgf() == bgf::unit<bgf::AbelianBgf>());
    for (int i = 0; i < 7; ++i)
      ASSERT_TRUE( SU3Cmp( zero, D[i] )(2e-14*i) );
  }
}

TEST(BGptSU3Test, inverse){
  ptSU3 Z;
  Z.ptU() = ptt::get_random_pt_matrix<MYORD>();
  ptSU3 W = inverse(Z);
  ptSU3 unit = Z*W;
  ASSERT_TRUE( unit.bgf() == bgf::unit<bgf::AbelianBgf>() );
  for (int i = 0; i < MYORD; ++i)
    ASSERT_TRUE( SU3Cmp( unit[i], SU3() )(1e-10) ) << i;
}

TEST(BGptSU3Test, expUnitMore){
  for (int _n = 0; _n < 1000; ++_n){
    ptt::PtMatrix<MYORD> A = ptt::get_random_pt_matrix<MYORD>();
    ptSU3 a, b;
    a.randomize(); a.bgf() = bgf::random();
    b.randomize(); b.bgf() = bgf::random();
    ptSU3 d = b;
    ptSU3 B = exp<bgf::AbelianBgf, MYORD>(A);
    ptSU3 C = exp<bgf::AbelianBgf, MYORD>(A * -1.);
    ptSU3 D = (B)*(C*b);
    // calculate the first few terms by hand ...
    // tree level
    ASSERT_TRUE(D.bgf() == d.bgf());
    for (int i = 0; i < 7; ++i)
      ASSERT_TRUE( SU3Cmp( d[i], D[i] )(1e-13*(i+1)) ) << i;
  }
}

TEST(BGptSU3Test, Multiplication){
  for (int _n = 0; _n < 1000; ++_n){
    ptSU3 a, b, c, d, e;
    a.randomize(); a.bgf() = bgf::random();
    b.randomize(); b.bgf() = bgf::random();
    c.randomize(); c.bgf() = bgf::random();
    d = (a*b)*c; e = a*(b*c);
    for (int i = 0; i < MYORD; ++i)
      ASSERT_TRUE( SU3Cmp( d[i], e[i] )(1e-13*(i+1)) ) << i;
  }
}    

TEST(BGptSU3Test, logThrows){
  ptSU3 A(bgf::random());
  if (IsZero<do_debug, bgf::AbelianBgf>::debug_on)
    ASSERT_THROW(log(A), IsNotZeroError);
  else
    ASSERT_NO_THROW(log(A));
}


TEST(BGptSU3Test, log){
  ptSU3 A(bgf::unit<bgf::AbelianBgf>());
  A.randomize();
  A.bgf().set_to_one();
  // this assumes A has unit background field
  ptsu3 B = log(A);
  // one loop
  ASSERT_TRUE( SU3Cmp(A[0], B[0])() );
  // two loop
  SU3 e = A[1] - A[0]*A[0]*0.5;
  ASSERT_TRUE( SU3Cmp(e, B[1])() );
  // three loop
  e = A[2] - (A[0]*A[1] + A[1]*A[0])*0.5 
    + A[0]*A[0]*A[0]/3;
  ASSERT_TRUE( SU3Cmp(e, B[2])() );
}

// Now, we want to check if log(exp(A)) = A holds...
TEST(BGptSU3Test, logexp){
  ptt::PtMatrix<MYORD> A = ptt::get_random_pt_matrix<MYORD>(), B;
  B = log(exp<bgf::AbelianBgf,MYORD>(A));
  ptSU3 C(bgf::unit<bgf::AbelianBgf>()), D;
  C.randomize();
  C.bgf().set_to_one();
  D = exp<bgf::AbelianBgf,MYORD>(log(C));
  // higher orders
  // a quick and dirty estimate gives that the round-off error should
  // be around 100 * epsilon * n at order n ...
  double eps = std::numeric_limits<double>::epsilon() * 100;
  for (int r = 0; r < 6; ++r){
    ASSERT_TRUE( SU3Cmp(A[r], B[r])(eps * (r+1)) );
    ASSERT_TRUE( SU3Cmp(C[r], D[r])(eps * (r+1)) );
  }
}

// we need a main function here because we have to initialize the
// background field class...

int main(int argc, char **argv) {
  bgf::get_abelian_bgf(0,0,T, L);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
