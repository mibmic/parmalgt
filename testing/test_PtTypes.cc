#include <PtTypes.hpp>
#include <Background.h>
#include <gtest/gtest.h>
#include <Helper.h>

const int ORD = 10;

TEST(PtMatrix, ApplyBgf){
  ptt::PtMatrix<ORD> A = ptt::get_random_pt_matrix<ORD>();
  ptt::PtMatrix<ORD> B(A);
  bgf::AbelianBgf U = bgf::unit<bgf::AbelianBgf>();
  A = A*U;
  A = U*A;
  for (int i = 0; i < ORD; ++i)
    ASSERT_TRUE( SU3Cmp(A[i],B[i])() );
}

TEST(PtMatrix, MultiplySelfT){
  ptt::PtMatrix<ORD> A = ptt::get_random_pt_matrix<ORD>();
  ptt::PtMatrix<ORD>::SU3_t U, nada;
  for (int i = 0; i < 3; ++i)
    U(i,i) = 1;
  for (int k = 0; k < ORD; ++k){
    ptt::PtMatrix<ORD> B;
    B[k] = U;
    for (int j = 0; j <= k; ++j)
      ASSERT_TRUE( SU3Cmp( (A*B)[j], nada)() );
    for (int j = 0; j < ORD-k-1; ++j)
      ASSERT_TRUE( SU3Cmp( (A*B)[j+k+1], A[j])() );
  }
}

TEST(PtMatrix, AddAssign){
  ptt::PtMatrix<ORD> A = ptt::get_random_pt_matrix<ORD>();
  ptt::PtMatrix<ORD> B = A * 2.;
  A += A;
  for (int i = 0; i < ORD; ++i)
    ASSERT_TRUE( SU3Cmp(A[i],B[i])() );
}

TEST(PtMatrix, reHrandom){
  ranlxd::Rand r(8126729);
  ptt::PtMatrix<ORD> A, B;
  for (int i = 0; i < ORD; ++i){
    A[i] = sun::SU3rand(r);
    B[i] = A[i];
  }
  B.reH();
  for (int i = 0; i < ORD; ++i)  ASSERT_TRUE( SU3Cmp(A[i],B[i])() );
}
