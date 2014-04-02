#include "Helper.h"
#include "gtest/gtest.h"

double Rand(){
  return 1.*rand()/RAND_MAX;
}

TEST(SU3CmpTest, EqualMatrices){
  SU3 A,B;
  for (int i = 0; i < 9; ++i){
    A[i] = Cplx(Rand(), Rand());
    B[i] = A[i];
  }
  ASSERT_TRUE(SU3Cmp(A,B)(0.0));
}

TEST(SU3CmpTest, NotEqualMatrices){
  SU3 A,B;
  for (int i = 0; i < 9; ++i){
    A[i] = Cplx(Rand(), Rand());
    B[i] = A[i];
  }
  for (int i = 8; i >= 0; --i){
    A = B;
    A[i] = Cplx(Rand(), Rand());
    ASSERT_FALSE(SU3Cmp(A,B)());
  }
}
