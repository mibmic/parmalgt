#include <MyMath.h>
#include <gtest/gtest.h>
#include <MyRand.h>
#include <Helper.h>

//TEST(SU3, DaggerMul){
//  MyRand r;
//  r.init(32311903);
//  SU3 A(SU3rand(r)), B(SU3rand(r)), C;
//  C = A*dag(B);
//  A ^= B;
//  ASSERT_TRUE(SU3Cmp(A,C)());
//}

TEST(SU3r, rehtest){
  MyRand r;
  r.init(8126729);
  SU3 A(SU3rand(r));
  SU3 B(A);
  B.reH();
  ASSERT_TRUE(SU3Cmp(A, B)());
}
