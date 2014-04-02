int PTORD = 6;

#include "gtest/gtest.h"
#include "Helper.h"
double Rand(){
  return rand()/RAND_MAX;
}

TEST(SU3, Multiplication){
  //
  // This is a little test scenario to expose a bug ...
  SU3 A, Acpy, alphaA; // A, and two copies
  Cplx alpha(r.Rand(), r.Rand()); // random complex 
  for (SU3::iterator ait = A.begin(), 
         aait = alphaA.begin(),
         acpit = Acpy.begin(); 
       ait != A.end(); ++ait, ++aait, ++acpit){
    *ait = Cplx(r.Rand(), r.Rand()); // randomize A
    *acpit = *ait; // copy a
    *aait = *ait*alpha; // alphaA = alpha * A
  }
  EXPECT_TRUE ( SU3Cmp( alphaA, alpha*A)() ); // works
  EXPECT_TRUE ( SU3Cmp( alphaA, A*alpha)() ); // works
  Acpy *= alpha; // <- here, it hits the fan
  EXPECT_TRUE ( SU3Cmp( alphaA, Acpy)() ); // does not work
}

// Test if initialized to zero
TEST(SU3, InitWithZero){
  SU3 A,B;
  // force A to zero
  for (SU3::iterator ait = A.begin();
       ait != A.end(); ++ait) *ait = Cplx(0,0);
  ASSERT_TRUE( SU3Cmp (A, B)() );
}
