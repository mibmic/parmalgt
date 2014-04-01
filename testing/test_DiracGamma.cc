#include <Background.h>
#include <gtest/gtest.h>
#include <LocalField.hpp>
#include <cstdlib>
#include <Helper.h>
#include <Kernels.hpp>


const int DIM = 4;

typedef SpinColor<4> Fermion;
typedef fields::LocalField< Fermion , DIM> ScalarFermionField;
typedef std::vector<ScalarFermionField> FermionField;

TEST(Operators, AddAssign){
  typedef fields::LocalField<int, 4> intField;
  intField::extents_t e;
  intField::neighbors_t nn;
  std::fill(e.begin(), e.end(), 10);
  intField A(e,1,0,nn), B(e,1,0,nn);
  A.apply_everywhere(randomize<intField>(123));
  B.apply_everywhere(randomize<intField>(431));
  intField C(A);
  A += B;
  for (intField::const_iterator a = A.begin(), e = A.end(), 
         b = B.begin(), c = C.begin(); a != e; ++a, ++b, ++c)
    ASSERT_EQ(*a, *b + *c);
}


TEST(Gamma, nPT){
ScalarFermionField::extents_t e;
ScalarFermionField::neighbors_t nn;
std::fill(e.begin(), e.end(), 4);
ScalarFermionField F(e,1,0,nn);

}
