#include <gtest/gtest.h>
#include <LocalField.hpp>
#include <cstdlib>
#include <Helper.h>
#include <Kernels.hpp>

template <class Field_t>
struct randomize {
  randomize(const int& seed) { std::srand(seed); }
  static const int dim = Field_t::dim;
  void operator()(Field_t& F, const pt::Point<dim>& n) const {
    F[n] = rand();
  }
  static const int n_cb = 0;
};

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

TEST(Operators, SubAssign){
  typedef fields::LocalField<int, 4> intField;
  intField::extents_t e;
  intField::neighbors_t nn;
  std::fill(e.begin(), e.end(), 10);
  intField A(e,1,0,nn), B(e,1,0,nn);
  A.apply_everywhere(randomize<intField>(123));
  B.apply_everywhere(randomize<intField>(431));
  intField C(A);
  A -= B;
  for (intField::const_iterator a = A.begin(), e = A.end(), 
         b = B.begin(), c = C.begin(); a != e; ++a, ++b, ++c)
    ASSERT_EQ(*a, *c - *b);
}

TEST(Operators, InnerProduct){
  typedef fields::LocalField<Cplx, 4> CplxField;
  CplxField::extents_t e;
  CplxField::neighbors_t nn;
  std::fill(e.begin(), e.end(), 10);
  CplxField A(e,1,0,nn), B(e,1,0,nn);
  A.apply_everywhere(randomize<CplxField>(123));
  B.apply_everywhere(randomize<CplxField>(431));
  Cplx manual = 0;
  for (CplxField::const_iterator a = A.begin(), e = A.end(), 
         b = B.begin(); a != e; ++a, ++b)
    manual += (*a)*(*b);
  ASSERT_TRUE(Cmp(manual, A*B)(1e-14));
}

TEST(Operators, ScalarProduct){
  typedef fields::LocalField<Cplx, 4> CplxField;
  CplxField::extents_t e;
  CplxField::neighbors_t nn;
  std::fill(e.begin(), e.end(), 10);
  CplxField A(e,1,0,nn);
  A.apply_everywhere(randomize<CplxField>(123));
  Cplx alpha = 5.234;
  CplxField B(A);
  A *= alpha;
  for (CplxField::const_iterator a = A.begin(), e = A.end(), 
         b = B.begin(); a != e; ++a, ++b)
    ASSERT_TRUE(Cmp(*a, *b * alpha)());
}

TEST(Buffer, BufferAndUnbuffer){
  typedef fields::LocalField<int, 4> intField;
  const int L = 10;
  intField::extents_t e;
  intField::neighbors_t nn;
  std::fill(e.begin(), e.end(), L);
  geometry::Geometry<4> g(e);
  intField A(e,1,0,nn), B(e,1,0,nn);
  A.apply_everywhere(randomize<intField>(123));
  std::vector<int> buffer(L*L*L);
  typedef kernels::Buffer<intField, std::vector<int>::iterator> Buffer_k;
  typedef kernels::Unbuffer<intField, std::vector<int>::const_iterator> Unbuffer_k;
  Buffer_k buf(buffer.begin());
  Unbuffer_k unbuf(buffer.begin());
  A.apply_on_timeslice(buf, 4);
  B.apply_on_timeslice(unbuf, 4);
  e[0] = 4;
  geometry::TimeSliceIter i(g.mk_point(e), g.get_extents());
  do {
    ASSERT_EQ(A[*i], B[*i]);
  } while((++i).is_good());
}

