#ifndef _TEST_FFT_H
#define _TEST_FFT_H
#include <gtest/gtest.h>

#include <LocalField.hpp>
#include <Background.h>
#include <Geometry.hpp>
#include <newQCDpt.h>
#include <Kernels.hpp>
#include <stdlib.h>
#include <IO.hpp>

#include<fft.hpp>



// space-time dimensions
const int DIM = 4;
const int L = 8;

// some short-hands
typedef bgf::ScalarBgf Bgf_t; // background field
typedef pt::Point<DIM> Point;
typedef pt::Direction<DIM> Direction;

// shorthand for gluon field
// typedef fields::LocalField<ptGluon, DIM> GluonField;

// shorthand for fermion field
typedef SpinColor<DIM> Fermion;
typedef fields::LocalField< Fermion , DIM> FermionField;
typedef FermionField::neighbors_t nt;


// TEST(fft, double){
//   typedef fields::LocalField< double , DIM> dField;
//   typedef dField::neighbors_t nt;
//   // lattice setup
//   // generate an array for to store the lattice extents
//   geometry::Geometry<DIM>::extents_t e;
//   // we want a L = 4 lattice
//   std::fill(e.begin(), e.end(), L);
//   dField Xi(e, 1, 0, nt());
//   dField Back(e, 1, 0, nt());
//   dField Old(e, 1, 0, nt());
//   fft::fft<dField,fft::pbc> ft(Xi);
//   for (dField::iterator it=Xi.begin();it!=Xi.end();++it)
//     *it = rand()/static_cast<double>(RAND_MAX);
//   Old=Xi;
//   ft.execute(Xi,Back,fft::x2p);
//   ft.execute(Back,Back,fft::p2x);
//   dField::iterator it1=Back.begin();
//   dField::iterator it2=Old.begin();
//   do {
//     double X = (*it1 - *it2);
//     // std::cout << X << std::endl;
//     ASSERT_TRUE( X<1e-14);
//   } while ( (++it1)!=Xi.end() && (++it2)!=Old.end() );
// }


TEST(fft, complex){
  typedef fields::LocalField< complex , DIM> cField;
  typedef cField::neighbors_t nt;
  // lattice setup
  // generate an array for to store the lattice extents
  geometry::Geometry<DIM>::extents_t e;
  // we want a L = 4 lattice
  std::fill(e.begin(), e.end(), L);
  cField Xi(e, 1, 0, nt());
  cField Back(e, 1, 0, nt());
  cField Old(e, 1, 0, nt());
  fft::fft<cField,fft::pbc> ft(Xi);
  for (cField::iterator it=Xi.begin();it!=Xi.end();++it)
    *it = complex(rand()/static_cast<double>(RAND_MAX),rand()/static_cast<double>(RAND_MAX));
  Old=Xi;
  // ft.execute(fft::x2p);
  // ft.execute(fft::p2x);
  ft.execute(Xi,Back,fft::x2p);
  ft.execute(Back,Back,fft::p2x);
  cField::iterator it1=Back.begin();
  cField::iterator it2=Old.begin();
  do {
    complex X = (*it1 - *it2);
    // std::cout << X << std::endl;
    ASSERT_TRUE( X.abs()<1e-10);
  } while ( (++it1)!=Back.end() && (++it2)!=Old.end() );
}




TEST(fft, sun){
  typedef fields::LocalField< sun::SU<3> , DIM> su3Field;
  typedef su3Field::neighbors_t nt;
  // lattice setup
  // generate an array for to store the lattice extents
  geometry::Geometry<DIM>::extents_t e;
  // we want a L = 4 lattice
  std::fill(e.begin(), e.end(), L);
  su3Field Xi(e, 1, 0, nt());
  su3Field Back(e, 1, 0, nt());
  su3Field Old(e, 1, 0, nt());
  fft::fft<su3Field,fft::pbc> ft(Xi);
  ranlxd::Rand R(1234);
  for (su3Field::iterator it=Xi.begin();it!=Xi.end();++it)
    *it = sun::SU3rand(R);
  Old=Xi;

  su3Field::iterator it0=Xi.begin();
  su3Field::iterator it1=Back.begin();
  su3Field::iterator it2=Old.begin();

  ft.execute(Xi,Back,fft::x2p);
  do {
    sun::SU<3> X = (*it1 - *it0);
    ASSERT_FALSE( X.norm()<1e-10);
  } while ( (++it1)!=Back.end() && (++it0)!=Xi.end() );

  it1=Back.begin();
  ft.execute(Back,Back,fft::p2x);
  do {
    sun::SU<3> X = (*it1 - *it2);
    ASSERT_TRUE( X.norm()<1e-10);
  } while ( (++it1)!=Back.end() && (++it2)!=Old.end() );
}


Fermion& randomize(Fermion& F) {
  for(typename Fermion::iterator it=F.begin();it!=F.end();++it) {
    (*it)[0] = complex(std::rand(),std::rand())/static_cast<double>(RAND_MAX);
    (*it)[1] = complex(std::rand(),std::rand())/static_cast<double>(RAND_MAX);
    (*it)[2] = complex(std::rand(),std::rand())/static_cast<double>(RAND_MAX);
  }
  return F;
}

template<int N>
double max(sun::Vec<N>& v) {
  double m(0),n;
  for( typename sun::Vec<N>::const_iterator it = v.begin(); it != v.end(); ++it)
    if( (n=(*it).abs()) > m ) m = n;
  return m;
}
template<int N>
double max(SpinColor<N>& f) {
  double m(max(f[0])),n;
  for(int i=1;i<N;++i)
    if( (n=max(f[i])) > m ) m = n;
  return m;
}

TEST(fft, pbc){
  // lattice setup
  // generate an array for to store the lattice extents
  geometry::Geometry<DIM>::extents_t e;
  // we want a L = 4 lattice
  std::fill(e.begin(), e.end(), L);

  FermionField Xi(e, 1, 0, nt());
  FermionField Back(e, 1, 0, nt());
  FermionField Old(e, 1, 0, nt());
  fft::fft<FermionField,fft::pbc> ft(Xi);
  for (FermionField::iterator it=Xi.begin();it!=Xi.end();++it)
    randomize(*it);
  Old=Xi;
  ft.execute(fft::x2p);
  ft.execute(fft::p2x);
  ft.execute(Xi,Back,fft::x2p);
  ft.execute(Back,Back,fft::p2x);

  FermionField::iterator it1=Back.begin();
  FermionField::iterator it2=Old.begin();
  do {
    Fermion X(*it1 - *it2);
    ASSERT_TRUE(max(X)<1e-14);
  } while ( (++it1)!=Back.end() && (++it2)!=Old.end() );

}

TEST(fft, inplacePBC){
  // lattice setup
  // generate an array for to store the lattice extents
  geometry::Geometry<DIM>::extents_t e;
  // we want a L = 4 lattice
  std::fill(e.begin(), e.end(), L);

  FermionField Xi(e, 1, 0, nt());
  FermionField Old(e, 1, 0, nt());
  fft::fft<FermionField,fft::pbc> ft(Xi);
  for (FermionField::iterator it=Xi.begin();it!=Xi.end();++it)
    randomize(*it);
  Old=Xi;
  ft.execute(fft::x2p);
  ft.execute(fft::p2x);

  FermionField::iterator it1=Xi.begin();
  FermionField::iterator it2=Old.begin();
  do {
    Fermion X(*it1 - *it2);
    ASSERT_TRUE(max(X)<1e-14);
  } while ( (++it1)!=Xi.end() && (++it2)!=Old.end() );
}



TEST(fft, abc){
  // lattice setup
  // generate an array for to store the lattice extents
  geometry::Geometry<DIM>::extents_t e;
  // we want a L = 4 lattice
  std::fill(e.begin(), e.end(), L);

  FermionField Xi(e, 1, 0, nt());
  FermionField Back(e, 1, 0, nt());
  FermionField Old(e, 1, 0, nt());
  fft::fft<FermionField,fft::abc> ft(Xi);
  for (FermionField::iterator it=Xi.begin();it!=Xi.end();++it)
    randomize(*it);
  Old=Xi;
  ft.execute(Xi,Back,fft::x2p);

  // {
  //   FermionField::iterator it1=Xi.begin();
  //   FermionField::iterator it2=Old.begin();
  //   do {
  //     Fermion X(*it1 - *it2);
  //     ASSERT_TRUE(max(X)<1e-14);
  //   } while ( (++it1)!=Back.end() && (++it2)!=Xi.end() );
  // }

  ft.execute(Back,Back,fft::p2x);

  FermionField::iterator it1=Back.begin();
  FermionField::iterator it2=Old.begin();
  do {
    Fermion X(*it1 - *it2);
    ASSERT_TRUE(max(X)<1e-14);
  } while ( (++it1)!=Back.end() && (++it2)!=Old.end() );

}

TEST(fft, inplaceABC){
  // lattice setup
  // generate an array for to store the lattice extents
  geometry::Geometry<DIM>::extents_t e;
  // we want a L = 4 lattice
  std::fill(e.begin(), e.end(), L);

  FermionField Xi(e, 1, 0, nt());
  FermionField Old(e, 1, 0, nt());
  fft::fft<FermionField,fft::abc> ft(Xi);
  for (FermionField::iterator it=Xi.begin();it!=Xi.end();++it)
    randomize(*it);
  Old=Xi;
  ft.execute(fft::x2p);
  ft.execute(fft::p2x);

  FermionField::iterator it1=Xi.begin();
  FermionField::iterator it2=Old.begin();
  do {
    Fermion X(*it1 - *it2);
    ASSERT_TRUE(max(X)<1e-14);
  } while ( (++it1)!=Xi.end() && (++it2)!=Old.end() );
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}



#endif //TEST_FFT_H
