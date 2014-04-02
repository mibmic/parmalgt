#ifndef _TEST_COMMUNICATOR_H
#define _TEST_COMMUNICATOR_H
#include <gtest/gtest.h>
#include <LocalField.hpp>
#include <newQCDpt.h>
#include <cstdlib>
#include <Helper.h>
#include <Kernels.hpp>

// space-time dimensions
const int DIM = 4;
// perturbative order
const int ORD = 2;
const int L = 16;


////////////////////////////////////
//
// ptSU3 communications

typedef BGptSU3<bgf::AbelianBgf, ORD> ptSU3;
typedef BGptGluon<bgf::AbelianBgf, ORD, 4> ptGluon;

void fill(ptSU3& M,int x,int dir) {
  for(int i=0;i<9;++i)
    M[0][i] = complex(x,i+10*dir);
}

template<class Field_t>
struct GluonFill {
  static const int n_cb = 0;
  GluonFill(const int& value) : val(value) { }
  void operator()(Field_t& F, const pt::Point<DIM>& n) const {
    for(pt::Direction<DIM> mu(0);mu.is_good();++mu)
      fill(F[n][mu],10000*val+int(n),int(mu));
  }
  int val;
};

std::ostream& operator<<(std::ostream& os, const complex& z) {
  std::cout << "(" << z.real()
	    << "," << z.imag()
	    << ")";
  return os;
}
std::ostream& operator<<(std::ostream& os, const sun::SU<3>& M) {
  for(int r=0;r<3;++r) {
    for(int c=0;c<3;++c)
      std::cout << M(r,c) << " ";
    std::cout << std::endl;
  }
  return os;
}
bool cmp(const sun::SU<3>& first,const sun::SU<3>& second) {
  for(int r=0;r<3;++r)
    for(int c=0;c<3;++c)
      if( (first(r,c).real() != second(r,c).real()) || 
	  (first(r,c).real() != second(r,c).real())  ) return false;
  return true;
}


TEST(Comunicator, SendRecvSU3){
  typedef fields::LocalField<ptGluon, DIM> GluonField;
  typedef typename GluonField::raw_pt raw_pt_t;
  typedef typename std::vector<ptGluon> vec_t;
  typedef typename std::vector<ptGluon>::const_iterator it_t;

  GluonField::extents_t e;
  GluonField::neighbors_t nn;
  std::fill(e.begin(), e.end(), L);
  GluonField U(e,1,0,nn);

  const int numprocs = U.get_communicator().numprocs();
  const int rank = U.get_communicator().rank();

  GluonFill<GluonField> c(rank+1);

  U.apply_everywhere(c);
  U.apply_everywhere(c);

  typedef geometry::ZeroBndIterator ZIterator;
  typedef geometry::TBndIterator TIterator;
  ZIterator i(U.mk_point(raw_pt_t{0,0,0,0}), e);
  ZIterator i1(U.mk_point(raw_pt_t{0,0,0,1}), e);
  TIterator f1(U.mk_point(raw_pt_t{0,0,0,e[3]-2}), e);
  TIterator f(U.mk_point(raw_pt_t{0,0,0,e[3]-1}), e);

  ptSU3 dn,up;
  const int prev = U.get_communicator().nb()[3].first+1;
  const int next = U.get_communicator().nb()[3].second+1;

  for( int irank = 0; irank < numprocs; ++irank) {

    if( U.get_communicator().rank() == irank)
      do {
	for(pt::Direction<DIM> mu(0);mu.is_good();++mu) {
	  fill(dn,10000*prev+(int)(*f1),int(mu));
	  fill(up,10000*next+(int)(*i1),int(mu));
	  ASSERT_TRUE( cmp(U[*f][mu][0],up[0]) );
	}
      } while ( (++i).is_good() && (++i1).is_good() &&
		(++f).is_good() && (++f1).is_good() );
    MPI_Barrier(MPI_COMM_WORLD);
  }

}


int main(int argc, char **argv) {
  comm::Communicator<fields::LocalField<ptGluon, DIM> >::init(argc,argv);
  //comm::Communicator<fields::LocalField<ptGluon, 4> >::init(argc,argv);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


#endif //TEST_COMMUNICATOR_H
