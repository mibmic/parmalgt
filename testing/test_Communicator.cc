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
const int L = 10;


template <class Field_t>
struct cfld {
  cfld(const int& value) : val(value) { }
  static const int dim = Field_t::dim;
  void operator()(Field_t& F, const pt::Point<dim>& n) const {
    F[n] = val;
  }
  static const int n_cb = 0;
  int val;
};


template <class Field_t>
struct tagfld {
  tagfld(const int& value) : val(value) { }
  static const int dim = Field_t::dim;
  void operator()(Field_t& F, const pt::Point<dim>& n) const {
    F[n] = 1000*val+int(n);
  }
  static const int n_cb = 0;
  int val;
};





TEST(Comunicator, Buffer){
  typedef fields::LocalField<int, DIM> intField;
  typedef intField::raw_pt raw_pt_t;
  typedef std::vector<int> vec_t;
  intField::extents_t e;
  intField::neighbors_t nn;
  std::fill(e.begin(), e.end(), L);
  intField A(e,1,0,nn);
  cfld<intField> c(A.get_communicator().rank()+1);
  A.apply_everywhere(c);
  A.apply_everywhere(c);

  ASSERT_EQ((A.get_communicator().send_buff()[0].first).size(), e[1]*e[2]*e[3] );
  ASSERT_EQ((A.get_communicator().send_buff()[3].first).size(), e[1]*e[2] );
  
  const int np = A.get_communicator().numprocs();

  typedef geometry::ZeroBndIterator ZIterator;
  typedef geometry::TBndIterator TIterator;
  ZIterator i(A.mk_point(raw_pt_t{0,0,0,0}), e);
  ZIterator i1(A.mk_point(raw_pt_t{0,0,0,1}), e);
  TIterator f1(A.mk_point(raw_pt_t{0,0,0,e[DIM-1]-2}), e);
  TIterator f(A.mk_point(raw_pt_t{0,0,0,e[DIM-1]-1}), e);
  
  for( int rank = 0; rank < np; ++rank) {
    // test send buffer    
    if( A.get_communicator().rank() == rank) {
      int expected = A.get_communicator().rank()+1;
      for( std::vector<int>::const_iterator it = A.get_communicator().send_buff()[3].first.begin();
	   it != A.get_communicator().send_buff()[3].first.end(); ++it)
	ASSERT_EQ(*it, expected);
      for( std::vector<int>::const_iterator it = A.get_communicator().send_buff()[3].second.begin();
	   it != A.get_communicator().send_buff()[3].second.end(); ++it)
	ASSERT_EQ(*it, expected);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}




TEST(Comunicator, SendRecv){
  typedef fields::LocalField<int, DIM> intField;
  typedef typename intField::raw_pt raw_pt_t;
  typedef typename std::vector<int> vec_t;
  intField::extents_t e;
  intField::neighbors_t nn;
  std::fill(e.begin(), e.end(), L);
  intField A(e,1,0,nn);
  cfld<intField> c(A.get_communicator().rank()+1);

  const int numprocs = A.get_communicator().numprocs();
  typedef typename std::vector<int>::const_iterator it_t;

  // initialize+wrong communications
  A.apply_everywhere(c);
  // initialize+communicate
  A.apply_everywhere(c);

  for( int rank = 0; rank < numprocs; ++rank) {
    // test send buffer    
    if( A.get_communicator().rank() == rank) {
      for( it_t it = A.get_communicator().recv_buff()[3].second.begin();
	   it != A.get_communicator().recv_buff()[3].second.end(); ++it)
	ASSERT_EQ(*it, (A.get_communicator().nb()[3].second+1));
    }
    if( A.get_communicator().rank() == rank) {
      for( it_t it = A.get_communicator().recv_buff()[3].first.begin();
	   it != A.get_communicator().recv_buff()[3].first.end(); ++it)
	ASSERT_EQ(*it, (A.get_communicator().nb()[3].first+1));
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

TEST(Comunicator, Unbuffer){
  typedef fields::LocalField<int, DIM> intField;
  typedef typename intField::raw_pt raw_pt_t;
  typedef typename std::vector<int> vec_t;
  intField::extents_t e;
  intField::neighbors_t nn;
  std::fill(e.begin(), e.end(), L);
  intField A(e,1,0,nn);

  typedef geometry::ZeroBndIterator ZIterator;
  typedef geometry::TBndIterator TIterator;
  ZIterator i(A.mk_point(raw_pt_t{0,0,0,0}), e);
  TIterator f(A.mk_point(raw_pt_t{0,0,0,e[DIM-1]-1}), e);

  cfld<intField> c(A.get_communicator().rank()+1);
  A.apply_everywhere(c);
  A.apply_everywhere(c);
  
  const int numprocs = A.get_communicator().numprocs();
  for( int rank = 0; rank < numprocs; ++rank)
    do {
      ASSERT_EQ(A[*i],(A.get_communicator().nb()[3].first+1));
      ASSERT_EQ(A[*f],(A.get_communicator().nb()[3].second+1));
    } while ( (++f).is_good() && (++i).is_good() );

  MPI_Barrier(MPI_COMM_WORLD);
}



TEST(Communicator,tag){
  typedef fields::LocalField<int, DIM> intField;
  typedef typename intField::raw_pt raw_pt_t;
  typedef typename std::vector<int> vec_t;
  intField::extents_t e;
  intField::neighbors_t nn;
  std::fill(e.begin(), e.end(), L);
  intField A(e,1,0,nn);
  tagfld<intField> tag(A.get_communicator().rank()+1);
  A.apply_everywhere(tag);
  A.apply_everywhere(tag);

  const int np = A.get_communicator().numprocs();
  const int rk = A.get_communicator().rank();

  typedef geometry::ZeroBndIterator ZIterator;
  typedef geometry::TBndIterator TIterator;
  ZIterator i(A.mk_point(raw_pt_t{0,0,0,0}), e);
  ZIterator i1(A.mk_point(raw_pt_t{0,0,0,1}), e);
  TIterator f1(A.mk_point(raw_pt_t{0,0,0,e[3]-2}), e);
  TIterator f(A.mk_point(raw_pt_t{0,0,0,e[3]-1}), e);
 
  const int first = A.get_communicator().nb()[3].first+1;
  const int second = A.get_communicator().nb()[3].second+1;

  for( int rank = 0; rank < np; ++rank) {
    if( rank == rk ) {
	do {
	  ASSERT_EQ(A[*i], (1000*first +int(*f1)) );
	  ASSERT_EQ(A[*f], (1000*second+int(*i1)) );
	} while ( (++i).is_good() && (++i1).is_good() && (++f).is_good() && (++f1).is_good() );
	std::cout << "\n";
      }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}



TEST(Communicator,Gather){
  typedef fields::LocalField<int, DIM> intField;
  intField::extents_t e;
  intField::neighbors_t nn;
  std::fill(e.begin(), e.end(), L);
  intField A(e,1,0,nn);


  const int np = A.get_communicator().numprocs();
  const int rk = A.get_communicator().rank();

  comm::Reduce<int> reduce(rk+1);
  int result = reduce();
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "reduce = " << np*(np+1)/2 << std::endl;;
  MPI_Barrier(MPI_COMM_WORLD);
  if( rk == 0 )
    ASSERT_EQ(result, np*(np+1)/2 );
  MPI_Barrier(MPI_COMM_WORLD);

}


// TEST(Comunicator, Finalize){
//   MPI_Barrier(MPI_COMM_WORLD);
//   comm::Communicator<fields::LocalField<int, 4> >::finalize();
// }


int main(int argc, char **argv) {
  comm::Communicator<fields::LocalField<int, DIM> >::init(argc,argv);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


#endif //TEST_COMMUNICATOR_H
