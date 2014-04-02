#include <LocalField.hpp>
#include <Background.h>
#include <Geometry.hpp>
#include <algorithm>
#include <newQCDpt.h>
#include <mpi.h>
#include <iostream>
#include <Methods.hpp>

#include <Communicator.hpp>

const int DIM = 4;
const int ORD = 6;

typedef BGptSU3<bgf::AbelianBgf, ORD> ptSU3;
typedef BGptGluon<bgf::AbelianBgf, ORD, DIM> ptGluon;
typedef fields::LocalField<ptGluon, DIM> GluonField;
typedef GluonField::neighbors_t nt;


// helper function to measure the trace of the field

struct MeasTrace {
  std::vector<double> traces;
  MeasTrace() : traces(ORD + 1, 0) { }
  static const int n_cb = 0;
  void operator()(GluonField& U,
                  const pt::Point<DIM>& n){
    for (pt::Direction<DIM> mu; mu.is_good(); ++mu){
      traces[0] += U[n][mu].bgf().Tr().real();
      for (int i = 0; i < ORD; ++i) 
        traces[i+1] += U[n][mu][i].tr().real();
    }
  }
};

// helper function to conveniently initialize the
// array of neighbors in four dimensions

inline nt neighbors(const int& tu, 
                    const int& td,
                    const int& xu,
                    const int& xd,
                    const int& yu,
                    const int& yd,
                    const int& zu,
                    const int& zd){
  nt result;
  result[0].first = tu;
  result[0].second = td;
  result[1].first = xu;
  result[1].second = xd;
  result[2].first = yu;
  result[2].second = yd;
  result[3].first = zu;
  result[3].second = zd;
  return result;
}


int main(int argc, char *argv[]) {
  // generate an array for to store the lattice extents
  geometry::Geometry<DIM>::extents_t e;
  // we want a L = 4 lattice
  std::fill(e.begin(), e.end(), 4);
  // // initialize parallel enviornment
  // int rank , numprocs;
  // MPI_Init(&argc, &argv);
  // MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // if (numprocs != 2){
  //   std::cout << "Please start the test program with two processes!\n";
  //   return 1;
  // }
  // // tell the process which are the neighbors
  // int zup = (rank == numprocs - 1) ? 0 : rank + 1;
  // int zdown = (!rank) ? numprocs - 1 : rank - 1;
  // nt n = neighbors(0,0,0,0,0,0, zup, zdown);
  const int buff_sz = 1;
  comm::Communicator<GluonField> comm(e);
  comm.init(argc,argv);
  GluonField U(e, 1);  
  // GluonField U(e, 1, rank, n);  
  meth::gu::detail::rand_gen_<GluonField> r(U);
  if (comm.rank() == 0){
    r.update();
    // U.test_send_fwd_z();
    std::cout << "real parts traces sent:\n";
    MeasTrace m;
    // U.measure_on_slice_with_bnd(m, pt::Direction<DIM>(3), 4);
    U.apply_everywhere(m);
    for (int i = 0; i <= ORD; ++i)
      std::cout << i << "\t" << m.traces[i] << std::endl;
  }
  else {
    // U.test_rec_bkw_z();
    std::cout << "real parts of traces received:\n";
    MeasTrace m;
    //    U.measure_on_slice_with_bnd(m, pt::Direction<DIM>(3), 4);
    U.apply_everywhere(m);
    for (int i = 0; i <= ORD; ++i)
      std::cout << i << "\t" << m.traces[i] << std::endl;
  }
  
  MPI_Finalize();
  return 0;
}
