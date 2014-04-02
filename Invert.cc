#include <LocalField.hpp>
#include <Background.h>
#include <Geometry.hpp>
#include <algorithm>
#include <newQCDpt.h>
#include <map>
#include <iostream>
#include <fstream>
#include <Kernels.hpp>
#include <uparam.hpp>
#include <stdlib.h>
#include <IO.hpp>
#include <newMyQCD.h>
#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
#endif
#include <signal.h>

#ifdef USE_MPI
#include <mpi.h>
#include <sstream>
#endif

#include <Inverter.hpp>

bool soft_kill = false;
int got_signal = 100;
void kill_handler(int s){
       soft_kill = true;
       got_signal = s;
       std::cout << "INITIATE KILL SEQUENCE\n"; 
}

// space-time dimensions
const int DIM = 4;
// perturbative order
const int ORD = 4;

// max number of iterations
const int COUNT_MAX = 1000;


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
//
// PARAMETERS to be read later

// lattice size
int L;
int T;
// s parameter for staggered
int s;
// total number of gauge updates
int NRUN;
// frequency of measurements
int MEAS_FREQ;
// testing gauge fixing option -- DO NOT TOUCH!
const int GF_MODE = 1;
// integration step and gauge fixing parameter
double taug;
double alpha;

// some short-hands
typedef bgf::ScalarBgf Bgf_t; // background field
// typedef bgf::AbelianBgf Bgf_t; // background field
typedef BGptSU3<Bgf_t, ORD> ptSU3; // group variables
typedef ptt::PtMatrix<ORD> ptsu3; // algebra variables
typedef BGptGluon<Bgf_t, ORD, DIM> ptGluon; // gluon
typedef pt::Point<DIM> Point;
typedef pt::Direction<DIM> Direction;

// shorthand for gluon field
typedef fields::LocalField<ptGluon, DIM> GluonField;
typedef GluonField::neighbors_t nt;

// shorthand for fermion field
typedef SpinColor<4> Fermion;
typedef fields::LocalField< Fermion , DIM> ScalarFermionField;
typedef std::vector<ScalarFermionField> FermionField;

// ... to set the background field ...
typedef kernels::SetBgfKernel<GluonField> SetBgfKernel;

// timing

struct Timer {
  double t, tmp;
  static double t_tot;
  Timer () : t(0.0) { };
  void start() { 
#ifdef _OPENMP
    tmp = omp_get_wtime(); 
#else
    tmp = clock();
#endif

  }
  void stop() { 
#ifdef _OPENMP
    double elapsed = omp_get_wtime() - tmp; 
#else
    double elapsed = ((double)(clock() - tmp))/CLOCKS_PER_SEC;
#endif
    t += elapsed;
    t_tot += elapsed;
  }
};

double Timer::t_tot = 0.0;

// helper function to feed int, double etc. to the parameters
template <typename T> 
std::string to_string(const T& x){
  std::stringstream sts;
  sts << x;
  return sts.str();
}


template<typename Inv>
void invertQ( const ScalarFermionField& src, GluonField& U, FermionField& dest ) {
  
  std::cout << "#Fermion Operator Inversion" << std::endl;

  double m = .1;
  ScalarFermionField Mxi (src);
  meth::Dirac<GluonField,ScalarFermionField,
  	      kernels::TreeLevelWilsonKernel> applyD(src,U,m);

  applyD(Mxi);
  double tol = 1e-10;

  if( inverter::BiCGstab<GluonField, ScalarFermionField, 
      kernels::TreeLevelWilsonKernel, COUNT_MAX>(U,Mxi,dest[0],m,tol)){
    std::cout << "#Error: unable to converge in less than " 
	      << COUNT_MAX 
	      << " iterations." 
	      << std::endl;
  }
  if( inverter::cgs<GluonField, ScalarFermionField, 
      kernels::TreeLevelWilsonKernel,COUNT_MAX>(U,Mxi,dest[1],m,tol)) {
    std::cout << "#Error: unable to converge in less than " 
	      << COUNT_MAX 
	      << " iterations." 
	      << std::endl;
    }
  
}




int main(int argc, char *argv[]) {

  signal(SIGUSR1, kill_handler);
  signal(SIGUSR2, kill_handler);
  signal(SIGXCPU, kill_handler);
  int rank;

  ////////////////////////////////////////////////////////////////////
  // read the parameters
  uparam::Param p;
  p.read("input");
  L = atof(p["L"].c_str());
  s = atoi(p["s"].c_str());
  alpha = atof(p["alpha"].c_str());
  taug = atof(p["taug"].c_str());
  NRUN = atoi(p["NRUN"].c_str());
  MEAS_FREQ = atoi(p["MEAS_FREQ"].c_str());
  T = L-s;
  // also write the number of space-time dimensions
  // and perturbative order to the parameters, to
  // make sure they are written in the .info file 
  // for the configurations stored on disk
  p.set("NDIM", to_string(DIM));
  p.set("ORD", to_string(ORD));
  ////////////////////////////////////////////////////////////////////
  //
  // timing stuff
  typedef std::map<std::string, Timer> tmap;
  tmap timings;
  ////////////////////////////////////////////////////////////////////
  //
  // random number generators
  srand(atoi(p["seed"].c_str()));
  ////////////////////////////////////////////////////////////////////
  //
  // lattice setup
  // generate an array for to store the lattice extents
  geometry::Geometry<DIM>::extents_t e;
  // we want a L = 4 lattice
  std::fill(e.begin(), e.end(), L);
  // for toric boundary: set the time extend to T
  e[0] = T ;

  // fields
  GluonField U(e, 1, 0, nt());
  // for (int t = 0; t < T; ++t){
  //   SetBgfKernel f(t);
  //   U.apply_on_timeslice(f, t);
  // }
  ScalarFermionField Xi  (e, 1, 0, nt());
  FermionField Psi;

  geometry::Geometry<DIM>::raw_pt_t coords={0,0,0,0};
  Point point = Xi.mk_point(coords);

  // source
  int n = 0;
  for ( ScalarFermionField::iterator it = Xi.begin(); it != Xi.end(); ++it, ++n )
    for( Direction mu(0); mu.is_good(); ++mu) 
      (*it)[mu][0] = Cplx(rand()/(double)RAND_MAX,rand()/(double)RAND_MAX);


  Psi.push_back(ScalarFermionField(e, 1, 0, nt()));
  Psi.push_back(ScalarFermionField(e, 1, 0, nt()));
  invertQ<int>( Xi, U, Psi );
  std::cout << (Xi - Psi[0])*(Xi - Psi[0]) << std::endl;
  std::cout << (Xi - Psi[1])*(Xi - Psi[1]) << std::endl;

  return 0;
}
