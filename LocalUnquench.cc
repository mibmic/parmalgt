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
#include <Methods.hpp>
#include <include/Kernels/generic/Plaquette.hpp>
#include <include/Kernels/generic/IO.hpp>
#include <signal.h>
#include <util.hpp>
#include <Timer.hpp>

#ifdef USE_MPI
#include <mpi.h>
#include <sstream>
#endif

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
const int ORD = 6;
// gauge improvement coefficient c1
const double c_1 =  -0.331;
// Number of fermion
const int Nf = 4;


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
// integration step and gauge fixing parameter
double taug;
double alpha;

// some short-hands
typedef bgf::ScalarBgf Bgf_t; // background field
//typedef bgf::TrivialBgf Bgf_t; // background field
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


//
// Make aliases for the Kernels ...
//

// ... for the gauge update/fixing ...

typedef kernels::StapleSqKernel<GluonField> StSqK;
typedef kernels::StapleReKernel<GluonField> StReK;
typedef StReK StK;

typedef kernels::ZeroModeSubtractionKernel<GluonField> ZeroModeSubtractionKernel;

// ... to set the background field ...
typedef kernels::SetBgfKernel<GluonField> SetBgfKernel;

// ... and for the measurements ...
typedef kernels::MeasureNormKernel<GluonField> MeasureNormKernel;
// typedef kernels::GammaUpperKernel<GluonField, kernels::init_helper_gamma> GammaUpperKernel;
// typedef kernels::GammaLowerKernel<GluonField, kernels::init_helper_gamma> GammaLowerKernel;
// typedef kernels::GammaUpperKernel<GluonField, kernels::init_helper_vbar> VbarUpperKernel;
// typedef kernels::GammaLowerKernel<GluonField, kernels::init_helper_vbar> VbarLowerKernel;
typedef kernels::UdagUKernel<GluonField> UdagUKernel;
typedef kernels::PlaqKernel<GluonField> PlaqKernel;

// ... and for the checkpointing.
typedef kernels::FileWriterKernel<GluonField> FileWriterKernel;
typedef kernels::FileReaderKernel<GluonField> FileReaderKernel;
typedef kernels::PRlgtReaderKernel<GluonField> PRlgtReaderKernel;

// Our measurement...

// Stuff we want to measure in any case
void measure_common(GluonField &U, const std::string& rep_str){
  // Norm of the Gauge Field
  MeasureNormKernel m;
  array_t<double, ORD+1>::Type other;
  io::write_file(U.apply_everywhere(m).reduce(other),
                 "Norm" + rep_str + ".bindat");
  PlaqKernel P;
  ptSU3 tmp = U.apply_everywhere(P).reduce()/U.vol();
  io::write_file<ptSU3, ORD>(tmp, tmp.bgf().Tr() , "Plaq" + rep_str + ".bindat");
  
}


// Stuff that makes sense only for a scalar background field.
void measure(GluonField &U, const std::string& rep_str,
             const bgf::ScalarBgf&){
  measure_common(U, rep_str);
}


// helper function to feed int, double etc. to the parameters
template <typename T> 
std::string to_string(const T& x){
  std::stringstream sts;
  sts << x;
  return sts.str();
}


int main(int argc, char *argv[]) {

#ifdef IMP_ACT
  StK::weights[0] = 1. - 8.*c_1;
  StK::weights[1] = c_1;
#endif

  signal(SIGUSR1, kill_handler);
  signal(SIGUSR2, kill_handler);
  signal(SIGXCPU, kill_handler);
  signal(SIGINT, kill_handler);

  // ////////////////////////////////////////////////////////////////////
  // //
  // // initialize MPI communicator
  // comm::Communicator<GluonField>::init(argc,argv);

  std::string rank_str = "";

  ////////////////////////////////////////////////////////////////////
  // read the parameters
  uparam::Param p;
  //  p.read("input" + rank_str);
  p.read("input");
  // also write the number of space-time dimensions
  // and perturbative order to the parameters, to
  // make sure they are written in the .info file 
  // for the configurations stored on disk
  p.set("NDIM", to_string(DIM));
  p.set("ORD",  to_string(ORD));
  std::ofstream of(("run.info"+rank_str).c_str(), std::ios::app);
  of << "INPUT PARAMETERS:\n";
  p.print(of);
  of.close();
  L = atof(p["L"].c_str());
  s = atoi(p["s"].c_str());
  alpha = atof(p["alpha"].c_str());
  taug = atof(p["taug"].c_str());
  NRUN = atoi(p["NRUN"].c_str());
  MEAS_FREQ = atoi(p["MEAS_FREQ"].c_str());
  T = L-s; // anisotropic lattice
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
  // we want a T x L^3 lattice
  std::fill(e.begin(), e.end(), L);
#ifdef USE_MPI
  e[DIM-1] = L/comm::Communicator<GluonField>::numprocs_ + 2;
#endif

  e[0] = T;
  // initialize background field get method
  //  bgf::get_abelian_bgf(0, 0, T, L, s);
  // we will have just one field
  GluonField U(e, 1, 0, nt());
  // U.comm.init(argc,argv);
  ////////////////////////////////////////////////////////////////////
  //
  // initialzie the background field of U or read config
  if ( p["read"] == "none" )
    for (int t = 0; t <= T; ++t){
      SetBgfKernel f(t);
      U.apply_on_timeslice(f, t);
    }
  else {
// #ifndef READ_FROM_PRLGT
//     FileReaderKernel fr(p);
// #else    
    std::ifstream is(p["read"],std::ifstream::binary);
    PRlgtReaderKernel fr(is);
    //#endif
    U.apply_everywhere(fr);
  }

  ScalarFermionField Xi(e, 1, 0, nt());
  FermionField Psi;
  for(int i=0;i<ORD;++i)
    Psi.push_back(ScalarFermionField(e, 1, 0, nt()));
  std::vector<double> mass(ORD);
  mass = {4., 0., -2.6057, 0., -4.2925, 0., -11.78};
  ////////////////////////////////////////////////////////////////////
  //
  // initialzie the random gaussian souce for fermions
  meth::fu::gaussian_source<ScalarFermionField> R(Xi);

  ////////////////////////////////////////////////////////////////////
  //
  // start the simulation
  int i_;
  for (i_ = 1; i_ <= NRUN && !soft_kill; ++i_){
    if (! (i_ % MEAS_FREQ) ) {
      timings["measurements"].start();
      measure(U, rank_str, Bgf_t());
      timings["measurements"].stop();
    }
    ////////////////////////////////////////////////////////
    //
    //  gauge update
    timings["Gauge Update"].start();
    meth::gu::RK1_update<GluonField, StK>(U, taug);
    timings["Gauge Update"].stop();

    ////////////////////////////////////////////////////////
    //
    //  gauge fixing
    timings["Gauge Fixing"].start();
    meth::gf::gauge_fixing(U, alpha);
    timings["Gauge Fixing"].stop();


    ////////////////////////////////////////////////////////
    //
    //  fermionic update
    timings["Fermionic Update"].start();
    R.update();
    Xi = R();
    meth::fu::invert<GluonField,ScalarFermionField,0>(U,Xi,Psi,mass);
    std::vector<kernels::FermionicUpdateKernel<GluonField,
    					       ScalarFermionField> > fk;
    for(Direction mu(0);mu.is_good();++mu)
      fk.push_back(kernels::FermionicUpdateKernel<GluonField,
    		   ScalarFermionField>(Xi,Psi,mu,taug*Nf));
    for(Direction mu(0);mu.is_good();++mu)
      U.apply_everywhere(fk[mu]);
    timings["Fermionic Update"].stop();

  } // end main for loop
  // write the gauge configuration
  if ( p["write"] != "none"){
    FileWriterKernel fw(p);
    U.apply_everywhere(fw);
  }
  // write out timings
  of.open(("run.info"+rank_str).c_str(), std::ios::app);
  of << "Timings:\n";
  for (tmap::const_iterator i = timings.begin(); i != timings.end();
       ++i){
    util::pretty_print(i->first, i->second.t, "s", of);
  }
  util::pretty_print("TOTAL", Timer::t_tot, "s", of);
  if (soft_kill)
    util::pretty_print("actual # of configs", i_, "", of);
  of.close();

  return 0;
}
