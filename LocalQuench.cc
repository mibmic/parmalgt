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

// some shorthands
typedef bgf::AbelianBgf Bgf_t; // background field
typedef BGptSU3<Bgf_t, ORD> ptSU3; // group variables
typedef ptt::PtMatrix<ORD> ptsu3; // algebra variables
typedef BGptGluon<bgf::AbelianBgf, ORD, DIM> ptGluon; // gluon
typedef pt::Point<DIM> Point;
typedef pt::Direction<DIM> Direction;

// shorthand for gluon field
typedef fields::LocalField<ptGluon, DIM> GluonField;
typedef GluonField::neighbors_t nt;


//
// Make aliases for the Kernels ...
//

// ... for the gauge update/fixing ...

#ifdef IMP_ACT // do we want an improved aciton?
// 1x1, and 2x1 staples
typedef kernels::StapleReKernel<Bgf_t, ORD, DIM> StK;
// we need these to implement the imrovement at the boundary
// NOTE however, that they have to be applied at t=1 and T=t-1
typedef kernels::LWProcessA<Bgf_t, ORD, DIM> PrAK;
typedef kernels::LWProcessB<Bgf_t, ORD, DIM> PrBK;
typedef kernels::TrivialPreProcess<Bgf_t, ORD, DIM> PrTK;
// workaround for template typedef
template <class PR> struct GUK {
  typedef  kernels::GaugeUpdateKernel <Bgf_t, ORD, DIM, StK, PR> type;
};
#else
typedef kernels::StapleSqKernel<Bgf_t, ORD, DIM> StK;
typedef kernels::TrivialPreProcess<Bgf_t, ORD, DIM> PrK;
typedef kernels::GaugeUpdateKernel <Bgf_t, ORD, DIM, StK, PrK> 
        GaugeUpdateKernel;
#endif
typedef kernels::ZeroModeSubtractionKernel<Bgf_t, ORD, DIM> ZeroModeSubtractionKernel;
typedef kernels::GaugeFixingKernel<GF_MODE, Bgf_t, ORD, DIM> GaugeFixingKernel;


// ... to set the background field ...
typedef kernels::SetBgfKernel<Bgf_t, ORD, DIM> SetBgfKernel;

// ... and for the measurements ...
typedef kernels::PlaqLowerKernel<Bgf_t, ORD, DIM> PlaqLowerKernel;
typedef kernels::PlaqUpperKernel<Bgf_t, ORD, DIM> PlaqUpperKernel;
typedef kernels::PlaqSpatialKernel<Bgf_t, ORD, DIM> PlaqSpatialKernel;
typedef kernels::MeasureNormKernel<Bgf_t, ORD, DIM> MeasureNormKernel;
typedef kernels::GammaUpperKernel<Bgf_t, ORD, DIM> GammaUpperKernel;
typedef kernels::GammaLowerKernel<Bgf_t, ORD, DIM> GammaLowerKernel;
typedef kernels::UdagUKernel<Bgf_t, ORD, DIM> UdagUKernel;
typedef kernels::TemporalPlaqKernel<Bgf_t, ORD, DIM> TemporalPlaqKernel;
typedef kernels::PlaqKernel<Bgf_t, ORD, DIM> PlaqKernel;
typedef kernels::GFMeasKernel<Bgf_t, ORD, DIM> GFMeasKernel;
typedef kernels::GFApplyKernel<Bgf_t, ORD, DIM> GFApplyKernel;

// ... and for the checkpointing.
typedef kernels::FileWriterKernel<Bgf_t, ORD, DIM> FileWriterKernel;
typedef kernels::FileReaderKernel<Bgf_t, ORD, DIM> FileReaderKernel;

// Our measurement
void measure(GluonField &U, const std::string& rep_str){

  GammaUpperKernel Gu(L);
  GammaLowerKernel Gl(L);
  U.apply_on_timeslice(Gu, T-1);
  U.apply_on_timeslice(Gl, 0);
  
  // Evaluate Gamma'a
  ptSU3 tmp = Gu.val + Gl.val;
  io::write_file<ptSU3, ORD>(tmp, tmp.bgf().Tr() , "Gp" + rep_str + ".bindat");
  
  // Norm of the Gauge Field
  MeasureNormKernel m;
  U.apply_everywhere(m);
  m.reduce();
  io::write_file(m.norm[0], "Norm" + rep_str + ".bindat");

}

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


int main(int argc, char *argv[]) {
#ifdef IMP_ACT
  //TODO: CROSS CHECK THESE
  StK::weights[0] = 5./3;
  StK::weights[1] = -1./12;
#else
  StK::weights[0] = 1.;
#endif
  signal(SIGUSR1, kill_handler);
  signal(SIGUSR2, kill_handler);
  signal(SIGXCPU, kill_handler);
  int rank;
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::string rank_str = "." + to_string(rank);
#else
  std::string rank_str = "";
#endif
  ////////////////////////////////////////////////////////////////////
  // read the parameters
  uparam::Param p;
  p.read("input" + rank_str);
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
  T = L-s;
  // also write the number of space-time dimensions
  // and perturbative order to the parameters, to
  // make sure they are written in the .info file 
  // for the configurations stored on disk
  p["NDIM"] = to_string(DIM);
  p["ORD"] = to_string(ORD);
  ////////////////////////////////////////////////////////////////////
  //
  // timing stuff
  typedef std::map<std::string, Timer> tmap;
  tmap timings;
  ////////////////////////////////////////////////////////////////////
  //
  // random number generators
  srand(atoi(p["seed"].c_str()));
#ifdef IMP_ACT
  GUK<PrAK>::type::rands.resize(L*L*L*(T+1));
  GUK<PrBK>::type::rands.resize(L*L*L*(T+1));
  GUK<PrTK>::type::rands.resize(L*L*L*(T+1));
  for (int i = 0; i < L*L*L*(T+1); ++i){
    GUK<PrAK>::type::rands[i].init(rand());
    GUK<PrBK>::type::rands[i].init(rand());
    GUK<PrTK>::type::rands[i].init(rand());
  }
#else
  GaugeUpdateKernel::rands.resize(L*L*L*(T+1));
  for (int i = 0; i < L*L*L*(T+1); ++i)
    GaugeUpdateKernel::rands[i].init(rand());
#endif
  ////////////////////////////////////////////////////////////////////
  //
  // lattice setup
  // generate an array for to store the lattice extents
  geometry::Geometry<DIM>::extents_t e;
  // we want a L = 4 lattice
  std::fill(e.begin(), e.end(), L);
  // for SF boundary: set the time extend to T + 1
  e[0] = T + 1;
  // we will have just one field
  GluonField U(e, 1, 0, nt());
  // initialize background field get method
  bgf::get_abelian_bgf(0, 0, T, L, s);
  ////////////////////////////////////////////////////////////////////
  //
  // initialzie the background field of U or read config
  if ( p["read"] == "none" )
    for (int t = 0; t <= T; ++t){
      SetBgfKernel f(t);
      U.apply_on_timeslice(f, t);
    }
  else {
    FileReaderKernel fr(p);
    U.apply_everywhere_serial(fr);
  }
  ////////////////////////////////////////////////////////////////////
  //
  // start the simulation
  for (int i_ = 1; i_ <= NRUN && !soft_kill; ++i_){
    if (! (i_ % MEAS_FREQ) ) {
      timings["measurements"].start();
      measure(U, rank_str);
      timings["measurements"].stop();
    }
    ////////////////////////////////////////////////////////
    //
    //  gauge update
#ifdef IMP_ACT
    // make vector of 'tirvially' pre-processed gauge update kernels
    std::vector<GUK<PrTK>::type> gut;
    for (Direction mu; mu.is_good(); ++mu)
      gut.push_back(GUK<PrTK>::type(mu, taug));
    // 1) temporal links at t=0 and t = T -1
    //    - modify c_0 -> c_0 + 2c_1
    StK::weights[0] += 2*StK::weights[1];
    //    - apply 'tirvially' pre-processed gauge update kernel
    U.apply_on_timeslice(gut[0], 0);
    U.apply_on_timeslice(gut[0], T-1);
    //    - set c_0 back to proper value
    StK::weights[0] = 5./3;
    // 2) Use 'special' GU kernels for spatial plaquettes at t=1 and T-1
    for (Direction k(1); k.is_good(); ++k){
      GUK<PrAK>::type gua (k, taug);
      GUK<PrBK>::type gub (k, taug);
      // There's a bug here
      // comment the next four lines if you don't want the program to crash!
      U.apply_on_timeslice(gua, 1); // <- CRASH HERE
      U.apply_on_timeslice(gub, 1);
      U.apply_on_timeslice(gua, T-1);
      U.apply_on_timeslice(gub, T-1);
    }
    // 3) Business as usual for t = 2,...,T-2, all directions and t = 1, mu = 0
    for (int t = 2; t <= T-2; ++t)
      for (Direction mu; mu.is_good(); ++mu)
        U.apply_on_timeslice(gut[mu], t);
    U.apply_on_timeslice(gut[0], 1);
#else
    std::vector<GaugeUpdateKernel> gu;
    for (Direction mu; mu.is_good(); ++mu)
      gu.push_back(GaugeUpdateKernel(mu, taug));
    timings["Gauge Update"].start();
    // for x_0 = 0 update the temporal direction only
    U.apply_on_timeslice(gu[0], 0);
    // for x_0 != 0 update all directions
    for (int t = 1; t < T; ++t)
      for (Direction mu; mu.is_good(); ++mu)
        U.apply_on_timeslice(gu[mu], t);
    timings["Gauge Update"].stop();
#endif
    ////////////////////////////////////////////////////////
    //
    //  gauge fixing
    GaugeFixingKernel gf(alpha);

    timings["Gauge Fixing"].start();

    GFMeasKernel gfm;
    U.apply_on_timeslice(gfm, 0);
    GFApplyKernel gfa(gfm.val, alpha, L);
    U.apply_on_timeslice(gfa, 0);

    for (int t = 1; t < T; ++t)
      U.apply_on_timeslice(gf, t);
    timings["Gauge Fixing"].stop();

  } // end main for loop
  // write the gauge configuration
  if ( p["write"] != "none"){
    FileWriterKernel fw(p);
    U.apply_everywhere_serial(fw);
  }
  // write out timings
  of.open(("run.info"+rank_str).c_str(), std::ios::app);
  of << "Timings:\n";
  for (tmap::const_iterator i = timings.begin(); i != timings.end();
       ++i){
    io::pretty_print(i->first, i->second.t, "s", of);
  }
  io::pretty_print("TOTAL", Timer::t_tot, "s", of);
  of.close();
  return 0;
}
