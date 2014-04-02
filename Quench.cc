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
#include <Kernels/generic/Plaquette.hpp>
#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
#endif
#include <signal.h>
#include <util.hpp>

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
// gauge improvement coefficient c1
const double c_1 =  -0.331;

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
//typedef bgf::ScalarBgf Bgf_t; // background field
typedef bgf::AbelianBgf Bgf_t; // background field
typedef BGptSU3<Bgf_t, ORD> ptSU3; // group variables
typedef ptt::PtMatrix<ORD> ptsu3; // algebra variables
typedef BGptGluon<Bgf_t, ORD, DIM> ptGluon; // gluon
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
typedef kernels::StapleReKernel<GluonField> StK;
// we need these to implement the imrovement at the boundary
// NOTE however, that they have to be applied at t=1 and T=t-1
typedef kernels::LWProcessA<GluonField> PrAK;
typedef kernels::LWProcessB<GluonField> PrBK;
typedef kernels::TrivialPreProcess<GluonField> PrTK;
// workaround for template typedef
template <class PR> struct GUK {
  typedef  kernels::GaugeUpdateKernel <GluonField, StK, PR> type;
};
#endif
typedef kernels::ZeroModeSubtractionKernel<GluonField> ZeroModeSubtractionKernel;

// ... to set the background field ...
typedef kernels::SetBgfKernel<GluonField> SetBgfKernel;

// ... and for the measurements ...
typedef kernels::MeasureNormKernel<GluonField> MeasureNormKernel;
typedef kernels::GammaUpperKernel<GluonField, kernels::init_helper_gamma> GammaUpperKernel;
typedef kernels::GammaLowerKernel<GluonField, kernels::init_helper_gamma> GammaLowerKernel;
typedef kernels::GammaUpperKernel<GluonField, kernels::init_helper_vbar> VbarUpperKernel;
typedef kernels::GammaLowerKernel<GluonField, kernels::init_helper_vbar> VbarLowerKernel;
typedef kernels::UdagUKernel<GluonField> UdagUKernel;
typedef plaq::Temporal<GluonField> TemporalPlaqKernel;
typedef kernels::PlaqKernel<GluonField> PlaqKernel;

// ... and for the checkpointing.
typedef kernels::FileWriterKernel<GluonField> FileWriterKernel;
typedef kernels::FileReaderKernel<GluonField> FileReaderKernel;

// Our measurement...

// Stuff we want to measure in any case
void measure_common(GluonField &U, const std::string& rep_str){
  //  Norm of the Gauge Field
  MeasureNormKernel m;
  Norm<ORD+1> res;

  U.apply_everywhere(m).reduce(res);

  // for(int rk=0;rk<comm::Communicator<GluonField>::numprocs_;++rk) {
  //   if( comm::Communicator<GluonField>::rank_ == rk ) {
  //     typename array_t<double, ORD+1>::Type::const_iterator it = res.begin();
  //     std::cout << "rank: " << rk << ":\t";
  //     do {
  // 	std::cout << *it << "\t";
  //     } while( (++it) != res.end() );
  //     std::cout << std::endl;
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  io::write_file(res,"Norm.bindat");
  // io::write_file(res,"Norm" + rep_str + ".bindat");
}

// Stuff that makes sense only for an Abelian background field.
void measure(GluonField &U, const std::string& rep_str,
             const bgf::AbelianBgf&){

  //PlaqKernel P;
  //io::write_file(U.apply_everywhere(P).val, "Plaq" + rep_str + ".bindat");

#ifndef HIGHER_ORDER_INT

  //TemporalPlaqKernel tp;
  //U.apply_on_timeslice(tp, 0);
  //U.apply_on_timeslice(tp, T-1);
  //

#ifndef IMP_ACT
  GammaUpperKernel Gu(L);
  GammaLowerKernel Gl(L);
#else
  GammaUpperKernel::weights imp_coeff;
  imp_coeff[0] = 1. - 8.*c_1;
  imp_coeff[1] = c_1;
  GammaUpperKernel Gu(L,imp_coeff);
  GammaLowerKernel Gl(L,imp_coeff);
#endif

  //  Evaluate Gamma'
  ptSU3 tmp;
  tmp = U.apply_on_timeslice(Gu, T-1).val
    + U.apply_on_timeslice(Gl, 0).val;
  
  io::write_file<ptSU3, ORD>(tmp, tmp.bgf().Tr() ,
			     "Gp.bindat");
  
  // io::write_file<ptSU3, ORD>(tmp, tmp.bgf().Tr() ,
  // 			     "Gp" + rep_str + ".bindat");
  
  
  // Evaluate F*Gamma'
  //  std::vector<Cplx> g = tmp.trace(), f = tp.val.trace(), gf(ORD+1, 0);
  //  for (int i = 0; i <= ORD; ++i)
  //    for (int j = 0; j <= i; ++j)
  //      gf[i] += g[j] * f[i - j];
  //
  //  io::write_file(gf, "FGamm" + rep_str + ".bindat");
  //
  //  VbarUpperKernel Vu(L);
  //  VbarLowerKernel Vl(L);
  //
  //  // Evaluate vbar
  //  tmp = U.apply_on_timeslice(Vu, T-1).val
  //    - U.apply_on_timeslice(Vl, 0).val;
  //  io::write_file<ptSU3, ORD>(tmp, tmp.bgf().Tr() ,
//                             "Vbar" + rep_str + ".bindat");
//
//  io::write_ptSUN(tp.val*tmp, "Fvbar" + rep_str + ".bindat");
#endif
  measure_common(U, rep_str);
}

// Stuff that makes sense only for a scalar background field.
void measure(GluonField &U, const std::string& rep_str,
             const bgf::ScalarBgf&){
  measure_common(U, rep_str);
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
  StK::weights[0] = 1. - 8.*c_1;
  StK::weights[1] = c_1;
#endif
  signal(SIGUSR1, kill_handler);
  signal(SIGUSR2, kill_handler);
  signal(SIGXCPU, kill_handler);
  signal(SIGINT, kill_handler);

  ////////////////////////////////////////////////////////////////////
  //
  // initialize MPI communicator
#ifdef USE_MPI
  comm::Communicator<GluonField>::init(argc,argv);
  std::string rank_str =  "." + to_string(comm::Communicator<GluonField>::rank_);
#else
  std::string rank_str = "";
#endif

  ////////////////////////////////////////////////////////////////////
  // read the parameters
  uparam::Param p;
  //  p.read("input" + rank_str);
  p.read("input");
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
  srand(atoi(p["seed"].c_str())+comm::Communicator<GluonField>::rank_*time(NULL));
#ifdef IMP_ACT
  GUK<PrAK>::type::rands.resize(L*L*L*(T+1));
  GUK<PrBK>::type::rands.resize(L*L*L*(T+1));
  GUK<PrTK>::type::rands.resize(L*L*L*(T+1));
  for (int i = 0; i < L*L*L*(T+1); ++i){
    GUK<PrAK>::type::rands[i].init(rand());
    GUK<PrBK>::type::rands[i].init(rand());
    GUK<PrTK>::type::rands[i].init(rand());
  }
#endif


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
  // for SF boundary: set the time extend to T + 1
  e[0] = T + 1;
  // initialize background field get method
  bgf::get_abelian_bgf(0, 0, T, L, s);
  // we will have just one field
  GluonField U(e, 1, 0, nt());

  MPI_Barrier(MPI_COMM_WORLD);
  ////////////////////////////////////////////////////////////////////
  //
  // initialzie the background field of U or read config
  if ( p["read"] == "none" )
    for (int t = 0; t <= T; ++t){
      SetBgfKernel f(t);
      U.apply_on_timeslice(f, t);
      U.apply_on_timeslice(f, t);
    }
  else {
    p.set("read", p["read"] + rank_str);
    FileReaderKernel fr(p);
    U.apply_everywhere(fr);
    U.apply_everywhere(fr);
  }

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

#ifdef IMP_ACT
    // In the case of an improved gauge action, we proceed with the
    // update according to choice "B" in Aoki et al., hep-lat/9808007
    // make vector of 'tirvially' pre-processed gauge update kernels
    std::vector<GUK<PrTK>::type> gut;
    for (Direction mu; mu.is_good(); ++mu)
      gut.push_back(GUK<PrTK>::type(mu, taug));
    timings["Gauge Update"].start();
    // 1) Use 'special' GU kernels for spatial plaquettes at t=1 and
    //    T-1, re reason for this is that the rectangular plaquettes
    //    with two links on the boundary have a weight of 3/2 c_1.
    for (Direction k(1); k.is_good(); ++k){
      GUK<PrAK>::type gua (k, taug);
      GUK<PrBK>::type gub (k, taug);
      U.apply_on_timeslice(gua, 1);
      U.apply_on_timeslice(gub, T-1);
    }
    // 2) Business as usual for t = 2,...,T-2, all directions and 
    //    t = 0,1 and T-1 for mu = 0
    for (int t = 2; t <= T-2; ++t)
      for (Direction mu; mu.is_good(); ++mu)
        U.apply_on_timeslice(gut[mu], t);
    U.apply_on_timeslice(gut[0], 0);
    U.apply_on_timeslice(gut[0], 1);
    U.apply_on_timeslice(gut[0], T-1);
    timings["Gauge Update"].stop();
#else
    timings["Gauge Update"].start();
    meth::gu::RK2_update(U, taug);
    timings["Gauge Update"].stop();
#endif
    ////////////////////////////////////////////////////////
    //
    //  gauge fixing
    timings["Gauge Fixing"].start();
    meth::gf::sf_gauge_fixing(U, alpha);
    timings["Gauge Fixing"].stop();

  } // end main for loop
  // write the gauge configuration
  if ( p["write"] != "none"){
    p.set("write", p["write"] + rank_str);
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

#ifdef USE_MPI
  comm::Communicator<GluonField>::finalize();
#endif  
  
  return 0;
}
