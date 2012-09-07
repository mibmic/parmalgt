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

bool soft_kill = false;
int got_signal = 100;
void kill_handler(int s){
       soft_kill = true;
       got_signal = s;
       std::cout << "INITIATE KILL SEQUENCE\n"; 
}
struct tri_coord{
  int Nm1;
  explicit tri_coord(int N) : Nm1(N-1) { }
  int operator()(int i, int j){
    return i * Nm1 - i * (i - 1) / 2 + j - i - 1;
  }
};

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
typedef kernels::GaugeUpdateKernel<Bgf_t, ORD, DIM> GaugeUpdateKernel;
typedef kernels::ZeroModeSubtractionKernel<Bgf_t, ORD, DIM> ZeroModeSubtractionKernel;
typedef kernels::GaugeFixingKernel<GF_MODE, Bgf_t, ORD, DIM> GaugeFixingKernel;

// ... to set the background field ...
typedef kernels::SetBgfKernel<Bgf_t, ORD, DIM> SetBgfKernel;

// ... and for the measurements ...
typedef kernels::PlaqLowerKernel<Bgf_t, ORD, DIM> PlaqLowerKernel;
typedef kernels::PlaqUpperKernel<Bgf_t, ORD, DIM> PlaqUpperKernel;
typedef kernels::PlaqSpatialKernel<Bgf_t, ORD, DIM> PlaqSpatialKernel;
typedef kernels::MeasureNormKernel<Bgf_t, ORD, DIM> MeasureNormKernel;

// ... and for the checkpointing.
typedef kernels::FileWriterKernel<Bgf_t, ORD, DIM> FileWriterKernel;
typedef kernels::FileReaderKernel<Bgf_t, ORD, DIM> FileReaderKernel;

// Our measurement
void measure(GluonField &U){
  std::vector<double> nor(ORD*2*3 + 2);
  SU3 dC; // derivative of C w.r.t. eta
  dC(0,0) = Cplx(0, -2./L);
  dC(1,1) = Cplx(0, 1./L);
  dC(2,2) = Cplx(0, 1./L);
  SU3 d_dC; // derivative of C w.r.t. eta and nu
  d_dC(0,0) = Cplx(0, 0);
  d_dC(1,1) = Cplx(0, 2./L);
  d_dC(2,2) = Cplx(0, -2./L);
  
  double pio3 = std::atan(1.)*4./3.;
  SU3 C, Cp;
  C(0,0)  = Cplx(0, -pio3/L);
  C(2,2)  = Cplx(0, pio3/L);
  Cp(0,0) = Cplx(0, -3.*pio3/L);
  Cp(1,1) = Cplx(0, 2.*pio3/L);
  Cp(2,2) = Cplx(0, pio3/L);

  // shorthand for V3
  int V3 = L*L*L;

  // testing: Measure the spatial plaq at one time slice

  PlaqSpatialKernel Pnew;
  U.apply_on_slice_with_bnd(Pnew, Direction(0), 1);
  io::write_file<ptSU3, ORD>(Pnew.val, Pnew.val.bgf().Tr(), "new_plaq.bindat");

  PlaqLowerKernel Pl; // Plaquettes U_{0,k} at t = 0
  PlaqUpperKernel Pu; // Plaquettes U_{0,k} at t = T - 1
  U.apply_on_slice_with_bnd(Pl, Direction(0), 0);
  U.apply_on_slice_with_bnd(Pu, Direction(0), T-1);

  PlaqUpperKernel Pm; // Plaquettes U_{0,k} at t = 1, ..., T - 2
  for (int t = 1; t < T - 1; ++t)
    U.apply_on_slice_with_bnd(Pm, Direction(0), t);

  PlaqSpatialKernel Ps; // Spatial plaq. at t = 1, ..., T-1
  for (int t = 1; t < T; ++t)
    U.apply_on_slice_with_bnd(Ps, Direction(0), t);

  // Evaluate Gamma'
  //ptSU3 tmp = Pl.val + Pu.val;
  ptSU3 t1 = Pl.val, t2 = Pu.val;
  std::for_each(t1.begin(), t1.end(), pta::mul(dC*C));
  std::for_each(t2.begin(), t2.end(), pta::mul(dag(Cp)*dC));
  ptSU3 tmp = t1 + t2;
  Cplx tree = tmp.bgf().ApplyFromLeft(dC).Tr();
  io::write_file<ptSU3, ORD>(tmp, tree, "Gp.bindat");

  // Evaluate v
  tmp = Pl.val + Pu.val;
  std::for_each(tmp.begin(), tmp.end(), pta::mul(d_dC));
  tree = tmp.bgf().ApplyFromLeft(d_dC).Tr();
  io::write_file<ptSU3, ORD>(tmp, tree, "v.bindat");

  // Evaluate bd_plaq (nomenclature as in MILC code)
  tmp = (Pl.val + Pu.val) / 6 / V3;
  tree = tmp.bgf().Tr();
  io::write_file<ptSU3, ORD>(tmp, tree, "bd_plaq.bindat");

  // Evaluate st_plaq
  tmp = (Pl.val + Pu.val + Pm.val) / 3 / V3 / T;
  tree = tmp.bgf().Tr();
  io::write_file<ptSU3, ORD>(tmp, tree, "st_plaq.bindat");

  // Eval. ss_plaq
  tmp = Ps.val / 3 / V3 / (T-1);
  tree = tmp.bgf().Tr();
  io::write_file<ptSU3, ORD>(tmp, tree, "ss_plaq.bindat");
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
  signal(SIGUSR1, kill_handler);
  signal(SIGUSR2, kill_handler);
  signal(SIGXCPU, kill_handler);
  ////////////////////////////////////////////////////////////////////
  // read the parameters
  uparam::Param p;
  p.read("input");
  std::cout << "INPUT PARAMETERS:\n";
  p.print();
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
  GaugeUpdateKernel::rands.resize(L*L*L*(T+1));
  for (int i = 0; i < L*L*L*(T+1); ++i)
    GaugeUpdateKernel::rands[i].init(rand());
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
    U.apply_everywhere(fr);
  }
  ////////////////////////////////////////////////////////////////////
  //
  // start the simulation
  for (int i_ = 1; i_ <= NRUN && !soft_kill; ++i_){
    if (! (i_ % MEAS_FREQ) ) {
      std::cout << i_;
      MeasureNormKernel m(ORD + 1);
      U.apply_everywhere(m);
      for (int i = 0 ; i < m.norm.size(); ++i)
        std::cout << " " << m.norm[i];
      std::cout << "\n";
      timings["measurements"].start();
      measure(U);
      timings["measurements"].stop();
    }
    ////////////////////////////////////////////////////////
    //
    //  gauge update
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
    ////////////////////////////////////////////////////////
    //
    //  zero mode subtraction
    std::vector<ZeroModeSubtractionKernel> z;
    double vinv = 1./L/L/L/L; // inverse volume
    for (Direction mu; mu.is_good(); ++mu){
      gu[mu].reduce();
      z.push_back(ZeroModeSubtractionKernel(mu, gu[mu].M[0]*vinv));
    }
    // for x_0 = 0 update the temporal direction only (as above)
    timings["Zero Mode Subtraction"].start();
    U.apply_on_timeslice(z[0], 0);
    // for x_0 != 0 update all directions (as above)
    for (int t = 1; t < T; ++t)
      for (Direction mu; mu.is_good(); ++mu)
        U.apply_on_timeslice(z[mu], t);
    timings["Zero Mode Subtraction"].stop();
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Double check the observables after gauge fixing
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Tue Sep  4 15:52:47 2012
     if (! (i_ % MEAS_FREQ) && false ) {
      std::cout << i_;
      MeasureNormKernel m(ORD + 1);
      U.apply_everywhere(m);
      for (int i = 0 ; i < m.norm.size(); ++i)
        std::cout << " " << m.norm[i];
      std::cout << "\n";
      timings["measurements"].start();
      measure(U);
      timings["measurements"].stop();
    }


    ////////////////////////////////////////////////////////
    //
    //  gauge fixing

  // store all plaquettes
  std::vector<std::vector<std::vector<Cplx> > > 
    Plaqs(L*L*L*(T+1), 
          std::vector<std::vector<Cplx> >(6, std::vector<Cplx>(ORD+1)));
  tri_coord f(4);

  for (Point n = U.g_begin(), e = U.g_end(); n != e; ++n)
    for (Direction mu; mu.is_good(); ++mu)
      for (Direction nu(mu + 1); nu.is_good(); ++nu)
        Plaqs.at(n).at(f(mu,nu)) = 
          (U[n][mu] * U[n+mu][nu] 
           * dag(U[n+nu][mu]) * dag(U[n][nu])).trace();

    GaugeFixingKernel gf(alpha);
    timings["Gauge Fixing"].start();
    for (int t = 1; t < T; ++t)
      U.apply_on_timeslice(gf, t);
    timings["Gauge Fixing"].stop();

    for (Point n = U.g_begin(), e = U.g_end(); n != e; ++n)
      for (Direction mu; mu.is_good(); ++mu)
        for (Direction nu(mu + 1); nu.is_good(); ++nu){
          std::vector<Cplx> Pnew =  (U[n][mu] * U[n+mu][nu] 
                                     * dag(U[n+nu][mu]) 
                                     * dag(U[n][nu])).trace();
          for (int i = 0; i < ORD+1; ++i){
            if ( fabs(Plaqs.at(n).at(f(mu,nu)).at(i).re - Pnew[i].re)
                 > 1e-15 || 
                 fabs(Plaqs.at(n).at(f(mu,nu)).at(i).im - Pnew[i].im)
                 > 1e-15){
              std::cout.precision(16);
              std::cout << Plaqs.at(n).at(f(mu,nu)).at(i) << std::endl;
              std::cout << Pnew[i] << std::endl;
              std::cout << "at n = " << n << std::endl;
              //throw std::exception();
            }
          }
        }


  } // end main for loop
  // write the gauge configuration
  if ( p["write"] != "none"){
    FileWriterKernel fw(p);
    U.apply_everywhere(fw);
  }
  // write out timings
  std::cout << "Timings:\n";
  for (tmap::const_iterator i = timings.begin(); i != timings.end();
       ++i){
    io::pretty_print(i->first, i->second.t, "s");
  }
  io::pretty_print("TOTAL", Timer::t_tot, "s");
  return 0;
}
