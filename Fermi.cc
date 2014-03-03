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

#include <Timer.hpp>
#include <Methods.hpp>


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
const int Nf = 2;


// max number of iterations
const int COUNT_MAX = 100;


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


//
// Make aliases for the Kernels ...
//

// ... to set the background field ...
typedef kernels::SetBgfKernel<GluonField> SetBgfKernel;



// helper function to feed int, double etc. to the parameters
template <typename T> 
std::string to_string(const T& x){
  std::stringstream sts;
  sts << x;
  return sts.str();
}



// // BCGstab
// template<int DIM>
// void inverter( GluonField& U, ScalarFermionField& x, double& m )
// {

//   const double inv_prec = 1e-5;
//   Cplx alpha, beta, omega;
//   Cplx nr, nr1;
//   Cplx norm_r, norm_r1;

//   //  b.randomize();  // ok, here preconditioning or something else...

//   ScalarFermionField b (x), r0(x), Ax(x);
//   WilsonTLKernel apply_from_x(U, x, m );
  
//   Ax.apply_everywhere(apply_from_x);
//   r0 = b - Ax;

//   ScalarFermionField p(x), p0(r0), r(r0), r0star(r), s(x);
//   WilsonTLKernel apply_from_p(U, p, m );
//   WilsonTLKernel apply_from_s(U, s, m );

//   ScalarFermionField Ap(x), As(x);

//   int count = 0;

//   nr1 = (r * r0star);
//   while( count < COUNT_MAX )
//     {
  
//       ++count;
//       Ap.apply_everywhere(apply_from_p);

//       nr = nr1;

//       Ap.apply_everywhere(apply_from_p);

//       alpha = nr / ( Ap * r0star );
//       s     = r - Ap * alpha;
      
//       As.apply_everywhere(apply_from_s);

//       omega = ( As * s ) / ( As * As );
//       x     = x + ( (p * alpha) + (s * omega) );
//       r     = s - As * omega;
//       nr1   = (r * r0star);
//       beta  = (nr1 / nr) * (alpha / omega);
//       p     = r + (p - Ap * omega) * beta;
      
//       std::cout << count << "\t" << r*r << "\n";
//     }


// }





int main(int argc, char *argv[]) {

  signal(SIGUSR1, kill_handler);
  signal(SIGUSR2, kill_handler);
  signal(SIGXCPU, kill_handler);
  int rank;

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
  // we want a L = 4 lattice
  std::fill(e.begin(), e.end(), L);
  // for SF boundary: set the time extend to T + 1
  e[0] = T;
  // we will have just one field
  GluonField U(e, 1, 0, nt());

  ScalarFermionField Xi(e, 1, 0, nt());
  FermionField Psi;

  meth::fu::gaussian_source<ScalarFermionField> R(Xi);
  for(int i=0;i<ORD;++i)
    Psi.push_back(ScalarFermionField(e, 1, 0, nt()));
  
  std::vector<double> mass(ORD);
  mass = {4., 0., -2.6057, 0., -4.2925, 0., -11.78};

  // R.update();
  // Xi = R();

  srand(1234);

  std::for_each(Xi.begin(),Xi.end(), 
		[](ScalarFermionField::data_t& i) { 
		  double r = 1./static_cast<double>(RAND_MAX);
		  for(Direction mu(0);mu.is_good();++mu)
		    for(int a=0;a<3;++a)
		      i[mu][a] = r*complex(rand(),rand()); });

  int ss=0;
  for(GluonField::iterator i=U.begin();i!=U.end();++i) { 
    for(Direction mu(0);mu.is_good();++mu)
      for(int oo=0;oo<ORD;++oo)
	for(int a=0;a<9;++a)
	  (*i)[mu][oo][a] = complex( .05+ss+.1+.1*mu-.1*oo+a,
				   .5*ss-mu*(a+1)+(oo+1));
    ++ss;
  }

  meth::fu::invert<GluonField,ScalarFermionField,0>(U,Xi,Psi,mass);
  meth::fu::invert<GluonField,ScalarFermionField,0>(U,Xi,Psi,mass);
  meth::fu::invert<GluonField,ScalarFermionField,0>(U,Xi,Psi,mass);
  std::for_each(Psi[2].begin(),Psi[2].end(), 
   		[](const ScalarFermionField::data_t& i) { std::cout << i;  });
  
  std::vector<kernels::FermionicUpdateKernel<GluonField,ScalarFermionField> > fk;
  for(Direction mu(0);mu.is_good();++mu)
    fk.push_back(kernels::FermionicUpdateKernel<GluonField,ScalarFermionField>(Xi,Psi,mu,taug*Nf));

  for(Direction mu(0);mu.is_good();++mu)
    U.apply_everywhere(fk[mu]);

  return 0;
}





