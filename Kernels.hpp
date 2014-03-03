#ifndef _KERNELS_H
#define _KERNELS_H
#include <Point.hpp>
#include <LocalField.hpp>
#include <PtTypes.hpp>
#include <Background.h>
#include <newQCDpt.h>
#include <newMyQCD.h>

#include <valarray>
#include <numeric>

#include <IO.hpp>
#include <uparam.hpp>
#include <Types.h>
#ifdef _OPENMP
#include <omp.h>
#else
namespace kernels {
  int omp_get_max_threads() { return 1; }
  int omp_get_thread_num() { return 0; }
}
#endif

#define FLD_INFO(F)					\
  typedef typename std_types<F>::ptGluon_t ptGluon;	\
  typedef typename std_types<F>::ptSU3_t ptSU3;		\
  typedef typename std_types<F>::ptsu3_t ptsu3;		\
  typedef typename std_types<F>::bgf_t BGF;		\
  typedef typename std_types<F>::point_t Point;		\
  typedef typename std_types<F>::direction_t Direction;	\
  static const int ORD = std_types<F>::order;		\
  static const int DIM = std_types<F>::n_dim;

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///  Gamma Matrices.
///
///  Indices and values of all the Dirac gamma and their products
///
///  \author Michele Brambilla <mib.mic@gmail.com>
///  \date Fri Nov 02 12:06:15 2012
///  

namespace dirac {

  // Indices of Dirac Gamma matrices
  int gmuind[15][4]  = {
    {3,2,1,0}, //gm1
    {3,2,1,0},
    {2,3,0,1},
    {2,3,0,1},
    {0,1,2,3}, //gm5
    {3,2,1,0}, //gm51
    {3,2,1,0}, 
    {2,3,0,1},
    {2,3,0,1},
    {0,1,2,3}, // gm12
    {1,0,3,2},
    {1,0,3,2},
    {1,0,3,2},
    {1,0,3,2},
    {0,1,2,3},
  };

  // Values of Dirac Gamma matrices
  Cplx gmuval[15][4] = {
    {Cplx(0,-1),Cplx(0,-1),Cplx(0, 1),Cplx(0, 1)}, //gm1
    {       -1 ,        1 ,        1 ,       -1 },
    {Cplx(0,-1),Cplx(0, 1),Cplx(0, 1),Cplx(0,-1)},
    {        1 ,        1 ,        1 ,        1 },
    {        1 ,        1 ,       -1 ,       -1 }, // gm5
    {Cplx(0, 1),Cplx(0, 1),Cplx(0, 1),Cplx(0, 1)}, //gm51
    {        1 ,       -1 ,        1 ,       -1 },
    {Cplx(0,1),Cplx(0, -1),Cplx(0, 1),Cplx(0,-1)},
    {       -1 ,       -1 ,        1 ,        1 },
    {Cplx(0,1),Cplx(0, -1),Cplx(0, 1),Cplx(0,-1)}, //gm12
    {       -1 ,        1 ,       -1 ,        1 },
    {Cplx(0,-1),Cplx(0,-1),Cplx(0, 1),Cplx(0, 1)},
    {Cplx(0, 1),Cplx(0, 1),Cplx(0, 1),Cplx(0, 1)},
    {       -1 ,        1 ,        1 ,       -1 },
    {Cplx(0,-1),Cplx(0, 1),Cplx(0, 1),Cplx(0,-1)}
  };

  // Factors multiplying each Dirac Gamma matrices to get transposed
  // ones
  const double gmT[4] = {-1.0, 1.0, -1.0, 1.0};

}


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///  Kernels.
///
///  Kernels can be measurements or updates for lattice fields.
///
///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
///  \date Thu May 24 17:55:49 2012

namespace kernels {
  
  template <class Field_t> 
  struct base_types {
    typedef typename Field_t::data_t data_t;
    static const int order = data_t::order;
    static const int n_dim = Field_t::dim;
    typedef pt::Point<n_dim> point_t;
    typedef pt::Direction<n_dim> direction_t;
  };

  /// struct to extract information on the field from the field type.
  template <class Field_t>
  struct std_types {
    typedef typename Field_t::data_t ptGluon_t;
    typedef typename ptGluon_t::pt_su3_t ptSU3_t;
    typedef typename ptSU3_t::pt_matrix_t ptsu3_t;
    typedef typename ptSU3_t::pt_matrix_t::SU3_t SU3_t;
    typedef typename ptGluon_t::bgf_t bgf_t;
    static const int order = ptGluon_t::order;
    static const int n_dim = Field_t::dim;
    typedef pt::Point<n_dim> point_t;
    typedef pt::Direction<n_dim> direction_t;
  };

  template <class Field_t>
  struct StapleSqKernel {
    // collect info about the field
    FLD_INFO(Field_t);

    typedef typename array_t<ptSU3, 1>::Type ptsu3_array_t;

    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 1;

    ptsu3_array_t val;
    Direction mu;

    StapleSqKernel(const Direction& nu) : mu(nu) {  }

    void operator()(Field_t& U, const Point& n) {
      // std::cout << val[0][0] << "\n";
      val[0].zero();
      // std::cout << val[0][0] << "\n";
      for(Direction nu; nu.is_good(); ++nu)
        if(nu != mu){
          val[0] += U[n + mu][nu] *  dag(U[n][nu] * U[n + nu][mu]);
          val[0] += dag(U[n-nu][mu] * U[n+mu-nu][nu]) * U[n - nu][nu];
        }
      // Close the staple
      val[0] = U[n][mu] * val[0] ;
    }
    
    ptSU3& reduce() { 
      return val[0]; 
    }

  };

  template <class Bgf, class SUN>
  void inplace_add(SUN& s, const Bgf& b){
    for (int i = 0; i < SUN::size; ++i)
      for (int j = 0; j < SUN::size; ++j)
	s(i,j) += b(i,j);
  }
  template <class SUN>
  void inplace_add(SUN& s, const bgf::AbelianBgf& b){
    for (int i = 0; i < SUN::size; ++i)
      s(i,i) += b[i];
  }
  template <class SUN>
  void inplace_add(SUN& s, const bgf::ScalarBgf& b){
    for (int i = 0; i < SUN::size; ++i)
      s(i,i) += b.val();
  }
  template <class SUN>
  void inplace_add(SUN& s, const bgf::TrivialBgf&){
    for (int i = 0; i < SUN::size; ++i)
      s(i,i) += 1;
  }
  


  ////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////
  ///
  /// Kernel to measure the staples for e.g. LW or Iwasaki aciton.
  ///
  /// Be careful when you use this to perform a gauge update. As was
  /// pointed out by Aoki et al. in hep-lat/9808007, generally the
  /// e.o.m do not hold if one naively uses the rectangular
  /// plaquettes. For a perturbative calculation, the weights of the
  /// rectangular loops at the boundary should be adjusted according
  /// to choice 'B' in the paper, c.f. eqns. (2.17) and (2.18).
  ///
  /// This is implemented in the pre- and post-processing kernels
  /// LWProcessA and LWProcessB.
  ///
  /// Commenty by D.H., Oct. 26, 2012

  template <class Field_t>
  struct StapleReKernel {

    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    typedef typename array_t<ptSU3, 2>::Type ptsu3_array_t;    
    typedef typename array_t<double, 2>::Type weight_array_t;    
    
    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 2;

    ptsu3_array_t val;
    Direction mu;
    static weight_array_t weights;
    
    StapleReKernel(const Direction& nu) : mu(nu) {  }

    ptSU3 two_by_one(Field_t& U, const Point& n, const Direction& nu){
      return U[n + mu][mu] * U[n + mu + mu][nu] * 
	dag(U[n][nu] * U[n+nu][mu] * U[n+nu+mu][mu])
	+ U[n+mu][mu] * dag( U[n-nu][mu] * 
			     U[n+mu-nu][mu] * U[n+mu+mu-nu][nu] ) * U[n-nu][nu]
	+ U[n+mu][nu] * dag( U[n-mu][nu] * 
			     U[n-mu+nu][mu] * U[n+nu][mu] ) * U[n-mu][mu]
	+ dag( U[n-nu-mu][mu] * U[n-nu][mu] * U[n-nu+mu][nu] ) * 
	U[n-nu-mu][nu] * U[n-mu][mu];
    }
    ptSU3 one_by_two(Field_t& U, const Point& n, const Direction& nu){
      return U[n + mu][nu] * U[n+mu+nu][nu] * 
	dag(U[n][nu] * U[n+nu][nu] * U[n+nu+nu][mu]) 
	+ dag( U[n-nu-nu][mu] * U[n-nu-nu+mu][nu] * U[n-nu+mu][nu] ) * 
	U[n-nu-nu][nu] * U[n-nu][nu];
    }
    ptSU3 one_by_one(Field_t& U, const Point& n, const Direction& nu){
      return U[n + mu][nu] *  dag(U[n][nu] * U[n + nu][mu])
	+ dag(U[n-nu][mu] * U[n+mu-nu][nu]) * U[n - nu][nu];
    }
    
    void operator()(Field_t& U, const Point& n) {      

      // is it better to use a foreach ?
      for( int i = 0; i < val.size(); ++i) {
	val[i].bgf() *= 0;
	val[i].zero();
      }

      for(Direction nu; nu.is_good(); ++nu)
        if(nu != mu) {
	  // 1x1 contribution
	  val[0] += one_by_one(U, n, nu);
	  // 2x1 contribution
	  val[1] += two_by_one(U, n, nu) + one_by_two(U, n, nu);
	}
      
      // Close the staple
      for( int i = 0; i < val.size(); ++i) {
	val[i] = U[n][mu] * val[i] ;
      }

    }
    ptSU3& reduce() { 
      val[0] *= weights[0];
      for( int i = 1; i < val.size(); ++i) val[0] += weights[i]*val[i] ;
      return val[0]; 
    }

  };

  template <class F>
  typename StapleReKernel<F>::weight_array_t StapleReKernel<F>::weights;

  template <class Field_t>
  struct TrivialPreProcess {
    static const int DIM = Field_t::dim;
    typedef typename std_types<Field_t>::point_t Point_t;
    typedef typename std_types<Field_t>::direction_t Direction_t;
    static void pre_process (Field_t& , const Point_t&, const Direction_t&) { }
    static void post_process (Field_t& , const Point_t&, const Direction_t& ) { }
  };

  // To be used at x with x_0 = a.
  // This modifies the background field, such that O(a) improvement
  // and the equations of motion hold at tree level.

  template <class Field_t>
  struct LWProcessA {
    static const int DIM = Field_t::dim;
    typedef typename std_types<Field_t>::point_t Point_t;
    typedef typename std_types<Field_t>::direction_t Direction_t;
    static void pre_process (Field_t& U, const Point_t& n, const Direction_t& k) { 
      static Direction_t t(0);
      U[n - t + k + k][t] *= 1.5;
      U[n - t - k][t] *= 1.5;
    }
    static void post_process (Field_t& U, const Point_t& n, const Direction_t& k) { 
      static Direction_t t(0);
      U[n - t + k + k][t] /= 1.5;
      U[n - t - k][t] /= 1.5;
    }
  };

  // To be used at x with x_0 = T - a.
  // This modifies the background field, such that O(a) improvement
  // and the equations of motion hold at tree level.
  template <class Field_t>
  struct LWProcessB {
    static const int DIM = Field_t::dim;
    typedef typename std_types<Field_t>::point_t Point_t;
    typedef typename std_types<Field_t>::direction_t Direction_t;
    static void pre_process (Field_t& U, const Point_t& n, const Direction_t& k) { 
      static Direction_t t(0);
      U[n - k][t] *= 1.5; // ...
      U[n + k + k][t] *= 1.5;
    }
    static void post_process (Field_t& U, const Point_t& n, const Direction_t& k) { 
      static Direction_t t(0);
      U[n - k][t] /= 1.5; // ...
      U[n + k + k][t] /= 1.5;
    }
  };


  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Kernel for the gauge update.
  ///
  ///  \tparam Field_t The type of field (I guess usually some sort of
  ///  gluon field) that the gauge update should be applied to.
  ///  \tparam StapleK_t The type of staple to use. This is employed
  ///  to implement e.g. improved gluon actions.
  ///  \tparam Process This must be a class that has two methods,
  ///  called pre_process and post_process. They are applied before
  ///  and after the gauge update is performed to be able to do things
  ///  like adjusting the weights of plaquettes at the boundary for
  ///  improved actions.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Thu May 24 17:47:43 2012

  template <class Field_t, 
	    class StapleK_t, class Process >
  struct GaugeUpdateKernel {

    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    typedef std::vector<Cplx>::iterator cpx_vec_it;
    typedef std::vector<std::vector<Cplx> >::iterator outer_cvec_it;

    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = StapleK_t::n_cb;    

    // zero momentum contribution
    std::vector<ptsu3> M;

    // for testing, c.f. below
    static std::vector<ranlxd::Rand> rands;
    Direction mu;
    
    double taug, stau;

    // on-the-fly plaquette measure
    std::vector<std::vector<Cplx> > plaq, pp;

    GaugeUpdateKernel(const Direction& nu, const double& t) :
      mu(nu), 
      M(omp_get_max_threads()), 
      taug(t), stau(sqrt(t)), 
      plaq(omp_get_max_threads(), std::vector<Cplx>(ORD+1)), 
      pp(omp_get_max_threads(), std::vector<Cplx>(ORD+1)) { }



    void operator()(Field_t& U, const Point& n) {
      ptSU3 W;

      // Make a Kernel to calculate and store the plaquette(s)
      StapleK_t st(mu); // maye make a vector of this a class member
      Process::pre_process(U,n,mu);
      st(U,n);
      Process::post_process(U,n,mu);
      
      pp[omp_get_thread_num()] = (st.val[0].trace()); // Save the 1x1 plaquette

      for(cpx_vec_it k = plaq[omp_get_thread_num()].begin(), 
	    j = pp[omp_get_thread_num()].begin(),
	    e = plaq[omp_get_thread_num()].end(); 
      	  k != e; ++k, ++j) *k += *j;

      ptsu3 tmp  = st.reduce().reH() * -taug;

      tmp[0] -= stau*sun::SU3rand(rands.at(n));
      U[n][mu] = exp<BGF, ORD>(tmp)*U[n][mu]; // back to SU3
      M[mu][omp_get_thread_num()] += get_q(U[n][mu]); // zero momentum contribution
    }
    
    void reduce(){
      std::for_each(M.begin()+1, M.end(), [&] (const ptsu3& i){ M[0] += i; } );
    }
    
  };

  

  template <class C, class P, class Q> std::vector<ranlxd::Rand>
  kernels::GaugeUpdateKernel<C,P,Q>::rands;

  template <class Field_t> struct RSU3Kernel {

    // collect info about the field
    static const int n_dim = Field_t::dim;
    typedef pt::Point<n_dim> Point;
    // Choose random number generator to use
    // Note that you have to change the same lines
    // in Methods.hpp

    // using RANLUX
    typedef std::vector<ranlxd::Rand> rand_vec_t;
    static rand_vec_t rands;
    static const int n_cb = 0;

    void operator()(Field_t& U, const Point& n) const {
      U[n] = sun::SU3rand(rands[n]);
    }
  };

  template <class C> typename kernels::RSU3Kernel<C>::rand_vec_t
  kernels::RSU3Kernel<C>::rands;

  template <class Field_t> struct RCVKernel {
    
    // collect info about the field
    static const int n_dim = Field_t::dim;
    typedef pt::Point<n_dim> Point;
    typedef typename base_types<Field_t>::direction_t Direction;
    
    // using RANLUX
    typedef std::vector<ranlxd::Rand> rand_vec_t;
    static rand_vec_t rands;
    static const int n_cb = 0;
    
    void operator()(Field_t& U, const Point& n) const {
      //      std::cout << "n = " << int(n) << std::endl;
      for(Direction mu(0);mu.is_good();++mu)
	U[n][mu] = sun::rand<3,ranlxd::Rand>(rands[int(n)]);
    }

  };
  
  template <class C> typename kernels::RCVKernel<C>::rand_vec_t
  kernels::RCVKernel<C>::rands;


  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Kernel for gauge fixing.
  ///
  ///  Note I implemented three different methods of which only the
  ///  third one seems to work fine at the moment.
  ///
  ///  \tparam METHOD  Gauge fixing mode. WARNING: Only mode 1 and 3
  ///  are tested and only 1 is fully trusted!
  ///  \tparam Field_t The kind of field to be used.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Thu May 24 17:49:09 2012
  ///

  template <int METHOD, class Field_t>
  class GaugeFixingKernel {
  public:
    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    explicit GaugeFixingKernel (const double& a) : alpha (a) { }
    void operator()(Field_t& U, const Point& n) const { 
      do_it(U, n, mode_selektor<METHOD>());
    }

    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 1;

  private:
    double alpha;
    template <int M> struct mode_selektor { };
    void do_it(Field_t& U, const Point& n, 
	       const mode_selektor<1>&) const {
      // exp version
      ptSU3 omega;
      for (Direction mu; mu.is_good(); ++mu)
	omega += U[n - mu][mu] - U[n][mu] ;
    
      ptSU3 Omega = exp<BGF, ORD>( alpha * omega.reH());
      ptSU3 OmegaDag = exp<BGF, ORD>( -alpha * omega.reH());
    
      for (Direction mu; mu.is_good(); ++mu){
	U[n][mu] = Omega * U[n][mu];
	U[n - mu][mu] *= OmegaDag;
      }
    }
    void do_it(Field_t& U, const Point& n,
	       const mode_selektor<2>&) const {
      ptSU3 omega;
      for (Direction mu; mu.is_good(); ++mu){
	omega += (U[n][mu] * dag(U[n][mu].bgf()) *
		  dag( U[n - mu][mu] ) * U[n - mu][mu].bgf());
      }
      ptSU3 Omega = exp<BGF, ORD>( alpha * omega.reH());
      ptSU3 OmegaDag = exp<BGF, ORD>( -alpha * omega.reH());
      for (Direction mu; mu.is_good(); ++mu){
	U[n][mu] = Omega * U[n][mu];
	U[n - mu][mu] *= OmegaDag;
      }
    }
    void do_it(Field_t& U, const Point& n,
	       const mode_selektor<3>&) const {
      ptSU3 omega;
      omega.zero();
      for (Direction mu; mu.is_good(); ++mu){
	ptSU3 Udag = dag(U[n - mu][mu]);
	BGF Vdag = U[n][mu].bgf().dag(), V = Udag.bgf().dag();
	omega += U[n][mu]*Vdag*Udag*V;
      }
      ptSU3 Omega = exp<BGF, ORD>( -alpha * omega.reH());
      ptSU3 OmegaDag = exp<BGF, ORD>( alpha * omega.reH());
      for (Direction mu; mu.is_good(); ++mu){
	U[n][mu] = Omega * U[n][mu];
	U[n - mu][mu] *= OmegaDag;
      }
    }

    void do_it(Field_t& U, const Point& n, 
	       const mode_selektor<4>&) const {
      // exp version
      ptSU3 omega;
      for (Direction mu; mu.is_good(); ++mu)
	omega += dag(U[n-mu][mu].bgf())*U[n - mu][mu] 
	  - U[n][mu]*dag(U[n][mu].bgf());
    
      ptSU3 Omega = exp<BGF, ORD>( alpha * omega.reH());
      ptSU3 OmegaDag = exp<BGF, ORD>( -alpha * omega.reH());
    
      for (Direction mu; mu.is_good(); ++mu){
	U[n][mu] = Omega * U[n][mu];
	U[n - mu][mu] *= OmegaDag;
      }
    }

  };
  
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Kernel taking care of zero mode subtraction.
  ///
  ///  \tparam BGF Background field to use.
  ///  \tparam ORD Perturbative order.
  ///  \tparam DIN Number of space-time dimensions.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Thu May 24 17:50:47 2012

  template <class Field_t>
  struct ZeroModeSubtractionKernel {
    
    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 0;

    Direction mu;
    ptsu3 M; // zero momentum contribution
    double norm;
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Constructor
    ///
    ///
    ///  \param nu Direction in which the functor will subtract the
    ///  zero modes.
    ///  \param N The sum over the zero modes to be subtracted.
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Thu May 24 17:51:17 2012
    ZeroModeSubtractionKernel(Field_t& U, 
			      const Direction& nu, const ptsu3& N) :
      mu(nu), M(N), norm(1./static_cast<double>(U.vol())) { }

    void operator()(Field_t& U, const Point& n) {
      U[n][mu] = exp<BGF,ORD>(get_q(U[n][mu])-norm*M);
    }
    
  };
  
  
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Kernel to set the background field to the usual Abelian one.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Thu May 24 17:52:27 2012
  template <class Field_t>
  struct SetBgfKernel {

    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 0;

    explicit SetBgfKernel(const int& t_in) : t(t_in) { }
    int t;
    void operator()(Field_t& U, const Point& n) const {
      impl(U, n, BGF());
    }

    void impl(Field_t& U, const Point& n, const bgf::AbelianBgf&) const {
      for (Direction mu; mu.is_good(); ++mu)
        U[n][mu].bgf() =  bgf::get_abelian_bgf(t, mu);
    }

    void impl(Field_t& U, const Point& n, const bgf::ScalarBgf&)
      const { }

  };


  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Kernel to measure the norm.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Thu May 24 17:52:56 2012
  template <class Field_t>
  struct MeasureNormKernel {

    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 0;
    std::vector<typename array_t<double, ORD+1>::Type> norm;

    explicit MeasureNormKernel() : norm(omp_get_max_threads()) { }

    void operator()(const Field_t& U, const Point& n) {
      std::vector<double> tmp = U[n].Norm();
      int i = omp_get_thread_num();
      for (int k = 0; k < ORD+1; ++k)
        norm[i][k] += tmp[k];
    }
    typename array_t<double, ORD+1>::Type reduce(){
      for (int i = 1, j = omp_get_max_threads(); i < j; ++i)
        for (int k = 0; k < ORD+1; ++k)
          norm[0][k] += norm[i][k];
      return norm[0];
    }
    typename array_t<double, ORD+1>::Type reduce(typename array_t<double, ORD+1>::Type& other){
      other = norm[0];
      for (int i = 1, j = omp_get_max_threads(); i < j; ++i)
        for (int k = 0; k < ORD+1; ++k) {
          norm[0][k] += norm[i][k];
	  other[k]   += norm[i][k];
	}
      return norm[0];
    }
    typename array_t<double, ORD+1>::Type reduce(Norm<ORD+1>& other){
      for (int k = 0; k < ORD+1; ++k)
	other[k] = norm[0][k];
      for (int i = 1, j = omp_get_max_threads(); i < j; ++i)
        for (int k = 0; k < ORD+1; ++k) {
          norm[0][k] += norm[i][k];
	  other[k] += norm[i][k];
	}
      return norm[0];
    }
  };

  // measures Udagger * U
  template <class Field_t>
  struct UdagUKernel {

    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 0;

    ptSU3 val;
    UdagUKernel () : val(bgf::zero<BGF>()) { }
    void operator()(Field_t& U, const Point& n){
      ptSU3 tmp(bgf::zero<BGF>());
      for (Direction mu; mu.is_good(); ++mu)
        tmp += dag(U[n][mu]) * U[n][mu];
#pragma omp critical
      val += tmp;
    }
  
  };

  template <class Field_t, class OutputIterator>
  struct Buffer {
    // collect info about the field
    static const int n_dim = Field_t::dim;
    typedef pt::Point<n_dim> point_t;
    typedef pt::Direction<n_dim> direction_t;

    // This may NOT be executed in parallel, so ...
    typedef void NoPar;
    
    // iterator to the positon where the next write goes
    OutputIterator oi;
    
    Buffer(OutputIterator o) : oi(o) { }

    void operator()(Field_t& U, const point_t& n) {
      *(oi++) = U[n];
    }
  };
  template <class Field_t, class InputIterator>
  struct Unbuffer {
    // collect info about the field
    static const int n_dim = Field_t::dim;
    typedef pt::Point<n_dim> point_t;
    typedef pt::Direction<n_dim> direction_t;

    // This may NOT be executed in parallel, so ...
    typedef void NoPar;
    
    // iterator to the positon where the next write goes
    InputIterator ii;
    
    Unbuffer(InputIterator i) : ii(i) { }

    void operator()(Field_t& U, const point_t& n){
      U[n] = *(ii++);
    }
  };

  // Measure the average paquette
  template <class Field_t>
  struct PlaqKernel {

    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;
    static const int nc  = ptSU3::SU3_t::size;
    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 0;
    
    const double norm = DIM*(DIM-1)*nc;

    std::vector<ptSU3> val;
    
    PlaqKernel () : val(omp_get_max_threads(),
			ptSU3(bgf::zero<BGF>())) { }
 
    void operator()(Field_t& U, const Point& n){
      ptSU3 tmp(bgf::zero<BGF>());
      for (Direction mu(0); mu.is_good(); ++mu)
        for (Direction nu(0); nu.is_good(); ++nu)
	  if( nu != mu )
	    tmp += U[n][mu] * U[n + mu][nu] * 
	      dag( U[n][nu] * U[n + nu][mu] );
      val[omp_get_thread_num()] += tmp;
    }

    const ptSU3& reduce() {
      for( auto it = val.begin()+1; it != val.end(); ++it) 
	val[0] += (*it);
      return (val[0] /= norm);
    }
    
  };
  
  // Kernel to construct the gauge fixing function at t = 0
  template <class Field_t>
  struct GFMeasKernel {

    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 0;

    ptsu3 val;

    void operator()(Field_t& U, const Point& n){
#pragma omp critical
      val += get_q(U[n][Direction(0)]);
    }
    
  };  
  // Kernel to execute the gauge fixing function at t = 0
  // (the one where we used U_0(\vec y, 0) to construct the gf 
  // function)
  template <class Field_t>
  struct GFApplyKernel {

    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    ptSU3 Omega, OmegaDag;
    
    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 0;

    GFApplyKernel (ptsu3 omega, const double& alpha,
		   const int& L) { 
      for (int r = 0; r < ORD; ++r)
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            if (i != j)
              omega[r](i,j) = 0;
      
      omega /= L*L*L;
      Omega = exp<BGF, ORD>( -alpha * omega.reH());
    }

    void operator()(Field_t& U, const Point& n){
      static Direction t(0);
      U[n][t] = Omega * U[n][t];
    }
    
  };
  template <class Field_t>
  struct GFApplyKernelTRBG {

    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    ptSU3 Omega, OmegaDag;
    
    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 0;

    GFApplyKernelTRBG (ptsu3 omega, const double& alpha,
		       const int& L) { 
      omega /= L*L*L;
      Omega = exp<BGF, ORD>( -alpha * omega.reH());
    }

    void operator()(Field_t& U, const Point& n){
      static Direction t(0);
      U[n][t] = Omega * U[n][t];
    }
    
  };

  // helper class to get around compile errors during initialization
  // of the kernels below ..
  template <class B> struct init_helper_gamma {
    void operator()(const int&, B&) const {}
  };
  template <> 
  struct init_helper_gamma<bgf::AbelianBgf> {
    void operator()(const int& L, bgf::AbelianBgf& Ctilde) const {
      Cplx ioL(0, 1./L);
      Ctilde[0] = -2.*ioL;
      Ctilde[1] = ioL;
      Ctilde[2] = ioL;
    }
  };
  template <class B> struct init_helper_vbar {
    void operator()(const int&, B&) const {}
  };
  template <> 
  struct init_helper_vbar<bgf::AbelianBgf> {
    void operator()(const int& L, bgf::AbelianBgf& Ctilde) const {
      Cplx ioL(0, 1./L);
      Ctilde[0] = 0;
      Ctilde[1] = 2.*ioL;
      Ctilde[2] = -2.*ioL;
    }
  };

#ifdef IMP_ACT
  // Measure Gamma at t = 0
  template <class Field_t, template <class C> class init_helper_t>
  struct GammaLowerKernel {

    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 0;

    // improvement coefficents
    typedef typename array_t<double, 2>::Type weights;
    weights c;

    ptSU3 val;
    // \tilde C = - [d_eta C]
    // at the t = 0 side, we have dagger(e^C) and hence an insertion
    // of \tilde C, since C itself is purely imaginary
    BGF Ctilde;
    explicit GammaLowerKernel (const int& L, const weights& c_) : val(bgf::zero<BGF>()), c(c_) { 
      init_helper_t<BGF>()(L, Ctilde);
    }
    void operator()(const Field_t& U, const Point& n){
      Direction t(0);
      ptSU3 tmp(bgf::zero<BGF>());
      for (Direction k(1); k.is_good(); ++k){
        // the 1x1 contribution
        tmp += (Ctilde * dag(U[n][k]) * U[n][t] * 
                U[n + t][k] * dag( U[n + k][t] )) * c[0];
        // the 2x1 contribution w\ two links @ the boundary
        // we actually have two contributions to the derivative form
        // each boundary link. However, we do not care because the
        // insertion of \tilde C commutes with the links. Hence, we
        // just multiply the weight with two, 
        //  w = 2 * (3/2) * c_1 = 3*(-1/12) = -1/4
        tmp += (Ctilde * dag(U[n][k]) * dag(U[n-k][k]) 
                * U[n-k][t] * U[n + t - k][k]
                * U[n + t][k] * dag( U[n + k][t] )) * 2. * (3./2.) * c[1];
        // the 1x2 contribution, with usual weight c_1
        tmp += (Ctilde * dag(U[n][k]) * U[n][t] 
                * U[n+t][t] * U[n+t+t][k] 
                * dag(U[n + k + t][t]) * dag(U[n + k][t])) * c[1];
      }
#pragma omp critical
      val += tmp;
    }
  };

  // Measure Gamma at t = T - a
  template <class Field_t, template <class C> class init_helper_t>
  struct GammaUpperKernel {

    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 0;

    // improvement coefficents
    typedef typename array_t<double, 2>::Type weights;
    weights c;

    ptSU3 val;
    // here, we need [d_eta C'], which is equal to -[d_eta C], hence
    // we can use Ctilde as above
    BGF Ctilde;
    explicit GammaUpperKernel (const int& L, const weights& c_) : val(bgf::zero<BGF>()), c(c_) { 
      init_helper_t<BGF>()(L, Ctilde);
    }
    void operator()(const Field_t& U, const Point& n){
      Direction t(0);
      ptSU3 tmp(bgf::zero<BGF>());
      for (Direction k(1); k.is_good(); ++k){
        // the 1x1 contribution
        tmp += (Ctilde * U[n + t][k] * dag(U[n + k][t])
                * dag(U[n][k]) * U[n][t]) * c[0];
        // the 2x1 contribution w\ two links @ the boundary
        // we actually have two contributions to the derivative form
        // each boundary link. However, we do not care because the
        // insertion of \tilde C commutes with the links. Hence, we
        // just multiply the weight with two, 
        //  w = 2 * (3/2) * c_1 = 3*(-1/12) = -1/4
        tmp += (Ctilde * U[n + t][k] * U[n + t + k][k]
                * dag(U[n + k + k][t]) * dag(U[n + k][k])
                * dag(U[n][k]) * U[n][t]) * 2. * (3./2.) * c[1];
        // the 1x2 contribution, with usual weight c_1
        tmp += (Ctilde * U[n + t][k] * dag(U[n + k][t])
                * dag(U[n + k - t][t]) * dag(U[n-t][k]) 
                * U[n - t][t] * U[n][t]) * c[1];
      }
#pragma omp critical
      val += tmp;
    }
  };

#else
  // Measure Gamma at t = 0
  template <class Field_t, template <class C> class init_helper_t>
  struct GammaLowerKernel {

    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 0;

    ptSU3 val;
    // \tilde C = - [d_eta C]
    // at the t = 0 side, we have dagger(e^C) and hence an insertion
    // of \tilde C, since C itself is purely imaginary
    BGF Ctilde;
    explicit GammaLowerKernel (const int &L) : val(bgf::zero<BGF>()) { 
      init_helper_t<BGF>()(L, Ctilde);
    }
    void operator()(const Field_t& U, const Point& n){
      Direction t(0);
      ptSU3 tmp(bgf::zero<BGF>());
      for (Direction k(1); k.is_good(); ++k)
        tmp += Ctilde * dag(U[n][k]) * U[n][t] * 
          U[n + t][k] * dag( U[n + k][t] );
#pragma omp critical
      val += tmp;
    }
  };

  // Measure Gamma at t = T - a
  template <class Field_t, template <class C> class init_helper_t>
  struct GammaUpperKernel {

    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 0;

    ptSU3 val;
    // here, we need [d_eta C'], which is equal to -[d_eta C], hence
    // we can use Ctilde as above
    BGF Ctilde;
    explicit GammaUpperKernel (const int& L) : val(bgf::zero<BGF>()) { 
      init_helper_t<BGF>()(L, Ctilde);
    }
    void operator()(const Field_t& U, const Point& n){
      Direction t(0);
      ptSU3 tmp(bgf::zero<BGF>());
      for (Direction k(1); k.is_good(); ++k)
        tmp += Ctilde * U[n + t][k] * dag(U[n + k][t])
          * dag(U[n][k]) * U[n][t];
#pragma omp critical
      val += tmp;
    }
  };
#endif


  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Writing a gluon to a file.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Fri May 25 15:59:06 2012

  template <class Field_t>
  class FileWriterKernel {
  public:
    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    // This may NOT be executed in parallel, so ...
    typedef void NoPar;

    explicit FileWriterKernel (uparam::Param& p) : o(p) { }

    void operator()(Field_t& U, const Point& n){
      for (Direction mu(0); mu.is_good(); ++mu)
#pragma omp critical
        U[n][mu].write(o);
    }
    io::CheckedOut o;
  private:
    // make n_cb private to prevent parallel application of this
    // kernel, because this would be a terrible idea
    static const int n_cb = 0;
  };
  
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Reading a gluon from a file.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Wed May 30 18:37:03 2012

  template <class Field_t>
  struct FileReaderKernel {
  public:
    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;

    // This may NOT be executed in parallel, so ...
    typedef void NoPar;

    explicit FileReaderKernel (uparam::Param& p) : i(p) { }

    void operator()(Field_t& U, const Point& n){
      for (Direction mu(0); mu.is_good(); ++mu)
        U[n][mu].read(i);
    }
    io::CheckedIn i;
  private:
    // make n_cb private to prevent parallel application of this
    // kernel, because this would be a terrible idea
    static const int n_cb = 0;
  };



  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Reading a gluon from a file.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Wed May 30 18:37:03 2012

  template <class Field_t>
  struct PRlgtReaderKernel {
  public:
    // collect info about the field
    typedef typename std_types<Field_t>::ptGluon_t ptGluon;
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;
    typedef typename std_types<Field_t>::bgf_t BGF;
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int ORD = std_types<Field_t>::order;
    static const int DIM = std_types<Field_t>::n_dim;
    static const int SIZE = 2 + ptSU3::storage_size;
    typename array_t<double,SIZE>::Type data;

    // This may NOT be executed in parallel, so ...
    typedef void NoPar;

    explicit PRlgtReaderKernel (std::ifstream& in) : is(in) { }

    void operator()(Field_t& U, const Point& n){
      SU3 t;
      for(Direction mu(0); mu.is_good(); ++mu) {
	is.read((char*)(&data[0]),(SIZE*sizeof(double))/sizeof(char) );
	for(int o=o;o<ORD;++o) {
	  for(int a=0;a<9;++a)
	    t[a]=data[a+ORD*o];
	  U[n][mu][o] = t;
	}

	// std::cout <<  U[n][mu].bgf() << std::endl;
	// // for(int b=0;b<3;++b)
	// //   for(int a=0;a<3;++a)
	// std::cout << U[n][mu][0] << std::endl << std::endl;

      } // Direction

      // int l;
      // std::cin >> l;
    }

  private:
    // make n_cb private to prevent parallel application of this
    // kernel, because this would be a terrible idea
    static const int n_cb = 0;
    std::ifstream& is;
  };




  namespace detail {

    struct LowBdy { };
    struct HighBdy { };
    struct NoBdy { };
    
    namespace helper {
      template<class F_t,class bdy>
      const F_t add(const F_t& S1,const F_t& S2, bdy) {
	F_t res(S1);
	return res+=S2;
      }
      
      template<class F_t>
      F_t add(const F_t& S1,const F_t& S2, LowBdy) {
	return std::move(S1-S2);
      }
      
      template<class F_t>
      F_t add(const F_t& S1,const  F_t& S2, HighBdy) {
	return std::move(S2-S1);
      }
    } // helper

    // non perturbative computation of
    // $U^\dagger_\mu(x-mu)(1+\gamma_mu)\Psi(x-mu)\pm U_\mu(x)(1-\gamma_mu)\Psi(x)
    template<class Gauge_t, class Field_t, class bdy>
    typename base_types<Field_t>::data_t Gamma(const Gauge_t& U, const Field_t& src,
					       const typename base_types<Field_t>::point_t& n,
					       const typename base_types<Field_t>::direction_t& mu, 
					       bdy) {
      typedef typename base_types<Field_t>::data_t F;
      typedef typename base_types<Field_t>::direction_t Direction;
      typedef typename base_types<Field_t>::point_t Point;

      F Xi1, Xi2;
      Point dn = n-Direction(mu);
      Point up = n+Direction(mu);

      for(Direction nu(0);nu.is_good();++nu) {
	Xi1[nu] = (src[dn][nu]+src[dn][dirac::gmuind[mu][nu]]*dirac::gmuval[mu][nu]);
	Xi2[nu] = (src[up][nu]-src[up][dirac::gmuind[mu][nu]]*dirac::gmuval[mu][nu]);
      }
      return helper::add(dag(U[n-mu][mu])*Xi1,U[n ][mu]*Xi2,bdy());
    }


    // perturbative computation of
    // $U^\dagger_\mu(x-mu)(1+\gamma_mu)\Psi(x-mu)\pm U_\mu(x)(1-\gamma_mu)\Psi(x)
    template<class Gauge_t, class Field_t, class bdy>
    typename base_types<Field_t>::data_t Gamma(const Gauge_t& U, const Field_t& src,
					       const typename base_types<Field_t>::point_t& n,
					       const typename base_types<Field_t>::direction_t& mu,
					       const int ord,
					       bdy) {
      typedef typename base_types<Field_t>::data_t F;
      typedef typename base_types<Field_t>::direction_t Direction;
      typedef typename base_types<Field_t>::point_t Point;

      F Xi1, Xi2;
      Point dn = n-Direction(mu);
      Point up = n+Direction(mu);

      for(Direction nu(0);nu.is_good();++nu) {
	Xi1[nu] = (src[dn][nu]+src[dn][dirac::gmuind[mu][nu]]*dirac::gmuval[mu][nu]);
	Xi2[nu] = (src[up][nu]-src[up][dirac::gmuind[mu][nu]]*dirac::gmuval[mu][nu]);
      }
      typename std_types<Gauge_t>::ptSU3_t d = dag(U[n-mu][mu]);
      return helper::add(d[ord]*Xi1,U[n ][mu][ord]*Xi2,bdy());
    }
  }


  
  //////////////////////////////////////////////////////////////////////
  ///
  /// Wilson dirac operator 
  template <class Field_t, class bdy>
  struct WilsonKernel : public base_types<Field_t> {
  public:
    // collect info about the field
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int DIM = std_types<Field_t>::n_dim;
    // fermion
    typedef SpinColor<DIM> Fermion;
    typedef typename fields::LocalField<Fermion, DIM> FermionField;

    // checker board hyper cube size
    static const int n_cb = 0;
    
    WilsonKernel(const Field_t& G, const FermionField& F, 
		 const double& m ) :
      U(G), src(F), mass(m) { };
    
    void operator() ( FermionField& dest, const Point& n) {
      dest[n] = mass*src;
      for( Direction mu(1); mu.is_good(); ++mu )
	dest[n] -= Gamma(U,src,n,mu,detail::NoBdy()) * .5;
      dest[n] -= Gamma(U,src,n,Direction(0),bdy()) * .5;
    }
  
  private:
    
    const double& mass;
    const Field_t& U;
    const FermionField& src;
  };



  //////////////////////////////////////////////////////////////////////
  ///
  /// Wilson dirac operator 
  template <class Field_t, class bdy>
  struct PTWilsonKernel : public base_types<Field_t> {
  public:
    // collect info about the field
    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;
    static const int DIM = std_types<Field_t>::n_dim;
    // fermion
    typedef SpinColor<DIM> Fermion;
    typedef typename fields::LocalField<Fermion, DIM> FermionField;

    // checker board hyper cube size
    static const int n_cb = 0;
    
    PTWilsonKernel(const Field_t& G, const std::vector<FermionField>& F, 
		 const std::vector<double>& m, int& oo ) :
      U(G), src(F), mass(m), o(oo) { };
    
    void operator() ( FermionField& dest, const Point& n) {

      // if(int(n)==0)
      // 	std::cout << "ord = " << o << std::endl;
      //      dest[n] *= 0;
      for( int kord = 0; kord < o; kord++) {
	// if(int(n)==0)
	//   std::cout << "\tmass[" << o-kord 
	// 	    << "] = " << mass[o-kord]
	// 	    << "\tsrc[" << kord
	// 	    << "]=\n" << src[o-kord][n]
	// 	    << std::endl;
      	// critical mass
      	dest[n] += src[kord][n]*mass[o-kord];
      	// interaction
      	for( Direction mu(1); mu.is_good(); ++mu )
      	  dest[n] -= Gamma(U,src[kord],n,mu,o-kord-1,detail::NoBdy()) * .5;
      	dest[n] -= Gamma(U,src[kord],n,Direction(0),o-kord-1,bdy()) * .5;
      } // kord
      // if(int(n)==0)
      // 	std::cout << std::endl;
	  
    }
  
  private:
    
    const std::vector<double>& mass;
    const Field_t& U;
    const std::vector<FermionField>& src;
    const int& o;
  };



  template <class Field_t, int boundary>
  struct WilsonPropagator {
  public:

    typedef typename base_types<Field_t>::data_t data_t; 
    typedef typename base_types<Field_t>::point_t Point;
    typedef typename base_types<Field_t>::direction_t Direction;
    typedef typename Field_t::raw_pt raw_pt;
    typedef typename Field_t::extents_t extents_t;
    static const int DIM = base_types<Field_t>::n_dim;

    typedef std::valarray<double> array_t;    

    // checker board hyper cube size
    // c.f. geometry and localfield for more info
    static const int n_cb = 0;

    WilsonPropagator(Field_t& other, const double m) : 
      mbare(m), ext(other.extents()), 
      k({2.0*M_PI/ext[0],2.0*M_PI/ext[1],2.0*M_PI/ext[2],2.0*M_PI/ext[3]}) { }
    

    void operator() (Field_t& dest, const Point& n) {
      do_it(dest,n,mode_selektor<boundary>());
    }

  private:
    const double mbare;
    const extents_t ext;
    const array_t k;

    template <int M> struct mode_selektor { };

    void do_it(Field_t& dest, const Point& n,mode_selektor<0>) {
      array_t p(DIM);
      if( int(n) != 0) {
	raw_pt x = dest.coords(n);
	for( Direction mu(0); mu.is_good(); ++mu )
	  p[int(mu)] = k[int(mu)]*x[int(mu)];
	propagator(dest[n],p);
      }
      else
	dest[n] *= 0;
    }

    void do_it(Field_t& dest, const Point& n,mode_selektor<1>) {
      array_t p(DIM);
      raw_pt x = dest.coords(n);
      p[0] = k[0]*(x[0]+.5);
      for( Direction mu(1); mu.is_good(); ++mu )
	p[int(mu)] = k[int(mu)]*x[int(mu)];
      propagator(dest[n],p);
    }

    void propagator(data_t& result, const array_t& p) {
      array_t pb(std::sin(p));
      array_t p2hat(std::pow(2*std::sin(.5*p),2));
      double M = mbare + .5*p2hat.sum();
      double den = 1./(M*M+std::inner_product(begin(pb),end(pb),begin(pb),0.0));
      data_t F(result*M);


      // std::cout << "M = "     << M
      // 		<< "\tden = " << den
      // 		<< std::endl;

      for( Direction mu(0); mu.is_good(); ++mu )
	for( Direction nu(0); nu.is_good(); ++nu )
	  F[nu] -= ( dirac::gmuval[mu][nu] * result[dirac::gmuind[mu][nu]] *
		     Cplx(0.,pb[int(mu)]) );
      result = F*den;
    }
    
  };

  
  
  ////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////
  /////
  /////  Perturbative application of the staggered dirac operator
  /////  
  /////
  /////  \author Michele Brambilla <mib.mic@gmail.com>
  /////  \date Fri Nov 02 16:23:46 2012
  //
  //template <class Field_t>
  //struct StaggeredPTKernel {
  //  // collect info about the field
  //  typedef typename std_types<Field_t>::ptGluon_t ptGluon;
  //  typedef typename std_types<Field_t>::ptSU3_t ptSU3;
  //  typedef typename std_types<Field_t>::ptsu3_t ptsu3;
  //  typedef typename std_types<Field_t>::bgf_t BGF;
  //  typedef typename std_types<Field_t>::point_t Point;
  //  typedef typename std_types<Field_t>::direction_t Direction;
  //  static const int ORD = std_types<Field_t>::order;
  //  static const int DIM = std_types<Field_t>::n_dim;
  //
  //  // fermion
  //  typedef SpinColor<1> Fermion;
  //  typedef fields::LocalField<Fermion, DIM> FermionField;
  //  typedef typename array_t<double, DIM>::Type double_array_t;    
  //  
  //  // checker board hyper cube size
  //  // c.f. geometry and localfield for more info
  //  static const int n_cb = 0;
  //
  //
  //
  //  StaggeredPTKernel( Field_t& G, FermionField& X, const double& m ) : U(G), src(X), mass(m) { Uorder = 0; std::cout << "Fix eta definition!!!!!\n";};
  //  
  //  void eta(const Point& n, double_array_t& phase) {
  //
  //  }
  //  
  //  void next_Uorder() { Uorder++; }
  //  void reset_Uorder() { Uorder = 0; }
  //
  //  void operator()( FermionField& dest, Point& n) {
  //    eta(n, phase);
  //    for( Direction nu(0); nu.is_good(); ++nu )
  //	dest[n][0] += .5*phase[(int)nu] * ( U[n][nu][Uorder] * src[n+nu][0] - U[n-nu][nu][Uorder] * src[n-nu][0] );
  //    //      mass
  //    dest[n][0] += mass * src[n][0];
  //
  //  }
  //
  //private:
  //  int Uorder;
  //  double mass;
  //  double_array_t phase;
  //  Field_t& U;
  //  FermionField& src;
  //
  //};

  template<class G, class F>
  struct FermionicUpdateKernel {

    typedef typename std_types<G>::ptSU3_t ptSU3;
    typedef typename std_types<G>::bgf_t BGF;
    typedef typename std_types<G>::point_t Point;
    typedef typename std_types<G>::direction_t Direction;
    static const int ORD = std_types<G>::order;
    static const int DIM = std_types<G>::n_dim;
    static const int Nc  = 3;//ptsu3::su3_array_t::C::N;
    //    typedef BGptSU3<bgf::TrivialBgf, ORD> ptsu3;
    typedef typename std_types<G>::ptsu3_t ptsu3;

    typedef typename base_types<F>::data_t Fermion;

    static const int n_cb = 0;

    const F& xi;
    const std::vector<F>& psi;
    const Direction mu;
    const double tauf;


    FermionicUpdateKernel(F& Xi_,
			  const std::vector<F>& Psi_, 
			  const Direction& dir, const double& t) : 
      xi(Xi_), psi(Psi_), mu(dir), tauf(-.5*t) { };

    void operator()(G& U, const Point& n) {
      ptSU3 W;
      ptsu3 W1;
      int i;
      typename array_t<Fermion,ORD-1>::Type tmp;

      for(int o=0;o<ORD-1;++o) {
	for(Direction nu(0);nu.is_good();++nu) {
      	  tmp[o][nu] = (psi[o][n][nu]+
      	  		psi[o][n][dirac::gmuind[mu][nu]]*dirac::gmuval[mu][nu]);
	}
      }

      for(int o=0;o<ORD-1;++o) {
      	for(int k=0;k<Nc;k++) {
      	  for(int j=0;j<Nc;j++) {
      	    for(Direction nu(0);nu.is_good();++nu) {
      	      W[o+1](k,j) += xi[n+mu][nu][j].conj() * tmp[o][nu][k];
      	    }
      	  }
      	}
      }

      W.bgf() *= 0;
      W1 = (W*dag(U[n][mu])).ptU();
      std::for_each(W1.begin(),W1.end(),[](SU3& s){ 
	  s -= s.dag(); s *= .5; 
	  complex tr(s.tr()/3.0);
	  s[0] -= tr; s[4] -=tr; s[8] -= tr;
	});

      U[n][mu] = exp<BGF, ORD>(W1*tauf)*U[n][mu];
    }
  };



}

#endif




