#define FLD_INFO(F) \
  typedef typename std_types<F>::ptGluon_t ptGluon;	\
  typedef typename std_types<F>::ptSU3_t ptSU3;		\
  typedef typename std_types<F>::ptsu3_t ptsu3;		\
  typedef typename std_types<F>::bgf_t BGF;		\
  typedef typename std_types<F>::point_t Point;		\
  typedef typename std_types<F>::direction_t Direction;	\
  static const int ORD = std_types<F>::order;		\
  static const int DIM = std_types<F>::n_dim;

namespace kernels {

  namespace gauge_update {
    ////////////////////////////////////////////////////////////
    //
    //  Gauge update with third-order Runge-Kutta scheme, first step.
    //
    //  \bug       CURRENTLY DOES NOT SEEM TO WORK!
    //  \date      Thu Feb 21 19:12:07 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>
    template <class Field_t, class StapleK_t, class RF_t>
    struct GU_RK_1 {
      
      // collect info about the field
      FLD_INFO(Field_t);
  
      // checker board hyper cube size
      // c.f. geometry and localfield for more info
      static const int n_cb = StapleK_t::n_cb;    
      
      Direction mu;
      double taug;
      double staug;
      Field_t *F;
      RF_t *R;
      
      GU_RK_1(const Direction& nu, const double& t, Field_t& FF, RF_t &RR) :
        mu(nu), taug(t), staug(std::sqrt(t)), F(&FF), R(&RR) { }
  
      void operator()(Field_t& U, const Point& n) {
        // Make a Kernel to calculate and store the plaquette(s)
        StapleK_t st(mu); // maye make a vector of this a class member
        st(U,n);
        (*F)[n][mu] = st.reduce() * taug;
        (*F)[n][mu][0] += (*R)[n] * staug;
        U[n][mu] = exp<BGF, ORD>((*F)[n][mu].reH() * -.25 )*U[n][mu]; // back to SU3
      }
    };
  
    ////////////////////////////////////////////////////////////
    //
    //  Gauge update with third-order Runge-Kutta scheme, second
    //  step.
    //
    //  \bug       CURRENTLY DOES NOT SEEM TO WORK!
    //  \date      Thu Feb 21 19:12:43 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>
    template <class Field_t, class StapleK_t, class RF_t>
    struct GU_RK_2 {
      
      // collect info about the field
      FLD_INFO(Field_t);

      // checker board hyper cube size
      // c.f. geometry and localfield for more info
      static const int n_cb = StapleK_t::n_cb;    
      
      Direction mu;
      double taug;
      double staug;
      Field_t *F;
      RF_t *R;
      
      GU_RK_2(const Direction& nu, const double& t, Field_t& FF, RF_t &RR) :
        mu(nu), taug(t), staug(std::sqrt(t)), F(&FF), R(&RR) { }
  
      void operator()(Field_t& U, const Point& n) {
        // Make a Kernel to calculate and store the plaquette(s)
        StapleK_t st(mu); // maye make a vector of this a class member
        st(U,n);
        (*F)[n][mu] = (*F)[n][mu] * -17./36 + 8./9 * st.reduce() * taug;
        (*F)[n][mu][0] += (*R)[n] * staug * 8./9;
        U[n][mu] = exp<BGF, ORD>( (*F)[n][mu].reH() * -1. )*U[n][mu]; // back to SU3
      }
    };

    ////////////////////////////////////////////////////////////
    //
    //  Gauge update with second-order Runge-Kutta scheme, first
    //  step.
    //
    //  \bug       CURRENTLY DOES NOT SEEM TO WORK!
    //  \date      Thu Feb 21 19:12:50 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>
    template <class Field_t, class StapleK_t, class RF_t>
    struct GU_RK_3 {
      
      // collect info about the field
      FLD_INFO(Field_t);

      // checker board hyper cube size
      // c.f. geometry and localfield for more info
      static const int n_cb = StapleK_t::n_cb;    
      
      Direction mu;
      double taug;
      double staug;
      Field_t *F;
      RF_t *R;
      
      GU_RK_3(const Direction& nu, const double& t, Field_t& FF, RF_t &RR) :
        mu(nu), taug(t), staug(std::sqrt(t)), F(&FF), R(&RR) { }
  
      void operator()(Field_t& U, const Point& n) {
        // Make a Kernel to calculate and store the plaquette(s)
        StapleK_t st(mu); // maye make a vector of this a class member
        st(U,n);
        (*F)[n][mu] = st.reduce() * 3./4 * taug -(*F)[n][mu];
        (*F)[n][mu][0] += (*R)[n] * staug * 3./4;
        U[n][mu] = exp<BGF, ORD>( (*F)[n][mu].reH() * -1)*U[n][mu]; // back to SU3
      }
    };
  
    ////////////////////////////////////////////////////////////
    //
    //  Gauge update with second-order Runge-Kutta scheme, first
    //  step.
    //
    //  \warning   See warning in GU_RK2_2.
    //
    //  \date      Thu Feb 21 19:28:54 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>
    template <class Field_t, class StapleK_t, class RF_t>
    struct GU_RK2_1 {
      
      // collect info about the field
      FLD_INFO(Field_t);
  
      // checker board hyper cube size
      // c.f. geometry and localfield for more info
      static const int n_cb = StapleK_t::n_cb;    
      
      Direction mu;
      double taug;
      double staug;
      Field_t *F, *Utilde;
      RF_t *R;
      
      GU_RK2_1(const Direction& nu, const double& t, Field_t& FF, RF_t &RR, Field_t& Util) :
        mu(nu), taug(t), staug(std::sqrt(t)), F(&FF), R(&RR), Utilde(&Util) { }
  
      void operator()(Field_t& U, const Point& n) {
        // Make a Kernel to calculate and store the plaquette(s)
        StapleK_t st(mu); // maye make a vector of this a class member
        st(U,n);
        (*F)[n][mu] = st.reduce();
        ptsu3 tmp = (*F)[n][mu].reH() * -taug;
        tmp[0] -= (*R)[n] * staug;
	//ptsu3 tmp;
	//tmp[0] -= (*R)[n] * staug;
        (*Utilde)[n][mu] = exp<BGF, ORD>(tmp)*U[n][mu]; // back to SU3
      }
    };

    ////////////////////////////////////////////////////////////
    //
    //  Gauge update with -order Runge-Kutta scheme, second step.
    //
    //  \warning   We are not sure if what is below called "Method I"
    //  or "Method II" is correct! Else it seems to work now.
    //
    //  \date      Thu Feb 21 19:29:02 2013
    //  \author    Dirk Hesse <dirk.hesse@fis.unipr.it>
    template <class Field_t, class StapleK_t, class RF_t>
    struct GU_RK2_2 {
      
      // collect info about the field
      FLD_INFO(Field_t);
      
      // checker board hyper cube size
      // c.f. geometry and localfield for more info
      static const int n_cb = StapleK_t::n_cb;    
      
      Direction mu;
      double taug;
      double staug;
      Field_t *F, *Utilde;
      RF_t *R;

      // zero momentum contribution
      std::vector<ptsu3> M;

      GU_RK2_2(const Direction& nu, const double& t, Field_t& FF, RF_t &RR, Field_t& Util) :
        mu(nu), taug(t), staug(std::sqrt(t)), F(&FF), R(&RR), Utilde(&Util), M(omp_get_max_threads()) { }
  
      void operator()(Field_t& U, const Point& n) {
        // Make a Kernel to calculate and store the plaquette(s)
        StapleK_t st(mu); // maye make a vector of this a class member
        st(*Utilde,n);
        (*F)[n][mu] += st.reduce();
	/*/ Method I
	(*F)[n][mu] *= -(0.5  + .25* taug) * taug;
	/*/  
	// Method II
	(*F)[n][mu] *= -.5 * taug;
        for (int i = 0; i < ORD - 2; ++i)
	  (*F)[n][mu][i + 2] += 0.5 * taug * (*F)[n][mu][i];
	//*/
        (*F)[n][mu][0] -= (*R)[n] * staug;
        U[n][mu] = exp<BGF, ORD>( (*F)[n][mu].reH() )*U[n][mu]; // back to SU3
	//ptsu3 tmp;
	//tmp[0] -= (*R)[n] * staug;
        //U[n][mu] = exp<BGF, ORD>(tmp)*U[n][mu]; // back to SU3
	M[omp_get_thread_num()] += get_q(U[n][mu]); // zero momentum contribution
      }

      // reduce vector of zero momentum contribution
      const ptsu3& reduce(){
	std::for_each(M.begin()+1, M.end(), [&] (const ptsu3& i){ M[0] += i; } );
	return M[0];
      }

    };

    template <class Field_t, class StapleK_t, class RF_t>
    struct GU_RK1 {
      
      // collect info about the field
      FLD_INFO(Field_t);
      
      // checker board hyper cube size
      // c.f. geometry and localfield for more info
      static const int n_cb = StapleK_t::n_cb;    
      
      Direction mu;
      double taug;
      double staug;
      Field_t *F;
      RF_t *R;

      // zero momentum contribution
      std::vector<ptsu3> M;
      
      GU_RK1(const Direction& nu, const double& t, Field_t& FF, RF_t &RR) :
        mu(nu), taug(t), staug(std::sqrt(t)), F(&FF), R(&RR), M(omp_get_max_threads()) { }
  
      void operator()(Field_t& U, const Point& n) {
        // Make a Kernel to calculate and store the plaquette(s)
        StapleK_t st(mu); // maye make a vector of this a class member
        st(U,n);
	ptsu3 tmp  = st.reduce().reH() * -taug;
	//ptsu3 tmp;
	tmp[0] -= (*R)[n] * staug;
        U[n][mu] = exp<BGF, ORD>(tmp)*U[n][mu]; // back to SU3
	M[omp_get_thread_num()] += get_q(U[n][mu]); // zero momentum contribution
      }

      // reduce vector of zero momentum contribution
      const ptsu3& reduce(){
	std::for_each(M.begin()+1, M.end(), [&] (const ptsu3& i){ M[0] += i; } );
	return M[0];
      }
    };
  } // end namespace gauge_update
}
