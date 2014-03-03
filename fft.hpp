#ifndef _FFT_H
#define _FFT_H

#include <fftw3.h>

namespace fft {

  enum {pbc, abc};
  enum {x2p, p2x};
  
  template<class Field_t, int boundary>
  struct fft {
    typedef typename Field_t::data_t data_t;
    typedef typename Field_t::Point Point;
    typedef typename Field_t::extents_t extents_t;
    typedef typename fields::detail::inplace_smul<Field_t,complex> scale;
    const int DIM = Field_t::dim;
    const int data_sz = sizeof(data_t)/(2*sizeof(double));
    const double ivol;
    // the factor 2 gets rid of fft_complex basic fftw3 data
    
    fft( Field_t& ref) : F(ref), k(M_PI/static_cast<double>(ref.extent(0))),
			 T(ref.extent(0)),ivol(1./static_cast<double>(ref.vol())) {
      int* extents = new int[DIM];
      for (int i=0;i<DIM;++i) extents[i] = F.extent(i);
      plan[0] = fftw_plan_many_dft(DIM, extents,data_sz,
      				   reinterpret_cast<fftw_complex*>(&(*F.begin())),
      				   NULL,data_sz,1,
      				   reinterpret_cast<fftw_complex*>(&(*F.begin())),
      				   NULL,data_sz,1,
      				   FFTW_FORWARD, FFTW_MEASURE);
      
      plan[1] = fftw_plan_many_dft(DIM, extents,data_sz,
      				   reinterpret_cast<fftw_complex*>(&(*F.begin())),
      				   NULL,data_sz,1,
      				   reinterpret_cast<fftw_complex*>(&(*F.begin())),
      				   NULL,data_sz,1,
      				   FFTW_BACKWARD, FFTW_MEASURE);  
      // DRY RUN to initialize plans without loss of data
      fftw_execute(plan[0]);
      fftw_execute(plan[1]);
    }

    void execute(const int& direction) {
      do_it(direction,mode_selektor<boundary>() );
    }
    void execute(Field_t& in, const int& direction) {
      do_it(in, direction,mode_selektor<boundary>() );
    }
    void execute(Field_t& in, Field_t& out, const int& direction) {
      do_it(in, out, direction,mode_selektor<boundary>() );
    }

  private:
    Field_t&  F;
    fftw_plan plan[2];
    const double k;
    const int T;
    
    template <int M> struct mode_selektor { };
      
    void do_it(const int& direction, const mode_selektor<pbc>) {
      fftw_execute(plan[direction]);
      if( direction == p2x ) {
	scale S(ivol);
	F.apply_everywhere(S);
      }
    }
    void do_it(Field_t& in, const int& direction, const mode_selektor<pbc>) {
      fftw_execute_dft(plan[direction],
		       reinterpret_cast<fftw_complex*>(&(*in.begin())),
		       reinterpret_cast<fftw_complex*>(&(*in.begin())));
      if( direction == p2x ) 
	for(int t=0;t<T;++t) {
	  scale S(ivol);
	  in.apply_on_timeslice(S,t);
	}
    }
    void do_it(Field_t& in, Field_t& out, const int& direction, const mode_selektor<pbc>) {
      fftw_execute_dft(plan[direction],
		       reinterpret_cast<fftw_complex*>(&(*in.begin())),
		       reinterpret_cast<fftw_complex*>(&(*out.begin())));
      if( direction == p2x ) {
	scale S(ivol);
	in.apply_everywhere(S);
      }
    }

    void do_it(const int& direction, const mode_selektor<abc>) {
      if( direction == x2p ) {
	for(int t=0;t<T;++t) {
	  scale S( complex(cos(k*t),sin(k*t)) );
	  F.apply_on_timeslice(S,t);
	}
	fftw_execute(plan[direction]);
      }
      else {
	  fftw_execute(plan[direction]);
	  for(int t=0;t<T;++t) {
	    scale S( complex(cos(k*t),-sin(k*t))*ivol );
	    F.apply_on_timeslice(S,t);
	  }
	}
    }
    void do_it(Field_t& in, Field_t& out, const int& direction, const mode_selektor<abc>) {
      if( direction == x2p ) {
	for(int t=0;t<T;++t) {
	  scale S( complex(cos(k*t),sin(k*t)) );
	  in.apply_on_timeslice(S,t);
	}
	fftw_execute_dft(plan[direction],
			 reinterpret_cast<fftw_complex*>(&(*in.begin())),
			 reinterpret_cast<fftw_complex*>(&(*out.begin() )));
	for(int t=0;t<T;++t) {
	  scale S( 1./complex(cos(k*t),sin(k*t)) );
	  in.apply_on_timeslice(S,t);
	}
      }
      else {
	fftw_execute_dft(plan[direction],
			 reinterpret_cast<fftw_complex*>(&(*in.begin())),
			 reinterpret_cast<fftw_complex*>(&(*out.begin() )));
	for(int t=0;t<T;++t) {
	  scale S( complex(cos(k*t),-sin(k*t))*ivol);
	  out.apply_on_timeslice(S,t);
	}
      }
    }
    
  };

} // namespace fft

#endif //FFT_H
