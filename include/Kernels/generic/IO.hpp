#ifndef _IO_KERNELS_H
#define _IO_KERNELS_H

#include <Kernels.hpp>
#include <Background.h>

#ifdef _OPENMP
#include <omp.h>
#else
namespace plaq {
  int omp_get_max_threads() { return 1; }
  int omp_get_thread_num() { return 0; }
}
#endif

namespace kernels {

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
  ///  Reading a gluon from a PRlgt file.                                                                
  ///                                                                                                    
  ///  \date Tue Mar 11 15:22:43 2014                                                                    
  ///  \author Michele Brambilla <mib.mic@gmail.com>                                                     
  template <class Field_t>
  struct PRlgtReaderKernel {
  public:
    // collect info about the field                                                                      
    typedef typename std_types<Field_t>::ptSU3_t ptSU3;
    typedef typename std_types<Field_t>::ptsu3_t ptsu3;

    typedef typename std_types<Field_t>::point_t Point;
    typedef typename std_types<Field_t>::direction_t Direction;

    // This may NOT be executed in parallel, so ...                                                      
    typedef void NoPar;

    explicit PRlgtReaderKernel (std::ifstream& in) : is(in) { }

    void operator()(Field_t& U, const Point& n){
      std::vector<double> v(ptSU3::storage_size);
      ptsu3 vv;
      for(Direction mu(0); mu.is_good(); ++mu) {
        is.read(reinterpret_cast<char*>(&v[0]), ptSU3::storage_size*sizeof(double));
        std::vector<double>::iterator it = v.begin()+2;
        vv.unbuffer(it);
        U[n][mu].ptU()=vv;
      }

    }

  private:
    // make n_cb private to prevent parallel application of this                                         
    // kernel, because this would be a terrible idea                                                     
    static const int n_cb = 0;
    std::ifstream& is;
  };

}

#endif //IO_KERNELS_H
