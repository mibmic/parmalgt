#ifndef _IO_H_
#define _IO_H_

#include <Types.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <uparam.hpp>
#ifdef USE_MPI
#include <mpi.h>
#endif
// Note the MD5 Copyright notice in the 
// implementation file IO.cc!


namespace comm {
  template<class T> struct Reduce;
}

namespace io {

  // The data on which md5 operates has to be accessed as double,
  // unsigned and unsigned char. This union will help here...
  
  namespace detail {
    // NOTE THAT THESE ARE ARCHITECTURE DEPENDENT!
    const int unsigned_per_double = 2;
    const int char_per_double = 8;

    union md5atom {
      double d[8];
      unsigned u[8 * unsigned_per_double];
      unsigned char c[8 * char_per_double];
    };


    //////////////////////////////////////////
    //
    // MB: actual writing
    inline void to_bin_file(std::ofstream& of, const Cplx& c){
      of.write(reinterpret_cast<const char*>(&c.real()), sizeof(double));
      of.write(reinterpret_cast<const char*>(&c.imag()), sizeof(double));
    }
    template <class ptSU3, int ORD>
    inline void write_file(const ptSU3& U, const Cplx& tree, const std::string& fname){
      std::ofstream of(fname.c_str(), std::ios_base::app |  std::ios_base::binary);
      to_bin_file(of, tree);
      for (int i = 0; i < ORD; ++i)
	to_bin_file(of, U[i].tr());
      of.close();
    }
    template <class CONT>
    inline void write_file(const CONT& c, const std::string& fname){
      std::ofstream of(fname.c_str(), std::ios_base::app |  std::ios_base::binary);
      for (typename CONT::const_iterator i = c.begin(), j = c.end(); i != j; ++i)
	of.write(reinterpret_cast<const char*>(&(*i)), sizeof(typename CONT::value_type));
      of.close();
    }
    template <class CONT>
    inline void write_ptSUN(const CONT& c, const std::string& fname){
      std::ofstream of(fname.c_str(), std::ios_base::app |  std::ios_base::binary);
      Cplx tmp = c.bgf().Tr();
      to_bin_file(of, tmp);
      for (typename CONT::const_iterator i = c.begin(), j = c.end(); i != j; ++i){
	tmp = i->Tr();
	of.write(reinterpret_cast<const char*>(&tmp), sizeof(Cplx));
      }
      of.close();
    }


  }

  class CheckedIo;

  std::ostream& operator <<(std::ostream& os, CheckedIo &io);

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Helper calculate the md5 checksum on-the-fly.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Fri May 25 16:25:45 2012

  class CheckedIo {
  public:
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Constructor.
    ///
    ///  Initialize the md5 algorithm.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Wed May 30 18:37:52 2012
    CheckedIo() : bcount(0), buffcnt(0) { 
      h[0] = 0x67452301;
      h[1] = 0xEFCDAB89;
      h[2] = 0x98BADCFE;
      h[3] = 0x10325476;
    } 
    ~CheckedIo() { }
    void finalize();
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Return the md5 as a vector.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Wed May 30 18:38:30 2012
    std::vector<unsigned> get_h() const {
      return std::vector<unsigned>(h, h+4);
    }
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Return the md5 as a string.
    ///
    ///  Note: This calls finialize, so it should only be called ONCE!
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Wed May 30 18:38:39 2012
    std::string md5(){
      finalize();
      unsigned char* w = reinterpret_cast<unsigned char*> (h);
      std::stringstream s;
      s << std::hex;
      for (int i = 0; i < 16; i++)
        s << (int) w[i];
      s << std::dec;
      return s.str();
    }
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Process a dobule.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Wed May 30 18:40:06 2012
    void process(const double& d){
      w.d[buffcnt++] = d;
      if (buffcnt == 8){
        md5process();
        buffcnt = 0;
        ++bcount;
      }
    }
  private:
    unsigned h[4];
    static const unsigned r[], k[];
    detail::md5atom w;
    // block count
    unsigned bcount;
    // buffer count
    int buffcnt;
    void md5process();
    unsigned lrol(const unsigned& u, const unsigned & shift) {
      return (u << shift) | (u >> (32 - shift));
    }
  };


  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Writes complex numbers to a file and calculates the md5
  ///  checksum on-the-fly.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Wed May 30 18:40:15 2012
  class CheckedOut : public CheckedIo {
  public:
    explicit CheckedOut(uparam::Param &p) : 
      CheckedIo(), os((p["write"] + ".cfg").c_str(), std::ios::trunc |
                      std::ios::binary ), param(p){
    }
    ~CheckedOut() { 
      os.close();
      param.set("md5", CheckedIo::md5());
      param.write(param["write"] + ".info");
    }
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    ///
    ///  Write a complex to disk.
    ///
    ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
    ///  \date Wed May 30 18:40:45 2012
    void write(const Cplx &c){
      CheckedIo::process(c.real());
      CheckedIo::process(c.imag());
      os.write(reinterpret_cast<char const*>(&(c.real())), sizeof(double));
      os.write(reinterpret_cast<char const*>(&(c.imag())), sizeof(double));
    }
  private:
    std::ofstream os;
    uparam::Param param;
  };

  class IoError : public std::exception {};


  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  ///
  ///  Read complex numbers from disk.
  ///
  ///  Check if the md5 checksum is right.
  ///
  ///  \author Dirk Hesse <herr.dirk.hesse@gmail.com>
  ///  \date Wed May 30 18:40:58 2012
  class CheckedIn : public CheckedIo {
  public:
    explicit CheckedIn(uparam::Param &p) : 
      CheckedIo(), is((p["read"] + ".cfg").c_str(), std::ios::binary ), param(p) {
      param.read(p["read"] + ".info");
      for (uparam::Param::const_iterator i = param.begin(); i!=
             param.end(); ++i){
        if (i->first == "md5" || i->first == "read" 
            || i->first == "write" || i->first == "NRUN")
          continue;
        if ( i->first == "seed" ) {
	  if( i->second == p[i->first] ) {
	    std::cout << "Change your seed! " << std::endl;
	    std::cout << "   in .info file : " << i->second 
		      << ", in current simulation " 
		      << p[i->first] << std::endl;
	    throw IoError();
	  }
	  else 
	    continue;
	}
        if (i->second != p[i->first]){
          std::cout << "Parameter mismatch: "<< i->first << std::endl;
          std::cout << "   in .info file : " << i->second 
                    << ", in current simulation: " 
                    << p[i->first] << std::endl;
          throw IoError();
        }
      }

    }
    ~CheckedIn() {
      is.close();
      std::string md5 = CheckedIo::md5();
      if (md5 != param["md5"]){
        std::cout << "'" << md5 << "'" << std::endl;
        std::cout << "'" << param["md5"] << "'" << std::endl;
        throw IoError();
      }
    }
    void read(Cplx& c){
      is.read(reinterpret_cast<char*>(&(c.real())), sizeof(double));
      is.read(reinterpret_cast<char*>(&(c.imag())), sizeof(double));
      CheckedIo::process(c.real());
      CheckedIo::process(c.imag());
    }
  private:
    std::ifstream is;
    uparam::Param param;
  };

  ////////////////////////////////////////////////////////////
  // writing binary data to files
  
  inline void to_bin_file(std::ofstream& of, const Cplx& c){
    of.write(reinterpret_cast<const char*>(&c.real()), sizeof(double));
    of.write(reinterpret_cast<const char*>(&c.imag()), sizeof(double));
  }
  template <class ptSU3, int ORD>
  inline void write_file(const ptSU3& U, const Cplx& tree, const std::string& fname){
#ifdef USE_MPI
    comm::Reduce<ptSU3> reduce(U);
    ptSU3 A = reduce();
    if(reduce.rank()==0) 
      detail::write_file<ptSU3,ORD>(A,tree*reduce.np(),fname);
    MPI_Barrier(MPI_COMM_WORLD);
#else
    detail::write_file<ptSU3,ORD>(U,tree,fname);
#endif
  }
  template <class CONT>
  inline void write_file(const CONT& c, const std::string& fname){
#ifdef USE_MPI
    comm::Reduce<CONT> reduce(c);
    CONT s = reduce();
    if(reduce.rank()==0)
      detail::write_file<CONT>(s,fname);
    MPI_Barrier(MPI_COMM_WORLD);
#else
    detail::write_file<CONT>(c,fname);
#endif
  }
  template <class CONT>
  inline void write_ptSUN(const CONT& c, const std::string& fname){
#ifdef USE_MPI
    comm::Reduce<CONT> reduce(c);
    CONT s = reduce();
    if(reduce.rank()==0)
      detail::write_ptSUN<CONT>(s,fname);
    MPI_Barrier(MPI_COMM_WORLD);
#else
    detail::write_ptSUN<CONT>(c,fname);
#endif
  }
}


#endif /* _IO_H_ */
