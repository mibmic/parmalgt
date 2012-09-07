#ifndef _IO_H_
#define _IO_H_

#include <iostream>
#include <fstream>
#include <MyMath.h>
#include <vector>
#include <sstream>
#include <uparam.hpp>


// Note the MD5 Copyright notice in the 
// implementation file IO.cc!

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
      param["md5"] = CheckedIo::md5();
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
      CheckedIo::process(c.re);
      CheckedIo::process(c.im);
      os.write(reinterpret_cast<char const*>(&(c.re)), sizeof(double));
      os.write(reinterpret_cast<char const*>(&(c.im)), sizeof(double));
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
            || i->first == "write" || i->first == "seed")
          continue;
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
      is.read(reinterpret_cast<char*>(&(c.re)), sizeof(double));
      is.read(reinterpret_cast<char*>(&(c.im)), sizeof(double));
      CheckedIo::process(c.re);
      CheckedIo::process(c.im);
    }
  private:
    std::ifstream is;
    uparam::Param param;
  };

  ////////////////////////////////////////////////////////////
  // writing binary data to files
  
  inline void to_bin_file(std::ofstream& of, const Cplx& c){
    of.write(reinterpret_cast<const char*>(&c.re), sizeof(double));
    of.write(reinterpret_cast<const char*>(&c.im), sizeof(double));
  }
  template <class ptSU3, int ORD>
  inline void write_file(const ptSU3& U, const Cplx& tree, const std::string& fname){
    std::ofstream of(fname.c_str(), std::ios_base::app |  std::ios_base::binary);
    to_bin_file(of, tree);
    for (int i = 0; i < ORD; ++i)
      to_bin_file(of, U[i].Tr());
    of.close();
  }

  inline void write_file(const std::vector<Cplx> vec, const std::string& fname){
    std::ofstream of(fname.c_str(), std::ios_base::app |  std::ios_base::binary);
    for (int i = 0; i < vec.size(); ++i)
      to_bin_file(of, vec[i]);
    of.close();
  }
  
  ////////////////////////////////////////////////////////////
  // formated cout for the timings/parameters
  template <typename T>
  inline void pretty_print(const std::string& s, const T& d,
                           const std::string& unit = ""){
    std::cout.width(25); 
    std::cout << s; 
    std::cout.width(0);
    std::cout << ": " << d << unit << std::endl;
  };


}


#endif /* _IO_H_ */
