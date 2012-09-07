#ifndef _MY_MATH_H_
#define _MY_MATH_H_

#include <cmath>
#include <iostream>
#include <string.h>
#include <stdio.h>

#include "MyRand.h"
#ifdef SSE
#include <emmintrin.h>
#include <pmmintrin.h>
#endif

const double D3  = 1.0 /  3.0;
const double D6  = 1.0 /  6.0;
const double D7  = 1.0 /  7.0;
const double D9  = 1.0 /  9.0;
const double D12 = 1.0 / 12.0;
const double D18 = 1.0 / 18.0;
const double sD3 =    sqrt(D3);

#define SZ_DB sizeof(double)
#define SZ_CP sizeof(Cplx)

//#define RANF

class Cplx;
class SqMatC;
class Vector;
class SU3;
class CVector;

// COMPLESSI

class Cplx {
public:
#ifdef SSE
    union{
    struct{
      double re;
      double im;
    };
    __m128d m;
  };
#else
  double re,im;
#endif
  

  Cplx (double r = 0., double i=0.) {
      re = r;
      im = i;
  }

  int write(FILE *filept) {
    if (fwrite(&re, SZ_DB, 2, filept))
	  return 1;
      return 0;
  }

  int read(FILE *filept) {
    if (fread(&re, SZ_DB, 2, filept))
	  return 1;
      return 0;
  }

  Cplx operator=(const Cplx& b) { re = b.re; im = b.im; return *this;}
  Cplx operator=(const double& b) { re = b; im = 0; return *this;}

  Cplx operator+(const Cplx& b) const { return Cplx(re+b.re, im+b.im); }
  Cplx operator-(const Cplx& b) const { return Cplx(re-b.re, im-b.im); }
  Cplx operator*(const Cplx& b) const { return Cplx(re*b.re-im*b.im, im*b.re + re*b.im); }
  Cplx operator/(const Cplx& b) const { 
    double mod_b = b.re*b.re+b.im*b.im;
    return Cplx((re * b.re  + im*b.im)/mod_b, (-re * b.im + im*b.re)/mod_b); }

  Cplx operator+(const double& b) const { return Cplx(re+b, im); }
  Cplx operator-(const double& b) const { return Cplx(re-b, im); }
  Cplx operator*(const double& b) const { return Cplx(re*b, b*im); }
  Cplx operator/(const double& b) const { return Cplx(re/b, im/b); }
  

  Cplx operator+=(const Cplx& b) { re += b.re; im +=b.im; return *this;}
  Cplx operator-=(const Cplx& b) { re -= b.re; im -=b.im; return *this;}
  Cplx operator*=(const Cplx& b) {
    Cplx app((re*b.re - im*b.im), (im*b.re + re*b.im));
    (*this) = app;
    return *this;
  }
  Cplx operator/=(const Cplx& b) {
    double mod_b = b.re*b.re+b.im*b.im;
    Cplx app(((re * b.re  + im*b.im)/mod_b), ((im*b.re - re*b.im)/mod_b));
    (*this) = app;
    return *this;
  }

  Cplx operator+=(const double& b) { re += b;          return *this;}
  Cplx operator-=(const double& b) { re -= b;          return *this;}
  Cplx operator*=(const double& b) { re *= b; im *= b; return *this;}
  Cplx operator/=(const double& b) { re /= b; im /=b;  return *this;}

  friend Cplx operator+(const double& a, const Cplx& b) { return Cplx(a+b.re, b.im); }
  friend Cplx operator-(const double& a, const Cplx& b) { return Cplx(a-b.re, -b.im); }
  friend Cplx operator*(const double& a, const Cplx& b) { return Cplx(a*b.re, a*b.im); }
  friend Cplx operator/(const double& a, const Cplx& b) {
    double mod_b = b.re*b.re+b.im*b.im;
    return Cplx((a * b.re)/mod_b, -(a * b.im)/mod_b); }
  
  void prout() {
    if(im >= 0)
      printf("%g + %g*i",re, im);
    else
      printf("%g - %g*i",re,-im);
  };
  
  Cplx operator-() const {  return Cplx(-re, -im); }
  Cplx operator~() const {  return Cplx(re, -im); }
  
  double mod() const { return sqrt(re*re+im*im); }

  bool operator ==(const Cplx& z) const {
    return ((z.re == re)&&(z.im == im));
  }

  bool operator !=(const Cplx& z) const {
    return !((z.re == re)&&(z.im == im));
  }

};

inline double  mod(Cplx);
inline double mod2(Cplx);
inline Cplx   cbrt(Cplx);
inline Cplx    sin(Cplx);
inline Cplx    cos(Cplx);
inline Cplx    tan(Cplx);
Cplx    exp(Cplx);
inline Cplx   asin(Cplx);
inline Cplx   acos(Cplx);
inline Cplx   atan(Cplx);

inline double mod(double x) { return fabs(x); }

inline bool operator==(double, Cplx);
inline bool operator!=(double, Cplx);




inline std::ostream& operator<<(std::ostream& os, const Cplx& c){
  os << "(" << c.re << "," << c.im << ")";
  return os;
}






class SU2 {
  friend class WVector;
  friend SU2 SU2rand();

public:
  Cplx whr[4];

  SU2 (){};

  SU2 (const Cplx *matr);
  
  ~SU2 (){}

  int write(FILE *filept) {
    if(fwrite(&whr, SZ_DB*4*2, 1, filept)){
      return 0;
    }
    return 1;
  }
  
  int read(FILE *filept) {
    if(fread(&whr, SZ_DB*4*2, 1, filept)){
      return 0;
    }
    return 1;
  }
  
  inline SU2 operator=(const SU2& A) {
    whr[0].re = A.whr[0].re;
    whr[0].im = A.whr[0].im;
    whr[1].re = A.whr[1].re;
    whr[1].im = A.whr[1].im;
    whr[2].im = A.whr[2].im;
    whr[2].re = A.whr[2].re;
    whr[3].re = A.whr[3].re;
    whr[3].im = A.whr[3].im;
    return *this;
  }
  
  inline SU2 operator+=(const SU2& A) {
    whr[0].re += A.whr[0].re;
    whr[0].im += A.whr[0].im;
    whr[1].re += A.whr[1].re;
    whr[1].im += A.whr[1].im;
    whr[2].re += A.whr[2].re;
    whr[2].im += A.whr[2].im;
    whr[3].re += A.whr[3].re;
    whr[3].im += A.whr[3].im;
    return *this;
};

  inline SU2 operator-=(const SU2& A) {
    whr[0].re -= A.whr[0].re;
    whr[0].im -= A.whr[0].im;
    whr[1].re -= A.whr[1].re;
    whr[1].im -= A.whr[1].im;
    whr[2].re -= A.whr[2].re;
    whr[2].im -= A.whr[2].im;
    whr[3].re -= A.whr[3].re;
    whr[3].im -= A.whr[3].im;
    return *this;
  };

  inline SU2 operator*=(const SU2& A){
    SU2 res;
    
    res.whr[0] = whr[0] * A.whr[0] + whr[1] * A.whr[2] ;
    res.whr[1] = whr[0] * A.whr[1] + whr[1] * A.whr[3] ;
    res.whr[2] = whr[2] * A.whr[0] + whr[3] * A.whr[2] ;
    res.whr[3] = whr[2] * A.whr[1] + whr[3] * A.whr[3] ;
    
    *this = res;
    return *this;
  }

  void operator*=(const Cplx& z) {
    whr[0] = z * whr[0];
    whr[1] = z * whr[1];
    whr[2] = z * whr[2];
    whr[3] = z * whr[3];
  }
  
  void operator/=(const Cplx& z) {
    Cplx den(1./z);
    whr[0] = den * whr[0];
    whr[1] = den * whr[1];
    whr[2] = den * whr[2];
    whr[3] = den * whr[3];
  }

  inline SU2 operator+(const SU2& A) const{
    SU2 res;
    res.whr[0].re = whr[0].re + A.whr[0].re;
    res.whr[0].im = whr[0].im + A.whr[0].im;
    res.whr[1].re = whr[1].re + A.whr[1].re;
    res.whr[1].im = whr[1].im + A.whr[1].im;
    res.whr[2].re = whr[2].re + A.whr[2].re;
    res.whr[2].im = whr[2].im + A.whr[2].im;
    res.whr[3].re = whr[3].re + A.whr[3].re;
    res.whr[3].im = whr[3].im + A.whr[3].im;
    return res;
  };
  
  inline SU2 operator-(const SU2& A) const {
    SU2 res;
    res.whr[0].re = whr[0].re - A.whr[0].re;
    res.whr[0].im = whr[0].im - A.whr[0].im;
    res.whr[1].re = whr[1].re - A.whr[1].re;
    res.whr[1].im = whr[1].im - A.whr[1].im;
    res.whr[2].re = whr[2].re - A.whr[2].re;
    res.whr[2].im = whr[2].im - A.whr[2].im;
    res.whr[3].re = whr[3].re - A.whr[3].re;
    res.whr[3].im = whr[3].im - A.whr[3].im;
    return res;
  };

  inline SU2 operator*(const SU2& A) const{
    SU2 res;
    res.whr[0] = whr[0] * A.whr[0] + whr[1] * A.whr[2] ;
    res.whr[1] = whr[0] * A.whr[1] + whr[1] * A.whr[3] ;
    res.whr[2] = whr[2] * A.whr[0] + whr[3] * A.whr[2] ;
    res.whr[3] = whr[2] * A.whr[1] + whr[3] * A.whr[3] ;
    return res;
  };
    

  inline SU2 operator*(const Cplx& z) const{
    SU2 res;
    res.whr[0] = z * whr[0];
    res.whr[1] = z * whr[1];
    res.whr[2] = z * whr[2];
    res.whr[3] = z * whr[3];
    return res;
  };

  inline SU2 operator/(const Cplx& z) const {
    SU2 res;
    Cplx den(1./z);
    res.whr[0] = den * whr[0];
    res.whr[1] = den * whr[1];
    res.whr[2] = den * whr[2];
    res.whr[3] = den * whr[3];
    return res;
  };

  inline SU2 operator*(const double& x) const {
    SU2 res;
    res.whr[0].re = x * whr[0].re;
    res.whr[0].im = x * whr[0].im;
    res.whr[1].re = x * whr[1].re;
    res.whr[1].im = x * whr[1].im;
    res.whr[2].re = x * whr[2].re;
    res.whr[2].im = x * whr[2].im;
    res.whr[3].re = x * whr[3].re;
    res.whr[3].im = x * whr[3].im;
    return res;
  };

  inline SU2 operator/(const double x) const {
    SU2 res;
    double den = 1./x;
    res.whr[0].re = den * whr[0].re;
    res.whr[0].im = den * whr[0].im;
    res.whr[1].re = den * whr[1].re;
    res.whr[1].im = den * whr[1].im;
    res.whr[2].re = den * whr[2].re;
    res.whr[2].im = den * whr[2].im;
    res.whr[3].re = den * whr[3].re;
    res.whr[3].im = den * whr[3].im;
    return res;
  };


  inline friend  SU2 operator*(const Cplx& z, const SU2& A) {
    SU2 res;
    res.whr[0].re = z.re * A.whr[0].re - z.im * A.whr[0].im;
    res.whr[0].im = z.re * A.whr[0].im + z.im * A.whr[0].re;
    res.whr[1].re = z.re * A.whr[1].re - z.im * A.whr[1].im;
    res.whr[1].im = z.re * A.whr[1].im + z.im * A.whr[1].re;
    res.whr[2].re = z.re * A.whr[2].re - z.im * A.whr[2].im;
    res.whr[2].im = z.re * A.whr[2].im + z.im * A.whr[2].re;
    res.whr[3].re = z.re * A.whr[3].re - z.im * A.whr[3].im;
    res.whr[3].im = z.re * A.whr[3].im + z.im * A.whr[3].re;
    return res;
  }


  inline friend  SU2 operator*(const double& x, const SU2& A) {
    SU2 res;
    res.whr[0].re = x * A.whr[0].re;
    res.whr[0].im = x * A.whr[0].im;
    res.whr[1].re = x * A.whr[1].re;
    res.whr[1].im = x * A.whr[1].im;
    res.whr[2].re = x * A.whr[2].re;
    res.whr[2].im = x * A.whr[2].im;
    res.whr[3].re = x * A.whr[3].re;
    res.whr[3].im = x * A.whr[3].im;
    return res;
  }

  Cplx Tr() const { return (whr[0]+whr[3]); }

  Cplx Det() const {  return ( whr[0]*whr[3]-whr[1]*whr[2]);  }

  SU2 operator~() const {
    SU2 res;
    res.whr[0].re =  whr[0].re;
    res.whr[0].im = -whr[0].im;
    res.whr[1].re =  whr[1].re;
    res.whr[1].im = -whr[1].im;
    res.whr[2].re =  whr[2].re;
    res.whr[2].im = -whr[2].im;
    res.whr[3].re =  whr[3].re;
    res.whr[3].im = -whr[3].im;
    return res;
  }

  SU2 operator-() const{
    SU2 res;
    res.whr[0].re = -whr[0].re;
    res.whr[0].im = -whr[0].im;
    res.whr[1].re = -whr[1].re;
    res.whr[1].im = -whr[1].im;
    res.whr[2].re = -whr[2].re;
    res.whr[2].im = -whr[2].im;
    res.whr[3].re = -whr[3].re;
    res.whr[3].im = -whr[3].im;
    return res;
}
  
  void tra(){
    Cplx app;
    app = whr[1]; 
    whr[1] = whr[2]; 
    whr[2] = app;
  }

  void dag() {
    Cplx app = whr[1];
    whr[0].im = -whr[0].im;
    whr[1].re =  whr[2].re;
    whr[1].im = -whr[2].im;
    whr[2].re =  app.re;
    whr[2].im = -app.im;
    whr[3].im = -whr[3].im;
  }

  void reH();

  void prout() {
    whr[0].prout();
    printf("\t");
    whr[1].prout();
    printf("\n");
    whr[2].prout();
    printf("\t");
    whr[3].prout();
    printf("\n\n");
  }

  bool operator==(const SU2& A) { return (!memcmp(whr, A.whr, 64)); }
  bool operator!=(const SU2& A) { return ( memcmp(whr, A.whr, 64)); }

};


inline Cplx Tr(const SU2& A) { return (A.whr[0] + A.whr[3]); }

inline Cplx Det(const SU2& A){  return ( A.whr[0]*A.whr[3]-A.whr[1]*A.whr[2]); }

inline SU2 tra(const SU2& A) {
  SU2 res;
  res.whr[0] = A.whr[0];
  res.whr[1] = A.whr[2];
  res.whr[2] = A.whr[1];
  res.whr[3] = A.whr[3];
  return res;
}

inline SU2 dag(const SU2& A) {
  SU2 res;
  res.whr[0].re =  A.whr[0].re;
  res.whr[0].im = -A.whr[0].im;
  res.whr[1].re =  A.whr[2].re;
  res.whr[1].im = -A.whr[2].im;
  res.whr[2].re =  A.whr[1].re;
  res.whr[2].im = -A.whr[1].im;
  res.whr[3].re =  A.whr[3].re;
  res.whr[3].im = -A.whr[3].im;
  return res;
}

SU2 SU2rand();









class SU3 {
  friend class CVector;
  friend SU3 SU3rand();
private:

public:
  Cplx whr[9];

  // Added by Dirk on Jan. 11, 2012
  // easy access to the elements through the () operator
  // const and non-const versions
  const Cplx& operator()(int i, int j) const  { return whr[i*3 + j]; }
  Cplx& operator()(int i, int j)  { return whr[i*3 + j]; }
  const Cplx& operator[](int i) const { return whr[i]; }
  Cplx& operator[](int i) { return whr[i]; }
  // iterators
  typedef Cplx const * const_iterator;
  typedef Cplx * iterator;
  iterator begin(void) {return whr;}
  const_iterator begin(void) const {return whr;}
  iterator end(void) {return whr + 9;}
  const_iterator end(void) const {return whr + 9;}
  // DH added on Feb. 7, 2012
  // Norm
  double Norm(void) const {
    double norm = 0;
    for (int i = 0; i < 9; ++i)
      norm += whr[i].re*whr[i].re + whr[i].im*whr[i].im;
    return sqrt(norm);
  }

  SU3 (){};

  SU3 (const Cplx *matr) {
    whr[0].re = matr[0].re;
    whr[0].im = matr[0].im;
    whr[1].re = matr[1].re;
    whr[1].im = matr[1].im;
    whr[2].re = matr[2].re;
    whr[2].im = matr[2].im;
    whr[3].re = matr[3].re;
    whr[3].im = matr[3].im;
    whr[4].re = matr[4].re;
    whr[4].im = matr[4].im;
    whr[5].re = matr[5].re;
    whr[5].im = matr[5].im;
    whr[6].re = matr[6].re;
    whr[6].im = matr[6].im;
    whr[7].re = matr[7].re;
    whr[7].im = matr[7].im;
    whr[8].re = matr[8].re;
    whr[8].im = matr[8].im;
  }

  
  ~SU3 (){}

  int write(FILE *filept) {
    if(fwrite(&whr, SZ_DB*9*2, 1, filept)){
      return 0;
    }
    return 1;
  }
  
  int read(FILE *filept) {
    if(fread(&whr, SZ_DB*9*2, 1, filept)){
      return 0;
    }
    return 1;
  }
  
  inline SU3 operator=(const SU3& A) {
    whr[0].re = A.whr[0].re; 
    whr[0].im = A.whr[0].im; 
    whr[1].re = A.whr[1].re; 
    whr[1].im = A.whr[1].im; 
    whr[2].im = A.whr[2].im; 
    whr[2].re = A.whr[2].re; 
    whr[3].re = A.whr[3].re; 
    whr[3].im = A.whr[3].im; 
    whr[4].re = A.whr[4].re; 
    whr[4].im = A.whr[4].im; 
    whr[5].re = A.whr[5].re; 
    whr[5].im = A.whr[5].im; 
    whr[6].re = A.whr[6].re; 
    whr[6].im = A.whr[6].im; 
    whr[7].re = A.whr[7].re; 
    whr[7].im = A.whr[7].im; 
    whr[8].re = A.whr[8].re; 
    whr[8].im = A.whr[8].im; 

    return *this;
  }
  
  inline SU3 operator+=(const SU3& A) {
    whr[0].re += A.whr[0].re;
    whr[0].im += A.whr[0].im;
    whr[1].re += A.whr[1].re;
    whr[1].im += A.whr[1].im;
    whr[2].re += A.whr[2].re;
    whr[2].im += A.whr[2].im;
    whr[3].re += A.whr[3].re;
    whr[3].im += A.whr[3].im;
    whr[4].re += A.whr[4].re;
    whr[4].im += A.whr[4].im;
    whr[5].re += A.whr[5].re;
    whr[5].im += A.whr[5].im;
    whr[6].re += A.whr[6].re;
    whr[6].im += A.whr[6].im;
    whr[7].im += A.whr[7].im;
    whr[7].re += A.whr[7].re;
    whr[8].re += A.whr[8].re;
    whr[8].im += A.whr[8].im;
    return *this;
};

  inline SU3 operator-=(const SU3& A) {  
    whr[0].re -= A.whr[0].re;
    whr[0].im -= A.whr[0].im;
    whr[1].re -= A.whr[1].re;
    whr[1].im -= A.whr[1].im;
    whr[2].re -= A.whr[2].re;
    whr[2].im -= A.whr[2].im;
    whr[3].re -= A.whr[3].re;
    whr[3].im -= A.whr[3].im;
    whr[4].re -= A.whr[4].re;
    whr[4].im -= A.whr[4].im;
    whr[5].re -= A.whr[5].re;
    whr[5].im -= A.whr[5].im;
    whr[6].re -= A.whr[6].re;
    whr[6].im -= A.whr[6].im;
    whr[7].im -= A.whr[7].im;
    whr[7].re -= A.whr[7].re;
    whr[8].re -= A.whr[8].re;
    whr[8].im -= A.whr[8].im;
    return *this;
  };

  inline SU3 operator*=(const SU3& A){
    Cplx tmp[9];
    
    tmp[0].re= whr[0].re*A.whr[0].re-whr[0].im*A.whr[0].im +
      whr[1].re*A.whr[3].re-whr[1].im*A.whr[3].im +
      whr[2].re*A.whr[6].re-whr[2].im*A.whr[6].im ;
    tmp[0].im= whr[0].re*A.whr[0].im+whr[0].im*A.whr[0].re + 
      whr[1].re*A.whr[3].im+whr[1].im*A.whr[3].re + 
      whr[2].re*A.whr[6].im+whr[2].im*A.whr[6].re ;    
    tmp[1].re= whr[0].re*A.whr[1].re-whr[0].im*A.whr[1].im
      +whr[1].re*A.whr[4].re-whr[1].im*A.whr[4].im	  
      +whr[2].re*A.whr[7].re-whr[2].im*A.whr[7].im;
    tmp[1].im= whr[0].re*A.whr[1].im+whr[0].im*A.whr[1].re
      +whr[1].re*A.whr[4].im+whr[1].im*A.whr[4].re	  
      +whr[2].re*A.whr[7].im+whr[2].im*A.whr[7].re;	  
    tmp[2].re= whr[0].re*A.whr[2].re-whr[0].im*A.whr[2].im
      +whr[1].re*A.whr[5].re-whr[1].im*A.whr[5].im	  
      +whr[2].re*A.whr[8].re-whr[2].im*A.whr[8].im;	  
    tmp[2].im= whr[0].re*A.whr[2].im+whr[0].im*A.whr[2].re
      +whr[1].re*A.whr[5].im+whr[1].im*A.whr[5].re	  
      +whr[2].re*A.whr[8].im+whr[2].im*A.whr[8].re;	  
    tmp[3].re= whr[3].re*A.whr[0].re-whr[3].im*A.whr[0].im
      +whr[4].re*A.whr[3].re-whr[4].im*A.whr[3].im	  
      +whr[5].re*A.whr[6].re-whr[5].im*A.whr[6].im;	  
    tmp[3].im= whr[3].re*A.whr[0].im+whr[3].im*A.whr[0].re
      +whr[4].re*A.whr[3].im+whr[4].im*A.whr[3].re	  
      +whr[5].re*A.whr[6].im+whr[5].im*A.whr[6].re;	  
    tmp[4].re= whr[3].re*A.whr[1].re-whr[3].im*A.whr[1].im
      +whr[4].re*A.whr[4].re-whr[4].im*A.whr[4].im	  
      +whr[5].re*A.whr[7].re-whr[5].im*A.whr[7].im;	  
    tmp[4].im= whr[3].re*A.whr[1].im+whr[3].im*A.whr[1].re
      +whr[4].re*A.whr[4].im+whr[4].im*A.whr[4].re	  
      +whr[5].re*A.whr[7].im+whr[5].im*A.whr[7].re;	  
    tmp[5].re= whr[3].re*A.whr[2].re-whr[3].im*A.whr[2].im
      +whr[4].re*A.whr[5].re-whr[4].im*A.whr[5].im	  
      +whr[5].re*A.whr[8].re-whr[5].im*A.whr[8].im;	  
    tmp[5].im= whr[3].re*A.whr[2].im+whr[3].im*A.whr[2].re
      +whr[4].re*A.whr[5].im+whr[4].im*A.whr[5].re	  
      +whr[5].re*A.whr[8].im+whr[5].im*A.whr[8].re;	  
    tmp[6].re= whr[6].re*A.whr[0].re-whr[6].im*A.whr[0].im
      +whr[7].re*A.whr[3].re-whr[7].im*A.whr[3].im	  
      +whr[8].re*A.whr[6].re-whr[8].im*A.whr[6].im;	  
    tmp[6].im= whr[6].re*A.whr[0].im+whr[6].im*A.whr[0].re
      +whr[7].re*A.whr[3].im+whr[7].im*A.whr[3].re	  
      +whr[8].re*A.whr[6].im+whr[8].im*A.whr[6].re;	  
    tmp[7].re= whr[6].re*A.whr[1].re-whr[6].im*A.whr[1].im
      +whr[7].re*A.whr[4].re-whr[7].im*A.whr[4].im	  
      +whr[8].re*A.whr[7].re-whr[8].im*A.whr[7].im;	  
    tmp[7].im= whr[6].re*A.whr[1].im+whr[6].im*A.whr[1].re
      +whr[7].re*A.whr[4].im+whr[7].im*A.whr[4].re	  
      +whr[8].re*A.whr[7].im+whr[8].im*A.whr[7].re;	  
    tmp[8].re= whr[6].re*A.whr[2].re-whr[6].im*A.whr[2].im
      +whr[7].re*A.whr[5].re-whr[7].im*A.whr[5].im	  
      +whr[8].re*A.whr[8].re-whr[8].im*A.whr[8].im;	  
    tmp[8].im= whr[6].re*A.whr[2].im+whr[6].im*A.whr[2].re
      +whr[7].re*A.whr[5].im+whr[7].im*A.whr[5].re	  
      +whr[8].re*A.whr[8].im+whr[8].im*A.whr[8].re;
    
    whr[0].re = tmp[0].re; 
    whr[0].im = tmp[0].im; 
    whr[1].re = tmp[1].re; 
    whr[1].im = tmp[1].im; 
    whr[2].re = tmp[2].re; 
    whr[2].im = tmp[2].im; 
    whr[3].re = tmp[3].re; 
    whr[3].im = tmp[3].im; 
    whr[4].re = tmp[4].re; 
    whr[4].im = tmp[4].im; 
    whr[5].re = tmp[5].re; 
    whr[5].im = tmp[5].im; 
    whr[6].re = tmp[6].re; 
    whr[6].im = tmp[6].im; 
    whr[7].re = tmp[7].re; 
    whr[7].im = tmp[7].im; 
    whr[8].re = tmp[8].re; 
    whr[8].im = tmp[8].im; 

    return *this;
  }

  //
  // This little execise is supposed to show a bug in the *=  operator
  // defined below. The plain multiplication works ...
  //

  //*/
  // My Version (DH)

  SU3& operator *= (const Cplx& z){
    for (iterator i = begin(); i != end(); ++i) *i *= z;
    return *this;
  }
  /*/
  // Old Version -- BUGGY
  
  void operator*=(const Cplx& z) {
    // The problem lies here ...
    // we should insert a
    // double tmp = whr[0].re;
    whr[0].re = z.re * whr[0].re - z.im * whr[0].im;
    // and replace this line with
    whr[0].im = z.re * whr[0].im + z.im * whr[0].re;
    //whr[0].im = z.re * whr[0].im + z.im * tmp;
    //                                   --------
    whr[1].re = z.re * whr[1].re - z.im * whr[1].im;
    whr[1].im = z.re * whr[1].im + z.im * whr[1].re;
    whr[2].re = z.re * whr[2].re - z.im * whr[2].im;
    whr[2].im = z.re * whr[2].im + z.im * whr[2].re;
    whr[3].re = z.re * whr[3].re - z.im * whr[3].im;
    whr[3].im = z.re * whr[3].im + z.im * whr[3].re;
    whr[4].re = z.re * whr[4].re - z.im * whr[4].im;
    whr[4].im = z.re * whr[4].im + z.im * whr[4].re;
    whr[5].re = z.re * whr[5].re - z.im * whr[5].im;
    whr[5].im = z.re * whr[5].im + z.im * whr[5].re;
    whr[6].re = z.re * whr[6].re - z.im * whr[6].im;
    whr[6].im = z.re * whr[6].im + z.im * whr[6].re;
    whr[7].re = z.re * whr[7].re - z.im * whr[7].im;
    whr[7].im = z.re * whr[7].im + z.im * whr[7].re;
    whr[8].re = z.re * whr[8].re - z.im * whr[8].im;
    whr[8].im = z.re * whr[8].im + z.im * whr[8].re;    
  }
  //*/
  void operator/=(const Cplx& z) { 
    Cplx den(1./z);
    whr[0].re = den.re * whr[0].re - den.im * whr[0].im;
    whr[0].im = den.re * whr[0].im + den.im * whr[0].re;
    whr[1].re = den.re * whr[1].re - den.im * whr[1].im;
    whr[1].im = den.re * whr[1].im + den.im * whr[1].re;
    whr[2].re = den.re * whr[2].re - den.im * whr[2].im;
    whr[2].im = den.re * whr[2].im + den.im * whr[2].re;
    whr[3].re = den.re * whr[3].re - den.im * whr[3].im;
    whr[3].im = den.re * whr[3].im + den.im * whr[3].re;
    whr[4].re = den.re * whr[4].re - den.im * whr[4].im;
    whr[4].im = den.re * whr[4].im + den.im * whr[4].re;
    whr[5].re = den.re * whr[5].re - den.im * whr[5].im;
    whr[5].im = den.re * whr[5].im + den.im * whr[5].re;
    whr[6].re = den.re * whr[6].re - den.im * whr[6].im;
    whr[6].im = den.re * whr[6].im + den.im * whr[6].re;
    whr[7].re = den.re * whr[7].re - den.im * whr[7].im;
    whr[7].im = den.re * whr[7].im + den.im * whr[7].re;
    whr[8].re = den.re * whr[8].re - den.im * whr[8].im;
    whr[8].im = den.re * whr[8].im + den.im * whr[8].re;
}

  inline SU3 operator+(const SU3& A) const {  
    SU3 res;
    res.whr[0].re = whr[0].re + A.whr[0].re; 
    res.whr[0].im = whr[0].im + A.whr[0].im; 
    res.whr[1].re = whr[1].re + A.whr[1].re; 
    res.whr[1].im = whr[1].im + A.whr[1].im; 
    res.whr[2].re = whr[2].re + A.whr[2].re; 
    res.whr[2].im = whr[2].im + A.whr[2].im; 
    res.whr[3].re = whr[3].re + A.whr[3].re; 
    res.whr[3].im = whr[3].im + A.whr[3].im; 
    res.whr[4].re = whr[4].re + A.whr[4].re; 
    res.whr[4].im = whr[4].im + A.whr[4].im; 
    res.whr[5].re = whr[5].re + A.whr[5].re; 
    res.whr[5].im = whr[5].im + A.whr[5].im; 
    res.whr[6].re = whr[6].re + A.whr[6].re; 
    res.whr[6].im = whr[6].im + A.whr[6].im; 
    res.whr[7].re = whr[7].re + A.whr[7].re; 
    res.whr[7].im = whr[7].im + A.whr[7].im; 
    res.whr[8].re = whr[8].re + A.whr[8].re; 
    res.whr[8].im = whr[8].im + A.whr[8].im; 
    return res;
  };
  
  inline SU3 operator-(const SU3& A) const {  
    SU3 res;
    res.whr[0].re = whr[0].re - A.whr[0].re; 
    res.whr[0].im = whr[0].im - A.whr[0].im; 
    res.whr[1].re = whr[1].re - A.whr[1].re; 
    res.whr[1].im = whr[1].im - A.whr[1].im; 
    res.whr[2].re = whr[2].re - A.whr[2].re; 
    res.whr[2].im = whr[2].im - A.whr[2].im; 
    res.whr[3].re = whr[3].re - A.whr[3].re; 
    res.whr[3].im = whr[3].im - A.whr[3].im; 
    res.whr[4].re = whr[4].re - A.whr[4].re; 
    res.whr[4].im = whr[4].im - A.whr[4].im; 
    res.whr[5].re = whr[5].re - A.whr[5].re; 
    res.whr[5].im = whr[5].im - A.whr[5].im; 
    res.whr[6].re = whr[6].re - A.whr[6].re; 
    res.whr[6].im = whr[6].im - A.whr[6].im; 
    res.whr[7].re = whr[7].re - A.whr[7].re; 
    res.whr[7].im = whr[7].im - A.whr[7].im; 
    res.whr[8].re = whr[8].re - A.whr[8].re; 
    res.whr[8].im = whr[8].im - A.whr[8].im; 
    return res;
  };

  inline SU3 operator*(const SU3& A) const {
    SU3 res;
    res.whr[0].re= whr[0].re*A.whr[0].re-whr[0].im*A.whr[0].im
      +whr[1].re*A.whr[3].re-whr[1].im*A.whr[3].im		
      +whr[2].re*A.whr[6].re-whr[2].im*A.whr[6].im;		
    res.whr[0].im= whr[0].re*A.whr[0].im+whr[0].im*A.whr[0].re
      +whr[1].re*A.whr[3].im+whr[1].im*A.whr[3].re		
      +whr[2].re*A.whr[6].im+whr[2].im*A.whr[6].re;		
    res.whr[1].re= whr[0].re*A.whr[1].re-whr[0].im*A.whr[1].im
      +whr[1].re*A.whr[4].re-whr[1].im*A.whr[4].im		
      +whr[2].re*A.whr[7].re-whr[2].im*A.whr[7].im;		
    res.whr[1].im= whr[0].re*A.whr[1].im+whr[0].im*A.whr[1].re
      +whr[1].re*A.whr[4].im+whr[1].im*A.whr[4].re		
      +whr[2].re*A.whr[7].im+whr[2].im*A.whr[7].re;		
    res.whr[2].re= whr[0].re*A.whr[2].re-whr[0].im*A.whr[2].im
      +whr[1].re*A.whr[5].re-whr[1].im*A.whr[5].im		
      +whr[2].re*A.whr[8].re-whr[2].im*A.whr[8].im;		
    res.whr[2].im= whr[0].re*A.whr[2].im+whr[0].im*A.whr[2].re
      +whr[1].re*A.whr[5].im+whr[1].im*A.whr[5].re		
      +whr[2].re*A.whr[8].im+whr[2].im*A.whr[8].re;		
    res.whr[3].re= whr[3].re*A.whr[0].re-whr[3].im*A.whr[0].im
      +whr[4].re*A.whr[3].re-whr[4].im*A.whr[3].im		
      +whr[5].re*A.whr[6].re-whr[5].im*A.whr[6].im;		
    res.whr[3].im= whr[3].re*A.whr[0].im+whr[3].im*A.whr[0].re
      +whr[4].re*A.whr[3].im+whr[4].im*A.whr[3].re		
      +whr[5].re*A.whr[6].im+whr[5].im*A.whr[6].re;		
    res.whr[4].re= whr[3].re*A.whr[1].re-whr[3].im*A.whr[1].im
      +whr[4].re*A.whr[4].re-whr[4].im*A.whr[4].im		
      +whr[5].re*A.whr[7].re-whr[5].im*A.whr[7].im;		
    res.whr[4].im= whr[3].re*A.whr[1].im+whr[3].im*A.whr[1].re
      +whr[4].re*A.whr[4].im+whr[4].im*A.whr[4].re		
      +whr[5].re*A.whr[7].im+whr[5].im*A.whr[7].re;		
    res.whr[5].re= whr[3].re*A.whr[2].re-whr[3].im*A.whr[2].im
      +whr[4].re*A.whr[5].re-whr[4].im*A.whr[5].im		
      +whr[5].re*A.whr[8].re-whr[5].im*A.whr[8].im;		
    res.whr[5].im= whr[3].re*A.whr[2].im+whr[3].im*A.whr[2].re
      +whr[4].re*A.whr[5].im+whr[4].im*A.whr[5].re		
      +whr[5].re*A.whr[8].im+whr[5].im*A.whr[8].re;		
    res.whr[6].re= whr[6].re*A.whr[0].re-whr[6].im*A.whr[0].im
      +whr[7].re*A.whr[3].re-whr[7].im*A.whr[3].im		
      +whr[8].re*A.whr[6].re-whr[8].im*A.whr[6].im;		
    res.whr[6].im= whr[6].re*A.whr[0].im+whr[6].im*A.whr[0].re
      +whr[7].re*A.whr[3].im+whr[7].im*A.whr[3].re		
      +whr[8].re*A.whr[6].im+whr[8].im*A.whr[6].re;		
    res.whr[7].re= whr[6].re*A.whr[1].re-whr[6].im*A.whr[1].im
      +whr[7].re*A.whr[4].re-whr[7].im*A.whr[4].im		
      +whr[8].re*A.whr[7].re-whr[8].im*A.whr[7].im;		
    res.whr[7].im= whr[6].re*A.whr[1].im+whr[6].im*A.whr[1].re
      +whr[7].re*A.whr[4].im+whr[7].im*A.whr[4].re		
      +whr[8].re*A.whr[7].im+whr[8].im*A.whr[7].re;		
    res.whr[8].re= whr[6].re*A.whr[2].re-whr[6].im*A.whr[2].im
      +whr[7].re*A.whr[5].re-whr[7].im*A.whr[5].im		
      +whr[8].re*A.whr[8].re-whr[8].im*A.whr[8].im;		
    res.whr[8].im= whr[6].re*A.whr[2].im+whr[6].im*A.whr[2].re
      +whr[7].re*A.whr[5].im+whr[7].im*A.whr[5].re 
      +whr[8].re*A.whr[8].im+whr[8].im*A.whr[8].re;

    return res;  
  };
    

  inline SU3 operator*(const Cplx& z) const {
    SU3 res;
    res.whr[0].re = z.re * whr[0].re - z.im * whr[0].im;
    res.whr[0].im = z.re * whr[0].im + z.im * whr[0].re;
    res.whr[1].re = z.re * whr[1].re - z.im * whr[1].im;
    res.whr[1].im = z.re * whr[1].im + z.im * whr[1].re;
    res.whr[2].re = z.re * whr[2].re - z.im * whr[2].im;
    res.whr[2].im = z.re * whr[2].im + z.im * whr[2].re;
    res.whr[3].im = z.re * whr[3].im + z.im * whr[3].re;
    res.whr[3].re = z.re * whr[3].re - z.im * whr[3].im;
    res.whr[4].re = z.re * whr[4].re - z.im * whr[4].im;
    res.whr[4].im = z.re * whr[4].im + z.im * whr[4].re;
    res.whr[5].re = z.re * whr[5].re - z.im * whr[5].im;
    res.whr[5].im = z.re * whr[5].im + z.im * whr[5].re;
    res.whr[6].re = z.re * whr[6].re - z.im * whr[6].im;
    res.whr[6].im = z.re * whr[6].im + z.im * whr[6].re;
    res.whr[7].re = z.re * whr[7].re - z.im * whr[7].im;
    res.whr[7].im = z.re * whr[7].im + z.im * whr[7].re;
    res.whr[8].re = z.re * whr[8].re - z.im * whr[8].im;
    res.whr[8].im = z.re * whr[8].im + z.im * whr[8].re;
    
  return res;
  };  

  inline SU3 operator/(const Cplx& z) const {
    SU3 res;
    Cplx den(1./z);
    res.whr[0].re = den.re * whr[0].re - den.im * whr[0].im;
    res.whr[0].im = den.re * whr[0].im + den.im * whr[0].re;
    res.whr[1].re = den.re * whr[1].re - den.im * whr[1].im;
    res.whr[1].im = den.re * whr[1].im + den.im * whr[1].re;
    res.whr[2].re = den.re * whr[2].re - den.im * whr[2].im;
    res.whr[2].im = den.re * whr[2].im + den.im * whr[2].re;
    res.whr[3].re = den.re * whr[3].re - den.im * whr[3].im;
    res.whr[3].im = den.re * whr[3].im + den.im * whr[3].re;
    res.whr[4].re = den.re * whr[4].re - den.im * whr[4].im;
    res.whr[4].im = den.re * whr[4].im + den.im * whr[4].re;
    res.whr[5].re = den.re * whr[5].re - den.im * whr[5].im;
    res.whr[5].im = den.re * whr[5].im + den.im * whr[5].re;
    res.whr[6].im = den.re * whr[6].im + den.im * whr[6].re;
    res.whr[6].re = den.re * whr[6].re - den.im * whr[6].im;
    res.whr[7].re = den.re * whr[7].re - den.im * whr[7].im;
    res.whr[7].im = den.re * whr[7].im + den.im * whr[7].re;
    res.whr[8].re = den.re * whr[8].re - den.im * whr[8].im;
    res.whr[8].im = den.re * whr[8].im + den.im * whr[8].re;
    return res;
  };  

  inline SU3 operator*(const double& x) const {
    SU3 res;
    res.whr[0].re = x * whr[0].re;
    res.whr[0].im = x * whr[0].im;
    res.whr[1].re = x * whr[1].re;
    res.whr[1].im = x * whr[1].im;
    res.whr[2].re = x * whr[2].re;
    res.whr[2].im = x * whr[2].im;
    res.whr[3].re = x * whr[3].re;
    res.whr[3].im = x * whr[3].im;
    res.whr[4].re = x * whr[4].re;
    res.whr[4].im = x * whr[4].im;
    res.whr[5].re = x * whr[5].re;
    res.whr[5].im = x * whr[5].im;
    res.whr[6].re = x * whr[6].re;
    res.whr[6].im = x * whr[6].im;
    res.whr[7].re = x * whr[7].re;
    res.whr[7].im = x * whr[7].im;
    res.whr[8].re = x * whr[8].re;
    res.whr[8].im = x * whr[8].im;
    
    return res;
  };  

  inline SU3 operator/(const double& x) const {
    SU3 res;
    double den = 1./x;
    res.whr[0].re = den * whr[0].re;
    res.whr[0].im = den * whr[0].im;
    res.whr[1].re = den * whr[1].re;
    res.whr[1].im = den * whr[1].im;
    res.whr[2].re = den * whr[2].re;
    res.whr[2].im = den * whr[2].im;
    res.whr[3].re = den * whr[3].re;
    res.whr[3].im = den * whr[3].im;
    res.whr[4].re = den * whr[4].re;
    res.whr[4].im = den * whr[4].im;
    res.whr[5].re = den * whr[5].re;
    res.whr[5].im = den * whr[5].im;
    res.whr[6].re = den * whr[6].re;
    res.whr[6].im = den * whr[6].im;
    res.whr[7].re = den * whr[7].re;
    res.whr[7].im = den * whr[7].im;
    res.whr[8].re = den * whr[8].re;		    
    res.whr[8].im = den * whr[8].im;
    return res;
  };  


  inline friend SU3 operator*(const Cplx& z, const SU3& A) {
    SU3 res;
    res.whr[0].re = z.re * A.whr[0].re - z.im * A.whr[0].im;
    res.whr[0].im = z.re * A.whr[0].im + z.im * A.whr[0].re;
    res.whr[1].re = z.re * A.whr[1].re - z.im * A.whr[1].im;
    res.whr[1].im = z.re * A.whr[1].im + z.im * A.whr[1].re;
    res.whr[2].re = z.re * A.whr[2].re - z.im * A.whr[2].im;
    res.whr[2].im = z.re * A.whr[2].im + z.im * A.whr[2].re;
    res.whr[3].re = z.re * A.whr[3].re - z.im * A.whr[3].im;
    res.whr[3].im = z.re * A.whr[3].im + z.im * A.whr[3].re;
    res.whr[4].re = z.re * A.whr[4].re - z.im * A.whr[4].im;
    res.whr[4].im = z.re * A.whr[4].im + z.im * A.whr[4].re;
    res.whr[5].re = z.re * A.whr[5].re - z.im * A.whr[5].im;
    res.whr[5].im = z.re * A.whr[5].im + z.im * A.whr[5].re;
    res.whr[6].re = z.re * A.whr[6].re - z.im * A.whr[6].im;
    res.whr[6].im = z.re * A.whr[6].im + z.im * A.whr[6].re;
    res.whr[7].re = z.re * A.whr[7].re - z.im * A.whr[7].im;
    res.whr[7].im = z.re * A.whr[7].im + z.im * A.whr[7].re;
    res.whr[8].re = z.re * A.whr[8].re - z.im * A.whr[8].im;
    res.whr[8].im = z.re * A.whr[8].im + z.im * A.whr[8].re;
    return res;
  }


  inline friend  SU3 operator*(const double& x, const SU3& A) {
    SU3 res;
    res.whr[0].re = x * A.whr[0].re;
    res.whr[0].im = x * A.whr[0].im;
    res.whr[1].re = x * A.whr[1].re;
    res.whr[1].im = x * A.whr[1].im;
    res.whr[2].re = x * A.whr[2].re;
    res.whr[2].im = x * A.whr[2].im;
    res.whr[3].re = x * A.whr[3].re;
    res.whr[3].im = x * A.whr[3].im;
    res.whr[4].re = x * A.whr[4].re;
    res.whr[4].im = x * A.whr[4].im;
    res.whr[5].re = x * A.whr[5].re;
    res.whr[5].im = x * A.whr[5].im;
    res.whr[6].re = x * A.whr[6].re;
    res.whr[6].im = x * A.whr[6].im;
    res.whr[7].re = x * A.whr[7].re;
    res.whr[7].im = x * A.whr[7].im;
    res.whr[8].re = x * A.whr[8].re;
    res.whr[8].im = x * A.whr[8].im;
    return res;
  }


  inline CVector operator*(const CVector&) const; // SU3*CVector
  inline CVector operator^(const CVector&) const; // dag(SU3)*CVector


  Cplx Tr() const { return (whr[0]+whr[4]+whr[8]);}

  Cplx Det(){  
    return (whr[0]*(whr[4]*whr[8]-whr[5]*whr[7]) -
	    whr[3]*(whr[1]*whr[8]-whr[2]*whr[7]) +
	    whr[6]*(whr[1]*whr[5]-whr[2]*whr[4]));
  }

  friend Cplx Det(const SU3& A){
    return (A.whr[0]*(A.whr[4]*A.whr[8]-A.whr[5]*A.whr[7]) -
	    A.whr[3]*(A.whr[1]*A.whr[8]-A.whr[2]*A.whr[7]) +
	    A.whr[6]*(A.whr[1]*A.whr[5]-A.whr[2]*A.whr[4]));
  }

  SU3 operator~() const;

  SU3 operator-() const{
    SU3 res(whr);
    res.whr[0].re = -res.whr[0].re;
    res.whr[0].im = -res.whr[0].im;
    res.whr[1].re = -res.whr[1].re;
    res.whr[1].im = -res.whr[1].im;
    res.whr[2].re = -res.whr[2].re;
    res.whr[2].im = -res.whr[2].im;
    res.whr[3].re = -res.whr[3].re;
    res.whr[3].im = -res.whr[3].im;
    res.whr[4].re = -res.whr[4].re;
    res.whr[4].im = -res.whr[4].im;
    res.whr[5].re = -res.whr[5].re;    
    res.whr[5].im = -res.whr[5].im;    
    res.whr[6].re = -res.whr[6].re;
    res.whr[6].im = -res.whr[6].im;
    res.whr[7].re = -res.whr[7].re;
    res.whr[7].im = -res.whr[7].im;
    res.whr[8].re = -res.whr[8].re;
    res.whr[8].im = -res.whr[8].im;
    return res;
}


  void tra() {
    Cplx app;
    app = whr[1]; whr[1] = whr[3]; whr[3] = app;
    app = whr[2]; whr[2] = whr[6]; whr[6] = app;
    app = whr[5]; whr[5] = whr[7]; whr[7] = app;
  }

  friend SU3 tra(const SU3 &A0){
    SU3 A;
    A.whr[0] = A0.whr[0];
    A.whr[1] = A0.whr[3];
    A.whr[2] = A0.whr[6];
    A.whr[3] = A0.whr[1];
    A.whr[4] = A0.whr[4];
    A.whr[5] = A0.whr[7];
    A.whr[6] = A0.whr[2];
    A.whr[7] = A0.whr[5];
    A.whr[8] = A0.whr[8];
    return A;
  }

  void dag();

  inline friend SU3 dag(const SU3 &A0){
    SU3 A = A0;
    Cplx app;

    app = A.whr[1];
    A.whr[1] = A.whr[3];
    A.whr[3] = app;
    app = A.whr[2];
    A.whr[2] = A.whr[6];
    A.whr[6] = app;
    app = A.whr[5];
    A.whr[5] = A.whr[7];
    A.whr[7] = app;

    for (int i = 0; i < 9; i++) A.whr[i].im = -A.whr[i].im;
    return A;
  }
  

  void reU();
  void reH();

  void prout();

  bool operator==(SU3);
  bool operator!=(SU3);

  friend SU3 exp(const SU3& A);
};

inline std::ostream& operator <<(std::ostream& os, const SU3& U){
  for(int i = 0; i < 9; i++){
    if(i%3 == 0)    os << std::endl;
    os << " ";
    os << U.whr[i];
  }
  os << std::endl;
}

Cplx Tr(const SU3& A);







class CVector {
  friend class SU3;
  friend class ptCvector;
public:
  Cplx whr[3];

  // Added: DH on Jan. 26, 2012
  // access operators
  Cplx& operator[] (const int& i) { return whr[i]; };
  const Cplx& operator[] (const int& i) const { return whr[i]; };

  static const int storage_size = 6;

  CVector (){};

  CVector (Cplx *vec) {
    whr[0] = vec[0];  
    whr[1] = vec[1];  
    whr[2] = vec[2];  
  }

  CVector (const CVector& V){
    whr[0] = V.whr[0];
    whr[1] = V.whr[1];
    whr[2] = V.whr[2];
  }

  int write(FILE *filept) {
    if(fwrite(&whr, SZ_DB*2, 3, filept)){
      return 0;
    }
    return 1;
  }
  
  int read(FILE *filept) {
    if(fread(&whr, SZ_DB*2, 3, filept)){
      return 0;
    }
    return 1;
  }


  CVector& operator=(const CVector& V) {
      whr[0].re = V.whr[0].re; 
      whr[0].im = V.whr[0].im; 
      whr[1].re = V.whr[1].re; 
      whr[1].im = V.whr[1].im; 
      whr[2].re = V.whr[2].re; 
      whr[2].im = V.whr[2].im; 
      return *this;
  }

  CVector& operator=(CVector& V) {
      whr[0].re = V.whr[0].re; 
      whr[0].im = V.whr[0].im; 
      whr[1].re = V.whr[1].re; 
      whr[1].im = V.whr[1].im; 
      whr[2].re = V.whr[2].re; 
      whr[2].im = V.whr[2].im; 
      return *this;
  }
  
  CVector operator+(const CVector& V) const{
    CVector res;
    res.whr[0].re = whr[0].re + V.whr[0].re;
    res.whr[0].im = whr[0].im + V.whr[0].im;
    res.whr[1].re = whr[1].re + V.whr[1].re;
    res.whr[1].im = whr[1].im + V.whr[1].im;
    res.whr[2].re = whr[2].re + V.whr[2].re;
    res.whr[2].im = whr[2].im + V.whr[2].im;
    return res;					       
  }
  						       
  CVector operator-(const CVector &V) const{			       
    CVector res;				       
    res.whr[0].re = whr[0].re - V.whr[0].re;
    res.whr[0].im = whr[0].im - V.whr[0].im;
    res.whr[1].re = whr[1].re - V.whr[1].re;
    res.whr[1].im = whr[1].im - V.whr[1].im;
    res.whr[2].re = whr[2].re - V.whr[2].re;
    res.whr[2].im = whr[2].im - V.whr[2].im;
    return res;
  };  

  CVector operator-() const{
    CVector res;				       
    res.whr[0].re = -whr[0].re;
    res.whr[0].im = -whr[0].im;
    res.whr[1].re = -whr[1].re;
    res.whr[1].im = -whr[1].im;
    res.whr[2].re = -whr[2].re;
    res.whr[2].im = -whr[2].im;
    return res;
  };  


  Cplx operator*(const CVector& V) const{
    Cplx res;
    res.re = ( whr[0].re * V.whr[0].re + whr[0].im * V.whr[0].im +
	       whr[1].re * V.whr[1].re + whr[1].im * V.whr[1].im +
	       whr[2].re * V.whr[2].re + whr[2].im * V.whr[2].im );
    res.im = ( whr[0].re * V.whr[0].im - whr[0].im * V.whr[0].re +
	       whr[1].re * V.whr[1].im - whr[1].im * V.whr[1].re +
	       whr[2].re * V.whr[2].im - whr[2].im * V.whr[2].re );
    return res;
  }

  void operator+=(const CVector&V ) {   
    whr[0].re += V.whr[0].re;
    whr[0].im += V.whr[0].im;
    whr[1].re += V.whr[1].re;
    whr[1].im += V.whr[1].im;
    whr[2].re += V.whr[2].re;
    whr[2].im += V.whr[2].im;
  }
  
  void operator-=(const CVector& V) {  
    whr[0].re -= V.whr[0].re;
    whr[0].im -= V.whr[0].im;
    whr[1].re -= V.whr[1].re;
    whr[1].im -= V.whr[1].im;
    whr[2].re -= V.whr[2].re;
    whr[2].im -= V.whr[2].im;
  }
  
  void operator*=(const Cplx& z) {
    CVector app;
    app.whr[0].re = whr[0].re * z.re - whr[0].im * z.im;
    app.whr[0].im = whr[0].re * z.im + whr[0].im * z.re;
    app.whr[1].re = whr[1].re * z.re - whr[1].im * z.im;
    app.whr[1].im = whr[1].re * z.im + whr[1].im * z.re;
    app.whr[2].re = whr[2].re * z.re - whr[2].im * z.im;
    app.whr[2].im = whr[2].re * z.im + whr[2].im * z.re;
    *this = app;
} 

  void operator/=(const Cplx& z) {
    Cplx inv = 1./z;
    CVector app;
    app.whr[0].re = whr[0].re * inv.re - whr[0].im * inv.im;
    app.whr[0].im = whr[0].re * inv.im + whr[0].im * inv.re;
    app.whr[1].re = whr[1].re * inv.re - whr[1].im * inv.im;
    app.whr[1].im = whr[1].re * inv.im + whr[1].im * inv.re;
    app.whr[2].re = whr[2].re * inv.re - whr[2].im * inv.im;
    app.whr[2].im = whr[2].re * inv.im + whr[2].im * inv.re;
    *this = app;
  }

  CVector operator*(const Cplx& z) const {
    CVector res;
    res.whr[0].re = whr[0].re * z.re - whr[0].im * z.im;
    res.whr[0].im = whr[0].re * z.im + whr[0].im * z.re;
    res.whr[1].re = whr[1].re * z.re - whr[1].im * z.im;
    res.whr[1].im = whr[1].re * z.im + whr[1].im * z.re;
    res.whr[2].re = whr[2].re * z.re - whr[2].im * z.im;
    res.whr[2].im = whr[2].re * z.im + whr[2].im * z.re;
    return res;
  }

  CVector operator/(const Cplx& z) const{
    Cplx inv = 1./z;
    CVector res;
    res.whr[0].re = whr[0].re * inv.re - whr[0].im * inv.im;
    res.whr[0].im = whr[0].re * inv.im + whr[0].im * inv.re;
    res.whr[1].re = whr[1].re * inv.re - whr[1].im * inv.im;
    res.whr[1].im = whr[1].re * inv.im + whr[1].im * inv.re;
    res.whr[2].re = whr[2].re * inv.re - whr[2].im * inv.im;
    res.whr[2].im = whr[2].re * inv.im + whr[2].im * inv.re;
    return res;
  }

  void operator*=(const double& x) { 
    whr[0].re *= x;
    whr[0].im *= x;
    whr[1].re *= x;
    whr[1].im *= x;
    whr[2].re *= x;
    whr[2].im *= x;
  }


  void operator/=(const double& x) {
    double inv = 1./x;
    whr[0].re *= inv;
    whr[0].im *= inv;
    whr[1].re *= inv;
    whr[1].im *= inv;
    whr[2].re *= inv;
    whr[2].im *= inv;
  }

  CVector operator*(const double& x) const{
    CVector res;
    res.whr[0].re = whr[0].re * x;
    res.whr[0].im = whr[0].im * x;
    res.whr[1].re = whr[1].re * x;
    res.whr[1].im = whr[1].im * x;
    res.whr[2].re = whr[2].re * x;
    res.whr[2].im = whr[2].im * x;
    return res;
  }
  
  CVector operator/(const double& x) const{
    double inv = 1./x;
    CVector res;
    res.whr[0].re = whr[0].re * inv;
    res.whr[0].im = whr[0].im * inv;
    res.whr[1].re = whr[1].re * inv;
    res.whr[1].im = whr[1].im * inv;
    res.whr[2].re = whr[2].re * inv;
    res.whr[2].im = whr[2].im * inv;
    return res;
  }


  friend CVector operator*(const Cplx& z, const CVector &V) {
    CVector res;
    res.whr[0].re = V.whr[0].re * z.re - V.whr[0].im * z.im;
    res.whr[0].im = V.whr[0].re * z.im + V.whr[0].im * z.re;
    res.whr[1].re = V.whr[1].re * z.re - V.whr[1].im * z.im;
    res.whr[1].im = V.whr[1].re * z.im + V.whr[1].im * z.re;
    res.whr[2].re = V.whr[2].re * z.re - V.whr[2].im * z.im;
    res.whr[2].im = V.whr[2].re * z.im + V.whr[2].im * z.re;
    return res;
  }

  CVector operator*(const SU3 &A) const{
    CVector res;
    res.whr[0].re = (whr[0].re * A.whr[0].re - whr[0].im * A.whr[0].im +
		     whr[1].re * A.whr[3].re - whr[1].im * A.whr[3].im +
		     whr[2].re * A.whr[6].re - whr[2].im * A.whr[6].im);
    res.whr[0].im = (whr[0].re * A.whr[0].im + whr[0].im * A.whr[0].re +
		     whr[1].re * A.whr[3].im + whr[1].im * A.whr[3].re +
		     whr[2].re * A.whr[6].im + whr[2].im * A.whr[6].re);
    res.whr[1].re = (whr[0].re * A.whr[1].re - whr[0].im * A.whr[1].im +
		     whr[1].re * A.whr[4].re - whr[1].im * A.whr[4].im +
		     whr[2].re * A.whr[7].re - whr[2].im * A.whr[7].im);
    res.whr[1].im = (whr[0].re * A.whr[1].im + whr[0].im * A.whr[1].re +
		     whr[1].re * A.whr[4].im + whr[1].im * A.whr[4].re +
		     whr[2].re * A.whr[7].im + whr[2].im * A.whr[7].re);
    res.whr[2].re = (whr[0].re * A.whr[2].re - whr[0].im * A.whr[2].im +
		     whr[1].re * A.whr[5].re - whr[1].im * A.whr[5].im +
		     whr[2].re * A.whr[8].re - whr[2].im * A.whr[8].im);
    res.whr[2].im = (whr[0].re * A.whr[2].im + whr[0].im * A.whr[2].re +
		     whr[1].re * A.whr[5].im + whr[1].im * A.whr[5].re +
		     whr[2].re * A.whr[8].im + whr[2].im * A.whr[8].re);
    return res;
  }


  void prout(){
    printf("\n");
    for(int i = 0; i < 3; i++) {
      whr[i].prout();
      printf("\n");
    }
  }


  double mod() {
    return sqrt(whr[0].re * whr[0].re + whr[0].im * whr[0].im + 
		whr[1].re * whr[1].re + whr[1].im * whr[1].im + 
		whr[2].re * whr[2].re + whr[2].im * whr[2].im );
  }
  
  double Norm() { return mod(); }

  void dag() { 
    whr[0].im = -whr[0].im ;
    whr[1].im = -whr[1].im ;
    whr[2].im = -whr[2].im ;
  }

  friend CVector dag(const CVector&V) { 
    CVector res;
    res.whr[0].re =  V.whr[0].re;
    res.whr[0].im = -V.whr[0].im;
    res.whr[1].re =  V.whr[1].re;
    res.whr[1].im = -V.whr[1].im;
    res.whr[2].re =  V.whr[2].re;
    res.whr[2].im = -V.whr[2].im;
    return res;
  }
  /// Added: Jan. 18, 2012 DH
  bool operator==(const CVector& other) const {
    for (const Cplx *i = this->whr, *j = other.whr; 
         i != this->whr + 3; ++i, ++j)
      if (*i != *j)
        return false;
    return true;
  }


  void randomize() {
    static MyRand r(1235431);
    for (int i = 0; i < 3; ++i)
      whr[i] = Cplx(r.Rand(), r.Rand());
  };

}; // end class CVector


inline CVector SU3::operator*(const CVector& V) const {
  CVector res;
  res.whr[0].re = (V.whr[0].re * whr[0].re - V.whr[0].im * whr[0].im +
		   V.whr[1].re * whr[1].re - V.whr[1].im * whr[1].im +
		   V.whr[2].re * whr[2].re - V.whr[2].im * whr[2].im);
  res.whr[0].im = (V.whr[0].re * whr[0].im + V.whr[0].im * whr[0].re +
		   V.whr[1].re * whr[1].im + V.whr[1].im * whr[1].re +
		   V.whr[2].re * whr[2].im + V.whr[2].im * whr[2].re);
  res.whr[1].re = (V.whr[0].re * whr[3].re - V.whr[0].im * whr[3].im +
		   V.whr[1].re * whr[4].re - V.whr[1].im * whr[4].im +
		   V.whr[2].re * whr[5].re - V.whr[2].im * whr[5].im);
  res.whr[1].im = (V.whr[0].re * whr[3].im + V.whr[0].im * whr[3].re +
		   V.whr[1].re * whr[4].im + V.whr[1].im * whr[4].re +
		   V.whr[2].re * whr[5].im + V.whr[2].im * whr[5].re);
  res.whr[2].re = (V.whr[0].re * whr[6].re - V.whr[0].im * whr[6].im +
		   V.whr[1].re * whr[7].re - V.whr[1].im * whr[7].im +
		   V.whr[2].re * whr[8].re - V.whr[2].im * whr[8].im);
  res.whr[2].im = (V.whr[0].re * whr[6].im + V.whr[0].im * whr[6].re +
		   V.whr[1].re * whr[7].im + V.whr[1].im * whr[7].re +
		   V.whr[2].re * whr[8].im + V.whr[2].im * whr[8].re);
  return res;
}


inline CVector SU3::operator^(const CVector& V) const {
  CVector res;
  res.whr[0].re = (V.whr[0].re * whr[0].re + V.whr[0].im * whr[0].im +
		   V.whr[1].re * whr[3].re + V.whr[1].im * whr[3].im +
		   V.whr[2].re * whr[6].re + V.whr[2].im * whr[6].im);
  res.whr[0].im = (V.whr[0].re * whr[0].im - V.whr[0].im * whr[0].re +
		   V.whr[1].re * whr[3].im - V.whr[1].im * whr[3].re +
		   V.whr[2].re * whr[6].im - V.whr[2].im * whr[6].re);
  res.whr[1].re = (V.whr[0].re * whr[1].re + V.whr[0].im * whr[1].im +
		   V.whr[1].re * whr[4].re + V.whr[1].im * whr[4].im +
		   V.whr[2].re * whr[7].re + V.whr[2].im * whr[7].im);
  res.whr[1].im = (V.whr[0].re * whr[1].im - V.whr[0].im * whr[1].re +
		   V.whr[1].re * whr[4].im - V.whr[1].im * whr[4].re +
		   V.whr[2].re * whr[7].im - V.whr[2].im * whr[7].re);
  res.whr[2].re = (V.whr[0].re * whr[2].re + V.whr[0].im * whr[2].im +
		   V.whr[1].re * whr[5].re + V.whr[1].im * whr[5].im +
		   V.whr[2].re * whr[8].re + V.whr[2].im * whr[8].im);
  res.whr[2].im = (V.whr[0].re * whr[2].im - V.whr[0].im * whr[2].re +
		   V.whr[1].re * whr[5].im - V.whr[1].im * whr[5].re +
		   V.whr[2].re * whr[8].im - V.whr[2].im * whr[8].re);
  return res;
}


SU3 SU3rand(MyRand&);

#endif

