#ifndef _TIMER_H
#define _TIMER_H

#include <map>
#include <iostream>

#ifdef HAVE_STDCXX_11
#include <chrono>

struct Timer {
  static std::chrono::duration<double> t_tot;
  Timer () : t(0.0) { };
  void start() { 
    t1 = std::chrono::high_resolution_clock::now();
  }
  void stop() {
    std::chrono::duration<double> elapsed = 
      std::chrono::duration_cast<std::chrono::seconds> (std::chrono::high_resolution_clock::now() - t1);
    t += elapsed;
    t_tot += elapsed;
  }
private:
  std::chrono::high_resolution_clock::time_point t1;
  std::chrono::duration<double> t;
};

#else

#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
#endif

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

#endif

double Timer::t_tot = 0.0;

#endif //TIMER_H
