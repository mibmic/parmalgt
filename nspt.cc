#include "nspt.h"
#include "GammaStuff.h"

ptSU3 W, W1, W2, Wsym;
ptSU3 *MM = new ptSU3[dim];

int link_c, site_c, site_up, xx[4], curr;
const int offset = dim+1;

extern Cplx* w;  // usero' come variabile d'appoggio
extern Cplx* w1; // usero' per placchetta 1x1
extern Cplx* w2; // usero' per placchetta 2x1
extern double* norm;
extern double* norm1;
extern Cplx** trU;
extern int ratio; 
extern ptSU3* U;
extern nspt_params_t nspt_pars;
extern act_params_t  act_pars;
extern thread_params_t thr_pars[NTHR];

extern std::ofstream plaqfile, normfile, logfile, trufile;

#ifdef __WILLOOP_2x2__
extern std::ofstream loop2x2;
#endif


extern MyRand Rand[NTHR];

#ifdef __PARALLEL_OMP__
ptSU3 Ww[NTHR], Ww1[NTHR], Ww2[NTHR], Wwsym[NTHR];
ptSU3 MMm[NTHR][dim];
SU3 P[NTHR];
Cplx ww[NTHR][allocORD+1], ww1[NTHR][allocORD+1], ww2[NTHR][allocORD+1];
double normm[NTHR][allocORD+1];
Cplx ttrU[NTHR][dim][allocORD];
int tid;

#endif

// for (single step of) Fourier Accelerated gauge fixing
int coord[dim];
ptSU3_fld* Wgauge;
fftw_plan *planFA;

// for momentum space analysis of the norm
fftw_plan *planGluon;

#ifdef __K_MOM_ANALYSIS__
double *knorm[allocORD];
extern FILE *FPkn;
#endif

#ifdef __TIMING__
extern PRlgtTime Time;
#endif



void gauge_wilson(ptGluon_fld& Umu){

#ifndef __PARALLEL_OMP__

  for(curr = 0; curr < act_pars.iVol; curr++){
    
    for(int mu = 0; mu < dim; mu++){
      // Preparo il link corrente
      link_c = get(&Umu, curr, mu);
      
      W1.zero();	
#if GAUGE_ACTION != WIL
      W2.zero();
#endif				
      // Calcolo la staple, eventualmente alnche la 2x1 su tutte
      // le direzioni ortogonali al link corrente
      for(int nu = 0; nu < dim; nu++){		
	if(nu != mu){					
	  W1 += Umu.staple(curr, mu, nu);			
#if GAUGE_ACTION != WIL
	  W2 += Umu.staple2x1(curr, mu, nu);			
#endif
	}						
      }						
      
      // Chido le staple a dare la placchetta e traccio
      W1 = U[link_c]*W1;
      W1.Tr(w);
      // Sommo al valore della placchetta e calcolo la norma
      w1[0] += w[0];
      for (int i1=1; i1 <= PTORD; i1++){		
	w1[i1] += w[i1];
	norm[i1-1] += ((U[link_c].ptU[i1-1]*dag(U[link_c].ptU[i1-1])).Tr()).re;
	trU[mu][i1-1] += U[link_c].ptU[i1-1].Tr();
      }

      
      // Se ho azioni improved faccio lo stesso sulle
      // parti di improvement
#if GAUGE_ACTION != WIL
      W2 = U[link_c]*W2;			
      W2.Tr(w);
      for (int i1=0; i1 <= PTORD; i1++){		
	w2[i1] += w[i1];
      }
      
      W1 = act_pars.c0*W1 + act_pars.c1*W2;
#endif
      // Costringo nell'algebra
      W1 = W1.reH();					
      W1 *= act_pars.tau_g;
      
      // Aggiungo fluttuazione al primo ordine
      W1.ptU[0] -= act_pars.stau*SU3rand(Rand[0]);
      U[link_c] = exp(W1)*U[link_c];	
      
      // Contributo del momento nullo
      MM[mu] += log(U[link_c]);		
      
      
    } //end mu
    
  } // fine siti


#else
  
  for(tid = 0;tid < NTHR; tid++) {
    for(int ord = 0; ord < PTORD; ord++){
      ww[tid][ord]  = 0;
      ww1[tid][ord] = 0;
      ww2[tid][ord] = 0;
      normm[tid][ord] = 0;
      for( int mu = 0; mu < dim; mu++)
	{
	  ttrU[tid][mu][ord] = Cplx(0,0);
	}
    }
    ww[tid][PTORD]  = 0;
    ww1[tid][PTORD] = 0;
    ww2[tid][PTORD] = 0;
    
    for(int mu = 0; mu < dim; mu++){
      MMm[tid][mu].zero();
    }
  }
  
  
#pragma omp parallel private(curr,tid,link_c) num_threads(NTHR) 
  {
    tid = omp_get_thread_num();
    for(int y0 = thr_pars[tid].xi[0]; y0 < thr_pars[tid].xf[0]; y0++){
      for(int y1 = thr_pars[tid].xi[1]; y1 < thr_pars[tid].xf[1]; y1++){
	for(int y2 = thr_pars[tid].xi[2]; y2 < thr_pars[tid].xf[2]; y2++){
	  for(int y3 = thr_pars[tid].xi[3]; y3 < thr_pars[tid].xf[3]; y3++){

	    curr = y3 + act_pars.sz[3]*(y2 + act_pars.sz[2]*(y1 + act_pars.sz[1]*y0) );


	    for(int mu = 0; mu < dim; mu++){
	      // Preparo il link corrente
	      link_c = get(&Umu, curr, mu);
      
	      Ww1[tid].zero();	
#if GAUGE_ACTION != WIL
	      Ww2[tid].zero();
#endif				
	      // Calcolo la staple, eventualmente anche la 2x1 su tutte
	      // le direzioni ortogonali al link corrente
	      for(int nu = 0; nu < dim; nu++){		
		if(nu != mu){					
		  Ww1[tid] += Umu.staple(curr, mu, nu);			
#if GAUGE_ACTION != WIL
		  Ww2[tid] += Umu.staple2x1(curr, mu, nu);			
#endif
		}						
	      }						
      
	      // Chido le staple a dare la placchetta e traccio
	      Ww1[tid] = U[link_c]*Ww1[tid];
	      Ww1[tid].Tr(ww[tid]);
	      // Sommo al valore della placchetta e calcolo la norma
	      ww1[tid][0] += ww[tid][0];
	      for (int i1=1; i1 <= PTORD; i1++){
		ww1[tid][i1] += ww[tid][i1];

		normm[tid][i1-1] += ((U[link_c].ptU[i1-1]*dag(U[link_c].ptU[i1-1])).Tr()).re;

		ttrU[tid][mu][i1-1] += U[link_c].ptU[i1-1].Tr();
	      }

	      // Se ho azioni improved faccio lo stesso sulle
	      // parti di improvement
#if GAUGE_ACTION != WIL
	      Ww2[tid] = U[link_c]*Ww2[tid];			
	      Ww2[tid].Tr(ww[tid]);
	      for (int i1=0; i1 <= PTORD; i1++){		
		ww2[tid][i1] += ww[tid][i1];
	      }
      
	      Ww1[tid] = act_pars.c0*Ww1[tid] + act_pars.c1*Ww2[tid];
#endif
	      // Costringo nell'algebra
	      Ww1[tid]  = Ww1[tid].reH();					
	      Ww1[tid] *= act_pars.tau_g;
      
	      // Aggiungo fluttuazione al primo ordine
	      Ww1[tid].ptU[0] -= act_pars.stau*SU3rand(Rand[tid]);
	      U[link_c] = exp(Ww1[tid])*U[link_c];	
      
	      // Contributo del momento nullo
	      MMm[tid][mu] += log(U[link_c]);		
      
      
	    } //end mu
	    
#if ntz > 1
#pragma omp barrier
#endif
	  } // end z
	  
#if nty > 1
#pragma omp barrier
#endif
	} // end y
	
#if ntx > 1
#pragma omp barrier
#endif
      } // end x
      
#if ntt > 1
#pragma omp barrier
#endif
    }  // end t

#pragma omp barrier  
  } // end parallel

  // Reduce from parallel
  for(tid = 0; tid < NTHR; tid++){
  
    for(int ord = 0; ord < PTORD; ord++){
      w1[ord] += ww1[tid][ord];
      w2[ord] += ww2[tid][ord];
      norm[ord] += normm[tid][ord];
      for(int mu = 0; mu < dim; mu++){
	trU[mu][ord] += ttrU[tid][mu][ord];
      }
    }
    w1[PTORD] += ww1[tid][PTORD];
    w2[PTORD] += ww2[tid][PTORD];

    for(int mu = 0; mu < dim; mu++){
      MM[mu] += MMm[tid][mu];
    }
  }

#endif

  
  for (int i1=0; i1 < PTORD; i1++){		
#if GAUGE_ACTION == WIL
    plaqfile << w1[i1].re/(double)(72*act_pars.iVol) << "\t" 
	     << w1[i1].im/(double)(72*act_pars.iVol) << std::endl;
#else
    plaqfile << w1[i1].re/(double)(72*act_pars.iVol) << "\t" 
	     << w1[i1].im/(double)(72*act_pars.iVol) << "\t"
	     << w2[i1].re/(double)(72*act_pars.iVol) << "\t" 
	     << w2[i1].im/(double)(72*act_pars.iVol) << std::endl;
#endif
    normfile << norm[i1]/(double)(3*dim*act_pars.iVol) << std::endl;
    for( int mu = 0; mu < dim; mu++)
      {
	trufile  << trU[mu][i1].re /(double)(3*act_pars.iVol) << "\t" 
		 << trU[mu][i1].im /(double)(3*act_pars.iVol) << "\t";
      }
    trufile  << std::endl;
  }

#if GAUGE_ACTION == WIL
  plaqfile << w1[PTORD].re/(double)(72*act_pars.iVol) << "\t" 
	   << w1[PTORD].im/(double)(72*act_pars.iVol) << std::endl;
#else
  plaqfile << w1[PTORD].re/(double)(72*act_pars.iVol) << "\t" 
	   << w1[PTORD].im/(double)(72*act_pars.iVol) << "\t"
	   << w2[PTORD].re/(double)(72*act_pars.iVol) << "\t" 
	   << w2[PTORD].im/(double)(72*act_pars.iVol) << std::endl;
#endif

} // fine evoluzione gluoni





SpinColor ZeroMom;
SpinColor XiTmp;
ptSpinColor PsiTmp;
#ifdef __PARALLEL_OMP__
SpinColor ZM[NTHR];
SpinColor XT[NTHR];
ptSpinColor PT[NTHR];
#endif

void fermion_wilson(ptGluon_fld& Umu, ptSpinColor_fld& Pmu, SpinColor_fld& Xi){

  memset((void*)&ZeroMom, 0, sizeof(SpinColor));

#ifndef __PARALLEL_OMP__
  // genero campo gaussiano

  Xi.gauss(Rand[0]);

  // Sottraggo il momento nullo
  for( int iv = 0; iv < act_pars.iVol; iv++){
    ZeroMom += Xi.psi[iv];
  }
  ZeroMom *= act_pars.rVol;

  for( int iv = 0; iv < act_pars.iVol; iv++){
    Xi.psi[iv] -= ZeroMom;
    Pmu.scfld->psi[iv] = Xi.psi[iv];
  }

#else

  // Sottraggo il momento nullo
  memset((void*)ZM, 0, NTHR*sizeof(SpinColor));
   
#pragma omp parallel private(ZeroMom) num_threads(NTHR)
  { 
    int tid = omp_get_thread_num();
    Xi.gauss(Rand,tid);
#pragma omp for
    for(int iv = 0; iv < act_pars.iVol; iv++){
      ZM[tid] += Xi.psi[iv];
    }
  }
  
  ZeroMom = ZM[0];
  for(int nt = 1; nt < NTHR; nt++){
    ZeroMom += ZM[nt];
  }
  ZeroMom *= act_pars.rVol;
  
#pragma omp parallel for num_threads(NTHR)
  for( int iv = 0; iv < act_pars.iVol; iv++){
    Xi.psi[iv] -= ZeroMom;
    Pmu.scfld->psi[iv] = Xi.psi[iv];
  } // pragma omp parallel for -> now single thread
  

#endif

  memset((void*)Pmu.psi, 0, act_pars.iVol*sizeof(ptSpinColor));
  Pmu.fillPT(PTORD-2,Umu);

#ifndef __PARALLEL_OMP__
  // cicla su tutti i siti
  for(curr = 0; curr < act_pars.iVol; curr++){
    site_c = Umu.Z->L[curr][dim];

    for( int mu = 0; mu < dim; mu++ ){
      
      // variabile d'appoggio
      W.zero();
      XiTmp   = Xi.psi[Umu.Z->L[curr][offset+mu]];
      // PsiTmp  = Pmu.psi[site_c];
      // PsiTmp += PsiTmp.gmleft(mu);

      for(int nu = 0; nu < dim; nu++)
	{
	  PsiTmp.psi[nu] = Pmu.psi[site_c].psi[nu] + gmuval[mu][nu]*Pmu.psi[site_c].psi[gmuind[mu][nu]];
	}

      // Calcola la forza
      for(int ord = 0; ord < PTORD-1; ord++){
	for(int k = 0; k < 3; k++){
	  for(int j = 0; j < 3; j++){
	    
	    for(int alpha = 0; alpha < dim; alpha++){  
	      W.ptU[ord+1].whr[j+3*k] += ~(XiTmp.psi[alpha].whr[j])*
		PsiTmp.psi[alpha].ptCV[ord].whr[k];
	    } // alpha
	    
	  } // colore, j
	} // colore, k
      } // ord
      
      // Step evoluzione
      W = W*dag(Umu.W[site_c].U[mu]);
      W.reH();
      Umu.W[site_c].U[mu]= exp(act_pars.tau_f*W)*Umu.W[site_c].U[mu];
    } // mu

  } // loop sui siti
#else
#pragma omp parallel private(tid,site_c,curr) num_threads(NTHR)
  {
    tid = omp_get_thread_num();

    for(int y0 = thr_pars[tid].xi[0]; y0 < thr_pars[tid].xf[0]; y0++){
      for(int y1 = thr_pars[tid].xi[1]; y1 < thr_pars[tid].xf[1]; y1++){
	for(int y2 = thr_pars[tid].xi[2]; y2 < thr_pars[tid].xf[2]; y2++){
	  for(int y3 = thr_pars[tid].xi[3]; y3 < thr_pars[tid].xf[3]; y3++){

	    curr = y3 + act_pars.sz[3]*(y2 + act_pars.sz[2]*(y1 + act_pars.sz[1]*y0) );
	    
	    site_c = Umu.Z->L[curr][dim];
	    
	    for( int mu = 0; mu < dim; mu++ ){
	      
	      // variabile d'appoggio
	      Ww[tid].zero();
	      XT[tid]   = Xi.psi[Umu.Z->L[curr][offset+mu]];
	      // PT[tid]  = Pmu.psi[site_c];
	      // PT[tid] += PT[tid].gmleft(mu);

	      for(int nu = 0; nu < dim; nu++)
		{
		  PT[tid].psi[nu] = Pmu.psi[site_c].psi[nu] + gmuval[mu][nu]*Pmu.psi[site_c].psi[gmuind[mu][nu]];
		}
	      
	      // Calcola la forza
	      for(int ord = 0; ord < PTORD-1; ord++){
		for(int k = 0; k < 3; k++){
		  for(int j = 0; j < 3; j++){
		    
		    for(int alpha = 0; alpha < dim; alpha++){  
		      Ww[tid].ptU[ord+1].whr[j+3*k] += ~(XT[tid].psi[alpha].whr[j])*
			PT[tid].psi[alpha].ptCV[ord].whr[k];
		    } // alpha
		    
		  } // colore, j
		} // colore, k
	      } // ord
	      
	      // Step evoluzione
	      Ww[tid] = Ww[tid]*dag(Umu.W[site_c].U[mu]);
	      Ww[tid].reH();
	      Umu.W[site_c].U[mu]= exp(act_pars.tau_f*Ww[tid])*Umu.W[site_c].U[mu];
	    } // mu
	    
#if ntz > 1
#pragma omp barrier
#endif
	  } // end z
	  
#if nty > 1
#pragma omp barrier
#endif
	} // end y
	
#if ntx > 1
#pragma omp barrier
#endif
      } // end x
      
#if ntt > 1
#pragma omp barrier
#endif
    }  // end t
    

#pragma omp barrier
  } // parallel
#endif

}


  
void zero_modes_subtraction(ptGluon_fld& Umu){
  // Normalizzo il modo nullo
  for (int i1=0; i1 < dim; i1++){
    MM[i1] *= act_pars.rVol;
  }
  
  // Campo nell'algebra e sottrazione dei momenti nulli
  // nota: non mi interessa l'ordine con cui lo faccio

  #ifndef __PARALLEL_OMP__
  for(int i = 0; i < act_pars.iVol; i++){
    for(int mu = 0; mu < dim; mu++){
      W = log(Umu.W[i].U[mu]);
      W -= MM[mu];
      W.reH();
      Umu.W[i].U[mu]  = exp(W);
    }
  }
  #else

  int curr;
#pragma omp parallel private(curr, tid) num_threads(NTHR) 
  {
    tid = omp_get_thread_num();
    for(int y0 = thr_pars[tid].xi[0]; y0 < thr_pars[tid].xf[0]; y0++){
      for(int y1 = thr_pars[tid].xi[1]; y1 < thr_pars[tid].xf[1]; y1++){
	for(int y2 = thr_pars[tid].xi[2]; y2 < thr_pars[tid].xf[2]; y2++){
	  for(int y3 = thr_pars[tid].xi[3]; y3 < thr_pars[tid].xf[3]; y3++){

	    curr = y3 + act_pars.sz[3]*(y2 + act_pars.sz[2]*(y1 + act_pars.sz[1]*y0) );

	    for(int mu = 0; mu < dim; mu++){
	      Ww[tid] = log(Umu.W[curr].U[mu]);
	      Ww[tid] -= MM[mu];
	      Ww[tid].reH();
	      Umu.W[curr].U[mu]  = exp(Ww[tid]);
	    } // mu

	  } // y3
	} // y2
      } // y1
    } // y0
    
  } // parallel

#endif

}








void stochastic_gauge_fixing(ptGluon_fld& Umu){

#ifndef __PARALLEL_OMP__
  for(curr = 0; curr < act_pars.iVol; curr++){

    // preparo il sito corrente
    site_c = Umu.Z->get(curr);
    //    site_c = Umu.Z->L[curr][dim];
    W.zero();
    
    // calcolo il DmuUmu e lo metto nell'algebra
    for(int mu = 0; mu < dim; mu++){
      W += Umu.W[site_c].U[mu] - Umu.W[Umu.Z->L[curr][mu]].U[mu];
    }
    W -= dag(W);
    
    // Forma alternativa
    // for(int mu = 0; mu < dim; mu++){
    //   W += log(Umu.W[site_c].U[mu])- log(Umu.W[Umu.Z->L[curr][mu]].U[mu]);
    // }

    W.Trless();
    
    // Preparo le trasformazioni di gauge
    W1 = exp( act_pars.alpha*W);
    W2 = exp(-act_pars.alpha*W);
    
    // Moltiplico sul link corrente a sx e sul precedente a dx
    for(int mu = 0; mu < dim; mu++){
      Umu.W[site_c].U[mu] = W1*Umu.W[site_c].U[mu];
      Umu.W[Umu.Z->L[curr][mu]].U[mu] = Umu.W[Umu.Z->L[curr][mu]].U[mu]*W2;
    }
    
  } // loop sui siti
  
#else

#pragma omp parallel private(curr,tid,site_c)  num_threads(NTHR) 
  {
    tid = omp_get_thread_num();
    
    for(int y0 = thr_pars[tid].xi[0]; y0 < thr_pars[tid].xf[0]; y0++){
      for(int y1 = thr_pars[tid].xi[1]; y1 < thr_pars[tid].xf[1]; y1++){
	for(int y2 = thr_pars[tid].xi[2]; y2 < thr_pars[tid].xf[2]; y2++){
	  for(int y3 = thr_pars[tid].xi[3]; y3 < thr_pars[tid].xf[3]; y3++){

	    curr = y3 + act_pars.sz[3]*(y2 + act_pars.sz[2]*(y1 + act_pars.sz[1]*y0) );
	    
	    site_c = Umu.Z->L[curr][dim];
	    
	    // calcolo il DmuUmu e lo metto nell'algebra
	    Ww[tid].zero();
	    for(int mu = 0; mu < dim; mu++){
	      Ww[tid] += Umu.W[site_c].U[mu] - Umu.W[Umu.Z->L[curr][mu]].U[mu];
	    }

	    // Ww[tid] = Umu.W[site_c].U[0] - Umu.W[Umu.Z->L[curr][0]].U[0];
	    // for(int mu = 1; mu < dim; mu++){
	    //   Ww[tid] += Umu.W[site_c].U[mu] - Umu.W[Umu.Z->L[curr][mu]].U[mu];
	    // }

	    Ww[tid] -= dag(Ww[tid]);
	    Ww[tid].Trless();
	    
	    // Preparo le trasformazioni di gauge
	    Ww1[tid] = exp( act_pars.alpha*Ww[tid]);
	    Ww2[tid] = exp(-act_pars.alpha*Ww[tid]);
	    
	    // Moltiplico sul link corrente a sx e sul precedente a dx
	    for(int mu = 0; mu < dim; mu++){
	      Umu.W[site_c].U[mu] = Ww1[tid]*Umu.W[site_c].U[mu];
	      Umu.W[Umu.Z->L[curr][mu]].U[mu] = Umu.W[Umu.Z->L[curr][mu]].U[mu]*Ww2[tid];
	    }

#if ntz > 1
#pragma omp barrier
#endif
	  } // end z
	  
#if nty > 1
#pragma omp barrier
#endif
	} // end y
	
#if ntx > 1
#pragma omp barrier
#endif
      } // end x
      
#if ntt > 1
#pragma omp barrier
#endif
    }  // end t


#pragma omp barrier
  } // end parallel
#endif


#ifdef __K_MOM_ANALYSIS__

  fftw_execute(planGluon[0]);
  for(curr = 0; curr < act_pars.iVol; curr++){
    for (int i1=0; i1 < allocORD; i1++){		
      knorm[i1][curr] = 0;
    }
    for(int mu = 0; mu < dim; mu++){
      for (int i1=0; i1 < allocORD; i1++){		
	knorm[i1][curr] += ( Umu.W[curr].U[mu].ptU[i1]*
			     dag(Umu.W[curr].U[mu].ptU[i1]) 
			     ).Tr().re;
      }
    }
  }

  fftw_execute(planGluon[1]);
  
  for(curr = 0; curr < act_pars.iVol; curr++){
    for(int mu = 0; mu < dim; mu++){
      Umu.W[curr].U[mu] *= act_pars.rVol;
    }
  }

  fwrite(knorm[0],sizeof(double),act_pars.iVol*allocORD ,FPkn);

#endif


}






void FAstochastic_gauge_fixing(ptGluon_fld& Umu){

#ifdef  __PARALLEL_OMP__
#pragma omp for schedule(static)
#endif
  for(int x = 0; x < act_pars.iVol; x++){
    Wgauge->W[Umu.Z->get(x)].zero();
    for(int mu = 0; mu < dim; mu++){
      Wgauge->W[Umu.Z->get(x)] += ( log(U[get(&Umu,x,mu)]) - 
				    log(U[get(&Umu,x,-1,mu,mu)]) );
    }
  }
  
  fftw_execute(planFA[0]);
  
#ifdef PBC
  Wgauge->W[0] *= 0.0;
#endif  
  
#ifdef  __PARALLEL_OMP__
#pragma omp for schedule(static)
#endif
  for(int x = 1; x < act_pars.iVol; x++){
    Umu.Z->get(x,coord);
    Wgauge->W[Umu.Z->get(x)] *= -(act_pars.alpha/(Umu.Z->p2hat[0][coord[0]]+
						  Umu.Z->p2hat[1][coord[1]]+
						  Umu.Z->p2hat[2][coord[2]]+
						  Umu.Z->p2hat[3][coord[3]]));
  }
  
  
  fftw_execute(planFA[1]);
  
#ifdef  __PARALLEL_OMP__
#pragma omp for schedule(static)
#endif
  for(int x = 0; x < act_pars.iVol; x++){
    Wgauge->W[x] /= (double)act_pars.iVol;
    Wgauge->W[x].reH();
    Wgauge->W[x] = exp(Wgauge->W[x]);
  }
    
  
#ifdef  __PARALLEL_OMP__
#pragma omp for schedule(static)
#endif
  for(int x = 0; x < act_pars.iVol; x++){
    for(int mu = 0; mu < dim; mu++){
      U[get(&Umu,x,mu)] = Wgauge->get(x)*Umu.get(x,mu)*dag(Wgauge->get(x,1,mu));
    }
  }
  

#ifdef __K_MOM_ANALYSIS__

  fftw_execute(planGluon[0]);

  for(curr = 0; curr < act_pars.iVol; curr++){
    for (int i1=0; i1 < allocORD; i1++){		
      knorm[i1][curr] = 0;
    }
    for(int mu = 0; mu < dim; mu++){
      for (int i1=0; i1 < allocORD; i1++){		
	knorm[i1][curr] += ( Umu.W[curr].U[mu].ptU[i1]*
			     dag(Umu.W[curr].U[mu].ptU[i1]) 
			     ).Tr().re;
      }
    }
  }

  fftw_execute(planGluon[1]);
  
  for(curr = 0; curr < act_pars.iVol; curr++){
    for(int mu = 0; mu < dim; mu++){
      Umu.W[curr].U[mu] *= act_pars.rVol;
    }
  }

  fwrite(knorm[0],sizeof(double),act_pars.iVol*allocORD ,FPkn);

#endif
  
}






// Beat step di evoluzione quenched
void NsptEvolve(ptGluon_fld& Umu){

#ifdef __TIMING__
  Time.reset();
#endif

  for( int t2 = 0; t2 < nspt_pars.Beat; t2++ ){
    
    // Inizializza a 0 le viariabili di appoggio, le placchette
    // e la norma
    for (int i1=0; i1 < dim; i1++){
      MM[i1].zero();
    }
    for (int i1=0; i1 <= PTORD; i1++){
      w1[i1]    = 0.0;
      norm[i1]  = 0.0;
      for (int mu=0; mu < dim; mu++){
	trU[mu][i1]   = 0.0;
      }
#if GAUGE_ACTION != WIL
      w2[i1]    = 0.0;
#endif
    }

    // Evoluzione della parte gluonica
#ifdef __TIMING__
    Time.tic_g();
#endif

    gauge_wilson(Umu);

#ifdef __TIMING__
    Time.toc_g();
#endif

    // Sottrazione dei momenti nulli
#ifdef __TIMING__
    Time.tic_zm();
#endif

    zero_modes_subtraction(Umu);

#ifdef __TIMING__
    Time.toc_zm();
#endif

    // Gauge fixing stocastico
#ifdef __TIMING__
    Time.tic_gf();
#endif

#ifndef __FA_GAUGE_FIXING__
    stochastic_gauge_fixing(Umu);
#else
    // Fourier acceleration
    FAstochastic_gauge_fixing(Umu);
#endif

#ifdef __TIMING__
    Time.toc_gf();
    Time.reduce();
#endif

  }

#ifdef __TIMING__
  Time.out();
#endif

}





// Beat step di evoluzione unquenched
void NsptEvolve(ptGluon_fld& Umu, ptSpinColor_fld& Pmu, SpinColor_fld& Xi){

#ifdef __TIMING__
  Time.reset();
#endif

  for( int t2 = 0; t2 < nspt_pars.Beat; t2++ ){
    
    // Inizializza a 0 le viariabili di appoggio, le placchette
    // e la norma
    for (int i1=0; i1 < dim; i1++){
      MM[i1].zero();
    }
    for (int i1=0; i1 <= PTORD; i1++){
      w1[i1] = 0.0;
      norm[i1]  = 0.0;
      for (int mu=0; mu < dim; mu++){
	trU[mu][i1]   = 0.0;
      }
#if GAUGE_ACTION != WIL
      w2[i1] = 0.0;
#endif
    }

    // Evoluzione della parte gluonica
#ifdef __TIMING__
    Time.tic_g();
#endif

    gauge_wilson(Umu);

#ifdef __TIMING__
    Time.toc_g();    
#endif

    // Sottrazione dei momenti nulli
#ifdef __TIMING__
    Time.tic_zm();
#endif

    zero_modes_subtraction(Umu);

#ifdef __TIMING__
    Time.toc_zm();
#endif

    // Gauge fixing stocastico
#ifdef __TIMING__
    Time.tic_gf();
#endif

#ifndef __FA_GAUGE_FIXING__
    stochastic_gauge_fixing(Umu);
#else
    // Fourier acceleration
    FAstochastic_gauge_fixing(Umu);
#endif

#ifdef __TIMING__
    Time.toc_zm();
#endif


    // Evoluzione della parte fermionica
#ifdef __TIMING__
    Time.tic_f();
#endif

    fermion_wilson(Umu, Pmu, Xi);

#ifdef __TIMING__
    Time.toc_f();
    Time.reduce();
#endif

  }

#ifdef __TIMING__
  Time.out();
#endif

}



// Confronta 2 placchette. Se sono uguali ritorna 0, altrimenti
// la differenza tra le due
int plaquette_check(Cplx* w1, Cplx* w2){
  // return ( memcmp(w1, w2, (PTORD+1)*sizeof(Cplx)) ); 
  for(int i1 = 0; i1 <= PTORD;i1++){
    if( (w1[i1]-w2[i1]).mod() > 1e-7 ) return 1;
  }
  return 0;
}





#ifdef __WILLOOP_2x2__

void WL2x2(ptGluon_fld& Umu) {
  
#ifndef __PARALLEL_OMP__
  ptSU3 tmp, tmp1;
#else
  ptSU3 tmp[NTHR], tmp1[NTHR];
#endif
  
  int a,b,d,e;
  

#ifndef __PARALLEL_OMP__
  for(int site_x = 0; site_x < act_pars.iVol; site_x++) {

    for(int mu = 3; mu > 0; --mu){
      a = Umu.Z->L[site_x][5+mu];
      b = Umu.Z->L[a][5+mu];
      tmp = Umu.W[site_x].U[mu] * Umu.W[a].U[mu];

      for( int nu = 0; nu < mu; ++nu){
	d = Umu.Z->L[site_x][5+nu];
	e = Umu.Z->L[d][5+nu];
	tmp1 += tmp * ( Umu.W[b].U[nu] * Umu.W[ Umu.Z->L[b][5+nu] ].U[nu] * 
			dag( Umu.W[site_x].U[nu] * Umu.W[d].U[nu] * Umu.W[e].U[mu] * 
			     Umu.W[ Umu.Z->L[e][5+mu] ].U[mu] ));

      } // nu
    } // mu
  } // site

  for( int i1 = 1; i1 < PTORD; i1+=2){
    loop2x2 << (tmp1.ptU[i1].Tr()).re*act_pars.rVol*D18 << "\t";
  }
  loop2x2 << std::endl;

#else
#pragma omp parallel private(tid,a,b,d,e)  num_threads(NTHR) 
  {
    int chunk = act_pars.iVol/NTHR;
    tid = omp_get_thread_num();
    for(int site_x = chunk*tid; site_x < chunk*(tid+1);site_x++){
      
      for(int mu = 3; mu > 0; --mu){
	a = Umu.Z->L[site_x][5+mu];
	b = Umu.Z->L[a][5+mu];
	tmp[tid] = Umu.W[site_x].U[mu] * Umu.W[a].U[mu];
	
	for( int nu = 0; nu < mu; ++nu){
	  d = Umu.Z->L[site_x][5+nu];
	  e = Umu.Z->L[d][5+nu];
	  tmp1[tid] += tmp[tid] * ( Umu.W[b].U[nu] * Umu.W[ Umu.Z->L[b][5+nu] ].U[nu] * 
				    dag( Umu.W[site_x].U[nu] * Umu.W[d].U[nu] * Umu.W[e].U[mu] * 
					 Umu.W[ Umu.Z->L[e][5+mu] ].U[mu] ));
	} // nu
      } // mu
      
    } // end site_x
    
  } // end parallel
  

  // Reduce
  for(int tid = 1; tid < NTHR; tid++){
    tmp1[0] += tmp1[tid];
  }

  // std::cout << std::endl;
  for( int i1 = 1; i1 <= PTORD; i1+=2){
    // std::cout << (tmp1[0].ptU[i1].Tr()).re*act_pars.rVol*D18 << "\t";
    loop2x2 << (tmp1[0].ptU[i1].Tr()).re*act_pars.rVol*D18 << "\t";
  }

  // std::cout << std::endl;
  loop2x2 << std::endl;

#endif

}

#endif


#ifdef __MANY_LOOP__

Cplx l21[allocORD+1];
Cplx l22[allocORD+1];
Cplx l42[allocORD+1];
Cplx l44[allocORD+1];


void ComputeLoops(ptGluon_fld &Umu){

  ptSU3 br, bl, mr, ml, tr, tl;
  ptSU3 ld, lu, cd, cu, rd, ru;


  ptSU3 w22,w21,w44,w42;
#ifdef __PARALLEL_OMP__
  ptSU3 t_w22[NTHR],t_w21[NTHR],t_w44[NTHR],t_w42[NTHR];
#endif

  ptSU3 tmp;

  int a,b,c,d,e,f,g,h,j,k;
  int i,l,m,n,o,p,q,r,s,t;

  w22.zero();
  w21.zero();
  w42.zero();
  w44.zero();

#ifdef __PARALLEL_OMP__
  for( int thr = 0; thr < NTHR; thr++)
    {
      t_w22[thr].zero();
      t_w21[thr].zero();
      t_w42[thr].zero();
      t_w44[thr].zero();    
    }
#endif

  for( int mu = 0; mu < dim-1; mu++)
    {
      for( int nu = mu+1; nu < dim; nu++)
	{

#ifdef __PARALLEL_OMP__
#pragma omp parallel for private(a,b,c,d,e,f,g,h,j,k, i,l,m,n,o,p,q,r,s,t,br, bl, mr, ml, tr, tl, ld, lu, cd, cu, rd, ru) num_threads(NTHR)
#endif
	  for( int x = 0; x < Umu.Z->Size; x++)
	    {
#ifdef __PARALLEL_OMP__
	      int thr = omp_get_thread_num();
#endif
	      a = Umu.Z->get(x, 1, mu);
	      b = Umu.Z->get(a, 1, mu);
	      c = Umu.Z->get(b, 1, nu);
	      e = Umu.Z->get(x, 1, nu);
	      d = Umu.Z->get(e, 1, mu);

  	      f = Umu.Z->get(c, 1, nu);
	      g = Umu.Z->get(d, 1, nu);
	      h = Umu.Z->get(e, 1, nu);


	      bl = Umu.W[x].U[mu]*Umu.W[a].U[mu];
	      ml = Umu.W[h].U[mu]*Umu.W[g].U[mu];
	      cd = Umu.W[b].U[nu]*Umu.W[c].U[nu];
	      ld = Umu.W[x].U[nu]*Umu.W[e].U[nu];

#ifdef  __PARALLEL_OMP__
	      t_w22[thr] += bl*cd*dag(ld*ml);
	      
	      t_w21[thr] += ( bl*Umu.W[b].U[nu]*
			      dag(Umu.W[x].U[nu]*Umu.W[e].U[mu]*Umu.W[d].U[mu]) +
			      Umu.W[x].U[mu]*Umu.W[a].U[nu]*Umu.W[d].U[nu]*
			      dag( ld*Umu.W[h].U[mu]) );	      
#else
	      w22 += bl*cd*dag(ld*ml);

	      w21 += ( bl*Umu.W[b].U[nu]*
		       dag(Umu.W[x].U[nu]*Umu.W[e].U[mu]*Umu.W[d].U[mu]) +
		       Umu.W[x].U[mu]*Umu.W[a].U[nu]*Umu.W[d].U[nu]*
		       dag( ld*Umu.W[h].U[mu]) );	      
#endif

              i = Umu.Z->get(h, 1, nu);
              j = Umu.Z->get(f, 1, nu);
              k = Umu.Z->get(j, 1, nu);
              n = Umu.Z->get(i, 1, nu);
              l = Umu.Z->get(n, 1, mu);
              m = Umu.Z->get(b, 1, mu);
              o = Umu.Z->get(m, 1, mu);
              p = Umu.Z->get(o, 1, nu);
              q = Umu.Z->get(p, 1, nu);
              r = Umu.Z->get(f, 1, mu);
              s = Umu.Z->get(q, 1, nu);
              t = Umu.Z->get(k, 1, mu);

              br = Umu.W[b].U[mu]*Umu.W[m].U[mu];
              mr = Umu.W[f].U[mu]*Umu.W[r].U[mu];
              tr = Umu.W[k].U[mu]*Umu.W[t].U[mu];
              tl = Umu.W[n].U[mu]*Umu.W[l].U[mu];
              lu = Umu.W[h].U[nu]*Umu.W[i].U[nu];             
              cu = Umu.W[f].U[nu]*Umu.W[j].U[nu];
              ru = Umu.W[q].U[nu]*Umu.W[s].U[nu];
              rd = Umu.W[o].U[nu]*Umu.W[p].U[nu];
              

#ifdef  __PARALLEL_OMP__
              t_w42[thr] += bl*(br*rd*dag(ml*mr) 
				+ cd*cu*dag(lu*tl))*dag(ld);
	      
              t_w44[thr] += bl*br*rd*ru*dag(ld*lu*tl*tr);	      
#else
              w42 += bl*(br*rd*dag(ml*mr) 
                         + cd*cu*dag(lu*tl))*dag(ld);

              w44 += bl*br*rd*ru*dag(ld*lu*tl*tr);
#endif
	    } // i
	} // nu
    } // mu



#ifdef  __PARALLEL_OMP__
  for( int thr = 0; thr < NTHR; thr++)
    {
      w21+=t_w21[thr];
      w22+=t_w22[thr];
      w42+=t_w42[thr];
      w44+=t_w44[thr];
    }
#endif


  w21.Tr(l21);
  w22.Tr(l22);
  w42.Tr(l42);
  w44.Tr(l44);

  const double loopNorm = 1./(3*2*3*(Umu.Z->Size));
  // loop1x2 << "Plaquette 1x2" << endl;
  for(int i1 = 0; i1 <= PTORD; i1+=2){  
    loop2x2 << l21[i1].re*.5*loopNorm << "\t";
  }					
  
  // loop2x2 << "Plaquette 2x2" << endl;
  for(int i1 = 0; i1 <= PTORD; i1+=2){  
    loop2x2 << l22[i1].re*loopNorm << "\t";
  }					
  
  // loop2x2 << "Plaquette 2x4" << endl;
  for(int i1 = 0; i1 <= PTORD; i1+=2){  
    loop2x2 << l42[i1].re*.5*loopNorm << "\t";
  }					
  
  // cout << "Plaquette 4x4" << endl;
  for(int i1 = 0; i1 <= PTORD; i1+=2){  
    loop2x2 << l44[i1].re*loopNorm << "\t";
  }					
  loop2x2 << std::endl;

}

#endif


// Misura a configurazione congelata la placchetta
void plaquette_measure(ptGluon_fld& Umu, act_params_t& act_pars){
#ifndef __PARALLEL_OMP__
  for(int i1 = 0; i1 < PTORD+1; i1++){
    w[i1] = 0.0;
  }					
  
  Cplx* app = new Cplx[PTORD+1];
  
  for(int i = 0; i < act_pars.iVol;i++){
    W1.zero();				
    link_c = Umu.Z->L[i][dim];
    for(int mu = 0; mu < dim-1; mu++){
      W.zero();				
      for(int nu = mu+1; nu < dim; nu++){
	if(nu != mu ){
	  W += Umu.staple(i, mu, nu);
	}
      }
      W1 = Umu.W[link_c].U[mu]*W;
      W1.Tr(app);
      for(int i1 = 0; i1 <= PTORD; i1++){
	w[i1] += app[i1];
      }
    }	
  }
     
  for(int i1 = 0; i1 <= PTORD; i1++){
    w[i1] /= (act_pars.iVol*36);
  }					

  W.zero();
  W1.zero();
  delete [] app;

#else

  for(int nt = 0; nt < NTHR; nt++){
    for(int i1 = 0; i1 < PTORD+1; i1++){
      ww[nt][i1] = 0.0;
    }				
  }
  //  Cplx* app = new Cplx[PTORD+1];
  
#pragma omp parallel private(tid,link_c) num_threads(NTHR)
  {
    int chunk = act_pars.iVol/NTHR;

    tid = omp_get_thread_num();    
    for(int site_x = chunk*tid; site_x < chunk*(tid+1);site_x++){
      Ww1[tid].zero();
      //      link_c = Umu.Z->L[site_x][dim];
      for(int mu = 0; mu < dim; mu++){
 	Ww[tid].zero();
 	for(int nu = 0; nu < dim; nu++){
 	  if(nu != mu ){
 	    Ww[tid] += Umu.staple(site_x, mu, nu);
 	  }
 	}
	// 	Ww1[tid] = Umu.W[link_c].U[mu]*Ww[tid];
 	Ww1[tid] = Umu.W[site_x].U[mu]*Ww[tid];
 	Ww1[tid].Tr(ww1[tid]);
	
 	for(int i1 = 0; i1 <= PTORD; i1++){
 	  ww[tid][i1] += ww1[tid][i1];
 	}
	
      }// altro link, nello stesso sito	
    } // fine siti
    
    Ww[tid].zero();
    Ww1[tid].zero();
  } // end parallel
#pragma omp barrier

  for(int i1 = 0; i1 <= PTORD; i1++){
    w[i1] = 0;
    for(int nt = 0; nt < NTHR; nt++){
      //      std::cout << ww[nt][i1].re << "\t";
      w[i1] += ww[nt][i1];
    }
    //    std::cout << std::endl;
    w[i1] /= (act_pars.iVol*72);		
  }					
  //  std::cout << std::endl;
#endif

}


// // Misura a configurazione congelata la placchetta
// void plaquette_measure(ptGluon_fld& Umu, act_params_t& act_pars){
// #ifndef __PARALLEL_OMP__
//   for(int i1 = 0; i1 < PTORD+1; i1++){
//     w[i1] = 0.0;
//   }					
  
//   Cplx* app = new Cplx[PTORD+1];
  
//   for(int i = 0; i < act_pars.iVol;i++){
//     W1.zero();				
//     link_c = Umu.Z->L[i][dim];
//     for(int mu = 0; mu < dim; mu++){
//       W.zero();				
//       for(int nu = 0; nu < dim; nu++){
// 	if(nu != mu ){
// 	  W += Umu.staple(i, mu, nu);
// 	}
//       }
//       W1 = Umu.W[link_c].U[mu]*W;
//       W1.Tr(app);
//       for(int i1 = 0; i1 <= PTORD; i1++){
// 	w[i1] += app[i1];
//       }
//     }	
//   }
     
//   for(int i1 = 0; i1 <= PTORD; i1++){
//     w[i1] /= (act_pars.iVol*72);
//   }					

//   W.zero();
//   W1.zero();
//   delete [] app;

// #else

//   for(int nt = 0; nt < NTHR; nt++){
//     for(int i1 = 0; i1 < PTORD+1; i1++){
//       ww[nt][i1] = 0.0;
//     }				
//   }
//   //  Cplx* app = new Cplx[PTORD+1];
  
// #pragma omp parallel private(tid,link_c) num_threads(NTHR)
//   {
//     tid = omp_get_thread_num();
    
//     for(int site_x = chunk*tid; site_x < chunk*(tid+1);site_x++){
//       Ww1[tid].zero();
//       //      link_c = Umu.Z->L[site_x][dim];
//       for(int mu = 0; mu < dim; mu++){
//  	Ww[tid].zero();
//  	for(int nu = 0; nu < dim; nu++){
//  	  if(nu != mu ){
//  	    Ww[tid] += Umu.staple(site_x, mu, nu);
//  	  }
//  	}
// 	// 	Ww1[tid] = Umu.W[link_c].U[mu]*Ww[tid];
//  	Ww1[tid] = Umu.W[site_x].U[mu]*Ww[tid];
//  	Ww1[tid].Tr(ww1[tid]);
	
//  	for(int i1 = 0; i1 <= PTORD; i1++){
//  	  ww[tid][i1] += ww1[tid][i1];
//  	}
	
//       }// altro link, nello stesso sito	
//     } // fine siti
    
//     Ww[tid].zero();
//     Ww1[tid].zero();
//   } // end parallel
// #pragma omp barrier

//   for(int i1 = 0; i1 <= PTORD; i1++){
//     w[i1] = 0;
//     for(int nt = 0; nt < NTHR; nt++){
//       //      std::cout << ww[nt][i1].re << "\t";
//       w[i1] += ww[nt][i1];
//     }
//     //    std::cout << std::endl;
//     w[i1] /= (act_pars.iVol*72);		
//   }					
//   //  std::cout << std::endl;
//  #endif

// }
