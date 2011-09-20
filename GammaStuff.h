#ifndef _GAMMA_STUFF_
#define _GAMMA_STUFF_

#include"MyMath.h"


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

Cplx gmuval[15][4] = {
  {Cplx(0,-1),Cplx(0,-1),Cplx(0,1),Cplx(0,1)}, //gm1
  {       -1 ,        1 ,       1 ,      -1 },
  {Cplx(0,-1),Cplx(0,1),Cplx(0,1),Cplx(0,-1)},
  {        1 ,       1 ,       1 ,        1 },
  {        1 ,       1 ,      -1 ,       -1 }, // gm5
  {Cplx(0,1),Cplx(0,1),Cplx(0,1),Cplx(0,1)}, //gm51
  {        1 ,      -1 ,       1 ,       -1 },
  {Cplx(0,1),Cplx(0,-1),Cplx(0,1),Cplx(0,-1)},
  {       -1 ,      -1 ,       1 ,        1 },
  {Cplx(0,1),Cplx(0,-1),Cplx(0,1),Cplx(0,-1)}, //gm12
  {       -1 ,       1 ,      -1 ,        1 },
  {Cplx(0,-1),Cplx(0,-1),Cplx(0,1),Cplx(0,1)},
  {Cplx(0, 1),Cplx(0, 1),Cplx(0,1),Cplx(0,1)},
  {       -1 ,       1 ,       1 ,       -1 },
  {Cplx(0,-1),Cplx(0,1),Cplx(0,1),Cplx(0,-1)}
};

#endif
