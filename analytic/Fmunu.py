#!/usr/bin/env python
import numpy as np
from math import pi
import sys

L = int(sys.argv[1])

Ck = np.matrix( [[-1.j * pi / 3 / L, 0, 0],
                 [0, 0, 0],
                 [0, 0, 1.j * pi / 3 / L]] )
CkpmCk = np.matrix ( [[-2.j / 3 * pi / L, 0, 0],
                      [0, 1.j / 3 * pi / L, 0],
                      [0, 0, 1.j / 3 * pi / L]] )

def Bk(t):
    return np.matrix(Ck + float(t) / L * CkpmCk)

def v(t):
    return np.diag(np.diag(np.exp(Bk(t))))

def vinv(t):
    return np.diag(np.diag(1./np.exp(Bk(t))))

def V(t, mu):
    if abs(mu) == 4:
        return np.matrix(np.identity(3) + 0j)
    if 0 < mu:
        return v(t)
    if 0 > mu:
        return vinv(t - 1)

def U(t, mu, nu):
    tmp = V(t, mu)
    if abs(mu) == 4:
        t += mu / 4
    tmp *= V(t, nu)
    if abs(nu) == 4:
        t += nu / 4
    tmp *= V(t, -mu)
    if abs(mu) == 4:
        t -= mu / 4
    tmp *= V(t, -nu)
    return tmp


def F(t, mu, nu):
    return 1. / 8 * (
        U(t, mu, nu) - U(t, nu, mu)
        #+ U(t, nu, -mu) - U(t, mu, -nu)
        #+ U(t, -mu, -nu) - U(t, -nu, -mu)
        #+ U(t, -nu, mu) - U(t, -mu, nu)
        )

def G0k():
    return CkpmCk/L

def Edens():
    return np.trace(F(L/2, 4, 1)**2).real / 2 * 3

def Edens_cont():
    return - pi**2 / L**4 


#print Edens()*L**3*2

#print F(L/2, 4, 1)*L**3
#print U(L/2, 4, 1)

# print with a factor of L**3 because of the sum in parmalgt
# and with a factor of three because of the sum over k
print np.sinh(G0k())**2 * L**3 * 3




