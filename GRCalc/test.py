from GR_Func import *
from Numberical_Calc_Geodesic import *

import matplotlib.pyplot as plt
import numba as nb

sym.init_printing()

@nb.jit()
def calc(GammaFunc, x_0, v_0, s, ds):
    v = v_0
    x = x_0
    x_result = x_0
    v_result = v_0
    for i in np.arange(0,s,ds):
        G = GammaFunc(*x)
        part = np.einsum("ijk,j->ik",G,v,optimize="True")
        dv = -np.array(np.einsum("ik,k->i",part,v,optimize="True"))

        x = x + v*ds
        v = v + dv*ds
        x_result = np.vstack((x_result,x))
        v_result = np.vstack((v_result,v))
    x_result = np.delete(x_result,0,0)
    v_result = np.delete(v_result,0,0)
    return x_result, v_result

t,r,theta,phi,psi = sym.symbols("t,r,theta,phi,psi",real=True)
k,R = sym.symbols("k,R")
G,M = sym.symbols("G,M")

R = 10

x = [theta,phi]

g = sym.matrices.diag(r**2,(r*sym.sin(theta))**2)

Gamma = CalcChristoffelSymbol(g,x)

GammaFunc = sym.lambdify(x,Gamma,"numpy")

x_0 = np.array([np.pi/2,0])
v_0 = np.array([0.1,0.1])
s = 100
ds = 0.0001

s_result = np.arange(0,s,ds)

x_result,v_result = calc(GammaFunc,x_0,v_0,s,ds)
