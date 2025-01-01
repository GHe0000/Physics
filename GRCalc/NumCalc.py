import numpy as np
import numba as nb
from scipy import integrate
from matplotlib import pyplot as plt

from constant import *

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# Constant
M = Me

G = 1
M = 1
c = 1

def CalcMatic(x):
    t, r, theta, phi = x
    g = np.diag([-1 + 2*G*M/(c**2*r),\
                 1/(1 - 2*G*M/(c**2*r)),\
                 r**2,\
                 r**2*np.sin(theta)**2])
    return g

def CalcVtTest(x,v_phi):
    g = CalcMatic(x)
    g_00 = g[0,0]
    g_33 = g[3,3]
    return np.sqrt((-c**2 - g_33*v_phi**2)/g_00)

def CalcChrs(t,r,theta,phi):
    chrs = np.zeros(shape=(4,4,4))
    chrs[0][0][1] = chrs[0][1][0] = G*M/(r*(c**2*r - 2*G*M))
    chrs[1][0][0] = G*M*(c**2*r - 2*G*M)/(c**4*r**3)
    chrs[1][1][1] = -G*M/(r*(c**2*r - 2*G*M))
    chrs[1][2][2] = -r + 2*G*M/c**2
    chrs[1][3][3] = (-c**2*r + 2*G*M)*np.sin(theta)**2/c**2
    chrs[2][1][2] = chrs[2][2][1] = 1/r
    chrs[2][3][3] = -np.sin(2*theta)/2
    chrs[3][1][3] = chrs[3][3][1] = 1/r
    chrs[3][2][3] = chrs[3][3][2] = 1/np.tan(theta)
    return chrs

def CalcDetla(x,v):
    chrs = CalcChrs(*x)
    dx = v
    part = np.einsum("ijk,j->ik",chrs,v,optimize="True")
    dv = -np.einsum("ik,k->i",part,v,optimize="True")
    return dx,dv

def func(s,X):
    x = X[0:4]
    v = X[4:8]
    dx, dv = CalcDetla(x,v)
    dX = np.hstack((dx,dv))
    return dX

s = 1000
ds = 1e-2

a = G*M/(c**2)

''' 
x_0 = np.array([0, 100, np.pi/2, 0])
v_t = CalcVtTest(x_0,v_phi=5000)
v_0 = np.array([v_t, 0, 0, 5000])
'''
vel = 0.55 * c
r_0 = 7*a
x_0 = np.array([0, r_0, np.pi/2, 0])
v_t = CalcVtTest(x_0,v_phi=(vel/r_0))
v_0 = np.array([v_t, 0, 0, (vel/r_0)])

X_0 = np.hstack((x_0,v_0))
S = np.arange(0,s,ds)

sol = integrate.solve_ivp(func,(0,s),X_0,\
                          method="RK45",\
                          rtol=0.25*ds,\
                          first_step=0.75*ds,\
                          max_step=5*ds)

t_t = sol.t
r_t = sol.y[1]
phi_t = sol.y[3]

X_t = r_t*np.cos(phi_t)
Y_t = r_t*np.sin(phi_t)

plt.plot(0,0,marker="o",color="coral")
plt.plot(X_t,Y_t)
plt.title(r"$R = 0.35R_s$、$v=0.55$")
dc = plt.Circle((0, 0), 2*a,fill=False)
plt.gca().add_artist(dc)
plt.show()
