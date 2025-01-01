import numpy as np
import scipy

import matplotlib.pyplot as plt
from matplotlib import animation

dx = 0.002                  # spatial separation
x = np.arange(0, 10, dx)    # spatial grid points

kx = 50                     # wave number
m = 1                       # mass
sigma = 0.5                 # width of initial gaussian wave-packet
x0 = 3.0                    # center of initial gaussian wave-packet

# 初始波函数
A = 1.0 / (sigma * np.sqrt(np.pi))  # normalization constant
psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)

# 势函数
V = np.zeros(x.shape)
for i, _x in enumerate(x): # enumerate 提供了一个迭代器，返回编号和内容
    if _x > 5:
        V[i] = 1000

# Laplace算符（差分）
D2 = scipy.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2

# 求解Schrodinger方程
hbar = 1
# hbar = 1.0545718176461565e-34

def psi_t(t, psi):
    return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)


# Solve the Initial Value Problem
dt = 0.001  # time interval for snapshots
t0 = 0.0    # initial time
tf = 0.2    # final time
t_eval = np.arange(t0, tf, dt)  # recorded time shots

print("Solving initial value problem")
sol = scipy.integrate.solve_ivp(psi_t,
                                t_span=[t0, tf],
                                y0=psi0,
                                t_eval=t_eval,
                                method="RK23")

print("Done.")
np.savez("sol.npz",\
         x = x,\
         V = V,\
         sol_y = np.array(sol.y),\
         sol_t = np.array(sol.t))
