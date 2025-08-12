import numpy as np
import numba as nb

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from SymplecticIntegrator import SPRK 

import time
from functools import wraps

def Timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function {func.__name__} : runtime: {end_time - start_time:.4f} s")
        return result
    return wrapper


k = 1.0
m = 1.0 

@nb.njit
def f(q):
    return -k*q

def scipy_f(t, y):
    q, p = y
    dq_dt = p / m
    dp_dt = -k*q
    return [dq_dt, dp_dt]

q0 = np.array([1.0])
p0 = np.array([0.0])

t = 5000.0
dt = 0.01

@Timer
def run_SPRK(f, m, y0, t, dt):
    t, q, p = SPRK(f, m, y0, t, dt)
    return t, q.T.flatten(), p.T.flatten()

@Timer
def run_rk45(scipy_f, y0, t, dt):
    sol = solve_ivp(scipy_f, [0, t], y0, t_eval=np.arange(0, t, dt), method='RK45')
    return sol.t, sol.y[0, :].flatten(), sol.y[1, :].flatten()

t1, q1, p1 = run_SPRK(f, m, (q0, p0), t, dt)
t2, q2, p2 = run_rk45(scipy_f, np.concatenate((q0, p0)), t, dt)

def calc_energy(q, p):
    t = p**2 / (2*m) 
    v = k * q**2 / 2
    return t + v

energy1 = calc_energy(q1, p1)
energy2 = calc_energy(q2, p2)

energy_error1 = (energy1 - energy1[0]) / energy1[0]
energy_error2 = (energy2 - energy2[0]) / energy2[0]

fig, axs = plt.subplots(2, 2)

axs[0,0].plot(q1, p1)
axs[0,0].set_title('SPRK')

axs[1,0].plot(q2, p2)
axs[1,0].set_title('RK45')

axs[0,1].plot(t1, energy1, label='SPRK')
axs[0,1].plot(t2, energy2, label='RK45')
axs[0,1].set_title('Energy')
axs[0,1].legend()

axs[1,1].plot(t1, energy_error1, label='SPRK')
axs[1,1].plot(t2, energy_error2, label='RK45')
axs[1,1].set_title('Energy Error')
axs[1,1].legend()

plt.show()
