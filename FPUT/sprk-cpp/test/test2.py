import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from functools import wraps

from SymplecticIntegrator import SPRK as SPRK_numba

import sys
import os

msys2_clang_bin_path = "D:/Software/msys2/clang64/bin"
if hasattr(os, 'add_dll_directory') and os.path.exists(msys2_clang_bin_path):
    os.add_dll_directory(msys2_clang_bin_path)
sys.path.append(os.path.dirname(__file__))
import sprk as sprk_cpp


def Timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function {func.__name__:<15}: runtime: {end_time - start_time:.4f} s")
        return result
    return wrapper


k = 1.0
m = 1.0

@nb.njit
def force_numba(q):
    return -k * q

@nb.njit
def force_py(q):
    return -k * q

def scipy_f(t, y):
    q, p = y
    dq_dt = p / m
    dp_dt = -k * q
    return [dq_dt, dp_dt]

q0 = np.array([1.0])
p0 = np.array([0.0])
t_total = 5000.0
dt = 0.01


@Timer
def run_SPRK_numba(f, m, y0, t, dt):
    t, q, p = SPRK_numba(f, m, y0, t, dt)
    return t, q.T.flatten(), p.T.flatten()

@Timer
def run_SPRK_cpp(f, m, y0, t, dt):
    # The C++ function expects q0 and p0 as separate arguments
    q_initial, p_initial = y0
    t, q, p = sprk_cpp.sprk_with_pyforce(f, m, q_initial, p_initial, t, dt)
    return t, q.T.flatten(), p.T.flatten()

@Timer
def run_rk45(scipy_f, y0, t, dt):
    sol = solve_ivp(scipy_f, [0, t], y0, t_eval=np.arange(0, t, dt), method='RK45')
    return sol.t, sol.y[0, :].flatten(), sol.y[1, :].flatten()

# --- Main Execution Block ---
print("Running simulations...")
t1, q1, p1 = run_SPRK_numba(force_numba, m, (q0, p0), t_total, dt)
t2, q2, p2 = run_rk45(scipy_f, np.concatenate((q0, p0)), t_total, dt)
t3, q3, p3 = run_SPRK_cpp(force_py, m, (q0, p0), t_total, dt)
print("-" * 30)


# --- Analysis and Plotting ---
def calc_energy(q, p):
    kinetic = p**2 / (2 * m)
    potential = k * q**2 / 2
    return kinetic + potential

energy1 = calc_energy(q1, p1)
energy2 = calc_energy(q2, p2)
energy3 = calc_energy(q3, p3)

energy_error1 = (energy1 - energy1[0]) / energy1[0]
energy_error2 = (energy2 - energy2[0]) / energy2[0]
energy_error3 = (energy3 - energy3[0]) / energy3[0]

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Integrator Comparison: Numba vs. C++ vs. Scipy RK45', fontsize=16)

axs[0,0].plot(q1, p1, color='blue')
axs[0,0].set_title('SPRK (Numba)')
axs[0,0].set_xlabel('Position (q)')
axs[0,0].set_ylabel('Momentum (p)')
axs[0,0].grid(True)

axs[0,1].plot(q3, p3, color='green')
axs[0,1].set_title('SPRK (C++)')
axs[0,1].set_xlabel('Position (q)')
axs[0,1].grid(True)

axs[0,2].plot(q2, p2, color='red')
axs[0,2].set_title('RK45 (Scipy)')
axs[0,2].set_xlabel('Position (q)')
axs[0,2].grid(True)

axs[1,0].plot(t1, energy1, label='SPRK (Numba)', color='blue')
axs[1,0].plot(t3, energy3, label='SPRK (C++)', color='green', linestyle='--')
axs[1,0].plot(t2, energy2, label='RK45 (Scipy)', color='red')
axs[1,0].set_title('Energy vs. Time')
axs[1,0].set_xlabel('Time (s)')
axs[1,0].set_ylabel('Energy')
axs[1,0].legend()
axs[1,0].grid(True)

axs[1,1].plot(t1, energy_error1, label='SPRK (Numba)', color='blue')
axs[1,1].plot(t3, energy_error3, label='SPRK (C++)', color='green', linestyle='--')
axs[1,1].set_title('Energy Error (SPRK Methods)')
axs[1,1].set_xlabel('Time (s)')
axs[1,1].set_ylabel('Relative Energy Error')
axs[1,1].legend()
axs[1,1].grid(True)

axs[1,2].plot(t2, energy_error2, label='RK45 (Scipy)', color='red')
axs[1,2].set_title('Energy Error (RK45 Method)')
axs[1,2].set_xlabel('Time (s)')
axs[1,2].legend()
axs[1,2].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
