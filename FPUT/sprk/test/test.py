import sys
import os

msys2_clang_bin_path = "D:/Software/msys2/clang64/bin"
if hasattr(os, 'add_dll_directory') and os.path.exists(msys2_clang_bin_path):
    os.add_dll_directory(msys2_clang_bin_path)
sys.path.append(os.path.dirname(__file__))

import numpy as np
import sprk

def force(q):
    return -q

mass = np.array([1.0])
q0 = np.array([1.0])
p0 = np.array([0.0])
dt = 0.01
t_total = 1000

t, q, p = sprk.sprk_with_pyforce(force, mass, q0, p0, t_total, dt)
print(t.shape, q.shape, p.shape)

def Energy(q, p):
    return 0.5 * mass * q ** 2 + 0.5 * p ** 2

E = Energy(q, p)

p_exact = -q * np.exp(-t / t_total)
E_exact = Energy(q0, p_exact)

import matplotlib.pyplot as plt

t_end = int(10 / dt)
plt.plot(t[:t_end], p[:t_end])
plt.plot(t[:t_end], p_exact[:t_end])
plt.show()

