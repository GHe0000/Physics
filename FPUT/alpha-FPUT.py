import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from tools.sprk import SPRK8 
from tools.sprk import Leapfrog

# 模拟参数
N = 32
dt = 0.1
alpha = 0.25
t_end = 30000

# 刚度矩阵
A = 2*np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
omega_sq, mode_shape = np.linalg.eigh(A)
omega = np.sqrt(omega_sq)

# 初始化
q0, p0 = np.zeros(N+2), np.zeros(N+2)
q0[1:N+1] = 4 * mode_shape[:,0]

def H(q, p):
    V = lambda x: 0.5 * x**2 + alpha/3 * x**3
    tmp = np.zeros_like(q)
    tmp[2:N+2] = p[1:N+1]
    return np.sum(p**2 + V(q-tmp), axis=1)

# 动力学模型
@nb.njit
def gradT(p):
    return p

@nb.njit
def gradV(q):
    u = q[1:N+1]
    u_p1 = q[2:N+2]
    u_m1 = q[0:N]
    F = np.zeros_like(q)
    linear = u_p1 - 2*u+ u_m1
    nolinear = ((u_p1 - u)**2 - (u - u_m1)**2)
    F[1:N+1] = linear + alpha * nolinear
    return -F

def energy_n(q, p, n):
    phi_n = mode_shape[:,n]
    xi = q[:,1:N+1] @ phi_n
    xi_dot = p[:,1:N+1] @ phi_n
    return 0.5 * (xi_dot**2 + omega_sq[n] * xi**2)

print("Running Simulation...")
t, q, p = Leapfrog(
    gradT=gradT,
    gradV=gradV,
    q0=q0,
    p0=p0,
    t=t_end,
    dt=dt
)
print("Simulation Complete.")

energies = np.column_stack([energy_n(q, p, n) for n in range(5)])

fix, ax = plt.subplots()
for i in range(5):
    ax.plot(t, energies[:, i], label=f"$E_{i+1}$")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Energy (J)")
ax.legend()
plt.show()

def calculate_hamiltonian(q, p):
    # 动能部分
    kinetic_energy = 0.5 * np.sum(p[1:N+1]**2)
    # 势能部分
    diff_q = q[1:N+2] - q[0:N+1]
    potential_energy = np.sum(0.5 * diff_q**2 + (alpha / 3.0) * diff_q**3)
    return kinetic_energy + potential_energy

H_t = np.array([calculate_hamiltonian(q[i], p[i]) for i in range(len(t))])
fig2, ax2 = plt.subplots(figsize=(12, 6))
energy_error = H_t - H_t[0]
ax2.plot(t, energy_error)
ax2.set_xlabel("Time", fontsize=14)
ax2.set_ylabel("Hamiltonian Error ($H(t) - H(0)$)", fontsize=14)
ax2.set_title("Hamiltonian Conservation Error", fontsize=16)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.show()

print("\nAnalysis:")
print(f"Initial total energy H(0) = {H_t[0]:.6f}")
print(f"Maximum energy error |H(t) - H(0)|_max = {np.max(np.abs(energy_error)):.6e}")
print(f"Mean energy error = {np.mean(energy_error):.6e}")
