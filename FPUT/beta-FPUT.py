import numpy as np
import numba as nb
import matplotlib.pyplot as plt

# Assuming the SPRK8 tool is in the specified path
from tools.sprk import SPRK8
from tools.sprk import Leapfrog

# 模拟参数
N = 32
dt = 0.1
beta = 0.5
t_end = 50000

# 刚度矩阵 (Stiffness Matrix)
A = 2*np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
omega_sq, mode_shape = np.linalg.eigh(A)
omega = np.sqrt(omega_sq)

# 初始化 (Initialization)
q0, p0 = np.zeros(N+2), np.zeros(N+2)
q0[1:N+1] = 4 * mode_shape[:,0]

# 动力学模型 (Dynamical Model)
@nb.njit
def gradT(p):
    return p

@nb.njit
def gradV(q):
    u = q[1:N+1]
    u_p1 = q[2:N+2]
    u_m1 = q[0:N]
    F = np.zeros_like(q)
    linear = u_p1 - 2*u + u_m1
    
    nolinear = ((u_p1 - u)**3 - (u - u_m1)**3)
    
    F[1:N+1] = linear + beta * nolinear # Use beta parameter
    return -F

def energy_n(q, p, n):
    """Calculates the energy in the nth normal mode."""
    phi_n = mode_shape[:,n]
    xi = q[:,1:N+1] @ phi_n
    xi_dot = p[:,1:N+1] @ phi_n
    return 0.5 * (xi_dot**2 + omega_sq[n] * xi**2)

print("Running FPUT-Beta Simulation...")
t, q, p = SPRK8(
    gradT=gradT,
    gradV=gradV,
    q0=q0,
    p0=p0,
    t=t_end,
    dt=dt
)
print("Simulation Complete.")

def calculate_hamiltonian(q, p):
    kinetic_energy = 0.5 * np.sum(p[1:N+1]**2)
    diff_q = q[1:N+2] - q[0:N+1]
    potential_energy = np.sum(0.5 * diff_q**2 + (beta / 4.0) * diff_q**4)
    return kinetic_energy + potential_energy

def calculate_mode_energy(q, p, mode_idx):
    phi_n = mode_shape[:, mode_idx]
    xi = np.dot(q[1:N+1], phi_n)
    xi_dot = np.dot(p[1:N+1], phi_n)
    return 0.5 * (xi_dot**2 + omega_sq[mode_idx] * xi**2)

H_t = np.array([calculate_hamiltonian(q[i], p[i]) for i in range(len(t))])

# 计算模能量 (无需改变)
num_modes_to_plot = 5
mode_energies = np.zeros((t.shape[0], num_modes_to_plot))
for i in range(t.shape[0]):
    for n in range(num_modes_to_plot):
        mode_energies[i, n] = calculate_mode_energy(q[i], p[i], n)

# --- 7. 绘图 ---

# 图1：不同模的能量随时间变化
plt.style.use('seaborn-v0_8-whitegrid')
fig1, ax1 = plt.subplots(figsize=(12, 6))
for i in range(num_modes_to_plot):
    ax1.plot(t, mode_energies[:, i], label=f'Mode ${i+1}$')

ax1.set_xlabel("Time", fontsize=14)
ax1.set_ylabel("Mode Energy", fontsize=14)
# MODIFIED: 更新图表标题
ax1.set_title(f"FPUT-beta Chain: Mode Energy Evolution (N={N}, $\u03B2$={beta})", fontsize=16)
ax1.legend()
ax1.set_yscale('log')
ax1.set_ylim(bottom=1e-3)
plt.tight_layout()
plt.show()

# 图2：哈密顿量误差随时间变化 (绘图代码无需改变)
fig2, ax2 = plt.subplots(figsize=(12, 6))
energy_error = H_t - H_t[0]
ax2.plot(t, energy_error)
ax2.set_xlabel("Time", fontsize=14)
ax2.set_ylabel("Hamiltonian Error ($H(t) - H(0)$)", fontsize=14)
ax2.set_title("Hamiltonian Conservation Error (FPUT-$\u03B2$)", fontsize=16) # MODIFIED
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.show()

print(f"\nAnalysis:")
print(f"Initial total energy H(0) = {H_t[0]:.6f}")
print(f"Maximum energy error |H(t) - H(0)|_max = {np.max(np.abs(energy_error)):.6e}")
print(f"Mean energy error = {np.mean(energy_error):.6e}")

