import numpy as np
import numba as nb
import matplotlib.pyplot as plt

# Assuming the SPRK8 tool is in the specified path
from tools.sprk import SPRK8

# 模拟参数
N = 32
dt = 0.01
beta = 2.0  # Changed from alpha to beta, a typical value for FPUT-beta
t_end = 20000

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
    
    # --- KEY CHANGE FOR FPUT-BETA ---
    # The nonlinear term is now cubic instead of quadratic
    nolinear = ((u_p1 - u)**3 - (u - u_m1)**3)
    
    F[1:N+1] = linear + beta * nolinear # Use beta parameter
    return -F

def H(q, p):
    def V(x):
        return 0.5 * x**2 + beta/4 * x**4
    tmp = np.zeros_like(q)
    tmp[2:N+2] = p[1:N+1]
    return np.sum(p**2 + V(q-tmp), axis=1)

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

# Calculate and plot the energy in the first 5 modes
energies = np.column_stack([energy_n(q, p, n) for n in range(5)])

fig, ax = plt.subplots(figsize=(10, 6))
for i in range(5):
    ax.plot(t, energies[:, i], label=f"$E_{i+1}$")
ax.set_xlabel("Time")
ax.set_ylabel("Modal Energy")
ax.set_title("FPUT-Beta Chain: Energy in Normal Modes")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
plt.show()

Ht = H(q, p)
print(f"Hmax = {np.max(Ht)}, Hmin = {np.min(Ht)}, Hmean = {np.mean(Ht)}")
plt.plot(t, Ht)
plt.show()
