import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from tools.sprk import SPRK8

def make_gradT(m_inv):
    @nb.njit
    def gradT_jitted(p):
        return p * m_inv
    return gradT_jitted

@nb.njit
def gradV(q):
    N = len(q)
    F = np.zeros(N)
    x = q[1:] - q[:-1]
    
    # V'(x) = (exp(2x) - 1)/2
    V_prime = 0.5 * (np.exp(2 * x) - 1)

    # F_i = V'(q_{i+1}-q_i) - V'(q_i-q_{i-1})
    F[1:-1] = V_prime[1:] - V_prime[:-1]
    return -F

def get_diatomic_normal_modes(N, m1, m2):
    """计算双原子链的简正模式（频率和振型）"""
    # 构建动力学矩阵
    D = np.zeros((N, N))
    masses = np.array([m1 if i % 2 == 0 else m2 for i in range(N)])
    
    for i in range(N):
        inv_sqrt_m_i = 1.0 / np.sqrt(masses[i])
        D[i, i] = 2.0 * inv_sqrt_m_i**2
        if i > 0:
            D[i, i-1] = -1.0 * inv_sqrt_m_i * (1.0 / np.sqrt(masses[i-1]))
        if i < N - 1:
            D[i, i+1] = -1.0 * inv_sqrt_m_i * (1.0 / np.sqrt(masses[i+1]))
            
    # 求解本征值问题
    omega_sq, C = eigh(D)
    
    # 本征向量（振型）
    mode_shapes = np.zeros_like(C)
    for i in range(N):
        mode_shapes[:, i] = C[:, i] / np.sqrt(masses)
        
    # 归一化振型
    for i in range(N):
        norm = np.sqrt(np.sum(masses * mode_shapes[:, i]**2))
        mode_shapes[:, i] /= norm
        
    return np.sqrt(omega_sq), mode_shapes

def energy_n(q, p, n, masses, omega_sq, mode_shapes):
    phi_n = mode_shapes[:, n]
    Q_n = np.sum(q[:, 1:-1] * masses * phi_n, axis=1)
    P_n = np.sum(p[:, 1:-1] * phi_n, axis=1)
    return 0.5 * (P_n**2 + omega_sq[n] * Q_n**2)

# ==============================================================================
# 分析函数
# ==============================================================================
def calculate_xi(modal_energies):
    N_particles = modal_energies.shape[0]
    n_steps = modal_energies.shape[1]
    L = N_particles // 2  # 声学模和光学模的分界

    # 1. 计算时间平均能量 E_bar_k(t)
    cumulative_energy = np.cumsum(modal_energies, axis=1)
    time_steps_array = np.arange(1, n_steps + 1)
    E_bar = cumulative_energy / time_steps_array

    # 提取光学模式的能量 (k > L, in 0-based index: k >= L)
    E_bar_optical = E_bar[L:, :]
    
    # 2. 计算 w_k(t)
    sum_E_bar_optical = np.sum(E_bar_optical, axis=0)
    sum_E_bar_optical[sum_E_bar_optical < 1e-12] = 1e-12 # 防止除以零
    w_k = E_bar_optical / sum_E_bar_optical

    # 3. 计算谱熵 eta(t)
    w_k[w_k < 1e-12] = 1e-12 # 防止 log(0)
    eta = -np.sum(w_k * np.log(w_k), axis=0)

    # 4. 计算 xi_tilde(t)
    sum_E_bar_all = np.sum(E_bar, axis=0)
    sum_E_bar_all[sum_E_bar_all < 1e-12] = 1e-12
    xi_tilde = sum_E_bar_optical / (0.5 * sum_E_bar_all)

    # 5. 计算最终的 xi(t)
    xi_t = xi_tilde * (np.exp(eta) / L)
    
    return xi_t

# ---- 模拟参数 ----
N_particles = 32   # 运动的粒子数 (为加快计算，适当减小)
N_total = N_particles + 2 # 总坐标数，包括两个固定的端点
dt = 0.5           # 时间步长
t_end = 500000     # 模拟总时长 (适当延长以观察热化)
delta_m = 1.0      # 质量差，这是破坏可积性的微扰！

m1 = 1.0 - delta_m / 2.0
m2 = 1.0 + delta_m / 2.0

# ---- 初始化质量和位置动量 ----
masses = np.array([m1 if i % 2 != 0 else m2 for i in range(N_particles)])
m_inv_particles = 1.0 / masses

m_total = np.ones(N_total)
m_total[1:-1] = masses
m_inv_total = np.ones(N_total)
m_inv_total[1:-1] = m_inv_particles

# 计算简正模式
omega, mode_shapes = get_diatomic_normal_modes(N_particles, m1, m2)
omega_sq = omega**2

# ---- 设置初始条件 ----
q0 = np.zeros(N_total)
p0 = np.zeros(N_total)

initial_energy = 4.0 # 适当减小能量
num_excited_modes = 5
energy_per_mode = initial_energy / num_excited_modes

for i in range(num_excited_modes):
    amplitude = np.sqrt(2 * energy_per_mode)
    p0[1:-1] += amplitude * mode_shapes[:, i]

gradT = make_gradT(m_inv_total)

# ---- 运行模拟 ----
print(f"Running Simulation for Δm = {delta_m}...")
t, q, p = SPRK8(
    gradT=gradT,
    gradV=gradV,
    q0=q0,
    p0=p0,
    t=t_end,
    dt=dt,
)
print("Simulation Complete.")

# ---- 分析和可视化 ----
print("Analyzing results...")
modal_energies = np.array([
    energy_n(q, p, n, masses, omega_sq, mode_shapes) for n in range(N_particles)
])

# ---- 绘制能量谱演化图 ----
fig1, ax1 = plt.subplots(figsize=(10, 6))
plot_times = [0, int(t_end/dt/100), int(t_end/dt/10), int(t_end/dt)]

for i, time_idx in enumerate(plot_times):
    time_val = t[time_idx]
    avg_window = max(1, int(time_idx * 0.1))
    start_idx = max(0, time_idx - avg_window)
    avg_energies = np.mean(modal_energies[:, start_idx:time_idx+1], axis=1)
    total_energy = np.sum(avg_energies)
    if total_energy > 1e-9:
        normalized_energies = avg_energies * N_particles / total_energy
    else:
        normalized_energies = avg_energies
    ax1.plot(np.arange(N_particles) / N_particles, normalized_energies, 
             label=f't = {time_val:.0f}', alpha=0.8)

ax1.axhline(1.0, color='k', linestyle='--', label='Equipartition')
ax1.axvline(0.5, color='gray', linestyle=':', label='Acoustic/Optical Boundary')
ax1.text(0.25, 50, 'Acoustic', ha='center', color='gray')
ax1.text(0.75, 50, 'Optical', ha='center', color='gray')
ax1.set_yscale('log')
ax1.set_ylim(1e-4, 1e2)
ax1.set_xlabel("Normalized Mode Index (k/N)")
ax1.set_ylabel("Normalized Modal Energy <E_k> / <E>")
ax1.set_title(f"Energy Distribution Evolution (Δm = {delta_m})")
ax1.legend()
ax1.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()

# ---- 计算并绘制热化指标 xi(t) ----
print("Calculating thermalization metric xi(t)...")
xi_t = calculate_xi(modal_energies)

# 寻找 T_eq (xi(t) = 0.5 的时刻)
try:
    # 使用线性插值找到更精确的 T_eq
    T_eq = np.interp(0.5, xi_t, t)
    print(f"Equipartition Time T_eq (at xi=0.5) ≈ {T_eq:.2f}")
except Exception as e:
    T_eq = -1
    print(f"Could not determine T_eq. System may not have thermalized. Error: {e}")

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(t, xi_t, label='$\\xi(t)$')
ax2.axhline(0.5, color='r', linestyle='--', label='Threshold = 0.5')

if T_eq > 0:
    ax2.axvline(T_eq, color='k', linestyle=':', 
                label=f'$T_{{eq}} \\approx {T_eq:.0f}$')

ax2.set_xscale('log')
ax2.set_xlabel("Time (log scale)")
ax2.set_ylabel("Thermalization Metric $\\xi(t)$")
ax2.set_title(f"Evolution of Thermalization Metric (Δm = {delta_m})")
ax2.set_ylim(0, 1.1)
ax2.legend()
ax2.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.show()
