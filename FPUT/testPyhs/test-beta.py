import numpy as np
import matplotlib.pyplot as plt
from pyhamsys import solve_ivp_symp

# --- 1. 模拟参数设置 ---
N = 32              # 粒子数
beta = 0.25         # MODIFIED: 修改 alpha 为 beta 参数
dt = 0.1            # 积分步长
t_end = 5000       # 模拟总时长

# MODIFIED: 更新打印信息
print("FPUT-beta chain simulation using pyhamsys")
print(f"Parameters: N={N}, beta={beta}, dt={dt}, t_end={t_end}")

# --- 2. 计算系统的线性模 (这部分无需改变) ---
# 系统的线性部分对应的刚度矩阵 A
A = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
# 求解特征值和特征向量，得到线性模的频率平方和振型
omega_sq, mode_shape = np.linalg.eigh(A)
omega = np.sqrt(omega_sq)

# --- 3. 定义哈密顿系统的梯度和流函数 ---

def gradT(p):
    """动能 T = sum(p_i^2 / 2) 的梯度 (无需改变)"""
    return p

def gradV(q):
    """
    势能 V 的梯度，即粒子受到的力。
    q 是一个 N+2 的数组，q[0] 和 q[N+1] 是固定的边界。
    """
    u = q[1:N+1]
    u_p1 = q[2:N+2]
    u_m1 = q[0:N]
    
    # 线性力部分保持不变
    linear_force = u_p1 - 2*u + u_m1
    
    # MODIFIED: 修改非线性力部分以匹配 FPUT-beta 模型
    # V'(x) = x + beta*x^3
    # F_i = V'(q_{i+1}-q_i) - V'(q_i-q_{i-1})
    # 非线性部分为: beta * [ (q_{i+1}-q_i)^3 - (q_i-q_{i-1})^3 ]
    nonlinear_force = beta * ((u_p1 - u)**3 - (u - u_m1)**3)
    
    F = np.zeros_like(q)
    F[1:N+1] = linear_force + nonlinear_force
    return F

# 流函数 flow_T 和 flow_V 的定义无需改变，因为它们只是调用 gradT 和 gradV
def flow_T(h, t, y):
    """动能 T(p) 对应的流 (Drift)"""
    q, p = np.split(y, 2)
    q_new = q + h * gradT(p)
    return np.concatenate([q_new, p])

def flow_V(h, t, y):
    """势能 V(q) 对应的流 (Kick)"""
    q, p = np.split(y, 2)
    p_new = p + h * gradV(q)
    return np.concatenate([q, p_new])

# --- 4. 设置初始条件 (无需改变) ---
q0 = np.zeros(N + 2)
p0 = np.zeros(N + 2)
q0[1:N+1] = 16 * mode_shape[:, 0]
y0 = np.concatenate([q0, p0])

# --- 5. 运行辛积分模拟 (无需改变) ---
t_span = (0, t_end)
t_eval = np.arange(0, t_end, 100) 

print("Running simulation...")
sol = solve_ivp_symp(
    chi=flow_V,
    chi_star=flow_T,
    t_span=t_span,
    y0=y0,
    step=dt,
    t_eval=t_eval,
    method='Verlet'
)
print("Simulation complete.")

q_traj, p_traj = np.split(sol.y, 2)

# --- 6. 后处理与分析 ---

# MODIFIED: 修改总哈密顿量的计算函数
def calculate_hamiltonian(q, p):
    # 动能部分不变
    kinetic_energy = 0.5 * np.sum(p[1:N+1]**2)
    # 势能部分
    diff_q = q[1:N+2] - q[0:N+1]
    # 修改势能项为 beta/4 * x^4
    potential_energy = np.sum(0.5 * diff_q**2 + (beta / 4.0) * diff_q**4)
    return kinetic_energy + potential_energy

# 计算单个模能量的函数 (无需改变)
def calculate_mode_energy(q, p, mode_idx):
    phi_n = mode_shape[:, mode_idx]
    xi = np.dot(q[1:N+1], phi_n)
    xi_dot = np.dot(p[1:N+1], phi_n)
    return 0.5 * (xi_dot**2 + omega_sq[mode_idx] * xi**2)

# 计算能量轨迹 (调用已修改的函数)
H_t = np.array([calculate_hamiltonian(q_traj[:, i], p_traj[:, i]) for i in range(sol.t.shape[0])])

# 计算模能量 (无需改变)
num_modes_to_plot = 5
mode_energies = np.zeros((sol.t.shape[0], num_modes_to_plot))
for i in range(sol.t.shape[0]):
    for n in range(num_modes_to_plot):
        mode_energies[i, n] = calculate_mode_energy(q_traj[:, i], p_traj[:, i], n)

# --- 7. 绘图 ---

# 图1：不同模的能量随时间变化
plt.style.use('seaborn-v0_8-whitegrid')
fig1, ax1 = plt.subplots(figsize=(12, 6))
for i in range(num_modes_to_plot):
    ax1.plot(sol.t, mode_energies[:, i], label=f'Mode ${i+1}$')

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
ax2.plot(sol.t, energy_error)
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
