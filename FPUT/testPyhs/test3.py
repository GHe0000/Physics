import numpy as np
import matplotlib.pyplot as plt
from pyhamsys import solve_ivp_symp

# --- 1. 模拟参数设置 ---
N = 32              # 粒子数
alpha = 0.25        # 非线性强度
dt = 0.1            # 积分步长
t_end = 10000       # 模拟总时长

print("FPUT-alpha chain simulation using pyhamsys")
print(f"Parameters: N={N}, alpha={alpha}, dt={dt}, t_end={t_end}")

# --- 2. 计算系统的线性模 ---
# 系统的线性部分对应的刚度矩阵 A
A = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
# 求解特征值和特征向量，得到线性模的频率平方和振型
omega_sq, mode_shape = np.linalg.eigh(A)
omega = np.sqrt(omega_sq)

# --- 3. 定义哈密顿系统的梯度和流函数 ---

def gradT(p):
    """动能 T = sum(p_i^2 / 2) 的梯度"""
    return p

def gradV(q):
    """
    势能 V 的梯度，即粒子受到的力。
    q 是一个 N+2 的数组，q[0] 和 q[N+1] 是固定的边界。
    """
    # 提取内部活动的粒子
    u = q[1:N+1]
    # 为了计算差分，需要左右两边的邻居
    u_p1 = q[2:N+2] # u_{i+1}
    u_m1 = q[0:N]   # u_{i-1}
    
    # 计算力 F = -gradV
    # 线性部分: (u_{i+1} - 2u_i + u_{i-1})
    linear_force = u_p1 - 2*u + u_m1
    # 非线性部分: alpha * [ (u_{i+1}-u_i)^2 - (u_i-u_{i-1})^2 ]
    nonlinear_force = alpha * ((u_p1 - u)**2 - (u - u_m1)**2)
    
    # 总力，并填充到 N+2 的数组中
    F = np.zeros_like(q)
    F[1:N+1] = linear_force + nonlinear_force
    return F

# pyhamsys 需要的是流函数
def flow_T(h, t, y):
    """动能 T(p) 对应的流 (Drift)"""
    q, p = np.split(y, 2)
    q_new = q + h * gradT(p)
    return np.concatenate([q_new, p])

def flow_V(h, t, y):
    """势能 V(q) 对应的流 (Kick)"""
    q, p = np.split(y, 2)
    p_new = p + h * gradV(q) # 注意：p_dot = -gradV，但这里 gradV 定义为了力
    return np.concatenate([q, p_new])

# --- 4. 设置初始条件 ---
# 初始化在第一个（最低频率）线性模上
q0 = np.zeros(N + 2)
p0 = np.zeros(N + 2)
q0[1:N+1] = 16 * mode_shape[:, 0] # 给予一个较大的初始振幅

# 组合成 pyhamsys 需要的初始状态向量 y0
y0 = np.concatenate([q0, p0])

# --- 5. 运行辛积分模拟 ---
t_span = (0, t_end)
# 为了节省内存，我们不会保存每一步的结果，而是每隔一段时间保存一次
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

# 从结果中分离出 q 和 p
q_traj, p_traj = np.split(sol.y, 2)

# --- 6. 后处理与分析 ---

# 定义计算总哈密顿量的函数
def calculate_hamiltonian(q, p):
    # 动能部分
    kinetic_energy = 0.5 * np.sum(p[1:N+1]**2)
    # 势能部分
    diff_q = q[1:N+2] - q[0:N+1]
    potential_energy = np.sum(0.5 * diff_q**2 + (alpha / 3.0) * diff_q**3)
    return kinetic_energy + potential_energy

# 定义计算单个模能量的函数
def calculate_mode_energy(q, p, mode_idx):
    phi_n = mode_shape[:, mode_idx]
    xi = np.dot(q[1:N+1], phi_n)
    xi_dot = np.dot(p[1:N+1], phi_n)
    return 0.5 * (xi_dot**2 + omega_sq[mode_idx] * xi**2)

def calculate_all_mode_energies(q, p):
    q_interior = q[1:N+1]
    p_interior = p[1:N+1]
    xi     = np.einsum("in,i->n", mode_shape, q_interior)     # 形状 (N,)
    xi_dot = np.einsum("in,i->n", mode_shape, p_interior)     # 形状 (N,)
    E_modes = 0.5 * (xi_dot**2 + omega_sq * xi**2)
    return E_modes

# 计算轨迹上每个点的总能量
H_t = np.array([calculate_hamiltonian(q_traj[:, i], p_traj[:, i]) for i in range(sol.t.shape[0])])

q_interior = q_traj[1:N+1, :]   # (N, n_steps)
p_interior = p_traj[1:N+1, :]   # (N, n_steps)
xi     = np.einsum("in,it->nt", mode_shape, q_interior)     # (N, n_steps)
xi_dot = np.einsum("in,it->nt", mode_shape, p_interior)     # (N, n_steps)
all_mode_energies = 0.5 * (xi_dot**2 + (omega_sq[:, None] * xi**2))   # (N, n_steps)
E_t = all_mode_energies.sum(axis=0)   # (n_steps,)

# 计算前 5 个模的能量
num_modes_to_plot = 5
mode_energies = np.zeros((sol.t.shape[0], num_modes_to_plot))

for i in range(sol.t.shape[0]):
    for n in range(num_modes_to_plot):
        mode_energies[i, n] = calculate_mode_energy(q_traj[:, i], p_traj[:, i], n)

# --- 7. 绘图 ---

fig1, ax1 = plt.subplots(figsize=(12, 6))
for i in range(num_modes_to_plot):
    ax1.plot(sol.t, mode_energies[:, i], label=f'Mode ${i+1}$')

ax1.set_xlabel("Time", fontsize=14)
ax1.set_ylabel("Mode Energy", fontsize=14)
ax1.set_title(f"FPUT-$\u03B1$ Chain: Mode Energy Evolution (N={N}, $\u03B1$={alpha})", fontsize=16)
ax1.legend()
ax1.set_yscale('log')
ax1.set_ylim(bottom=1e-3) # 设置y轴最小值以更好地显示能量转移
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(12, 6))
energy_error = H_t - H_t[0]
ax2.plot(sol.t, energy_error)
ax2.set_xlabel("Time", fontsize=14)
ax2.set_ylabel("Hamiltonian Error ($H(t) - H(0)$)", fontsize=14)
ax2.set_title("Hamiltonian Conservation Error", fontsize=16)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.show()

fig3, ax3 = plt.subplots(figsize=(12, 6))
Ee = E_t - E_t[0]
ax3.plot(sol.t, Ee)
plt.tight_layout()
plt.show()

print("\nAnalysis:")
print(f"Initial total H(0) = {H_t[0]:.6f}")
print(f"Maximum H error |H(t) - H(0)|_max = {np.max(np.abs(energy_error)):.6e}")
print(f"Mean H error = {np.mean(energy_error):.6e}")

print(f"Initial total E(0) = {E_t[0]:.6f}")
print(f"Maximum E error |E(t) - E(0)|_max = {np.max(np.abs(Ee)):.6e}")
print(f"Mean E error = {np.mean(Ee):.6e}")
