import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# --- 1. 无量纲化参数设置 ---

# 宇宙学参数 (大小适中, 无单位)
Omega_m = 0.315
Omega_Lambda = 0.685

# 模拟参数 (无单位)
N = 5000          # 粒子数量
L_box = 1.0      # 盒子边长定义为1个单位
epsilon = 0.01   # 引力软化长度 (为盒子尺寸的一小部分)

# 从物理时间到无量纲时间的转换
# 哈勃常数 H0 ≈ 67.74 km/s/Mpc ≈ 1 / 14.4 Gyr^-1
H0_inv_Gyr = 14.4
t_start_phys_Gyr = 0.5
t_end_phys_Gyr = 13.8

# 无量纲时间
t_tilde_start = t_start_phys_Gyr / H0_inv_Gyr
t_tilde_end = t_end_phys_Gyr / H0_inv_Gyr
N_steps = 200     # 时间步数

# 核心无量纲加速度常数
C_accel = (3.0 * Omega_m) / (8.0 * np.pi)

# --- 2. 无量纲物理方程 ---
def dimensionless_friedmann(t_tilde, a):
    """无量纲弗里德曼方程 da/d(t_tilde)"""
    if a <= 0: return 0
    return np.sqrt(Omega_m / a + Omega_Lambda * a**2)

# --- 3. 计算尺度因子 a(t_tilde) ---
print("Calculating dimensionless scale factor a(t_tilde)...")
t_tilde_eval = np.linspace(t_tilde_start, t_tilde_end, N_steps)
dt_tilde = t_tilde_eval[1] - t_tilde_eval[0]

# 初始尺度因子 a 可以通过求解 t(a) 的积分反向得到，或使用近似
# 这里我们用一个合理的近似值
a_initial = (t_start_phys_Gyr / t_end_phys_Gyr)**(2./3.)

t_span = [t_tilde_start, t_tilde_end]
sol = solve_ivp(
    fun=dimensionless_friedmann,
    t_span=t_span,
    y0=[a_initial],
    t_eval=t_tilde_eval,
    method='Radau' # 依然使用稳健的求解器
)
if not sol.success:
    raise RuntimeError(f"ODE solver failed: {sol.message}")
a_t = sol.y.flatten()
print("Scale factor calculation complete.")

# --- 4. 生成无量纲初始条件 ---
def generate_initial_conditions():
    """位置和动量都是无量纲的"""
    points_per_dim = int(np.ceil(N**(1/3.)))
    grid = np.linspace(0, L_box, points_per_dim)
    xx, yy, zz = np.meshgrid(grid, grid, grid)
    pos = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    pos = pos[:N]
    # 扰动大小是网格间距的一小部分
    pos += (L_box / points_per_dim) * (np.random.rand(N, 3) - 0.5) * 0.1
    p_momentum = np.zeros_like(pos) # 无量纲动量初始为0
    return pos, p_momentum

# --- 5. 无量纲N-Body计算核心 ---
def get_dimensionless_acceleration(pos):
    """计算无量纲加速度矢量 A_tilde"""
    acc = np.zeros_like(pos)
    # 所有粒子质量相同，为 1/N，这个因子在主循环中统一处理
    for i in range(N):
        diff = pos[i] - pos
        diff = diff - L_box * np.round(diff / L_box)
        dist_sq = np.sum(diff**2, axis=1) + epsilon**2
        dist_sq[i] = np.inf
        inv_dist_cubed = dist_sq**(-1.5)
        # Sum over all other particles j
        acc[i] = np.sum(diff * inv_dist_cubed[:, np.newaxis], axis=0)
    # 注意，这里的 acc 还没有乘以 -1/N
    return -acc / N

# --- 6. 无量纲主模拟循环 ---
print("Starting dimensionless N-body simulation...")
pos, p_mom = generate_initial_conditions()
initial_pos = pos.copy()

# 初始半步 kick
acc_tilde = get_dimensionless_acceleration(pos)
p_mom += C_accel * (1.0 / a_t[0]) * acc_tilde * (dt_tilde / 2.0)

# 主循环
for i in range(N_steps - 1):
    # (Drift) 更新位置 (full step)
    a_mid = (a_t[i] + a_t[i+1]) / 2.0
    pos += (p_mom / a_mid**2) * dt_tilde
    pos %= L_box # 应用周期性边界条件

    # (Kick) 更新动量 (full step)
    acc_tilde = get_dimensionless_acceleration(pos)
    p_mom += C_accel * (1.0 / a_t[i+1]) * acc_tilde * dt_tilde

    if (i+1) % 20 == 0 or i == N_steps - 2:
        phys_time = t_tilde_eval[i+1] * H0_inv_Gyr
        print(f"Step {i+1}/{N_steps-1}, Cosmic Time: {phys_time:.2f} Gyr, a: {a_t[i+1]:.3f}")

print("Simulation finished.")
final_pos = pos

# --- 7. 可视化结果 ---
def plot_results(initial, final, box_size, t_start_Gyr, t_end_Gyr):
    fig = plt.figure(figsize=(18, 9))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(initial[:, 0], initial[:, 1], initial[:, 2], s=5, c='b', alpha=0.7)
    ax1.set_title(f"Initial Conditions (t = {t_start_Gyr:.2f} Gyr)")
    ax1.set_xlabel("X / L")
    ax1.set_ylabel("Y / L")
    ax1.set_zlabel("Z / L")
    ax1.set_xlim(0, box_size)
    ax1.set_ylim(0, box_size)
    ax1.set_zlim(0, box_size)
    ax1.view_init(elev=20., azim=30)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(final[:, 0], final[:, 1], final[:, 2], s=5, c='r', alpha=0.7)
    ax2.set_title(f"Final Structure (t = {t_end_Gyr:.2f} Gyr)")
    ax2.set_xlabel("X / L")
    ax2.set_ylabel("Y / L")
    ax2.set_zlabel("Z / L")
    ax2.set_xlim(0, box_size)
    ax2.set_ylim(0, box_size)
    ax2.set_zlim(0, box_size)
    ax2.view_init(elev=20., azim=30)
    plt.tight_layout()
    plt.show()

plot_results(initial_pos, final_pos, L_box, t_start_phys_Gyr, t_end_phys_Gyr)

# 绘制尺度因子 a(t)
plt.figure(figsize=(8, 6))
plt.plot(t_tilde_eval * H0_inv_Gyr, a_t) # 将横坐标转换回物理时间 Gyr
plt.title("Scale Factor of the Universe")
plt.xlabel("Time [Gyr]")
plt.ylabel("Scale Factor a(t)")
plt.grid(True)
plt.show()
