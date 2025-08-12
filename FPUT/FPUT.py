import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import sprk

# -------------------
# 参数
# -------------------
N = 32                  # 内部自由度数量
alpha = 0.25            # 非线性系数
mass = np.ones(N)       # 每个内部节点质量为 1
dt = 0.05
T_total = 500

# -------------------
# 力函数（Numba 加速）
# -------------------
@nb.njit
def fput_force(q):
    force = np.zeros_like(q)
    for i in range(1, N-1):
        dq_left = q[i] - q[i-1]
        dq_right = q[i+1] - q[i]
        force[i] = -(dq_right - dq_left) \
                   - alpha * (dq_right**2 - dq_left**2)
    # 固定边界 => q[0]=0, q[N-1]=0 已在 q 定义中体现
    return force

# -------------------
# 初始条件：第一模态激发
# -------------------
A = np.diag(np.full(N, 2.0)) \
  + np.diag(np.full(N-1, -1.0), 1) \
  + np.diag(np.full(N-1, -1.0), -1)
eigvals, eigvecs = np.linalg.eigh(A)

# q0, p0 含固定边界
q0 = np.zeros(N)
p0 = np.zeros_like(q0)

# 第一模态形状
q0[:] = 0.1 * eigvecs[:, 0]  # 匹配长度 N

# -------------------
# 调用 SPRK 辛积分器
# -------------------
t_arr, q_arr, p_arr = sprk.SPRK(fput_force, mass, (q0, p0), T_total, dt)

# -------------------
# 模态能量计算
# -------------------
def modal_energy(q, p, eigvecs, eigvals):
    """计算模态能量"""
    Qm = q @ eigvecs
    Pm = p @ eigvecs
    return 0.5 * (Pm**2 + eigvals * Qm**2)

E_modes = np.zeros((len(t_arr), N))
for i in range(len(t_arr)):
    E_modes[i] = modal_energy(q_arr[i], p_arr[i], eigvecs, eigvals)

# -------------------
# 绘图：前 5 个模态能量
# -------------------
plt.figure(figsize=(8, 4))
for m in range(5):
    plt.plot(t_arr, E_modes[:, m], label=f"Mode {m+1}")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.legend()
plt.title("FPUT Recurrence (small amplitude)")
plt.tight_layout()
plt.show()
