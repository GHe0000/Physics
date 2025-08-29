import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from pyhs import HamSys, solve_ivp_sympext


# -----------------------
# 1. 定义 FPUT-beta 系统
# -----------------------
def make_fput_beta(N=32, beta=0.1):
    q = sp.symbols(f'q0:{N}')
    p = sp.symbols(f'p0:{N}')
    t = sp.Symbol('t')

    # 固定边界
    q_ext = [0, *q, 0]

    # 势能
    V = 0
    for j in range(N + 1):
        dq = q_ext[j + 1] - q_ext[j]
        V += 0.5 * dq ** 2 + beta / 4 * dq ** 4

    # 动能
    T = sum(pi ** 2 for pi in p) / 2
    H = sp.Lambda((q, p, t), T + V)

    hs = HamSys(ndof=N, btype='pq')
    hs.compute_vector_field(lambda q, p, t: T + V, output=False)
    hs.hamiltonian = H  # 保存符号 Hamiltonian
    return hs


# -----------------------
# 2. 模能量计算
# -----------------------
def mode_energy(sol, N):
    q, p = np.split(sol.y, 2)
    M = len(sol.t)

    # 正交变换到模态坐标
    k = np.arange(1, N + 1)
    n = np.arange(1, N + 1)
    S = np.sqrt(2 / (N + 1)) * np.sin(np.pi * np.outer(k, n) / (N + 1))

    Q = S @ q
    P = S @ p
    omega = 2 * np.sin(k * np.pi / (2 * (N + 1)))

    E = 0.5 * (P ** 2 + (omega[:, None] ** 2) * Q ** 2)
    return E


# -----------------------
# 3. 主程序
# -----------------------
if __name__ == "__main__":
    N = 32
    beta = 0.25
    hs = make_fput_beta(N, beta)

    # 初始条件：激发第一模
    q0 = np.sin(np.pi * np.arange(1, N + 1) / (N + 1))
    p0 = np.zeros(N)
    y0 = np.concatenate([q0, p0])

    # 演化
    t_span = (0, 2000)
    step = 0.1
    t_eval = np.linspace(*t_span, 2001)

    sol = solve_ivp_sympext(
        hs, t_span, y0, step=step, t_eval=t_eval,
        method="BM4", check_energy=True
    )

    # 计算模能量
    E_modes = mode_energy(sol, N)

    # -----------------------
    # 4. 绘图
    # -----------------------
    plt.figure(figsize=(10, 6))
    for m in range(4):  # 画前 4 个模
        plt.plot(sol.t, E_modes[m], label=f"Mode {m + 1}")
    plt.xlabel("Time")
    plt.ylabel("Mode Energy")
    plt.title("FPUT-β Mode Energy Evolution")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Hamiltonian 演化（检查保辛性）
    H_vals = [hs.hamiltonian(q, p, 0) for q, p in zip(sol.y[:N].T, sol.y[N:].T)]
    H0 = H_vals[0]

    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, np.array(H_vals) - H0)
    plt.xlabel("Time")
    plt.ylabel("ΔH")
    plt.title("Hamiltonian Conservation Check")
    plt.grid(True)
    plt.show()
