import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import h5py

from scipy.linalg import eigh

from tools.timer import FunctionTimer
from tools.timer import LapTimer

np.random.seed(3407)

# 模型常数
N = 32 # 粒子数
L = N // 2 # 晶胞
delta = 0.1
epsilon = 0.01
beta = 0.0

# 模拟参数
dt = 0.1
t_max = 10**4
# t_max = 200

# 保存参数
save_dt = 200 # 每隔多少计算的时间保存一次
save_path = "./save/data.h5"

# 粒子质量
m1 = 1 - delta / 2
m2 = 1 + delta / 2

# Function

# 哈密顿量
def H(q, p, m):
    V = lambda x: x**2/2  + beta * x**4/4
    q_m1 = np.concatenate(([0.],q[:-1]))
    H = np.sum(p**2/(2*m)) + np.sum(V(q-q_m1)) + V(q[-1])
    return H

# 简振模
# def normal_mode():
#     # NOTE: 通过刚度矩阵进行特征值计算得到的简振模
#     A = 2*np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
#     M = np.diag(m)
#     omega_sq, U = eigh(A, b=M)
#     return np.sqrt(omega_sq), U

def normal_mode(): 
    # WARNING: 尝试复现原始论文的简振模，但存在一些问题
    k = np.arange(1, L+1) # 波数
    
    tmp = np.sqrt(1 - (4*m1*m2)/((m1+m2)**2) * np.sin(k*np.pi/(2*L+1))**2)
    omega_sq_plus = (m1+m2)/(m1*m2) * (1+tmp) # plus 为 optical branch（光学支）
    omega_sq_minus = (m1+m2)/(m1*m2) * (1-tmp) # minus 为 acoustic branch（声学支）
    omega_sq = np.concatenate((omega_sq_minus, omega_sq_plus[::-1]))
    omega = np.sqrt(omega_sq)

    u_plus = np.zeros((N,L))
    u_minus = np.zeros((N,L))

    k_idx = np.arange(1, L+1)[:, np.newaxis]
    l_idx = np.arange(1, N//2 + 1)[:, np.newaxis]

    tmp1 = np.sin(2*l_idx*k_idx*np.pi/(2*L+1))
    tmp2 = np.sin(2*(l_idx-1)*k_idx*np.pi/(2*L+1))
    tmp3 = np.sin(2*k_idx*np.pi/(2*L+1))

    u_plus[0::2, :] = (tmp1 + tmp2) / tmp3
    u_plus[1::2, :] = (2 - m1*omega_sq_plus) * tmp1 / tmp3

    u_minus[0::2, :] = (tmp1 + tmp2) / tmp3
    u_minus[1::2, :] = (2 - m1*omega_sq_minus) * tmp1 / tmp3

    u = np.concatenate((u_minus, u_plus[::-1, :]), axis=1)

    U, _ = np.linalg.qr(u)
    return omega, U

# 计算每个模的能量
def Ek(q, p, m, omega, U):
    Q = np.einsum('ji,j,j->i', U, m, q) # 更高效的写法
    P = np.einsum('ji,j->i', U, p)
    # Q = np.einsum('j,jk->k', q, U)
    # P = np.einsum('j,jk->k', p/m, U)
    Ek = 0.5 * (P**2 + omega**2 * Q**2)
    return Ek

# 初始化
def initialize(m, omega, U, epsilon, k=0.1):
    omega = omega.copy()
    U = U.copy()

    sort_idx = np.argsort(omega)
    omega = omega[sort_idx]
    U = U[:, sort_idx]

    N_excited = int(k*N)
    N_excited = 1 if N_excited < 1 else N_excited

    Etot = N * epsilon
    Ek = np.zeros(N)
    Ek[:N_excited] = Etot / N_excited

    phi_k = np.random.uniform(0, 2*np.pi, N)
    Q = np.sqrt(2*Ek / omega**2) * np.sin(phi_k)
    P = np.sqrt(2*Ek) * np.cos(phi_k)

    # q0 = np.einsum('k,jk->j', Q, U)
    # p0 = m * np.einsum('k,jk->j', P, U)

    q0 = np.einsum('ij,j->i', U, Q)
    p0 = np.einsum('i,ij,j->i', m, U, P)

    return q0, p0

if __name__ == '__main__':
    # 初始化
    m = np.zeros(N)
    m[0::2] = m1
    m[1::2] = m2

    omega, U = normal_mode()
    q0, p0 = initialize(m, omega, U, epsilon)

    plt.plot(Ek(q0, p0, m, omega, U))
    plt.show()
    print(f"H={H(q0, p0, m)}, sum Ek={np.sum(Ek(q0, p0, m, omega, U))}")
