import numpy as np
import numba as nb
import mathplotlib.pyplot as plt

from tools.timer import Timer

# 模型常数
N = 64 # 粒子数
L = N // 2 # 晶胞
beta = 0.1
delta = 0.1
epsilon = 0.1

# 模拟参数
dt = 0.5
t_max = 10**4

m1 = 1 - delta / 2
m2 = 1 + delta / 2

m = np.zeros(N)
m[0::2] = m1
m[1::2] = m2

# 计算简振模
@Timer
def calc_normal_mode():
    k = np.arange(1, L+1) # 波数
    
    tmp = np.sqrt(1 - (4*m1*m2)/((m1+m2)**2) * np.sin(k*np.pi/(2*L+1))**2)
    omega_sq_plus = (m1+m2)/(m1*m2) * (1+tmp)
    omega_sq_minus = (m1+m2)/(m1*m2) * (1-tmp)
    omega_sq = np.concatenate((omega_sq_plus, omega_sq_minus))
    omega = np.sqrt(omega_sq)

    u_plus = np.zeros((N,L))
    u_minus = np.zeros((N,L))

    k_idx = np.arange(1, L+1)[:, np.newaxis]
    j_idx = np.arange(1, N+1)[np.newaxis, :]

    l_idx = (j_idx+1) // 2

    tmp1 = np.sin(2*l_idx*k_idx*np.pi/(2*L+1))
    tmp2 = np.sin(2*(l_idx-1)*k_idx*np.pi/(2*L+1))
    tmp3 = np.sin(2*k_idx*np.pi/(2*L+1))

    u_plus[0::2, :] = (tmp1 + tmp2) / tmp3
    u_plus[1::2, :] = (2 - m1*omega_sq_plus) * tmp1 / tmp3

    u_minus[0::2, :] = (tmp1 + tmp2) / tmp3
    u_minus[1::2, :] = (2 - m1*omega_sq_minus) * tmp1 / tmp3

    u = np.concatenate((u_plus, u_minus), axis=1)

    U, _ = np.linalg.qr(u)

    sort_idx = np.argsort(omega)
    omega = omega[sort_idx]
    U = U[:, sort_idx]
    return omega, U

def initialize(omega, U):

    pass







