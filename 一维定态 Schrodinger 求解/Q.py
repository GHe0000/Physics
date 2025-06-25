# 使用计算机来计算一维势能的定态波函数

from math import factorial
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import eigsh

# 初始化参数
hbar = 1
m = 1
w = 1
X = 10
N = 2000
dx = 20 / N
En = 9


# 离散化空间坐标
x = np.linspace(-10, 10, N)

# 定义一维势能
# 有限深方势阱

v = 0.5 * w**2 * x**2

def exactPsi(x, n):
    A = ((m*w)/(np.pi*hbar))**(1/4) * 1/np.sqrt(2**n * factorial(n))
    xi = np.sqrt(m*w/hbar) * x
    Hn = hermite(n)
    psi = A * Hn(xi) * np.exp(-(xi**2)/2)
    return psi

def exactE(n):
    E = (n + 0.5) * hbar * w
    return E


# a = 6
# def voltageFunction(x):
#     if 0 < x < a:
#         return 0
#     else:
#         return 10**6 # 用大势能替代无穷大

# def exactPsi(x, n):
#     if not(0 < x < a):
#         return 0
#     A = np.sqrt(2/a)
#     psi = A * np.sin((n*np.pi)/a * x)
#     return psi
#
# def exactE(n):
#     E = (n**2 * np.pi**2 * hbar**2)/(2*m*a**2)
#     return E

# v = np.array([voltageFunction(x_) for x_ in x])
V = diags(v, 0)

# 构造 Hermitian 矩阵
A = np.ones(N)
D = spdiags([1 * A, -2 * A, 1 * A], [-1, 0, 1], N, N)
D = (-(hbar**2) / (2 * m)) * (1 / dx**2) * D

# 总哈密顿量
H = D + V

# 求特征值和特征向量
Val, Vec = eigsh(H, k=En, which='SA')

# 画图
plt.rcParams['figure.figsize'] = [12, 9]
plt.rcParams['figure.dpi'] = 300

for i in range(En):
    psi = Vec[:, i]
    E = Val[i]
    ax1 = plt.subplot(3, 3, i + 1)
    psi_exact = [exactPsi(x_, i)**2 for x_ in x]
    ax1.plot(x, psi_exact, 'r')
    ax1.plot(x, psi**2 / np.sum(psi**2 * dx),'--g', label=r'$|\Psi(x)|^2$')
    E_exact = exactE(i)
    ax1.set_xlabel('x')
    ax1.set_ylabel(r'$|\Psi(x)|^2$', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(x, v, label='V(x)', color='b')
    ax2.set_ylabel('V(x)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    plt.title(f'n={i} E={E:.3f} '+r'$E_{exact}$'+f'={E_exact:.3f}')

#plt.tight_layout() # 自动调整子图
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.savefig("Q2.png")
plt.show()
