# 使用计算机来计算一维势能的定态波函数

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import eigsh

# 初始化参数
hbar = 1
m = 1
X = 10
N = 200
dx = 2 * X / N
En = 9

# 离散化空间坐标
x = np.linspace(-X, X, N)

# 定义一维势能
# 这里以谐振子势能为例子
v = 0.5 * x**2
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
for i in range(En):
    psi = Vec[:, i]
    E = Val[i]
    ax1 = plt.subplot(3, 3, i + 1)
    ax1.plot(x, psi**2 / np.sum(psi**2 * dx), label=r'$|\Psi(x)|^2$', color='g')
    ax1.set_xlabel('x')
    ax1.set_ylabel(r'$|\Psi(x)|^2$', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(x, v, label='V(x)', color='b')
    ax2.set_ylabel('V(x)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    plt.title(f'n={i+1} E={E:.3f}')

#plt.tight_layout() # 自动调整子图
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()
