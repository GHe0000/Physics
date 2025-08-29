# 概念性代码，具体API请参考官方文档
import numpy as np
import pyhamsys as phs

# 1. 定义哈密顿系统的演化
# 定义与动能T(p)相关的流动
def flow_T(dt, y):
    q, p = y
    q_new = q + dt * p # 假设 p 是广义速度
    return np.array([q_new, p])

# 定义与势能V(q)相关的流动
def flow_V(dt, y):
    q, p = y
    p_new = p - dt * (-np.sin(q)) # 以单摆的势能为例 V(q) = -cos(q)
    return np.array([q, p_new])

# 2. 设置初始条件和积分参数
y0 = np.array([np.pi / 2, 0.0]) # 初始角度和动量
t_span = (0, 100)
dt = 0.1

# 3. 使用辛积分器求解
# pyHamSys 提供了高阶辛积分方法
sol = phs.solve_ivp_symp(
    chi=flow_T,          # 对应哈密顿量的一部分
    chi_star=flow_V,     # 对应哈密顿量的另一部分
    y0=y0,
    t_span=t_span,
    step=dt,
    method='BM4'         # 选择一个四阶的辛积分方法
)

# sol.y 将包含积分后的轨迹
import matplotlib.pyplot as plt

plt.plot(sol.t, sol.y[0], label='q')
plt.show()
