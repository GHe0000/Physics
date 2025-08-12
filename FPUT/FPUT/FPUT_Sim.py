import numpy as np
import matplotlib.pyplot as plt
import time as pytime

# 参数
N = 32
delta = np.sqrt(1/8)
alpha = 1.0
steps = 20000

# 刚度矩阵与模态
A = 2*np.eye(N-1) - np.eye(N-1, k=1) - np.eye(N-1, k=-1)
e1, e2 = np.linalg.eigh(A)  # eigh 自动排序

# 初始化
u, v = np.zeros(N+1), np.zeros(N+1)
u[1:N] = 4*e2[:,0]  # 激励第1模

# 力函数 (向量化版本)
def Ffunc_vectorized(u):
    """
    使用NumPy向量化运算计算力。
    u[0] 和 u[N] 作为固定为0的边界，使得端点的计算公式可以被统一处理。
    """
    NF = np.zeros(N+1)
    
    # 为 i = 1..N-1 创建切片
    u_p1 = u[2:N+1]  # 代表 u_{i+1}
    u_i  = u[1:N]    # 代表 u_{i}
    u_m1 = u[0:N-1]  # 代表 u_{i-1}
    
    # 向量化计算线性和非线性部分
    # u[0]和u[N]为0的边界条件使得端点(i=1和i=N-1)的计算自动符合要求
    linear_part = u_p1 - 2*u_i + u_m1
    nonlinear_part = alpha * ((u_p1 - u_i)**2 - (u_i - u_m1)**2)
    
    NF[1:N] = linear_part + nonlinear_part
    return NF

# 能量计算
def Efunc(u, v, n):
    """计算第n个模态的能量"""
    w = e2[:,n]
    xi1 = np.dot(u[1:N], w)
    xi2 = np.dot(v[1:N], w)
    return 0.5*(xi2**2 + e1[n]*xi1**2)

# --- 记录执行时间并运行 ---
print("Running vectorized simulation...")
start_time = pytime.time()

# 时间积分（Velocity Verlet）
O = Ffunc_vectorized(u) # 初始力
energies = [[] for _ in range(5)]
time = []

for t in range(steps):
    # 记录时间和能量
    time.append(t*delta*4)
    for k in range(5):
        energies[k].append(Efunc(u, v, k))
    
    # Velocity Verlet积分器的内部循环
    for _ in range(4):
        # 更新位移
        u[1:N] += v[1:N]*delta + 0.5*O[1:N]*delta**2
        # 计算新位置下的力
        F = Ffunc_vectorized(u)
        # 更新速度
        v[1:N] += 0.5*(F[1:N] + O[1:N])*delta
        # 为下个子步骤更新力
        O = F

end_time = pytime.time()
print(f"Simulation finished in {end_time - start_time:.4f} seconds.")

# --- 绘图 ---
plt.figure(figsize=(10, 6))
for k in range(5):
    # 使用LaTeX格式化图例标签
    plt.plot(time, energies[k], label=f"$E_{k+1}$")

plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Energy Evolution of the First 5 Modes")
plt.legend()
plt.grid(True)
plt.show()
