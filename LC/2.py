
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# 参数设置
# -----------------------------
mu0 = 4*np.pi*1e-7  # 真空磁导率 [H/m]

# 永磁体参数
M0 = 1e6           # 磁化强度 [A/m]
a = 0.01           # 磁铁边长 [m]
h = 0.005          # 磁铁厚度 [m]
Nx, Ny = 4, 4      # 棋盘阵列尺寸

# 热解石墨参数
L = 0.008          # 石墨片边长 [m]
t = 0.001          # 厚度 [m]
chi_para = -2e-4   # 平行磁化率
chi_perp = -5e-4   # 垂直磁化率

# -----------------------------
# 磁偶极子磁场函数
# -----------------------------
def B_dipole(r, m):
    x, y, z = r
    r_mag = np.linalg.norm(r)
    if r_mag < 1e-6:
        return np.zeros(3)
    r_hat = r / r_mag
    return mu0/(4*np.pi) * (3*np.dot(m,r_hat)*r_hat - m)/r_mag**3

# -----------------------------
# 棋盘阵磁场叠加
# -----------------------------
def B_total(pos):
    B = np.zeros(3)
    for i in range(Nx):
        for j in range(Ny):
            sign = (-1)**(i+j)
            m = np.array([0,0,sign*M0*a**3])  # 近似立方体体积
            r0 = np.array([i*a, j*a, 0])
            r_vec = pos - r0
            B += B_dipole(r_vec, m)
    return B

# -----------------------------
# 磁势能计算
# -----------------------------
def magnetic_potential(pos, theta=0):
    B = B_total(pos)
    # 旋转矩阵
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta),  np.cos(theta), 0],
                   [0, 0, 1]])
    chi_tensor = np.diag([chi_para, chi_para, chi_perp])
    chi_rot = Rz @ chi_tensor @ Rz.T
    m_ind = (chi_rot @ B)/mu0 * L**2 * t
    U = -0.5*np.dot(m_ind, B)
    return U

# -----------------------------
# 势阱面扫描
# -----------------------------
x = np.linspace(0, Nx*a, 50)
y = np.linspace(0, Ny*a, 50)
z = 0.005  # 固定高度进行水平势阱扫描
theta = 0  # 初始角度

X, Y = np.meshgrid(x, y)
U = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        U[i,j] = magnetic_potential(np.array([X[i,j], Y[i,j], z]), theta)

# -----------------------------
# 绘制三维势阱
# -----------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U, cmap='viridis', alpha=0.8)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('磁势能 U [J]')
ax.set_title('热解石墨水平势阱')

# -----------------------------
# 搜索稳定位置（力矩最小）
# -----------------------------
z_vals = np.linspace(0.001, 0.02, 50)
theta_vals = np.linspace(0, np.pi/2, 20)

stable_positions = []
for z0 in z_vals:
    for theta0 in theta_vals:
        # 数值近似力梯度
        delta = 1e-4
        pos = np.array([Nx*a/2, Ny*a/2, z0])
        U0 = magnetic_potential(pos, theta0)
        # 水平方向梯度
        Fx = (magnetic_potential(pos + np.array([delta,0,0]), theta0) - 
              magnetic_potential(pos - np.array([delta,0,0]), theta0))/(2*delta)
        Fy = (magnetic_potential(pos + np.array([0,delta,0]), theta0) - 
              magnetic_potential(pos - np.array([0,delta,0]), theta0))/(2*delta)
        # 力矩梯度
        dtheta = 1e-4
        tau = (magnetic_potential(pos, theta0+dtheta) - U0)/dtheta
        if np.abs(Fx)<1e-7 and np.abs(Fy)<1e-7 and np.abs(tau)<1e-8:
            stable_positions.append((pos.copy(), theta0))

# 绘制水平投影稳定位置
for pos, theta0 in stable_positions:
    ax.scatter(pos[0], pos[1], magnetic_potential(pos, theta0), color='r', s=50)

plt.show()

print("稳定位置 (中心坐标 & 旋转角):", stable_positions)
