import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.optimize import minimize
from matplotlib.patches import Rectangle

# --- 1. 定义物理常数和材料参数 ---
# 物理常数
mu0 = 4 * np.pi * 1e-7  # 真空磁导率 (T*m/A)
g = 9.81  # 重力加速度 (m/s^2)

# 热解石墨片参数
L_pg = 5e-3  # 边长 (m)
t_pg = 0.5e-3  # 厚度 (m)
rho_pg = 2200  # 密度 (kg/m^3)
chi_perp = -4.5e-4  # 垂直磁化率 (SI)
chi_para = -0.5e-4  # 平行磁化率 (SI)
m_pg = rho_pg * L_pg**2 * t_pg # 质量

# 永磁体参数
L_mag = 5e-3  # 边长 (m)
h_mag = 2.5e-3  # 厚度 (m)
Br = 1.3  # 剩磁 (T)

# 磁铁阵列 (4x4)
rows, cols = 4, 4
x_mag_centers = np.linspace(-(cols-1)/2 * L_mag, (cols-1)/2 * L_mag, cols)
y_mag_centers = np.linspace(-(rows-1)/2 * L_mag, (rows-1)/2 * L_mag, rows)

# --- 2. 磁场计算函数 ---
def B_field_single_magnet(x, y, z, xc, yc, zc, Br_eff):
    """计算单个永磁体在(x,y,z)点的磁场"""
    Bx, By, Bz = 0, 0, 0
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                u = x - xc + i * L_mag / 2
                v = y - yc + j * L_mag / 2
                w = z - zc + k * h_mag / 2
                r = np.sqrt(u**2 + v**2 + w**2)
                
                term1 = np.log(r - u) if r - u > 1e-12 else 0
                term2 = np.log(r - v) if r - v > 1e-12 else 0
                term3 = np.arctan2(u * v, w * r) if w * r != 0 else 0

                Bx += i * j * k * (-v * term1)
                By += i * j * k * (-u * term2)
                Bz += i * j * k * (w * term3)

    return (Br_eff / (2 * np.pi)) * np.array([Bx, By, Bz])


def B_field_array(x, y, z):
    """计算整个棋盘阵列在(x,y,z)点的总磁场"""
    B_total = np.zeros(3)
    for i in range(rows):
        for j in range(cols):
            Br_eff = Br * (-1)**(i + j)
            B_total += B_field_single_magnet(x, y, z, x_mag_centers[j], y_mag_centers[i], -h_mag/2, Br_eff)
    return B_total

# --- 3. 势能计算函数 ---
def magnetic_energy_density(x, y, z, theta):
    """计算在(x,y,z)点的磁能密度，考虑石墨片的旋转"""
    # 旋转矩阵
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    
    B = B_field_array(x, y, z)
    B_prime = np.dot(np.linalg.inv(R), B) # 将磁场转换到石墨片的坐标系
    
    Bx_p, By_p, Bz_p = B_prime[0], B_prime[1], B_prime[2]
    
    energy_density = -(1 / (2 * mu0)) * (chi_para * (Bx_p**2 + By_p**2) + chi_perp * Bz_p**2)
    return energy_density

def total_potential_energy(params):
    """计算石墨片的总势能"""
    z, theta = params
    x, y = 0, 0 # 假设石墨片中心在阵列中心上方
    
    # 磁势能 (通过对石墨片面积积分)
    integrand = lambda xp, yp: magnetic_energy_density(xp, yp, z, theta)
    # 由于dplquad对高振荡函数可能不稳定，这里使用简单的数值积分近似
    # 使用一个简单的均值乘以面积来近似，适用于z较大，B场变化相对平缓的情况
    # 更精确的方法需要更复杂的数值积分
    N_points = 5
    x_points = np.linspace(-L_pg/2, L_pg/2, N_points)
    y_points = np.linspace(-L_pg/2, L_pg/2, N_points)
    U_m_val = 0
    for xp in x_points:
        for yp in y_points:
            # 考虑旋转
            x_rot = xp * np.cos(theta) - yp * np.sin(theta)
            y_rot = xp * np.sin(theta) + yp * np.cos(theta)
            U_m_val += magnetic_energy_density(x_rot, y_rot, z, theta)
    U_m = (U_m_val / (N_points**2)) * (L_pg**2 * t_pg)
    
    # 重力势能
    U_g = m_pg * g * z
    
    return U_m + U_g

# --- 4. 寻找稳定悬浮点 ---
# 初始猜测值 [z, theta]
initial_guess = [2.5e-3, np.pi/4]

# 使用优化算法寻找势能最小值
result = minimize(total_potential_energy, initial_guess, bounds=[(1e-3, 10e-3), (-np.pi, np.pi)])
stable_z, stable_theta = result.x
min_energy = result.fun

print(f"稳定悬浮高度 z = {stable_z * 1000:.2f} mm")
print(f"稳定旋转角度 theta = {np.degrees(stable_theta):.2f} 度")
print(f"最低总势能 = {min_energy:.2e} J")


# --- 5. 可视化绘图 ---
# 磁场分布图
grid_res = 100
x_range = np.linspace(-1.5 * cols/2 * L_mag, 1.5 * cols/2 * L_mag, grid_res)
y_range = np.linspace(-1.5 * rows/2 * L_mag, 1.5 * rows/2 * L_mag, grid_res)
X, Y = np.meshgrid(x_range, y_range)
Bz = np.zeros_like(X)

for i in range(grid_res):
    for j in range(grid_res):
        Bz[i, j] = B_field_array(X[i, j], Y[i, j], stable_z)[2]

# 绘制磁场和石墨片位置
fig, ax = plt.subplots(figsize=(8, 8))
# 绘制磁场Bz分量
c = ax.contourf(X * 1000, Y * 1000, Bz, levels=50, cmap='coolwarm')
fig.colorbar(c, ax=ax, label='Magnetic Field Bz (T) at z={:.2f}mm'.format(stable_z*1000))
# 绘制磁铁
for i in range(rows):
    for j in range(cols):
        color = 'red' if (i + j) % 2 == 0 else 'blue'
        ax.add_patch(Rectangle((x_mag_centers[j]*1000 - L_mag*500, y_mag_centers[i]*1000 - L_mag*500),
                               L_mag*1000, L_mag*1000,
                               facecolor=color, alpha=0.8, edgecolor='black'))

# 绘制悬浮的石墨片
pg_corners = np.array([
    [-L_pg/2, -L_pg/2], [L_pg/2, -L_pg/2],
    [L_pg/2, L_pg/2], [-L_pg/2, L_pg/2],
    [-L_pg/2, -L_pg/2]
])
R_theta = np.array([[np.cos(stable_theta), -np.sin(stable_theta)],
                    [np.sin(stable_theta), np.cos(stable_theta)]])
pg_corners_rotated = np.dot(pg_corners, R_theta.T)
ax.plot(pg_corners_rotated[:, 0] * 1000, pg_corners_rotated[:, 1] * 1000, 'g-', linewidth=3, label='Pyrolytic Graphite')
ax.fill(pg_corners_rotated[:, 0] * 1000, pg_corners_rotated[:, 1] * 1000, 'g', alpha=0.4)


ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_title('Magnetic Field Distribution and Stable Position of Pyrolytic Graphite')
ax.set_aspect('equal', adjustable='box')
ax.legend()
plt.grid(True)
plt.show()
