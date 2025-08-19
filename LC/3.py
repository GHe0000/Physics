
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.patches import Rectangle

# -----------------------------
# 1. 常数和参数
# -----------------------------
mu0 = 4*np.pi*1e-7
g = 9.81

# 石墨片
L_pg, t_pg = 5e-3, 0.5e-3
rho_pg = 2200
chi_perp, chi_para = -4.5e-4, -0.5e-4
m_pg = rho_pg * L_pg**2 * t_pg

# 永磁体
L_mag, h_mag = 5e-3, 2.5e-3
Br = 1.3
rows, cols = 4, 4
x_mag_centers = np.linspace(-(cols-1)/2*L_mag, (cols-1)/2*L_mag, cols)
y_mag_centers = np.linspace(-(rows-1)/2*L_mag, (rows-1)/2*L_mag, rows)

# -----------------------------
# 2. 矩形永磁体磁场解析公式
# -----------------------------
def B_rect_magnet(x, y, z, xc, yc, zc, Lx, Ly, Lz, Br):
    """解析计算矩形磁铁在(x,y,z)的磁场"""
    # 转到磁铁中心坐标系
    X = np.array([x - (xc - Lx/2), x - (xc + Lx/2)])
    Y = np.array([y - (yc - Ly/2), y - (yc + Ly/2)])
    Z = np.array([z - (zc - Lz/2), z - (zc + Lz/2)])

    Bx = By = Bz = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                sign = (-1)**(i+j+k)
                xi, yj, zk = X[i], Y[j], Z[k]
                R = np.sqrt(xi**2 + yj**2 + zk**2) + 1e-12  # 避免除零
                Bx += sign * np.arctan2(yj*zk, xi*R)
                By += sign * np.arctan2(xi*zk, yj*R)
                Bz += sign * np.arctan2(xi*yj, zk*R)
    factor = Br / (4*np.pi)
    return np.array([Bx, By, Bz]) * factor

def B_field_array(x, y, z):
    """棋盘阵磁场叠加"""
    B_total = np.zeros(3)
    for i in range(rows):
        for j in range(cols):
            Br_eff = Br * (-1)**(i+j)
            B_total += B_rect_magnet(x, y, z,
                                     x_mag_centers[j], y_mag_centers[i], 0,
                                     L_mag, L_mag, h_mag, Br_eff)
    return B_total

# -----------------------------
# 3. 磁能密度，考虑绕 x,y,z 旋转
# -----------------------------
def magnetic_energy_density(x, y, z, angles):
    """angles: [alpha, beta, gamma] 绕 x,y,z 旋转"""
    alpha, beta, gamma = angles
    # 旋转矩阵：Z-Y-X 欧拉角
    Rx = np.array([[1,0,0],
                   [0,np.cos(alpha),-np.sin(alpha)],
                   [0,np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma),0],
                   [np.sin(gamma),  np.cos(gamma),0],
                   [0,0,1]])
    R = Rz @ Ry @ Rx
    B = B_field_array(x, y, z)
    B_local = R.T @ B  # 转到石墨片局部坐标系
    Bx, By, Bz = B_local
    return -(1/(2*mu0)) * (chi_para*(Bx**2+By**2) + chi_perp*Bz**2)

# -----------------------------
# 4. 总势能
# -----------------------------
def total_potential_energy(params):
    z, alpha, beta, gamma = params
    N_points = 3
    x_points = np.linspace(-L_pg/2, L_pg/2, N_points)
    y_points = np.linspace(-L_pg/2, L_pg/2, N_points)
    U_m_val = 0
    for xp in x_points:
        for yp in y_points:
            U_m_val += magnetic_energy_density(xp, yp, z, [alpha, beta, gamma])
    U_m = (U_m_val / (N_points**2)) * (L_pg**2 * t_pg)
    U_g = m_pg * g * z
    return U_m + U_g

# -----------------------------
# 5. 寻找稳定位置
# -----------------------------
initial_guess = [2.5e-3, 0, 0, 0]  # [z, alpha, beta, gamma]
bounds = [(1e-3, 10e-3), (-np.pi/2,np.pi/2), (-np.pi/2,np.pi/2), (-np.pi,np.pi)]
res = minimize(total_potential_energy, initial_guess, bounds=bounds)
stable_z, alpha, beta, gamma = res.x
print(f"稳定高度 z={stable_z*1000:.2f} mm")
print(f"绕 x,y,z 的旋转角 alpha={np.degrees(alpha):.2f}, beta={np.degrees(beta):.2f}, gamma={np.degrees(gamma):.2f} 度")
print(f"总势能 {res.fun:.2e} J")
