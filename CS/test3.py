import numpy as np
import matplotlib.pyplot as plt
import numba

# --- 模拟参数 ---
N_PARTICLES = 2000         # 粒子数量 (建议范围: 500-2000)
BOX_SIZE = 1.0             # 模拟盒子的边长 (单位: Mpc)
G = 6.674e-11 * (1.989e30 / (3.086e22)**3) # 引力常数 (单位: Mpc^3 / (M_sun * s^2))
                           # 为了便于模拟，我们通常会使用简化的 G=1
G_sim = 1.0

SOFTENING = 0.01           # 引力软化参数，避免奇异点
TIME_STEP = 0.01           # 时间步长
N_STEPS = 500              # 总模拟步数
PLOT_EVERY_N_STEPS = 5     # 每隔多少步更新一次图像

# --- Numba 加速的核心计算函数 ---

@numba.jit(nopython=True)
def apply_periodic_boundaries(positions, box_size):
    """施加周期性边界条件"""
    return positions % box_size

@numba.jit(nopython=True)
def get_acceleration(positions, masses, G, softening, box_size):
    """
    计算所有粒子的加速度 (N^2 直接求和)
    使用了最小镜像约定 (Minimum Image Convention)
    """
    n = positions.shape[0]
    accel = np.zeros_like(positions)

    for i in range(n):
        for j in range(i + 1, n):
            # 计算粒子 j 相对于粒子 i 的位移矢量
            delta_pos = positions[j] - positions[i]

            # 最小镜像约定：处理周期性边界
            for dim in range(delta_pos.shape[0]):
                if delta_pos[dim] > box_size / 2:
                    delta_pos[dim] -= box_size
                elif delta_pos[dim] < -box_size / 2:
                    delta_pos[dim] += box_size

            # 计算距离的平方 + 软化
            dist_sq = np.sum(delta_pos**2) + softening**2
            inv_r3 = dist_sq**(-1.5)

            # 计算引力加速度
            force_magnitude = G * inv_r3
            accel_i = force_magnitude * delta_pos * masses[j]
            accel_j = -force_magnitude * delta_pos * masses[i]

            accel[i] += accel_i
            accel[j] += accel_j

    return accel


@numba.jit(nopython=True)
def leapfrog_kick(velocities, accelerations, dt):
    """Leapfrog积分的第一步和第三步：Kick (速度更新)"""
    return velocities + accelerations * dt

@numba.jit(nopython=True)
def leapfrog_drift(positions, velocities, dt, box_size):
    """Leapfrog积分的第二步：Drift (位置更新)"""
    new_positions = positions + velocities * dt
    # 在漂移后应用周期性边界
    return apply_periodic_boundaries(new_positions, box_size)


# --- 主程序 ---
def main():
    """
    模拟主函数
    """
    # 1. 初始化粒子
    np.random.seed(42) # 为了结果可复现
    
    # 初始位置：在盒内均匀随机分布
    positions = np.random.rand(N_PARTICLES, 2) * BOX_SIZE
    
    # 初始速度：设为0 (宇宙初始时速度扰动很小)
    velocities = np.zeros_like(positions)
    
    # 粒子质量：假设所有粒子质量相同
    masses = np.ones(N_PARTICLES) * 1.0 # 简化单位

    # 计算初始加速度
    accelerations = get_acceleration(positions, masses, G_sim, SOFTENING, BOX_SIZE)

    # 2. 设置 Matplotlib 可视化
    plt.ion() # 开启交互模式
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 3. 模拟主循环
    for step in range(N_STEPS):
        # 使用Leapfrog (KDK - Kick-Drift-Kick) 积分法
        # Kick (半步)
        velocities = leapfrog_kick(velocities, accelerations, TIME_STEP / 2.0)
        
        # Drift (整步)
        positions = leapfrog_drift(positions, velocities, TIME_STEP, BOX_SIZE)
        
        # 计算新位置下的加速度
        accelerations = get_acceleration(positions, masses, G_sim, SOFTENING, BOX_SIZE)
        
        # Kick (另外半步)
        velocities = leapfrog_kick(velocities, accelerations, TIME_STEP / 2.0)

        # 4. 可视化
        if step % PLOT_EVERY_N_STEPS == 0:
            ax.clear()
            ax.scatter(positions[:, 0], positions[:, 1], s=2, c='white', alpha=0.8)
            ax.set_title(f'2D Cosmological Simulation | Step: {step}/{N_STEPS}')
            ax.set_xlabel('X (Mpc)')
            ax.set_ylabel('Y (Mpc)')
            ax.set_xlim(0, BOX_SIZE)
            ax.set_ylim(0, BOX_SIZE)
            ax.set_aspect('equal', adjustable='box')
            ax.set_facecolor('black')
            fig.canvas.draw()
            plt.pause(0.001)

    # 模拟结束后保持窗口显示
    plt.ioff()
    ax.set_title(f'Final State | Step: {N_STEPS}')
    plt.show()

if __name__ == '__main__':
    main()
