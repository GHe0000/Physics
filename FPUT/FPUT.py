# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class FPUTChain:
    """
    模拟费米-帕斯塔-乌拉姆-青柳 (FPUT-β) 链的类。
    
    这个模型描述了一维链上通过非线性弹簧相互连接的一系列粒子。
    固定边界条件 (q_0 = q_{N+1} = 0)。
    
    运动方程为:
    m * d^2(q_i)/dt^2 = k * (q_{i+1} - 2*q_i + q_{i-1}) + β * [(q_{i+1} - q_i)^3 - (q_i - q_{i-1})^3]
    
    为了简化，我们通常设 m=1, k=1。
    """

    def __init__(self, N, beta, dt):
        """
        初始化 FPUT 链。
        
        参数:
        N (int): 内部可移动粒子的数量。
        beta (float): 非线性相互作用的强度系数。
        dt (float): 模拟的时间步长。
        """
        self.N = N  # 粒子数
        self.beta = beta  # 非线性系数
        self.dt = dt  # 时间步长
        
        # 粒子位移 q 和动量 p (速度, 因为 m=1)。
        # 数组长度为 N+2 以包含固定的边界点 q_0 和 q_{N+1}。
        self.q = np.zeros(N + 2, dtype=float)
        self.p = np.zeros(N + 2, dtype=float)
        
        # 预先计算正规模变换矩阵，用于分析能量分布。
        # 正规模坐标 Q_k = sqrt(2/(N+1)) * sum_{j=1 to N} q_j * sin(k*j*pi/(N+1))
        self.mode_matrix = np.zeros((N, N))
        for k in range(1, N + 1):
            for j in range(1, N + 1):
                self.mode_matrix[k-1, j-1] = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * k * j / (N + 1))

    def set_initial_conditions(self, mode_k, amplitude):
        """
        设置初始条件，将所有能量集中在单个正规模上。
        
        参数:
        mode_k (int): 要激发的初始模式编号 (1, 2, ..., N)。
        amplitude (float): 初始模式的振幅。
        """
        self.q = np.zeros(self.N + 2, dtype=float)
        self.p = np.zeros(self.N + 2, dtype=float)
        
        # 根据正规模的定义设置初始位移
        for i in range(1, self.N + 1):
            self.q[i] = amplitude * np.sin(np.pi * mode_k * i / (self.N + 1))
            
    def calculate_acceleration(self):
        """
        根据当前位移计算每个粒子的加速度。
        """
        # 使用 np.roll 高效计算相邻位移差
        q_plus_1 = np.roll(self.q, -1)
        q_minus_1 = np.roll(self.q, 1)
        
        # 线性力部分 (k=1)
        linear_force = q_plus_1 - 2 * self.q + q_minus_1
        
        # 非线性力部分
        term1 = q_plus_1 - self.q
        term2 = self.q - q_minus_1
        nonlinear_force = self.beta * (term1**3 - term2**3)
        
        # 加速度 a = F/m (m=1)
        accel = linear_force + nonlinear_force
        
        # 保持边界固定
        accel[0] = accel[-1] = 0.0
        return accel

    def time_step(self):
        """
        使用 Velocity-Verlet 算法演化系统一个时间步长。
        这是一个稳定且保持能量守恒性良好的数值积分方法。
        """
        # 1. 计算当前加速度
        accel_current = self.calculate_acceleration()
        
        # 2. 更新位移: q(t+dt) = q(t) + p(t)*dt + 0.5*a(t)*dt^2
        self.q += self.p * self.dt + 0.5 * accel_current * self.dt**2
        
        # 3. 计算新位置下的加速度
        accel_next = self.calculate_acceleration()
        
        # 4. 更新速度: p(t+dt) = p(t) + 0.5 * (a(t) + a(t+dt)) * dt
        self.p += 0.5 * (accel_current + accel_next) * self.dt

    def get_mode_energies(self):
        """
        计算每个正规模的能量。
        对于谐振子部分，能量 E_k = 0.5 * (P_k^2 + omega_k^2 * Q_k^2)
        """
        # 变换到正规模坐标
        # q_internal 和 p_internal 只包含可移动的粒子
        q_internal = self.q[1:-1]
        p_internal = self.p[1:-1]
        
        Q_k = self.mode_matrix @ q_internal
        P_k = self.mode_matrix @ p_internal
        
        # 计算每个模式的角频率 omega_k
        k_vals = np.arange(1, self.N + 1)
        omega_k_sq = (2 * np.sin(np.pi * k_vals / (2 * (self.N + 1))))**2 * 2 # 乘以2使其与文献一致
        
        # 计算每个模式的能量（仅谐波部分）
        mode_energies = 0.5 * (P_k**2 + omega_k_sq * Q_k**2)
        return mode_energies

    def get_total_energy(self):
        """计算系统的总能量（动能 + 势能）"""
        # 动能
        kinetic_energy = 0.5 * np.sum(self.p[1:-1]**2)
        
        # 势能
        stretches = self.q[1:] - self.q[:-1]
        potential_energy = 0.5 * np.sum(stretches**2) + (self.beta / 4.0) * np.sum(stretches**4)
        
        return kinetic_energy + potential_energy

def run_simulation(N, beta, dt, sim_time, init_mode, init_amplitude, num_modes_to_plot):
    """封装的模拟函数，用于运行一次完整的模拟并返回结果"""
    chain = FPUTChain(N=N, beta=beta, dt=dt)
    chain.set_initial_conditions(mode_k=init_mode, amplitude=init_amplitude)
    
    n_steps = int(sim_time / dt)
    time_points = np.arange(n_steps) * dt
    
    mode_energy_history = np.zeros((n_steps, num_modes_to_plot))
    total_energy_history = np.zeros(n_steps)

    print(f"开始模拟 (β = {beta})...")
    for i in range(n_steps):
        chain.time_step()
        mode_energies = chain.get_mode_energies()
        mode_energy_history[i, :] = mode_energies[:num_modes_to_plot]
        total_energy_history[i] = chain.get_total_energy()
        
        if (i+1) % (n_steps // 10) == 0:
            print(f"进度 (β = {beta}): {100 * (i+1) / n_steps:.0f}%")
            
    return time_points, mode_energy_history, total_energy_history

def main():
    """主函数：设置、运行两次模拟并进行对比可视化"""
    # --- 通用模拟参数 ---
    N = 31              # 粒子数
    DT = 0.1            # 时间步长
    SIM_TIME = 10000    # 总模拟时间
    INIT_MODE = 1       # 初始激发的模式
    INIT_AMPLITUDE = 1.0 # 初始振幅
    NUM_MODES_TO_PLOT = 5 # 绘制能量的模式数量

    # --- 参数对比 ---
    BETA_LOW = 0.1      # 低非线性强度 (非热化)
    BETA_HIGH = 1.0     # 高非线性强度 (热化)

    # --- 运行两次模拟 ---
    time_low, modes_low, total_e_low = run_simulation(
        N, BETA_LOW, DT, SIM_TIME, INIT_MODE, INIT_AMPLITUDE, NUM_MODES_TO_PLOT)
    
    time_high, modes_high, total_e_high = run_simulation(
        N, BETA_HIGH, DT, SIM_TIME, INIT_MODE, INIT_AMPLITUDE, NUM_MODES_TO_PLOT)

    print("模拟完成。正在绘制对比结果...")
    
    # --- 绘图 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    
    # --- 左侧图: 非热化 (低 β) ---
    ax1_low, ax2_low = axes[:, 0]
    
    # 图1.1: 各模式能量
    for k in range(NUM_MODES_TO_PLOT):
        ax1_low.plot(time_low, modes_low[:, k], label=f'模式 {k+1}')
    ax1_low.set_title(f'非热化: 能量演化 (β={BETA_LOW})')
    ax1_low.set_ylabel('模式能量')
    ax1_low.legend()
    ax1_low.grid(True, linestyle='--', alpha=0.6)
    
    # 图1.2: 总能量
    initial_total_energy_low = total_e_low[0]
    energy_fluctuation_low = (total_e_low - initial_total_energy_low) / initial_total_energy_low
    ax2_low.plot(time_low, energy_fluctuation_low)
    ax2_low.set_title('总能量相对变化')
    ax2_low.set_xlabel('时间')
    ax2_low.set_ylabel('(E(t) - E(0)) / E(0)')
    ax2_low.grid(True, linestyle='--', alpha=0.6)
    ax2_low.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # --- 右侧图: 热化 (高 β) ---
    ax1_high, ax2_high = axes[:, 1]

    # 图2.1: 各模式能量
    for k in range(NUM_MODES_TO_PLOT):
        ax1_high.plot(time_high, modes_high[:, k], label=f'模式 {k+1}')
    ax1_high.set_title(f'热化: 能量演化 (β={BETA_HIGH})')
    ax1_high.legend()
    ax1_high.grid(True, linestyle='--', alpha=0.6)

    # 图2.2: 总能量
    initial_total_energy_high = total_e_high[0]
    energy_fluctuation_high = (total_e_high - initial_total_energy_high) / initial_total_energy_high
    ax2_high.plot(time_high, energy_fluctuation_high)
    ax2_high.set_title('总能量相对变化')
    ax2_high.set_xlabel('时间')
    ax2_high.grid(True, linestyle='--', alpha=0.6)
    ax2_high.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout()
    # 使用中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
    plt.show()

if __name__ == '__main__':
    main()
