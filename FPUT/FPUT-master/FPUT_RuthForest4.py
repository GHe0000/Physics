# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class FPUTChain:
    """
    一个使用 NumPy 重构和优化的 FPUT 链模拟器。
    
    该类模拟了一个具有线性和非线性（α 和 β 型）相互作用的振子链。
    它采用了 4 阶 Forest-Ruth 辛积分器，以确保长期的能量稳定性和准确性，
    这与用户提供的原始代码中的积分方法思想一致。
    
    运动方程:
    d^2(q_i)/dt^2 = (q_{i+1} - 2q_i + q_{i-1}) 
                    + α * [(q_{i+1} - q_i)^2 - (q_i - q_{i-1})^2] 
                    + β * [(q_{i+1} - q_i)^3 - (q_i - q_{i-1})^3]
    """

    def __init__(self, N, alpha, beta, dt):
        """
        初始化 FPUT 链。
        
        参数:
        N (int): 内部可移动粒子的数量。
        alpha (float): FPU-α (三次势) 非线性强度系数。
        beta (float): FPU-β (四次势) 非线性强度系数。
        dt (float): 模拟的时间步长。
        """
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        
        # 粒子位移 q 和动量 p。数组长度为 N+2 以包含固定的边界点。
        self.q = np.zeros(N + 2, dtype=float)
        self.p = np.zeros(N + 2, dtype=float)
        
        # 预计算正规模变换矩阵，用于能量分析。
        self.mode_matrix = np.zeros((N, N))
        for k in range(1, N + 1):
            for j in range(1, N + 1):
                self.mode_matrix[k-1, j-1] = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * k * j / (N + 1))
        
        # 4阶辛积分器的系数 (Forest-Ruth/Yoshida)
        x0 = -2**(1/3) / (2 - 2**(1/3))
        x1 = 1 / (2 - 2**(1/3))
        self.c_coeffs = np.array([x1/2, (x0+x1)/2, (x0+x1)/2, x1/2])
        self.d_coeffs = np.array([x1, x0, x1, 0])

    def set_initial_conditions(self, mode_k, amplitude):
        """
        设置初始条件，将所有能量集中在单个正规模上。
        """
        self.q.fill(0)
        self.p.fill(0)
        
        # 内部振子的索引
        indices = np.arange(1, self.N + 1)
        # 根据正规模定义设置初始位移
        self.q[1:-1] = amplitude * np.sin(np.pi * mode_k * indices / (self.N + 1))
            
    def calculate_acceleration(self):
        """
        根据当前位移矢量化地计算每个粒子的加速度。
        """
        # 使用 np.roll 高效计算相邻位移差
        q_plus_1 = np.roll(self.q, -1)
        q_minus_1 = np.roll(self.q, 1)
        
        stretch_right = q_plus_1 - self.q
        stretch_left = self.q - q_minus_1
        
        # 线性力 (对应二次势, a=1)
        linear_force = stretch_right - stretch_left
        
        # 非线性力
        alpha_force = self.alpha * (stretch_right**2 - stretch_left**2)
        beta_force = self.beta * (stretch_right**3 - stretch_left**3)
        
        accel = linear_force + alpha_force + beta_force
        
        # 保持边界固定
        accel[0] = accel[-1] = 0.0
        return accel

    def time_step(self):
        """
        使用 4 阶 Forest-Ruth 辛积分算法演化系统一个时间步长。
        """
        # 这是一个4步积分过程
        for i in range(4):
            self.q += self.c_coeffs[i] * self.p * self.dt
            accel = self.calculate_acceleration()
            self.p += self.d_coeffs[i] * accel * self.dt

    def get_mode_energies(self):
        """
        计算每个正规模的能量（谐波近似）。
        """
        q_internal = self.q[1:-1]
        p_internal = self.p[1:-1]
        
        Q_k = self.mode_matrix @ q_internal
        P_k = self.mode_matrix @ p_internal
        
        k_vals = np.arange(1, self.N + 1)
        omega_k_sq = (2 * np.sin(np.pi * k_vals / (2 * (self.N + 1))))**2 * 2
        
        mode_energies = 0.5 * (P_k**2 + omega_k_sq * Q_k**2)
        return mode_energies

    def get_total_energy(self):
        """计算系统的总能量（动能 + 势能）。"""
        kinetic_energy = 0.5 * np.sum(self.p[1:-1]**2)
        
        stretches = self.q[1:] - self.q[:-1]
        linear_potential = 0.5 * np.sum(stretches**2)
        alpha_potential = (self.alpha / 3.0) * np.sum(stretches**3)
        beta_potential = (self.beta / 4.0) * np.sum(stretches**4)
        
        potential_energy = linear_potential + alpha_potential + beta_potential
        return kinetic_energy + potential_energy

def main():
    """主函数：设置、运行模拟并可视化结果。"""
    # --- 模拟参数 (源于用户提供的代码) ---
    # 用户代码 N=71 似乎包括了两个固定的端点。
    # 我们的 N 定义为可移动的粒子数。
    N_OSCILLATORS = 69 
    ALPHA = 0.0          # 对应用户代码中的 'b'
    BETA = 1000.0        # 对应用户代码中的 'c' (这是一个非常强的非线性)
    DT = 0.125           # 时间步长
    
    # 用户代码的 Niter=433482, 总时间为 54185.25, 运行时间会非常长。
    # 这里缩短模拟时间以便快速查看结果。
    SIM_TIME = 5000
    
    INIT_MODE = 1        # 初始激发的模式 (sin(pi*i/(N-1)) 对应最低阶模式)
    INIT_AMPLITUDE = 1.0 # 初始振幅
    NUM_MODES_TO_PLOT = 10 # 要追踪和绘制的模式数量

    # --- 设置和运行模拟 ---
    chain = FPUTChain(N=N_OSCILLATORS, alpha=ALPHA, beta=BETA, dt=DT)
    chain.set_initial_conditions(mode_k=INIT_MODE, amplitude=INIT_AMPLITUDE)
    
    n_steps = int(SIM_TIME / DT)
    time_points = np.arange(n_steps) * DT
    
    mode_energy_history = np.zeros((n_steps, NUM_MODES_TO_PLOT))
    total_energy_history = np.zeros(n_steps)
    
    print("开始模拟...")
    initial_energy = chain.get_total_energy()
    print(f"初始总能量: {initial_energy:.4f}")

    for i in range(n_steps):
        chain.time_step()
        if i % (n_steps // 100) == 0: # 每 1% 更新一次能量记录
            mode_energies = chain.get_mode_energies()
            mode_energy_history[i, :] = mode_energies[:NUM_MODES_TO_PLOT]
            total_energy_history[i] = chain.get_total_energy()
        
        if (i+1) % (n_steps // 10) == 0:
            print(f"进度: {100 * (i+1) / n_steps:.0f}%")
    
    final_energy = chain.get_total_energy()
    print(f"模拟完成。最终总能量: {final_energy:.4f}")
    print(f"能量变化: {(final_energy - initial_energy) / initial_energy * 100:.6f}%")

    # --- 绘图 (模仿用户代码的输出) ---
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 图1: 最终时刻链的位移形状
    pos_indices = np.arange(N_OSCILLATORS)
    ax1.plot(pos_indices, chain.q[1:-1], '-o', markersize=4)
    ax1.set_title(f'最终时刻 (T={SIM_TIME}) 链的位移形状')
    ax1.set_xlabel('粒子索引')
    ax1.set_ylabel('位移')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 图2: 各个模式的能量随时间的变化
    # 过滤掉未记录的零点
    recorded_steps = time_points[::n_steps // 100]
    recorded_energies = mode_energy_history[::n_steps // 100]
    
    for k in range(NUM_MODES_TO_PLOT):
        ax2.plot(recorded_steps, recorded_energies[:, k], label=f'模式 {k+1}')
    ax2.set_title(f'前 {NUM_MODES_TO_PLOT} 个模式的能量演化 (β={BETA})')
    ax2.set_xlabel('时间')
    ax2.set_ylabel('模式能量 (谐波近似)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
