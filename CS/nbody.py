# coding: utf-8
"""
这是一个精简的单文件版本，用于二维宇宙学 N-body 模拟。
它整合了初始条件生成、粒子-网格模拟和最终可视化。

工作流程:
1.  **InitialConditions**: 使用傅里叶方法生成一个高斯随机场作为初始引力势。
2.  **Zeldovich**: 使用泽尔多维奇近似，根据引力势设置粒子的初始位置和速度。
3.  **Simulation**:
    - 使用粒子-网格 (Particle-Mesh) 方法进行演化。
    - 使用 Cloud-in-Cell (CIC) 方法将粒子质量分配到网格。
    - 在傅里叶空间中求解泊松方程得到引力。
    - 使用蛙跳积分器 (Leap-Frog) 更新粒子的状态。
    - 将不同时间点的数据保存到磁盘。
4.  **Visualization**:
    - 读取保存的数据。
    - 通过计算初始网格单元的面积变化来可视化密度场。
    - 使用 Matplotlib 生成最终图像。
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass
import numpy as np
from numpy import fft
from scipy.integrate import quad
import numba
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, Tuple
from functools import partial, reduce
import operator
from matplotlib import pyplot as plt

# --- 模块: 初始条件 (源于 cft.py) ---

def _wave_number(s: Tuple[int, ...]):
    N = s[0]
    i = np.indices(s)
    return np.where(i > N / 2, i - N, i)

def _k_power(k, n):
    """安全地计算 |k| 的 n 次幂，避免 k=0 时的除零错误。"""
    with np.errstate(divide='ignore'):
        return np.where(k == 0, 0, k**n)

@dataclass
class Box:
    """定义模拟空间"""
    dim: int
    N: int
    L: float

    def __post_init__(self):
        self.res = self.L / self.N
        self.shape = (self.N,) * self.dim
        self.size = self.N ** self.dim
        self.K = _wave_number(self.shape) * 2 * np.pi / self.L
        self.k_squared = (self.K**2).sum(axis=0)
        self.k_max = self.N * np.pi / self.L

class Filter(ABC):
    """在傅里叶空间中操作的滤波器基类"""
    def __init__(self, f_k: Callable):
        self._f_k = f_k
    def __call__(self, K):
        return self._f_k(K)
    def __mul__(self, other):
        return Filter(lambda K: self(K) * other(K))

class PowerLaw(Filter):
    """幂律功率谱 P(k) = k^n"""
    def __init__(self, n: float):
        super().__init__(lambda K: _k_power((K**2).sum(axis=0), n / 2.))

class Scale(Filter):
    """高斯平滑滤波器"""
    def __init__(self, B: Box, sigma: float):
        t = sigma**2
        def f(K):
            return reduce(
                operator.mul,
                (np.exp(t / B.res**2 * (np.cos(k * B.res) - 1)) for k in K)
            )
        super().__init__(f)

class Cutoff(Filter):
    """硬截断滤波器，去除超出奈奎斯特频率的模式"""
    def __init__(self, B: Box):
        super().__init__(lambda K: np.where((K**2).sum(axis=0) <= B.k_max**2, 1, 0))

class Potential(Filter):
    """从密度到势的变换（求解泊松方程）"""
    def __init__(self):
        super().__init__(lambda K: -_k_power((K**2).sum(axis=0), -1.))

def generate_gaussian_field(B: Box, P: Filter, T: Filter = Filter(lambda K: 1), seed: int = None):
    """生成一个应用了变换 T 的高斯随机场"""
    if seed is not None:
        np.random.seed(seed)
    # 在傅里叶空间生成白噪声
    wn_f = fft.fftn(np.random.normal(0, 1, B.shape))
    # 应用功率谱和变换
    field_f = wn_f * np.sqrt(P(B.K)) * T(B.K)
    # 返回到实空间
    return fft.ifftn(field_f).real

# --- 模块: N-body 物理和积分器 (源于 nbody.py) ---

@dataclass
class Cosmology:
    """定义宇宙学模型"""
    H0: float
    OmegaM: float
    OmegaL: float

    @property
    def OmegaK(self):
        return 1 - self.OmegaM - self.OmegaL

    @property
    def G(self):
        return 3. / 2 * self.OmegaM * self.H0**2

    def da(self, a):
        """ a * H(a) """
        return self.H0 * a * np.sqrt(
            self.OmegaL + self.OmegaM * a**-3 + self.OmegaK * a**-2
        )

# 定义一个爱因斯坦-德西特 (EdS) 宇宙模型
EdS = Cosmology(70.0, 1.0, 0.0)

@numba.jit(nopython=True)
def mass_deposition_cic_2d(pos: np.ndarray, shape: Tuple[int, int], tgt: np.ndarray):
    """使用 Numba 加速的 2D CIC 质量分配"""
    for i in range(len(pos)):
        idx0, idx1 = int(np.floor(pos[i, 0])), int(np.floor(pos[i, 1]))
        f0, f1 = pos[i, 0] - idx0, pos[i, 1] - idx1
        w00 = (1 - f0) * (1 - f1)
        w10 = f0 * (1 - f1)
        w01 = (1 - f0) * f1
        w11 = f0 * f1
        tgt[idx0 % shape[0], idx1 % shape[1]] += w00
        tgt[(idx0 + 1) % shape[0], idx1 % shape[1]] += w10
        tgt[idx0 % shape[0], (idx1 + 1) % shape[1]] += w01
        tgt[(idx0 + 1) % shape[0], (idx1 + 1) % shape[1]] += w11

class Interpolator2D:
    """双线性插值器，用于将网格上的力插值到粒子位置"""
    def __init__(self, grid_data):
        self.data = grid_data
        self.shape = grid_data.shape

    def __call__(self, x):
        X1 = np.floor(x).astype(int)
        X2 = (X1 + 1)
        xm = x % 1.0
        xn = 1.0 - xm
        
        # 周期性边界条件
        X1 %= np.array(self.shape)
        X2 %= np.array(self.shape)

        f1 = self.data[X1[:, 0], X1[:, 1]]
        f2 = self.data[X2[:, 0], X1[:, 1]]
        f3 = self.data[X1[:, 0], X2[:, 1]]
        f4 = self.data[X2[:, 0], X2[:, 1]]

        return (f1 * xn[:, 0] * xn[:, 1] +
                f2 * xm[:, 0] * xn[:, 1] +
                f3 * xn[:, 0] * xm[:, 1] +
                f4 * xm[:, 0] * xm[:, 1])

def gradient_2nd_order(F, i, res):
    """二阶中心差分计算梯度"""
    return (np.roll(F, -1, axis=i) - np.roll(F, 1, axis=i)) / (2 * res)

Vector = TypeVar("Vector", bound=np.ndarray)

@dataclass
class State(Generic[Vector]):
    """封装模拟状态"""
    time: float
    position: Vector
    momentum: Vector

class HamiltonianSystem(ABC, Generic[Vector]):
    """哈密顿系统抽象基类"""
    @abstractmethod
    def positionEquation(self, s: State[Vector]) -> Vector: raise NotImplementedError
    @abstractmethod
    def momentumEquation(self, s: State[Vector]) -> Vector: raise NotImplementedError

class PoissonVlasov(HamiltonianSystem[np.ndarray]):
    """泊松-弗拉索夫系统，描述了 N-body 模拟的物理"""
    def __init__(self, box, cosmology, particle_mass):
        self.box = box
        self.cosmology = cosmology
        self.particle_mass = particle_mass
        self.delta = np.zeros(self.box.shape, dtype='f8')
        self.potential_kernel = Potential()(self.box.K)

    def positionEquation(self, s: State[np.ndarray]) -> np.ndarray:
        a = s.time
        da = self.cosmology.da(a)
        # 物理速度 v = p / (a^2 * da/dt)
        return s.momentum / (a**2 * da)

    def momentumEquation(self, s: State[np.ndarray]) -> np.ndarray:
        a = s.time
        da = self.cosmology.da(a)
        
        # 1. 质量分配 (CIC)
        x_grid = s.position / self.box.res
        self.delta.fill(0.0)
        mass_deposition_cic_2d(x_grid, self.box.shape, self.delta)
        
        # 2. 计算密度扰动 delta
        self.delta *= self.particle_mass
        self.delta -= 1.0
        
        # 3. 求解泊松方程
        delta_f = fft.fftn(self.delta)
        phi_f = delta_f * self.potential_kernel
        phi = fft.ifftn(phi_f).real
        
        # 4. 计算引力场 (g = -nabla(phi))
        acc_x = -gradient_2nd_order(phi, 0, self.box.res)
        acc_y = -gradient_2nd_order(phi, 1, self.box.res)
        
        # 5. 插值力到粒子
        interp_ax = Interpolator2D(acc_x)
        interp_ay = Interpolator2D(acc_y)
        acc = np.c_[interp_ax(x_grid), interp_ay(x_grid)]
        
        # 动量方程 dp/dt = - (G/a) * acc / da
        return -acc * self.cosmology.G / a / da

class Initializer:
    """使用泽尔多维奇近似初始化粒子"""
    def __init__(self, B_mass: Box, B_force: Box, cosmology: Cosmology, phi: np.ndarray):
        self.bm = B_mass
        self.bf = B_force
        self.cosmology = cosmology
        
        # 计算初始位移场 u = -nabla(phi)
        self.u_x = -gradient_2nd_order(phi, 0, self.bm.res)
        self.u_y = -gradient_2nd_order(phi, 1, self.bm.res)
        
        # 粒子网格的初始坐标
        self.grid_coords = np.indices(self.bm.shape).transpose(1, 2, 0).reshape(-1, 2) * self.bm.res

    def get_initial_state(self, a_init: float) -> State[np.ndarray]:
        # 从网格插值位移场到粒子位置
        interp_ux = Interpolator2D(self.u_x)
        interp_uy = Interpolator2D(self.u_y)
        u = np.c_[interp_ux(self.grid_coords), interp_uy(self.grid_coords)]
        
        # Zeldovich Approx: x = q + D(a)*u, p = a^2 * dD/dt * u
        # For EdS, D(a) = a, so p = a * u
        X = self.grid_coords + a_init * u
        P = a_init * u
        return State(time=a_init, position=X, momentum=P)

    @property
    def particle_mass(self):
        """每个模拟粒子代表的质量（相对于平均密度）"""
        return (self.bf.N / self.bm.N)**self.bm.dim

# --- 模块: 模拟循环和数据保存 ---

def leap_frog_step(dt: float, h: HamiltonianSystem[Vector], s: State[Vector]) -> State[Vector]:
    """执行一步蛙跳积分"""
    s.momentum += (dt / 2.0) * h.momentumEquation(s)
    s.position += dt * h.positionEquation(s)
    s.momentum += (dt / 2.0) * h.momentumEquation(s)
    s.time += dt
    return s

def run_simulation(stepper, halt_condition, init_state, save_times, data_dir):
    """运行整个模拟循环"""
    state = init_state
    save_counter = 0

    while not halt_condition(state):
        state = stepper(state)
        
        # 检查是否需要保存
        if save_counter < len(save_times) and state.time >= save_times[save_counter]:
            print(f"Saving state at time a = {state.time:.3f}")
            fn = os.path.join(data_dir, f'x.{int(round(state.time * 1000)):05d}.npy')
            with open(fn, 'wb') as f:
                np.save(f, state.position)
                np.save(f, state.momentum)
            save_counter += 1
    return state

# --- 模块: 可视化 (源于 phase_plot.py) ---

def get_box_triangles(box: Box):
    """为初始的方形网格创建三角剖分"""
    idx = np.arange(box.size, dtype=int).reshape(box.shape)
    x0 = idx[:-1, :-1].flatten()
    x1 = idx[:-1, 1:].flatten()
    x2 = idx[1:, :-1].flatten()
    x3 = idx[1:, 1:].flatten()
    upper = np.c_[x0, x1, x2]
    lower = np.c_[x3, x2, x1]
    return np.r_[upper, lower]

def get_triangle_area(x: np.ndarray, y: np.ndarray, t: np.ndarray):
    """计算所有三角形的（有向）面积"""
    return 0.5 * (x[t[:, 0]] * (y[t[:, 1]] - y[t[:, 2]]) +
                  x[t[:, 1]] * (y[t[:, 2]] - y[t[:, 0]]) +
                  x[t[:, 2]] * (y[t[:, 0]] - y[t[:, 1]]))

def plot_simulation_snapshots(mass_box, triangles, plot_times, data_dir, output_filename):
    """加载数据并生成多面板图像"""
    cols = len(plot_times)
    fig, axs = plt.subplots(1, cols, figsize=(4 * cols, 4.5), constrained_layout=True)
    if cols == 1: axs = [axs]

    for i, t in enumerate(plot_times):
        fn = os.path.join(data_dir, f'x.{int(round(t * 1000)):05d}.npy')
        with open(fn, "rb") as f:
            pos = np.load(f)

        # 密度正比于三角形面积的倒数
        area = np.abs(get_triangle_area(pos[:, 0], pos[:, 1], triangles))
        # 避免除零
        density = mass_box.res**2 / (area + 1e-12)
        
        # 为了更好的可视化，对三角形按面积排序
        sorting = np.argsort(area)[::-1]

        ax = axs[i]
        # 使用 tripcolor 进行绘图
        tripcolor = ax.tripcolor(
            pos[:, 0], pos[:, 1], triangles[sorting],
            facecolors=np.log10(density[sorting]),
            cmap='magma', vmin=-1, vmax=2.5
        )
        ax.set_title(f"a = {t}")
        ax.set_aspect('equal')
        ax.set_xlim(0, mass_box.L)
        ax.set_ylim(0, mass_box.L)
        ax.set_xlabel("x [Mpc/h]")
        if i == 0:
            ax.set_ylabel("y [Mpc/h]")

    cbar = fig.colorbar(tripcolor, ax=axs, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label(r'$\log_{10}(\rho/\bar{\rho})$')
    
    fig.suptitle("Evolution of Cosmological Structures")
    fig.savefig(output_filename, dpi=150)
    print(f"Visualization saved to {output_filename}")


# --- 主函数 ---

def main():
    # 模拟参数
    N_MASS = 256      # 质量（粒子）网格分辨率
    N_FORCE = 512     # 力计算网格分辨率
    BOX_L = 50.0      # 盒子大小 (Mpc/h)
    A_INIT = 0.02     # 初始尺度因子
    A_FINAL = 2.0     # 最终尺度因子
    DT = 0.02         # 时间步长
    SEED = 42         # 随机数种子
    
    DATA_DIR = "data"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. 定义模拟盒子和宇宙学
    mass_box = Box(dim=2, N=N_MASS, L=BOX_L)
    force_box = Box(dim=2, N=N_FORCE, L=BOX_L)
    cosmology = EdS

    # 2. 生成初始条件
    print("Generating initial conditions...")
    # P(k) ~ k^-2 (对于2D，这类似于3D中的n=-1)
    power_spectrum = PowerLaw(-2.0) * Scale(mass_box, 0.5) * Cutoff(mass_box)
    # 乘以一个振幅因子
    A = 10 
    phi = generate_gaussian_field(mass_box, power_spectrum, Potential(), SEED) * A

    # 3. 设置粒子和物理系统
    initializer = Initializer(mass_box, force_box, cosmology, phi)
    initial_state = initializer.get_initial_state(A_INIT)
    system = PoissonVlasov(force_box, cosmology, initializer.particle_mass)

    # 4. 运行模拟
    print("Starting N-body simulation...")
    stepper = partial(leap_frog_step, DT, system)
    halt_condition = lambda s: s.time > A_FINAL
    
    # 定义要保存和绘图的时间点
    plot_times = [0.5, 1.0, 2.0]
    
    run_simulation(stepper, halt_condition, initial_state, plot_times, DATA_DIR)
    print("Simulation finished.")

    # 5. 可视化结果
    print("Generating visualization...")
    triangles = get_box_triangles(mass_box)
    plot_simulation_snapshots(mass_box, triangles, plot_times, DATA_DIR, "structure_evolution.png")


if __name__ == "__main__":
    main()
