# 一个辛积分器库
# Symplectic Partitioned Runge-Kutta
# 8 阶的辛分区 Runge-Kutta 法求解器
import numpy as np
import numba as nb
from typing import Tuple, Callable

def _SPRK_loop(force_func, mass, q0, p0, dt, n_step):
    # 辛积分器的常数，来自文献：
    # Laskar, J., & Robutel, P. (2001). High order symplectic integrators for the Solar System. Celestial Mechanics and Dynamical Astronomy, 80(1), 39-62.

    C_COEFFS = np.array([
        0.195557812560339,
        0.433890397482848,
        -0.207886431443621,
        0.078438221400434,
        0.078438221400434,
        -0.207886431443621,
        0.433890397482848,
        0.195557812560339,
    ])
    
    D_COEFFS = np.array([
        0.0977789062801695,
        0.289196093121589,
        0.252813583900000,
        -0.139788583301759,
        -0.139788583301759,
        0.252813583900000,
        0.289196093121589,
        0.0977789062801695,
    ])

    q_save = np.zeros((n_step+1, len(q0)))
    p_save = np.zeros((n_step+1, len(p0)))

    q_save[0] = q0
    p_save[0] = p0

    q = q0.copy()
    p = p0.copy()

    for i in range(n_step):
        for j in range(8):
            q += D_COEFFS[j] * (p / mass) * dt
            p += C_COEFFS[j] * force_func(q) * dt
        q_save[i+1] = q
        p_save[i+1] = p
    return q_save, p_save    


def SPRK(
        force_func: Callable[[np.ndarray], np.ndarray],
        mass: np.ndarray,
        y0: Tuple[np.ndarray, np.ndarray],
        t: float,
        dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    一个 8 阶的辛分区 Runge-Kutta 法 Hamilton 方程求解器. 

    此函数使用由Laskar & Robutel提出的 8 阶辛积分方法，对形如
    H(q,p) = T(p) + V(q) 的可分离哈密顿系统进行数值积分.

    函数会自动检查 `force_func` 是否被Numba编译，以实现高性能计算.

    Parameters
    ----------
    force_func : callable
        计算力的函数，`f(q) = -dV(q)/dq`
        其函数签名为 `f(q) -> F`，其中 q 和 F 均为 NumPy 数组.
        为了获得最佳性能，此函数应由 Numba 的 njit 装饰器编译.
    mass : np.ndarray
        每个粒子的质量.
    y0 : tuple of np.ndarray
        一个包含初始位置和初始动量的元组 `(q0, p0)`.
    t : float
        总积分时间.
    dt : float
        每个积分步长的时间间隔.

    Returns
    -------
    t_eval : np.ndarray
        从0到总积分时间的时刻数组，形状为 `(n_step + 1,)`.
    q : np.ndarray
        位置的轨迹数组，形状为 `(n_step + 1, N)`.
    p : np.ndarray
        动量的轨迹数组，形状为 `(n_step + 1, N)`.
    """

    q0, p0 = y0

    q0 = np.asarray(q0, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)

    n_step = int(t / dt)

    t = np.linspace(0, t, n_step+1)

    if isinstance(force_func, nb.core.dispatcher.Dispatcher):
        loop_func = nb.njit(_SPRK_loop)
    else:
        print("Warning: force_func is not numba-compiled, may be slow.")
        loop_func = _SPRK_loop
    q, p = loop_func(force_func, mass, q0, p0, dt, n_step)
    return t, q, p

