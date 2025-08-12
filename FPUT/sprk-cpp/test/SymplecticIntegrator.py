# 一个辛积分器库
import numpy as np
import numba as nb

def _SPRK_loop(force_func, mass, q0, p0, dt, n_step):
    """
    SPRK 的内部循环，使用 numba 加速
    """

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

def SPRK(force_func, mass, y0, t, dt):
    """
    一个 8 阶的辛分区 Runge-Kutta 法 Hamilton 方程求解器. 
    
    此函数用来计算一个形如 H(q,p) = T(p) + V(q) 的 Hamilton 系统的演化
    
    Args：
        force_func(callable): 力函数，计算 `f(q)`，建议使用 numba 加速
        mass(np.ndarray): 粒子的质量.
        y0(tuple): 初始条件 `(q0, p0)`
        t(float): 起始时间
        dt(float): 时间步长
    
    Returns:
        tuple: `(t, q, p)`
        - t(np.ndarray): 时间序列
        - q(np.ndarray): 对应时间的位置
        - p(np.ndarray): 对应时间的速度
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

