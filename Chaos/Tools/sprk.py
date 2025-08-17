'''
Symplectic Partitioned Runge-Kutta
8 阶的辛分区 Runge-Kutta 法求解器
'''
import numpy as np
import numba as nb

def _SPRK_core(gradT, gradV, q0, p0, dt, n_step):
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
    q_save = np.zeros((n_step + 1, len(q0)))
    p_save = np.zeros((n_step + 1, len(p0)))

    q_save[0] = q0
    p_save[0] = p0

    q = q0.copy()
    p = p0.copy()

    for i in range(n_step):
        for j in range(8):
            q += D_COEFFS[j] * gradT(p) * dt
            p -= C_COEFFS[j] * gradV(q) * dt
        q_save[i+1] = q
        p_save[i+1] = p
    return q_save, p_save

def SPRK8(
        gradT: callable[[np.ndarray], np.ndarray],
        gradV: callable[[np.ndarray], np.ndarray],
        q0: np.ndarray,
        p0: np.ndarray,
        t: float,
        dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    一个通用的 8 阶辛分区 Runge-Kutta 法 Hamilton 方程求解器.

    此函数使用由 Laskar & Robutel 提出的 8 阶辛积分方法，对形如
    H(q,p) = T(p) + V(q) 的可分离哈密顿系统进行数值积分.
    
    与原始版本不同，此函数接受动能关于动量的梯度 (dT/dp) 和
    势能关于位置的梯度 (dV/dq) 作为输入，使其更具通用性.

    函数会自动检查梯度函数是否被 Numba 编译，以实现高性能计算.

    Parameters
    ----------
    gradT : callable
        计算动能梯度 dT/dp 的函数.
        函数签名为 `f(p) -> np.ndarray`.
        为了获得最佳性能，此函数应由 Numba 的 njit 装饰器编译.
        
    gradV : callable
        计算势能梯度 dV/dq 的函数.
        函数签名为 `f(q) -> np.ndarray`.
        为了获得最佳性能，此函数应由 Numba 的 njit 装饰器编译.
        
    q0 : np.ndarray
        初始位置.
        
    p0 : np.ndarray
        初始动量.
        
    t : float
        总计算时间.
        
    dt : float
        每个积分步长的时间间隔.

    Returns
    -------
    t_eval : np.ndarray
        从 0 到总积分时间的时刻数组，形状为 `(n_step + 1,)`.
    q : np.ndarray
        位置的轨迹数组，形状为 `(n_step + 1, N)`.
    p : np.ndarray
        动量的轨迹数组，形状为 `(n_step + 1, N)`.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)

    n_step = int(t / dt)
    t_eval = np.linspace(0, t, n_step + 1)

    # 检查两个函数是否都被 Numba 编译
    is_jitted = (isinstance(gradT, nb.core.dispatcher.Dispatcher) and
                 isinstance(gradV, nb.core.dispatcher.Dispatcher))

    if is_jitted:
        loop_func = nb.njit(_SPRK_core)
    else:
        print("Warning: One or both gradient functions are not numba-compiled, may be slow.")
        loop_func = _SPRK_core
    q, p = loop_func(gradT, gradV, q0, p0, dt, n_step)
    return t_eval, q, p
