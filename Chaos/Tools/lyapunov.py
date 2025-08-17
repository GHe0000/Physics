'''
Lyapunov 指数相关
'''

import numpy as np
import numba as nb

@nb.njit()
def lyapunov_exponent(stepper, x0, dt, n_step, renorm_step):
    dim = x0.size
    x = x0.copy()
    Q = np.eye(dim)

    lambda_sum = np.zeros(dim)
    n_cycle = n_step // renorm_step
    t_save = np.arange(1, n_cycle+1) * renorm_step * dt
    lambda_save = np.zeros((n_cycle, dim))

    for cycle in range(n_cycle):
        for _ in range(renorm_step):
            x, Q = stepper(x, Q, dt)
        # 正交化
        Q, R = np.linalg.qr(Q)
        Q = np.ascontiguousarray(Q) # 转为 C 顺序，从而加速
        diag_R = np.diag(R)
        Q[:, diag_R < 0] *= -1
        lambda_sum += np.log(np.abs(diag_R) + 1e-100) # 防止 log(0)
        lambda_save[cycle] = lambda_sum / t_save[cycle]
    return t_save, lambda_save

# 数值 Jaccobian
@nb.njit()
def numerical_jacobian(f, x, dt):
    pass
