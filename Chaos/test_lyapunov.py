import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from tools.lyapunov import lyapunov_exponent

sigma, rho, beta = 10.0, 28.0, 8.0/3.0

@nb.njit()
def lorenz_rhs(x):
    dx = np.empty(3)
    dx[0] = sigma * (x[1] - x[0])
    dx[1] = x[0]*(rho - x[2]) - x[1]
    dx[2] = x[0]*x[1] - beta*x[2]
    return dx

@nb.njit()
def lorenz_jac(x):
    J = np.empty((3,3))
    J[0,0] = -sigma; J[0,1] = sigma;   J[0,2] = 0.0
    J[1,0] = rho-x[2]; J[1,1] = -1.0;  J[1,2] = -x[0]
    J[2,0] = x[1];     J[2,1] = x[0];  J[2,2] = -beta
    return J

@nb.njit()
def lorenz_stepper(x, Q, dt):
    k1 = lorenz_rhs(x)
    k2 = lorenz_rhs(x + 0.5*dt*k1)
    k3 = lorenz_rhs(x + 0.5*dt*k2)
    k4 = lorenz_rhs(x + dt*k3)
    x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    J1 = lorenz_jac(x) @ Q
    J2 = lorenz_jac(x + 0.5*dt*k1) @ (Q + 0.5*dt*J1)
    J3 = lorenz_jac(x + 0.5*dt*k2) @ (Q + 0.5*dt*J2)
    J4 = lorenz_jac(x + dt*k3) @ (Q + dt*J3)
    Q_next = Q + (dt/6.0)*(J1 + 2*J2 + 2*J3 + J4)

    return x_next, Q_next

x0 = np.array([1.0, 1.0, 1.0])
dt = 0.01
n_step = 200_000
renorm_step = 10

t, lams = lyapunov_exponent(lorenz_stepper, x0, dt, n_step, renorm_step)

# 绘图
plt.figure(figsize=(8,5))
for i in range(lams.shape[1]):
    plt.plot(t, lams[:,i], label=f"λ{i+1}")
plt.axhline(0, color="k", linestyle="--")
plt.legend()
plt.show()

print(lams[-1])
