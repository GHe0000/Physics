import numpy as np


class Yo8:
    """Yoshida 8th-order symplectic integrator."""

    def __init__(self):
        self.c, self.d = self._yo8_coeffs()

    def _yo8_coeffs(self):
        # Yoshida recursive construction (8th order)
        def base():
            return np.array([0.5]), np.array([1.0])

        c, d = base()
        for n in range(1, 4):  # build up to 8th order
            x1 = 1 / (2 - 2**(1 / (2*n + 1)))
            x0 = 1 - 2 * x1
            c = np.concatenate([x1 * c, [x0 * c[0]], x1 * c[::-1]])
            d = np.concatenate([x1 * d, [x0 * d[0]], x1 * d[::-1]])
        return c, d

    def step(self, p, q, gradT, gradV, H, dt):
        """One Yo8 step."""
        for ci, di in zip(self.c, self.d):
            q = q + ci * dt * gradT(p)  # drift
            p = p - di * dt * gradV(q)  # kick
        return p, q


import matplotlib.pyplot as plt

# -------------------
# β-FPUT 模型定义
# -------------------
def make_beta_fput(N, beta=1.0):
    def T(p):
        return 0.5 * np.sum(p**2)
    def V(q):
        dq = np.diff(np.concatenate(([0], q, [0])))
        return 0.5 * np.sum(dq**2) + beta/4 * np.sum(dq**4)
    def H(p,q):
        return T(p) + V(q)
    def gradT(p):
        return p
    def gradV(q):
        bq = np.concatenate(([0.0], q, [0.0]))     # 固定端 q_0=q_{N+1}=0
        dq = np.diff(bq)                            # 长度 N+1，dq_i = q_{i+1}-q_i
        s  = dq + beta * dq**3
        return s[:-1] - s[1:]                       # ∂V/∂q_i = s_{i-1} - s_i

    return gradT, gradV, H

# -------------------
# 测试模拟
# -------------------
N = 32
beta = 0.01
dt = 0.01
steps = 20000

# 初始条件：激发第一个正弦模
q = 0.1 * np.sin(np.pi * np.arange(1, N+1) / (N+1))
p = np.zeros(N)

gradT, gradV, H = make_beta_fput(N, beta)
yo8 = Yo8()

E0 = H(p,q)
energies = []
times = []

for step in range(steps):
    if step % 100 == 0:
        energies.append(H(p,q))
        times.append(step*dt)
    p, q = yo8.step(p, q, gradT, gradV, H, dt)

# -------------------
# 绘制能量随时间
# -------------------
plt.plot(times, (energies-E0)/E0, label="Total Energy")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.legend()
plt.show()



deltaE = (energies - E0) / E0
print(f"{np.mean(deltaE):.4f}, {np.max(deltaE):.4f}, {np.min(deltaE):.4f}")
print("sum(c)=", yo8.c.sum(), "sum(d)=", yo8.d.sum())
print("c[:5]=", yo8.c[:5])
print("d[:5]=", yo8.d[:5])
