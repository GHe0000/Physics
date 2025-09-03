import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from tools.timer import LapTimer

np.random.seed(3407)

N = 32
delta = 0.5
epsilon = 0.1
beta = 0.25

t_end = 10**7
dt = 0.1
dt_save = 100

m1 = 1 - delta / 2
m2 = 1 + delta / 2

def calc_H(q, p, m):
    V = lambda x: (np.exp(2*x) - 2*x - 1)/4
    q_m1 = np.concatenate((np.zeros(1), q[:-1]))
    T = np.sum(V(q - q_m1)) + V(q[-1])
    U = np.sum(p**2 / (2 * m))
    return T + U

def normal_mode(m):
    # NOTE: 使用论文的简振模的公式计算出的 H 和 sum E 差距过大
    # 这里暂时使用刚度矩阵通过广义特征值求解
    K = 2*np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    M = np.diag(m)
    omega_sq, U = eigh(K, b=M)
    omega = np.sqrt(np.maximum(0, omega_sq))
    return omega, U

def calc_Ek(q, p, m, omega, U):
    Q = np.einsum('ji,j,j->i', U, m, q)
    P = np.einsum('ji,j->i', U, p)
    return 0.5 * (P**2 + omega**2 * Q**2)

def initialize(m, omega, U, epsilon, k=0.1):
    N = len(m)
    sort_idx = np.argsort(omega)
    omega = omega[sort_idx]
    U = U[:, sort_idx]

    N_excited = int(k * N)
    N_excited = 1 if N_excited < 1 and N > 0 else N_excited
    
    Etot = N * epsilon
    Ek = np.zeros(N)
    Ek[:N_excited] = Etot / N_excited

    phi_k = np.random.uniform(0, 2 * np.pi, N)
    
    Q = np.zeros(N)
    valid_omega = omega > 1e-9
    Q[valid_omega] = np.sqrt(2 * Ek[valid_omega] / omega[valid_omega]**2) * np.sin(phi_k[valid_omega])
    P = np.sqrt(2 * Ek) * np.cos(phi_k)

    q0 = np.einsum('ij,j->i', U, Q)
    p0 = np.einsum('i,ij,j->i', m, U, P)
    return q0, p0

@nb.njit()
def Yo8_step(q, p, m, dt):
    q = q.copy()
    p = p.copy()

    def gradT(p, m):
        return p / m 

    def gradV(q):
        dV = lambda x: x + beta * x**3
        q_m1 = np.concatenate((np.zeros(1), q[:-1]))
        q_p1 = np.concatenate((q[1:], np.zeros(1)))
        return dV(q - q_m1) - dV(q_p1 - q)

    C_COEFFS = np.array([0.521213104349955, 1.431316259203525, 0.988973118915378,
                         1.298883627145484, 1.216428715985135, -1.227080858951161,
                         -2.031407782603105, -1.698326184045211, -1.698326184045211,
                         -2.031407782603105, -1.227080858951161, 1.216428715985135,
                         1.298883627145484, 0.988973118915378, 1.431316259203525,
                         0.521213104349955])
    D_COEFFS = np.array([1.04242620869991, 1.82020630970714, 0.157739928123617,
                         2.44002732616735, -0.007169894197081, -2.44699182370524,
                         -1.61582374150097, -1.780828626589452, -1.61582374150097,
                         -2.44699182370524, -0.007169894197081, 2.44002732616735,
                         0.157739928123617, 1.82020630970714, 1.04242620869991])
    for i in range(15):
        p -= C_COEFFS[i] * gradV(q) * dt
        q += D_COEFFS[i] * gradT(p, m) * dt
    p -= C_COEFFS[15] * gradV(q) * dt
    return q, p

@nb.njit()
def calc_chunk(q, p, m, dt, n_steps):
    for step in range(n_steps):
        q, p = Yo8_step(q, p, m, dt)
    return q, p

if __name__ == '__main__':
    m = np.zeros(N)
    m[0::2] = m1
    m[1::2] = m2

    omega, U = normal_mode(m)
    q0, p0 = initialize(m, omega, U, epsilon, k=0.1)
    Ek= calc_Ek(q0, p0, m, omega, U)
    
    sort_idx = np.argsort(omega)
    plt.figure(figsize=(10, 6))
    plt.plot(Ek[sort_idx], 'o-')
    plt.grid(True)
    plt.show()

    H = calc_H(q0, p0, m)
    print(f"H={H}, sum Ek={np.sum(Ek)}, H-sum Ek={H-np.sum(Ek)}")

    q = q0.copy()
    p = p0.copy()
    n_chunks = int(t_end / dt_save)
    Ekt = np.zeros((n_chunks, N))
    Ht = np.zeros(n_chunks)
    Tt = np.arange(n_chunks) * dt_save

    n_print = n_chunks // 100
    timer = LapTimer()
    timer()
    for chunk in range(n_chunks):
        q, p = calc_chunk(q, p, m, dt, int(dt_save / dt))
        Ht[chunk] = calc_H(q, p, m)
        Ekt[chunk] = calc_Ek(q, p, m, omega, U)
        if chunk % n_print == 0:
            pt = timer()
            print(f"{chunk}/{n_chunks}: {pt:.2f}s, Delta H={Ht[chunk]-Ht[0]:.8f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Tt, Ht)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(10):
        ax.plot(Tt, Ekt[:, i], label=f'k={i}')
    plt.legend()
    plt.show()
