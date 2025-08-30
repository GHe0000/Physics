import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.linalg import eigh

np.random.seed(3407)

N = 32
delta = 0.
epsilon = 0.1
alpha = 0.25

t_end = 1E4
dt = 0.1

m1 = 1 - delta / 2
m2 = 1 + delta / 2

def calc_H(q, p, m):
    V = lambda x: x**2 / 2.0 + alpha * x**3 / 3.0
    q_m1 = np.zeros(N)
    q_m1[1:] = q[:-1]
    T = np.sum(V(q - q_m1)) + V(q[-1])
    U = np.sum(p**2 / (2 * m))
    return T + U

def normal_mode(m):
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
def SPRK8_step(q, p, m, dt):
    q = q.copy()
    p = p.copy()

    def gradT(p, m):
        return p / m

    def gradV(q):
        dV = lambda x: x + alpha * x**2
        q_m1 = np.zeros(N)
        q_m1[1:] = q[:-1]
        q_p1 = np.zeros(N)
        q_p1[:-1] = q[1:]
        return dV(q - q_m1) - dV(q_p1 - q)

    # 积分器常数
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

    for i in range(8):
        q += D_COEFFS[i] * gradT(p, m) * dt
        p -= C_COEFFS[i] * gradV(q) * dt
    return q, p


if __name__ == '__main__':
    m = np.zeros(N)
    m[0::2] = m1
    m[1::2] = m2

    omega, U = normal_mode(m)
    q0, p0 = initialize(m, omega, U, epsilon, k=0)
    Ek = calc_Ek(q0, p0, m, omega, U)

    sort_idx = np.argsort(omega)
    plt.figure(figsize=(10, 6))
    plt.plot(Ek[sort_idx], 'o-')
    plt.grid(True)
    plt.show()

    H = calc_H(q0, p0, m)
    print(f"H={H}, sum Ek={np.sum(Ek)}, H-sum Ek={H-np.sum(Ek)}")

    q = q0.copy()
    p = p0.copy()
    nsteps = int(t_end / dt)

    Ekt = np.zeros((nsteps+1, 5))
    Ht = np.zeros(nsteps+1)
    t_span = np.arange(nsteps+1) * dt

    Ekt[0] = calc_Ek(q, p, m, omega, U)[:5]
    Ht[0] = calc_H(q, p, m)

    for step in range(nsteps):
        q, p = SPRK8_step(q, p, m, dt)
        Ekt[step+1] = calc_Ek(q, p, m, omega, U)[:5]
        Ht[step+1] = calc_H(q, p, m)

    Hend = calc_H(q, p, m)
    Ekend = calc_Ek(q, p, m, omega, U)
    print(f"H_t={Hend}, sum Ek_t={np.sum(Ekend)}, H_t-sum Ek_t={Hend-np.sum(Ekend)}")
    sort_idx = np.argsort(omega)
    plt.figure(figsize=(10, 6))
    plt.plot(Ekend[sort_idx], 'o-')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(t_span, Ekt[:, i], label=f'q{i}')
    plt.grid(True)
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()
    ax2.plot(t_span, (Ht-Ht[0])/Ht[0], label='error', color='r')
    ax2.set_ylabel('error')
    ax.plot(t_span, Ht, label='H', color='b')
    ax.set_xlabel('t')
    ax.set_ylabel('H')
    ax.grid(True)
    lines, labels = ax.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax.legend(lines + bars, labels + bar_labels, loc='upper left')
    plt.show()
    print(f"max error = {np.max(np.abs( (Ht-Ht[0])/Ht[0] ))}")
