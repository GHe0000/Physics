import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from scipy.linalg import eigh

from tools.timer import LapTimer

np.random.seed(3407)

N = 32
delta = 0.5
epsilon = 0.1
beta = 0.2

t_end = 10**7
dt = 0.1
t_save = 10**6

m1 = 1 - delta / 2
m2 = 1 + delta / 2

def calc_H(q, p, m):
    V = lambda x: x**2 / 2.0 + beta * x**4 / 4.0
    q_m1 = np.concatenate((np.zeros(1), q[:-1]))
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
def Yo8_step(q, p, m, dt):
    q = q.copy()
    p = p.copy()
    def gradT(p, m):
        return p / m 

    def dV(x):
        return x + beta * x**3

    def gradV(q):
        q_m1 = np.concatenate((np.zeros(1), q[:-1]))
        q_p1 = np.concatenate((q[1:], np.zeros(1)))
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


@nb.njit(fastmath=True, nogil=True)
def calc_chunk(q0, p0, m, n_steps, dt):
    # 这里 q0, p0, 并不保存在 q_save, p_save 中
    q_save = np.zeros((n_steps, q0.shape[0]))
    p_save = np.zeros((n_steps, p0.shape[0]))

    q = q0.copy()
    p = p0.copy()

    for step in range(n_steps):
        q, p = Yo8_step(q, p, m, dt)
        q_save[step] = q
        p_save[step] = p 
    return q_save, p_save

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

    
    q_last = q0.copy()
    p_last = p0.copy()

    timer = LapTimer()
    n_chunk = int(t_end / t_save)
    
    for chunk_idx in range(n_chunk):
        q_save, p_save = calc_chunk(q_last, p_last, m, int(t_save / dt), dt)
        q_last = q_save[-1]
        p_last = p_save[-1]
        Et = np.sum(calc_Ek(q_last, p_last, m, omega, U))
        Ht = calc_H(q_last, p_last, m)
        calc_time = timer()
        print(f"chunk {chunk_idx+1}/{n_chunk}, H={Ht}, E={Et}, {calc_time:.2f}")

    q = q_last.copy()
    p = p_last.copy()

    H_t = calc_H(q, p, m)
    Ek_t = calc_Ek(q, p, m, omega, U)
    print(f"H_t={H_t}, sum Ek_t={np.sum(Ek_t)}, H_t-sum Ek_t={H_t-np.sum(Ek_t)}")
    sort_idx = np.argsort(omega)
    plt.figure(figsize=(10, 6))
    plt.plot(Ek_t[sort_idx], 'o-')
    plt.grid(True)
    plt.show()
