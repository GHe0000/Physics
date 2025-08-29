import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

np.random.seed(3407)

N = 32
delta = 0.5
epsilon = 0.01
beta = 0.0

m1 = 1 - delta / 2
m2 = 1 + delta / 2

def calc_H(q, p, m):
    V = lambda x: x**2 / 2.0 + beta * x**4 / 4.0
    q_m1 = np.concatenate(([0.], q[:-1]))
    T = np.sum(V(q - q_m1)) + V(q[-1])
    U = np.sum(p**2 / (2 * m))
    return T + U

def normal_mode(m):
    K = (2*np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1))
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

if __name__ == '__main__':
    m = np.zeros(N)
    m[0::2] = m1
    m[1::2] = m2

    omega, U = normal_mode(N, m)
    q0, p0 = initialize(m, omega, U, epsilon, k=0.1)
    Ek= calc_Ek(q0, p0, m, omega, U)
    
    sort_idx = np.argsort(omega)
    plt.figure(figsize=(10, 6))
    plt.plot(Ek[sort_idx], 'o-')
    plt.grid(True)
    plt.show()

    H = calc_H(q0, p0, m)
    print(f"H={H}, sum Ek={np.sum(Ek)}, H-sum Ek={H-np.sum(Ek)}")

