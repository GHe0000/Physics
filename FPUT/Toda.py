
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh

"""
Toda chain (periodic) with 8th-order Symplectic Partitioned Runge–Kutta (Yoshida-like) integrator.
- Verifies near-constant Toda invariants via Lax spectrum (eigenvalues of L)
- Shows FPU/Toda-like recurrence via shape correlation R(t)
- Numba-accelerated force and integrator inner loops

Potential (standard Toda, set a=b=1):
    V(x) = exp(-x) + x - 1
V'(x) = 1 - exp(-x)
Equations:
    dq_i/dt = p_i / m_i
    dp_i/dt = V'(q_i - q_{i-1}) - V'(q_{i+1} - q_i)   (periodic)

Tip: use smaller dt for better conservation; increase n_steps for clearer recurrences.
"""

# ---------------------- Parameters ----------------------
N = 32                 # number of particles
periodic = True        # periodic boundary (integrable periodic Toda)
m = np.ones(N)         # masses

# Toda parameters (kept as 1 here). If you want, generalize as: V(x)=A/B**2*(np.exp(-B*x)+B*x-1)
A = 1.0
B = 1.0

# Time integration
T_total = 200.0
# dt needs to be sufficiently small for 8th-order SPRK stability and energy conservation
# Reduce dt if you observe drift in invariants.
dt = 0.02
n_steps = int(T_total / dt)

save_every = 5  # sample every k steps for plotting/diagnostics

# ---------------------- SPRK(8) Coefficients ----------------------
C_COEFFS = np.array([
    0.195557812560339,
    0.433890397482848,
    -0.207886431443621,
    0.078438221400434,
    0.078438221400434,
    -0.207886431443621,
    0.433890397482848,
    0.195557812560339,
], dtype=np.float64)

D_COEFFS = np.array([
    0.0977789062801695,
    0.289196093121589,
    0.2528135839,
    -0.139788583301759,
    -0.139788583301759,
    0.2528135839,
    0.289196093121589,
    0.0977789062801695,
], dtype=np.float64)

# ---------------------- Numba-accelerated kernels ----------------------
@nb.njit(cache=True, fastmath=True)
def vprime(x, A, B):
    # V'(x) for V(x) = A/B^2 * (exp(-B*x) + B*x - 1) => V'(x) = A/B * (1 - exp(-B*x))
    return (A / B) * (1.0 - np.exp(-B * x))

@nb.njit(cache=True, fastmath=True)
def gradT(p, m):
    out = np.empty_like(p)
    for i in range(p.size):
        out[i] = p[i] / m[i]
    return out

@nb.njit(cache=True, fastmath=True)
def gradV_periodic(q, A, B):
    """Compute dH/dq for periodic Toda: V'(q_i - q_{i-1}) - V'(q_{i+1} - q_i)."""
    N = q.size
    g = np.empty_like(q)
    for i in range(N):
        im1 = i - 1 if i > 0 else N - 1
        ip1 = i + 1 if i < N - 1 else 0
        left = vprime(q[i] - q[im1], A, B)
        right = vprime(q[ip1] - q[i], A, B)
        g[i] = left - right
    return g

@nb.njit(cache=True, fastmath=True)
def sprk8_step(q, p, m, dt, A, B):
    # One full 8-stage SPRK update (in-place)
    for s in range(8):
        # Drift (update q)
        v = gradT(p, m)
        for i in range(q.size):
            q[i] += D_COEFFS[s] * v[i] * dt
        # Kick (update p)
        g = gradV_periodic(q, A, B)
        for i in range(p.size):
            p[i] -= C_COEFFS[s] * g[i] * dt

@nb.njit(cache=True, fastmath=True)
def hamiltonian(q, p, m, A, B):
    # H = sum p^2/(2m) + sum V(q_{i+1}-q_i)
    N = q.size
    H = 0.0
    for i in range(N):
        H += 0.5 * p[i] * p[i] / m[i]
    for i in range(N):
        ip1 = i + 1 if i < N - 1 else 0
        x = q[ip1] - q[i]
        # V(x) = A/B^2 * (exp(-B*x) + B*x - 1)
        H += A / (B * B) * (np.exp(-B * x) + B * x - 1.0)
    return H

# ---------------------- Lax matrix & invariants (NumPy) ----------------------
def lax_matrix(q, p, A=1.0, B=1.0):
    """Build symmetric tridiagonal Lax matrix L for periodic Toda in Flaschka vars.
    Using: a_i = 0.5 * sqrt(A) * exp(-B*(q_{i+1}-q_i)/2)
           b_i = -0.5 * p_i / sqrt(B) * sqrt(B)  (we keep B in a_i; b_i = -0.5 p_i)
    Here we set: a_i = 0.5 * np.sqrt(A) * np.exp(-0.5*B*(q_{i+1}-q_i)), b_i = -0.5 * p_i.
    The eigenvalues of L are constants of motion.
    """
    N = q.size
    a = 0.5 * np.sqrt(A) * np.exp(-0.5 * B * (np.roll(q, -1) - q))
    b = -0.5 * p.copy()
    L = np.zeros((N, N), dtype=np.float64)
    np.fill_diagonal(L, b)
    # periodic off-diagonals
    for i in range(N - 1):
        L[i, i + 1] = a[i]
        L[i + 1, i] = a[i]
    L[0, -1] = a[-1]
    L[-1, 0] = a[-1]
    return L

# ---------------------- Initial conditions ----------------------
def initial_conditions(N, mode=1, amp=0.5, seed=0):
    """Small-amplitude sinusoidal displacement, zero mean, zero total momentum.
    For periodic BC: q_i = amp * sin(2π * mode * i / N), p_i = 0.
    This typically exhibits clear recurrences in Toda.
    """
    i = np.arange(N)
    q0 = amp * np.sin(2.0 * np.pi * mode * i / N)
    p0 = np.zeros(N)
    # remove mean displacement to avoid drift of center of mass (not needed but neat)
    q0 -= np.mean(q0)
    return q0, p0

# ---------------------- Simulation ----------------------
def run():
    q, p = initial_conditions(N, mode=1, amp=0.4)

    # Record diagnostics
    ns = n_steps // save_every + 1
    t_rec = np.zeros(ns)
    H_rec = np.zeros(ns)
    R_rec = np.zeros(ns)  # shape recurrence metric
    eig_err = np.zeros(ns)

    q0 = q.copy()
    p0 = p.copy()

    # reference eigenvalues (invariants)
    L0 = lax_matrix(q0, p0, A, B)
    lam0 = np.sort(eigvalsh(L0))

    H_rec[0] = hamiltonian(q, p, m, A, B)
    R_rec[0] = 0.0
    eig_err[0] = 0.0

    qi = q.copy()
    pi = p.copy()

    sidx = 1
    for step in range(1, n_steps + 1):
        sprk8_step(qi, pi, m, dt, A, B)
        if step % save_every == 0:
            t = step * dt
            t_rec[sidx] = t
            H_rec[sidx] = hamiltonian(qi, pi, m, A, B)
            # normalized shape recurrence: ||q(t)-q(0)|| / ||q(0)||
            denom = np.linalg.norm(q0) + 1e-15
            R_rec[sidx] = np.linalg.norm(qi - q0) / denom
            # Lax spectrum deviation
            L = lax_matrix(qi, pi, A, B)
            lam = np.sort(eigvalsh(L))
            eig_err[sidx] = np.max(np.abs(lam - lam0))
            sidx += 1

    return t_rec, H_rec, R_rec, eig_err, q0, p0, qi, pi

# ---------------------- Plotting ----------------------
def main():
    t_rec, H_rec, R_rec, eig_err, q0, p0, qf, pf = run()

    fig1 = plt.figure(figsize=(8, 4))
    plt.plot(t_rec, H_rec - H_rec[0])
    plt.xlabel('t')
    plt.ylabel('Energy drift H(t) - H(0)')
    plt.title('Total energy conservation (SPRK8)')
    plt.tight_layout()

    fig2 = plt.figure(figsize=(8, 4))
    plt.plot(t_rec, R_rec)
    plt.xlabel('t')
    plt.ylabel('R(t) = ||q(t)-q(0)|| / ||q(0)||')
    plt.title('Toda recurrence (lower is closer)')
    plt.tight_layout()

    fig3 = plt.figure(figsize=(8, 4))
    plt.plot(t_rec, eig_err)
    plt.xlabel('t')
    plt.ylabel('max |λ(t) - λ(0)|')
    plt.title('Integrability check via Lax spectrum invariance')
    plt.tight_layout()

    # Example snapshot of configuration at final time vs initial
    fig4 = plt.figure(figsize=(8, 4))
    i = np.arange(N)
    plt.plot(i, q0, label='q(0)')
    plt.plot(i, qf, label='q(t_final)')
    plt.xlabel('site i')
    plt.ylabel('q_i')
    plt.title('Initial vs final displacement')
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
