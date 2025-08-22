import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import h5py

from tools.timer import Timer

# 模型常数
N = 32 # 粒子数
L = N // 2 # 晶胞
beta = 0.1
delta = 0.1
epsilon = 0.1

# 模拟参数
dt = 0.5
t_max = 10**5

# 保存参数
save_dt = 200 # 每隔多少计算的时间保存一次
save_path = "./save/data.h5"

m1 = 1 - delta / 2
m2 = 1 + delta / 2

m = np.zeros(N)
m[0::2] = m1
m[1::2] = m2

@nb.njit()
def SPRK8_step(q, p, m, dt):
    def gradT(p, m):
        return p / m
    def gradV(q):
        pass
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

# 计算简振模
@Timer
def calc_normal_mode():
    k = np.arange(1, L+1) # 波数
    
    tmp = np.sqrt(1 - (4*m1*m2)/((m1+m2)**2) * np.sin(k*np.pi/(2*L+1))**2)
    omega_sq_plus = (m1+m2)/(m1*m2) * (1+tmp) # plus 为 optical branch（光学支）
    omega_sq_minus = (m1+m2)/(m1*m2) * (1-tmp) # minus 为 acoustic branch（声学支）
    omega_sq = np.concatenate((omega_sq_minus, omega_sq_plus[::-1]))
    omega = np.sqrt(omega_sq)

    u_plus = np.zeros((N,L))
    u_minus = np.zeros((N,L))

    k_idx = np.arange(1, L+1)[:, np.newaxis]
    j_idx = np.arange(1, N+1)[np.newaxis, :]

    #l_idx = (j_idx+1) // 2

    l_idx = np.arange(1, N//2 + 1)[:, np.newaxis]

    tmp1 = np.sin(2*l_idx*k_idx*np.pi/(2*L+1))
    tmp2 = np.sin(2*(l_idx-1)*k_idx*np.pi/(2*L+1))
    tmp3 = np.sin(2*k_idx*np.pi/(2*L+1))

    u_plus[0::2, :] = (tmp1 + tmp2) / tmp3
    u_plus[1::2, :] = (2 - m1*omega_sq_plus) * tmp1 / tmp3

    u_minus[0::2, :] = (tmp1 + tmp2) / tmp3
    u_minus[1::2, :] = (2 - m1*omega_sq_minus) * tmp1 / tmp3

    u = np.concatenate((u_minus, u_plus[::-1, :]), axis=1)

    U, _ = np.linalg.qr(u)
    return omega, U

def calc_E_k(p, q, m, omega, U):
    # p, q shoud be 1d array
    Q = np.einsum('j,jk->k', q, U)
    P = np.einsum('j,jk->k', p/m, U)
    E = 0.5 * (P**2 + omega**2 * Q**2)
    return E

def initialize(m, omega, U, epsilon, k=0.1):
    sort_idx = np.argsort(omega)
    omega = omega[sort_idx]
    U = U[:, sort_idx]

    N = m.shape[0]

    N_excited = int(k * N)
    N_excited = 1 if N_excited < 1 else N_excited

    E_tot = N * epsilon
    E_k = np.zeros(N)
    E_k[:N_excited] = E_tot / N_excited

    phi_k = np.random.uniform(0, 2*np.pi, N)

    Q = np.sqrt(2*E_k / omega**2) * np.sin(phi_k)
    P = np.sqrt(2*E_k) * np.cos(phi_k)

    q0 = np.einsum('k,jk->j', Q, U)
    p0 = m * np.einsum('k,jk->j', P, U)

    return q0, p0


@nb.njit()
def calc_chunk(q0, p0, m, n_steps, dt):
    q_save = np.zeros((n_steps, q0.shape[0]))
    p_save = np.zeros((n_steps, p0.shape[0]))
    # 注意：这里 q0, p0 并不会存储在 q_save, p_save 中
    q = q0.copy()
    p = p0.copy()

    for step in range(n_steps):
        q, p = SPRK8_step(q, p, m, dt)
        q_save[step] = q
        p_save[step] = p

    return q_save, p_save 

if __name__ == '__main__':
    # 初始化粒子
    omega, U = calc_normal_mode()

    q0, p0 = initialize(m, omega, U, epsilon)

    p = p0.copy()
    q = q0.copy()

    # 初始化保存
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('q', data=q, maxshape=(None, N), chunks=True)
        f.create_dataset('p', data=p, maxshape=(None, N), chunks=True)
        f.create_dataset('t', data=np.array([0.0]), maxshape=(None,), chunks=True)

    n_chunk = int(t_max / save_dt)
    q_last = q.copy()
    p_last = p.copy()

    for chunk_idx in range(n_chunk):
        t_start = chunk_idx * save_dt
        t_end = (chunk_idx+1) * save_dt

        q_save, p_save = calc_chunk(q, p, m, int(save_dt/dt), dt)
        t_save = np.arange(t_start, t_end, dt)

        with h5py.File(save_path, 'a') as f:
            dset_q = f['q']
            dset_p = f['p']
            dset_t = f['t']

            current_size = dset_t.shape[0]
            new_size = current_size + t_save.shape[0]

            dset_q.resize(new_size, axis=0)
            dset_p.resize(new_size, axis=0)
            dset_t.resize(new_size, axis=0)

            dset_q[current_size:] = q_save
            dset_p[current_size:] = p_save
            dset_t[current_size:] = t_save

        q_last = q_save[-1]
        p_last = p_save[-1]
        print("{chunk_idx+1}/{n_chunk}", end="\r")

    print("Calc done.")

    E_k_start = calc_E_k(p0, q0, m, omega, U)
    E_k_end = calc_E_k(p_last, q_last, m, omega, U)

    omega_idx = np.arange(1, omega.shape[0]+1)
    plt.plot(omega_idx, E_k_start, label='start')
    plt.plot(omega_idx, E_k_end, label='end')
    plt.legend()
    plt.show()
