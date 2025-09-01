import numpy as np
import matplotlib.pyplot as plt

def SPRK8_step(q, p, gradT, gradV, dt):
    q = q.copy()
    p = p.copy()
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
        q += D_COEFFS[i] * gradT(p) * dt
        p -= C_COEFFS[i] * gradV(q) * dt
    return q, p

def Yo_step(q, p, gradT, gradV, dt):
    q = q.copy()
    p = p.copy()

    alpha_s = np.array([0.5])
    n_stage = 4
    for n in range(1, n_stage):
        x1 = 1 / (2 - 2**(1 / (2*n + 1)))
        x0 = 1 - 2*x1
        alpha_new = np.concatenate([x1*alpha_s, x0*alpha_s, x1*np.flip(alpha_s)])
        alpha_s = alpha_new

    D_COEFFS = alpha_s.copy()
    C_COEFFS = alpha_s.copy()

    for i in range(len(C_COEFFS)):
        q += D_COEFFS[i] * gradT(p) * dt
        p -= C_COEFFS[i] * gradV(q) * dt
    return q, p

def Yo8_step(q, p, gradT, gradV, dt):
    q = q.copy()
    p = p.copy()

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
    # C_COEFFS = np.array([0.521213104349955, 1.431316259203525, 0.988973118915378,
    #                      1.298883627145484, 1.216428715985135, -1.227080858951161,
    #                      -2.031407782603105, -1.698326184045214, -1.698326184045214,
    #                      -2.031407782603105, -1.227080858951161, 1.216428715985135,
    #                      1.298883627145484, 0.988973118915378, 1.431316259203525,
    #                      0.521213104349955])
    # D_COEFFS = np.array([1.04242620869991, 1.82020630970714, 0.157739928123617,
    #                      2.44002732616735, -0.007169894197081, -2.44699182370524,
    #                      -1.61582374150097, -1.780828626589454, -1.61582374150097,
    #                      -2.44699182370524, -0.007169894197081, 2.44002732616735,
    #                      0.157739928123617, 1.82020630970714, 1.04242620869991])
    for i in range(15):
        p -= C_COEFFS[i] * gradV(q) * dt
        q += D_COEFFS[i] * gradT(p) * dt
    p -= C_COEFFS[15] * gradV(q) * dt
    return q, p

def Yos6_step(q, p, gradT, gradV, dt):
    q = q.copy()
    p = p.copy()
    a = [0.784513610477560,
         0.235573213359357,
        -1.17767998417887,
         1.31518632068390]
    C_COEFFS = np.array([a[0]/2, a[0]/2,
                         a[1]/2, a[1]/2,
                         a[2]/2, a[2]/2,
                         a[3]/2, a[3]/2,
                         a[2]/2, a[2]/2,
                         a[1]/2, a[1]/2,
                         a[0]/2])
    D_COEFFS = np.array([a[0]/2, a[0]/2,
                         a[1]/2, a[1]/2,
                         a[2]/2, a[2]/2,
                         a[3]/2, a[3]/2,
                         a[2]/2, a[2]/2,
                         a[1]/2, a[1]/2,
                         a[0]/2])
    for i in range(len(C_COEFFS)):
        q += D_COEFFS[i] * gradT(p) * dt
        p -= C_COEFFS[i] * gradV(q) * dt
    return q, p

def test_func(step_func):
    m, k = 1.0, 1.0
    gradT = lambda p: p/m
    gradV = lambda q: k*q
    dt = 0.01
    n_steps = 10**5
    Hs = []
    q, p = np.array([1.0]), np.array([0.0])
    for _ in range(n_steps):
        Hs.append(0.5*p**2/m + 0.5*k*q**2)
        q, p = step_func(q, p, gradT, gradV, dt)
    Hs = np.array(Hs)
    error = (Hs - Hs[0]) / Hs[0]
    print(f"{step_func.__name__}: mean error={np.mean(error)}")
    plt.plot(error)
    plt.show()

test_func(Yo8_step)
test_func(SPRK8_step)
test_func(Yos6_step)
test_func(Yo_step)
