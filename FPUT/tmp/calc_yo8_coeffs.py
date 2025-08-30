import numpy as np

np.set_printoptions(precision=15) # 控制打印精度

# Solution A
w = np.array([
    -0.161582374150097E1,
    -0.244699182370524E1,
    -0.716989419708120E-2,
     0.244002732616735E1,
     0.157739928123617E0,
     0.182020630970714E1,
     0.104242620869991E1,
])

w0 = 1 - 2*np.sum(w)

m = len(w)

d = [w[m-1-i] for i in range(m)] + [w0]

c = (
    [0.5 * w[m-1]] +
    [0.5 * (w[m-1-i] + w[m-2-i]) for i in range(1, m)] +
    [0.5 * (w[0] + w0)]
)

c_str = np.array2string(np.array(c), separator=',')
d_str = np.array2string(np.array(d), separator=',')

print(f"C=\n {c_str}")
print(f"D=\n {d_str}")
