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

d_half = np.array(w[::-1])
d = np.concatenate((d_half,
		    np.array([w0]),
		    d_half[::-1]))


w_tmp = np.concatenate((np.array([w0]),w))
w_tmp_p1 = np.concatenate((w_tmp[1:], np.zeros(1)))
c_half = (w_tmp + w_tmp_p1)/2
c = np.concatenate((c_half[::-1], c_half))

c_str = np.array2string(np.array(c), separator=',')
d_str = np.array2string(np.array(d), separator=',')

print(f"C=\n {c_str}")
print(f"D=\n {d_str}")

print(f"sum c={np.sum(c)}")
print(f"sum d={np.sum(d)}")
