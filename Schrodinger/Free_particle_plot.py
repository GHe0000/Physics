import numpy as np
import matplotlib.pyplot as plt
import math

from numba import jit

import time

a = 0.5
h = 1
m = 1

start_tot = time.time()

@jit(nopython=True)
def grate(func, a, b, n, x_0, t):
    x = np.linspace(a, b, n)
    y = func(x, x_0, t)
    dx = x[1] - x[0]
    return np.sum(y*dx)

@jit(nopython=True)
def f(k, x_0, t):
    return ((np.sin(a*k))/k) * np.exp(1j * (k*x_0-(h*(k**2))*t))

@jit(nopython=True)
def psi(x, t):
    ans = grate(f, -100, 100, 10000, x,t)
    ans = 1/(np.pi*np.sqrt(2*a)) * ans
    ans = abs(ans) ** 2
    return(ans)

#@jit(nopython=True)
def caculate(x, t):
    start = time.time()
    y = [0] * len(x)
    i = -1
    for x_i in x:
        i = i + 1
        y[i] = psi(x[i], t)
    print("Finish!"+\
           " Time:" + str(time.time() - start)+\
           " Total:" + str(time.time() - start_tot)
         )
    return y

#----------

x = np.arange(-5, 5, 0.001)

y0 = [0] * len(x)
i = -1
for x_i in x:
    i = i + 1
    if x_i <= 0.5 and x_i >= -0.5:
        y0[i] = (1/math.sqrt(2*a))
    else:
        y0[i] = 0

y1 = caculate(x,0.1)

y2 = caculate(x,0.2)
y3 = caculate(x,0.3)
y4 = caculate(x,0.4)
y5 = caculate(x,0.5)

plt.title("Free particle($\hbar$ = "+str(h)\
         +",m = "+str(m)\
         +",a = "+str(a)+")")

plt.xlabel("x")
plt.ylabel("$| \Psi (x) | ^ 2$")

plt.plot(x, y0, label=("t = 0"))
plt.plot(x, y1, label=("t = 0.1"))
plt.plot(x, y2, label=("t = 0.2"))
plt.plot(x, y3, label=("t = 0.3"))
plt.plot(x, y4, label=("t = 0.4"))
plt.plot(x, y5, label=("t = 0.5"))

plt.legend()
plt.show()

