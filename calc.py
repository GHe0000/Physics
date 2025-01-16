import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

m = 0.1 * 10**-3
R = 5*10**-2
g = 9.8
mu0 = 4*np.pi*10**-7
I = 1000
S = 0.1 * 10**-4
r = 10**-3
d = 0.1 * 10**-2

def ode(y, t):
    a, da = y
    daN = da
    ddaN = -g/R*np.sin(a) - (mu0**2 * I**2 * R * S**2)/(4*(np.pi**2)*r*m*R) * da * (np.sin(a)**2)/(d+R*np.cos(a))**4
    return [daN, ddaN]

t = np.linspace(0, 2*10**6, 2*10**8)
y0 = [-np.pi/2, 0]

# 记录计算时间
t1 = time.time()
sol = odeint(ode, y0, t)
t2 = time.time()

print('计算时间：', t2-t1)

# 保存数据（用压缩格式保存）
# np.savez_compressed('data.npz', t=t, sol=sol)

plt.plot(t, sol[:,0])
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')
plt.show()
