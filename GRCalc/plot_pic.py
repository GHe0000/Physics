import numpy as np
import matplotlib.pyplot as plt

sol1 = np.load("sol11.npy")
sol2 = np.load("sol21.npy")
sol3 = np.load("sol31.npy")

def func(sol):
    r = sol[1]
    theta = sol[3]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x,y

x1,y1=func(sol1)
x2,y2=func(sol2)
x3,y3=func(sol3)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

plt.subplot(1,3,1)
plt.plot(x2,y2)
plt.plot(0,0,marker="o",color="coral")
plt.gca().add_artist(plt.Circle((0, 0), 2,fill=False))
plt.gca().set_aspect(1)
plt.title(r"$v=\frac{1}{2}-0.001$")

plt.subplot(1,3,2)
plt.plot(x1,y1)
plt.plot(0,0,marker="o",color="coral")
plt.gca().add_artist(plt.Circle((0, 0), 2,fill=False))
plt.gca().set_aspect(1)
plt.title(r"$v=\frac{1}{2}$")

plt.subplot(1,3,3)
plt.plot(x3,y3)
plt.plot(0,0,marker="o",color="coral")
plt.gca().add_artist(plt.Circle((0, 0), 2,fill=False))
plt.gca().set_aspect(1)
plt.title(r"$v=\frac{1}{2}$+0.001")

plt.suptitle(r"位于大于$3 R_s$的稳定轨道"+"\n"\
             +r"$R_s=2$、$R=7$、恰圆周运动速度$v_c = \frac{1}{2}$")

plt.show()
