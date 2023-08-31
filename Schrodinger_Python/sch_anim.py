import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation


# 数据导入
sol = np.load("./sol.npz")
x = sol["x"]
V = sol["V"]
sol_y = sol["sol_y"]
sol_t = sol["sol_t"]

# 动画
fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot(2, 1, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(-1, 3)
title = ax1.set_title('')
line11 = ax1.plot([], [], "k--", label=r"$V(x) \times 0.001$")[0]
line12 = ax1.plot([], [], "b", label=r"$\vert \psi \vert^2$")[0]
plt.legend(loc=1, fontsize=8, fancybox=False)

ax2 = plt.subplot(2, 1, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(-2, 2)
line21 = ax2.plot([], [], "k--", label=r"$V(x) \times 0.001$")[0]
line22 = ax2.plot([], [], "r", label=r"$Re\{ \psi \}$")[0]
plt.legend(loc=1, fontsize=8, fancybox=False)

update_objects = [line11,line12,line21,line22]

def init():
    for i in update_objects:
        i.set_data([],[])
    return update_objects

def update(frames):
    update_objects[0].set_data(x, V * 0.001)
    update_objects[2].set_data(x, V * 0.001)
    update_objects[1].set_data(x, np.abs(sol_y[:, frames])**2)
    update_objects[3].set_data(x, np.real(sol_y[:, frames]))
    return update_objects

anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(sol_t), interval=100, blit=True)
plt.show()
