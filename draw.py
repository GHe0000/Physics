import numpy as np
from matplotlib import pyplot as plt

data = np.load('data200.npz')
t = data['t']
sol = data['sol']

plt.plot(t, sol[:,0])
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')
plt.show()

t = t[:100]
sol = sol[:100]

plt.plot(t, sol[:,0])
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')
plt.show()
