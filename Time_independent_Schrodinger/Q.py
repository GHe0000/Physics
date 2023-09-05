import numpy as np
import scipy

import matplotlib.pyplot as plt

# 设定常数
hbar = 1
m = 1

x_start = -10
x_finish = 10

n = 200

dx = (x_finish - x_start) / n
En = 9

x = np.linspace(x_start, x_finish, 200)
V = 1/2 * (x ** 2)
v = np.diag(V)

# Hermitian矩阵
D = np.ones(n)
A = np.diag(-2 * D) + np.diag(D[1:],-1) + np.diag(D[1:],1)
A = (-(hbar**2)/(2*m))*(1/(dx**2))*A
H = A + V

Val,Vec = scipy.sparse.linalg.eigs(H, En)

import pdb
#pdb.set_trace()

Phi_1 = (((m*1)/(np.pi*hbar))**(1/4)*np.sqrt((2*m*1)/(hbar)))*x*np.exp(-(m*1)/(2*hbar) * x**2)
Phi_1 = Phi_1 ** 2
plt.plot(x,np.abs(Vec[:,1]**2)/np.sum(np.abs(Vec[:,1]**2)*dx))
plt.plot(x,Phi_1)
plt.show()
