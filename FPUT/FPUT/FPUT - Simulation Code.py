import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 32
delta = np.sqrt(1/8)
A = np.zeros((N-1, N-1))

for i in range(N-1):
    A[i][i] = 2
for i in range(N-2):
    A[i][i+1] = -1
    A[i+1][i] = -1
    
e1, e2 = np.linalg.eig(A)
sort = e1.argsort()
e1, e2 = e1[sort], e2[:,sort]

def Initialize():
    u, v = np.zeros(N+1), np.zeros(N+1)
    for i in range(1,N): u[i] = 4*e2[i-1][0]
    return u, v

def Ffunc(u, alpha):
    NF = np.zeros(N)
    NF[1] = u[2] - 2*u[1] + alpha*(u[2]-u[1])**2 - alpha*u[1]**2
    for i in range(2, N-1):
        NF[i] = u[i+1] - 2*u[i] + u[i-1] + alpha*(u[i+1]-u[i])**2 - alpha*(u[i]-u[i-1])**2
    NF[N-1] = -2*u[N-1] + u[N-2] + alpha*u[N-1]**2 - alpha*(u[N-1]-u[N-2])**2
    return NF

def Ufunc(o, u ,v):
    NU = np.zeros(N)
    for i in range(1, N):
       NU[i] = u[i] + v[i]*delta + 0.5*o[i]*delta**2
    return NU

def Vfunc(f, o ,v):
    NV = np.zeros(N)
    for i in range(1, N):
       NV[i] = v[i] + 0.5*delta*(f[i] + o[i])
    return NV

def Efunc(u, v, n):
    w = [e2[i][n] for i in range(N-1)]
    xi1 = np.dot(u[1:N], w)
    xi2 = np.dot(v[1:N], w)
    return 0.5*(xi2**2+e1[n]*xi1**2)
    
def AnimateOscillation(alpha, NrFr):
    global F, O, U, V, X
    
    U, V = Initialize()
    O = Ffunc(U, alpha)     

    fig, ax = plt.subplots(figsize=(19.20, 7.00))
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 32)
    ax.set_xticks([i for i in range(N+1)])
    ax.set_xlabel("Assigned mass number")
    ax.set_ylabel("Equilibrium point displacement")
    X = [i+1 for i in range(N-1)]
    l, = plt.plot(X, U[1:N], marker="o", markersize=10, linestyle="None", color="black")
    plt.grid()
    
    def aloop(t):
        global F, O, U, V, X
        print(f"Frame: {t}")
        
        ax.set_title(f"Time = {round(t*delta*4)}")
        l.set_data(X, U[1:N])
    
        for i in range(4):
            U = Ufunc(O, U, V)
            F = Ffunc(U, alpha)
            V = Vfunc(F, O, V)
            O = F
        
    anim = FuncAnimation(fig, aloop, frames = NrFr, interval = 1, repeat=False)
    anim.save("Oscillation.gif", fps = 30)
    
def AnimatePlot(alpha, NrFr):
    global F, O, U, V, X
    
    U, V = Initialize()
    O = Ffunc(U, alpha)     

    PlotList = [[0], [[0], [0], [0], [0], [0], [0]]]
    fig, ax = plt.subplots(figsize=(19.20, 3.60))
    ax.set_ylim(-0.0015, 0.08)
    ax.set_xlim(0, 5000)
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    
    l0, = plt.plot(PlotList[0], PlotList[1][0], label="$E_0$")
    l1, = plt.plot(PlotList[0], PlotList[1][1], label="$E_1$")
    l2, = plt.plot(PlotList[0], PlotList[1][2], label="$E_2$")
    l3, = plt.plot(PlotList[0], PlotList[1][3], label="$E_3$")
    l4, = plt.plot(PlotList[0], PlotList[1][4], label="$E_4$")
    lines = [l0, l1, l2, l3, l4]
    
    plt.grid()
    plt.legend()
    
    def aloop(t):
        global F, O, U, V, X
        print(f"Frame: {t}")
        
        PlotList[0].append(t*delta*4)
        for l, i in zip(lines, range(5)):
            PlotList[1][i].append(Efunc(U, V, i))
            l.set_data(PlotList[0], PlotList[1][i])
    
        for i in range(4):
            U = Ufunc(O, U, V)
            F = Ffunc(U, alpha)
            V = Vfunc(F, O, V)
            O = F
        
    anim = FuncAnimation(fig, aloop, frames = NrFr, interval = 1, repeat=False)
    anim.save("Plot.gif", fps = 30)
    
AnimateOscillation(1, 3540)
AnimatePlot(1, 3540)

