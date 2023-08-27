import numpy as np
import scipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# 定义常量
G=6.67408e-11

m_nd=1.989e+30
r_nd=5.326e+12
v_nd=30000
t_nd=79.91*365*24*3600*0.51

K1=G*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd

m1=1.1
m2=0.907
m3=1.0

r1=np.array([-0.5,0,0])
r2=np.array([0.5,0,0])
r3=np.array([0,1,0])

v1=np.array([0.01,0.01,0])
v2=np.array([-0.05,0,-0.1])
v3=np.array([0,-0.01,0])


# 待求解微分方程
def equation(w,t,G,m1,m2,m3):
    r1=w[:3]
    r2=w[3:6]
    r3=w[6:9]

    v1=w[9:12]
    v2=w[12:15]
    v3=w[15:18]

    r12=np.linalg.norm(r2-r1)
    r13=np.linalg.norm(r3-r1)
    r23=np.linalg.norm(r3-r2)
   
    dv1bydt=K1*m2*(r2-r1)/r12**3+K1*m3*(r3-r1)/r13**3
    dv2bydt=K1*m1*(r1-r2)/r12**3+K1*m3*(r3-r2)/r23**3
    dv3bydt=K1*m1*(r1-r3)/r13**3+K1*m2*(r2-r3)/r23**3
    dr1bydt=K2*v1
    dr2bydt=K2*v2
    dr3bydt=K2*v3

    r12_derivs=np.concatenate((dr1bydt,dr2bydt))
    r_derivs=np.concatenate((r12_derivs,dr3bydt))
    v12_derivs=np.concatenate((dv1bydt,dv2bydt))
    v_derivs=np.concatenate((v12_derivs,dv3bydt))
    derivs=np.concatenate((r_derivs,v_derivs))
    return derivs

init_params=np.concatenate((r1,r2,r3,v1,v2,v3),axis=0)
time_span=np.linspace(0,20,500)

sol=scipy.integrate.odeint(equation,init_params,time_span,args=(G,m1,m2,m3))

r1_sol=sol[:,:3]
r2_sol=sol[:,3:6]
r3_sol=sol[:,6:9]

fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
'''
line1 = ax.plot([],[],[],color="red")[0]
line2 = ax.plot([],[],[],color="blue")[0]
line3 = ax.plot([],[],[],color="green")[0]

update_objects = [line1,line2,line3]

ax.set_xlim(-2.5,1)
ax.set_ylim(0,1.2)
ax.set_zlim(-4,1)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Three Body Problem")
#ax.legend()

# 动画初始化
def init():
    for i in update_objects[0:3]:
        i.set_data([],[])
        i.set_3d_properties([])
    return update_objects

# 更新函数
def update(frames):
    update_objects[0].set_data(r1_sol[:frames,:2].T)
    update_objects[0].set_3d_properties(r1_sol[:frames,2])
    update_objects[1].set_data(r2_sol[:frames,:2].T)
    update_objects[1].set_3d_properties(r2_sol[:frames,2])
    update_objects[2].set_data(r3_sol[:frames,:2].T)
    update_objects[2].set_3d_properties(r3_sol[:frames,2])
    return update_objects

anim = animation.FuncAnimation(fig,update,init_func=init,frames=500,interval=100)
'''
#anim.save('ThreeBodyProblem.mp4',writer='ffmpeg',dpi=300)
#plt.show()

#'''
ax.plot(r1_sol[:,0],r1_sol[:,1],r1_sol[:,2],color="red")
ax.plot(r2_sol[:,0],r2_sol[:,1],r2_sol[:,2],color="blue")
ax.plot(r3_sol[:,0],r3_sol[:,1],r3_sol[:,2],color="green")

ax.scatter(r1_sol[-1,0],r1_sol[-1,1],r1_sol[-1,2],color="red",marker="o",s=10,label="A")
ax.scatter(r2_sol[-1,0],r2_sol[-1,1],r2_sol[-1,2],color="blue",marker="o",s=10,label="B")
ax.scatter(r3_sol[-1,0],r3_sol[-1,1],r3_sol[-1,2],color="green",marker="o",s=10,label="C")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Three Body Problem")
ax.legend()

plt.show()
#'''
