from scipy import *
import matplotlib.pyplot as plt


def force(x_prv,x,x_nxt,l,y,z):
   f=0.0
   f=l*((x_prv-x)+(x_nxt-x))+y*((-x_prv+x)**2+(x_nxt-x)**2)+z*((-x_prv+x)**3+(x_nxt-x)**3)
   return f


if __name__=='__main__':
   
   dt=0.1
   nmodes=10
   ak=0.0#weight for fourier mode
   ak_dot=0.0
   Niter=32110
   N=66+1
   
   Einit=0.0
   Efinal=0.0
   
   a=1 #spring constant of harmonic force
   b=1
   c=1 #spring constant of anharmonic force
   
   #instantiate veloities,displacements, its ghost and the force array

   k1=zeros(N,dtype=float)
   k2=zeros(N,dtype=float)
   k3=zeros(N,dtype=float)
   k4=zeros(N,dtype=float)
   
   ghost_1=zeros(N,dtype=float)
   ghost_2=zeros(N,dtype=float)
   ghost_3=zeros(N,dtype=float)

#   vel=zeros(N,dtype=float)
#   vel_ghost=zeros(N,dtype=float)
   disp=zeros(N,dtype=float)
   disp_ghost=disp
   foor1=zeros(N,dtype=float)
   foor2=zeros(N,dtype=float)
   spring=linspace(0,N,N)
   #storage array for mode distribution
   foorier=zeros(nmodes,dtype=float)
   mods=linspace(1,nmodes,nmodes)
#initialize the displacement

   for i in range(N):
      disp[i]=10*sin((pi*i)/(N-1)) #This is just pure sine mode
      disp_ghost=disp
      ghost_1=disp
      ghost_2=disp
      ghost_3=disp

   plt.plot(spring,disp)
   plt.show()
    
   for i in range(1,N-1):
       Einit+=0.5*a*((disp[i-1]-disp[i])**2+(disp[i+1]-disp[i])**2)+b*0.25*((disp[i-1]-disp[i])**4+(disp[i+1]-disp[i])**4)
   print(Einit) 
   for k in range(Niter):
       
      for i in range(1,N-1):
         k1[i]+=dt*force(disp_ghost[i-1],disp_ghost[i],disp[i+1],a,b,c)
      ghost_1=disp_ghost+0.5*k1
       
      for i in range(1,N-1):    
         k2[i]+=dt*force(ghost_1[i-1],ghost_1[i],ghost_1[i+1],a,b,c)
      ghost_2=disp_ghost+0.5*k2
   
      for i in range(1,N-1): 
         k3[i]+=dt*force(ghost_2[i-1],ghost_2[i],ghost_2[i+1],a,b,c)
      ghost_3=disp_ghost+k3
   
      for i in range(1,N-1):
         k4[i]+=dt*force(ghost_3[i-1],ghost_3[i],ghost_3[i+1],a,b,c)
       
      disp=disp_ghost+(1/6)*(k1+2*(k2+k3)+k4)*dt
      disp_ghost=disp
       
#      for i in range(1,N-1):
#         foor1[i]= force(disp_ghost[i-1],disp_ghost[i],disp[i+1],a,b)
#      for i in range(1,N-1):
#         disp[i]=disp_ghost[i]+vel_ghost[i]*dt+0.5*foor1[i]*(dt)**2
#      for i in range(1,N-1):   
#         foor2[i]=force(disp[i-1],disp[i],disp[i+1],a,b)
#      for i in range(1,N-1):
#         vel[i]=vel_ghost[i]+0.5*(foor1[i]+foor2[i])*dt
#      disp_ghost=disp
#      vel_ghost=vel
   
   for i in range(1,N-1):
      Efinal+=0.5*a*((disp[i-1]-disp[i])**2+(disp[i+1]-disp[i])**2)+b*0.25*((disp[i-1]-disp[i])**4+(disp[i+1]-disp[i])**4)+(0.5*((1/6)*(k1[i]+2*(k2[i]+k3[i])+k4[i]))**2)
   print(Efinal)
    
   for mode in range(1,nmodes):
      for i in range(N):
         ak+=disp[i]*sin((i*mode*pi)/N)
         ak_dot+=(1/6)*(k1[i]+2*(k2[i]+k3[i])+k4[i])*sin((i*mode*pi)/N)
      foorier[mode]=0.5*(ak_dot**2)+2*(ak**2)*(sin((mode*pi/(2*N)))**2)
      ak=0.0
      ak_dot=0.0
   plt.figure()
   plt.subplot(2,1,1)
   plt.plot(spring,disp)
   plt.subplot(2,1,2)
   plt.plot(mods,foorier)
   plt.show()
