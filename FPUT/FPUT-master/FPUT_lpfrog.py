from scipy import *
import matplotlib.pyplot as plt


def force(x_prv,x,x_nxt,m,n,p):
   f=m*((x_nxt-x)-(x-x_prv))+n*((x_nxt-x)**2-(x-x_prv)**2)+p*((x_nxt-x)**3-(x-x_prv)**3)
   return f


if __name__=='__main__':
   
   dt=0.15
   nmodes=11
   ak=0.0#weight for fourier mode
   Niter=6513114
   N=102
   a=2 #spring constant of harmonic force
   b=0.0 #spring constant of phi-3 force
   c=77. #spring constant of phi-4 force
   E=0.0

   #instantiate k1,k2,k3,k4 these are the RK4 "ghost" displacements

   k1=zeros(N,dtype=float)
   k2=zeros(N,dtype=float)
   k3=zeros(N,dtype=float)
   k4=zeros(N,dtype=float)

   v1=zeros(N,dtype=float)
   v2=zeros(N,dtype=float)
   v3=zeros(N,dtype=float)
   v4=zeros(N,dtype=float)


  #placeholder and actual string displacements

   disp=zeros(N,dtype=float)
   disp_ghost=disp

   ghost_1=zeros(N,dtype=float)
   ghost_2=zeros(N,dtype=float)
   ghost_3=zeros(N,dtype=float)


   #position index
   pos_ind=linspace(0,N,N)

   #list of modes
   modes=linspace(1,nmodes,nmodes)
  
   #alternative just to track first given Fourier  modes
   fr=100                            #proportaion to time lapse per frame 
   Ntrak=Niter//fr                    #number of frames
   ttrack=linspace(0,Ntrak,Ntrak)    #time index
   foorier_sim=zeros((nmodes,Ntrak),dtype=float)      
   trek=0
   
   #running scalar for mode distribution
   ak=0.0
   ak_dot=0.0

   #initialize the displacement
   
   for i in range(N):
      disp[i]=sin((pi*i)/(N-1)) #This is just pure sine mode
      disp_ghost[i]=disp[i]
   print(disp_ghost)
   for i in range(1,N-1):
      E+=0.5*(v3[i])**2+0.5*a*((disp[i+1]-disp[i])**2+(disp[i-1]-disp[i])**2)+b*(1/3)*((disp[i]-disp[i-1])**3+(disp[i+1]-disp[i])**3)+c*0.25*((disp[i]-disp[i-1])**4+(disp[i+1]-disp[i])**4)
   print(E)

   for k in range(Niter):
#      if k%(Niter//100) ==0:
#         c*=2
      if k==0:
         for i in range(1,N-1):
            v3[i]+=0.5*dt*force(disp[i-1],disp[i],disp[i+1],a,b,c)
      for i in range(1,N-1):
         disp[i]+=dt*v3[i]

      for i in range(1,N-1):
         v3[i]+=dt*force(disp[i-1],disp[i],disp[i+1],a,b,c)
    

   #update the velocities
#      for i in range(1,N-1):
#         v1[i]=v3[i]+force(disp_ghost[i-1],disp_ghost[i],disp_ghost[i+1],a,b,c)*dt

#      for i in range(1,N-1):
#         v2[i]=v3[i]+dt*force(ghost_1[i-1],ghost_1[i],ghost_1[i+1],a,b,c)

   
#      v3=(0.5)*(v1+v2)

         
      disp_ghost=disp
      if k%fr==0 and k>0:
         for freq in range(1,nmodes):
            for i in range(N):
               ak+=disp[i]*sin(i*freq*pi/N)
               ak_dot+=v3[i]*sin(i*freq*pi/N)
            foorier_sim[freq][trek]=0.5*ak_dot**2+2*ak**2*sin(freq*pi/(2*N))**2
            ak=0.0
            ak_dot=0.0
         trek+=1

   E=0.0
   for i in range(1,N-1): 
      E+=0.5*v3[i]**2+a*0.5*((disp[i+1]-disp[i])**2+(disp[i-1]-disp[i])**2)+b*(1/3)*((-disp[i-1]+disp[i])**3+(disp[i+1]-disp[i])**3)+c*0.25*((-disp[i-1]+disp[i])**4+(disp[i+1]-disp[i])**4)
   print(E)
   plt.figure()
   plt.subplot(2,1,1)
   plt.plot(pos_ind,disp)
   plt.subplot(2,1,2)
   for freq in range(nmodes):
      plt.plot(ttrack,foorier_sim[freq][:])
   plt.show()
