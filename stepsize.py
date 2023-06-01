"""
The code below was written by @author: Diana Nitzschke (https://github.com/DianaNtz) and
is an implementation of the embedded Dormand-Prince Runge-Kutta Method with adaptive step size control
"""
import numpy as np
import matplotlib.pyplot as plt
#some initial values
t0=0
tfinal=10
steps=2**5
dt=(tfinal-t0)/(steps)
u0=0.0000001
a=6
#error tolerance
tol=10**(-8)
#differential equation function f
def f(t,u):
    return -(t-a)*u
#setting up arrays
t=np.empty(1, dtype='double')
t[0]=t0
tn=t0
u=np.empty(1, dtype='double')
u[0]=u0
un=u0
#time integration loop
while(tn<=10):    
    k1=dt*f(tn,un)
    k2=dt*f(tn+(1/5)*dt,un+(1/5)*k1)
    k3=dt*f(tn+(3/10)*dt,un+(3/40)*k1+(9/40)*k2)
    k4=dt*f(tn+(4/5)*dt,un+(44/45)*k1-(56/15)*k2+(32/9)*k3)
    k5=dt*f(tn+(8/9)*dt,un+(19372/6561)*k1-(25360/2187)*k2+(64448/6561)*k3-(212/729)*k4)
    k6=dt*f(tn+dt,un+(9017/3168)*k1-(355/33)*k2+(46732/5247)*k3+(49/176)*k4-(5103/18656)*k5)
    k7=dt*f(tn+dt,un+(35/384)*k1+(500/1113)*k3+(125/192)*k4-(2187/6784)*k5+(11/84)*k6)
    #order 5 coefficients
    b1=35/384
    b2=0
    b3=500/1113
    b4=125/192
    b5=-(2187/6784)
    b6=11/84
    b7=0
    #order 4 coefficients
    tildeb1=5179/57600
    tildeb2=0
    tildeb3=7571/16695
    tildeb4=393/640
    tildeb5=-(92097/339200)
    tildeb6=187/2100
    tildeb7=1/40
    err=(b1-tildeb1)*k1+(b2-tildeb2)*k2+(b3-tildeb3)*k3+(b4-tildeb4)*k4+(b5-tildeb5)*k5+(b6-tildeb6)*k6+(b7-tildeb7)*k7
    print(err)
    un=un+b1*k1+b2*k2+b3*k3+b4*k4+b5*k5+b6*k6+b7*k7
    tn=tn+dt
    t=np.append(t,tn)
    u=np.append(u,un)
    if(np.abs(err)>tol):
        dt=0.95*dt*(tol/np.abs(err))**(0.25)
    if(np.abs(err) <= tol):
        dt=0.95*dt*(tol/np.abs(err))**(0.2)
#analytical solution       
ua=u0*np.exp(-0.5*(t-a*2)*t)
#plotting analytical vs numerical solutions for order 5
ax1 = plt.subplots(1, sharex=True, figsize=(10,5))          
plt.plot(t,ua,color='black',linestyle='-',linewidth=3,label="$u_a(t)$")
plt.plot(t,u,color='deepskyblue',linestyle='-.',linewidth=3,label = "$u_n(t)$")
plt.xlabel("t",fontsize=19) 
plt.ylabel(r' ',fontsize=19,labelpad=20).set_rotation(0)
plt.ylim([0,8])
plt.xlim([t0,tfinal]) 
plt.xticks(fontsize= 17)
plt.yticks(fontsize= 17) 
plt.legend(loc=2,fontsize=19,handlelength=3) 
plt.savefig("stepsize.pdf")
plt.show()
