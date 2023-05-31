"""
The code below was written by @author: Diana Nitzschke and
is an implementation of dense Runge-Kutta methods for the Dormand–Prince
embedded Runge-Kutta method of order 4 and 5.
"""
import numpy as np
import matplotlib.pyplot as plt
#differential equation function f
def f(t,u):
    return -(t-6)*u
#Dormand–Prince embedded Runge-Kutta method of order 4 and 5
def RungeKutta(t0,tfinal,steps,u0,string):
    dt=(tfinal-t0)/(steps)
    t=np.zeros(steps+1, dtype='double')
    tn=t0
    u=np.zeros(steps+1, dtype='double')
    un=u0
    n=0
    if(string=="DorPr5" or string=="DorPr4"):
        n=7
        tcoff=np.array([0,1/5,3/10,4/5,8/9,1,1])
        c=np.array([[0,0,0,0,0,0],[1/5,0,0,0,0,0],[3/40,9/40,0,0,0,0],[44/45,-56/15,32/9,0,0,0],
                    [19372/6561,-(25360/2187),(64448/6561),-(212/729),0,0],
                    [(9017/3168),-(355/33),(46732/5247),(49/176),-(5103/18656),0],
                    [(35/384),0,(500/1113),(125/192),-(2187/6784),(11/84)]])
        if(string=="DorPr5"):
            b=np.array([35/384,0,500/1113,125/192,-(2187/6784),11/84,0])
        if(string=="DorPr4"):
            b=np.array([5179/57600,0,7571/16695,393/640,-(92097/339200),187/2100,1/40])        
    k=np.zeros([n, steps+1])
    for i in range(0,steps+1):    
        t[i]=tn     
        u[i]=un
        for j in range(0,n):           
            if(j==0):
                k[j][i]=dt*f(tn+tcoff[0]*dt,un)
            else:
                bb=c[j][0:j]
                bbb=0
                for l in range(0,len(bb)):
                    bbb=bbb+bb[l]*k[l][i]
                k[j][i]=dt*f(tn+tcoff[j]*dt,un+bbb)
        
        bbb=0
        for g in range(0,n):
            bbb=bbb+b[g]*k[g][i]
        un=un+bbb
        tn=tn+dt
    return t,u,k,string   