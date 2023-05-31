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
#dense Runge-Kutta method 
def interpolate(t,u,k,sigma,string,c8=2/5,gamma=-1/30000,a86=1/20):
    dt=t[1]-t[0]
    tneu=t+sigma*dt
    if(string=="DorPr5" or string=="DorPr4"):
        sig=np.array([1/2*sigma, 1/3*sigma**2,1/4*sigma**3,1/5*sigma**4, 1/20*sigma**4-gamma,0,0,0])
        A=np.array([[ 3/10, 4/5, 8/9, 1, 1, c8, 0, 0], 
                    [ (3/10)**2, (4/5)**2, (8/9)**2, 1, 1, c8**2, 0, 0], 
                    [ (3/10)**3, (4/5)**3, (8/9)**3, 1, 1, c8**3, 0, 0],
                    [ (3/10)**4, (4/5)**4, (8/9)**4, 1, 1, c8**4, 0, 0],
                    [(1/5)**3*(9/40), -(1/5)**3*(56/15)+(3/10)**3*(32/9), -(1/5)**3*(25360/2187)+(3/10)**3*(64448/6561)-(4/5)**3*(212/729), -(1/5)**3*(355/33)+(3/10)**3*(46732/5247)+(4/5)**3*(49/176)-(8/9)**3*(5103/18656), (3/10)**3*(500/1113)+(4/5)**3*(125/192)-(8/9)**3*(2187/6784)+11/84, 0, 0, 0],
                    [0, 32/9*(9/40), (64448/6561)*(9/40)+(212/729)*(56/15), (46732/5247)*(9/40)-(49/176)*(56/15)+(5103/18656)*(25360/2187), (500/1113)*(9/40)-(125/192)*(56/15)+(2187/6784)*(25360/2187)-(11/84)*(355/33), 0, -1, 0],
                    [9/40*(3/10), -56/15*(4/5), -25360/2187*(8/9),-355/33, 0, 0, 0, -c8],
                    [9/40, -56/15, -25360/2187,-355/33, 0, 0, 0, -1]
                    ])
        Ainv=np.linalg.inv(A)
        vb8=Ainv.dot(sig)
        bstrich8=vb8[-3]
        gamma1=vb8[-2]
        gamma2=vb8[-1]
        vec=np.array([0.5*c8**2-a86,(1/3)*c8**3-a86,gamma/bstrich8-a86,-gamma1/bstrich8-a86*(-355/33), -gamma2/bstrich8])
        C=np.array([[1/5,3/10, 4/5, 8/9, 1],
                    [(1/5)**2,(3/10)**2, (4/5)**2, (8/9)**2, 1],
                    [(1/5)**3,(3/10)**3, (4/5)**3, (8/9)**3, 1],
                    [0,9/40, -56/15, -25360/2187, 0],
                    [1, 0, 0, 0, 0]
            ])
        Cinv=np.linalg.inv(C)
        va8=Cinv.dot(vec)
        b1strich=1-vb8[0]-vb8[1]-vb8[2]-vb8[3]-vb8[4]-vb8[5]
        a81=c8-np.sum(va8)-a86
        avec8=np.array([a81,va8[0],va8[1],va8[2],va8[3],a86,va8[4]])
        bstrichvec=np.array([b1strich,0,vb8[0],vb8[1],vb8[2],vb8[3],vb8[4],vb8[5]])
        k8=dt*f(t+c8*dt,u+avec8[0]*k[0]+avec8[1]*k[1]+avec8[2]*k[2]+avec8[3]*k[3]+avec8[4]*k[4]+avec8[5]*k[5]+avec8[6]*k[6])
        uneu=u+(bstrichvec[0]*k[0]+bstrichvec[1]*k[1]+bstrichvec[2]*k[2]+bstrichvec[3]*k[3]+bstrichvec[4]*k[4]+bstrichvec[5]*k[5]+bstrichvec[6]*k[6]+bstrichvec[7]*k8)*sigma

    return tneu,uneu  