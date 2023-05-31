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
#plotting and error calculation
nende=13
nstart=7
n=np.zeros(nende-nstart+1, dtype='double')
error=np.zeros(nende-nstart+1, dtype='double')
errorinter=np.zeros(nende-nstart+1, dtype='double')
method=["DorPr4","DorPr5"]
string=""
for j in range(0,len(method)):
        #print(method[j])
        string=string+method[j]+":"+"\n"
        for i in range(nstart,nende+1):
                t0=0
                tfinal=10
                steps=2**i
                u0=0.0000001
                
                t,u,k,s= RungeKutta(t0,tfinal,steps,u0,method[j])
                ua=u0*np.exp(-0.5*(t-6*2)*t)
                
                ax1 = plt.subplots(1, sharex=True, figsize=(10,5))          
                plt.plot(t,ua,color='black',linestyle='-',linewidth=3,label="$u_a(t)$")
                plt.plot(t,u,color='deepskyblue',linestyle='-.',linewidth=3,label = "$u_n(t)$")
                plt.xlabel("t",fontsize=19) 
                plt.ylabel(r' ',fontsize=19,labelpad=20).set_rotation(0)
                plt.ylim([0,np.max(u)])
                plt.xlim([t0,tfinal]) 
                plt.xticks(fontsize= 17)
                plt.yticks(fontsize= 17) 
                plt.title(s+" n={0:.0f}".format(2**i),fontsize=22)
                plt.legend(loc=2,fontsize=19,handlelength=3) 
                plt.show()
                
                dt=(tfinal-t0)/(steps)
                err=np.max(np.abs(ua[:steps]-u[:steps]))
                string=string+" n=2^"+str(i)+" Error: "+str(err)+"\n"
                error[i-nstart]=err
                
                t1,u1=interpolate(t,u,k,0.2,s)
                ua1=u0*np.exp(-0.5*(t1-6*2)*t1)
                
                ax1 = plt.subplots(1, sharex=True, figsize=(10,5))          
                plt.plot(t1,ua1,color='black',linestyle='-',linewidth=3,label="$u_a(t)$")
                plt.plot(t1,u1,color='deepskyblue',linestyle='-.',linewidth=3,label = "$u_{inter}(t)$")
                plt.xlabel("t",fontsize=19) 
                plt.ylabel(r' ',fontsize=19,labelpad=20).set_rotation(0)
                plt.ylim([0,np.max(u1)])
                plt.xlim([t1[0],t1[-1]]) 
                plt.xticks(fontsize= 17)
                plt.yticks(fontsize= 17) 
                plt.title(s+" interpolated n={0:.0f}".format(2**i),fontsize=22)
                plt.legend(loc=2,fontsize=19,handlelength=3) 
                plt.show()
                
                errinter=np.max(np.abs(ua1[:steps]-u1[:steps]))
                errorinter[i-nstart]=errinter
                string=string+" n=2^"+str(i)+" Error: "+str(errinter)+" (interpolated)"+"\n"
                n[i-nstart]=2**i
        filename='data/method_'+s+'_error.npz'
        np.savez(filename, n_loaded=n, error_loaded=error)
        filename='data/method_'+s+'_errorinter.npz'
        np.savez(filename, n_loaded=n, errorinter_loaded=errorinter)
        ax1 = plt.subplots(1, sharex=True, figsize=(6,5))  
        plt.xscale('log', base=2) 
        plt.yscale('log', base=10) 
        plt.plot(n,error,color='black',linestyle='-',linewidth=3,label = "non-interpolated")
        plt.plot(n,errorinter,color='blue',linestyle=':',linewidth=3,label = "interpolated")
        plt.xlabel("n",fontsize=19) 
        plt.ylabel('Error',fontsize=19)
        plt.ylim([np.min(error),np.max(error)])
        plt.xlim([n[0],n[-1]]) 
        plt.xticks(n,fontsize= 17)
        plt.yticks(fontsize= 17) 
        plt.legend(loc=1,fontsize=19,handlelength=3) 
        plt.title("Error "+s+" ",fontsize=22)
        plt.savefig("figures/method_"+s+"_error.pdf")
        plt.show()
        
        if(j!=len(method)-1):
            string=string+"\n"
print(string)
file = open("Errors.txt", "w")
file.write(string)
file.close()
#convergence order calculation
string=""
for j in range(0,len(method)):
    
    s=method[j]
    filename='data/method_'+s+'_error.npz'
    error_data=np.load(filename)
    error=error_data["error_loaded"]
    n=error_data["n_loaded"]
    filename='data/method_'+s+'_errorinter.npz'
    errorinter_data=np.load(filename)
    error_inter=errorinter_data["errorinter_loaded"]
   
    for i in range(len(n)-2,len(n)-1):
        c1=error[i]/error[i+1]
        c2=error_inter[i]/error_inter[i+1]
        string=string+"2^p"+" for "+s+": "+str(c1)+"\n"+"2^p for "+s+": "+str(c2)+"(interpolated)"+"\n"
        if(j!=len(method)-1):
            string=string+"\n"
print(string)
file = open("Convergence.txt", "w")
file.write(string)
file.close()