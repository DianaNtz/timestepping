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