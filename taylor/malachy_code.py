import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint

def taylor(x,c):
    ts = [x**n/math.factorial(n) for n in range(ncoef)]
    return np.dot(np.array(c),np.array(ts))

def fourier(x,c):
    fs = [np.sin(n*x/distance) for n in range(ncoef)]
    # fs = [1]
    # n=1
    # while len(fs) < ncoef:
    #     fs.append(np.sin(n*x/distance))
    #     if len(fs) < ncoef:
    #         fs.append(np.cos(n*x/distance))
    #     n+=1
    return np.dot(np.array(c),np.array(fs))

def deriv(f,t,param): # return derivatives of the array f[x,dx/dt,theta,dtheta/dt]
    x = f[0]
    dxdt = f[1]

    #With driving
    # d2xdt2 = gammaD*omega0**2*np.cos(omegaD*t) - 2*betaD*dxdt - omega0**2*np.sin(x)
    #Without
    d2xdt2 = - taylor(x, c)/m*dxdt - k*x/m
    
    return [dxdt,d2xdt2]


ncoef = 4
k = 1
m = 1
c = [random.uniform(0,1.5)]
while len(c) < ncoef:
    rand = random.uniform(-1.5,1.5)
    c.append(rand)

#Define the time array
max_time = 10
n_steps = 10000
time = np.linspace(0.0,max_time,n_steps)

yinit = [0,0] # initial values of x0 and dx0, respectively
param = np.array([0.2]) #Setting the value of gammaD in the param array so that it can be passed to the deriv function
f_solun = odeint(deriv,yinit,time,args=(param,))

plt.figure()
plt.plot(time, f_solun[:,0])
plt.xlabel('Time [s]')
plt.ylabel('x')
plt.show()