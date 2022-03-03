import numpy as np
import random
import matplotlib.pyplot as plt
import math

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

def deriv(f,t,param): # return derivatives of the array f[phi,dphi/dt,theta,dtheta/dt]
    phi = f[0]
    dphidt = f[1]
    gammaD = param[0]
    
    #dphi/dt
    dphidt = dphidt 
    #d2phi/dt2 

    #With driving
    # d2phidt2 = gammaD*omega0**2*np.cos(omegaD*t) - 2*betaD*dphidt - omega0**2*np.sin(phi)
    #Without
    d2phidt2 = - 2*taylor(phi)*dphidt - omega0**2*np.sin(phi)
    
    return [dphidt,d2phidt2]


ncoef = 4
omegaD = 2*pi
omega0 = 1.5*omegaD
betaD = omega0/4
c = [random.uniform(0,1.5)]
while len(c) < ncoef:
    rand = random.uniform(-1.5,1.5)
    c.append(rand)

#Define the time array
max_time = 10
n_steps = 10000
time = linspace(0.0,max_time,n_steps)

yinit = [0,0] # initial values of phi0 and dphi0, respectively
param = np.array([0.2]) #Setting the value of gammaD in the param array so that it can be passed to the deriv function
f_solun = odeint(deriv,yinit,time,args=(param,))

figure()
plot(time, f_solun[:,0])
xlabel('Time [s]')
ylabel('Phi')
show()
#/