import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
# import sys
# from PyQt5.QtWidgets import QApplication
from matplotlib.animation import FuncAnimation
# import pyautogui
from celluloid import Camera

def taylor(x,c):
    ts = [x**n/math.factorial(n) for n in range(ncoef)]
    return np.dot(np.array(c),np.array(ts))

def fourier(x,c):
    fs = [1]
    L = 1
    n=1
    while len(fs) < ncoef:
        fs.append(np.sin(n*x/L))
        if len(fs) < ncoef:
            fs.append(np.cos(n*x/L))
        n+=1
    return np.dot(np.array(c),np.array(fs))
  
def make_coeffs(c,lb,ub):
    while len(c) < ncoef:
        c.append(random.uniform(lb,ub))
    return c

def k(t,x):
    return taylor(x,ck)

def m(t,x):
    return np.exp(-.1*t)

def b(t,x):
    return taylor(x,cb)

def deriv(f,t): # return derivatives of the array f[x,dx/dt,theta,dtheta/dt]
    x, v = f[0], f[1]
    linear_matrix = np.array([[0,1],[- k(t,x)/m(t,x), - b(t,x)/m(t,x)]])
    return np.dot(linear_matrix, f)


ncoef = 4
cm,cb,ck = [], [], []
cm = make_coeffs(cm,0,1)
cb = make_coeffs(cb,0,1)
ck = make_coeffs(ck,0,1)

#Define the time array and other plotting stuff
max_time = 10
n_steps = 10000

time = np.linspace(0.0,max_time,n_steps)
space = np.linspace(-1,1,n_steps)

m_func = [m(t,0) for t in time]
b_func = [b(0,x) for x in space]
k_func = [k(0,x) for x in space]

yinit = [1,0] # initial values of x0 and dx0, respectively
f_solun = odeint(deriv,yinit,time)
equilibrium = [0 for n in range(len(time))]

# desired_sizex, desired_sizey = 800, 800
# app = QApplication(sys.argv)
# screen = app.screens()[0]
# my_dpi = screen.physicalDotsPerInch()
# app.quit()
# width, height = pyautogui.size()
# print(width, height)
# print(my_dpi)


plt.figure()
plt.plot(time, f_solun[:,0])
plt.plot(time, equilibrium)

plt.xlabel('Time [s]')
plt.ylabel('x')
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(12, 12))

camera = Camera(fig)

for i in range(0, n_steps, 50):
    axs[0, 0].plot(f_solun[:,0][:i], time[:i])
    axs[0, 0].plot(equilibrium[:i], time[:i])
    axs[0, 0].set_title('System Position Solution')
    axs[0, 0].set(xlabel='Position', ylabel='Time')

    axs[0, 1].plot(f_solun[:,1][:i], time[:i])
    axs[0, 1].plot(equilibrium[:i], time[:i])
    axs[0, 1].set_title('System Velocity Solution')
    axs[0, 1].set(xlabel='Velocity', ylabel='Time')

    axs[0, 2].plot(f_solun[:,1][:i], time[:i])
    axs[0, 2].plot(equilibrium[:i], time[:i])
    axs[0, 2].set_title('System Solution')
    axs[0, 2].set(xlabel='Velocity', ylabel='Time')

    axs[1, 0].plot(time[:i], m_func[:i])
    axs[1, 0].set_title('Mass Function')
    axs[1, 0].set(xlabel='Time', ylabel='Coefficient Value')

    axs[1, 1].plot(space[:i], b_func[:i])
    axs[1, 1].plot(equilibrium[:i],space[:i])
    axs[1, 1].set_title('Drag Function')
    axs[1, 1].set(xlabel='Position', ylabel='Coefficient Value')

    axs[1, 2].plot(space[:i], k_func[:i])
    axs[1, 2].plot(equilibrium[:i],space[:i])
    axs[1, 2].set_title('k Function')
    axs[1, 2].set(xlabel='Position', ylabel='Coefficient Value')
    print("Snap " + str(i))
    camera.snap()
print("Ready to capture")
animation = camera.animate()
animation.save('test_animation.mp4')