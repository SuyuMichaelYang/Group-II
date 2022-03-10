import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
from mpl_toolkits import mplot3d
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

def sharkTooth(z):
    teeth = []
    w = 1
    A = .4
    y0 = .5
    n = 1
    while n <= 10:
        teeth.append( -((-1)**(n+1)/(1.707*n)) * np.sin(2*w*n*z))
        n+=1
    return y0 + A* np.sum(teeth)

def k(t,x,v):
    return 1

def m(t,x,v):
    M =  .3*np.cos(.05*t**2)**2 + .1#np.cos(.1*t**2)**2 + .1 #np.cos(.5*t)**2 + .1
    if M <= 0:
        M = 10**(-10)
    return M
def b(t,x):
    return  .05

def deriv(f,t): # return derivatives of the array f[x,dx/dt,theta,dtheta/dt]
    x, v = f[0], f[1]
    M,B,K = m(t,x,v), b(t,x), k(t,x,v)
    linear_matrix = np.array([[0,1],[- K/M, - B/M]])
    phase_space.append([x,M*v])
    time_list.append(t)
    mass_list.append(M)
    return np.dot(linear_matrix, f)


ncoef = 4
cm,cb,ck = [], [], []
cm = make_coeffs(cm,0,1)
cb = make_coeffs(cb,0,1)
ck = make_coeffs(ck,0,1)

#Define the time array and other plotting stuff
max_time = 50
n_steps = 100000
time = np.linspace(0.0,max_time,n_steps)
space = np.linspace(-1,1,n_steps)
phase_spaces, masses_list, times_list, radii = [], [], [], []

f_soluns = []
ICs = [[1,0],[.9,0], [.95,0],[1,.1]]
for trial in ICs:
    phase_space, mass_list, time_list, radius = [], [], [], []
    f_soluns.append(odeint(deriv,trial,time))
    phase_spaces.append(phase_space)
    for vector in phase_space:
        radius.append(np.linalg.norm(vector))
    radii.append(radius)
    masses_list.append(mass_list)
    times_list.append(time_list)

mt_func = [m(t,0,0) for t in time]
b_func = [b(0,x) for x in space]
k_func = [k(0,x,0) for x in space]
equilibrium = [0 for n in range(len(time))]

def subplots():

    '''plt.figure()
    plt.plot(time, f_soluns[0][:,0])
    plt.plot(time, equilibrium)
    plt.xlabel('Time [s]')
    plt.ylabel('x')
    plt.show()
    plt.savefig('foo.png')'''

    fig, axs = plt.subplots(2, 4, figsize=(12, 12))

    axs[0, 0].plot(f_soluns[0][:,0], time)
    axs[0, 0].plot(equilibrium, time)
    axs[0, 0].set_title('System Position Solution')
    axs[0, 0].set(xlabel='Position', ylabel='Time')

    axs[0, 1].plot(f_soluns[0][:,1], time)
    axs[0, 1].plot(equilibrium, time)
    axs[0, 1].set_title('System Velocity Solution')
    axs[0, 1].set(xlabel='Velocity', ylabel='Time')

    axs[1, 1].plot(f_soluns[0][:,0], f_soluns[0][:,1])
    axs[1, 1].set_title('Phase Space')
    axs[1, 1].set(xlabel='Position', ylabel='Velocity')

    for sol in phase_spaces:
        axs[1, 2].plot(np.array(sol)[:,0], np.array(sol)[:,1])
    axs[1, 2].set_title('Phase Space')
    axs[1, 2].set(xlabel='Position', ylabel='Momentum')

    axs[1, 0].plot(times_list[0], masses_list[0])
    axs[1, 0].set_title('Mass Function')
    axs[1, 0].set(xlabel='Time', ylabel='Coefficient Value')

    axs[0, 3].plot(times_list[0], radii[0])
    axs[0, 3].set_title('Radius vs. Time')
    axs[0, 3].set(xlabel='Time', ylabel='||p,x||')

    i = 0
    while i < len(radii):
        axs[1, 3].plot(times_list[i], radii[i])
        i += 1
    axs[1, 3].set_title('Radius vs. Time')
    axs[1, 3].set(xlabel='Time', ylabel='||p,x||')

    axs[0, 2].plot(np.array(phase_spaces[0])[:,0], np.array(phase_spaces[0])[:,1])
    axs[0, 2].set_title('Phase Space')
    axs[0, 2].set(xlabel='Position', ylabel='Momentum')

    plt.show()
def three_d():
    f = lambda x, y: np.sin(np.sqrt(x ** 2 + y ** 2))
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
def animate():
    fig, axs = plt.subplots(2, 4, figsize=(12, 12))
    camera = Camera(fig)

    for i in range(0, n_steps, 1000):
        '''plt.figure()
        plt.plot(time, f_soluns[0][:,0])
        plt.plot(time, equilibrium)
        plt.xlabel('Time [s]')
        plt.ylabel('x')
        plt.show()
        plt.savefig('foo.png')'''

        axs[0, 0].plot(f_soluns[0][:,0][:i], time[:i])
        axs[0, 0].plot(equilibrium[:i], time[:i])
        axs[0, 0].set_title('System Position Solution')
        axs[0, 0].set(xlabel='Position', ylabel='Time')

        axs[0, 1].plot(f_soluns[0][:,1][:i], time[:i])
        axs[0, 1].plot(equilibrium[:i], time[:i])
        axs[0, 1].set_title('System Velocity Solution')
        axs[0, 1].set(xlabel='Velocity', ylabel='Time')

        axs[1, 1].plot(f_soluns[0][:,0][:i], f_soluns[0][:,1][:i])
        axs[1, 1].set_title('Phase Space')
        axs[1, 1].set(xlabel='Position', ylabel='Velocity')

        for sol in phase_spaces:
            axs[1, 2].plot(np.array(sol)[:,0][:math.floor(i/n_steps*len(sol))], np.array(sol)[:,1][:math.floor(i/n_steps*len(sol))])
        axs[1, 2].set_title('Phase Space')
        axs[1, 2].set(xlabel='Position', ylabel='Momentum')

        axs[1, 0].plot(times_list[0][:math.floor(i/n_steps*len(times_list[0]))], masses_list[0][:math.floor(i/n_steps*len(masses_list[0]))])
        axs[1, 0].set_title('Mass Function')
        axs[1, 0].set(xlabel='Time', ylabel='Coefficient Value')

        axs[0, 3].plot(times_list[0][:math.floor(i/n_steps*len(times_list[0]))], radii[0][:math.floor(i/n_steps*len(radii[0]))])
        axs[0, 3].set_title('Radius vs. Time')
        axs[0, 3].set(xlabel='Time', ylabel='||p,x||')

        j = 0
        while j < len(radii):
            axs[1, 3].plot(times_list[j][:math.floor(i/n_steps*len(times_list[j]))], radii[j][:math.floor(i/n_steps*len(radii[j]))])
            j += 1
        axs[1, 3].set_title('Radius vs. Time')
        axs[1, 3].set(xlabel='Time', ylabel='||p,x||')

        axs[0, 2].plot(np.array(phase_spaces[0])[:,0][:math.floor(i/n_steps*len(phase_spaces[0]))], np.array(phase_spaces[0])[:,1][:math.floor(i/n_steps*len(phase_spaces[0]))])
        axs[0, 2].set_title('Phase Space')
        axs[0, 2].set(xlabel='Position', ylabel='Momentum')
        print("Snap " + str(i))
        camera.snap()
    print("Ready to capture")
    animation = camera.animate()
    animation.save('test_animation.mp4')



animate()