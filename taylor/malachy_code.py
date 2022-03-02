import numpy as np
import random
import matplotlib.pyplot as plt
import math

def taylor(x,c):
    ts = [x**n/math.factorial(n) for n in range(ncoef)]
    return np.dot(np.array(c),np.array(ts))

def fourier(x,c):
    fs = [1]
    n=1
    while len(fs) < ncoef:
        fs.append(np.sin(n*x/distance))
        if len(fs) < ncoef:
            fs.append(np.cos(n*x/distance))
        n+=1
    return np.dot(np.array(c),np.array(fs)) 


ncoef = 4
c = [random.uniform(0,1.5)]
while len(c) < ncoef:
    rand = random.uniform(-1.5,1.5)
    c.append(rand)
