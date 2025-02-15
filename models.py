#!/usr/bin/env python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def kuramoto1(y,t):
    dydt = np.zeros((y.shape[0],))

    w = np.loadtxt('Data/frequencies.dat')

    k = w.shape[0]

    J = np.loadtxt('Data/connectivity.dat')

    for i in range(k):
        sum = 0.0
        for j in range(k):
            sum += J[i,j] * np.sin(y[j] - y[i])

        dydt[i] = w[i] + sum

    return(dydt)

def kuramoto2(y,t):
    dydt = np.zeros((y.shape[0],))

    w = np.loadtxt('Data/frequencies.dat')

    k = w.shape[0]

    J = np.loadtxt('Data/connectivity.dat')

    for i in range(k):
        sum=0.0
        for j in range(k):
            sum += J[i,j]*(np.sin(y[j]-y[i]-1.05)) + 0.33*np.sin(2*(y[i]-y[j]))
        dydt[i] = w[i] + sum

    return(dydt)

def michaelis_menten(y,t):
    dydt = np.zeros((y.shape[0],))

    J = np.loadtxt('Data/generated_data/connectivity.dat')

    k = J.shape[0]

    for i in range(k):
        sum=0.0
        for j in range(k):
            sum += (J[i,j] * (y[j]/(1+y[j])))
        dydt[i] = -y[i] + sum

    return(dydt)

def roessler(y,t):
    dydt = np.zeros(y.shape[0])

    J = np.loadtxt('Data/connectivity.dat')
    N = J.shape[0]

    for i in range(N):
        sum = 0.0
        for j in range(N):
            sum += J[i,j]*np.sin(y[(3*j)+0])

        dydt[(3*i)+0] = -y[(3*i)+1] - y[(3*i)+2] + sum
        dydt[(3*i)+1] = y[(3*i)+0] + 0.1*y[(3*i)+1]
        dydt[(3*i)+2] = 0.1 + (y[(3*i)+2]*(y[(3*i)+0]-18))

    return(dydt)

if __name__ == "__main__":
    init = 1. + np.random.uniform(0.,1.,size=(6,))
    tspan = np.arange(0,10,1)
    y = odeint(kuramoto2, init, tspan)
    print(y)
    plt.plot(tspan,y)
    plt.savefig("test.pdf")
