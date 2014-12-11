Module-3
========
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 08:47:03 2014

@author: sathyanarayan
"""

import numpy as np
import matplotlib.pyplot as plt

def sodIC1():
    nx = 81
    dx = .25
    dt = .0002   
    x = np.linspace(-10,10,81)
    
    nt = np.ceil(.01/dt)
    
    #initial conditions
    wl = np.array([1., 0, 100000.])
    wr = np.array([0.125, 0, 10000.])
    
    U = np.ones((3,nx))
    
    U[:,:40] = build_U(wl[0],wl[1],wl[2])
    U[:,40:] = build_U(wr[0],wr[1],wr[2])
    
    return U, dx, dt, nx, x, int(nt)


    
def build_U(rho, u, p):
    gamma = 1.4
    e = p / ((gamma-1)*rho)
    e_T = e + u**2/2
    
    U = np.array([[rho],[rho*u],[rho*e_T]])
    
    return U
    

def build_flux(U_in):
    '''Takes a 3x1 vector U and generates the flux vector for Conserved Euler Eqs'''
    gamma = 1.4
    u1, u2, u3 = U_in[0], U_in[1], U_in[2]
    F = np.array([u2,u2**2/u1+(gamma-1)*(u3-u2**2/(2*u1)),\
                  u2/u1*(u3+(gamma-1)*(u3-u2**2/(2*u1)))])
                  
    return F

    

def richtmyer(U, dx, dt, nx, nt, damp=0):
    UN = np.ones((3,nx))
    UN_plus = np.ones((3,nx))
    UN_minus = np.ones((3,nx))
    
    for i in range(nt):
        UN_plus[:,:-1] = .5*(U[:,1:]+U[:,:-1]) -\
        dt/(2*dx)*(build_flux(U[:,1:]) - build_flux(U[:,:-1]))
        
        UN_minus[:,1:] = UN_plus[:,:-1]  
        UN[:,1:-1] = U[:,1:-1] - dt/dx *\
        (build_flux(UN_plus[:,1:-1]) - build_flux(UN_minus[:,1:-1])) +\
        damp * (U[:,2:] - 2*U[:,1:-1] + U[:,:-2])
        
        UN[:,0] = UN[:,1]
        UN[:,-1] = UN[:,-2]
        
        U[:,:] = UN[:,:]
        
    return U

def decompose_U(U_in):
    '''Extract pressure, velocity, sound speed, density, entropy and Mach number from U'''
    gamma = 1.4
    
    vel = U_in[1,:]/U_in[0,:]
    pres = (gamma - 1)*(U_in[2,:] - .5 * U_in[1,:]**2 / U_in[0,:])
    rho = U_in[0,:]
    c = np.sqrt(gamma * pres / rho)
    S = pres/rho**gamma
    
    return vel, pres, rho, c, S
    
U, dx, dt, nx, x, nt = sodIC1()
U = richtmyer(U, dx, dt, nx, nt)
vel, pres, rho, c, S = decompose_U(U)

def plot_shock_tube(vel, pres, rho, c, S, x):
    plt.figure(figsize=(14,7))
    
    plt.subplot(2,3,1)
    plt.plot(x,vel)
    plt.title('Velocity')
    
    plt.subplot(2,3,2)
    plt.plot(x,pres)
    plt.title('Pressure')
    
    plt.subplot(2,3,3)
    plt.plot(x,rho)
    plt.title('Density')
    
    plt.subplot(2,3,4)
    plt.plot(x,vel/c)
    plt.title('Mach Number')
    
    plt.subplot(2,3,5)
    plt.plot(x,c)
    plt.title('Speed of sound')
    
    plt.subplot(2,3,6)
    plt.plot(x,S)
    plt.title('Entropy')
    


plot_shock_tube(vel, pres, rho, c, S, x)

print(x[50],vel[50],pres[50],rho[50])
