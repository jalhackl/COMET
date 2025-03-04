import numpy as np
import matplotlib.pyplot as plt
import process_data
import plot_functions

from compare_clusterings import *
from plot_functions import *
from process_data import *

def return_planets_3d_with_com_motion(initial_conditions = [
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),                # Sun
    (1.0, 0.0, 0.0, 0.0, 2 * np.pi, 0.0),                # Earth
    (5.2, 0.0, 0.0, 0.0, 2 * np.pi / np.sqrt(5.2), 0.0), # Jupiter
    (1.5, 0.0, 0.0, 0.0, 2 * np.pi / np.sqrt(1.5), 0.0)  # Mars
],

# Masses of the 'Sun' and the planets
masses = [1.0, 3.003e-6, 9.545e-4, 3.213e-7] ,

# velocity for circular motion

com_velocity = (0.0, 2 * np.pi, 0.0) , # In AU/year
                                                      
                                      
T=4, dt=0.001, sun_explicitly=True, radius=1, period = 1, linear_period = 1):
    G = 4 * np.pi**2  # Gravitational constant in AU^3 / (yr^2 * Solar mass)
    num_bodies = len(initial_conditions)
    N = int(T / dt)
    epsilon = 1e-5  
    
    x = np.zeros((num_bodies, N+1))
    y = np.zeros((num_bodies, N+1))
    z = np.zeros((num_bodies, N+1))
    vx = np.zeros((num_bodies, N+1))
    vy = np.zeros((num_bodies, N+1))
    vz = np.zeros((num_bodies, N+1))
    
    for i in range(num_bodies):
        x[i, 0], y[i, 0], z[i, 0] = initial_conditions[i][0], initial_conditions[i][1], initial_conditions[i][2]
        vx[i, 0] = initial_conditions[i][3] #+ com_velocity[0]
        vy[i, 0] = initial_conditions[i][4] #+ com_velocity[1]
        vz[i, 0] = initial_conditions[i][5] #+ com_velocity[2]
    
    for i in range(N):

        com_x, com_y, com_z, com_vx, com_vy, com_vz = 0, 0, 0, 0, 0, 0
        if radius:
            omega = 2 * np.pi / period
            com_x = radius * np.cos(omega * i * dt)
            com_y = radius * np.sin(omega * i * dt)
            com_vx += -radius * omega * np.sin(omega * i * dt)
            com_vy += radius * omega * np.cos(omega * i * dt)

        if linear_period:
            phase = np.floor(i * dt / linear_period)
            direction = (-1) ** phase  
            com_vx += com_velocity[0] * direction
            com_vy += com_velocity[1] * direction
            com_vz += com_velocity[2] * direction


        kx = np.zeros((num_bodies, 4))
        ky = np.zeros((num_bodies, 4))
        kz = np.zeros((num_bodies, 4))
        kvx = np.zeros((num_bodies, 4))
        kvy = np.zeros((num_bodies, 4))
        kvz = np.zeros((num_bodies, 4))
        
        for k in range(4):
            for p in range(num_bodies):
                if k == 0:
                    px, py, pz = x[p, i], y[p, i], z[p, i]
                    pvx, pvy, pvz = vx[p, i] + com_vx, vy[p, i] + com_vy, vz[p, i] + com_vz
                else:
                    px = x[p, i] + 0.5 * kx[p, k-1] * dt
                    py = y[p, i] + 0.5 * ky[p, k-1] * dt
                    pz = z[p, i] + 0.5 * kz[p, k-1] * dt
                    pvx = vx[p, i] + 0.5 * kvx[p, k-1] * dt + com_vx
                    pvy = vy[p, i] + 0.5 * kvy[p, k-1] * dt + com_vy
                    pvz = vz[p, i] + 0.5 * kvz[p, k-1] * dt + com_vz
                

                if sun_explicitly:
                    ax, ay, az = 0, 0, 0
                else:
                    re = np.sqrt(px**2 + py**2 + pz**2)
                    ax = -G * px / re**3
                    ay = -G * py / re**3
                    az = -G * pz / re**3


                for q in range(num_bodies):
                    if p != q:
                        qx = x[q, i] + 0.5 * kx[q, k-1] * dt if k > 0 else x[q, i]
                        qy = y[q, i] + 0.5 * ky[q, k-1] * dt if k > 0 else y[q, i]
                        qz = z[q, i] + 0.5 * kz[q, k-1] * dt if k > 0 else z[q, i]
                        r = np.sqrt((px - qx)**2 + (py - qy)**2 + (pz - qz)**2) + epsilon
                        ax -= G * masses[q] * (px - qx) / r**3
                        ay -= G * masses[q] * (py - qy) / r**3
                        az -= G * masses[q] * (pz - qz) / r**3
                
                kx[p, k] = pvx
                ky[p, k] = pvy
                kz[p, k] = pvz
                kvx[p, k] = ax
                kvy[p, k] = ay
                kvz[p, k] = az
        
        for p in range(num_bodies):
            x[p, i+1] = x[p, i] + dt * (kx[p, 0] + 2*kx[p, 1] + 2*kx[p, 2] + kx[p, 3]) / 6
            y[p, i+1] = y[p, i] + dt * (ky[p, 0] + 2*ky[p, 1] + 2*ky[p, 2] + ky[p, 3]) / 6
            z[p, i+1] = z[p, i] + dt * (kz[p, 0] + 2*kz[p, 1] + 2*kz[p, 2] + kz[p, 3]) / 6
            vx[p, i+1] = vx[p, i] + dt * (kvx[p, 0] + 2*kvx[p, 1] + 2*kvx[p, 2] + kvx[p, 3]) / 6
            vy[p, i+1] = vy[p, i] + dt * (kvy[p, 0] + 2*kvy[p, 1] + 2*kvy[p, 2] + kvy[p, 3]) / 6
            vz[p, i+1] = vz[p, i] + dt * (kvz[p, 0] + 2*kvz[p, 1] + 2*kvz[p, 2] + kvz[p, 3]) / 6
    
    positions = [(x[p], y[p], z[p]) for p in range(num_bodies)]
    return positions