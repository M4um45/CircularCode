# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:48:19 2023

@author: edgar
"""

import numpy as np
from scipy.special import jv, jn_zeros
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#%% Parameters
Nr = 50
N_phi = 50
N_steps = 10000
radius = 1
c = 1
dphi = 2*np.pi/N_phi
dr = radius/Nr
dt = 0.001
CFL=c*dt*(1/dr+1/dphi)
print(CFL)



#%% Initial conditions
r = np.linspace(0, radius, Nr)
phi = np.linspace(0, 2*np.pi, N_phi)
R, phi = np.meshgrid(r, phi)
X = R*np.cos(phi)
Y = R*np.sin(phi)
#%% Modos Simples
m=1
n=1
kth_zero = jn_zeros(m, n)[n-1]
Z = np.cos(m*phi) * jv(m, kth_zero*R/radius)
u = np.zeros((N_steps, Nr, N_phi))
u[0, :, :] = Z.T/10

#%% Combinación de modos# 
m1=1
n1=2
kth_zero1= jn_zeros(m1, n1)[n1-1]
m2=2
n2=1
kth_zero2= jn_zeros(m2, n2)[n2-1]
Z = np.cos(m2*phi) * jv(m2, kth_zero2*R/radius)+np.cos(m1*phi) * jv(m1, kth_zero1*R/radius)
u = np.zeros((N_steps, Nr, N_phi))
u[0, :, :] = Z.T

#%% Plot de combinación inicial
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
M = u[0]  # Or choose another time step to visualize

ax.plot_surface(X.T, Y.T, M, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
#%% Stepping
k1 = (c*dt)**2/dr**2
for t in range(0, N_steps-1):
    for i in range(0, Nr-1):
        for j in range(0, N_phi-1):
            ri = max(r[i], 0.5*dr)  # To avoid the singularity at r=0
            k2 = (c*dt)**2/(2*ri*dr)
            k3 = (c*dt)**2/(dphi*ri)**2
            u[t+1, i, j] = 2*u[t, i, j] - u[t-1, i, j] \
            + k1*(u[t, i+1, j] - 2*u[t, i, j] + u[t, i-1, j])\
            + k2*(u[t, i+1, j] - u[t, i-1, j])\
            + k3*(u[t, i, j+1] - 2*u[t, i, j] + u[t, i, j-1])

        u[t+1, i, -1] = u[t+1, i, 0]  # Update the values for phi=2*pi

#%% Plot 11
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Z = u[5000]  
ax.set_title("Modo {}, {}".format(m, n))
ax.plot_surface(X.T, Y.T, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
#%%

fig.savefig('modo11membrana.pdf', bbox_inches='tight')
#%%
fig, ax = plt.subplots()

ax.contour(X.T, Y.T, Z)
ax.set_aspect('equal')
#%% Para modos simples

fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')

# Inicializa la superficie 3D
Z_3d = u[0]
surf = [ax3d.plot_surface(X.T, Y.T, Z_3d, cmap='viridis')]
# Set Z-axis limits
u_flat = u.flatten()

maximo= np.max(u_flat)
minimo=np.min(u_flat)
if abs(maximo)<=abs(minimo):    
    ax3d.set_zlim(-abs(minimo), abs(minimo))
else:
    ax3d.set_zlim(-maximo, maximo)
ax3d.set_title("Modo {}, {}".format(m, n))
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.view_init(azim=50, elev=13)
# Función de actualización para la animación
def update(i):
    index = 25 * i  # Increase the index by 2 at each step
    if index < len(u):
        Z_3d = u[index]  # Datos para la iteración actual
        surf[0].remove()  # Elimina la superficie anterior
        surf[0] = ax3d.plot_surface(X.T, Y.T, Z_3d, cmap='viridis')  # Actualiza el gráfico 3D

    


ani = FuncAnimation(fig, update, frames=len(u), interval=1, blit=False, repeat=False)
# Set the view angle (azimuth and elevation)


plt.show()



#%% Para modos compuestos

fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')

Z_3d = u[0]
surf = [ax3d.plot_surface(X.T, Y.T, Z_3d, cmap='viridis')]

u_flat = u.flatten()
maximo = np.max(u_flat)
minimo = np.min(u_flat)

if abs(maximo) <= abs(minimo):
    ax3d.set_zlim(-abs(minimo), abs(minimo))
else:
    ax3d.set_zlim(-maximo, maximo)

ax3d.set_title("Modo {}, {} + Modo {}, {}".format(m1, n1, m2, n2))
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.view_init(azim=70, elev=13)

def update(i):
    Z_3d = u[i]
    surf[0].remove()
    surf[0] = ax3d.plot_surface(X.T, Y.T, Z_3d, cmap='viridis')
    
    # Update the view angle for rotation
    azim = i * 1/2  # Adjust the factor to control the rotation speed
    ax3d.view_init(azim=azim, elev=13)

ani = FuncAnimation(fig, update, frames=len(u), interval=50, blit=False, repeat=False)

plt.show()


#%%



# Save the animation as a GIF using PillowWriter
ani.save("modo41.gif", writer="pillow", fps=60, dpi=80)

plt.show()