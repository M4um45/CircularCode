# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:58:24 2023

@author: Manuel
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
m=3
n=4
kth_zero = jn_zeros(m, n)[n-1]
Z = np.cos(m*phi) * jv(m, kth_zero*R/radius)#a tiempo t=0
u = np.zeros((N_steps, Nr, N_phi))
u[0, :, :] = Z.T/10
#%% Plot de combinación inicial
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
M = u[0]  # Or choose another time step to visualize

ax.plot_surface(X.T, Y.T, M, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
#%% Evolución temproal


for t in range (N_steps):
    u[t,:,:]=Z.T * np.cos(c * jn_zeros(m, n)[n-1] * t)

#%% Plot 11
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
Z = u[7000]  # Or choose another time step to visualize # Crece hasta 150 maximo en 610
# Recien en 900 empieza a cambiar el sentido
ax.plot_surface(X.T, Y.T, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Modo {}, {}".format(m, n))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#ax.set_zlim(-181, 181)
#ax.view_init(azim=35, elev=13)
plt.show()
#%%
fig.savefig('modo34analitico.pdf', bbox_inches='tight')
#%%
fig, ax = plt.subplots()

ax.contour(X.T, Y.T, Z)
ax.set_aspect('equal')
#%% Animation

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
    ax3d.set_zlim(-maximo, maximo)
else:
    ax3d.set_zlim(-maximo, maximo)

#ax3d.set_zlim(-250, 250)
ax3d.set_title("Modo {}, {}".format(m, n))
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.view_init(azim=50, elev=13)
# Función de actualización para la animación
def update(i):
    index = i*10
    if index < len(u):
        Z_3d = u[index]
        surf[0].remove()
        surf[0] = ax3d.plot_surface(X.T, Y.T, Z_3d, cmap='viridis')

ani = FuncAnimation(fig, update, frames=range(N_steps), interval=10, blit=False, repeat=False)


plt.show()