# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 08:36:35 2023

@author: Manuel
"""

import numpy as np
from scipy.special import jn_zeros
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%% Parameters
Nr = 50
N_phi = 50
N_steps = 10000
radius =1
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

#%% Tipos de forzantes
sigma=1
def forzante(x,tipo):
    if tipo==1:
        return np.sin(x)
    if tipo==2:
        if x == 0:
            return 1
        else:
            return np.sin(x) / x
    if tipo==3:
        return np.exp(-x**2 / (2 * sigma**2))
    if tipo==4:
        return ((x**2 - sigma**2) / sigma**4) * np.exp(-x**2 / (2 * sigma**2))

#%% Selecciión de modo y tipo de forzante
m=1
n=1
frec= jn_zeros(m, n)[n-1]*c/radius

print("la frecuencia es",frec,"modo",m,n)

tipo=1
#frec=12
b=2
#%% Stepping
b=2
u = np.zeros((N_steps, Nr, N_phi))
k1 = (c*dt)**2/dr**2
for t in range(0, N_steps-1):
    for i in range(0, Nr):
        for j in range(-1, N_phi-1):
            ri = max(r[i], 0.5*dr)  # To avoid the singularity at r=0
            k2 = (c*dt)**2/(2*ri*dr)
            k3 = (c*dt)**2/(dphi*ri)**2
            if i==Nr-1:
                u[t+1, i, j]=1/(1+b*dt/2)*(2*u[t, i, j] - u[t-1, i, j]\
                +k1*( 2*u[t,i-1,j] -2*u[t,i,j] ) \
                +k3*(u[t, i, j+1] - 2*u[t, i, j] + u[t, i, j-1]) \
                +b*dt/2*(u[t-1,i,j]) )
            else:
                u[t+1, i, j] = 1/(1+b*dt/2)*(2*u[t, i, j] - u[t-1, i, j] \
                + k1*(u[t, i+1, j] - 2*u[t, i, j] + u[t, i-1, j])\
                + k2*(u[t, i+1, j] - u[t, i-1, j])\
                + k3*(u[t, i, j+1] - 2*u[t, i, j] + u[t, i, j-1])\
                +b*dt/2*(u[t-1,i,j]))#damping term
                if i in range (0,2):
                    u[t+1,i,j]=u[t+1,i,j]+1/5*forzante(frec*dt*t, tipo)
                
                #if i==24 and j==0:
                    #u[t+1,i,j]=u[t+1,i,j]+1/5*forzante(frec*dt*t, tipo)
                #else:
                    #u[t+1, i, -1] = u[t+1, i, 0]

#%% Plot 11
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
Z = u[9800]  # Or choose another time step to visualize # Crece hasta 150 maximo en 610
# Recien en 900 empieza a cambiar el sentido
ax.plot_surface(X.T, Y.T, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Modo {}, {}".format(m, n))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#ax.view_init(azim=70, elev=13)
plt.show()
#%%
fig.savefig('modo11bordelibre.pdf', bbox_inches='tight')
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
#ax3d.view_init(azim=70, elev=13)
# Función de actualización para la animación
def update(i):
    index = i * 50
    if index < len(u):
        Z_3d = u[index]
        surf[0].remove()
        surf[0] = ax3d.plot_surface(X.T, Y.T, Z_3d, cmap='viridis')

ani = FuncAnimation(fig, update, frames=range(N_steps), interval=10, blit=False, repeat=False)


plt.show()
#%% Store
# Specify the filename for the GIF
gif_filename = 'bordelibre.gif'

# Save the animation as a GIF using PillowWriter
ani.save("bordelibre.gif", writer="pillow", fps=60, dpi=80)

plt.show()