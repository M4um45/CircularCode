# -*- coding: utf-8 -*- 
"""

Created on Tue Oct 31 15:48:19 2023 

@author: ManuelBorraSantarcieri
"""
#paquetes a usar
import numpy as np
from scipy.special import jn_zeros
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%% Parametros a usar
Nr = 50 # discretización radial
N_phi = 50 # discretización angular
N_steps = 10 # discretización temporal
radius = 1 
c = 1
dphi = 2*np.pi/N_phi # paso angular
dr = radius/Nr # paso radial
dt = 0.001 # paso temporal
CFL=c*dt*(1/dr+1/dphi)
print(CFL)



#%% Vectores 
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


#%% Selecciión de modo y tipo de forzante para la solución analitia omitir
m=2
n=1
frec= jn_zeros(m, n)[n-1]*c/radius

print("la frecuencia es",frec,"modo",m,n)

tipo=1
frec=10 #en caso de desear frecuancias arbitrarias descomentar


#%% Para modos con m distinto de 0 Forzante simple
u = np.zeros((N_steps, Nr, N_phi))# vector de onda
k1 = (c*dt)**2/dr**2
for t in range(0, N_steps-1):
    for i in range(0, Nr-1):
        for j in range(-1, N_phi-1):
            ri = max(r[i], 0.5*dr)  # To avoid the singularity at r=0
            k2 = (c*dt)**2/(2*ri*dr)
            k3 = (c*dt)**2/(dphi*ri)**2
            u[t+1, i, j] = 2*u[t, i, j] - u[t-1, i, j] \
            + k1*(u[t, i+1, j] - 2*u[t, i, j] + u[t, i-1, j])\
            + k2*(u[t, i+1, j] - u[t, i-1, j])\
            + k3*(u[t, i, j+1] - 2*u[t, i, j] + u[t, i, j-1])
            if i==24 and j==0:
                u[t+1,i,j]=u[t+1,i,j]+1/5*forzante(frec*dt*t, tipo)
            else:
                u[t+1, i, -1] = u[t+1, i, 0]
            
 
#%% Para modos con m distinto de 0 Forzante doble
u = np.zeros((N_steps, Nr, N_phi))# vector de onda
k1 = (c*dt)**2/dr**2
for t in range(0, N_steps-1):
    for i in range(0, Nr-1):
        for j in range(-1, N_phi-1):
            ri = max(r[i], 0.5*dr)  # To avoid the singularity at r=0
            k2 = (c*dt)**2/(2*ri*dr)
            k3 = (c*dt)**2/(dphi*ri)**2
            u[t+1, i, j] = 2*u[t, i, j] - u[t-1, i, j] \
            + k1*(u[t, i+1, j] - 2*u[t, i, j] + u[t, i-1, j])\
            + k2*(u[t, i+1, j] - u[t, i-1, j])\
            + k3*(u[t, i, j+1] - 2*u[t, i, j] + u[t, i, j-1])
            if i==24 and j==0:
                u[t+1,i,j]=u[t+1,i,j]+1/5*forzante(frec*dt*t, tipo)
            elif i==24 and j==24:
                u[t+1,i,j]=u[t+1,i,j]+1/5*forzante(frec*dt*t, tipo)
            else:
                u[t+1, i, -1] = u[t+1, i, 0]
            
            
#%% Para forzante central modos simétricos
u = np.zeros((N_steps, Nr, N_phi))# vector de onda
k1 = (c*dt)**2/dr**2
for t in range(0, N_steps-1):
    for i in range(0, Nr-1):
        for j in range(-2, N_phi-1):
            ri = max(r[i], 0.5*dr)  # To avoid the singularity at r=0
            k2 = (c*dt)**2/(2*ri*dr)
            k3 = (c*dt)**2/(dphi*ri)**2
            u[t+1, i, j] = 2*u[t, i, j] - u[t-1, i, j] \
            + k1*(u[t, i+1, j] - 2*u[t, i, j] + u[t, i-1, j])\
            + k2*(u[t, i+1, j] - u[t, i-1, j])\
            + k3*(u[t, i, j+1] - 2*u[t, i, j] + u[t, i, j-1])
            if i in range (0,2):#forzante cilindrico
                u[t+1,i,j]=u[t+1,i,j]+1/5*forzante(frec*dt*t, tipo)


#%% Plots
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
Z = u[9]  # Or choose another time step to visualize # Crece hasta 150 maximo en 610
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
ax.view_init(azim=50, elev=13)
plt.show()
#%%
fig.savefig('doble.png', bbox_inches='tight')

#%% Animation

fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')

# Inicializa la superficie 3D
Z_3d = u[0]
surf = [ax3d.plot_surface(X.T, Y.T, Z_3d, cmap='viridis')]

# Set Z-axis limits
u_flat = u.flatten()
maximo = np.max(u_flat)
minimo = np.min(u_flat)

if abs(maximo) <= abs(minimo):
    ax3d.set_zlim(-maximo, maximo)
else:
    ax3d.set_zlim(-maximo, maximo)

ax3d.set_title("Modo {}, {}".format(m, n))
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.view_init(azim=50, elev=13)

# Añade un texto para mostrar el valor de i
text = ax3d.text2D(0.05, 0.95, "", transform=ax3d.transAxes, color='red')

def update(i):
    index = i * 25  # No es necesario multiplicar por 25 si i ya es el índice que necesitas
    if index < len(u):
        Z_3d = u[index]
        surf[0].remove()
        surf[0] = ax3d.plot_surface(X.T, Y.T, Z_3d, cmap='viridis')
        
        # Actualiza el texto con el valor de i
        #text.set_text("t = {}".format(index))

ani = FuncAnimation(fig, update, frames=range(N_steps), interval=10, blit=False, repeat=False)

plt.show()
#%% Store
# Specify the filename for the GIF


# Save the animation as a GIF using PillowWriter
ani.save("21.gif", writer="pillow", fps=60, dpi=80)

plt.show()

