# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:37:33 2023

@author: Manuel
"""
import numpy as np
from scipy.special import jn_zeros
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
import time
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

#%% MODE 11
m=0
n=1
omega = jn_zeros(m, n)[n-1]*c/radius

print("la frecuencia es ", omega, "radian/s; ", "modo es ",m,n)
# EL FORZANTE NO PUEDE ESTAR EN EL MEDIO SI NO EXITA OTROS MODOS
#%% suma de modos
m=0
n=1
m1=0
n1=2 
m2=0
n2=3
omega0=jn_zeros(m, n)[n-1]*c/radius
omega1=jn_zeros(m1, n1)[n1-1]*c/radius
omega2=jn_zeros(m2, n2)[n2-1]*c/radius
omega = omega0+omega1+omega2
# a mayor frecuencia es necesario más nsteps 


#%% Stepping
inicio = time.time()
u = np.zeros((N_steps, Nr, N_phi))
k1 = (c*dt)**2/dr**2
#u[0,12,29] = 1
for t in range(0, N_steps-1):
    for i in range(0, Nr-1):
        ri = max(r[i], 0.5*dr)  # To avoid the singularity at r=0
        k2 = (c*dt)**2/(2*ri*dr)
        for j in range(-1, N_phi-1):
            k3 = (c*dt)**2/(dphi*ri)**2
            u[t+1, i, j] = 2*u[t, i, j] - u[t-1, i, j] \
            + k1*(u[t, i+1, j] - 2*u[t, i, j] + u[t, i-1, j])\
            + k2*(u[t, i+1, j] - u[t, i-1, j])\
            + k3*(u[t, i, j+1] - 2*u[t, i, j] + u[t, i, j-1])
            #if i==12 and j==29:
                #u[t+1,i,j]=u[t+1,i,j]+1/5*np.sin(omega*dt*t)
            #if i in range (0,2):
               # u[t+1,i,j]=u[t+1,i,j]+1/5*np.sin(omega*dt*t)
            if i==24 and j==0:
                u[t+1,i,j]=u[t+1,i,j]+1/5*np.sin(omega*dt*t)
final= time.time()
tiempo=final-inicio

#Tiempo de ejecución para Nsteps=1e+5 se obtienen 14.53 minutos
#para N=1e4 91 segundos, para N=1e3 8 segundos. El orden del código es de N**2
#cte=tiempo/(50**2*1e5)=3.4e-6 valores del mismo orden para otros N. Implica que
# sabemos cual es la constante para este código.


#%%              
uu = u[:,0,0]#oscilación del forzante Cambiar dependiendo la posición asignada
plt.figure('uu')
plt.plot(uu)
plt.show()
#%%
uuFourier = np.fft.fft(uu)#transformada de fourier en la posición del forzante
u_flat = uuFourier.flatten()

maximo= np.max(abs(u_flat))
uuFourier=uuFourier/maximo

plt.figure('FFT(uu)')
plt.plot(np.abs(uuFourier))
plt.show()



#%%
plt.figure('FFT(uu)')
plt.plot(np.abs(uuFourier[:20]))
# Find the peak value and its index
peak_value = np.max(np.abs(uuFourier[:20]))
peak_index = np.argmax(np.abs(uuFourier[:20]))

# Add a marker at the peak point
plt.scatter(peak_index, peak_value, color='red', marker='x', label=f'Peak: {peak_value:.2f} at {peak_index}')

plt.legend()
plt.show()

print("frecuencia aproximada",peak_index/10*2*np.pi)
print("frecuenia exacta",omega)
#The maximum value appears to be at 4, but in fact it is at 2.405/2pi*10 = 3.827
#%%

#%%
limite=40
fig=plt.figure('FFT(uu)')
x=np.linspace(0,limite-1,limite)/10* 2 * np.pi
plt.plot(x,np.abs(uuFourier[:limite]))
plt.xlabel("Frecuencia Hz")
plt.ylabel("Amplitud (normalizada)")
# Find peaks
peaks, _ = find_peaks(np.abs(uuFourier[:limite]), height=0.013)

# Get peak values and indices
peak_values = np.abs(uuFourier[peaks])
peak_indices = peaks/10 * 2 * np.pi

# Add markers at the peak points
plt.scatter(peak_indices, peak_values, color='red', marker='x')

# Annotate each peak with its index and value
for i, (index, value) in enumerate(zip(peak_indices, peak_values)):
    formatted_index = "{:.2f}".format(index)  # Format index to three decimal places
    formatted_value = "{:.2f}".format(value)  # Format value to three decimal places
    plt.annotate(f'Frec: {formatted_index}', (index, value),
                 textcoords="offset points", xytext=(0, 10), ha='center', color='blue')

plt.show()

# Calculate and print the result for each peak
for index, value in zip(peak_indices, peak_values):
    result = index
    formatted_value = "{:.2f}".format(value)  # Format value to three decimal places
    formatted_result = "{:.2f}".format(result)  # Format result to three decimal places
    print(f'Index: {index:.2f}, Value: {formatted_value}, Result: {formatted_result}')
    
# omega esta en el indice del maximo cumple con los esperado
#omega0 también
#%%
fig.savefig('FTTarbi.png', bbox_inches='tight')
#%% En el caso de usar un frecuancia arbitraria centrada

for j in range (10):
    for i in range (1,5):
        print("frecuencias", jn_zeros(j, i)[i-1]*c/radius,"modo",j,i)



  
    
