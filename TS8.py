# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 19:51:39 2021

@author: ADRIAN CAFA
"""
####Enunciado

# Para una señal x(k)=a1⋅sen(Ω1⋅k)+n(k)
# siendo

# Ω1=Ω0+fr⋅2πN
# Ω0=π2
# y las variables aleatorias definidas por

# fr∼U(−12,12)
# n∼N(0,σ2)
# Evalúe el siguiente estimador de Ω1
# Ω^W1=arg maxf{PW^}
# basado en el periodograma de Welch evaluado en 3). Del mismo modo, evalúe otro estimador de la PSD para crear otro estimador de Ω1
# Ω^X1=arg maxf{PX^}
# Considere 200 realizaciones de 1000 muestras para cada experimento. Cada realización debe tener un SNR tal que el pico de la senoidal esté 3 y 10 db 
# por encima del piso de ruido impuesto por n(k).


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.fft import fft, fftshift
import matplotlib.patches as mpatches
from pandas import DataFrame
from IPython.display import HTML
#interactive plotting in separate window
%matplotlib qt

#######################################################################################################################
#%% Inicio de la simulación
#######################################################################################################################
plt.close('all')

fs = 1000 # frecuencia de muestreo (Hz)
R = 200 # realizaciones
N = 1000 # Muestras

figura=0
 
ts = 1/fs  # tiempo de muestreo
df = fs/N  # resolución espectral

t=np.arange(0,1,ts)
f=np.arange(0,fs,df)

# Señal indicada:
####################
a0=2
omega_0=np.pi/2
M = 200
fr=np.random.uniform(low=-2, high=2, size=M)
omega_1=omega_0+fr*2*np.pi/N
## NOTA: como uno es de dimension (200,1) y el otro es de dimension (1000,1) no puedo hacer el producto
## cambio los ejes para que sea (200,1)*(1,1000)

x=np.sin(2*np.pi*omega_1.reshape(1,200)*(fs/(2*np.pi))*t.reshape(1000,1))
noise = np.random.normal(size=L*M)







