# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 22:24:26 2021

@author: ADRIAN CAFA
"""
## Inicialización del Notebook del TP3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame
#interactive plotting in separate window
%matplotlib qt

# Insertar aquí el código para inicializar tu notebook
########################################################
### 1) Compruebe experimentalmente las propiedades de sesgo y varianza del periodograma.

# Simular para los siguientes tamaños de señal
#N = np.array([10, 50, 100, 250, 500, 1000, 5000], dtype=np.float)

mu, sigma = 0, 2 # mean and standard deviation


M = 200  # numero de muestras
N = np.array([10, 50, 100, 250, 500, 1000, 5000])

for i in range(len(N)):
    # Genero señal aleatoria:
    x = np.random.normal(mu, sigma,size=(N[i]))
    X = np.fft.fft(x,axis=0)
    f = np.linspace(0, np.pi, len(X))
    
    Pxx = (1/N[i]) * abs(X)**2
    data = np.ones_like(Pxx)
    true_PSD= N[i] * data
    

    # plot
    plt.figure(figsize=(10, 4))
    plt.stem(f, Pxx, 'C0')
    plt.plot(f, true_PSD, 'C1', label=r'$\Phi_{xx}(e^{j \Omega})$')
    plt.title('Estimacion del PSD')
    plt.xlabel(r'$\Omega$')
    plt.axis([0, np.pi, 0, N[i]])
 
    # calculo del sesgo y varianza del periodograma
    print("############################")
    print("N[i]: {}".format(N[i]))
    print('Sesgo del periodograma: \t {0:1.4f}'.format(np.mean(Pxx-1)))
    print('Varianza de el periodograma: \t {0:1.4f}'.format(np.var(Pxx)))
    print("############################")