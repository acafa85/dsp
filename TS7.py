# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 20:50:22 2021

@author: ADRIAN CAFA
"""

"""
En esta tarea continuaremos con el capítulo 14 de Holton. DSP Principles and App.
Para ello, realizaremos el punto 3.a del TP2.

3- Genere la siguiente señal

$$x_{(k)}=a_0*sen(Ω1*k)$$
siendo

$$a_0=2$$$$Ω1=Ω0+fr⋅2πN$$$$Ω0=π2$$
y la variable aleatoria definida por la siguiente distribución de probabilidad

$fr∼U(−2,2)$

Considere 200 realizaciones (muestras tomadas de fr) para cada ventana analizada en la TS6. Se pide entonces:

3.a) Grafique los histogramas de $|Xiw(Ω0)|$

siendo

$|X^i_w(Ω)|=|F\{x(k)*w_i(k)\}|$ para la i-ésima ventana de las 5 utilizadas en la TS6. El sesgo y la varianza se definen para este caso como:

Grafique los 5 histogramas juntos, o cuide que todos los gráficos tengan el mismo rango de valores en X para facilitar la comparación visual. 3.b) Calcule experimentalmente el sesgo y la varianza del siguiente estimador:$$\hat a_0=|X^i_w(Ω0)|$$siendo

$sa=E{\hat a_0}−a0$

$va=var\{\hat a0\}=E\{(\hat a0 −E\{\hat a0\})^2\}$ y pueden aproximarse cuando consideramos los valores esperados como las medias muestrales

$E{a0^}=μa^=1M∑j=0M−1aj^$

$sa=μa^−a0$

$va=1/M∑j=0M−1(aj^−μa^)^2$

Estimación de Amplitud

sa va Rectangular
Bartlett
Hann
Blackman
Flat-top
Bonus Visualizar los 5 histogramas juntos
"""



import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')
#interactive plotting in separate window
%matplotlib qt

# Setup inline graphics: Esto lo hacemos para que el tamaño de la salida, 
# sea un poco más adecuada al tamaño del documento
mpl.rcParams['figure.figsize'] = (15,10)

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import plotly.graph_objects as go
from IPython.display import HTML
import pandas as pd

plt.close('all')


N = 1000
fs = 1000
fase = 0
f = np.linspace(0, (N-1)*(fs/N), N)
t = np.linspace(0, (N-1)/fs, N)
dfnorm = fs/(2*np.pi) # resolución espectral normalizada en radianes

############ Defino Ventanas

box = np.repeat(1,N)
barlett = signal.windows.bartlett(N)
hann = signal.windows.hann(N)
blackman = signal.windows.blackman(N)
flattop = signal.windows.flattop(N)

windows = [box,barlett,hann,blackman,flattop]
nameswindows = ['box','barlett','hann','blackman','flattop']
vw = np.arange(0,len(windows),1) # Secuencia de índices para las ventanas (usada en el for)

############

a0 = 2 # amplitud
fr = np.random.uniform(-2,2,200) # Variable aleatoria con distribución uniforme
vfr =  np.arange(0,len(fr),1) # Secuencia de índices para la variable aleatoria (usada en el for)

omega0 = np.pi/2
omega0norm = int(omega0 * dfnorm)
omega1 = omega0 + fr*(2*np.pi/N)

matriz = np.empty((N, len(fr)*5)) # N filas por muestras y 1000 columnas por 200 realizaciones y 5 ventanas
matrizdft = np.empty((N, len(fr)*5)) # idem valores transformados

######### Para cada ventana hay 200 realizaciones, 
######### cada una de ellas calculada con un distinto Omega 1 (dependiente de la variable aleatoria)

for w in vw:
    for i in vfr:
        columna = w*200+i
        matriz [:,columna] = a0 * np.sin(2 * np.pi * (omega1[i] * dfnorm) * t + fase) * windows[w]
        matrizdft [:,columna] = np.abs((1/N)*fft(matriz[:,columna]))

######### Se extrae los valores de las realizaciones en "Omega0" Hz para cada ventana y se arma la matriz f250.

f250 = np.empty((len(fr), len(windows)))
for z in vw:
    ini = z*len(fr)
    fin = (z+1)*len(fr)
    f250 [:,z] = matrizdft[omega0norm,ini:fin]#/max(matrizdft[omega0norm,ini:fin])
    # valores normalizados para que el valor máximo sea 1.
    
    # Columna 0 = box
    # Columna 1 = barlett
    # Columna 2 = hann
    # Columna 3 = blackman
    # Columna 4 = flat-top

plt.figure(1)

nbins = 20
colors = ['red', 'b', 'lime','y','m']
[plt.hist(f250[:,ii], nbins, density=True, histtype='bar', color=colors[ii], label=nameswindows[ii]) for ii in vw]
#plt.hist(f250[:,0], nbins, density=True, histtype='bar', color=colors, label=nameswindows)
plt.title('Valor del módulo de la DFT en la frecuencia '+'{:3.0f}'.format(omega0norm)+' Hz para las 5 ventanas utilizadas', fontsize=20)
plt.xlabel('Módulo DFT', fontsize=15)
plt.ylabel('Frecuencia', fontsize=15)
plt.legend(fontsize=15)

plt.show()



################################################################

medmuestral = np.mean(f250, axis=0)

sesgo = medmuestral - a0

varianza = np.empty((1, len(windows)))
for i in vw:
    varianza[:,i] = sum((f250[:,i] - medmuestral[i])**2)/len(fr)



resultados = [ 
                   [sesgo[0],varianza[0,0]],
                   [sesgo[1],varianza[0,1]], 
                   [sesgo[2],varianza[0,2]], 
                   [sesgo[3],varianza[0,3]], 
                   [sesgo[4],varianza[0,4]] 
                 ]
resultados
df = pd.DataFrame(resultados, columns=['$s_a$', '$v_a$'],
               index=[  
                        'Rectangular',
                        'Bartlett',
                        'Hann',
                        'Blackman',
                        'Flat-top'
                     ])
#HTML(df.to_html())
df.round(2)


################################### Le saco el sesgo y le sumo a0 para que quede centrado en a0

f250_corregido = np.zeros_like(f250)
for z in vw:
    f250_corregido [:,z] = f250[:,z]-sesgo[z]
    
plt.figure(2)

nbins = 20
colors = ['red', 'b', 'lime','y','m']
[plt.hist(f250_corregido[:,ii], nbins, density=True, histtype='bar', alpha=0.9, color=colors[ii], label=nameswindows[ii]) for ii in vw]
#plt.hist(f250[:,0], nbins, density=True, histtype='bar', color=colors, label=nameswindows)
plt.title('Valor del módulo de la DFT en la frecuencia '+'{:3.0f}'.format(omega0norm)+' Hz para las 5 ventanas utilizadas', fontsize=20)
plt.xlabel('Módulo DFT', fontsize=15)
plt.ylabel('Frecuencia', fontsize=15)
plt.legend(fontsize=15)

plt.show()