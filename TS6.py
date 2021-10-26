# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 18:36:08 2021

@author: ADRIAN CAFA
"""
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from pandas import DataFrame
from IPython.display import HTML
#interactive plotting in separate window
%matplotlib qt

plt.close('all')

N = 1000
Npad = 10 * N
fs = 1000
f = np.linspace(0, (N-1)*(fs/N), N)
f_pad = np.linspace(0, (Npad-1)*(fs/(Npad)), Npad)
t_pad = np.linspace(0, (Npad-1)/fs, Npad)
t = np.linspace(0, (N-1)/fs, N)



# 1) Implemente las funciones para generar las siguientes ventanas w(k):

# Bartlett ó triangular Hann Blackman Flat-top Ayuda: Puede consultar el capítulo 7.3 del mismo libro para obtener las 
# expresiones analíticas de cada ventana.
# a) Para cada ventana grafique w(k) y |W(Ω)|, para N = 1000 muestras, normalizando w(k) de forma tal que la amplitud máxima de 
# |W(Ω)| (lóbulo principal) sea 0 dB.

# Ayuda: Posiblemente convenga utilizar zero-padding para visualizar mejor |W(Ω)|.

####################################################################
###########        Rectangular             #########################
####################################################################

box = signal.windows.boxcar(N)

plt.figure(1)

plt.plot(t,box)
plt.title("Ventana rectangular")
plt.ylabel("Mag")
plt.xlabel("tiempo [seg]")
axes_hdl = plt.gca()
plt.show()

box_fft = np.abs((1/N)*fft(box, n= Npad)) # veo el modulo por eso no debo multiplicar por 2
maxElement_box_fft = np.amax(box_fft)
box_fft = box_fft/maxElement_box_fft # Normalizo para que la pot max sea 1 dB

plt.figure(2)

plt.plot(f_pad, 20* np.log10(box_fft),':x', lw=2, label='$ k0 = $' + 'box')
plt.title("PSD en dB - Ventana rectangular")
plt.ylabel("PSD [dB]")
plt.xlabel("frec[Hz]")
plt.xlim(0,10)
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()



####################################################################
###########        Bartlett ó triangular   #########################
####################################################################

barlett = signal.windows.bartlett(N)

plt.figure(3)

plt.plot(t,barlett)
plt.title("Ventana de Barlett")
plt.ylabel("Magnitud")
plt.xlabel("tiempo [segs]")

axes_hdl = plt.gca()

plt.show()

barlett_fft = np.abs((1/N)*fft(barlett, n= Npad)) # veo el modulo por eso no debo multiplicar por 2
maxElement_barlett_fft = np.amax(barlett_fft)
barlett_fft = barlett_fft/maxElement_barlett_fft # Normalizo para que la pot max sea 1 dB



plt.figure(4)

plt.plot(f_pad, 20* np.log10(barlett_fft),':x', lw=2, label='$ k0 = $' + 'barlett')
plt.title("PSD en dB - Ventana Barlett")
plt.ylabel("PSD [dB]")
plt.xlabel("frec[Hz]")
plt.xlim(0,10)
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


####################################################################
###########        Hann                    #########################
####################################################################

hann = signal.windows.hann(N)

plt.figure(5)

plt.plot(t,hann)
plt.title("Ventana de Hann")
plt.ylabel("Magnitud")
plt.xlabel("tiempo [segs]")
axes_hdl = plt.gca()
plt.show()


hann_fft = np.abs((1/N)*fft(hann, n= Npad)) # veo el modulo por eso no debo multiplicar por 2
maxElement_hann_fft = np.amax(hann_fft)
hann_fft = hann_fft/maxElement_hann_fft # Normalizo para que la pot max sea 1 dB

plt.figure(6)

plt.plot(f_pad, 20* np.log10(hann_fft),':x', lw=2, label='$ k0 = $' + 'hann')
plt.title("PSD en dB - Ventana hann")
plt.ylabel("PSD [dB]")
plt.xlabel("frec[Hz]")
plt.xlim(0,10)
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


####################################################################
###########        Blackman                #########################
####################################################################

blackman = signal.windows.blackman(N)

plt.figure(7)

plt.plot(t,blackman)
plt.title("Ventana blackman")
plt.ylabel("Magnitud")
plt.xlabel("tiempo [segs]")
axes_hdl = plt.gca()

plt.show()

blackman_fft = np.abs((1/N)*fft(blackman, n= Npad)) # veo el modulo por eso no debo multiplicar por 2
maxElement_blackman_fft = np.amax(blackman_fft)
blackman_fft = blackman_fft/maxElement_blackman_fft # Normalizo para que la pot max sea 1 dB

plt.figure(8)

plt.plot(f_pad, 20* np.log10(blackman_fft),':x', lw=2, label='$ k0 = $' + 'blackman')
plt.title("PSD en dB - Ventana blackman")
plt.ylabel("PSD [dB]")
plt.xlabel("frec[Hz]")
plt.xlim(0,10)
axes_hdl = plt.gca()
axes_hdl.legend()

plt.show()

####################################################################
###########        Flat-top                #########################
####################################################################

flattop = signal.windows.flattop(N)

plt.figure(9)

plt.plot(t,flattop)
plt.title("Ventana flattop")
plt.ylabel("Magnitud")
plt.xlabel("tiempo [segs]")
axes_hdl = plt.gca()
plt.show()


flattop_fft = np.abs((1/N)*fft(flattop, n= Npad)) # veo el modulo por eso no debo multiplicar por 2
maxElement_flattop_fft = np.amax(flattop_fft)
flattop_fft = flattop_fft/maxElement_flattop_fft # Normalizo para que la pot max sea 1 dB

plt.figure(10)

plt.plot(f_pad, 20* np.log10(flattop_fft),':x', lw=2, label='$ k0 = $' + 'flattop')
plt.title("PSD en dB - Ventana flattop")
plt.ylabel("PSD [dB]")
plt.xlabel("frec[Hz]")
plt.xlim(0,10)
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


############################## Resultados


# b) Construya una tabla en la que figuren dos mediciones de la anchura del lóbulo principal de cada ventana:

# Ω0 será la frecuencia del primer cruce por cero, es decir |W(Ω)| =0 y Ω1, aquella frecuencia tal que |W(Ω1)|=2√2∨−3 dB y el valor 
# de transferencia máxima de los lóbulos secundarios (expresado en dB) W2=max{|W(Ω)|}, ∀ Ω≥Ω0
# Sugerencia: Normalice Ω0,1 por Δf=fS√N para facilitar la comparación.
# Descripción de las ventanas

# Descripción de las ventanas

# Ω0 Ω1 W2 Rectangular
# Bartlett
# Hann
# Blackman
# Flat-top


#######################################

#| window function | Ω0 | Ω1 | W2 |
#|---|:---:|:---:|:---:|
#| [rectangular](#Rectangular-Window) | 1 | 0.5 | -12.5 dB |
#| [Barlett](#Triangular-Window) | 2 | 0.6 | -26 |
#| [Hann](#Hann-Window) | 2 | 0.71 | -31 |
#| [Blackman](#Blackman-Window) | 3 | 0.8 | -57.5 |
#| [Flat-Top](#Flat-Top-Window) | 5.7 | 1.9 | -92.9 |





