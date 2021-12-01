# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 20:50:22 2021

@author: ADRIAN CAFA
"""

import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')
#interactive plotting in separate window
%matplotlib qt


from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import plotly.graph_objects as go
from IPython.display import HTML
import pandas as pd
import scipy.signal as sig
import scipy.io as sio
import scipy
import pylab
# Setup inline graphics: Esto lo hacemos para que el tamaño de la salida, 
# sea un poco más adecuada al tamaño del documento
mpl.rcParams['figure.figsize'] = (15,10)

plt.close('all')

# Para listar las variables que hay en el archivo
print(sio.whosmat('ecg.mat'))

# Variables
# [('ecg_lead', (1129116, 1), 'double'), 
# ('qrs_pattern1', (113, 1), 'double'), 
# ('heartbeat_pattern1', (411, 1), 'double'), 
# ('heartbeat_pattern2', (441, 1), 'double'), 
# ('qrs_detections', (1903, 1), 'double')]

mat_struct = sio.loadmat('ecg.mat')
fs = 1000 #Hz
#ecg = mat_struct['ecg_lead']
ecg = mat_struct['ecg_lead']
qrs = mat_struct['qrs_detections']

d_muestras1 = 200
d_muestras2 = 350

#%%Armo mi matriz de latidos con el delta muestras para adelante y atras
#la matriz contiene tantos elementos coomo qrs_detections de intervalo 
#d_muestras1+d_muestras2
latidos_matrix= [ (ecg[int(i-d_muestras1):int(i+d_muestras2)]) for i in qrs ]
##Con esto me aseguro que quede como array y no como lista de arrays
array_latidos=np.hstack(latidos_matrix)

array_latidos=array_latidos - np.mean(array_latidos,axis=0)
#t = scipy.arange(len(ecg))/fs
#pylab.plot(t, ecg)
#pylab.show()

#t = scipy.arange(len(array_latidos))/fs

array_latidos = array_latidos[:,:50] - np.mean(array_latidos[:,:50],axis=0)

array_latidos_padded = np.pad(array_latidos,pad_width=((2000,2000),(0,0)),mode='constant')


#plt.plot(array_latidos)
plt.plot(array_latidos_padded)




# #%%Verifico encontrar un latido
# plt.figure(figura)
# plt.plot(muestras_tot,ecg)
# figura+=1
# pos=qrs_detections[1000]
# d_muestras1=200
# d_muestras2=300
# plt.xlim(pos-d_muestras1,pos+d_muestras2)

###############################################################################




#qrs_detections = mat_struct['qrs_detections']
#win = np.round(np.array([.15,.35])*fs)








