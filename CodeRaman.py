#Raman Analysis Code

#Imports//Importations
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import deconvolve

#Load Files//Charger des fichiers   
datafile = '<PATH>'
data = np.genfromtxt(datafile,delimiter='',unpack=True, skip_header = 0, dtype=float)

#Split to x and y components//Diviser en composantes x et y
x,y = data

#Plotting//Traçage
fig = plt.figure()
ax = fig.add_subplot()

ax.plot(x,y)
ax.set_xlim(0)
ax.set_xlabel('Raman Shift (cm$^{-1}$)')
ax.set_ylim(0)
ax.set_ylabel('Intensity (counts)')
ax.set_title('Un-deconvolved Raman Spectrum')

#Find Peaks//Trouver des sommets
peaks, properties = find_peaks(y,
                               height=y.mean(),
                               prominence=6,
                               distance=20)

#Find Local Minimum//Trouver le minimum local
valley_y = min(y[peaks[0]:peaks[1]])
valley_x = (x[np.where(y == valley_y)[0]])[0]
valley_x,valley_y

ax.axvline(x[peaks[0]],color = 'r',label = f'D Band:{x[peaks[0]]}')
ax.axvline(x[peaks[1]],color = 'g',label = f'G Band: {x[peaks[1]]}')
ax.axvline(valley_x,color = 'y',label = f'Local Minimum: {valley_x}')
#ax.axvline(x[peaks[2]],color = 'pink',label = f'D Band:{x[peaks[2]]}')
ax.legend()
plt.show()

#Print Results//Imprimer les résultats
print(f'D-Band: {x[peaks[0]]}')
print(f'G Band: {x[peaks[1]]}')
print(f'Local Minimum: {valley_x}')
