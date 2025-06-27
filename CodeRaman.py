#Raman Analysis Code

#Imports//Importations
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import deconvolve

#Functions//Fonctions
 # def CheckForText(data):
 #  """
 #  Checks file for text at any point during it
 #  Args:
 #  data : np.ndarray
 #        The datafile being checked

 #  Returns : 
 #  result : int
 #          1 if there is an error
 #          0 if there is no error
 #  """
  

#Load Files//Charger des fichiers
datafile = '<PATH>'
data = np.genfromtxt(datafile,delimiter=' ',unpack=True)

#Replace Commas With Points//Remplacer les virgules par des points
data = np.char.replace(data,",",".")

#Split to x and y components//Diviser en composantes x et y
x,y = data

#Checks//Vérifications
# if CheckForText(data) == 1:
#   print("Ensure File has no headers or other text//Assurez-vous que le fichier ne contient pas d'en-têtes ou d'autre texte")

#Plotting//Traçage
fig = plt.figure()
ax = fig.add_subplot()

ax.plot(x,y)
ax.set_xlim(0)
ax.set_xlabel('Raman Shift (cm$^{-1}$)')
ax.set_ylim(0)
ax.set_ylabel('Intensity (counts)')
ax.set_title('Un-deconvolved Raman Spectrum')
plt.show()

#Finding Peaks//Trouver des sommets
find_peaks(data,
           height=,
           prominence=)

#




