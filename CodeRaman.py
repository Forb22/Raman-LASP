#Raman Analysis Code

#Imports//Importations
import numpy as np #arrays//tableaux
import scipy.constants as sc #scientific constants for gaussian etc.//constantes scientifiques pour gaussian etc.
import matplotlib.pyplot as plt #plotting//Traçage
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from pybaselines import Baseline
from scipy.integrate import quad

#Functions for Deconvolution//Fonctions pour déconvolution
#Single Gaussian 
def gaussian(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

#Single Lorentzian
def lorentzian(x, A, mu, fwhm):
    gamma = fwhm / 2  # gamma = Half-width at half-maximum
    return A*(gamma**2/((x-mu)**2 + gamma**2))

#4 Lorentzians, 1 Gaussian
def sadezkyetal(x, A1, mu1, fwhm1, A2, mu2, fwhm2, A3, mu3, fwhm3, A4, mu4, fwhm4, A5, mu5, sigma5):
    return (lorentzian(x, A1, mu1, fwhm1) +
            lorentzian(x, A2, mu2, fwhm2) +
            lorentzian(x, A3, mu3, fwhm3) +
            lorentzian(x, A4, mu4, fwhm4) +
            gaussian(x, A5, mu5, sigma5) )

def order2(x, A1, mu1, fwhm1, A2, mu2, fwhm2, A3, mu3, fwhm3, A4, mu4, fwhm4):
    return (lorentzian(x, A1, mu1, fwhm1) +
            lorentzian(x, A2, mu2, fwhm2) +
            lorentzian(x, A3, mu3, fwhm3) +
            lorentzian(x, A4, mu4, fwhm4))

def chisquared(ys, model_ys, uncertainty):
    return ( ((ys - model_ys) /model_ys)**2 ).sum()
    
sample_names = ['NOJP2', 'NOJP7a', 'NOJP9', 'NOJP12a', 'NOJP13', 'NOJP14']
area_numbers = [1,2,3,4,5]
laser_wavelengths = [532]

#Lists for results processing//listes pour le traitement des résultats
d_bands = ['D Band']
g_bands = ['G Band']
minima = ['Local Minimum']
names = ['Name']
chi_squareds = []
reduced_chi_squareds = ['Reduced Chi Squared']
G = ['G']
D1 = ['D1']
D2 = ['D2']
D3 = ['D3']
D4 = ['D4']

for sample_name in sample_names:
    for area_number in area_numbers:
        for laser_wavelength in laser_wavelengths:
            
          #Load Files//Charger des fichiers
          datafile = f'/Users/guy/Desktop/Sherbrooke_Lab_Data/Raman Data/Fixed/{sample_name} Area {area_number} {laser_wavelength}_fixed.txt'
          data = np.genfromtxt(datafile,delimiter='',unpack=True, skip_header = 0, dtype=float)

          #Split to x and y components//Diviser en composantes x et y
          x,y = data
          
          #defining a search area//définir une zone de recherche
          search_area_y = y[np.where(x > 1200)[0][0]:np.where(x > 1800)[0][0]] #peaks search area
          search_area_x = x[np.where(x > 1200)[0][0]:np.where(x > 1800)[0][0]] #curve fit search area
          
          fit_search_area_x = x[np.where(x > 500)[0][0]:np.where(x > 2000)[0][0]]
          fit_search_area_y = y[np.where(x > 500)[0][0]:np.where(x > 2000)[0][0]]
          
          #remove Fluoresecence slope
          baseline_fitter = Baseline(x_data=fit_search_area_x)
          half_window_1 = 15
          fit_1, params_1 = baseline_fitter.std_distribution(fit_search_area_y, half_window_1, smooth_half_window=10)
          fit_search_area_y = fit_search_area_y - fit_1
          
          baseline_fitter_2 = Baseline(x_data=x)
          half_window_2 = 2
          fit_2, params_2 = baseline_fitter_2.std_distribution(y, half_window_2, smooth_half_window=10)
          edited_y = y - fit_2

          #Defining another search area for 2nd Order Peaks
          fit2_s_a_x = x[np.where(x > 2000)[0][0]:]
          fit2_s_a_y = edited_y[np.where(x > 2000)[0][0]:]
          
          #Plotting//Traçage
          fig = plt.figure()
          ax = fig.add_subplot()
          ax.plot(x,edited_y, zorder = 10)

          #ax.plot(fit2_s_a_x,fit2_s_a_y,label = 'Raw Data - Fluorescence Baseline')
          ax.set_xlim()
          ax.set_xlabel('Raman Shift (cm$^{-1}$)')
          ax.set_ylabel('Intensity (counts)')
          ax.set_title(f'Raman Spectrum: {sample_name} Area {area_number} {laser_wavelength}')
          
          if laser_wavelength == 532:
            #Find Peaks//Trouver des sommets
            temp_peaks, properties = find_peaks(search_area_y,
                                                height= (search_area_y.mean()),
                                                prominence=4,
                                                distance=20)

          else: #Edit find_peaks as appropriate//modifiez « find_peaks » selon vos besoins
            #Find Peaks//Trouver des sommets
            temp_peaks, properties = find_peaks(search_area_y,
                                                height= (search_area_y.mean()),
                                                prominence=6,
                                                distance=80)
              

          peaks = [np.where(y == search_area_y[temp_peaks[0]])[0][0],np.where(y == search_area_y[temp_peaks[1]])[0][0]]

          #Find Local Minima//Trouver le minimum local
          valley_y = min(y[peaks[0]:peaks[1]])
          valley_x = (x[np.where(y == valley_y)[0]])[0]
          valley_x,valley_y

          #Plot Bands
          ax.axvline(x[peaks[0]],color = 'r',label = f'D Band:{x[peaks[0]]}')
          ax.axvline(x[peaks[1]],color = 'g',label = f'G Band: {x[peaks[1]]}')
          ax.axvline(valley_x,color = 'y',label = f'Local Minimum: {valley_x}')
          
          
          #########################
          #Curve Fitting
          #########################
          
          # Initial guesses: [A1, mu1, sigma1, A2, mu2, sigma2...]
          p0 = [y[peaks[0]], 1350, 10, y[peaks[1]], 1590, 10, y[peaks[0]], 1620, 10, y[peaks[1]], 1180, 10, y[peaks[0]]/2, 1500, 10]
          
          p02 = [y[peaks[0]], 2450, 10, y[peaks[1]], 2700, 10, y[peaks[0]], 2900, 10, y[peaks[1]], 3100, 10]

          #Bounds For curve_fit()
          lower_bounds = [0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0]
          upper_bounds = [np.inf, np.inf, np.inf,np.inf, np.inf, np.inf,np.inf, np.inf, np.inf,np.inf, np.inf, np.inf,np.inf, np.inf, np.inf]
          bounds = (lower_bounds, upper_bounds)
          bounds2 = (lower_bounds[0:12],upper_bounds[0:12])
          try:    
            sadezkyetal_fit, _ = curve_fit(sadezkyetal, fit_search_area_x, fit_search_area_y, p0=p0, bounds = bounds, maxfev=10000)
            sadezkyetal_fit_y = sadezkyetal(fit_search_area_x, *sadezkyetal_fit)
  
            order2_fit, _ = curve_fit(order2, fit2_s_a_x, fit2_s_a_y, p0=p02, maxfev = 10000, bounds = bounds2)
            order2_fit_y = order2(fit2_s_a_x, *order2_fit)
            
          except Exception:
            print(f"Couldn't Fit {sample_name} Area {area_number} {laser_wavelength}! Moving to the next sample") 
            pass

          else:
            #First Order Plotting
            ax.plot(fit_search_area_x, sadezkyetal_fit_y, label=f'Sadezkye Deconvolution', linestyle='--')
            ax.plot(x, lorentzian(x,sadezkyetal_fit[0], sadezkyetal_fit[1], sadezkyetal_fit[2]),label = 'D1')
            ax.plot(x, lorentzian(x,sadezkyetal_fit[3], sadezkyetal_fit[4], sadezkyetal_fit[5]),label = 'G')
            ax.plot(x, lorentzian(x,sadezkyetal_fit[6], sadezkyetal_fit[7], sadezkyetal_fit[8]),label = 'D2')
            ax.plot(x, lorentzian(x,sadezkyetal_fit[9], sadezkyetal_fit[10], sadezkyetal_fit[11]),label = 'D4')
            ax.plot(x, gaussian(x,sadezkyetal_fit[12], sadezkyetal_fit[13], sadezkyetal_fit[14]),label = 'D3')
  
            #Second Order Plotting
            ax.plot(fit2_s_a_x, order2_fit_y, label = '2nd Order Deconvolution')
            ax.plot(x, lorentzian(x, order2_fit[0], order2_fit[1], order2_fit[2]))
            ax.plot(x, lorentzian(x, order2_fit[3], order2_fit[4], order2_fit[5]))
            ax.plot(x, lorentzian(x, order2_fit[6], order2_fit[7], order2_fit[8]))
            ax.plot(x, lorentzian(x, order2_fit[9], order2_fit[10], order2_fit[11]))
            
            G.append(sadezkyetal_fit[4])
            D4.append(sadezkyetal_fit[10])
            D2.append(sadezkyetal_fit[7])
            D1.append(sadezkyetal_fit[1])
            D3.append(sadezkyetal_fit[13])
            
          #Show Plot
          ax.legend()
          plt.show()
          

          #Chi Squared (Quality of Fit) Calculations
          R = (min(fit_search_area_y) - max(fit_search_area_y))
          y_err = np.ones(len(fit_search_area_y))*(R/(2*np.sqrt(len(fit_search_area_y))))
          chi_squared = chisquared(fit_search_area_y,sadezkyetal_fit_y,y_err)
          
          # print(f'Calculated Chi Squared: {chi_squared}')
          # print(f'Reduced Chi Squared: {chi_squared/(len(fit_search_area_y) - len(bounds))}')
          # print(f'Expected Chi Squared: {len(fit_search_area_y) - len(bounds)} ± {np.sqrt(2*(len(fit_search_area_y) - len(bounds)))}')

          names.append(f'{sample_name} Area {area_number} {laser_wavelength}nm')
          d_bands.append(x[peaks[0]])
          g_bands.append(x[peaks[1]])
          minima.append(valley_x)
          chi_squareds.append(chi_squared)
          reduced_chi_squareds.append(chi_squared/(len(fit_search_area_y) - len(bounds)))

# chi_squareds = np.array(chi_squareds)
# print(chi_squareds.mean())
# print(max(chi_squareds) - min(chi_squareds))
#to write to a .csv file//pour écrire dans un fichier .csv
results = np.array([names,d_bands,g_bands,minima,reduced_chi_squareds,G,D1,D2,D3,D4])
outputfile = '/Users/guy/Desktop/Sherbrooke_Lab_Data/Plot_1_egg_yolk.csv'
np.savetxt(outputfile, results, delimiter=',', fmt = '%s')
