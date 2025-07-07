#Raman Analysis Code

###############
#Imports//Importations
###############
import numpy as np #arrays//tableaux
import pandas as pd #Results Handling in DataFrames
import scipy.constants as sc #scientific constants for gaussian etc.//constantes scientifiques pour gaussian etc.
import matplotlib.pyplot as plt #plotting//Traçage
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from pybaselines import Baseline
from scipy.integrate import quad

###############
#Functions for Deconvolution//Fonctions pour déconvolution
###############
#Single Gaussian 
def gaussian(x, A, mu, sigma):
    """
    Calculates a Gaussian peak profile.

    Args:
        x (np.ndarray): The independent variable (i.e. Raman shift).
        A (float): The amplitude (height) of the peak.
        mu (float): The center (position) of the peak.
        sigma (float): The standard deviation (width) of the peak.

    Returns:
        np.ndarray: The calculated Gaussian values corresponding to each x.
    """
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

#Single Lorentzian
def lorentzian(x, A, mu, fwhm):
    """
    Calculates a Lorentzian peak profile.

    Args:
        x (np.ndarray): The independent variable (i.e. Raman shift).
        A (float): The amplitude (height) of the peak.
        mu (float): The center (position) of the peak.
        fwhm (float): The Full Width at Half Maximum of the peak.

    Returns:
        np.ndarray: The calculated Lorentzian values corresponding to each x.
    """
    gamma = fwhm / 2  # gamma = Half-width at half-maximum
    return A*(gamma**2/((x-mu)**2 + gamma**2))

#First Order
#4 Lorentzians, 1 Gaussian [Sadezky et al. 2005]
def order1(x, A1, mu1, fwhm1, A2, mu2, fwhm2, A3, mu3, fwhm3, A4, mu4, fwhm4, A5, mu5, sigma5):
    """
    Constructs a model for the first-order Raman spectrum of carbon blacks.

    Combines four Lorentzian peaks and one Gaussian peak to model the
    first-order region (approx. 1000-1800 cm-1) of the Raman spectrum for
    disordered carbon materials. This deconvolution is based on the five-band
    model used for carbon blacks by [Sadezky et al. 2005]

    - Peak 1 (Lorentzian): D1 band
    - Peak 2 (Lorentzian): G band
    - Peak 3 (Lorentzian): D2 band
    - Peak 4 (Lorentzian): D4 band
    - Peak 5 (Gaussian):   D3 band

    Args:
        x (np.ndarray): The independent variable, representing the Raman shift in cm-1.

        A... (float): Amplitude of the band.
        mu... (float): Center position of the band.
        fwhm... (float): Full Width at Half Maximum of the band.

        sigma5 (float): Standard deviation of the D3 band. (as opposed to fwhm for the lorentzians)

    Returns:
        np.ndarray: The calculated total intensity of the composite spectrum
                    corresponding to each value in x.
    """
    return (lorentzian(x, A1, mu1, fwhm1) +
            lorentzian(x, A2, mu2, fwhm2) +
            lorentzian(x, A3, mu3, fwhm3) +
            lorentzian(x, A4, mu4, fwhm4) +
            gaussian(x, A5, mu5, sigma5))

#Second Order
#4 Lorentzians [Sadezky et al. 2005]
def order2(x, A1, mu1, fwhm1, A2, mu2, fwhm2, A3, mu3, fwhm3, A4, mu4, fwhm4):
    """
    Constructs a model for the second-order Raman spectrum of carbon blacks.

    Combines four Lorentzian peaks to model the second-order
    region (approx. 2000-3500 cm-1) of the Raman spectrum for disordered 
    carbon materials. This deconvolution is based on the four-band model used
    for carbon blacks by [Sadezky et al. 2005]

    Args:
        x (np.ndarray): The independent variable, representing the Raman shift in cm-1.

        A... (float): Amplitude of the band.
        mu... (float): Center position of the band.
        fwhm... (float): Full Width at Half Maximum of the band.

    Returns:
        np.ndarray: The calculated total intensity of the spectrum
                    corresponding to each value in x.
    """
    return (lorentzian(x, A1, mu1, fwhm1) +
            lorentzian(x, A2, mu2, fwhm2) +
            lorentzian(x, A3, mu3, fwhm3) +
            lorentzian(x, A4, mu4, fwhm4))

def chisquared(ys, model_ys):
    """
    Calculates the Chi Squared value for a model to help
    determine quality of fit.
    
    Args:
        ys (np.ndarray): The recorded data
        model_ys (np.ndarray): The data from the fitted model
        
    Returns:
        Chi Squared (float): The calculated chi squared value using ( ((ys - model_ys) /model_ys)**2 ).sum()
    """
    return ( ((ys - model_ys) /model_ys)**2 ).sum()


##################
#Input/Output Lists (format input file names as '{sample_name} Area {area_number} {laser_wavelength}.txt')
##################
sample_names = ['NOJP2', 'NOJP7a', 'NOJP9', 'NOJP12a', 'NOJP13', 'NOJP14']
area_numbers = [1,2,3,4,5]
laser_wavelengths = [532]

#List for results processing//listes pour le traitement des résultats
#To store a dictionary of results for each spectrum
all_results = []

#################
#Main Loop
#################
for sample_name in sample_names:
    for area_number in area_numbers:
        for laser_wavelength in laser_wavelengths:
            
            #Load Files//Charger des fichiers
            datafile = f'/<FILEPATH>/{sample_name} Area {area_number} {laser_wavelength}_fixed.txt'
            data = np.genfromtxt(datafile,delimiter='',unpack=True, skip_header = 0, dtype=float)

            #Split to x and y components//Diviser en composantes x et y
            x,y = data
            
            #defining a search area//définir une zone de recherche
            search_area_y = y[np.where(x > 1200)[0][0]:np.where(x > 1800)[0][0]] #peaks search area
            search_area_x = x[np.where(x > 1200)[0][0]:np.where(x > 1800)[0][0]] #curve fit search area
            
            fit_search_area_x = x[np.where(x > 500)[0][0]:np.where(x > 2000)[0][0]]
            fit_search_area_y = y[np.where(x > 500)[0][0]:np.where(x > 2000)[0][0]]
            
            
            #remove Fluoresecence slope
            #Baseline Correction around 1st order peaks
            baseline_fitter = Baseline(x_data=fit_search_area_x)
            half_window_1 = 15
            fit_1, params_1 = baseline_fitter.std_distribution(fit_search_area_y, half_window_1, smooth_half_window=10)
            fit_search_area_y = fit_search_area_y - fit_1
            
            #Around 2nd order peaks
            baseline_fitter_2 = Baseline(x_data=x)
            half_window_2 = 2
            fit_2, params_2 = baseline_fitter_2.std_distribution(y, half_window_2, smooth_half_window=10)
            edited_y = y - fit_2
            
            #defining a search area for 2nd Order peaks fit
            fit2_search_area_x = x[np.where(x > 2000)[0][0]:]
            fit2_search_area_y = edited_y[np.where(x > 2000)[0][0]:]
            
            #Plotting//Traçage
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(x,edited_y, zorder = 10, label = 'Raw Data minus Fluorescence Baseline')

            ax.set_xlim(800)
            ax.set_xlabel('Raman Shift (cm$^{-1}$)')
            ax.set_ylabel('Intensity (counts)')
            ax.set_title(f'Raman Spectrum: {sample_name} Area {area_number} {laser_wavelength}')

            #Find Peaks (pre-deconvolution)//Trouver des sommets
            if laser_wavelength == 532:
                temp_peaks, properties = find_peaks(search_area_y,
                                                    height= (search_area_y.mean()),
                                                    prominence=4,
                                                    distance=20)

            else: #Edit find_peaks as appropriate for other wavelengths//modifiez « find_peaks » selon vos besoins
                temp_peaks, properties = find_peaks(search_area_y,
                                                    height= (search_area_y.mean()),
                                                    prominence=6,
                                                    distance=80)
                

            peaks = [np.where(y == search_area_y[temp_peaks[0]])[0][0],np.where(y == search_area_y[temp_peaks[1]])[0][0]]

            #Find Local Minimum between First order Peaks//Trouver le minimum local
            valley_y = min(y[peaks[0]:peaks[1]])
            valley_x = (x[np.where(y == valley_y)[0]])[0]
            valley_x,valley_y
            
            #Plotting Bands
            ax.axvline(x[peaks[0]],color = 'r',label = f'D Band:{x[peaks[0]]}')
            ax.axvline(x[peaks[1]],color = 'g',label = f'G Band: {x[peaks[1]]}')
            ax.axvline(valley_x,color = 'y',label = f'Local Minimum: {valley_x}')
            
            #########################
            #Curve Fitting to Deconvolve Peaks
            #########################

            # Initial guesses for Curve Fitting: [A1, mu1, sigma1, A2, mu2, sigma2...]
            p0 = [y[peaks[0]], 1350, 10, y[peaks[1]], 1590, 10, y[peaks[0]], 1620, 10, y[peaks[1]], 1180, 10, y[peaks[0]]/2, 1500, 10]
            p02 = [y[peaks[0]], 2450, 10, y[peaks[1]], 2700, 10, y[peaks[0]], 2900, 10, y[peaks[1]], 3100, 10]

            #Bounds For curve_fit()
            lower_bounds = [0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0] 
            upper_bounds = [np.inf, np.inf, np.inf,np.inf, np.inf, np.inf,np.inf, np.inf, np.inf,np.inf, np.inf, np.inf,np.inf, np.inf, np.inf]
            bounds = (lower_bounds, upper_bounds)
            bounds2 = (lower_bounds[0:12],upper_bounds[0:12])
            try:    
                order1_fit, _ = curve_fit(order1, fit_search_area_x, fit_search_area_y, p0=p0, bounds = bounds, maxfev=10000)
                order1_fit_y = order1(fit_search_area_x, *order1_fit)

                order2_fit, _ = curve_fit(order2, fit2_search_area_x, fit2_search_area_y, p0=p02, maxfev = 10000, bounds = bounds2)
                order2_fit_y = order2(fit2_search_area_x, *order2_fit)
            
            #Error Handling
            except ValueError as e:
                print(f"Couldn't fit due to a data problem (ValueError). Moving to next sample.")
                print(f"  Details: {e}")

            except RuntimeError as e:
                print(f"Couldn't fit because the algorithm failed (RuntimeError). Moving to next sample.")
                print(f"  Details: {e}")

            except Exception as e:
                print(f"An UNEXPECTED error occurred: {type(e).__name__}. Moving to next sample.")
                print(f"  Details: {e}")

            
            else:
                
                #First Order Plots
                ax.plot(fit_search_area_x, order1_fit_y, label=f'Sadezkye Deconvolution', linestyle='--')
                ax.plot(x, lorentzian(x,order1_fit[0], order1_fit[1], order1_fit[2]),label = 'D1')
                ax.plot(x, lorentzian(x,order1_fit[3], order1_fit[4], order1_fit[5]),label = 'G')
                ax.plot(x, lorentzian(x,order1_fit[6], order1_fit[7], order1_fit[8]),label = 'D2')
                ax.plot(x, lorentzian(x,order1_fit[9], order1_fit[10], order1_fit[11]),label = 'D4')
                ax.plot(x, gaussian(x,order1_fit[12], order1_fit[13], order1_fit[14]),label = 'D3')

                #Second Order Plots
                ax.plot(fit2_search_area_x, order2_fit_y, label = '2nd Order Deconvolution')
                ax.plot(x, lorentzian(x, order2_fit[0], order2_fit[1], order2_fit[2]))
                ax.plot(x, lorentzian(x, order2_fit[3], order2_fit[4], order2_fit[5]))
                ax.plot(x, lorentzian(x, order2_fit[6], order2_fit[7], order2_fit[8]))
                ax.plot(x, lorentzian(x, order2_fit[9], order2_fit[10], order2_fit[11]))
                
                #########################
                #Integration for Area Under Deconvolved Peaks
                #########################

                first_order_total_A = quad(order1,fit_search_area_x[0],fit_search_area_x[-1], args = tuple(order1_fit))
                D1_A = quad(lorentzian,fit_search_area_x[0],fit_search_area_x[-1],args = (order1_fit[0], order1_fit[1], order1_fit[2]))
                D2_A = quad(lorentzian,fit_search_area_x[0],fit_search_area_x[-1],args = (order1_fit[6], order1_fit[7], order1_fit[8]))
                D3_A = quad(gaussian,fit_search_area_x[0],fit_search_area_x[-1],args = (order1_fit[12], order1_fit[13], order1_fit[14]))
                D4_A = quad(lorentzian,fit_search_area_x[0],fit_search_area_x[-1],args = (order1_fit[9], order1_fit[10], order1_fit[11]))
                G_A = quad(lorentzian,fit_search_area_x[0],fit_search_area_x[-1],args = (order1_fit[3], order1_fit[4], order1_fit[5]))

                #Chi Squared (Quality of Fit) Calculations 
                #(for first order, second order deconvolution is for validation of first order deconvolution positions only)
                chi_squared = chisquared(fit_search_area_y,order1_fit_y)

                current_results = {
                    'Name': f'{sample_name} Area {area_number} {laser_wavelength}nm',

                    #Un-deconvolved band positions
                    'D_Band_Raw': x[peaks[0]],
                    'G_Band_Raw': x[peaks[1]],
                    'Local_Minimum': valley_x,

                    #Fit Quality Monitoring
                    'Chi_Squared': chi_squared,
                    'Reduced_Chi_Squared': chi_squared / (len(fit_search_area_y) - len(order1_fit)),

                    #First Order Band Positions
                    'G_Position': order1_fit[4],
                    'D1_Position': order1_fit[1],
                    'D2_Position': order1_fit[7],
                    'D3_Position': order1_fit[13],
                    'D4_Position': order1_fit[10],

                    #First Order Band Heights
                    'G_Height': order1_fit[3],
                    'D1_Height': order1_fit[0],
                    'D2_Height': order1_fit[6],
                    'D3_Height': order1_fit[12],
                    'D4_Height': order1_fit[9],

                    #First Order Band Areas
                    'G_Area': G_A[0],
                    'D1_Area': D1_A[0],
                    'D2_Area': D2_A[0],
                    'D3_Area': D3_A[0],
                    'D4_Area': D4_A[0]
                }

                #Append the dictionary for this spectrum to our master list
                all_results.append(current_results)

            #Show Plot
            ax.legend()
            plt.show()

#to write to a .csv file//pour écrire dans un fichier .csv
#Convert the list of dictionaries into a pandas DataFrame
results_df = pd.DataFrame(all_results)

#Define the output file path
outputfile = '<FILEPATH>/Plot_1_egg_yolk.csv'

#Save the DataFrame to a CSV file.
#The `index=False` argument prevents pandas from writing a new index column.
results_df.to_csv(outputfile, index=False)

print("\n-------------------------------------------")
print(f"Analysis complete. Results saved to:\n{outputfile}")
print("-------------------------------------------")
