#Raman Analysis Code

###############
#Imports//Importations
###############

import numpy as np #arrays//tableaux
import pandas as pd #Results Handling in DataFrames
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

def gaussian_area(A, sigma):
    """Calculates the area of a Gaussian peak profile.

    Args:
        A (float): The amplitude (height) of the peak.
        sigma (float): The standard deviation (width) of the peak.

    Returns:
        float: The area of the Gaussian peak profile.
        """
    return A*(sigma/np.sqrt(2*np.pi))

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

def lorentzian_area(A,fwhm):
    """
    Calculates the area of a lorentzian peak profile

    Args:
        A (float): The amplitude (height) of the peak
        fwhm (float): The full width at half maximum

    Returns:
        float: the area of the Lorentzian peak profile
    """
    return A*(np.pi/2)*fwhm

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

def chisquared(ys, model_ys, y_err):
    """
    Calculates the Chi Squared value for a model to help
    determine quality of fit.

    Args:
        ys (np.ndarray): The recorded data
        model_ys (np.ndarray): The data from the fitted model
        y_err (np.ndarray): The Measurement errors on the data

    Returns:
        Chi Squared (float): The calculated chi squared value using ( ((ys - model_ys) / y_err)**2 ).sum()
    """
    return ( ((ys - model_ys) / y_err)**2 ).sum()


##################
#Input/Output Lists (format input file names as '{sample_name} Area {area_number} {laser_wavelength}.txt')
##################
sample_names = ['NOBP6','NOBP7a','NOBP10','NOBP11c','NOBP13','NOJP2','NOJP7a','NOJP5a','NOJP9','NOJP12a','NOJP13','NOJP14','NOMP13','NOMP3a','NOMP6a','NOMP8','NOMP12', 'NOJC1a','NOJC2','NOJC3']
area_numbers = [1,2,3,4,5]
laser_wavelengths = [532]

######################
#Adjustible Parameters
######################

DATA_TRIM_START = 0
DATA_TRIM_END = 3500

BASELINE_FITTER_LAMBDA = 1e6 #Strictness of the baseline fitter curve. Bigger -> Stricter

FIND_PEAKS_SEARCH_AREA_START = 1200 #Pre-Deconvolution
FIND_PEAKS_SEARCH_AREA_END = 1750

FIRST_ORDER_SEARCH_AREA_START = 500
FIRST_ORDER_SEARCH_AREA_END = 2000

SECOND_ORDER_SEARCH_AREA_START = 2000
#SECOND_ORDER_SEARCH_AREA_END left blank for now to have the fit go between SECOND_ORDER_SEARCH_AREA_START and the end of the data

PLOT_X_START = 100
PLOT_X_END = 2000

FIND_PEAKS_PROMINENCE = 4

GAIN = 5.172
READOUT_NOISE_ELECTRONS = 4

#Define the output file path
OUTPUTFILE = '/Users/guy/Desktop/Sherbrooke_Lab_Data/Raman Data/Master Data TST.csv'

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
            datafile = f'/Users/guy/Desktop/Sherbrooke_Lab_Data/Raman Data/ALL 532/{sample_name} Area {area_number} {laser_wavelength}_01.txt'
            #datafile = f'/Users/guy/Desktop/Sherbrooke_Lab_Data/T+F3/{sample_name} Site {area_number}_01.txt'
            data = np.genfromtxt(datafile,delimiter='',unpack=True, skip_header = 0, dtype=float)

            #Split to x and y components//Diviser en composantes x et y
            x,y = data

            #Use to trim data
            #y = y[np.where(x > DATA_TRIM_START)[0][0]:np.where(x > DATA_TRIM_END)[0][0]]
            #x = x[np.where(x > DATA_TRIM_START)[0][0]:np.where(x > DATA_TRIM_END)[0][0]]

            #remove Fluoresecence slope
            baseline_fitter_2 = Baseline(x_data=x)
            fit_2, params_2 = baseline_fitter_2.arpls(y, lam = BASELINE_FITTER_LAMBDA)
            edited_y = y - fit_2

            #defining a search area for pre-deconvolution peaks//définir une zone de recherche
            search_area_y = edited_y[np.where(x > FIND_PEAKS_SEARCH_AREA_START)[0][0]:np.where(x > FIND_PEAKS_SEARCH_AREA_END)[0][0]]
            search_area_x = x[np.where(x > FIND_PEAKS_SEARCH_AREA_START)[0][0]:np.where(x > FIND_PEAKS_SEARCH_AREA_END)[0][0]]

            #defining a search area for 1st order peaks fit
            fit_search_area_x = x[np.where(x > FIRST_ORDER_SEARCH_AREA_START)[0][0]:np.where(x > FIRST_ORDER_SEARCH_AREA_END)[0][0]]
            fit_search_area_y = edited_y[np.where(x > FIRST_ORDER_SEARCH_AREA_START)[0][0]:np.where(x > FIRST_ORDER_SEARCH_AREA_END)[0][0]]

            #defining a search area for 2nd Order peaks fit
            fit2_search_area_x = x[np.where(x > SECOND_ORDER_SEARCH_AREA_START)[0][0]:]
            fit2_search_area_y = edited_y[np.where(x > SECOND_ORDER_SEARCH_AREA_START)[0][0]:]

            #Plotting//Traçage
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(x,edited_y, zorder = 10, label = 'Raw Data minus Fluorescence')
            ax.plot(x,y, label = 'Raw Data')
            ax.set_xlim(PLOT_X_START, PLOT_X_END)
            ax.set_xlabel('Raman Shift (cm$^{-1}$)')
            ax.set_ylabel('Intensity (counts)')
            ax.set_title(f'Raman Spectrum: {sample_name} Area {area_number} {laser_wavelength}')

            #Find Peaks (pre-deconvolution)//Trouver des sommets
            if laser_wavelength == 532:
                temp_peaks, properties = find_peaks(search_area_y,
                                                    height= (search_area_y.mean()),
                                                    prominence=FIND_PEAKS_PROMINENCE,
                                                    distance=len(edited_y)/25)

            else: #Edit find_peaks as appropriate for other wavelengths//modifiez « find_peaks » selon vos besoins
                temp_peaks, properties = find_peaks(search_area_y,
                                                    height= (search_area_y.mean()),
                                                    prominence=FIND_PEAKS_PROMINENCE,
                                                    distance=len(edited_y)/25)


            peaks = [np.where(edited_y == search_area_y[temp_peaks[0]])[0][0],np.where(edited_y == search_area_y[temp_peaks[1]])[0][0]]

            max_height = max(edited_y[peaks[0]],edited_y[peaks[1]])

            #Find Local Minimum between First order Peaks//Trouver le minimum local
            valley_y = min(edited_y[peaks[0]:peaks[1]])
            valley_x = (x[np.where(edited_y == valley_y)[0]])[0]

            #Plotting Bands
            ax.axvline(x[peaks[0]],color = 'r',label = f'D Band:{x[peaks[0]]}')
            ax.axvline(x[peaks[1]],color = 'g',label = f'G Band: {x[peaks[1]]}')
            ax.axvline(valley_x,color = 'y',label = f'Local Minimum: {valley_x}')

            #########################
            #Curve Fitting to Deconvolve Peaks
            #########################

            # Initial guesses for Curve Fitting: [A1, mu1, sigma1, A2, mu2, sigma2...]
            p0 = [max_height/2,  1350, 50, #D1
                  max_height/2,  1580, 50,  #G
                  max_height/2,  1620, 50,  #D2
                  max_height/5,  1200, 200, #D4
                  max_height/10, 1530, 50] #D3

            p02 = [max_height/2, 2450, 10,
                   max_height/2, 2700, 10,
                   max_height/2, 2900, 10,
                   max_height/2, 3100, 10]

            #Bounds For curve_fit()
            lower_bounds_1 = [0, 1300, 0, #D1
                              0, 1530, 0, #G
                              0, 1570, 0, #D2
                              0, 1100, 0, #D4
                              0, 1450, 0] #D3

            upper_bounds_1 = [max_height, 1400, np.inf, #D1
                              max_height, 1630, np.inf, #G
                              max_height, 1670, np.inf, #D2
                              max_height, 1250, np.inf, #D4
                              max_height, 1550, 100] #D3
            bounds_1 = (lower_bounds_1, upper_bounds_1)

            lower_bounds_2 = [0, 2250, 0, #D1
                              0, 2250, 0, #G
                              0, 2250, 0, #D2
                              0, 2250, 0] #D4

            upper_bounds_2 = [max_height, 3500, np.inf,
                              max_height, 3500, np.inf,
                              max_height, 3500, np.inf,
                              max_height, 3500, np.inf]

            bounds_2 = (lower_bounds_2,upper_bounds_2)

            try:
                order1_fit, _ = curve_fit(order1, fit_search_area_x, fit_search_area_y, p0=p0, bounds = bounds_1, maxfev=1000, loss = 'huber')
                order1_fit_y = order1(fit_search_area_x, *order1_fit)

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
                ax.plot(fit_search_area_x, order1_fit_y, label=f'1st Order Deconvolution', linestyle='--')
                ax.plot(x, lorentzian(x,order1_fit[0], order1_fit[1], order1_fit[2]),label = 'D1')
                ax.plot(x, lorentzian(x,order1_fit[3], order1_fit[4], order1_fit[5]),label = 'G')
                ax.plot(x, lorentzian(x,order1_fit[6], order1_fit[7], order1_fit[8]),label = 'D2')
                ax.plot(x, lorentzian(x,order1_fit[9], order1_fit[10], order1_fit[11]),label = 'D4')
                ax.plot(x, gaussian(x,order1_fit[12], order1_fit[13], order1_fit[14]),label = 'D3', color = 'y')

                #########################
                #Integration for Area Under Deconvolved Peaks
                #########################

                first_order_total_A = quad(order1,fit_search_area_x[0],fit_search_area_x[-1], args = tuple(order1_fit))
                D1_A = lorentzian_area(order1_fit[0], order1_fit[2])
                D2_A = lorentzian_area(order1_fit[6], order1_fit[8])
                D3_A = gaussian_area(order1_fit[12], order1_fit[14])
                D4_A = lorentzian_area(order1_fit[9], order1_fit[11])
                G_A = lorentzian_area(order1_fit[3], order1_fit[5])


                #Chi Squared (Quality of Fit) Calculations
                #(for first order, second order deconvolution is for validation of first order deconvolution positions only)


                readout_noise_counts = READOUT_NOISE_ELECTRONS/GAIN

                total_electrons = (y[np.where(x > FIRST_ORDER_SEARCH_AREA_START)[0][0]:np.where(x > FIRST_ORDER_SEARCH_AREA_END)[0][0]] * GAIN)
                baseline_electrons = (fit_2[np.where(x > FIRST_ORDER_SEARCH_AREA_START)[0][0]:np.where(x > FIRST_ORDER_SEARCH_AREA_END)[0][0]] * GAIN)

                shot_noise_electrons = np.sqrt(total_electrons + baseline_electrons)
                shot_noise_counts = shot_noise_electrons/(GAIN)

                y_err = np.sqrt((shot_noise_counts**2) + (readout_noise_counts**2))
                chi_squared = chisquared(fit_search_area_y,order1_fit_y,y_err)

                current_results = {
                    'Name': f'{sample_name} Area {area_number} {laser_wavelength}nm',

                    #Fit Quality Monitoring
                    'Chi_Squared': chi_squared,
                    'Reduced_Chi_Squared': chi_squared / (len(fit_search_area_y) - len(order1_fit)),

                    #Un-deconvolved band positions
                    'D_Band_Raw': x[peaks[0]],
                    'G_Band_Raw': x[peaks[1]],
                    'Local_Minimum': valley_x,

                    #Un-Deconvolved band heights
                    'D_Band_Raw_Height': y[peaks[0]],
                    'G_Band_Raw_Height': y[peaks[1]],
                    'Local_Minimum_Height': valley_y,

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
                    'G_Area': G_A,
                    'D1_Area': D1_A,
                    'D2_Area': D2_A,
                    'D3_Area': D3_A,
                    'D4_Area': D4_A,

                    ## Height Ratios (e.g., D1_Height / G_Height)
                    # Denominator: G_Height (order1_fit[3])
                    'D1_G_Height_Ratio': order1_fit[0] / order1_fit[3] if order1_fit[3] != 0 else None,
                    'D2_G_Height_Ratio': order1_fit[6] / order1_fit[3] if order1_fit[3] != 0 else None,
                    'D3_G_Height_Ratio': order1_fit[12] / order1_fit[3] if order1_fit[3] != 0 else None,
                    'D4_G_Height_Ratio': order1_fit[9] / order1_fit[3] if order1_fit[3] != 0 else None,

                    # Denominator: D1_Height (order1_fit[0])
                    'D2_D1_Height_Ratio': order1_fit[6] / order1_fit[0] if order1_fit[0] != 0 else None,
                    'D3_D1_Height_Ratio': order1_fit[12] / order1_fit[0] if order1_fit[0] != 0 else None,
                    'D4_D1_Height_Ratio': order1_fit[9] / order1_fit[0] if order1_fit[0] != 0 else None,

                    # Denominator: D2_Height (order1_fit[6])
                    'D3_D2_Height_Ratio': order1_fit[12] / order1_fit[6] if order1_fit[6] != 0 else None,
                    'D4_D2_Height_Ratio': order1_fit[9] / order1_fit[6] if order1_fit[6] != 0 else None,

                    # Denominator: D3_Height (order1_fit[12])
                    'D4_D3_Height_Ratio': order1_fit[9] / order1_fit[12] if order1_fit[12] != 0 else None,

                    ## Area Ratios (e.g., D1_Area / G_Area)
                    # Denominator: G_Area (G_A)
                    'D1_G_Area_Ratio': D1_A / G_A if G_A != 0 else None,
                    'D2_G_Area_Ratio': D2_A / G_A if G_A != 0 else None,
                    'D3_G_Area_Ratio': D3_A / G_A if G_A != 0 else None,
                    'D4_G_Area_Ratio': D4_A / G_A if G_A != 0 else None,

                    # Denominator: D1_Area (D1_A)
                    'D2_D1_Area_Ratio': D2_A / D1_A if D1_A != 0 else None,
                    'D3_D1_Area_Ratio': D3_A / D1_A if D1_A != 0 else None,
                    'D4_D1_Area_Ratio': D4_A / D1_A if D1_A != 0 else None,

                    # Denominator: D2_Area (D2_A)
                    'D3_D2_Area_Ratio': D3_A / D2_A if D2_A != 0 else None,
                    'D4_D2_Area_Ratio': D4_A / D2_A if D2_A != 0 else None,

                    # Denominator: D3_Area (D3_A)
                    'D4_D3_Area_Ratio': D4_A / D3_A if D3_A != 0 else None,
                }

                #Append the dictionary for this spectrum to master list
                all_results.append(current_results)

                try:
                    order2_fit, _ = curve_fit(order2, fit2_search_area_x, fit2_search_area_y, p0=p02, maxfev = 100, bounds = bounds_2)
                    order2_fit_y = order2(fit2_search_area_x, *order2_fit)
                    #Error Handling
                except ValueError as e:
                    print(f"Couldn't fit 2nd Order due to a data problem (ValueError). Moving to next sample.")
                    print(f"  Details: {e}")

                except RuntimeError as e:
                    print(f"Couldn't fit 2nd Order because the algorithm failed (RuntimeError). Moving to next sample.")
                    print(f"  Details: {e}")

                except Exception as e:
                    print(f"An UNEXPECTED error in 2nd Order Fit occurred: {type(e).__name__}. Moving to next sample.")
                    print(f"  Details: {e}")

                else:
                    #Second Order Plots
                    #ax.plot(fit2_search_area_x, order2_fit_y, label = '2nd Order Deconvolution')
                    ax.plot(x, lorentzian(x, order2_fit[0], order2_fit[1], order2_fit[2]))
                    ax.plot(x, lorentzian(x, order2_fit[3], order2_fit[4], order2_fit[5]))
                    ax.plot(x, lorentzian(x, order2_fit[6], order2_fit[7], order2_fit[8]))
                    ax.plot(x, lorentzian(x, order2_fit[9], order2_fit[10], order2_fit[11]))



            #Show Plot
            ax.legend()
            plt.savefig(f'{sample_name}_{area_number}_{laser_wavelength}.png')
            plt.close()

#to write to a .csv file//pour écrire dans un fichier .csv
#Convert the list of dictionaries into a pandas DataFrame
results_df = pd.DataFrame(all_results)

#Save the DataFrame to a CSV file.
#The `index=False` argument prevents pandas from writing a new index column.
results_df.to_csv(OUTPUTFILE, index=False)

print("\n-------------------------------------------")
print(f"Analysis complete. Results saved to:\n{OUTPUTFILE}")
print("-------------------------------------------")
