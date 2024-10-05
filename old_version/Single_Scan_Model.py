import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def fit_and_save_spectra(waves, intensities, x_val, y_val):
    """
    Fits the spectra, calculates ID/IG ratios, saves data to files, and returns relevant data.

    Parameters:
    - waves: array-like, wavelengths of the Raman spectra.
    - intensities: array-like, intensity values corresponding to the wavelengths.
    - x_val: float, X coordinate value.
    - y_val: float, Y coordinate value.

    Returns:
    - y_fit_id: array, fitted values for I-D peak.
    - y_fit_ig: array, fitted values for I-G peak.
    - y_fit_id2: array, fitted values for I-D2 peak.
    - y_fit_ig2: array, fitted values for I-G2 peak.
    - y_fit_combined: array, combined fitted values.
    - r_squared: float, goodness of fit measure.
    - ratio_id_ig_intensity: float, intensity ratio of I-D to I-G.
    """
    # Define the Gaussian function
    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

    # Define the combined Gaussian function (all four peaks)
    def quadruple_gaussian(x, amp1, mean1, stddev1, amp2, mean2, stddev2,
                           amp3, mean3, stddev3, amp4, mean4, stddev4):
        return (gaussian(x, amp1, mean1, stddev1) + 
                gaussian(x, amp2, mean2, stddev2) +
                gaussian(x, amp3, mean3, stddev3) + 
                gaussian(x, amp4, mean4, stddev4))

    def fit_id_ig(x_data, y_data):
        # Initial guesses for I-D and I-G
        initial_guess_id_ig = [
            max(y_data) * 0.4, 1400, 50,  # I-D: amplitude, mean, stddev
            max(y_data) * 0.4, 1600, 30   # I-G: amplitude, mean, stddev
        ]
        
        # Define bounds for parameters
        lower_bounds = [
            0, 1350, 20, 
            0, 1550, 20
        ]
        upper_bounds = [
            np.inf, 1450, 60, 
            np.inf, 1650, 60
        ]
        
        # Perform the curve fitting with bounds
        popt_id_ig, _ = curve_fit(
            lambda x, amp1, mean1, stddev1, amp2, mean2, stddev2: 
            gaussian(x, amp1, mean1, stddev1) + gaussian(x, amp2, mean2, stddev2),
            x_data, y_data, p0=initial_guess_id_ig, bounds=(lower_bounds, upper_bounds), maxfev=5000
        )

        # Extract the best-fit parameters for I-D and I-G
        amp1, mean1, stddev1, amp2, mean2, stddev2 = popt_id_ig
       
        # Calculate the areas
        area_id = amp1 * stddev1 * np.sqrt(2 * np.pi)
        area_ig = amp2 * stddev2 * np.sqrt(2 * np.pi)
        
        return amp1, mean1, stddev1, amp2, mean2, stddev2, area_id, area_ig

    def fit_id2_ig2(x_data, y_data, amp1, mean1, stddev1, amp2, mean2, stddev2):
        # Initial guesses for I-D2 and I-G2
        initial_guess_id2_ig2 = [
            max(y_data) * 0.1, 1320, 50,  # I-D2: amplitude, mean, stddev
            max(y_data) * 0.1, 1500, 50   # I-G2: amplitude, mean, stddev
        ]

        # Define bounds for parameters
        lower_bounds = [
            amp1 * 0.5, mean1 - 10, 20,  # Lower bounds for I-D
            amp2 * 0.5, mean2 - 30, 20,  # Lower bounds for I-G
            0, 1250, 30,            # Lower bounds for I-D2
            0, 1500, 30             # Lower bounds for I-G2
        ]
        upper_bounds = [
            amp1 * 1.4, mean1 + 10, 80,     # Upper bounds for I-D
            amp2 * 1.4, mean2 + 30, 80,     # Upper bounds for I-G
            max(y_data) * 0.25, 1380, 100,  # Upper bounds for I-D2
            max(y_data) * 0.25, 1600, 100   # Upper bounds for I-G2
        ]

        # Perform the curve fitting with bounds
        popt_all, _ = curve_fit(
            lambda x, amp1, mean1, stddev1, amp2, mean2, stddev2, amp3, mean3, stddev3, amp4, mean4, stddev4: 
            gaussian(x, amp1, mean1, stddev1) + gaussian(x, amp2, mean2, stddev2) +
            gaussian(x, amp3, mean3, stddev3) + gaussian(x, amp4, mean4, stddev4),
            x_data, y_data,
            p0=[amp1, mean1, stddev1, amp2, mean2, stddev2, *initial_guess_id2_ig2],
            bounds=(lower_bounds, upper_bounds), maxfev=20000
        )

        # Extract the best-fit parameters for all peaks
        amp1, mean1, stddev1, amp2, mean2, stddev2, amp3, mean3, stddev3, amp4, mean4, stddev4 = popt_all

        return amp1, mean1, stddev1, amp2, mean2, stddev2, amp3, mean3, stddev3, amp4, mean4, stddev4

    def calculate_r_squared(y_data, y_fit_combined, x_data):
        
        # Calculate the residual sum of squares
        ss_res = np.sum((y_data - y_fit_combined) ** 2)
        
        # Calculate the total sum of squares
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        
        # Calculate the R-squared value
        r_squared = 1 - (ss_res / ss_tot)
        
        return r_squared

    # Convert input arrays to pandas DataFrame
    df = pd.DataFrame({'Wave': waves, 'Intensity': intensities})
       
    # Initialize fitting arrays with zeros
    y_fit_id = np.zeros_like(intensities)
    y_fit_ig = np.zeros_like(intensities)
    y_fit_id2 = np.zeros_like(intensities)
    y_fit_ig2 = np.zeros_like(intensities)
    y_fit_combined = np.zeros_like(intensities)

    # Perform the initial fitting for I-D and I-G
    amp1, mean1, stddev1, amp2, mean2, stddev2, area_id, area_ig = fit_id_ig(waves, intensities)

    # Perform the 4-peak fitting
    popt_all = fit_id2_ig2(waves, intensities, amp1, mean1, stddev1, amp2, mean2, stddev2)

    # Extract the fitted parameters
    amp1, mean1, stddev1, amp2, mean2, stddev2, amp3, mean3, stddev3, amp4, mean4, stddev4 = popt_all

    # Recalculate the fits
    y_fit_id = gaussian(df['Wave'].values, amp1, mean1, stddev1)
    y_fit_ig = gaussian(df['Wave'].values, amp2, mean2, stddev2)
    y_fit_id2 = gaussian(df['Wave'].values, amp3, mean3, stddev3)
    y_fit_ig2 = gaussian(df['Wave'].values, amp4, mean4, stddev4)
    y_fit_combined = y_fit_id + y_fit_ig + y_fit_id2 + y_fit_ig2

    # Calculate goodness of fit metrics
    r_squared = calculate_r_squared(intensities, y_fit_combined, waves)


    # Calculate the ID-IG ratio using the fitted peak intensity
    ratio_id_ig_intensity = amp1 / amp2 if amp2 != 0 else 0

    return y_fit_id, y_fit_ig, y_fit_id2, y_fit_ig2, y_fit_combined, r_squared, ratio_id_ig_intensity
