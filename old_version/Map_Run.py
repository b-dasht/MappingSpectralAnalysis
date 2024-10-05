import os
import numpy as np
import pandas as pd
from itertools import product
from scipy.signal import savgol_filter
from Process_RamanMap import process_raman_file
from Extract_Spectra import extract_spectra
from Single_Scan_Model import fit_and_save_spectra
from Baseline_Fix import baseline_correction
from Spectra_Plots import plot_spectra
from Spectra_Plots import plot_heatmap

# Define the input files
input_files = ['1.txt', '2.txt', '3.txt'] # Enter the name of the files in the working directory to be processed and analysed. An example is provided.

for file in input_files:
    
    input_file = file

    # Define the output folder (creates a new folder if it does not exist)
    output_dir = f"Fittings_{input_file[:-4]}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the Raman file to prepare the data
    overall_data = process_raman_file(input_file, 800, 1900, 5)

    # Initialize new columns with NaN values
    new_columns = ['Smoothed Intensity', 'Baseline', 'Baseline Corrected', 'D Band Fit', 
                   'G Band Fit', 'D2 Band Fit', 'G2 Band Fit', 'Combined Fit']
    for col in new_columns:
        overall_data[col] = np.nan

    # Find the maximum X and Y values to loop through each position
    max_x = int(overall_data['X'].max())
    max_y = int(overall_data['Y'].max())

    # Initialize results list to store R-Squared and ID/IG ratio information
    results = []

    # Define the dictionary for storing results inside the loop
    fit_results = {
        'Smoothed Intensity': None,
        'Baseline': None,
        'Baseline Corrected': None,
        'D Band Fit': None,
        'G Band Fit': None,
        'D2 Band Fit': None,
        'G2 Band Fit': None,
        'Combined Fit': None
    }

    # Loop through all X and Y combinations
    for x_val, y_val in product(range(max_x + 1), range(max_y + 1)):
        
        # Extract the wavenumbers, raw spectra and denoised spectra for the current X and Y position
        waves, raw_intensities, denoised_intensities = extract_spectra(overall_data, x_val, y_val)

        # Perform data smoothing
        smoothed_intensities = savgol_filter(denoised_intensities, window_length=11, polyorder=1)

        # Perform baseline correction
        corrected_intensities, baseline = baseline_correction(waves, smoothed_intensities, polyorder=1, numstd=0.1)

        # Call the fitting function and get results
        id_fit, ig_fit, id2_fit, ig2_fit, combined_fit, r_squared, ratio_id_ig_intensity = fit_and_save_spectra(
            waves, corrected_intensities, x_val, y_val)

        # Update the fit_results dictionary with the computed results
        fit_results['Smoothed Intensity'] = smoothed_intensities
        fit_results['Baseline'] = baseline
        fit_results['Baseline Corrected'] = corrected_intensities
        fit_results['D Band Fit'] = id_fit
        fit_results['G Band Fit'] = ig_fit
        fit_results['D2 Band Fit'] = id2_fit
        fit_results['G2 Band Fit'] = ig2_fit
        fit_results['Combined Fit'] = combined_fit

        # Update the DataFrame with results
        for col, result_array in fit_results.items():
            overall_data.loc[(overall_data['X'] == x_val) & (overall_data['Y'] == y_val), col] = np.interp(
                overall_data.loc[(overall_data['X'] == x_val) & (overall_data['Y'] == y_val), 'Wave'], 
                waves, result_array
            )

        # Store results
        results.append([x_val, y_val, r_squared, ratio_id_ig_intensity])

        # Save data and fitting results to text files
        output_file = os.path.join(output_dir, f"FittingResults-{x_val},{y_val}.txt")
        with open(output_file, 'w') as f:
            f.write("Wavenumber\tRaw Intensity\tDenoised Intensity\tSmoothed Intensity\tBaseline\tBaseline Corrected\tD Band Fit\tG Band Fit\tD2 Band Fit\tG2 Band Fit\tCombined Fit\n")
            for wave, raw_intensity, denoised_intensity, smoothed_intensity, baseline_value, corrected_intensity, id_fit_value, ig_fit_value, id2_fit_value, ig2_fit_value, combined_fit_value in zip(
                    waves, raw_intensities, denoised_intensities, smoothed_intensities, baseline, corrected_intensities, id_fit, ig_fit, id2_fit, ig2_fit, combined_fit):
                f.write(f"{wave:.2f}\t{raw_intensity:.2f}\t{denoised_intensity:.2f}\t{smoothed_intensity:.2f}\t{baseline_value:.2f}\t{corrected_intensity:.2f}\t{id_fit_value:.2f}\t{ig_fit_value:.2f}\t{id2_fit_value:.2f}\t{ig2_fit_value:.2f}\t{combined_fit_value:.2f}\n")

        # Plot spectra
        plot_spectra(input_file, output_dir, x_val, y_val, waves, raw_intensities, denoised_intensities, smoothed_intensities, baseline, corrected_intensities, id_fit, ig_fit, id2_fit, ig2_fit, combined_fit)
        
        # Print progress
        print(f"Completed fitting and saving for X={x_val}, Y={y_val}")

    # Create DataFrame from results
    results_data = pd.DataFrame(results, columns=['X', 'Y', 'R-Squared', 'ID/IG Intensity'])
    
    # Plot heatmap of the ID-IG Ratios
    intensity = results_data.pivot(index='X', columns='Y', values='ID/IG Intensity')
    plot_heatmap('Intensity', intensity, input_file[:-4], output_dir, max_x, max_y, 0.5, 1.2)
    
    # Define the output file name for results
    output_file = os.path.join(output_dir, f"{input_file[:-4]}_fittings.txt")

    # Save the Results table
    results_data.to_csv(output_file, index=False, sep='\t')

    # Save overall data
    overall_data_output_file = os.path.join(output_dir, f"{input_file[:-4]}-data.txt")
    overall_data.to_csv(overall_data_output_file, index=False, sep='\t')
