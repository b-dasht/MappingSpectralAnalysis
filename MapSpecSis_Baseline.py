from itertools import product
from pybaselines import polynomial
import numpy as np

def baseline_correction(baseline_data, y_col=0, x_col=3, pol_order=2, numstd=0.7, baseline_fig=False):
    """
    Loops through the baseline_data and applies baseline correction to the designated y-axis baseline_data column
    of each spectra. Optionally plots the original and baseline corrected baseline_data.

    Parameters
    ----------
    baseline_data: baseline_dataFrame
        The baseline_dataframe containing the spectra at different X and Y positions that is to be
        baseline corrected
    y_col : int, optional
        The column number for the y-axis baseline_data (e.g., Intensity) to be baseline corrected.
        Default is set to 0, which will use the values of the last column.
    x_col : int, optional
        The column number for the x-axis baseline_data (e.g., Raman shift). The default is 3.
    pol_order : int
        The polynomial order to use for the fitting of the baseline.
    numstd : float, optional
        The number of standard deviations to include when thresholding for the baseline. Has the
        effect of shifting the baseline lower (smaller numstd) or higher (larger numstd).
        Default is 0.7.
    baseline_fig : bool, optional
        If True, plots the input baseline_data, the generated baseline and the baseline-correctedbaseline_data.
        Default is False.

    Returns
    -------
    baseline_corrected : baseline_dataframe
        The original baseline_dataframe with 2 additional columns with the header names 'Baseline' and
        'Baseline Corrected' that contains the fitted baseline baseline_data and the baseline corrected
        baseline_data.
    """

    # Column name identification
    pos_col_1 = baseline_data.columns[0]
    pos_col_2 = baseline_data.columns[1]
    x_column = baseline_data.columns[x_col - 1]
    y_column = baseline_data.columns[y_col - 1]

    baselines = []
    y_corrected = []

    # Determines the number of X and Y positions in the baseline_dataset
    range_xpos = np.arange(int(baseline_data[pos_col_1].min()), int(baseline_data[pos_col_1].max()) + 1)
    range_ypos = np.arange(int(baseline_data[pos_col_2].min()), int(baseline_data[pos_col_2].max()) + 1)

    # Loops for each X and Y position in the map
    for x_pos, y_pos in product(range_xpos, range_ypos):

        # Filters baseline_data to the X and Y position combination of the loop
        filtered_baseline_data = baseline_data[(baseline_data[pos_col_1] == x_pos) & (baseline_data[pos_col_2] == y_pos)]

        # Extracts the x (e.g., Raman shift) and y (e.g., Intensity) data from filtered_baseline_data
        x, y = filtered_baseline_data[x_column].values.astype(float), filtered_baseline_data[y_column].values.astype(float)

        # Fits a baseline according to the set criteria
        baseline = polynomial.imodpoly(y, x_data=x, poly_order = pol_order, num_std = numstd)
        baseline = np.array(baseline[0])

        # Collects the fitted baselines for later saving
        baselines.extend(baseline)

        # Removes the fitted baseline from the y-axis baseline_data
        baseline_corrected = np.array(y) - baseline

        # Collects the baseline corrected y-axis baseline_data for later saving
        y_corrected.extend(baseline_corrected)


    baseline_data['Baseline'] = baselines
    baseline_data['Baseline Corrected'] = y_corrected

    if baseline_fig:

        import matplotlib.pyplot as plt

        # Plotting the original, baseline and baseline corrected baseline_data
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))

        # Original baseline_data plot
        ax[0].plot(baseline_data[x_column], baseline_data[y_column], label='Original baseline_data', color='blue')
        ax[0].plot(baseline_data[x_column], baseline_data['Baseline'], label='Baseline', color='yellow', linestyle='--')
        ax[0].set_title('Original baseline_data and Fitted Baseline')
        ax[0].set_xlabel(x_column)
        ax[0].set_ylabel(y_column)
        ax[0].legend()

        # Baseline corrected baseline_data plot
        ax[1].plot(baseline_data[x_column], baseline_data['Baseline Corrected'], label='Baseline Corrected baseline_data', color='orange')
        ax[1].set_title('Baseline Corrected baseline_data')
        ax[1].set_xlabel(x_column)
        ax[1].set_ylabel(y_column)

        plt.tight_layout()
        plt.show()

    return baseline_data

# Example usage
# from MapSpecSis_ProcessFile import process_file
# from MapSpecSis_SpikeRemoval import spike_removal
# from MapSpecSis_PCA import pca_noise_removal
# from MapSpecSis_Smooth import smooth_data
# file = ".txt" # Enter file name here
# processed_data = process_file(file)
# processed_data = spike_removal(processed_data)
# processed_data = pca_noise_removal(processed_data, pca_comp=6)
# processed_data = smooth_data(processed_data)
# baseline_corrected_data = baseline_correction(processed_data, y_col=7, x_col=3, pol_order=2, numstd=0.7, baseline_fig=True)