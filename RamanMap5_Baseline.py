from itertools import product
from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np

def baseline_correction(baseline_data, y_col=0, x_col=3, lam=3e7, p=0.02, num_iter=20, baseline_fig=False):
    """
    Loops through the baseline_data and applies ALS baseline correction to the designated y-axis
    column of each spectrum. Optionally plots the original and baseline-corrected data.

    Parameters
    ----------
    baseline_data: DataFrame
        The DataFrame containing the spectra at different X and Y positions to be
        baseline corrected.
    y_col : int, optional
        The column number for the y-axis data (e.g., Intensity) to be baseline corrected.
        Default is set to 0, which will use the values of the last column.
    x_col : int, optional
        The column number for the x-axis data (e.g., Raman shift). Default is 3.
    lam : float, optional
        The smoothing parameter for the ALS baseline algorithm. Larger values result in smoother baselines.
        Default is 3e7.
    p : float, optional
        The asymmetry parameter for the ALS baseline algorithm. Must be between 0 and 1. Lower values
        fit peaks better. Default is 0.02.
    num_iter : int, optional
        Number of baseline correction iterations to perform. Default is 20.
    baseline_fig : bool, optional
        If True, plots the input data, the generated baseline, and the baseline-corrected data.
        Default is False.

    Returns
    -------
    baseline_corrected : DataFrame
        The original DataFrame with 2 additional columns named 'Baseline' and 'Baseline Corrected'
        that contain the fitted baseline data and the baseline-corrected data.
    """

    # Column name identification
    pos_col_1 = baseline_data.columns[0]
    pos_col_2 = baseline_data.columns[1]
    x_column = baseline_data.columns[x_col - 1]
    y_column = baseline_data.columns[y_col - 1]

    baselines = []
    y_corrected = []

    # Determine the number of X and Y positions in the dataset
    range_xpos = np.arange(int(baseline_data[pos_col_1].min()), int(baseline_data[pos_col_1].max()) + 1)
    range_ypos = np.arange(int(baseline_data[pos_col_2].min()), int(baseline_data[pos_col_2].max()) + 1)

    # Loops for each X and Y position in the map
    for x_pos, y_pos in product(range_xpos, range_ypos):

        # Filters data to the X and Y position combination of the loop
        filtered_data = baseline_data[(baseline_data[pos_col_1] == x_pos) & (baseline_data[pos_col_2] == y_pos)]

        # Extracts the y-axis data (e.g., Intensity)
        y = filtered_data[y_column].values.astype(float)

        # Length of the data
        L = len(y)

        # Construct the second derivative matrix
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(D.transpose())  # Second derivative matrix scaled by lambda

        # Initial weights (equal weighting)
        w = np.ones(L)

        # ALS baseline correction iterations
        for _ in range(num_iter):
            W = sparse.diags(w, 0)  # Diagonal weight matrix
            Z = W + D  # Combined weight and smoothness matrices
            baseline = spsolve(Z, w * y)  # Solve the linear system
            w = p * (y > baseline) + (1 - p) * (y <= baseline)  # Update weights

        # Collect the fitted baselines for later saving
        baselines.extend(baseline)

        # Removes the fitted baseline from the y-axis data
        baseline_corrected = y - baseline

        # Collects the baseline-corrected y-axis data for later saving
        y_corrected.extend(baseline_corrected)

    baseline_data['Baseline'] = baselines
    baseline_data['Baseline Corrected'] = y_corrected

    if baseline_fig:

        import matplotlib.pyplot as plt

        # Plotting the original, baseline, and baseline-corrected data
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))

        # Original data plot
        ax[0].plot(baseline_data[x_column], baseline_data[y_column], label='Original data', color='blue')
        ax[0].plot(baseline_data[x_column], baseline_data['Baseline'], label='Baseline', color='yellow', linestyle='--')
        ax[0].set_title('Original Data and Fitted Baseline')
        ax[0].set_xlabel(x_column)
        ax[0].set_ylabel(y_column)
        ax[0].legend()

        # Baseline-corrected data plot
        ax[1].plot(baseline_data[x_column], baseline_data['Baseline Corrected'], label='Baseline Corrected Data', color='orange')
        ax[1].set_title('Baseline Corrected Data')
        ax[1].set_xlabel(x_column)
        ax[1].set_ylabel(y_column)

        plt.tight_layout()
        plt.show()

    return baseline_data

# Example usage
# from RamanMap1_ProcessFile import process_file
# from RamanMap2_SpikeRemoval import spike_removal
# from RamanMap3_PCA import pca_noise_removal
# from RamanMap4_Smooth import smooth_data
# file = "C:/Users/Benny/OneDrive - University of Birmingham/Machine Guides & Info/Raman/Raman - ID-IG Map/ASPN-raw.txt"
# file = "C:/Users/Administrator/Dropbox/Personal/PythonScripts/ASPN-raw.txt"
# processed_data = process_file(file)
# processed_data = spike_removal(processed_data)
# processed_data = pca_noise_removal(processed_data, pca_comp=6)
# processed_data = smooth_data(processed_data)
# baseline_corrected_data = baseline_correction(processed_data, y_col=0, x_col=3, lam=3e7, p=0.02, num_iter=20, baseline_fig=True)
