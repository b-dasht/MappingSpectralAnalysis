from itertools import product
import numpy as np

def normalise_data(normalising_data, max_val, y_col=0, each_spectra=True, x_col=3, normal_fig=False):
    """
    Normalises the y-axis values from the data. Can be performed in one go (only one y-axis
    datapoint at the set maximum value) or individually for each spectra (each spectra has one
    datapoint at the set maximum valuepoint). Optionally plots the original and normalised data.

    Parameters
    ----------
    normalising_data : DataFrame
        The dataframe containing the spectra at different X and Y positions that is to be normalised.
    max_val : int
        The new maximum limit for the normalised data (e.g., 100 or 1).
    y_col : int, optional
        The column number for the y-axis data (e.g., Intensity). Default is set to 0, which will
        use the values of the last column.
    each_spectra : bool, optional
        If True, normalises the y-axis data in each spectra individually. Otherwise normalises all
        y-axis data as a group. Default is True.
    x_col : int, optional
        The column number for the x-axis data (e.g., Raman shift). The default is 3.
    normal_fig : bool, optional
        If True, plots the original data and the normalised data. Default is False.

    Returns
    -------
    normalised_df : DataFrame
        The original dataframe with an additional column with the header 'Normalised'
        that contains the normalised y-axis data
    """

    # Column name identification
    x_column = normalising_data.columns[x_col - 1]
    y_column = normalising_data.columns[y_col - 1]

    # If true, runs for each spectra. Otherwise, runs them as one big group
    if each_spectra:

        # Column name identification
        pos_col_1 = normalising_data.columns[0]
        pos_col_2 = normalising_data.columns[1]

        # Initiates an array for storing the normalised data from each position
        y_normalised = []

        # Determines the number of X and Y positions in the normalising_dataset
        range_xpos = np.arange(int(normalising_data[pos_col_1].min()), int(normalising_data[pos_col_1].max()) + 1)
        range_ypos = np.arange(int(normalising_data[pos_col_2].min()), int(normalising_data[pos_col_2].max()) + 1)

        # Loops for each X and Y position in the map
        for x_pos, y_pos in product(range_xpos, range_ypos):


            # Filters normalising_data to the X and Y position combination of the loop
            filtered_normalising_data = normalising_data[(normalising_data[pos_col_1] == x_pos) & (normalising_data[pos_col_2] == y_pos)]

            # Extracts the y-axis values (e.g., Intensity) from filtered_normalising_data
            y = filtered_normalising_data[y_column].values.astype(float)

            # Normalises the values to the set max value
            y_min_zero = y - min(y)
            normalised = max_val * (y_min_zero/max(y_min_zero))

            # Collects the fitted baselines for later saving
            y_normalised.extend(normalised)
    else:

        y_min_zero = normalising_data[y_column] - min(normalising_data[y_column])
        y_normalised = max_val * (y_min_zero/max(y_min_zero))

    # Creates new column and stores the normalised data
    normalising_data['Normalised'] = y_normalised

    # If true, plots the figure
    if normal_fig:

        import matplotlib.pyplot as plt

        # Plotting the original and normalised data
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))

        # Plot of original data
        ax[0].plot(normalising_data[x_column], normalising_data[y_column], label='Original data', color='blue')
        ax[0].set_title('Original Data')
        ax[0].set_xlabel(x_column)
        ax[0].set_ylabel(y_column)
        ax[0].legend()

        # Plot of normalised data
        ax[1].plot(normalising_data[x_column], normalising_data['Normalised'], label='Normalised data', color='orange')
        ax[1].set_title('Normalised Data')
        ax[1].set_xlabel(x_column)
        ax[1].set_ylabel(y_column)

        plt.tight_layout()
        plt.show()

    return normalising_data


# Example usage
# from RamanMap1_ProcessFile import process_file
# from RamanMap2_SpikeRemoval import spike_removal
# from RamanMap3_PCA import pca_noise_removal
# from RamanMap4_Smooth import smooth_data
# from RamanMap5_Baseline import baseline_correction
# from RamanMap6_Crop import crop_data
# file = "C:/Users/Benny/OneDrive - University of Birmingham/Machine Guides & Info/Raman/Raman - ID-IG Map/ASPN-raw.txt"
# processed_data = process_file(file)
# processed_data = spike_removal(processed_data)
# processed_data = pca_noise_removal(processed_data, pca_comp=6)
# processed_data = smooth_data(processed_data)
# processed_data = baseline_correction(processed_data)
# processed_data = crop_data(processed_data, 800, 1800)
# normalised_data1 = normalise_data(processed_data,100, y_col=9, each_spectra=True, x_col=3, normal_fig=True)
# normalised_data2 = normalise_data(processed_data,100, y_col=9, each_spectra=False, x_col=3, normal_fig=True)
