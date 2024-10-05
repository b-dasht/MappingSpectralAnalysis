from scipy.signal import savgol_filter

def smooth_data(smooth_data, y_col=0, x_col= 3, window_len=11, pol_order=1, smooth_fig=False):
    """
    Smooths the final column containing the spectra data based on the given window length
    and polynomial order. Can optionally plot the original and smoothed data.

    Parameters
    ----------
    smooth_data : DataFrame
        The dataframe containing the spectra at different X and Y positions that is to be smoothed.
    y_col : int, optional
        The column number for the y-axis data (e.g., Intensity). Default is set to 0, which will
        use the values of the last column.
    x_col : int, optional
        The column number for the x-axis data (e.g., Raman shift). The default is 3.
    window_len: int
        The window length to be used during the smoothing process. Default is 11.
    pol_order : int
        The polynomial order of the smoothing process.
    smooth_fig : bool, optional
        If True, plots the original data and the smoothed data. Default is False.

    Returns
    -------
    smooth_data : DataFrame
        The original dataframe with an additional column with the header name 'Smoothed' that
        contains the smoothed data.

    """

    # Column name identification
    x_column = smooth_data.columns[x_col - 1]
    y_column = smooth_data.columns[y_col - 1]

    # Smooth the data
    smoothed_data = savgol_filter(smooth_data[y_column], window_length=window_len, polyorder=pol_order)

    # Add the smoothed data to the original dataframe
    smooth_data["Smoothed"] = smoothed_data

    # Plotting if required
    if smooth_fig:

        import matplotlib.pyplot as plt

        # Plotting the original vs smoothed data
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))

        # Original data plot
        ax[0].plot(smooth_data[x_column], smooth_data[y_column], label='Original Data')
        ax[0].set_title('Original Data')
        ax[0].set_xlabel(x_column)
        ax[0].set_ylabel(y_column)

        # Smoothed data plot
        ax[1].plot(smooth_data[x_column], smooth_data['Smoothed'], label='Smoothed Data', color='orange')
        ax[1].set_title('Smoothed Data')
        ax[1].set_xlabel(x_column)
        ax[1].set_ylabel(y_column)

        plt.tight_layout()
        plt.show()

    return smooth_data

# Example usage
# from MapSpecSis_ProcessFile import process_file
# from MapSpecSis_SpikeRemoval import spike_removal
# from MapSpecSis_PCA import pca_noise_removal
# file = ".txt" # Enter file name here
# processed_data = process_file(file)
# processed_data = spike_removal(processed_data)
# processed_data = pca_noise_removal(processed_data, pca_comp=6)
# smoothed_data = smooth_data(processed_data, y_col=6, x_col=3, window_len=11, pol_order=1, smooth_fig=True)
