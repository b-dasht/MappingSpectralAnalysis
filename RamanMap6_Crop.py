def crop_data(cropping_data, lower_bound, upper_bound, x_col=3, crop_fig=False):
    """
    Crops the spectra data based on the given x-axis bounds and optionally plots the original
    and cropped data.

    Parameters
    ----------
    cropping_data : DataFrame
        The dataframe containing the spectra at different X and Y positions that is to be cropped.
    lower_bound : float
        The lower bound for the cropping.
    upper_bound : float
        The upper bound for the cropping.
    x_col : int, optional
        The column number for the x-axis data (e.g., Raman shift). The default is 3.
    crop_fig : bool, optional
        If True, plots the original data and the cropped data. Default is False.

    Returns
    -------
    cropped_df : DataFrame
        The cropped original dataframe with the same headings.

    """

    # Column name identification
    x_column = cropping_data.columns[x_col - 1]

    # Filter out rows with x-axis data less than or greater than the set lower and upper bounds
    cropped_df = cropping_data[(cropping_data[x_column] >= lower_bound) &
                                 (cropping_data[x_column] <= upper_bound)]

    # Reset the index to create sequential X and Y values
    cropped_df.reset_index(drop=True, inplace=True)

    # Plotting if required
    if crop_fig:

        import matplotlib.pyplot as plt

        # Plotting the original vs cropped data
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))

        # Original data plot
        original_x = cropping_data[x_column]
        original_y = cropping_data.iloc[:, -1]
        ax[0].plot(original_x, original_y, label='Original Data')
        ax[0].set_title('Original Data')
        ax[0].set_xlim([original_x.min(), original_x.max()])
        ax[0].set_xlabel(x_column)
        ax[0].set_ylabel(cropping_data.columns[-1])

        # Cropped data plot
        cropped_x = cropped_df[x_column]
        cropped_y = cropped_df.iloc[:, -1]
        ax[1].plot(cropped_x, cropped_y, label='Cropped Data', color='orange')
        ax[1].set_title('Cropped Data')
        ax[1].set_xlim([original_x.min(), original_x.max()])
        ax[1].set_xlabel(x_column)
        ax[1].set_ylabel(cropping_data.columns[-1])

        plt.tight_layout()
        plt.show()

    return cropped_df

# Example usage
# from RamanMap1_ProcessFile import process_file
# from RamanMap2_SpikeRemoval import spike_removal
# from RamanMap3_PCA import pca_noise_removal
# from RamanMap4_Smooth import smooth_data
# from RamanMap5_Baseline import baseline_correction
# file = "C:/Users/Benny/OneDrive - University of Birmingham/Machine Guides & Info/Raman/Raman - ID-IG Map/ASPN-raw.txt"
# processed_data = process_file(file)
# processed_data = spike_removal(processed_data)
# processed_data = pca_noise_removal(processed_data, pca_comp=6)
# processed_data = smooth_data(processed_data)
# processed_data = baseline_correction(processed_data)
# cropped_data = crop_data(processed_data, 800, 1800, x_col=3, crop_fig=True)
