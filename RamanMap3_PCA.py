import pandas as pd
from sklearn.decomposition import PCA

def pca_noise_removal(denoising_data, pca_comp=None, pca_var=None, y_col=0, x_col=3, pca_fig=False):
    """
    Utilises PCA to reduce noise in the data by recreating each spectrum using only the main principal
    components of the dataset or by retaining a set amount of the total variance in the data. Returns
    the data with a new column for the denoised data and prints the variance removed.

    Parameters
    ----------
    denoising_data : DataFrame
        A DataFrame containing the spectra at different X and Y positions that is to be PCA
        denoised. X and Y columns should be the first two columns of the DataFrame.
    pca_comp : int, optional
        The number of principal components to retain during the reconstruction. If specified,
        `pca_var` should not be set.
    pca_var : float, optional
        The percentage of the total variance in the data to retain during the PCA reconstruction.
        The value should be between 0 and 1. If specified, `pca_comp` should not be set.
    y_col : int, optional
        The column containing the y-axis data (e.g., Intensity values). Default is set to 0,
        which will use the values of the last column.
    x_col : int, optional
        The column containing the x-axis data (e.g., Raman shift values). Default is 3 as
        first two columns are for X and Y positions.

    pca_fig : bool, optional
        If True, plots the original data and the PCA-denoised data. Default is False.

    Returns
    -------
    denoising_df : DataFrame
        The original DataFrame with an additional column with the header name 'Denoised' that
        contains the PCA-denoised data.
    """
    # Check to ensure that exactly one of pca_comp or pca_var is specified
    if pca_comp is not None and pca_var is not None:
        raise ValueError("Please specify one of 'pca_comp' or 'pca_var'. Both cannot be selected at the same time.")

    if pca_comp is None and pca_var is None:
        raise ValueError("Please specify either 'pca_comp' or 'pca_var'. One of them must be selected.")

    # Column name identification
    pos_col_1 = denoising_data.columns[0]
    pos_col_2 = denoising_data.columns[1]
    x_data_col = denoising_data.columns[x_col - 1]
    y_data_col = denoising_data.columns[y_col - 1]

    # Create pivot table to structure data for PCA
    pivot_table = denoising_data.pivot_table(
        index=[pos_col_1, pos_col_2],
        columns=x_data_col,
        values=y_data_col,
        fill_value=0
    )

    # Extract the x (features) and y (samples) for PCA
    x = pivot_table.columns.values
    y = pivot_table.values

    # Perform PCA to reduce dimensionality
    pca = PCA(n_components=pca_comp if pca_comp is not None else pca_var)
    transformed_y = pca.fit_transform(y)
    denoised_y = pca.inverse_transform(transformed_y)

    # Print the total variance retained for the component method
    if pca_comp is not None:
        retained_var = 100 * sum(pca.explained_variance_ratio_)
        print(f"Total variance retained: {retained_var:.2f}%")


    # Reconstruct DataFrame with PCA-denoised intensity values
    denoised_data = pd.DataFrame(denoised_y, columns=x, index=pivot_table.index)
    denoised_data_stacked = denoised_data.stack().reset_index()
    denoised_data_stacked.columns = [pos_col_1, pos_col_2, x_data_col, 'Denoised']

    # Merge the PCA-denoised data back into the original dataframe
    denoised_df = pd.merge(
        denoising_data,
        denoised_data_stacked,
        on=[pos_col_1, pos_col_2, x_data_col],
        how='left'
    )

    # Sort the DataFrame by X and Y
    denoised_df = denoised_df.sort_values(by=[pos_col_1, pos_col_2])

    # Reset the index to create sequential X and Y values
    denoised_df.reset_index(drop=True, inplace=True)

    # Plotting if required
    if pca_fig:

        import matplotlib.pyplot as plt

        # Plotting the original vs PCA-denoised data
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))

        # Original data plot
        ax[0].plot(x, y.T) #
        ax[0].set_title('Original Data')
        ax[0].set_xlabel(x_data_col)
        ax[0].set_ylabel(y_data_col)

        # PCA denoised data plot
        ax[1].plot(x, denoised_y.T)
        ax[1].set_title('PCA Denoised Data')
        ax[1].set_xlabel(x_data_col)
        ax[1].set_ylabel(y_data_col)

        plt.tight_layout()
        plt.show()

    return denoised_df

# Example usage with pca_comp and pca_var
# from RamanMap1_ProcessFile import process_file
# from RamanMap2_SpikeRemoval import spike_removal
# file = "C:/Users/Benny/OneDrive - University of Birmingham/Machine Guides & Info/Raman/Raman - ID-IG Map/ASPN-raw.txt"
# processed_data = process_file(file)
# processed_data = spike_removal(processed_data)
# denoised_data_comp = pca_noise_removal(processed_data, y_col=5, x_col=3, pca_comp=6, pca_fig=True)
# denoised_data_var = pca_noise_removal(processed_data, y_col=5, x_col=3, pca_var=0.4, pca_fig=True)
