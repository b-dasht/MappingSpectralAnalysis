import pandas as pd
from sklearn.decomposition import PCA

def process_raman_file(file_path, lower_bound, upper_bound, pca_comp):
    """
    Processes the Raman spectroscopy data file by performing PCA to reduce dimensionality
    and reconstructing a DataFrame with X, Y, Wave, Raw Intensity, and Denoised Intensity columns.

    Parameters:
    - file_path: the path to the Raman spectroscopy data file.
    - pca_comp: the number of principal components to use in PCA.

    Returns:
    - processed_data: DataFrame containing X, Y, Wave, Raw Intensity, and Denoised Intensity columns.
    """

    # Read the file into a pandas DataFrame
    raw_data = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None)
    raw_data.columns = ['X', 'Y', 'Wave', 'Intensity']

    # Adjust X and Y values so that the smallest value is 0
    min_x = raw_data['X'].min()
    min_y = raw_data['Y'].min()
    raw_data['X'] = raw_data['X'] - min_x
    raw_data['Y'] = raw_data['Y'] - min_y

    # Pivot the DataFrame to get the intensity matrix
    pivot_table = raw_data.pivot_table(index=['X', 'Y'], columns='Wave', values='Intensity', fill_value=0)
    wave_values = pivot_table.columns.values
    intensity_matrix = pivot_table.values

    # Perform PCA to reduce dimensionality
    pca = PCA(n_components=pca_comp)
    transformed_intensity_matrix = pca.fit_transform(intensity_matrix)
    denoised_intensity_matrix = pca.inverse_transform(transformed_intensity_matrix)

    # Reconstruct the DataFrame with PCA-denoised intensity values
    denoised_data = pd.DataFrame(denoised_intensity_matrix, columns=wave_values, index=pivot_table.index)
    denoised_data_stacked = denoised_data.stack().reset_index()
    denoised_data_stacked.columns = ['X', 'Y', 'Wave', 'Denoised Intensity']

    # Add the original raw intensity values
    raw_data_pivot = raw_data.pivot_table(index=['X', 'Y'], columns='Wave', values='Intensity', fill_value=0)
    raw_data_stacked = raw_data_pivot.stack().reset_index()
    raw_data_stacked.columns = ['X', 'Y', 'Wave', 'Raw Intensity']

    # Merge the PCA-denoised data with the original raw data
    processed_data = pd.merge(raw_data_stacked, denoised_data_stacked, on=['X', 'Y', 'Wave'])

    # Sort the DataFrame by X and Y
    processed_data = processed_data.sort_values(by=['X', 'Y'])
    
    # Filter out rows with wave numbers less than or greater than the set lower and upper bounds
    processed_data = processed_data[(processed_data['Wave'] >= lower_bound) & (processed_data['Wave'] <= upper_bound)]

    # Reset the index to create sequential X and Y values
    processed_data.reset_index(drop=True, inplace=True)

    return processed_data

