import os
import pandas as pd

def process_file(file_path, headers=True, col_names=[], zero_xy=False):
    """
    Processes and creates a DataFrame from a raw Raman mapping data file. Also saves the processed
    Dataframe as a tab-separated file with the name: file_path + "_processed". Column names can be
    automatically assigned based on the first row or manually specified (must not have column names
    in the first row for this choice). Additionally, the minimum X and Y position values can also be
    zeroed using their minimum values.

    Parameters
    ----------
    file_path : str
        The full path to the raw data file to be processed.
    headers : bool, optional
        Indicates whether the first row of the file contains column headers. If True, the first
        row is used as the header. If False, custom column names must be provided via `col_names`.
        Default is True.
    col_names : list of str, optional
        A list of column names to use if `headers` is False. This list should match the number
        of columns in the file. Default is an empty list.
    zero_xy : bool, optional
        If True, the function will modify the values in the first two columns to start from zero and
        increase in positive direction. This is useful for normalising position data. Default is False.

    Returns
    -------
    processed_data : pandas.DataFrame
        A DataFrame containing the data from the input file.
    """

    # Finds the path of the current directory
    current_dir = os.path.dirname(file_path)

    # Identifies the filename (including the filetype extension)
    filename_withext = os.path.basename(file_path)

    # Separate the filename and the extension
    filename, _ = os.path.splitext(filename_withext)

    # Creates a new folder in the same directory called Processed + the file name
    output_dir = f"{current_dir}/Processed_{filename}"
    os.makedirs(output_dir, exist_ok=True)

    # Reads the data, separates columns by whitespace/tabs, sets the first row as the header
    if headers:

        # Reads the data from the file and sets the headers to the first row
        processed_data = pd.read_csv(file_path, sep=r'\s+', header=0)

        # Removes any # in the column names
        processed_data.columns = processed_data.columns.str.replace('#', '', regex=False)

        # Zeros the minimum x and y position values. Only performed if zero_xy = True
        if zero_xy:

            # Identify the first two column names
            col_1 = processed_data.columns[0]
            col_2 = processed_data.columns[1]

            processed_data[col_1] = processed_data[col_1] - processed_data[col_1].min()
            processed_data[col_2] = processed_data[col_2] - processed_data[col_2].min()
    else:

        # Reads the data from the file and sets the column names based on the set col_names list
        processed_data = pd.read_csv(file_path, sep=r'\s+', names=col_names)

        # Check to ensure that the number of columns provided match the data
        if len(col_names) != processed_data.shape[1]:

            raise ValueError(f"Number of columns in col_names ({len(col_names)}) does not match the number of columns in the file ({processed_data.shape[1]}).")

        # Zeros the minimum x and y position values. Only performed if zero_xy = True
        if zero_xy:
            processed_data[col_names[0]] = processed_data[col_names[0]] - processed_data[col_names[0]].min()
            processed_data[col_names[1]] = processed_data[col_names[1]] - processed_data[col_names[1]].min()

    # Save the dataframe to a tab-separated text file
    processed_data.to_csv(f'{output_dir}/{filename}_processed.txt', index=False,sep='\t')

    return processed_data

# Example run
# file1 = "C:/Users/Benny/OneDrive - University of Birmingham/Machine Guides & Info/Raman/Raman - ID-IG Map/ASPN-raw.txt"
# processed_data_1 = process_file(file1, headers=True, zero_xy=True)
# file2 = "C:/Users/Benny/OneDrive - University of Birmingham/Machine Guides & Info/Raman/Raman - ID-IG Map/ASPN-raw.txt"
# processed_data_2 = process_file(file2, headers=False, col_names=['X', 'Y', 'Wave', 'Intensity'], zero_xy=False)
