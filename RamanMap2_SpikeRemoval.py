import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.signal import find_peaks, peak_widths, savgol_filter
from pybaselines import polynomial
from scipy import interpolate


def spike_removal(spike_remove_data, y_col=0, x_col=3, pol_order=2, expand_window=20,
                  prom_width_ratio=200, overall_plot=False, individual_plots=False):
    """
    Detects and removes spikes in each spectra at different X and Y position combinations
    the spectra data using a set peak width to prominence ratio. Due to detection and removal
    issues, the edges (final 2 data points either side) of the spectra are smoothed to remove
    potential spikes. The X and Y position columns must be the first two columns in the dataframe.

    Parameters
    ----------
    spike_remove_data : dataframe
        A dataframe containing the spectral data with headings corresponding to x_name and y_name
        that is to have the spikes/cosmic rays removed.
    y_col : int, optional
        The column containing the y-axis data (e.g., Intensity values). Default is set to 0, which
        will use the values of the last column.
    x_col : int, optional
        The column containing the x-axis data (e.g., Raman shift values). Default is 3 as
        first two columns are for X and Y positions.
    pol_order : int, optional
        The polynomial order to use for the baseline fitting. Default is 2.
    expand_window : int, optional
        The expanded window size (each side of spikes) for evaluating and removing spikes.
        Default is 20.
    prom_width_ratio : float, optional
        The ratio of prominence to width for spike identification. Default is 200.
    overall_plot : bool, optional
        If True, plots the original and corrected data. Default is False.
    individual_plots : bool, optional
        If True, creates magnified plots around each detected spike. Default is False.

    Returns
    -------
    spike_remove_data : dataframe
        The original dataframe with an additional column 'Spike Corrected' containing the
        corrected intensity data.
    """

    # Column name identification
    pos_col_1 = spike_remove_data.columns[0]
    pos_col_2 = spike_remove_data.columns[1]
    x_column = spike_remove_data.columns[x_col - 1]
    y_column = spike_remove_data.columns[y_col - 1]

    # Determines the number of X and Y positions in the dataset
    range_xpos = np.arange(int(spike_remove_data[pos_col_1].min()), int(spike_remove_data[pos_col_1].max()) + 1)
    range_ypos = np.arange(int(spike_remove_data[pos_col_2].min()), int(spike_remove_data[pos_col_2].max()) + 1)

    # Counters for tracking the number of spikes in the data and the spike counter for individual figures
    total_spike_counter = 0
    figure_spike_counter = 0

    # Array to store the spike corrected y-values for later return of the dataframe
    new_y = []

    # Arrays for storing the x and y position of detected spikes. Used for marking peaks on overall plot
    spike_peaks_x = []
    spike_peaks_y = []

    # Array for storing the data around each detected spike for later plotting
    individual_plots_data = []

    # Loops for each X and Y position in the map
    for x_pos, y_pos in product(range_xpos, range_ypos):

        # Filters spike_remove_data to the X and Y position combination of the loop
        filtered_data = spike_remove_data[(spike_remove_data[pos_col_1] == x_pos) & (spike_remove_data[pos_col_2] == y_pos)]

        # Extracts the x (e.g., Raman shift) and y (e.g., Intensity) data from filtered_data
        x, y = filtered_data[x_column].values.astype(float), filtered_data[y_column].values.astype(float)

        # Creates an array for indexing each identified spike
        spikes = np.zeros(len(y), dtype=bool)

        # Smooths the boundary values at each end of the spectrum using nearby entries
        # This is done as spikes at boundaries have tendency to not be identified or removed.
        start_smooth = savgol_filter(y[2:5], window_length=2, polyorder=1)
        end_smooth = savgol_filter(y[-5:-2], window_length=2, polyorder=1)
        y[:2] = start_smooth[:2]
        y[-2:] = end_smooth[-2:]

        # Initial peak detection in spectra using no thresholds.
        peaks, data = find_peaks(y, prominence=0)

        # Extracts prominence data for each identified peak
        prominences = data['prominences']

        # Finds the peak widths and left/right end points for each identified peak
        widths, _, wlend, wrend = peak_widths(y, peaks)

        # Loops for each peak to identify the ones that meet the set prominence to width ratio.
        for peak, prom, width, lend, rend in zip(peaks, prominences, widths, wlend, wrend):
            if prom / width > prom_width_ratio:

                # If ratio is met, sets the value of the spikes array to True for the full width
                # of the peak (plus a few extra datapoints at each end)
                start_idx = max(int(lend) - 3, 0)
                end_idx = min(int(rend) + 3, len(y))
                spikes[start_idx:end_idx] = True

                # If overaal plot is True, stores the x and y peak positions for later plotting
                if overall_plot:
                    spike_peaks_x.append(x[peak])
                    spike_peaks_y.append(y[peak])

        # Flattens the spikes array to only keep the indices that are True
        spikes_only = np.flatnonzero(spikes)

        # Copies the y data (intensity) for correcting after spike removal
        y_corrected = y.copy()

        # When spikes are detected, groups nearby spikes together into regions
        if spikes_only.size > 0:
            spike_regions = np.split(spikes_only, np.where(np.diff(spikes_only) > 5)[0] + 1)

            # Keeps a count of the total number of spikes detected throughout the map
            total_spike_counter += len(spike_regions)

            # Loops for each spike region and removes spikes using a baseline removal method
            for region in spike_regions:

                # Finds the start and end indices of the region
                region_start, region_end = region[0], region[-1]

                # Where possible, expands the start and end window for removing spikes (set by expand_window)
                window_start = max(region_start - expand_window, 0)
                window_end = min(region_end + expand_window, len(y))

                # Creates an array from the start to end index
                window = np.arange(window_start, window_end)

                # Finds the indices in the window that are not spikes
                no_spikes = window[~spikes[window]]

                # Only performs the spike removal if if non spike datapoints present
                if len(no_spikes) > 1:

                    # Creates a baseline using the data points that are not spikes
                    baseline = polynomial.imodpoly(y[no_spikes], no_spikes, poly_order=pol_order)[0]

                    # Creates a linear function to estimate the baseline values in the window
                    baseline_function = interpolate.interp1d(no_spikes, baseline, kind='linear', fill_value='extrapolate')

                    # Replaces the spike data with the predicted intensity values for the baseline spike correction
                    y_corrected[region] = baseline_function(region)

        # To store all the corrected data, updates new_y with the spike corrected data
        new_y.extend(y_corrected)

        # Stores the plotting data for each individual spike (if set to True and spikes identified)
        if individual_plots and spikes_only.size > 0:

            # Loops for each region in the groups within spike_regions
            for region in spike_regions:

                # Counter for the figure name
                figure_spike_counter += 1

                # Finds the start and end indices of the region
                region_start, region_end = region[0], region[-1]

                # Where possible, expands the start and end window for removing spikes (set by expand_window)
                window_start = max(region_start - expand_window, 0)
                window_end = min(region_end + expand_window, len(y))

                # Identifies the peaks within this region
                peaks_in_region = peaks[(peaks >= region_start) & (peaks <= region_end)]

                # Identifies the prominences and widths of the peaks within this region
                proms_in_region, widths_in_region = zip(*[(prominences[peaks == p][0], widths[peaks == p][0]) for p in peaks_in_region])

                # Finds the peaks in the region that meet the set prominence width ratio
                spikes_in_region = [p for p, prom, width in zip(peaks_in_region, proms_in_region, widths_in_region)
                                             if prom / width > prom_width_ratio]

                # Stores the information for each plot for later use
                fig_data = {'x': x[window_start:window_end],
                            'y': y[window_start:window_end],
                            'y_corrected': y_corrected[window_start:window_end],
                            'x_spikes': x[spikes_in_region],
                            'y_spikes': y[spikes_in_region],
                            'title': f'Spike {figure_spike_counter} found at position: ({x_pos},{y_pos})',
                            }
                individual_plots_data.append(fig_data)

    print(f'Total number of unique spikes detected across all positions: {total_spike_counter}')

    # Only performed if overall_plot is set to True. Notice: Due to the continuous line plot,
    # there will be strange lines that go across the spectra as it moves to new spectra from
    # different positions.
    if overall_plot:

        # Plot of the original data with markings for the identified spikes
        plt.figure(figsize=(12, 6))
        plt.plot(spike_remove_data[x_column], spike_remove_data[y_column], color='blue', label='Original Data')
        plt.scatter(spike_peaks_x, spike_peaks_y, color='red', label='Detected Spike Peaks', zorder=5, s=50)
        plt.xlabel('Raman Shift $cm^{-1}$')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Original Data with Detected Spike Peaks (Prominence/Width Check)')
        plt.legend()
        plt.show()

        # Plot of the spike corrected data
        plt.figure(figsize=(12, 6))
        plt.plot(spike_remove_data[x_column], new_y, color='green', label='Corrected Data')
        plt.xlabel('Raman Shift $cm^{-1}$')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Spectra after Spike Removal')
        plt.legend()
        plt.show()

    # Only performed if individual_plots is set to True.
    if individual_plots:

        # Plots the spectra of each identified spike using the previous data in individual_plots_data
        for data in individual_plots_data:
            plt.figure(figsize=(8, 4))
            plt.plot(data['x'], data['y'], color='blue', label='Original Data')
            plt.scatter(data['x_spikes'], data['y_spikes'], color='red', label='Detected Spikes', zorder=5)
            plt.plot(data['x'], data['y_corrected'], color='green', linestyle='--', label='Corrected Data')
            plt.xlabel('Raman Shift $cm^{-1}$')
            plt.ylabel('Intensity (a.u.)')
            plt.title(data['title'])
            plt.legend()
            plt.show()

    # Creates a new column called Spike Corrected to store spike corrected data (new_y)
    spike_remove_data['Spike Corrected'] = new_y

    return spike_remove_data

# Example run
# from RamanMap1_ProcessFile import process_file
# file = "C:/Users/Benny/OneDrive - University of Birmingham/Machine Guides & Info/Raman/Raman - ID-IG Map/ASPN-raw.txt"
# processed_data = process_file(file)
# spikeless_data = spike_removal(processed_data, y_col=4, x_col=3, pol_order=2, overall_plot=True, individual_plots=True)
