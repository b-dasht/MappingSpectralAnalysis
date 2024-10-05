from pybaselines import polynomial
import numpy as np

def baseline_correction(waves, intensities, polyorder=2, numstd=1):
    """
    Applies baseline correction to a single spectrum of intensity data using the Improved Modified Polynomial (IModPoly) method.

    Parameters:
      - waves: array representing the wavenumber values.
      - intensities: array of intensity values corresponding to each wavenumber.
      - plot: boolean indicating whether to plot the results.

    Returns:
      - intensities_corrected: array of baseline-corrected intensity values.
    """
    # Perform baseline correction using imodpoly
    baseline = polynomial.imodpoly(intensities, x_data=waves, poly_order = polyorder, num_std = numstd)
    baseline = np.array(baseline[0])  # Ensure baseline is a numpy array
    intensities_corrected = np.array(intensities) - baseline

    return intensities_corrected, baseline