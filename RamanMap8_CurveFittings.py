import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
from itertools import product


def als_baseline(y, lam=5e10, p=0.001, n_iter=10):
    """
    Asymmetric Least Squares baseline correction.

    Parameters:
        y (array): Input signal.
        lam (float): Smoothness parameter.
        p (float): Asymmetry parameter.
        n_iter (int): Number of iterations.

    Returns:
        array: Computed baseline.
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.T)  # Compute D^T * D

    w = np.ones(L)
    for _ in range(n_iter):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = W + D
        baseline = spsolve(Z, w * y)
        w = p * (y > baseline) + (1 - p) * (y < baseline)

    return baseline

def pseudo_voigt(x, *params):
    num_components = len(params) // 4
    y = np.zeros_like(x)
    for i in range(num_components):
        amp, pos, width, gfrac = params[i * 4:(i + 1) * 4]
        sigma = width / np.sqrt(2 * np.log(2))
        gamma = width / 2
        gaussian = amp * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2))
        lorentzian = amp * (gamma ** 2) / ((x - pos) ** 2 + gamma ** 2)
        y += gfrac * gaussian + (1 - gfrac) * lorentzian
    return y

# Validate parameters and apply defaults
def validate_parameters(fits, regions, base, x_range):
    validated_fits = []
    default_fwhm_range = (0, np.inf)
    default_intensity_range = (0, 100)
    default_gaussian_range = [0, 1]

    validated_regions = []
    for i, region in enumerate(regions):
        region_defaults = {
            "Name": f"Region {i + 1}",
            "Start": min(x_range),
            "End": max(x_range),
        }

        # Apply defaults to regions
        for key, value in region_defaults.items():
            if key not in region or region[key] in [None, ""]:
                region[key] = value

        validated_regions.append(region)

    for sublist in fits:  # Iterate over each list in fits
        validated_sublist = []  # Maintain sublist structure
        for i, comp in enumerate(sublist):
            defaults = {
                "Name": f"Comp {i + 1}",
                "Gaussian Range": default_gaussian_range,
                "Position": np.mean(x_range),
                "Position Range": x_range,
                "Intensity (%)": 100,
                "Intensity Range (%)": default_intensity_range,
                "FWHM": 30,
                "FWHM Range": default_fwhm_range,
                "Subsequent Fitting Restrictions (%)": 20,
                "Region": 1,
            }

            # Apply defaults only if the value is missing, None, or empty
            for key, value in defaults.items():
                if key not in comp or comp[key] in [None, ""]:
                    comp[key] = value

            # Validate ranges
            if not (min(comp["Intensity Range (%)"]) <= comp["Intensity (%)"] <= max(comp["Intensity Range (%)"])):
                raise ValueError(f"Invalid intensity for component {comp["Name"]}")
            if not (min(comp["Position Range"]) <= comp["Position"] <= max(comp["Position Range"])):
                raise ValueError(f"Invalid position for component {comp["Name"]}")
            if not (min(comp["FWHM Range"]) <= comp["FWHM"] <= max(comp["FWHM Range"])):
                raise ValueError(f"Invalid FWHM for component {comp["Name"]}")

            validated_sublist.append(comp)  # Append component to its sublist

        validated_fits.append(validated_sublist)  # Append sublist to main list



    # Ensure no overlapping regions
    sorted_regions = sorted(validated_regions, key=lambda r: r["Start"])
    for i in range(1, len(sorted_regions)):
        if sorted_regions[i]["Start"] < sorted_regions[i - 1]["End"]:
            raise ValueError(f"Regions overlap: {sorted_regions[i - 1]} and {sorted_regions[i]}")

    # Validate `base` parameters (ensuring missing or empty values get defaults)
    base_defaults = {
        "Smoothness Parameter (Lambda)": 6e7,
        "Asymmetry Parameter (p)": 0.002,
        "Number of Iterations": 20,
    }

    for key, value in base_defaults.items():
        if key not in base or base[key] in [None, ""]:
            base[key] = value

    return validated_fits, validated_regions, base  # Keep `base` as a dictionary

# Main curve fitting function
def fit_data(fitting_data, x_col=3, y_col=0, fits=[], base=None, regions=None, fitting_figs=False):

    pos_col_1, pos_col_2 = fitting_data.columns[:2]
    x_axis, y_axis = fitting_data.columns[x_col - 1], fitting_data.columns[y_col - 1]

    range_xpos = np.arange(int(fitting_data[pos_col_1].min()), int(fitting_data[pos_col_1].max()) + 1)
    range_ypos = np.arange(int(fitting_data[pos_col_2].min()), int(fitting_data[pos_col_2].max()) + 1)

    x_range = [fitting_data.iloc[:, x_col - 1].min(), fitting_data.iloc[:, x_col - 1].max()]
    fits, regions, base = validate_parameters(fits, regions, base, x_range)

    fitting_details = []
    r_squared_store = {}
    baseline_array = np.zeros(len(fitting_data))
    component_fits_dict = {comp["Name"]: np.zeros(len(fitting_data)) for stage in fits for comp in stage}
    combined_fit_array = np.zeros(len(fitting_data))


    for x_pos, y_pos in product(range(5), range(1)): #(range_xpos, range_ypos):

        # Filter spectrum for current (x, y)
        filtered_index = (fitting_data[pos_col_1] == x_pos) & (fitting_data[pos_col_2] == y_pos)
        filtered_data = fitting_data[filtered_index]

        data_x = filtered_data[x_axis].values.astype(float)
        data_y = filtered_data[y_axis].values.astype(float)

        # Compute ALS baseline correction
        baseline = als_baseline(data_y, lam=base["Smoothness Parameter (Lambda)"], p=base["Assymetry Parameter (p)"],
                                n_iter=base["Number of Iterations"])
        data_y_corrected = data_y - baseline

        # Store baseline in full-size array
        baseline_array[filtered_index] = baseline
        combined_fit = np.zeros_like(data_y_corrected)

        # Max intensity for this spectrum
        max_intensity = np.max(data_y)

        initial_guess, bounds_lower, bounds_upper = [], [], []

        for stage_idx, fit_stage in enumerate(fits):
            for region in regions:
                region_name = region["Name"]
                region_start = region["Start"]
                region_end = region["End"]

                mask = (region_start <= data_x) & (data_x <= region_end)
                region_x = data_x[mask]
                region_y = data_y_corrected[mask]

                region_fits = []
                for previous_stage in range(stage_idx + 1):  # Include previous components
                    region_fits += [comp for comp in fits[previous_stage] if comp["Region"] == regions.index(region) + 1]

                #Skip to the next loop if there are not components to this for this region during this loop
                if not region_fits:
                    continue

                initial_guess, bounds_lower, bounds_upper = [], [], []

                for comp in region_fits:
                    # Apply stepwise fitting restrictions if not first fit
                    combined_fit[mask] = 0


                    latest_fit = next((fit for fit in reversed(fitting_details)
                                       if fit["X"] == x_pos and fit["Y"] == y_pos and fit["Component"] == comp["Name"]), None)

                    # Set previous values in one go if a matching fit is found
                    set_amp, set_pos, set_fwhm = (
                        (latest_fit["Amplitude"], latest_fit["Position"], latest_fit["FWHM"]) if latest_fit else (None, None, None)
                    )
                    if latest_fit and comp["Name"] in latest_fit["Component"]:
                        min_amp_bound = set_amp * (1 - (comp["Subsequent Fitting Restrictions (%)"] / 100))
                        max_amp_bound = set_amp * (1 + (comp["Subsequent Fitting Restrictions (%)"] / 100))
                        min_pos_bound = set_pos * (1 - (comp["Subsequent Fitting Restrictions (%)"] / 100))
                        max_pos_bound = set_pos * (1 + (comp["Subsequent Fitting Restrictions (%)"] / 100))
                        min_fwhm_bound = set_fwhm * (1 - (comp["Subsequent Fitting Restrictions (%)"] / 100))
                        max_fwhm_bound = set_fwhm * (1 + (comp["Subsequent Fitting Restrictions (%)"] / 100))
                    else:
                        set_amp = (comp["Intensity (%)"] / 100) * max_intensity
                        min_amp_bound = (min(comp["Intensity Range (%)"]) / 100) * max_intensity
                        max_amp_bound = (max(comp["Intensity Range (%)"]) / 100) * max_intensity
                        set_pos = comp["Position"]
                        min_pos_bound = min(comp["Position Range"])
                        max_pos_bound = max(comp["Position Range"])
                        set_fwhm = comp["FWHM"]
                        min_fwhm_bound = min(comp["FWHM Range"])
                        max_fwhm_bound = max(comp["FWHM Range"])


                    lower_gfrac, upper_gfrac = min(comp["Gaussian Range"]), max(comp["Gaussian Range"])

                    initial_guess.extend([
                        set_amp, set_pos, set_fwhm, (lower_gfrac + upper_gfrac) / 2
                    ])
                    bounds_lower.extend([
                        min_amp_bound, min_pos_bound, min_fwhm_bound, lower_gfrac
                    ])
                    bounds_upper.extend([
                        max_amp_bound, max_pos_bound, max_fwhm_bound, upper_gfrac
                    ])

                try:
                    # Perform curve fitting
                    popt, _ = curve_fit(pseudo_voigt, region_x, region_y, p0=initial_guess, bounds=(bounds_lower, bounds_upper))
                    region_combined_fit = pseudo_voigt(region_x, *popt)

                    # Store combined fit for this region
                    combined_fit[mask] += region_combined_fit

                    for i, comp in enumerate(region_fits):
                        component_fit = pseudo_voigt(region_x, *popt[i * 4:(i + 1) * 4])
                        filtered_array = component_fits_dict[comp["Name"]][filtered_index]
                        filtered_array[mask] = component_fit
                        component_fits_dict[comp["Name"]][filtered_index] = filtered_array

                        # Remove existing entry for this X, Y, Component to avoid duplicates
                        fitting_details = [entry for entry in fitting_details if not (
                            entry["X"] == x_pos and entry["Y"] == y_pos and entry["Component"] == comp["Name"]
                        )]
                        # Store fitting parameters
                        fitting_details.append({
                            "X": x_pos,
                            "Y": y_pos,
                            "Region": region_name,
                            "Component": comp["Name"],
                            "Amplitude": popt[i * 4],
                            "Position": popt[i * 4 + 1],
                            "FWHM": popt[i * 4 + 2],
                            "Gaussian Fraction": popt[i * 4 + 3],
                            "Area": np.abs(np.trapz(component_fit - baseline[mask], region_x))
                        })


                except Exception as e:
                    print(f"Error during fitting of Region: {region_name} in Position: X={x_pos}, Y={y_pos}: {e}")

                    # Assign zero values for failed fits
                    for comp in region_fits:
                        component_fits_dict[comp["Name"]][filtered_index] = 0

            combined_fit_array[filtered_index] = combined_fit

        if fitting_figs:
            plt.figure(figsize=(8, 6))
            plt.plot(data_x, data_y, "k-", label="Original Spectrum", linewidth=2)

            r_squared_text = ""  # Initialize text string for storing R² values

            # Iterate over regions and plot baseline, combined fit, and calculate R² separately
            for region in regions:
                region_mask = (data_x >= region["Start"]) & (data_x <= region["End"])

                plt.axvline(region["Start"], linestyle="dotted", color="blue")
                plt.axvline(region["End"], linestyle="dotted", color="blue")
                plt.text((region["Start"] + region["End"]) / 2, (min(data_y) - 0.05), region["Name"],
                         ha="center", fontsize=10, color="blue", va="top")

                # Plot baseline within the region
                plt.plot(data_x[region_mask], baseline[region_mask], "-",
                         label="Baseline" if region == regions[0] else "", linewidth=2.5, color="grey")

                # Plot combined fit within the region
                plt.plot(data_x[region_mask], (combined_fit + baseline)[region_mask], "b-",
                         label="Combined Fit" if region == regions[0] else "", linewidth=2)

                # Compute R² for the current region
                ss_total_region = np.sum((data_y[region_mask] - np.mean(data_y[region_mask])) ** 2)
                ss_residual_region = np.sum((data_y[region_mask] - (combined_fit[region_mask] + baseline[region_mask])) ** 2)
                r_squared_region = 1 - (ss_residual_region / ss_total_region) if ss_total_region > 0 else np.nan
                r_squared_store[(x_pos, y_pos, region["Name"])] = r_squared_region

                # Append R² value to text box content
                r_squared_text += f"$R^2$ ({region['Name']}) = {r_squared_region:.4f}\n"

                # Iterate over all stored components and plot them within the region
                for comp_name, comp_fit_values in component_fits_dict.items():
                    plt.plot(data_x[region_mask], comp_fit_values[filtered_index][region_mask] + baseline[region_mask],
                             "--", label=comp_name, alpha=0.7)
            # Add single R² text box in the top right
            plt.text(0.95, 0.95, r_squared_text.strip(), transform=plt.gca().transAxes,
                     verticalalignment="top", horizontalalignment="right", fontsize=10,
                     bbox=dict(facecolor="white", edgecolor="black"))


            # Plot formatting
            plt.xlabel(x_axis.replace("_", " "))
            plt.ylabel(y_axis.replace("_", " "))
            plt.title(f"Fitting Results for ({x_pos}, {y_pos})")
            plt.legend(loc=2)
            plt.show()
        else:
            print(f"Completed fitting of X={x_pos}, Y={y_pos}")

    # Organise fitting_data columns
    fitting_data["Baseline"] = baseline_array
    for comp in component_fits_dict:
        fitting_data[f"{comp} Component"] = component_fits_dict[comp]
        fitting_data[f"{comp} Component (+ Baseline)"] = component_fits_dict[comp] + baseline_array
    fitting_data["Combined Fit"] = combined_fit_array
    fitting_data["Combined Fit (+ Baseline)"] = combined_fit_array + baseline_array

    fitting_details = pd.DataFrame(fitting_details)

    return fitting_data, fitting_details, r_squared_store


# Example usage
from RamanMap1_ProcessFile import process_file
from RamanMap2_SpikeRemoval import spike_removal
from RamanMap3_PCA import pca_noise_removal
from RamanMap4_Smooth import smooth_data
from RamanMap5_Baseline import baseline_correction
from RamanMap6_Crop import crop_data
from RamanMap7_Normalise import normalise_data

file = "C:/Users/Benny/OneDrive - University of Birmingham/Machine Guides & Info/Raman/Raman - ID-IG Map/ASPN-raw.txt"
# file = "C:/Users/Administrator/Dropbox/Personal/PythonScripts/ASPN-raw.txt"
processed_data = process_file(file, zero_xy=True)
processed_data = spike_removal(processed_data)
processed_data = pca_noise_removal(processed_data, pca_comp=6)
processed_data = smooth_data(processed_data, window_len=20)
processed_data = baseline_correction(processed_data)
# processed_data = crop_data(processed_data, 950, 1800)
processed_data = normalise_data(processed_data, each_spectra=False)


# Define multiple modelling regions
regions = [
    {
        "Name": "Region 1",
        "Start": 1000,
        "End": 1750
    },
    {
        "Name": "Region 2",
        "Start": 500,
        "End": 950
    }
]

base = {
        "Smoothness Parameter (Lambda)": 6e7,
        "Assymetry Parameter (p)": 0.002,
        "Number of Iterations": 20
        }
# Define fits with "Gaussian Range"
first_fits = [
    {
     "Name": "D",
     "Gaussian Range": [0.8, 1],
     "Position": 1370,
     "Position Range": (1340, 1405),
     "Intensity (%)": 100,
     "Intensity Range (%)": (0, 100),
     "FWHM": 30,
     "FWHM Range": (20, 100),
     "Subsequent Fitting Restrictions (%)": 5,
     "Region": 1
     },
    {
     "Name": "G",
     "Gaussian Range": [0.8, 1],
     "Position": 1625,
     "Position Range": (1590, 1640),
     "Intensity (%)": 100,
     "Intensity Range (%)": (0, 100),
     "FWHM": 30,
     "FWHM Range": (20, 100),
     "Subsequent Fitting Restrictions (%)": 5,
     "Region": 1
     }
]

second_fits = [
    {
     "Name": "D2",
     "Gaussian Range": [0.9, 1],
     "Position": 1200,
     "Position Range": (1120, 1280),
     "Intensity (%)": 10,
     "Intensity Range (%)": (0, 30),
     "FWHM": 80,
     "FWHM Range": (50, 150),
     "Subsequent Fitting Restrictions (%)": 10,
     "Region": 1
     },
    {
     "Name": "G2",
     "Gaussian Range": [0.9, 1],
     "Position": 1570,
     "Position Range": (1530, 1590),
     "Intensity (%)": 10,
     "Intensity Range (%)": (0, 30),
     "FWHM": 80,
     "FWHM Range": (50, 100),
     "Subsequent Fitting Restrictions (%)": 10,
     "Region": 1
     }
]

third_fits = [
    {
     "Name": "Third1",
     "Gaussian Range": [0.9, 1],
     "Position": 550,
     "Position Range": (520, 700),
     "Intensity (%)": 10,
     "Intensity Range (%)": (0, 200),
     "FWHM": 20,
     "FWHM Range": (0, 100),
     "Subsequent Fitting Restrictions (%)": 10,
     "Region": 2
     },
    {
     "Name": "Third2",
     "Gaussian Range": [0.9, 1],
     "Position": 700,
     "Position Range": (620, 900),
     "Intensity (%)": 10,
     "Intensity Range (%)": (0, 100),
     "FWHM": 20,
     "FWHM Range": (0, 100),
     "Subsequent Fitting Restrictions (%)": 10,
     "Region": 2
     }
]

fits = [first_fits, second_fits, third_fits]

fitted_data, fitting_parameters, r_squared = fit_data(
    processed_data,
    x_col=3,
    y_col=0,
    fits=fits,
    base=base,
    regions=regions,
    fitting_figs=True,
)
