# Raman Mapping Spectra Processing and Analysis

This repository contains scripts primarily designed for processing and analysing Raman mapping data obtained using the Renishaw WiRE software package. However, the scripts can easily be adapted for other software packages and data types.

## Features
- **Old Version**: 
  - Functional but contains bugs and lacks generalisation.
  - Originally developed for performing ID/IG ratio mapping of carbon fibres using Raman spectroscopy.
  
- **New Version**: 
  - More general and adaptable, providing greater flexibility for a variety of datasets.

## Upcoming Features
- **Fitting Script** (in development):
  - The new version of the fitting process will be implemented in two stages:
  
    1. **Initial Fit**: A preliminary fit where users can optionally select specific peaks (e.g., dominant peaks within the spectra).
    2. **Final Fit**: Adds other components, with constraints applied to limit how much the peaks from the first fit are allowed to vary.

This two-stage fitting process is designed to address the limitations of simultaneously modelling all components, aiming to improve precision and repeatability. It is intended to make the analysis more reliable and adaptable, particularly when dealing with less well-defined components or peaks.

The final aim is to develop this into a lightweight library for easier integration into other scripts and workflows.
