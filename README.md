# Raman Mapping Spectra Processing and Analysis

This repository contains scripts primarily designed for processing and analysing Raman mapping data obtained using the Renishaw WiRE software package. However, the scripts can be easily adapted for other software packages and data types.

## Features
- **Old Version**: 
  - Functional but contains bugs and lacks generalisation.
  - Initially developed for performing ID/IG ratio mapping of carbon fibres using Raman spectroscopy.
  
- **New Version**: 
  - More general and adaptable, offering improved flexibility for various datasets.

## Upcoming Features
- **Fitting Script** (in development):
  - The new version of the fitting process will be implemented in two stages:
  
    1. **Initial Fit**: An optional preliminary fit where users can select specific peaks (e.g., dominant peaks).
    2. **Final Fit**: Adds other components, with constraints applied to limit how much the initially chosen peaks can vary from their initial fit.

This two-stage fitting process is designed to overcome the limitations of modelling all components simultaneously, aiming to improve precision and repeatability. It is intended to make analysis easier and facilitate future expansion of the tool.
