# Raman Mapping Spectra Processing and Analysis

This repository contains scripts designed for processing, modelling, and analysing Raman mapping data. With minor adjustments, the same approach can also be applied to other mapping data types.

The toolbox is highly customisable and supports both single-stage fitting (using predefined component lists) and multi-stage fitting.

The main advantage of this tool lies in **multi-stage fitting**, where components and restrictions are progressively introduced in successive stages. This approach helps stabilise the fitting process, particularly in regions of the spectrum with multiple overlapping components.

Without this staged approach, primary peaks can shift significantly as secondary peaks vary between positions, sometimes even leading to component “switching”. Such behaviour complicates analysis, reduces reliability, and makes automatic fitting challenging, especially when scaling to large datasets.

## Why Multi-Stage Fitting?
- Prevents unintended shifts of primary peaks when secondary components vary.
- Reduces the risk of component switching between fits.
- Improves repeatability and reliability across mapping datasets.
- Allows flexible control over peak restrictions through user-defined bounds.

## Example Workflow
Although more stages can be used, a typical two-stage process looks like this:
1. **First Stage Fit** – Perform an initial fitting using only the primary user-defined components (with their bounds and restrictions). All other potential components are ignored.
2. **Second Stage Fit** – Add additional user-defined components. Parameters from the first stage (e.g., position, intensity, width) are reused as starting points, but their allowed ranges are now restricted relative to the user’s chosen tolerance.

This progressive fitting strategy is intended to make the analysis more precise, repeatable, and scalable, while remaining flexible for complex spectra.
