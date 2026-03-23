# Copula Modeling for Bivariate Reliability (Time vs. Mileage)

## Background & Goal
You requested to investigate the statistical dependency between Failure Mileage and Failure Time using Copula functions. Since the reliability of diesel engine parts decreases as both time and mileage increase, these variables are not independent. Copulas allow us to decouple the marginal distributions from the dependence structure. 

This implementation plan details the Jupyter Notebook cells (Cells 9-11) we will create to model, fit, and visualize this relationship.

## User Review Required
> [!IMPORTANT]
> **1. Marginal Distributions Formulation**: 
> Currently, the plan transforms the raw `(Time, Mileage)` observations into uniform pseudo-observations `(U, V)` using the **Empirical CDF**. This is standard in purely evaluating Copula dependency without the bias of assuming a strict unified parametric marginal distribution. Does your assignment require using the previously fitted theoretical functions (e.g., Mixed Weibull) to compute the `(U, V)` inputs, or is the universal Empirical CDF sufficient?
> 
> **2. Choice of Copulas**: 
> The proposed plan evaluates 4 classic Archimedean/Elliptical Copulas (Clayton, Gumbel, Frank, Gaussian). Are there any other specific Copulas you would like integrated?

## Proposed Changes

### Cell 9: Data Correlation and Visual Exploration
- **Data Load**: Pair `fail_time` and `mileage_val` directly from `cleaned_data.csv`.
- **Rank Correlation Analysis**: Calculate and print Pearson's $r$, Spearman's $\rho$, and Kendall's $\tau$.
- **2D Joint Distribution Plot**: Generate a scatter plot overlaid with dual marginal histograms on the axes (using `seaborn.jointplot`) to visually establish the foundational correlation shape. (Output: SVG format, Times New Roman, no hardcoded title).

### Cell 10: Copula Parameter Estimation & Goodness-of-Fit
- **Pseudo-observations Transformation**: Map raw data to $[0, 1]^2$ space using the Empirical CDF.
- **Dependency Modeling**: Parameterize and fit the 4 Copula functions:
  - **Clayton Copula**: Asymmetric, strong left-tail dependence (sensitive to simultaneous early failures).
  - **Gumbel Copula**: Asymmetric, strong right-tail dependence (sensitive to simultaneous late wear-out).
  - **Frank Copula**: Symmetric dependence, good general fit for wide-ranging correlations.
  - **Gaussian Copula**: Symmetric baseline without tail bias.
- **Estimation Methodology**: Maximum Likelihood Estimation (MLE) or exact Canonical Inversion derived from Kendall's $\tau$, depending on Copula solvability.
- **Output**: Export estimated Copula parameters ($\theta$) and the Akaike Information Criterion (AIC) scores to a CSV file (`result/copula_analysis/copula_gof_results.csv`).

### Cell 11: Copula 3D & 2D Visualization
- Generate visual plots specifically isolating the best-fitting Copula (the one with the lowest AIC).
- **Plot 1 (2D Scatter & Contour)**: A 2D contour map of the Copula density, overlaid with the actual $[U, V]$ pseudo-observations scattered as points.
- **Plot 2 (3D Surface - PDF)**: A 3D wireframe/surface plot of the Bivariate Copula Probability Density Function $c(u,v)$.
- All visuals strictly adhere to prior formatting rules: PNG/SVG export, Times New Roman, English axis labels, and no overarching titles to ensure manuscript-ready purity.

## Verification Plan
1. **Sanity Check**: Review output correlations to confirm strong positive dependence (meaning longer time = higher mileage).
2. **AIC Assessment**: The lowest AIC mathematically proves the most accurate joint-behavior representation for this specific dataset.
