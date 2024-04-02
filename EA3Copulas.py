#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:35:14 2024

@author: Utilisateur
"""

import yfinance as yf
import numpy as np
from copulas.multivariate import GaussianMultivariate
from copulas.bivariate import Clayton, Frank, Gumbel
from copulas.univariate import UniformUnivariate
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
from pycop import archimedean

assets = {
    "S&P 500": "^GSPC",
    "MSCI Emerging Markets": "EEM",
    "Bloomberg Commodity Index": "CMOD.MI",
    "USD/CNY": "CNY=X",
    "German Bonds": "EXSB.DE" 
}

time_periods = {
    "Pre-Financial Crisis Growth": ("2005-01-01", "2007-12-31"),
    "Post-Financial Crisis Recovery": ("2009-01-01", "2011-12-31"),
    "COVID-19 Pandemic Impact": ("2020-01-01", "2022-12-31"),
    "Late-2017 Market Rally": ("2017-09-01", "2018-02-28"),
    "Trade War Uncertainty": ("2018-01-01", "2020-12-31"),
    "Renminbi Devaluation": ("2014-01-01", "2016-12-31"),
    "Trade War Period": ("2018-03-01", "2019-12-31"),
    "Normalisation of ECB Monetary Policy": ("2017-01-09", "2018-12-31"),
    "Chinese Economic Boom": ("2005-01-01", "2007-01-01")
}

def empirical_log_likelihood(copula, data, n_samples=100):
    samples = copula.sample(n_samples)
    empirical_likelihoods = copula.pdf(samples)
    return np.mean(np.log(empirical_likelihoods))

def transform_to_uniform(marginals):
    # Assuming 'marginals' is a pandas Series
    return marginals.rank(method='average') / (len(marginals) + 1)

def empirical_log_likelihood(copula, data, n_samples=100):
    samples = copula.sample(n_samples)
    empirical_likelihoods = copula.pdf(samples)
    return np.mean(np.log(empirical_likelihoods))

def calculate_aic(log_likelihood, num_parameters):
    return 2 * num_parameters - 2 * log_likelihood

def find_best_copula(asset1_returns, asset2_returns):
    # Transform to uniform distributions
    asset1_uniform = transform_to_uniform(asset1_returns)
    asset2_uniform = transform_to_uniform(asset2_returns)
    data = np.column_stack((asset1_uniform, asset2_uniform))

    # List of copulas to evaluate
    copulas = [
        GaussianMultivariate(),
        Clayton(),
        Frank(),
        Gumbel(),
        UniformUnivariate()
    ]

    aic_scores = {}
    best_copula = None
    best_aic = np.inf

    for copula in copulas:
        try:
            copula.fit(data)
            log_likelihood = empirical_log_likelihood(copula, data)
            # Assuming the 'parameters' can somehow give us the number of parameters for AIC calculation.
            # This is a simplification; you'll need to adjust based on your actual copula objects.
            num_parameters = 1  # Placeholder. Determine the correct number based on your copula type.
            aic = calculate_aic(log_likelihood, num_parameters)
            aic_scores[copula.__class__.__name__] = aic
            
            if aic < best_aic:
                best_aic = aic
                best_copula = copula
                
            print(f"{copula.__class__.__name__}: Log-likelihood = {log_likelihood}, AIC = {aic}")

        except Exception as e:
            print(f"An error occurred while fitting the copula: {e}")

    print(f"Best copula based on AIC: {best_copula} with AIC = {best_aic}")
    return best_copula, aic_scores




def calculate_dependency(asset1, asset2, time_period_name):
    asset1_data = yf.download(assets[asset1], start=time_periods[time_period_name][0], end=time_periods[time_period_name][1])
    asset2_data = yf.download(assets[asset2], start=time_periods[time_period_name][0], end=time_periods[time_period_name][1])

    if asset1_data.empty or asset2_data.empty:
        print(f"Data for {asset1} or {asset2} is empty. Skipping...")
        return
    
    asset1_returns = asset1_data['Close'].pct_change().dropna()
    asset2_returns = asset2_data['Close'].pct_change().dropna()

    asset1_returns, asset2_returns = asset1_returns.align(asset2_returns, join='inner')

    best_copula = find_best_copula(asset1_returns, asset2_returns)
    print(f"Best copula for {asset1} / {asset2} during {time_period_name}: {best_copula}")

    aligned_asset1, aligned_asset2 = asset1_returns.align(asset2_returns, join='inner')

    #Dependency mesures
    pearson_corr, pearson_pval = pearsonr(aligned_asset1, aligned_asset2)
    spearman_corr, spearman_pval = spearmanr(aligned_asset1, aligned_asset2)
    kendall_tau, kendall_pval = kendalltau(aligned_asset1, aligned_asset2)

    # Print the results
    print(f"{asset1} / {asset2} during {time_period_name}")
    print(f"Pearson Correlation Coefficient (r): {pearson_corr:.4f}, p-value: {pearson_pval:.4g}")
    print(f"Spearman's Rank Correlation Coefficient (rho): {spearman_corr:.4f}, p-value: {spearman_pval:.4g}")
    print(f"Kendall's Tau Correlation Coefficient (tau): {kendall_tau:.4f}, p-value: {kendall_pval:.4g}")
    print("\n")
    

for pair, periods in {
    ("S&P 500", "MSCI Emerging Markets"): ["Pre-Financial Crisis Growth", "Post-Financial Crisis Recovery", "COVID-19 Pandemic Impact"],
    ("Bloomberg Commodity Index", "S&P 500"): ["Late-2017 Market Rally", "Trade War Uncertainty", "COVID-19 Pandemic Impact"],
    ("MSCI Emerging Markets", "USD/CNY"): ["Chinese Economic Boom", "Renminbi Devaluation", "Trade War Uncertainty"],
    ("Bloomberg Commodity Index", "USD/CNY"): ["Late-2017 Market Rally", "Trade War Period", "COVID-19 Pandemic Impact"],
    ("German Bonds", "Bloomberg Commodity Index"): ["Normalisation of ECB Monetary Policy", "Trade War Uncertainty", "COVID-19 Pandemic Impact"]
}.items():
    for period in periods:
        calculate_dependency(pair[0], pair[1], period)
        

# Download data for S&P 500 and MSCI Emerging Markets for the specified period
sp500_data = yf.download("^GSPC", start="2005-01-01", end="2007-12-31")
emerging_data = yf.download("EEM", start="2005-01-01", end="2007-12-31")

# Calculate returns for S&P 500 and MSCI Emerging Markets
sp500_returns = sp500_data['Close'].pct_change().dropna()
emerging_returns = emerging_data['Close'].pct_change().dropna()

# Subsample data to reduce size
sp500_returns_subsampled = sp500_returns[::800]  # Sample every 800 data points
emerging_returns_subsampled = emerging_returns[::800]

# Create Gumbel copula
copula = archimedean(family="gumbel")

# Plot the CDF for the pair (S&P 500, MSCI Emerging Markets) during the specified period
copula.plot_cdf([sp500_returns_subsampled, emerging_returns_subsampled], plot_type="3d", Nsplit=50)
plt.title("Cumulative Distribution Function (CDF) of the Gumbel Copula for the Pair")
plt.xlabel("S&P 500 Returns")
plt.ylabel("MSCI Emerging Markets Returns")
plt.show()

# Plot the PDF for the pair (S&P 500, MSCI Emerging Markets) during the specified period
copula.plot_pdf([sp500_returns_subsampled, emerging_returns_subsampled], plot_type="3d", Nsplit=50, cmap="cividis")
plt.title("Probability Density Function (PDF) of the Gumbel Copula for the Pair")
plt.xlabel("S&P 500 Returns")
plt.ylabel("MSCI Emerging Markets Returns")
plt.show()


sp500_returns_uniform = transform_to_uniform(sp500_returns)
emerging_returns_uniform = transform_to_uniform(emerging_returns)

# Clayton and Gumbel
def calculate_tail_dependence(copula_name, parameter):
    if copula_name == "Clayton":
        lower_tail = 2 ** (-1 / parameter)
        upper_tail = 0  # Clayton copula has no upper tail dependence
        return lower_tail, upper_tail
    elif copula_name == "Gumbel":
        upper_tail = 2 - 2 ** (1 / parameter)
        lower_tail = 0  # Gumbel copula has no lower tail dependence
        return lower_tail, upper_tail
    else:
        return None, None  # For other copulas or non-Archimedean copulas

## For Clayton
best_copula_name = "Clayton"  
best_copula_param = 1.5 
lower_tail_dep, upper_tail_dep = calculate_tail_dependence(best_copula_name, best_copula_param)

print(f"Lower Tail Dependence: {lower_tail_dep}")
print(f"Upper Tail Dependence: {upper_tail_dep}")


copula = archimedean(family="clayton")

# Plot the CDF for the Clayton copula
copula.plot_cdf([sp500_returns_subsampled, emerging_returns_subsampled], plot_type="3d", Nsplit=50)
plt.title("Clayton Copula CDF for S&P 500 / MSCI Emerging Markets during COVID-19 Pandemic Impact")
plt.xlabel("S&P 500 Returns")
plt.ylabel("MSCI Emerging Markets Returns")
plt.show()

# Plot the PDF for the Clayton copula
copula.plot_pdf([sp500_returns_subsampled, emerging_returns_subsampled], plot_type="3d", Nsplit=50, cmap="cividis")
plt.title("Clayton Copula PDF for S&P 500 / MSCI Emerging Markets during COVID-19 Pandemic Impact")
plt.xlabel("S&P 500 Returns")
plt.ylabel("MSCI Emerging Markets Returns")
plt.show()

### For Frank : 

copula = archimedean(family="frank")

# Plot the CDF for the Frank copula
copula.plot_cdf([sp500_returns, emerging_returns], plot_type="3d", Nsplit=50)
plt.title("Frank Copula CDF for S&P 500 / MSCI Emerging Markets")
plt.xlabel("S&P 500 Returns")
plt.ylabel("MSCI Emerging Markets Returns")
plt.show()

# Plot the PDF for the Frank copula
copula.plot_pdf([sp500_returns, emerging_returns], plot_type="3d", Nsplit=50, cmap="cividis")
plt.title("Frank Copula PDF for S&P 500 / MSCI Emerging Markets")
plt.xlabel("S&P 500 Returns")
plt.ylabel("MSCI Emerging Markets Returns")
plt.show()
