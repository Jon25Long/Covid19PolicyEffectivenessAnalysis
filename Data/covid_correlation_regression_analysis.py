#!/usr/bin/env python3
"""
Program: covid_correlation_regression_analysis.py
Purpose: Analyze COVID-19 data with correlation and regression analysis
Date: December 31, 2025
Description: This program performs:
  1. Import owid-covid-data.csv
  2. Remove missing values
  3. Pearson correlation between stringency_index and reproduction_rate
  4. Spearman correlation between stringency_index and reproduction_rate
  5. Linear regression between stringency_index and reproduction_rate
  6. Calculate variance for both variables
  7. Export all results to Excel
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
import os
from datetime import datetime

# Define file paths
data_folder = "/Users/jonlong/PublicHealthPortfolio/my-projects/policy-effectiveness-analysis/Data"
input_file = os.path.join(data_folder, "owid-covid-data.csv")
output_file = os.path.join(data_folder, "covid_analysis_results.xlsx")

print("=" * 70)
print("COVID-19 Correlation and Regression Analysis")
print("=" * 70)
print()

# Step 1: Import the CSV file
print("Step 1: Importing data...")
covid_raw = pd.read_csv(input_file)
print(f"  - Loaded {len(covid_raw):,} rows and {len(covid_raw.columns)} columns")
print()

# Step 2: Remove missing values for the two variables of interest
print("Step 2: Removing missing values...")
covid_clean = covid_raw[['stringency_index', 'reproduction_rate']].dropna()
print(f"  - Original rows: {len(covid_raw):,}")
print(f"  - Rows with complete data: {len(covid_clean):,}")
print(f"  - Rows removed: {len(covid_raw) - len(covid_clean):,}")
print()

# Extract the two variables
stringency = covid_clean['stringency_index'].values
reproduction = covid_clean['reproduction_rate'].values

# Step 3 & 4: Pearson and Spearman Correlations
print("Step 3 & 4: Computing correlations...")

# Pearson correlation
pearson_corr, pearson_pval = stats.pearsonr(stringency, reproduction)
print(f"  - Pearson correlation: {pearson_corr:.6f}")
print(f"  - Pearson p-value: {pearson_pval:.6e}")

# Spearman correlation
spearman_corr, spearman_pval = stats.spearmanr(stringency, reproduction)
print(f"  - Spearman correlation: {spearman_corr:.6f}")
print(f"  - Spearman p-value: {spearman_pval:.6e}")
print()

# Step 5: Linear Regression
print("Step 5: Performing linear regression...")

# Reshape data for sklearn
X = stringency.reshape(-1, 1)
y = reproduction

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Get predictions
y_pred = model.predict(X)

# Calculate regression statistics
r_squared = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

# Calculate standard errors and t-statistics
n = len(y)
residuals = y - y_pred
rss = np.sum(residuals**2)
se_residual = np.sqrt(rss / (n - 2))

# Standard error for slope
x_mean = np.mean(X)
se_slope = se_residual / np.sqrt(np.sum((X - x_mean)**2))

# Standard error for intercept
se_intercept = se_residual * np.sqrt(1/n + x_mean**2 / np.sum((X - x_mean)**2))

# T-statistics
t_slope = model.coef_[0] / se_slope
t_intercept = model.intercept_ / se_intercept

# P-values
p_slope = 2 * (1 - stats.t.cdf(abs(t_slope), n - 2))
p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), n - 2))

# F-statistic for overall model
tss = np.sum((y - np.mean(y))**2)
f_statistic = (tss - rss) / rss * (n - 2)
p_f_statistic = 1 - stats.f.cdf(f_statistic, 1, n - 2)

print(f"  - Intercept: {model.intercept_:.6f}")
print(f"  - Slope: {model.coef_[0]:.6f}")
print(f"  - R-squared: {r_squared:.6f}")
print(f"  - RMSE: {rmse:.6f}")
print(f"  - F-statistic: {f_statistic:.6f}")
print()

# Step 6: Calculate Variance
print("Step 6: Calculating variance and descriptive statistics...")

stringency_stats = {
    'n': len(stringency),
    'mean': np.mean(stringency),
    'std': np.std(stringency, ddof=1),
    'variance': np.var(stringency, ddof=1),
    'min': np.min(stringency),
    'max': np.max(stringency)
}

reproduction_stats = {
    'n': len(reproduction),
    'mean': np.mean(reproduction),
    'std': np.std(reproduction, ddof=1),
    'variance': np.var(reproduction, ddof=1),
    'min': np.min(reproduction),
    'max': np.max(reproduction)
}

print(f"  Stringency Index:")
print(f"    - Mean: {stringency_stats['mean']:.6f}")
print(f"    - Std Dev: {stringency_stats['std']:.6f}")
print(f"    - Variance: {stringency_stats['variance']:.6f}")
print()
print(f"  Reproduction Rate:")
print(f"    - Mean: {reproduction_stats['mean']:.6f}")
print(f"    - Std Dev: {reproduction_stats['std']:.6f}")
print(f"    - Variance: {reproduction_stats['variance']:.6f}")
print()

# Step 7: Export All Results to Excel
print("Step 7: Exporting results to Excel...")

# Create Excel writer
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    
    # Sheet 1: Cleaned Data Sample (first 100 observations)
    covid_sample = covid_clean.head(100)
    covid_sample.to_excel(writer, sheet_name='Data Sample', index=True)
    
    # Sheet 2: Descriptive Statistics and Variance
    variance_df = pd.DataFrame({
        'Variable': ['stringency_index', 'reproduction_rate'],
        'N': [stringency_stats['n'], reproduction_stats['n']],
        'Mean': [stringency_stats['mean'], reproduction_stats['mean']],
        'Std_Dev': [stringency_stats['std'], reproduction_stats['std']],
        'Variance': [stringency_stats['variance'], reproduction_stats['variance']],
        'Min': [stringency_stats['min'], reproduction_stats['min']],
        'Max': [stringency_stats['max'], reproduction_stats['max']]
    })
    variance_df.to_excel(writer, sheet_name='Descriptive Statistics', index=False)
    
    # Sheet 3: Correlation Results
    correlation_df = pd.DataFrame({
        'Correlation_Type': ['Pearson', 'Spearman'],
        'Variable1': ['stringency_index', 'stringency_index'],
        'Variable2': ['reproduction_rate', 'reproduction_rate'],
        'Correlation': [pearson_corr, spearman_corr],
        'P_Value': [pearson_pval, spearman_pval]
    })
    correlation_df.to_excel(writer, sheet_name='Correlation Results', index=False)
    
    # Sheet 4: Regression Parameter Estimates
    regression_params_df = pd.DataFrame({
        'Variable': ['Intercept', 'stringency_index'],
        'Coefficient': [model.intercept_, model.coef_[0]],
        'Standard_Error': [se_intercept, se_slope],
        't_Value': [t_intercept, t_slope],
        'P_Value': [p_intercept, p_slope]
    })
    regression_params_df.to_excel(writer, sheet_name='Regression Parameters', index=False)
    
    # Sheet 5: Regression Model Fit Statistics
    fit_stats_df = pd.DataFrame({
        'Statistic': ['R-Squared', 'Adjusted R-Squared', 'RMSE', 'MSE', 'N'],
        'Value': [
            r_squared,
            1 - (1 - r_squared) * (n - 1) / (n - 2),
            rmse,
            mse,
            n
        ]
    })
    fit_stats_df.to_excel(writer, sheet_name='Model Fit Statistics', index=False)
    
    # Sheet 6: Regression ANOVA Table
    anova_df = pd.DataFrame({
        'Source': ['Model', 'Residual', 'Total'],
        'DF': [1, n - 2, n - 1],
        'Sum_of_Squares': [tss - rss, rss, tss],
        'Mean_Square': [(tss - rss) / 1, rss / (n - 2), np.nan],
        'F_Value': [f_statistic, np.nan, np.nan],
        'P_Value': [p_f_statistic, np.nan, np.nan]
    })
    anova_df.to_excel(writer, sheet_name='ANOVA Table', index=False)
    
    # Sheet 7: Summary
    summary_df = pd.DataFrame({
        'Analysis': [
            'Data Import',
            'Data Cleaning',
            'Pearson Correlation',
            'Spearman Correlation',
            'Linear Regression',
            'Variance Calculation',
            'Export Date'
        ],
        'Details': [
            f'{len(covid_raw):,} rows imported',
            f'{len(covid_clean):,} complete cases',
            f'r = {pearson_corr:.4f}, p = {pearson_pval:.4e}',
            f'rho = {spearman_corr:.4f}, p = {spearman_pval:.4e}',
            f'R² = {r_squared:.4f}, F = {f_statistic:.2f}',
            f'Var(stringency) = {stringency_stats["variance"]:.2f}, Var(reproduction) = {reproduction_stats["variance"]:.4f}',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
    })
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print(f"  - Results exported to: {output_file}")
print()

print("=" * 70)
print("Analysis Complete!")
print("=" * 70)
print()
print("Summary of Results:")
print(f"  • Pearson correlation: {pearson_corr:.4f} (p = {pearson_pval:.4e})")
print(f"  • Spearman correlation: {spearman_corr:.4f} (p = {spearman_pval:.4e})")
print(f"  • Linear regression R²: {r_squared:.4f}")
print(f"  • Regression equation: reproduction_rate = {model.intercept_:.4f} + {model.coef_[0]:.6f} * stringency_index")
print(f"  • Variance(stringency_index): {stringency_stats['variance']:.2f}")
print(f"  • Variance(reproduction_rate): {reproduction_stats['variance']:.4f}")
print()
