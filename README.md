# COVID-19 Policy Effectiveness Analysis

## Overview

This project analyzes the relationship between government policy stringency and COVID-19 transmission rates using the **Oxford COVID-19 Government Response Tracker** (OxCGRT) stringency index and viral reproduction rates from the Our World in Data (OWID) COVID-19 dataset.

## Research Question

**Does government policy strictness actually reduce viral transmission?**

This analysis explores whether stricter containment policies (lockdowns, stay-at-home orders, school closures) correlate with reduced transmission rates as measured by the reproduction rate (R).

## Key Variables

- **Independent Variable**: `stringency_index` - Oxford COVID-19 Government Response Tracker composite measure of policy stringency (0-100 scale)
- **Dependent Variable**: `reproduction_rate` - The average number of new infections caused by a single infected individual

## Project Structure

```
policy-effectiveness-analysis/
├── README.md                           # This file
├── .gitignore                          # Git ignore patterns
├── Data/                               # Data files and analysis scripts
│   ├── covid_data_cleaner.py           # Data preprocessing script
│   ├── covid_correlation_regression_analysis.py  # Python statistical analysis
│   ├── covid_correlation_regression_analysis.sas # SAS statistical analysis
│   ├── covid_analysis_results.xlsx     # Analysis output (generated)
│   └── owid-covid-data-cleaned.xlsx    # Cleaned dataset (generated)
└── notebooks/                          # Jupyter notebooks
    └── policy_effectiveness_analysis-final.ipynb  # Complete analysis workflow
```

## Data Source

- **Dataset**: Our World in Data COVID-19 Dataset
- **URL**: <https://github.com/owid/covid-19-data>
- **File**: `owid-covid-data.csv`
- **Citation**: Hasell, J., Mathieu, E., Beltekian, D. et al. (2020). A cross-country database of COVID-19 testing. *Scientific Data*, 7, 345.

## Methodology

### 1. Data Cleaning

- Import OWID COVID-19 dataset
- Remove columns with >80% missing values
- Drop rows with missing values for key variables
- Export cleaned dataset

### 2. Statistical Analysis

**Correlation Analysis:**

- Pearson correlation coefficient (parametric)
- Spearman rank correlation coefficient (non-parametric)

**Regression Analysis:**

- Linear regression modeling stringency_index → reproduction_rate
- Calculate R², RMSE, standard errors, and confidence intervals
- Test statistical significance of relationship

**Variance Analysis:**

- Calculate variance for both independent and dependent variables

### 3. Visualization & Interpretation

- Time series plots
- Scatter plots with regression lines
- Distribution analysis
- Missing data patterns

## Implementation

### Python Implementation

**Prerequisites:**

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl
```

**Run Data Cleaning:**

```bash
cd Data
python covid_data_cleaner.py
```

**Run Statistical Analysis:**

```bash
python covid_correlation_regression_analysis.py
```

**Run Jupyter Notebook:**

```bash
cd notebooks
jupyter notebook policy_effectiveness_analysis-final.ipynb
```

### SAS Implementation

**Prerequisites:**

- SAS 9.4 or later
- PROC IMPORT, PROC CORR, PROC REG capabilities

**Run Analysis:**

```sas
%include 'Data/covid_correlation_regression_analysis.sas';
```

## Expected Outputs

1. **Console Output:**
   - Data import summary
   - Sample size before/after cleaning
   - Correlation coefficients and p-values
   - Regression coefficients, R², and significance tests
   - Variance calculations

2. **Excel File** (`covid_analysis_results.xlsx`):
   - Summary statistics
   - Correlation results
   - Regression coefficients and diagnostics
   - Confidence intervals

3. **Visualizations:**
   - Scatter plots with trend lines
   - Time series analysis
   - Distribution plots
   - Missing data heatmaps

## Key Findings

*Note: Run the analysis to generate findings based on your dataset version and time period.*

The analysis tests the hypothesis that stricter government policies (higher stringency index) are associated with lower viral transmission (lower reproduction rate). Results may challenge or confirm prevailing narratives about lockdown effectiveness.

## Information Theory Context

According to information theory, information content is inversely proportional to the probability of an outcome: **I ∝ 1/P(outcome)**. Surprising results contain more information than expected ones. This analysis explores whether the relationship between policy stringency and transmission is as straightforward as commonly assumed.

## Technical Notes

- **Missing Data Handling**: Complete case analysis (listwise deletion)
- **Statistical Significance Level**: α = 0.05
- **Regression Assumptions**: Linearity, independence, homoscedasticity, normality of residuals should be verified
- **Temporal Considerations**: Time lag effects between policy implementation and transmission changes are not explicitly modeled in basic correlation/regression

## Limitations

1. Ecological fallacy: Country-level aggregates may not reflect individual-level relationships
2. Confounding variables: Many factors influence transmission beyond policy stringency
3. Temporal dynamics: Policies may have delayed effects not captured in concurrent correlations
4. Data quality: Reproduction rate estimates have inherent uncertainty
5. Missing data: Complete case analysis may introduce bias

## Future Extensions

- [ ] Add time-lagged correlation analysis
- [ ] Include confounding variables (vaccination rates, seasonality, variants)
- [ ] Implement panel data regression models
- [ ] Add country-specific stratified analyses
- [ ] Explore non-linear relationships
- [ ] Add causal inference methods (e.g., difference-in-differences)

## References

- Hale, T., Angrist, N., Goldszmidt, R. et al. (2021). A global panel database of pandemic policies (Oxford COVID-19 Government Response Tracker). *Nature Human Behaviour*, 5, 529-538.
- World Health Organization. (2020). *Considerations for implementing and adjusting public health and social measures in the context of COVID-19*.

## Author

Jon Long  
Public Health Portfolio Project

## License

This project is for educational and portfolio purposes. Data usage is subject to OWID terms and Oxford COVID-19 Government Response Tracker license.

---

*Last Updated: January 8, 2026*
