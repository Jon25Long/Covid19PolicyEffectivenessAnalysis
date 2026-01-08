# Policy Effectiveness Analysis

## Overview

This project examines the relationship between government policy stringency and COVID-19 transmission rates to answer a critical public health question: **Does government policy strictness actually reduce viral transmission?**

Using data from Our World in Data (OWID), this analysis explores the correlation between the Oxford Stringency Index and reproduction rates across countries during the COVID-19 pandemic.

## Research Question

**Primary Question**: Does government policy strictness actually reduce viral transmission?

**Independent Variable**: `stringency_index` (Oxford COVID-19 Government Response Tracker)  
**Dependent Variable**: `reproduction_rate` (viral transmission rate)  
**Analysis Type**: Time series correlation and regression analysis

## Information Theory Context

This analysis explores a **non-obvious relationship** where the outcome is not immediately predictable. According to information theory, information content is inversely proportional to the probability of an outcome: **I ∝ 1/P(outcome)**. Surprising results contain more information than expected ones.

The prevailing narrative suggested strict containment policies (lockdowns, stay-at-home orders, school closures) would substantially reduce transmission rates. This project tests whether empirical data supports this hypothesis.

## Project Structure

```
policy-effectiveness-analysis/
├── README.md                                    # This file
├── Data/                                        # Data files and scripts
│   ├── owid-covid-data.csv                     # OWID COVID-19 dataset
│   ├── owid-covid-data-cleaned.xlsx            # Cleaned data
│   ├── covid_correlation_regression_analysis.py # Python analysis script
│   ├── covid_correlation_regression_analysis.sas # SAS analysis script
│   ├── covid_data_cleaner.py                   # Data preprocessing
│   └── covid_analysis_results.xlsx             # Analysis outputs
├── notebooks/                                   # Analysis notebooks
│   └── policy_effectiveness_analysis-final.ipynb
└── .gitignore                                  # Git exclusions
```

## Data Source

**Dataset**: Our World in Data (OWID) COVID-19 Dataset  
**URL**: <https://github.com/owid/covid-19-data>

The OWID COVID-19 dataset provides comprehensive, daily-updated COVID-19 statistics including:

- Policy stringency measures (Oxford Government Response Tracker)
- Epidemiological indicators (cases, deaths, reproduction rates)
- Healthcare capacity metrics
- Demographic and economic variables

**Key Variables**:

- `date`: Observation date
- `location`: Country/region name
- `stringency_index`: Government response stringency (0-100 scale)
- `reproduction_rate`: Effective reproduction number (R)
- `new_cases_per_million`: Daily new cases per million population

## Analysis Methods

### Statistical Techniques

1. **Pearson Correlation**: Measures linear relationship between stringency and reproduction rate
2. **Spearman Correlation**: Assesses monotonic relationship (rank-based)
3. **Linear Regression**: Models reproduction rate as a function of stringency index
4. **Variance Analysis**: Examines variability in both variables

### Scripts and Tools

- **Python Analysis** ([covid_correlation_regression_analysis.py](Data/covid_correlation_regression_analysis.py))
  - Pandas for data manipulation
  - SciPy for statistical tests
  - Scikit-learn for regression modeling
  - Excel export with formatted results

- **Jupyter Notebook** ([policy_effectiveness_analysis-final.ipynb](notebooks/policy_effectiveness_analysis-final.ipynb))
  - Interactive exploratory data analysis
  - Visualizations (scatter plots, distributions)
  - Quality assessment and missing value analysis
  - Comprehensive statistical interpretation

- **SAS Analysis** ([covid_correlation_regression_analysis.sas](Data/covid_correlation_regression_analysis.sas))
  - PROC CORR for correlation analysis
  - PROC REG for regression modeling

## Key Findings

The analysis evaluates:

- **Correlation strength**: Is there a significant relationship between policy stringency and transmission?
- **Effect magnitude**: How much does a 1-unit increase in stringency affect reproduction rate?
- **Statistical significance**: Are observed relationships unlikely due to chance?
- **Practical significance**: Do the findings have real-world policy implications?

## Getting Started

### Prerequisites

```bash
# Python requirements
pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl jupyter
```

### Running the Analysis

**Option 1: Python Script**

```bash
cd Data
python3 covid_correlation_regression_analysis.py
```

**Option 2: Jupyter Notebook**

```bash
cd notebooks
jupyter lab policy_effectiveness_analysis-final.ipynb
```

**Option 3: SAS**

```sas
/* Update file paths in the script, then run */
%include 'covid_correlation_regression_analysis.sas';
```

## Results Output

The Python analysis exports results to `Data/covid_analysis_results.xlsx` containing:

1. **Descriptive Statistics**: Mean, standard deviation, variance for both variables
2. **Pearson Correlation**: Coefficient and p-value
3. **Spearman Correlation**: Coefficient and p-value
4. **Regression Results**: Coefficients, R², standard errors, confidence intervals
5. **Model Diagnostics**: Residuals, predictions, fit statistics

## Technical Notes

### Data Quality Considerations

- **Missing Values**: Only complete cases are used (rows with both stringency and reproduction rate)
- **Temporal Alignment**: Policy effects may have time lags not captured in simultaneous analysis
- **Geographic Variation**: Country-specific factors may moderate the relationship
- **Policy Heterogeneity**: Stringency index aggregates diverse policy types

### Reproducibility

All analysis scripts use relative paths where possible. If you encounter path errors in the Python script, update the `data_folder` variable in [covid_correlation_regression_analysis.py](Data/covid_correlation_regression_analysis.py):

```python
# Line 27 - Update to your absolute path
data_folder = "/path/to/policy-effectiveness-analysis/Data"
```

## Applications

This analysis framework can be adapted for:

- Policy impact evaluation in other public health domains
- Comparative effectiveness research
- Time-series intervention studies
- Correlation vs. causation case studies

## Related Projects

This project is part of the **Public Health Data Science Portfolio**:

- **COVID-19 Tracker**: 10 high-information analyses exploring pandemic patterns
- **Healthcare Data Science**: Reference examples for epidemiological analysis
- **Tools & Frameworks**: Reusable components for public health research

## License

**Data**: Our World in Data COVID-19 dataset is licensed under Creative Commons BY  
**Code**: Analysis scripts are original work, available for reuse with attribution

## References

1. Hale, T., et al. (2021). "A global panel database of pandemic policies (Oxford COVID-19 Government Response Tracker)." *Nature Human Behaviour*, 5(4), 529-538.
2. Mathieu, E., et al. (2021). "A global database of COVID-19 vaccinations." *Nature Human Behaviour*, 5(7), 947-953.
3. Our World in Data. (2023). "Coronavirus Pandemic (COVID-19)." Retrieved from <https://ourworldindata.org/coronavirus>

## Contact

This project is part of a public health data science portfolio demonstrating technical expertise in:

- Statistical analysis and hypothesis testing
- Time-series epidemiological modeling
- Multi-platform implementation (Python, SAS)
- Reproducible research practices

---

**Last Updated**: January 7, 2026  
**Project Type**: Public Health Policy Analysis  
**Status**: Complete
