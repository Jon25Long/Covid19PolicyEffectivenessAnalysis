# COVID-19 Policy Effectiveness Analysis

**Analyzing the relationship between government policy stringency and viral transmission during the COVID-19 pandemic**

## 📋 Overview

This project investigates a critical public health question: **Does government policy stringency actually reduce viral transmission?** Using data from the Our World in Data (OWID) COVID-19 dataset, this analysis explores the relationship between the Oxford Stringency Index and disease reproduction rates.

### Research Question

Does the stringency of government COVID-19 policies correlate with reduced viral transmission rates?

### Key Findings

- **Unexpected Result**: Analysis revealed a weak **positive** correlation (r = 0.2945) between stringency and reproduction rate—contrary to conventional expectations
- **Low Explanatory Power**: Policy stringency explains only 8.67% of variance in reproduction rates
- **High Information Content**: The surprising result provides valuable insights about the complexity of pandemic response effectiveness

## 🎯 Motivation

Following **information theory principles**, this analysis focuses on a non-obvious relationship where outcomes are not immediately predictable. According to information theory:

$$I \propto \frac{1}{P(\text{outcome})}$$

Surprising results contain **more information** than expected ones. While analyzing "total cases vs deaths" has low information content (obvious correlation), examining "policy stringency vs transmission" provides genuine insights because the outcome challenges assumptions.

## 📊 Dataset

**Source**: [Our World in Data COVID-19 Database](https://github.com/owid/covid-19-data)

**Variables Used**:

- `stringency_index` - Oxford COVID-19 Government Response Tracker (0-100 scale)
- `reproduction_rate` - Effective reproduction number (R)

**Data Coverage**:

- Global dataset spanning multiple countries
- Time series data from pandemic onset through data collection date

## 🔬 Analysis Components

### 1. Jupyter Notebook Analysis

**File**: [`notebooks/policy_effectiveness_analysis-final.ipynb`](notebooks/policy_effectiveness_analysis-final.ipynb)

Comprehensive analysis including:

- Data quality assessment and missing value analysis
- Exploratory data analysis with visualizations
- Correlation analysis (Pearson and Spearman)
- Linear regression modeling
- Statistical significance testing
- Interpretation framework with causal considerations

### 2. Python Script: Correlation & Regression

**File**: [`Data/covid_correlation_regression_analysis.py`](Data/covid_correlation_regression_analysis.py)

Automated analysis pipeline that:

- Imports OWID COVID-19 data
- Removes missing values
- Performs Pearson and Spearman correlations
- Conducts linear regression analysis
- Calculates variance for both variables
- Exports results to Excel with formatted output

### 3. Data Cleaning Script

**File**: [`Data/covid_data_cleaner.py`](Data/covid_data_cleaner.py)

Preprocessing utility that:

- Imports raw COVID-19 dataset
- Identifies columns with >80% missing values
- Removes sparse columns and incomplete rows
- Exports cleaned data to Excel format

### 4. SAS Analysis

**File**: [`Data/covid_correlation_regression_analysis.sas`](Data/covid_correlation_regression_analysis.sas)

Parallel implementation in SAS for:

- Cross-verification of Python results
- Demonstrating multi-language statistical competency
- Enterprise analytics environment compatibility

## 📁 Project Structure

```
policy-effectiveness-analysis/
├── README.md                          # This file
├── .gitignore                         # Git exclusions
├── Data/
│   ├── owid-covid-data.csv           # Raw dataset
│   ├── owid-covid-data-cleaned.xlsx  # Cleaned dataset
│   ├── covid_analysis_results.xlsx   # Analysis outputs
│   ├── covid_data_cleaner.py         # Data preprocessing script
│   ├── covid_correlation_regression_analysis.py  # Python analysis
│   └── covid_correlation_regression_analysis.sas # SAS analysis
└── notebooks/
    └── policy_effectiveness_analysis-final.ipynb  # Main analysis notebook
```

## 🚀 Getting Started

### Prerequisites

**Python Environment**:

```bash
python >= 3.8
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
jupyter
openpyxl
```

**SAS** (optional): SAS 9.4 or later for `.sas` script execution

### Installation

1. **Clone the repository** (or navigate to project folder):

```bash
cd /path/to/PublicHealthPortfolio/my-projects/policy-effectiveness-analysis
```

1. **Install Python dependencies**:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter openpyxl
```

1. **Download the dataset**:

```bash
# Download from OWID GitHub
wget https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv -P Data/
```

Alternatively, visit [OWID COVID-19 Data Repository](https://github.com/owid/covid-19-data/tree/master/public/data) and download manually to the `Data/` folder.

### Usage

#### Option 1: Run Jupyter Notebook (Recommended)

```bash
cd notebooks
jupyter notebook policy_effectiveness_analysis-final.ipynb
```

Or use JupyterLab:

```bash
jupyter lab policy_effectiveness_analysis-final.ipynb
```

#### Option 2: Run Python Script

```bash
cd Data
python covid_correlation_regression_analysis.py
```

Output will be saved to `Data/covid_analysis_results.xlsx`

#### Option 3: Data Cleaning Only

```bash
cd Data
python covid_data_cleaner.py
```

#### Option 4: SAS Analysis

```bash
# In SAS environment
%include "/path/to/Data/covid_correlation_regression_analysis.sas";
```

## 📈 Key Results

### Statistical Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Pearson Correlation (r)** | 0.2945 | Weak positive correlation |
| **p-value** | < 0.001 | Statistically significant |
| **R² (Variance Explained)** | 8.67% | Low explanatory power |
| **Spearman Correlation (ρ)** | Similar | Confirms relationship |

### Critical Insights

1. **Reverse Causality**: Stricter policies may be **implemented in response** to rising transmission, not causing reductions
2. **Temporal Lag**: Policy effects require time to manifest—cross-sectional analysis cannot capture this
3. **Confounding Variables**: Unmeasured factors (variant emergence, healthcare capacity, population density) likely dominate
4. **Implementation Gap**: Stringency measures on paper ≠ actual behavioral compliance

### Information Theory Perspective

The **low variance explained** (8.67%) indicates that `stringency_index` provides **limited information** about `reproduction_rate` in cross-sectional analysis. This surprising result carries **high information content** precisely because it contradicts conventional assumptions and highlights analytical complexity.

## 🔍 Methodological Considerations

### Strengths

✅ Large sample size from global dataset  
✅ Rigorous statistical testing (Pearson, Spearman, regression)  
✅ Transparent documentation of unexpected findings  
✅ Information theory framework for result interpretation  
✅ Multiple analysis implementations (Python, SAS, Jupyter)

### Limitations

⚠️ **Cross-sectional design** - Cannot establish temporal causation  
⚠️ **Ecological fallacy** - Country-level data may not reflect individual behavior  
⚠️ **Omitted variable bias** - Many unmeasured confounders  
⚠️ **Policy compliance gap** - Stringency index measures policy, not adherence  
⚠️ **Variant heterogeneity** - Different viral variants across time/space

## 🔮 Future Research Directions

### Recommended Extensions

1. **Temporal Analysis**: Implement lagged models to examine policy effects over time
2. **Panel Data Methods**: Fixed-effects models to control for country-specific factors
3. **Causal Inference**: Difference-in-differences or regression discontinuity designs
4. **Mediator Analysis**: Examine pathways through mobility, testing, or compliance
5. **Heterogeneity Analysis**: Stratify by region, income level, or pandemic phase
6. **Alternative Metrics**: Explore individual policy components beyond composite index

### Advanced Techniques

- Interrupted time series analysis
- Synthetic control methods
- Instrumental variable regression
- Bayesian structural time series models

## 📚 References

### Data Source

- Mathieu, E., Ritchie, H., Rodés-Guirao, L., et al. (2020). *Our World in Data COVID-19 Dataset*. [https://github.com/owid/covid-19-data](https://github.com/owid/covid-19-data)

### Stringency Index

- Hale, T., Angrist, N., Goldszmidt, R., et al. (2021). *A global panel database of pandemic policies (Oxford COVID-19 Government Response Tracker)*. Nature Human Behaviour.

### Information Theory Framework

- Shannon, C. E. (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal.

## 🛠️ Technical Skills Demonstrated

- **Python**: pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn
- **Statistical Analysis**: Correlation, regression, hypothesis testing
- **SAS Programming**: PROC CORR, PROC REG, data manipulation
- **Jupyter Notebooks**: Interactive analysis and visualization
- **Data Cleaning**: Handling missing data, outlier detection
- **Reproducible Research**: Documented workflows, version control
- **Information Theory**: Applying theoretical frameworks to empirical research
- **Critical Thinking**: Interpreting unexpected results, identifying limitations

## 📄 License

This project is part of a personal portfolio. Data is sourced from Our World in Data (CC BY 4.0 license). Analysis code and documentation are available for educational and reference purposes.

## 👤 Author

**Jonathan Long**  
Public Health Data Analyst  
Portfolio: [Covid19PolicyEffectivenessAnalysis](https://github.com/jonlong)

## 📧 Contact

For questions or collaboration opportunities, please reach out via GitHub or portfolio contact information.

---

**Last Updated**: January 8, 2026  
**Project Status**: ✅ Complete  
**Analysis Type**: Cross-sectional correlation and regression analysis
