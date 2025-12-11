# covid19-tracker
basic data analysis with jupyter notebook

# COVID-19 Data Analysis Project

## 📌 Project Description
A comprehensive analysis of global COVID-19 case data from Our World in Data. This project tracks infection trends, mortality rates, and vaccination progress across countries.

**Data Coverage**: January 2020 through August 2024 (2020-01-01 to 2024-08-04)

## 🎯 Objectives
1. Analyze global pandemic progression
2. Compare country-level responses
3. Visualize key metrics (cases, deaths, vaccinations)
4. Identify patterns in outbreak waves

## 🛠️ Tools Used
- Python 3
- Pandas (Data manipulation)
- Matplotlib/Seaborn (Visualization)
- Jupyter Notebook (Analysis environment)

## 📊 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/covid19-tracker.git
   ```
2. **Activate the covid19_tracker virtual environment** (required for all notebooks):
   ```bash
   conda activate covid19_tracker
   ```
   This environment includes all required dependencies (pandas, numpy, matplotlib, seaborn, scipy, statsmodels, IPython).

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook covid19_analysis.ipynb
   ```
   
   **Note**: For generated analysis notebooks in `notebooks/`, use the "COVID-19 Tracker (Python 3.11 AMD GPU)" kernel in VS Code or Jupyter.

## 💡 Key Insights
- Case fatality rates varied significantly by country
- Vaccination rollout showed strong correlation with case reduction
- Seasonal patterns emerged in northern/southern hemispheres

## 📝 Reflections
This project demonstrated how data science can provide actionable insights during global health crises. Future improvements could include real-time data pipelines and interactive dashboards.

---

## 🔬 High-Information Content Analyses (NEW)

**Date Added**: December 5, 2025  
**Location**: `notebooks/`  
**Documentation**: See [HIGH_INFORMATION_ANALYSES.md](HIGH_INFORMATION_ANALYSES.md) and [ANALYSES_README.md](ANALYSES_README.md)

### Motivation
Following information theory principles, we've created **10 focused notebooks** exploring non-obvious relationships where outcomes are NOT immediately predictable. These analyses have higher information content than obvious correlations (e.g., cases vs deaths).

### Available Analyses

1. **Policy Effectiveness** - Does stringency reduce transmission?
2. **Healthcare Capacity** - Do hospital beds save lives?
3. **Wealth-Mortality Paradox** - Do rich nations fare better?
4. **Age Structure** - How much do demographics matter?
5. **Comorbidity Impact** - Do chronic diseases predict outcomes?
6. **Testing Strategy** - What testing level captures reality?
7. **Vaccination Effectiveness** - Real-world vaccine impact
8. **Reporting Accuracy** - Which countries under-report deaths?
9. **Urbanization** - Do cities spread disease faster?
10. **Development Gap** - Does HDI protect populations?

### Quick Start
```bash
cd notebooks
jupyter lab policy_effectiveness_analysis.ipynb
```

Each notebook includes:
- Research question with information theory context
- Statistical analysis (correlation, regression, p-values)
- Visualizations and interpretation framework
- Policy implications

**See**: [ANALYSES_README.md](ANALYSES_README.md) for details.
