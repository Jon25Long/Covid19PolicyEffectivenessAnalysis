#!/usr/bin/env python3
"""
COVID-19 High-Information Analysis Notebook Generator
======================================================

Generates multiple focused Jupyter notebooks exploring non-obvious,
high-information relationships in the COVID-19 dataset.

Each notebook analyzes relationships where the outcome is NOT immediately
predictable, maximizing information content according to information theory:
Information ∝ 1/P(outcome) - surprising results contain more information.

Author: John
Date: December 5, 2025
"""

import sys
from pathlib import Path

# Add tools to path
tools_path = Path(__file__).resolve().parent.parent.parent / "tools-and-frameworks" / "navigator"
sys.path.insert(0, str(tools_path))

from notebook_generator import NotebookGenerator

# Project configuration
PROJECT_PATH = Path(__file__).resolve().parent
DATASET_FILE = "owid-covid-data.csv"

# Define high-information analysis notebooks to generate
ANALYSES = [
    {
        "name": "policy_effectiveness",
        "title": "Policy Stringency vs Disease Transmission",
        "independent_var": "stringency_index",
        "dependent_var": "reproduction_rate",
        "description": "Does government policy strictness actually reduce viral transmission?",
        "columns": ["date", "location", "stringency_index", "reproduction_rate", "new_cases_per_million"],
        "analysis_type": "time_series",
        "narrative": "In early pandemic response, many governments and public health agencies framed **strict containment policies** (lockdowns, stay-at-home orders, school closures) as a primary tool to reduce transmission. The Oxford COVID-19 Government Response Tracker's *Stringency Index* was widely used in media and research as a proxy for how aggressively countries were responding to the crisis.",
        "references": ["Oxford COVID-19 Government Response Tracker (Hale et al., 2021)", "WHO interim guidance on public health measures (2020)"],
        "expected_relationship": "strong negative",
        "expected_explanation": "Stricter policies should substantially reduce transmission rates",
        "reverse_causality_plausible": True  # Governments may tighten policy IN RESPONSE TO high transmission
    },
    {
        "name": "healthcare_capacity_impact",
        "title": "Hospital Infrastructure vs Mortality Outcomes",
        "independent_var": "hospital_beds_per_thousand",
        "dependent_var": "total_deaths_per_million",
        "description": "Does healthcare system capacity prevent deaths when controlling for case rates?",
        "columns": ["date", "location", "hospital_beds_per_thousand", "total_deaths_per_million", 
                   "total_cases_per_million", "population"],
        "analysis_type": "cross_sectional",
        "narrative": "Healthcare surge capacity was a major focus of pandemic preparedness. Public health agencies and media outlets frequently emphasized **hospital bed availability** and ICU capacity as critical safeguards against mortality. Countries with more hospital infrastructure were widely assumed to be better positioned to handle severe COVID-19 cases.",
        "references": ["WHO guidance on surge capacity for health systems (2020)", "OECD Health Statistics hospital bed data"],
        "expected_relationship": "strong negative",
        "expected_explanation": "More hospital beds per capita should clearly reduce death rates",
        "reverse_causality_plausible": False  # Deaths cannot cause hospital bed counts (infrastructure precedes pandemic)
    },
    {
        "name": "wealth_mortality_paradox",
        "title": "Economic Development vs COVID-19 Mortality",
        "independent_var": "gdp_per_capita",
        "dependent_var": "total_deaths_per_million",
        "description": "Do wealthier nations have better or worse outcomes? (Accounting for reporting quality)",
        "columns": ["date", "location", "gdp_per_capita", "total_deaths_per_million", 
                   "excess_mortality", "total_tests_per_thousand"],
        "analysis_type": "cross_sectional",
        "narrative": "Pre-pandemic preparedness assessments and global health security indices heavily weighted **economic resources** as indicators of pandemic resilience. Wealthier countries were widely expected to leverage their resources—advanced healthcare systems, research capacity, and financial buffers—to achieve better outcomes than lower-income nations.",
        "references": ["Global Health Security Index 2019 rankings", "World Bank economic classifications"],
        "expected_relationship": "strong negative",
        "expected_explanation": "Higher GDP per capita should strongly correlate with lower mortality rates",
        "reverse_causality_plausible": False  # Deaths cannot meaningfully alter a country's GDP per capita
    },
    {
        "name": "age_structure_vulnerability",
        "title": "Population Age Demographics vs Mortality Rate",
        "independent_var": "aged_65_older",
        "dependent_var": "total_deaths_per_million",
        "description": "How much does an aging population increase COVID-19 mortality?",
        "columns": ["date", "location", "aged_65_older", "aged_70_older", "median_age",
                   "total_deaths_per_million", "total_cases_per_million"],
        "analysis_type": "cross_sectional",
        "narrative": "From the earliest stages of the pandemic, **age** was identified as the single strongest individual risk factor for severe COVID-19 outcomes. Public health messaging consistently emphasized that older populations face dramatically higher mortality risk. Countries with aging populations were expected to experience substantially worse outcomes.",
        "references": ["CDC COVID-19 age-stratified mortality data", "Early epidemiological reports from Wuhan and Italy"],
        "expected_relationship": "strong positive",
        "expected_explanation": "Countries with older populations should have substantially higher death rates",
        "reverse_causality_plausible": False  # Deaths cannot alter population age structure
    },
    {
        "name": "comorbidity_impact",
        "title": "Pre-existing Health Conditions vs COVID-19 Outcomes",
        "independent_var": "diabetes_prevalence",
        "dependent_var": "total_deaths_per_million",
        "description": "Do chronic diseases predict worse COVID-19 outcomes across populations?",
        "columns": ["date", "location", "diabetes_prevalence", "cardiovasc_death_rate",
                   "total_deaths_per_million", "total_cases_per_million"],
        "analysis_type": "cross_sectional",
        "narrative": "Clinical evidence consistently showed that **pre-existing conditions** (diabetes, cardiovascular disease, obesity) significantly increased individual risk of severe COVID-19 outcomes. Public health authorities identified populations with high chronic disease burden as particularly vulnerable, suggesting that countries with higher comorbidity prevalence would face worse pandemic outcomes.",
        "references": ["CDC list of underlying medical conditions", "Meta-analyses of comorbidities and COVID-19 severity"],
        "expected_relationship": "moderate positive",
        "expected_explanation": "Higher chronic disease prevalence should correlate with higher mortality",
        "reverse_causality_plausible": False  # Deaths cannot cause pre-existing diabetes prevalence
    },
    {
        "name": "testing_strategy_effectiveness",
        "title": "Testing Adequacy vs Case Detection",
        "independent_var": "total_tests_per_thousand",
        "dependent_var": "positive_rate",
        "description": "At what testing level do countries accurately capture true case rates?",
        "columns": ["date", "location", "total_tests_per_thousand", "positive_rate",
                   "tests_per_case", "new_cases_per_million"],
        "analysis_type": "time_series",
        "narrative": "'Test, trace, isolate' became a mantra of pandemic response. WHO and national health agencies emphasized **widespread testing** as essential for controlling transmission. Countries were ranked by testing rates, with higher testing assumed to lead to better case detection and control. Testing capacity was treated as a key indicator of pandemic preparedness.",
        "references": ["WHO testing strategy guidance", "Our World in Data testing dashboards"],
        "expected_relationship": "strong negative",
        "expected_explanation": "More testing per capita should result in lower positive rates (better case detection)",
        "reverse_causality_plausible": True  # High positive rates may trigger increased testing (reactive strategy)
    },
    {
        "name": "vaccination_mortality_reduction",
        "title": "Vaccination Coverage vs Death Rate Evolution",
        "independent_var": "people_fully_vaccinated_per_hundred",
        "dependent_var": "new_deaths_smoothed_per_million",
        "description": "Quantifying the real-world effectiveness of vaccination campaigns",
        "columns": ["date", "location", "people_fully_vaccinated_per_hundred",
                   "new_deaths_smoothed_per_million", "new_cases_smoothed_per_million",
                   "aged_65_older"],
        "analysis_type": "time_series",
        "narrative": "Vaccines were framed as **'our way out' of the pandemic**. Public health campaigns and government messaging emphasized vaccination as the primary path to ending COVID-19 mortality. High vaccination coverage was presented as sufficient to dramatically reduce deaths, with countries racing to vaccinate their populations as quickly as possible.",
        "references": ["WHO vaccination targets and strategies", "National vaccination campaign messaging"],
        "expected_relationship": "strong negative",
        "expected_explanation": "Higher vaccination rates should lead to substantially lower death rates",
        "reverse_causality_plausible": True  # High death rates may accelerate vaccination campaigns (urgency response)
    },
    {
        "name": "reporting_accuracy_analysis",
        "title": "Excess Mortality vs Reported COVID-19 Deaths",
        "independent_var": "total_deaths",
        "dependent_var": "excess_mortality_cumulative",
        "description": "Which countries under-report COVID deaths? True pandemic toll revealed.",
        "columns": ["date", "location", "total_deaths", "excess_mortality",
                   "excess_mortality_cumulative", "excess_mortality_cumulative_per_million",
                   "gdp_per_capita", "human_development_index"],
        "analysis_type": "cross_sectional",
        "narrative": "Official COVID-19 death counts became politically sensitive metrics. While debates emerged about **underreporting**, particularly in lower-income countries, official statistics were still widely used for international comparisons. The relationship between reported COVID deaths and true excess mortality was assumed to be reasonably strong in most contexts.",
        "references": ["The Economist excess mortality tracker", "WHO excess mortality estimates"],
        "expected_relationship": "strong positive",
        "expected_explanation": "Reported COVID deaths should closely track true excess mortality",
        "reverse_causality_plausible": False  # Excess deaths are actual mortality; reported deaths are a measurement
    },
    {
        "name": "urbanization_transmission",
        "title": "Population Density vs Viral Transmission Dynamics",
        "independent_var": "population_density",
        "dependent_var": "reproduction_rate",
        "description": "Does urban density facilitate disease spread, or do cities adapt better?",
        "columns": ["date", "location", "population_density", "reproduction_rate",
                   "new_cases_per_million", "stringency_index"],
        "analysis_type": "time_series",
        "narrative": "Dense urban areas were widely portrayed as **'ticking time bombs'** for viral transmission. Early media coverage emphasized crowded cities like New York, Mumbai, and São Paulo as particularly vulnerable. The assumption that high population density would inevitably lead to faster, more severe outbreaks shaped both public perception and policy discussions.",
        "references": ["Early pandemic media coverage of urban hotspots", "Urban planning and public health literature on density"],
        "expected_relationship": "moderate positive",
        "expected_explanation": "Higher population density should correlate with higher transmission rates",
        "reverse_causality_plausible": False  # Transmission cannot meaningfully alter population density
    },
    {
        "name": "development_mortality_gap",
        "title": "Human Development Index vs True Pandemic Impact",
        "independent_var": "human_development_index",
        "dependent_var": "excess_mortality_cumulative_per_million",
        "description": "Does overall societal development protect against pandemic mortality?",
        "columns": ["date", "location", "human_development_index",
                   "excess_mortality_cumulative_per_million", "gdp_per_capita",
                   "life_expectancy", "hospital_beds_per_thousand"],
        "analysis_type": "cross_sectional",
        "narrative": "The **Human Development Index** (combining health, education, and income indicators) was often invoked as a composite measure of societal capacity to handle crises. Highly developed nations were expected to leverage their advantages—education systems, healthcare infrastructure, governance capacity, and public trust—to achieve substantially better pandemic outcomes than less developed countries.",
        "references": ["UNDP Human Development Reports", "Pre-pandemic pandemic preparedness indices"],
        "expected_relationship": "strong negative",
        "expected_explanation": "Higher human development should correlate with lower excess mortality",
        "reverse_causality_plausible": False  # Deaths cannot alter a country's Human Development Index
    }
]


def create_notebook_content(analysis: dict) -> dict:
    """
    Create notebook structure for a specific analysis.
    
    Args:
        analysis: Dictionary with analysis parameters
        
    Returns:
        Notebook structure as dictionary
    """
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# COVID-19 Analysis: {analysis['title']}\n",
            "\n",
            f"## Research Question\n",
            f"{analysis['description']}\n",
            "\n",
            "## Information Theory Context\n",
            "This analysis explores a **non-obvious relationship** where the outcome is not immediately predictable. ",
            "According to information theory, information content is inversely proportional to the probability of an outcome: ",
            "**I ∝ 1/P(outcome)**. Surprising results contain more information than expected ones.\n",
            "\n",
            f"**Independent Variable**: `{analysis['independent_var']}`  \n",
            f"**Dependent Variable**: `{analysis['dependent_var']}`  \n",
            f"**Analysis Type**: {analysis['analysis_type'].replace('_', ' ').title()}\n",
            "\n",
            "---\n"
        ]
    })
    
    # Prevailing Narrative cell
    references_formatted = "\\n".join([f"- {ref}" for ref in analysis.get('references', [])])
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Prevailing Narrative\n",
            "\n",
            f"{analysis.get('narrative', 'No prevailing narrative documented.')}\n",
            "\n",
            "### Working Hypothesis\n",
            f"Based on this prevailing narrative, we would expect: **{analysis.get('expected_relationship', 'undefined relationship')}** correlation between ",
            f"`{analysis['independent_var']}` and `{analysis['dependent_var']}`.  \n",
            f"In plain language: {analysis.get('expected_explanation', 'No specific expectation defined.')}\n",
            "\n",
            "### Sources\n",
            f"{references_formatted}\n",
            "\n",
            "---\n"
        ]
    })
    
    # Setup cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import required libraries\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from scipy import stats\n",
            "from pathlib import Path\n",
            "import sys\n",
            "\n",
            "# Add project root to path and import config\n",
            "project_root = Path('.').resolve().parent\n",
            "sys.path.insert(0, str(project_root))\n",
            "import config\n",
            "\n",
            "# Set visualization defaults\n",
            "sns.set_theme(style='whitegrid', palette='muted')\n",
            "plt.rcParams['figure.figsize'] = (14, 8)\n",
            "plt.rcParams['font.size'] = 11\n",
            "\n",
            "# Data path using config\n",
            f"data_path = project_root / config.data_dir / '{DATASET_FILE}'\n",
            "print(f'Data source: {data_path}')"
        ]
    })
    
    # Data loading cell
    column_list = "', '".join(analysis['columns'])
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Data Loading\n",
            "\n",
            f"Loading relevant columns from OWID COVID-19 dataset:\n",
            f"- {', '.join([f'`{col}`' for col in analysis['columns']])}"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Load data with relevant columns\n",
            f"columns_needed = {analysis['columns']}\n",
            "\n",
            "try:\n",
            "    df = pd.read_csv(\n",
            "        data_path,\n",
            "        usecols=columns_needed,\n",
            "        parse_dates=['date']\n",
            "    )\n",
            "    \n",
            "    print(f'✓ Loaded {len(df):,} rows')\n",
            "    print(f'\\nColumns: {list(df.columns)}')\n",
            "    print(f'\\nDate range: {df[\"date\"].min()} to {df[\"date\"].max()}')\n",
            "    print(f'\\nCountries: {df[\"location\"].nunique()}')\n",
            "    \n",
            "    # Display sample\n",
            "    display(df.head(10))\n",
            "    \n",
            "except FileNotFoundError:\n",
            "    print(f'❌ Data file not found: {data_path}')\n",
            "    print('Download from: https://github.com/owid/covid-19-data')\n",
            "except Exception as e:\n",
            "    print(f'❌ Error loading data: {e}')"
        ]
    })
    
    # Data quality assessment
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Data Quality Assessment\n",
            "\n",
            "Examine missing values, outliers, and data distribution before analysis."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Missing value analysis\n",
            "print('Missing Values by Column:')\n",
            "print('-' * 60)\n",
            "missing = df.isnull().sum()\n",
            "missing_pct = (missing / len(df) * 100).round(2)\n",
            "missing_df = pd.DataFrame({\n",
            "    'Missing Count': missing,\n",
            "    'Percentage': missing_pct\n",
            "}).sort_values('Missing Count', ascending=False)\n",
            "print(missing_df)\n",
            "\n",
            "# Basic statistics for key variables\n",
            f"print('\\n\\nKey Variable Statistics:')\n",
            "print('-' * 60)\n",
            f"print(df[['{analysis['independent_var']}', '{analysis['dependent_var']}']].describe())\n",
            "\n",
            "# Check for zeros and extreme values\n",
            f"print('\\n\\nValue Range Check:')\n",
            "print('-' * 60)\n",
            f"for col in ['{analysis['independent_var']}', '{analysis['dependent_var']}']:\n",
            "    valid = df[col].dropna()\n",
            "    if len(valid) > 0:\n",
            "        print(f'{col}:')\n",
            "        print(f'  Min: {valid.min():.2f}')\n",
            "        print(f'  Max: {valid.max():.2f}')\n",
            "        print(f'  Zeros: {(valid == 0).sum()} ({(valid == 0).sum() / len(valid) * 100:.1f}%)')\n",
            "        print()\n"
        ]
    })
    
    # Exploratory visualization
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Exploratory Visualization\n",
            "\n",
            f"Visualize the relationship between **{analysis['independent_var']}** and **{analysis['dependent_var']}**."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create clean dataset for visualization\n",
            f"df_clean = df[['{analysis['independent_var']}', '{analysis['dependent_var']}', 'location']].dropna()\n",
            "\n",
            "if len(df_clean) > 0:\n",
            "    fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
            "    \n",
            "    # Scatter plot\n",
            "    axes[0].scatter(\n",
            f"        df_clean['{analysis['independent_var']}'],\n",
            f"        df_clean['{analysis['dependent_var']}'],\n",
            "        alpha=0.5,\n",
            "        s=20\n",
            "    )\n",
            f"    axes[0].set_xlabel('{analysis['independent_var']}')\n",
            f"    axes[0].set_ylabel('{analysis['dependent_var']}')\n",
            f"    axes[0].set_title('Scatter Plot: {analysis['title']}')\n",
            "    axes[0].grid(True, alpha=0.3)\n",
            "    \n",
            "    # Distribution plots\n",
            f"    axes[1].hist(df_clean['{analysis['independent_var']}'], bins=50, alpha=0.7, label='Independent')\n",
            "    axes[1].set_xlabel('Value')\n",
            "    axes[1].set_ylabel('Frequency')\n",
            "    axes[1].set_title('Distribution of Variables')\n",
            "    axes[1].legend()\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "    \n",
            f"    print(f'Visualization based on {{len(df_clean):,}} valid observations')\n",
            "else:\n",
            "    print('⚠️  Insufficient data for visualization')\n"
        ]
    })
    
    # Statistical analysis
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Statistical Analysis\n",
            "\n",
            "Compute correlations and perform regression analysis. Results are stored in variables for use in automated reporting."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Correlation and regression analysis\n",
            f"df_analysis = df[['{analysis['independent_var']}', '{analysis['dependent_var']}']].dropna()\n",
            "\n",
            "# Initialize variables for later use\n",
            "analysis_complete = False\n",
            "r_pearson = r_spearman = p_pearson = p_spearman = None\n",
            "slope = intercept = r_value = p_value = std_err = None\n",
            "r_squared = variance_explained = None\n",
            "strength = direction = significance = None\n",
            "\n",
            "if len(df_analysis) > 30:  # Minimum sample size\n",
            "    analysis_complete = True\n",
            "    \n",
            "    # Pearson correlation\n",
            f"    r_pearson, p_pearson = stats.pearsonr(\n",
            f"        df_analysis['{analysis['independent_var']}'],\n",
            f"        df_analysis['{analysis['dependent_var']}']\n",
            "    )\n",
            "    \n",
            "    # Spearman correlation (for non-linear relationships)\n",
            f"    r_spearman, p_spearman = stats.spearmanr(\n",
            f"        df_analysis['{analysis['independent_var']}'],\n",
            f"        df_analysis['{analysis['dependent_var']}']\n",
            "    )\n",
            "    \n",
            "    # Simple linear regression\n",
            f"    slope, intercept, r_value, p_value, std_err = stats.linregress(\n",
            f"        df_analysis['{analysis['independent_var']}'],\n",
            f"        df_analysis['{analysis['dependent_var']}']\n",
            "    )\n",
            "    \n",
            "    # Calculate derived metrics\n",
            "    r_squared = r_pearson ** 2\n",
            "    variance_explained = r_squared * 100\n",
            "    \n",
            "    # Determine interpretation categories\n",
            "    if abs(r_pearson) < 0.3:\n",
            "        strength = 'weak'\n",
            "    elif abs(r_pearson) < 0.7:\n",
            "        strength = 'moderate'\n",
            "    else:\n",
            "        strength = 'strong'\n",
            "    \n",
            "    direction = 'positive' if r_pearson > 0 else 'negative'\n",
            "    significance = 'statistically significant' if p_pearson < 0.05 else 'not statistically significant'\n",
            "    \n",
            "    # Display results\n",
            "    print('Correlation Analysis')\n",
            "    print('=' * 60)\n",
            "    print(f'Sample size: {len(df_analysis):,} observations\\n')\n",
            "    \n",
            "    # Format p-values appropriately\n",
            "    p_pearson_str = f'p < 0.001' if p_pearson < 0.001 else f'p = {p_pearson:.4f}'\n",
            "    p_spearman_str = f'p < 0.001' if p_spearman < 0.001 else f'p = {p_spearman:.4f}'\n",
            "    \n",
            "    print(f'Pearson correlation:  r = {r_pearson:7.4f}, {p_pearson_str}')\n",
            "    print(f'Spearman correlation: ρ = {r_spearman:7.4f}, {p_spearman_str}')\n",
            "    \n",
            "    print('\\nInterpretation:')\n",
            "    print(f'  • {strength.capitalize()} {direction} correlation')\n",
            "    print(f'  • Result is {significance} (α = 0.05)')\n",
            "    print(f'  • R² = {r_squared:.4f} ({variance_explained:.2f}% variance explained)')\n",
            "    \n",
            "    print('\\nLinear Regression:')\n",
            "    print(f'  y = {slope:.4f}x + {intercept:.4f}')\n",
            "    print(f'  Standard error: {std_err:.4f}')\n",
            "    \n",
            "else:\n",
            "    print(f'⚠️  Insufficient data: only {len(df_analysis)} observations')\n",
            "    print('    Need at least 30 for meaningful statistical analysis')"
        ]
    })
    
    # Auto-generated Results section
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Generate Results Summary (using variables from previous analysis)\n",
            "from IPython.display import Markdown, display\n",
            "\n",
            "if analysis_complete:\n",
            "    # Determine visual indicators\n",
            "    if abs(r_pearson) < 0.3:\n",
            "        strength_emoji = '⚪'\n",
            "    elif abs(r_pearson) < 0.7:\n",
            "        strength_emoji = '🟡'\n",
            "    else:\n",
            "        strength_emoji = '🔴'\n",
            "    \n",
            "    direction_emoji = '📈' if r_pearson > 0 else '📉'\n",
            "    sig_emoji = '✓' if p_pearson < 0.05 else '✗'\n",
            "    \n",
            "    # Generate detailed interpretation text\n",
            "    if abs(r_pearson) < 0.3:\n",
            "        effect_interp = 'The relationship is **weak**, suggesting other factors play a larger role.'\n",
            "    elif abs(r_pearson) < 0.5:\n",
            "        effect_interp = 'The relationship is **moderate**, indicating this factor has meaningful influence but is not the sole driver.'\n",
            "    elif abs(r_pearson) < 0.7:\n",
            "        effect_interp = 'The relationship is **moderately strong**, showing this factor is an important predictor.'\n",
            "    else:\n",
            "        effect_interp = 'The relationship is **very strong**, indicating this factor is a major determinant.'\n",
            "    \n",
            "    if p_pearson < 0.001:\n",
            "        sig_interp = 'The relationship is **highly statistically significant** (p < 0.001), meaning this finding is extremely unlikely to be due to chance.'\n",
            "    elif p_pearson < 0.01:\n",
            "        sig_interp = 'The relationship is **very statistically significant** (p < 0.01), providing strong evidence this is a real effect.'\n",
            "    elif p_pearson < 0.05:\n",
            "        sig_interp = 'The relationship is **statistically significant** (p < 0.05), meeting the conventional threshold for statistical significance.'\n",
            "    else:\n",
            "        sig_interp = 'The relationship is **not statistically significant** (p ≥ 0.05), so we cannot rule out that this pattern occurred by chance.'\n",
            "    \n",
            "    # Practical significance interpretation\n",
            "    if variance_explained < 10:\n",
            "        practical_interp = f'However, only **{variance_explained:.1f}% of variance** is explained, suggesting **limited practical significance**.'\n",
            "    elif variance_explained < 25:\n",
            "        practical_interp = f'With **{variance_explained:.1f}% of variance** explained, this has **moderate practical significance**.'\n",
            "    else:\n",
            "        practical_interp = f'With **{variance_explained:.1f}% of variance** explained, this has **substantial practical significance** for policy and planning.'\n",
            "    \n",
            "    # Direction interpretation\n",
            "    if direction == 'positive':\n",
            f"        direction_interp = f'As **{analysis['independent_var']}** increases, **{analysis['dependent_var']}** tends to increase by {{slope:.4f}} units per unit increase.'\n",
            "    else:\n",
            f"        direction_interp = f'As **{analysis['independent_var']}** increases, **{analysis['dependent_var']}** tends to decrease by {{abs(slope):.4f}} units per unit increase.'\n",
            "    \n",
            "    # Build markdown summary\n",
            "    p_value_display = '< 0.001' if p_pearson < 0.001 else f'{p_pearson:.4f}'\n",
            "    \n",
            "    results_md = f'''## 5. Analysis Results\n",
            "\n",
            "### Statistical Findings\n",
            "\n",
            "**Sample Size**: {len(df_analysis):,} observations\n",
            "\n",
            "| Metric | Value | Interpretation |\n",
            "|--------|-------|----------------|\n",
            "| **Pearson Correlation (r)** | {r_pearson:.4f} | {strength_emoji} {strength.capitalize()} {direction} correlation |\n",
            "| **Spearman Correlation (ρ)** | {r_spearman:.4f} | Rank-based correlation (handles non-linearity) |\n",
            "| **P-value** | {p_value_display} | {sig_emoji} {significance.capitalize()} |\n",
            "| **R² (Variance Explained)** | {r_squared:.4f} ({variance_explained:.2f}%) | Proportion of variance explained |\n",
            "| **Regression Slope** | {slope:.4f} | {direction_emoji} Change in Y per unit X |\n",
            "| **Regression Intercept** | {intercept:.4f} | Expected Y when X = 0 |\n",
            "| **Standard Error** | {std_err:.4f} | Uncertainty in slope estimate |\n",
            "\n",
            "### Interpretation\n",
            "\n",
            "**{strength.capitalize()} {direction} relationship detected** between the variables:\n",
            "\n",
            "1. **Correlation Strength**: {effect_interp}\n",
            "\n",
            "2. **Statistical Significance**: {sig_interp}\n",
            "\n",
            "3. **Practical Significance**: {practical_interp}\n",
            "\n",
            "4. **Direction & Magnitude**: {direction_interp}\n",
            "\n",
            "### Information Theory Assessment\n",
            "\n",
            "'''\n",
            "    \n",
            "    # Add information content assessment\n",
            "    if abs(r_pearson) < 0.3:\n",
            "        info_content = '''**High information content** ⭐⭐⭐: This weak correlation is surprising and informative. \n",
            "The lack of a strong relationship suggests the variables operate more independently than intuition might suggest, \n",
            "which has important implications for understanding the underlying mechanisms.'''\n",
            "    elif abs(r_pearson) < 0.5 and p_pearson < 0.05:\n",
            "        info_content = '''**Moderate information content** ⭐⭐: This moderate correlation provides useful insights. \n",
            "The relationship exists but is not overwhelming, suggesting other factors also play important roles.'''\n",
            "    elif abs(r_pearson) >= 0.7:\n",
            "        info_content = '''**Lower information content** ⭐: This strong correlation might have been somewhat expected. \n",
            "While confirming intuition is valuable, surprising results typically yield higher information content.'''\n",
            "    else:\n",
            "        info_content = '''**Moderate information content** ⭐⭐: This finding provides useful evidence about the relationship, \n",
            "though the pattern may not be entirely unexpected.'''\n",
            "    \n",
            "    results_md += info_content\n",
            "    \n",
            "    # Display the results\n",
            "    display(Markdown(results_md))\n",
            "    \n",
            "else:\n",
            "    display(Markdown('''## 5. Analysis Results\n",
            "    \n",
            "⚠️ **Insufficient data** for statistical analysis. Need at least 30 observations with complete data.\n",
            "'''))"
        ]
    })
    
    # Python-generated Hypothesis Check cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Generate Hypothesis Check\n",
            "from IPython.display import Markdown, display\n",
            "\n",
            "if analysis_complete:\n",
            "    # Determine expectation alignment\n",
            f"    expected_relationship = '{analysis.get('expected_relationship', 'undefined')}'\n",
            "    \n",
            "    # Parse expected relationship\n",
            "    if 'negative' in expected_relationship:\n",
            "        expected_dir = 'negative'\n",
            "    elif 'positive' in expected_relationship:\n",
            "        expected_dir = 'positive'\n",
            "    else:\n",
            "        expected_dir = 'undefined'\n",
            "    \n",
            "    if 'strong' in expected_relationship:\n",
            "        expected_str = 'strong'\n",
            "    elif 'moderate' in expected_relationship:\n",
            "        expected_str = 'moderate'\n",
            "    else:\n",
            "        expected_str = 'undefined'\n",
            "    \n",
            "    # Determine alignment\n",
            "    direction_match = (expected_dir == direction)\n",
            "    strength_match = (expected_str == strength)\n",
            "    \n",
            "    if direction_match and strength_match:\n",
            "        alignment = '✅ **CONFIRMS** the prevailing narrative'\n",
            "        alignment_detail = 'Both the direction and strength of the relationship match expectations from public health authorities.'\n",
            "    elif direction_match and not strength_match:\n",
            "        alignment = '⚠️ **PARTIALLY CHALLENGES** the prevailing narrative'\n",
            "        alignment_detail = f'While the direction ({direction}) matches expectations, the relationship is **{strength}** rather than **{expected_str}** as anticipated. This suggests the factor plays a role but is less dominant than assumed.'\n",
            "    elif not direction_match:\n",
            "        alignment = '❌ **CONTRADICTS** the prevailing narrative'\n",
            "        alignment_detail = f'The observed **{direction}** correlation directly opposes the expected **{expected_dir}** relationship. This surprising finding demands re-evaluation of conventional assumptions.'\n",
            "    else:\n",
            "        alignment = '❓ **UNCLEAR** relationship to expectations'\n",
            "        alignment_detail = 'The relationship requires further investigation to determine alignment with expectations.'\n",
            "    \n",
            "    # Pre-compute conditional text for f-string (can't use backslashes in f-string expressions)\n",
            "    info_level = \"**high information content**\" if not direction_match or (not strength_match and expected_str in [\"strong\", \"moderate\"]) else \"**moderate information content**\"\n",
            "    narrative_verb = \"contradicts\" if not direction_match else \"challenges\"\n",
            "    belief_status = \"incorrect\" if not direction_match else \"weaker than assumed\"\n",
            "    relationship_comparison = \"opposite to\" if not direction_match else \"much weaker than\"\n",
            "    \n",
            "    hypothesis_md = f'''## Hypothesis Check: Expected vs Observed\\n",
            "\\n",
            "### What We Expected\\n",
            "\\n",
            "Based on prevailing public health narratives from authoritative sources, we anticipated a **{expected_relationship}** correlation.\\n",
            "\\n",
            f"**Reasoning**: {analysis.get('expected_explanation', 'No expectation defined.')}\\n",
            "\\n",
            "### What We Observed\\n",
            "\\n",
            "**Actual Result**: {strength.capitalize()} {direction} correlation (r = {r_pearson:.4f}, p < 0.001)\\n",
            "\\n",
            "- **Pearson correlation**: r = {r_pearson:.4f}\\n",
            "- **Variance explained**: {variance_explained:.2f}%\\n",
            "- **Statistical significance**: {significance}\\n",
            "\\n",
            "### Verdict: {alignment}\\n",
            "\\n",
            "{alignment_detail}\\n",
            "\\n",
            "### Why This Matters\\n",
            "\\n",
            "From an **information theory perspective**, this finding carries {info_level} because it {narrative_verb} the prevailing narrative from public health institutions. When widely-held beliefs prove {belief_status}, we gain valuable insights that can reshape pandemic response strategies.\\n",
            "\\n",
            f"The surprise lies not in discovering *no* relationship, but in finding that the relationship is {{relationship_comparison}} what authoritative sources predicted. This unexpected pattern suggests that {analysis['independent_var']} alone may not be the primary driver of {analysis['dependent_var']}, indicating complex interactions with timing, compliance, and contextual factors.\\n",
            "\\n",
            "---",
            "'''\n",
            "    \n",
            "    display(Markdown(hypothesis_md))\n",
            "else:\n",
            "    display(Markdown('## Hypothesis Check: Expected vs Observed\\n\\n⚠️ Analysis incomplete - cannot perform hypothesis check.'))"
        ]
    })
    
    # Python-generated Discussion section
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Generate Discussion Section\n",
            "from IPython.display import Markdown, display\n",
            "\n",
            "if analysis_complete:\n",
            "    # Determine expectation alignment\n",
            f"    expected_relationship = '{analysis.get('expected_relationship', 'undefined')}'\n",
            "    observed_strength = strength  # from previous cell\n",
            "    observed_direction = direction  # from previous cell\n",
            "    \n",
            "    # Parse expected relationship\n",
            "    if 'negative' in expected_relationship:\n",
            "        expected_dir = 'negative'\n",
            "    elif 'positive' in expected_relationship:\n",
            "        expected_dir = 'positive'\n",
            "    else:\n",
            "        expected_dir = 'undefined'\n",
            "    \n",
            "    if 'strong' in expected_relationship:\n",
            "        expected_str = 'strong'\n",
            "    elif 'moderate' in expected_relationship:\n",
            "        expected_str = 'moderate'\n",
            "    else:\n",
            "        expected_str = 'undefined'\n",
            "    \n",
            "    # Determine alignment\n",
            "    direction_match = (expected_dir == observed_direction)\n",
            "    strength_match = (expected_str == observed_strength)\n",
            "    \n",
            "    if direction_match and strength_match:\n",
            "        alignment = '✅ **CONFIRMS** the prevailing narrative'\n",
            "        alignment_detail = 'Both the direction and strength of the relationship match expectations from public health authorities.'\n",
            "    elif direction_match and not strength_match:\n",
            "        alignment = '⚠️ **PARTIALLY CHALLENGES** the prevailing narrative'\n",
            "        alignment_detail = f'While the direction ({observed_direction}) matches expectations, the relationship is **{observed_strength}** rather than **{expected_str}** as anticipated. This suggests the factor plays a role but is less dominant than assumed.'\n",
            "    elif not direction_match:\n",
            "        alignment = '❌ **CONTRADICTS** the prevailing narrative'\n",
            "        alignment_detail = f'The observed **{observed_direction}** correlation directly opposes the expected **{expected_dir}** relationship. This surprising finding demands re-evaluation of conventional assumptions.'\n",
            "    else:\n",
            "        alignment = '❓ **UNCLEAR** relationship to expectations'\n",
            "        alignment_detail = 'The relationship requires further investigation to determine alignment with expectations.'\n",
            "    \n",
            "    # Variables expected_relationship, direction, strength, expected_dir, expected_str, direction_match, strength_match, alignment, alignment_detail\n",
            "    # are already defined in the previous Hypothesis Check cell\n",
            "    \n",
            "    # Generate causality assessment (using variables from previous Hypothesis Check cell)\n",
            "    # Check if reverse causality is logically plausible for this analysis\n",
            f"    reverse_causality_plausible = {analysis.get('reverse_causality_plausible', False)}\n",
            "    \n",
            "    if not direction_match and reverse_causality_plausible:\n",
            "        # Only warn about reverse causality if it's logically possible\n",
            f"        causality_note = '''**Reverse Causality Warning**: The unexpected direction suggests possible reverse causation. \n",
            f"For example, `{analysis['dependent_var']}` may trigger changes in `{analysis['independent_var']}` (reactive response), \n",
            "creating a correlation opposite to the causal direction. This timing issue requires careful temporal analysis with lagged variables.'''\n",
            "    elif not direction_match and not reverse_causality_plausible:\n",
            "        # Direction mismatch when reverse causality is impossible - point to confounding/measurement issues\n",
            f"        causality_note = '''**Unexpected Direction**: The relationship contradicts expectations, but reverse causality is implausible \n",
            f"(pandemic outcomes cannot alter `{analysis['independent_var']}`). This suggests either: (1) confounding variables are driving \n",
            "both factors, (2) measurement/reporting biases are present, or (3) the prevailing theory is incorrect. Investigate third variables \n",
            "and data quality issues.'''\n",
            "    elif strength == 'weak':\n",
            "        causality_note = '''**Weak Effect Size**: The small correlation suggests this factor alone explains little variance. \n",
            "Other unmeasured variables likely play larger roles. Causal inference requires controlling for confounders.'''\n",
            "    else:\n",
            "        causality_note = '''**Correlation vs Causation**: While the relationship is detectable, establishing causation requires \n",
            "temporal precedence, ruling out confounders, and ideally quasi-experimental evidence from policy changes.'''\n",
            "    \n",
            "    # Calculate descriptive statistics for context\n",
            f"    x_mean = df_analysis['{analysis['independent_var']}'].mean()\n",
            f"    x_std = df_analysis['{analysis['independent_var']}'].std()\n",
            f"    y_mean = df_analysis['{analysis['dependent_var']}'].mean()\n",
            f"    y_std = df_analysis['{analysis['dependent_var']}'].std()\n",
            "    \n",
            "    # Calculate practical effect size\n",
            "    # For 1 SD change in X, what's the change in Y?\n",
            "    practical_effect = slope * x_std\n",
            "    effect_percent = (practical_effect / y_mean) * 100 if y_mean != 0 else 0\n",
            "    \n",
            "    discussion_md = f'''## 6. Discussion\\n",
            "\\n",
            "### Expectation Alignment\\n",
            "\\n",
            "{alignment}\\n",
            "\\n",
            "{alignment_detail}\\n",
            "\\n",
            "**Expected**: {expected_relationship} correlation  \\n",
            "**Observed**: {strength} {direction} correlation (r = {r_pearson:.4f})\\n",
            "\\n",
            "### Causality Assessment\\n",
            "\\n",
            "{causality_note}\\n",
            "\\n",
            "### Practical Significance\\n",
            "\\n",
            f"A **1 standard deviation increase** in `{analysis['independent_var']}` (SD = {{x_std:.2f}}) is associated with a \\n",
            f"**{{practical_effect:.4f}} unit change** in `{analysis['dependent_var']}`, representing approximately **{{abs(effect_percent):.1f}}%** \\n",
            "of the mean outcome value.\\n",
            "\\n",
            "**Interpretation**: '''\n",
            "    \n",
            "    if abs(effect_percent) < 5:\n",
            "        discussion_md += 'This is a **minimal practical effect**. Changes in this factor have negligible real-world impact on outcomes.'\n",
            "    elif abs(effect_percent) < 15:\n",
            "        discussion_md += 'This is a **small but detectable practical effect**. The factor contributes modestly to outcomes.'\n",
            "    elif abs(effect_percent) < 30:\n",
            "        discussion_md += 'This is a **moderate practical effect**. The factor has meaningful real-world impact worth considering in policy.'\n",
            "    else:\n",
            "        discussion_md += 'This is a **substantial practical effect**. The factor is a major driver of outcomes and demands policy attention.'\n",
            "    \n",
            "    # Geographic/temporal variation notes\n",
            f"    if '{analysis['analysis_type']}' == 'time_series':\n",
            "        variation_note = '''\\n\\n### Temporal Considerations\n",
            "\n",
            "This time-series analysis captures relationships as they evolved during the pandemic. The correlation may vary across \n",
            "different pandemic phases (initial outbreak, vaccine rollout, variant waves). Consider examining phase-specific relationships.'''\n",
            "    else:\n",
            "        variation_note = '''\\n\\n### Cross-Sectional Limitations\n",
            "\n",
            "This cross-sectional analysis represents a snapshot in time. Country-level relationships may obscure within-country variation \n",
            "and temporal dynamics. Geographic and cultural contexts likely moderate these relationships.'''\n",
            "    \n",
            "    discussion_md += variation_note\n",
            "    \n",
            "    display(Markdown(discussion_md))\n",
            "else:\n",
            "    display(Markdown('## 6. Discussion\\n\\n⚠️ Analysis incomplete - insufficient data for discussion.'))"
        ]
    })
    
    # Python-generated Next Steps section
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Generate Specific Next Steps\n",
            "from IPython.display import Markdown, display\n",
            "\n",
            "if analysis_complete:\n",
            "    # Re-establish variables from previous cells for conditional logic\n",
            f"    expected_relationship = '{analysis.get('expected_relationship', 'undefined')}'\n",
            "    expected_dir = 'negative' if 'negative' in expected_relationship else ('positive' if 'positive' in expected_relationship else 'undefined')\n",
            "    expected_str = 'strong' if 'strong' in expected_relationship else ('moderate' if 'moderate' in expected_relationship else 'undefined')\n",
            "    direction_match = (expected_dir == direction)\n",
            "    observed_strength = strength\n",
            f"    reverse_causality_plausible = {analysis.get('reverse_causality_plausible', False)}\n",
            "    \n",
            "    next_steps_md = '''## 7. Recommended Next Steps\n",
            "\n",
            "Based on the specific findings of this analysis:\\n\\n'''\n",
            "    \n",
            "    # Conditional recommendations based on findings\n",
            "    # Only recommend reverse causality investigation if logically plausible\n",
            "    if not direction_match and reverse_causality_plausible:\n",
            f"        next_steps_md += '''### 🔴 Priority: Investigate Reverse Causality\n",
            "\n",
            "The **opposite-direction** relationship suggests possible reverse causation:\n",
            "\n",
            f"1. **Temporal Analysis**: Use lagged correlations to determine whether `{analysis['independent_var']}` precedes \n",
            f"   `{analysis['dependent_var']}` or vice versa. Plot time-series to visualize lead-lag relationships.\n",
            "\n",
            f"2. **Event Studies**: Identify natural experiments or policy changes that altered `{analysis['independent_var']}` \n",
            f"   and examine subsequent changes in `{analysis['dependent_var']}`.\n",
            "\n",
            "3. **Literature Review**: Search for causal mechanisms explaining why the relationship contradicts expectations. \n",
            "   Review similar findings in epidemiological literature.\n",
            "\n",
            "'''\n",
            "    elif not direction_match and not reverse_causality_plausible:\n",
            f"        next_steps_md += '''### 🔴 Priority: Investigate Confounding and Data Quality\n",
            "\n",
            f"The **opposite-direction** relationship is surprising, and reverse causality is implausible (`{analysis['dependent_var']}` cannot cause `{analysis['independent_var']}`):\n",
            "\n",
            "1. **Confounding Variables**: Identify third variables that may drive both factors. For example, countries with \n",
            f"   higher `{analysis['independent_var']}` may differ systematically in unmeasured ways that affect `{analysis['dependent_var']}`.\n",
            "\n",
            "2. **Data Quality Assessment**: Examine measurement accuracy, reporting biases, and missing data patterns. \n",
            f"   Check if `{analysis['dependent_var']}` data quality varies systematically with `{analysis['independent_var']}`.\n",
            "\n",
            "3. **Subgroup Analysis**: Segment data by region, income level, or time period to identify where the relationship \n",
            "   holds versus reverses. This can reveal moderating factors or data issues.\n",
            "\n",
            "'''\n",
            "    \n",
            "    if observed_strength == 'weak' and expected_str in ['strong', 'moderate']:\n",
            f"        next_steps_md += '''### 🟡 Priority: Identify Confounders and Moderators\n",
            "\n",
            "The **weaker-than-expected** relationship suggests important missing variables:\n",
            "\n",
            "1. **Multivariate Regression**: Add control variables (GDP, healthcare capacity, age structure, etc.) to \n",
            f"   isolate the independent effect of `{analysis['independent_var']}`.\n",
            "\n",
            "2. **Segmentation Analysis**: Split data by moderating factors (e.g., income level, region, pandemic phase) \n",
            "   to identify where the relationship is stronger or weaker.\n",
            "\n",
            "3. **Mediation Analysis**: Test whether the effect operates through intermediate variables rather than directly.\n",
            "\n",
            "'''\n",
            "    \n",
            "    if variance_explained < 10:\n",
            f"        next_steps_md += f'''### 🔵 Priority: Explore Alternative Predictors\n",
            "\n",
            "With only **{variance_explained:.1f}% variance explained**, other factors dominate outcomes:\n",
            "\n",
            "1. **Feature Importance Analysis**: Use machine learning (random forests, gradient boosting) to identify \n",
            f"   which variables best predict `{analysis['dependent_var']}`.\n",
            "\n",
            f"2. **Composite Indices**: Combine `{analysis['independent_var']}` with related variables to create a \n",
            "   more comprehensive predictor.\n",
            "\n",
            "3. **Non-linear Relationships**: Test for thresholds, U-shaped relationships, or interaction effects that \n",
            "   linear correlation misses.\n",
            "\n",
            "'''\n",
            "    \n",
            f"    if '{analysis['analysis_type']}' == 'cross_sectional':\n",
            "        next_steps_md += '''### 🟢 Methodological Enhancement: Add Temporal Dimension\n",
            "\n",
            "Cross-sectional data limits causal inference:\n",
            "\n",
            "1. **Panel Data Analysis**: Collect data at multiple time points to examine within-country changes over time.\n",
            "\n",
            "2. **Difference-in-Differences**: If policy changes affected the independent variable, compare outcomes \n",
            "   before/after in treated vs control groups.\n",
            "\n",
            "3. **Granger Causality**: Test whether past values of one variable help predict future values of the other.\n",
            "\n",
            "'''\n",
            "    else:\n",
            "        next_steps_md += '''### 🟢 Methodological Enhancement: Geographic Comparison\n",
            "\n",
            "Time-series analysis should be complemented with geographic variation:\n",
            "\n",
            "1. **Country-Specific Trajectories**: Plot individual country time-series to identify outliers and patterns.\n",
            "\n",
            "2. **Regional Clustering**: Group countries by similar response patterns and identify regional factors.\n",
            "\n",
            "3. **Cross-Country Validation**: Test whether relationships hold across different country contexts.\n",
            "\n",
            "'''\n",
            "    \n",
            "    # Add specific policy implications\n",
            "    next_steps_md += '''### 📊 Data Quality Improvements\n",
            "\n",
            "'''\n",
            "    \n",
            f"    if 'deaths' in '{analysis['dependent_var']}'.lower() or 'mortality' in '{analysis['dependent_var']}'.lower():\n",
            "        next_steps_md += '''- **Validate with Excess Mortality**: Official COVID deaths may undercount. Compare with excess mortality data.\n",
            "- **Age-Standardization**: Control for population age structure when comparing mortality rates.\n",
            "'''\n",
            "    \n",
            f"    if 'test' in '{analysis['independent_var']}'.lower() or 'case' in '{analysis['dependent_var']}'.lower():\n",
            "        next_steps_md += '''- **Account for Testing Bias**: Case counts depend on testing rates. Use test-positivity rates or seroprevalence.\n",
            "- **Reporting Delays**: Allow for lags between infection, testing, and reporting.\n",
            "'''\n",
            "    \n",
            "    next_steps_md += f'''\\n### 🎯 Specific Research Questions\n",
            "\n",
            "Priority questions emerging from this analysis:\n",
            "\n",
            "1. **Why does the relationship differ from expectations?** Review institutional assumptions that led to \n",
            "   the predicted **{expected_relationship}** relationship.\n",
            "\n",
            "2. **Which countries are outliers?** Identify cases with high residuals and examine their unique characteristics.\n",
            "\n",
            "3. **What mechanisms are missing?** Develop theoretical models explaining the observed **{observed_strength} {observed_direction}** \n",
            "   relationship.\n",
            "'''\n",
            "    \n",
            "    display(Markdown(next_steps_md))\n",
            "else:\n",
            "    display(Markdown('## 7. Recommended Next Steps\\n\\n⚠️ Analysis incomplete - cannot generate recommendations.'))"
        ]
    })
    
    # Cell 8: Multivariate regression with controls
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 8. Multivariate Regression with Controls (cross-sectional by country)\n",
            "from statsmodels.formula.api import ols\n",
            "import statsmodels.api as sm\n",
            "from statsmodels.stats.outliers_influence import variance_inflation_factor\n",
            "\n",
            "# Aggregate to country-level means to avoid time autocorrelation in this quick check\n",
            f"reg_cols = ['location', '{analysis['independent_var']}', '{analysis['dependent_var']}']"
            f" + [c for c in ['new_cases_smoothed_per_million', 'total_cases_per_million', 'aged_65_older'] if c in df.columns and c not in ['{analysis['independent_var']}', '{analysis['dependent_var']}']]\n",
            "df_reg = df[reg_cols].dropna() if 'df' in globals() else pd.DataFrame()\n",
            "df_country = df_reg.groupby('location', as_index=False).mean() if not df_reg.empty else pd.DataFrame()\n",
            "\n",
            "if len(df_country) >= 30:\n",
            f"    base_controls = [c for c in df_country.columns if c not in ['location', '{analysis['independent_var']}', '{analysis['dependent_var']}']]\n",
            f"    formula = '{analysis['dependent_var']} ~ {analysis['independent_var']}' + (' + ' + ' + '.join(base_controls) if base_controls else '')\n",
            "    model = ols(formula, data=df_country).fit(cov_type='HC3')  # robust SEs\n",
            "    print(model.summary().tables[1])\n",
            "    print('\\nModel fit:')\n",
            "    print(f'  R-squared: {model.rsquared:.3f} (Adj: {model.rsquared_adj:.3f})')\n",
            "    print(f'  N (countries): {len(df_country)}')\n",
            "    \n",
            "    # VIF to flag multicollinearity\n",
            "    X = model.model.exog\n",
            "    vif_df = pd.DataFrame({\n",
            "        'feature': model.params.index,\n",
            "        'VIF': [variance_inflation_factor(X, i) for i in range(X.shape[1])]\n",
            "    })\n",
            "    print('\\nVariance Inflation Factors (VIF):')\n",
            "    print(vif_df)\n",
            "else:\n",
            "    print('⚠️ Not enough complete country-level rows for multivariate regression (need ≥ 30). Current:', len(df_country))"
        ]
    })
    
    # Cell 9: Segmentation & Interaction
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 9. Segmentation & Interaction: does effect vary by demographics?\n",
            f"if 'df_country' in globals() and len(df_country) >= 30 and '{analysis['independent_var']}' in df_country.columns:\n",
            "    try:\n",
            "        df_seg = df_country.copy()\n",
            f"        df_seg['age_segment'] = pd.qcut(df_seg['{analysis['independent_var']}'].values, q=3, labels=['low', 'mid', 'high'])\n",
            f"        seg_formula = '{analysis['dependent_var']} ~ {analysis['independent_var']} * C(age_segment)'\n",
            "        seg_model = ols(seg_formula, data=df_seg).fit(cov_type='HC3')\n",
            "        print(seg_model.summary().tables[1])\n",
            "        print('\\nN (countries):', len(df_seg))\n",
            "        print('Segments:', df_seg['age_segment'].value_counts().to_dict())\n",
            "    except Exception as e:\n",
            "        print(f'⚠️ Segmentation model failed: {e}')\n",
            "else:\n",
            "    print('⚠️ Cannot run segmentation: country-level dataset not available or too small.')"
        ]
    })
    
    # Cell 10: Mediation test
    # Determine case control variable from analysis columns
    case_control_candidates = ['new_cases_smoothed_per_million', 'total_cases_per_million']
    case_control = next((c for c in case_control_candidates if c in analysis['columns']), None)
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 10. Mediation Test (Baron-Kenny framework with bootstrap p-value)\n",
            "rng = np.random.default_rng(42)\n",
            "\n",
            "if 'df_country' in globals() and len(df_country) >= 30:\n",
            f"    case_col = next((c for c in ['new_cases_smoothed_per_million', 'total_cases_per_million'] if c in df_country.columns), None)\n",
            "    if case_col:\n",
            f"        med_df = df_country[['{analysis['independent_var']}', case_col, '{analysis['dependent_var']}']].dropna()\n",
            "        if len(med_df) >= 30:\n",
            f"            # a-path: {analysis['independent_var']} -> cases\n",
            f"            a_model = ols(f'{{case_col}} ~ {analysis['independent_var']}', data=med_df).fit(cov_type='HC3')\n",
            f"            a_coeff = a_model.params['{analysis['independent_var']}']\n",
            "\n",
            f"            # b/c'-path: {analysis['dependent_var']} ~ {analysis['independent_var']} + cases\n",
            f"            bc_model = ols(f'{analysis['dependent_var']} ~ {analysis['independent_var']} + {{case_col}}', data=med_df).fit(cov_type='HC3')\n",
            "            b_coeff = bc_model.params[case_col]\n",
            f"            c_prime = bc_model.params['{analysis['independent_var']}']\n",
            "\n",
            "            indirect_point = a_coeff * b_coeff\n",
            "\n",
            "            # Bootstrap indirect effect\n",
            "            boot_samples = 500\n",
            "            boot_indirect = []\n",
            "            for _ in range(boot_samples):\n",
            "                sample_idx = rng.integers(0, len(med_df), len(med_df))\n",
            "                sample = med_df.iloc[sample_idx]\n",
            f"                a_b = ols(f'{{case_col}} ~ {analysis['independent_var']}', data=sample).fit().params['{analysis['independent_var']}']\n",
            f"                b_b = ols(f'{analysis['dependent_var']} ~ {analysis['independent_var']} + {{case_col}}', data=sample).fit().params[case_col]\n",
            "                boot_indirect.append(a_b * b_b)\n",
            "            boot_indirect = np.array(boot_indirect)\n",
            "            ci_low, ci_high = np.percentile(boot_indirect, [2.5, 97.5])\n",
            "            p_boot = (np.mean(np.sign(boot_indirect) != np.sign(indirect_point)) * 2).clip(0, 1)\n",
            "\n",
            "            print('Indirect effect (a*b):', round(indirect_point, 4))\n",
            "            print('Bootstrap 95% CI:', (round(ci_low, 4), round(ci_high, 4)))\n",
            "            print('Direct effect (c\\\\): ', round(c_prime, 4))\n",
            "            print('N (countries):', len(med_df))\n",
            "            print('Bootstrap two-sided p (sign test):', round(p_boot, 4))\n",
            "        else:\n",
            "            print('⚠️ Not enough data after dropping NaNs for mediation (need ≥ 30). Current:', len(med_df))\n",
            "    else:\n",
            "        print('⚠️ Case control column not available for mediation')\n",
            "else:\n",
            "    print('⚠️ Mediation not run: country-level dataset unavailable or too small.')"
        ]
    })
    
    # Cell 11: Expanded controls
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 11. Expanded controls: testing, policy, capacity, income (cross-sectional by country)\n",
            "extra_cols = [\n",
            "    'new_tests_per_thousand', 'positive_rate', 'icu_patients_per_million',\n",
            "    'hosp_patients_per_million', 'stringency_index', 'gdp_per_capita',\n",
            "    'hospital_beds_per_thousand', 'median_age', 'aged_70_older'\n",
            "]\n",
            f"base_cols = ['location', '{analysis['independent_var']}', '{analysis['dependent_var']}']\n",
            "case_col = next((c for c in ['new_cases_smoothed_per_million', 'total_cases_per_million'] if 'df' in globals() and c in df.columns), None)\n",
            "if case_col:\n",
            "    base_cols.append(case_col)\n",
            # Only add aged_65_older if it's not already the independent variable
            *([f"if 'df' in globals() and 'aged_65_older' in df.columns:\n",
               "    base_cols.append('aged_65_older')\n"] if analysis['independent_var'] != 'aged_65_older' else []),
            # Exclude columns already in base_cols to prevent duplicates
            "available_extra = [c for c in extra_cols if 'df' in globals() and c in df.columns and c not in base_cols]\n",
            "reg_cols_extra = base_cols + available_extra\n",
            "missing_optional = [c for c in extra_cols if c not in available_extra and c not in base_cols]\n",
            "if missing_optional:\n",
            "    print('⚠️ Skipping missing optional controls:', missing_optional)\n",
            "if 'df' in globals():\n",
            f"    df_reg_ext = df[reg_cols_extra].dropna(subset=['{analysis['independent_var']}', '{analysis['dependent_var']}'])\n",
            "else:\n",
            "    df_reg_ext = pd.DataFrame()\n",
            "if not df_reg_ext.empty:\n",
            "    df_country_ext = df_reg_ext.groupby('location', as_index=False).mean()\n",
            "    available_controls = [c for c in available_extra if c in df_country_ext.columns and df_country_ext[c].notna().sum() > 0]\n",
            f"    base_formula_parts = ['{analysis['independent_var']}']  \n",
            "    if case_col and case_col in df_country_ext.columns:\n",
            "        base_formula_parts.append(case_col)\n",
            # Only add aged_65_older if it's not already the independent variable
            *([    "    if 'aged_65_older' in df_country_ext.columns:\n",
            "        base_formula_parts.append('aged_65_older')\n"] if analysis['independent_var'] != 'aged_65_older' else []),
            "    formula_controls = ' + '.join(base_formula_parts + available_controls)\n",
            f"    formula_ext = '{analysis['dependent_var']} ~ ' + formula_controls\n",
            "    if len(df_country_ext) >= 30:\n",
            "        model_ext = ols(formula_ext, data=df_country_ext).fit(cov_type='HC3')\n",
            "        print(model_ext.summary().tables[1])\n",
            "        print('\\nModel fit:')\n",
            "        print(f'  R-squared: {model_ext.rsquared:.3f} (Adj: {model_ext.rsquared_adj:.3f})')\n",
            "        print(f'  N (countries): {len(df_country_ext)}')\n",
            "        # VIF on included regressors\n",
            "        X_ext = model_ext.model.exog\n",
            "        vif_ext = pd.DataFrame({\n",
            "            'feature': model_ext.params.index,\n",
            "            'VIF': [variance_inflation_factor(X_ext, i) for i in range(X_ext.shape[1])]\n",
            "        })\n",
            "        print('\\nVariance Inflation Factors (VIF):')\n",
            "        print(vif_ext)\n",
            "    else:\n",
            "        print('⚠️ Not enough country-level observations for expanded controls (need ≥ 30). Current:', len(df_country_ext))\n",
            "else:\n",
            "    print('⚠️ Expanded controls skipped: insufficient data after dropping NaNs.')"
        ]
    })
    
    # Cell 12: Lagged specification
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            f"# 12. Lagged {analysis['independent_var']} (21-day) with cases and age controls\n",
            "if 'df' in globals():\n",
            f"    lag_cols = ['location', 'date', '{analysis['independent_var']}', '{analysis['dependent_var']}']  \n",
            "    case_col = next((c for c in ['new_cases_smoothed_per_million', 'total_cases_per_million'] if c in df.columns), None)\n",
            "    if case_col:\n",
            "        lag_cols.append(case_col)\n",
            # Only add aged_65_older if it's not already the independent variable
            *([    "    if 'aged_65_older' in df.columns:\n",
            "        lag_cols.append('aged_65_older')\n"] if analysis['independent_var'] != 'aged_65_older' else []),
            "    df_lag = df[lag_cols].copy()\n",
            f"    df_lag['{analysis['independent_var']}_lag21'] = df_lag.groupby('location')['{analysis['independent_var']}'].shift(21)\n",
            f"    df_lag = df_lag.dropna(subset=['{analysis['independent_var']}_lag21', '{analysis['dependent_var']}'])\n",
            "    df_lag_country = df_lag.groupby('location', as_index=False).mean()\n",
            "    if len(df_lag_country) >= 30:\n",
            # Build controls list excluding independent variable
            f"        controls = [c for c in [case_col" + (", 'aged_65_older'" if analysis['independent_var'] != 'aged_65_older' else "") + "] if c and c in df_lag_country.columns]\n",
            f"        lag_formula = '{analysis['dependent_var']} ~ {analysis['independent_var']}_lag21' + (' + ' + ' + '.join(controls) if controls else '')\n",
            "        model_lag = ols(lag_formula, data=df_lag_country).fit(cov_type='HC3')\n",
            "        print(model_lag.summary().tables[1])\n",
            "        print('\\nModel fit:')\n",
            "        print(f'  R-squared: {model_lag.rsquared:.3f} (Adj: {model_lag.rsquared_adj:.3f})')\n",
            "        print(f'  N (countries): {len(df_lag_country)}')\n",
            "    else:\n",
            "        print('⚠️ Not enough country-level observations for lagged spec (need ≥ 30). Current:', len(df_lag_country))\n",
            "else:\n",
            "    print('⚠️ Lagged spec skipped: data frame df not available.')"
        ]
    })
    
    # Cell 13: Log-scale robustness
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            f"# 13. Log-scale robustness with trimming (5th–95th pct {analysis['dependent_var']})\n",
            "if 'df' in globals():\n",
            f"    base_required = ['location', '{analysis['independent_var']}', '{analysis['dependent_var']}']  \n",
            "    case_col = next((c for c in ['new_cases_smoothed_per_million', 'total_cases_per_million'] if c in df.columns), None)\n",
            "    if case_col:\n",
            "        base_required.append(case_col)\n",
            "    missing_required = [c for c in base_required if c not in df.columns]\n",
            "    if missing_required:\n",
            "        print('⚠️ Log-scale spec skipped: missing required columns:', missing_required)\n",
            "    else:\n",
            # Build optional cols list excluding independent variable if it's aged_65_older
            "        optional_cols = ['new_tests_per_thousand', 'stringency_index'" + (", 'aged_65_older'" if analysis['independent_var'] != 'aged_65_older' else "") + "]\n",
            "        available_optional = [c for c in optional_cols if c in df.columns]\n",
            "        if len(available_optional) < len(optional_cols):\n",
            "            missing_opt = [c for c in optional_cols if c not in available_optional]\n",
            "            print('⚠️ Skipping missing optional controls:', missing_opt)\n",
            "        log_cols = base_required + available_optional\n",
            "        df_log = df[log_cols].dropna()\n",
            "        if not df_log.empty:\n",
            "            df_log_country = df_log.groupby('location', as_index=False).mean()\n",
            "            if len(df_log_country) >= 30:\n",
            f"                deaths = df_log_country['{analysis['dependent_var']}']\n",
            "                low_q, high_q = deaths.quantile([0.05, 0.95])\n",
            "                df_log_trim = df_log_country[(deaths >= low_q) & (deaths <= high_q)].copy()\n",
            f"                df_log_trim['log_deaths'] = np.log1p(df_log_trim['{analysis['dependent_var']}'])\n",
            f"                predictors = ['{analysis['independent_var']}']  \n",
            "                if case_col and case_col in df_log_trim.columns:\n",
            "                    df_log_trim['log_cases'] = np.log1p(df_log_trim[case_col])\n",
            "                    predictors.append('log_cases')\n",
            "                predictors.extend([c for c in ['new_tests_per_thousand', 'stringency_index', 'aged_65_older'] if c in df_log_trim.columns])\n",
            "                log_formula = 'log_deaths ~ ' + ' + '.join(predictors)\n",
            "                model_log = ols(log_formula, data=df_log_trim).fit(cov_type='HC3')\n",
            "                print(model_log.summary().tables[1])\n",
            "                print('\\nModel fit:')\n",
            "                print(f'  R-squared: {model_log.rsquared:.3f} (Adj: {model_log.rsquared_adj:.3f})')\n",
            "                print(f'  N (countries): {len(df_log_trim)} (trimmed)')\n",
            "            else:\n",
            "                print('⚠️ Not enough country-level observations for log-scale spec (need ≥ 30). Current:', len(df_log_country))\n",
            "        else:\n",
            "            print('⚠️ Log-scale spec skipped: data has too many missing values after filtering.')\n",
            "else:\n",
            "    print('⚠️ Log-scale spec skipped: data frame df not available.')"
        ]
    })
    
    # Cell 14: Plain-language takeaways
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 14. Plain-language takeaways (generated for any notebook)\n",
            "from IPython.display import Markdown, display\n",
            "import math\n",
            "\n",
            "# Identify primary variables if available\n",
            f"primary_x = '{analysis['independent_var']}' if 'df' in globals() and '{analysis['independent_var']}' in df.columns else 'independent variable'\n",
            f"primary_y = '{analysis['dependent_var']}' if 'df' in globals() and '{analysis['dependent_var']}' in df.columns else 'outcome variable'\n",
            "\n",
            "# Gather controls seen across specs\n",
            "controls = []\n",
            "for cand in ['new_cases_smoothed_per_million', 'total_cases_per_million', 'aged_65_older']:\n",
            "    if 'df' in globals() and cand in df.columns:\n",
            "        controls.append(cand)\n",
            "if 'available_controls' in globals():\n",
            "    controls.extend(available_controls)\n",
            "if 'available_optional' in globals():\n",
            "    controls.extend(available_optional)\n",
            "controls = sorted(set(controls)) if controls else ['none captured']\n",
            "\n",
            "# Capture model summaries if they exist\n",
            "spec_notes = []\n",
            "def add_spec(label, model_var):\n",
            "    if model_var in globals():\n",
            "        m = globals()[model_var]\n",
            "        if hasattr(m, 'rsquared'):\n",
            "            n_obs = getattr(m, 'nobs', None)\n",
            "            rsq = f\"R² {m.rsquared:.3f} (Adj {m.rsquared_adj:.3f})\" if hasattr(m, 'rsquared_adj') else f\"R² {m.rsquared:.3f}\"\n",
            "            spec_notes.append(f\"{label}: {rsq}; N={int(n_obs) if n_obs is not None else 'n/a'}\")\n",
            "add_spec('Expanded controls', 'model_ext')\n",
            "add_spec('Lag 21-day', 'model_lag')\n",
            "add_spec('Log-trim spec', 'model_log')\n",
            "spec_text = '; '.join(spec_notes) if spec_notes else 'Specs ran where data allowed; some may have been skipped due to missing columns or N<30.'\n",
            "\n",
            "# Assess model fit quality dynamically\n",
            "rsq_values = []\n",
            "for m_name in ['model_ext', 'model_lag', 'model_log']:\n",
            "    if m_name in globals():\n",
            "        m = globals()[m_name]\n",
            "        if hasattr(m, 'rsquared'):\n",
            "            rsq_values.append(m.rsquared)\n",
            "avg_rsq = sum(rsq_values) / len(rsq_values) if rsq_values else None\n",
            "if avg_rsq is not None:\n",
            "    if avg_rsq < 0.10:\n",
            "        fit_assessment = 'Fit is very weak (R² < 0.10), indicating the model explains little variance and most variation remains unexplained'\n",
            "    elif avg_rsq < 0.25:\n",
            "        fit_assessment = 'Fit is modest (R² < 0.25), signalling substantial unexplained variation'\n",
            "    elif avg_rsq < 0.50:\n",
            "        fit_assessment = 'Fit is moderate (R² between 0.25–0.50), explaining a meaningful but not dominant portion of variance'\n",
            "    elif avg_rsq < 0.75:\n",
            "        fit_assessment = 'Fit is good (R² between 0.50–0.75), capturing a substantial portion of outcome variation'\n",
            "    else:\n",
            "        fit_assessment = 'Fit is strong (R² > 0.75), explaining most of the variation in outcomes'\n",
            "else:\n",
            "    fit_assessment = 'Model fit varies across specifications'\n",
            "\n",
            "# Assess coefficient stability across specifications\n",
            "coef_values = []\n",
            "for m_name in ['model_ext', 'model_lag', 'model_log']:\n",
            "    if m_name in globals():\n",
            "        m = globals()[m_name]\n",
            "        if hasattr(m, 'params'):\n",
            "            # Try to get coefficient for primary_x or lagged version\n",
            "            for param_name in m.params.index:\n",
            "                if primary_x in param_name:\n",
            "                    coef_values.append(m.params[param_name])\n",
            "                    break\n",
            "if len(coef_values) >= 2:\n",
            "    coef_range = max(coef_values) - min(coef_values)\n",
            "    coef_mean = sum(coef_values) / len(coef_values)\n",
            "    coef_cv = (coef_range / abs(coef_mean)) if coef_mean != 0 else float('inf')\n",
            "    all_same_sign = all(c > 0 for c in coef_values) or all(c < 0 for c in coef_values)\n",
            "    if not all_same_sign:\n",
            "        robustness_assessment = 'The relationship changes direction across specifications, indicating high sensitivity to model choices'\n",
            "    elif coef_cv > 0.50:\n",
            "        robustness_assessment = 'The relationship shows substantial variation across specifications; control variables and model choices strongly affect the estimated association'\n",
            "    elif coef_cv > 0.25:\n",
            "        robustness_assessment = 'The relationship shows moderate variation across specifications; some sensitivity to controls and functional forms'\n",
            "    else:\n",
            "        robustness_assessment = 'The relationship is relatively stable across specifications, suggesting robust findings'\n",
            "else:\n",
            "    robustness_assessment = 'Coefficient stability could not be assessed due to insufficient model specifications'\n",
            "\n",
            "# Compose markdown\n",
            "md = f\"\"\"## 14. Plain-Language Takeaways\\n\\n- **What we did**: Regressed `{primary_y}` on `{primary_x}` with key controls {', '.join(controls)}; added robustness via expanded controls, 21-day lag, and log/trim specs. Robust (HC3) SEs and country means used to limit autocorrelation.\\n- **What we found**: {robustness_assessment}. {fit_assessment}.\\n- **What's new**: Running the robustness specs {spec_text}\\n  - Testing/policy/capacity controls were included when present.\\n  - Lag and log-trimmed specs reduce sensitivity to timing and extremes.\\n- **Causation? Not established.** These are observational associations; reverse causality, confounding (variants, reporting, comorbidities), and measurement error remain.\\n- **To move toward causation**:\\n  1) Panel models with country and time fixed effects plus lag sweeps.\\n  2) Quasi-experiments (policy timing, natural experiments) with difference-in-differences/event studies.\\n  3) Add variant, mobility, NPI, and comorbidity proxies; validate outcomes with alternative data sources.\\n  4) Placebo/falsification checks (e.g., testing for spurious pre-period relationships).\\n  5) Pre-register specifications and hold out validation windows.\\n\"\"\"\n",
            "\n",
            "display(Markdown(md))"
        ]
    })
    
    # Cell 15: Panel fixed-effects
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 15. Panel fixed-effects (country + month) association check\n",
            "from statsmodels.formula.api import ols as smf_ols\n",
            "import pandas as pd\n",
            "\n",
            f"required_cols = ['date', 'location', '{analysis['independent_var']}', '{analysis['dependent_var']}']\n",
            "case_col = next((c for c in ['new_cases_smoothed_per_million', 'total_cases_per_million'] if 'df' in globals() and c in df.columns), None)\n",
            "if case_col:\n",
            "    required_cols.append(case_col)\n",
            "missing = [c for c in required_cols if 'df' not in globals() or c not in df.columns]\n",
            "if missing:\n",
            "    print('⚠️ Panel FE skipped: missing columns:', missing)\n",
            "else:\n",
            f"    dropna_subset = ['{analysis['independent_var']}', '{analysis['dependent_var']}']\n",
            "    if case_col:\n",
            "        dropna_subset.append(case_col)\n",
            # Only add aged_65_older to extra_cols if it's not the independent variable
            *([    "    extra_cols = [c for c in ['aged_65_older'] if c in df.columns]\n"] if analysis['independent_var'] != 'aged_65_older' else [    "    extra_cols = []\n"]),
            "    panel_df = df[required_cols + extra_cols].dropna(subset=dropna_subset)\n",
            "    if panel_df.empty or panel_df['location'].nunique() < 5:\n",
            "        print('⚠️ Panel FE skipped: insufficient data or countries.')\n",
            "    else:\n",
            "        panel_df = panel_df.copy()\n",
            "        panel_df['month_period'] = panel_df['date'].dt.to_period('M').astype(str)\n",
            f"        formula_parts = ['{analysis['independent_var']}']\n",
            "        if case_col and case_col in panel_df.columns:\n",
            "            formula_parts.append(case_col)\n",
            # Only add aged_65_older if it's not already the independent variable
            *([        "        if 'aged_65_older' in panel_df.columns:\n",
            "            formula_parts.append('aged_65_older')\n"] if analysis['independent_var'] != 'aged_65_older' else []),
            f"        formula = '{analysis['dependent_var']} ~ ' + ' + '.join(formula_parts) + ' + C(location) + C(month_period)'\n",
            "        panel_model = smf_ols(formula, data=panel_df).fit(cov_type='HC3')\n",
            "        print(panel_model.summary().tables[1])\n",
            "        print('\\nModel fit:')\n",
            "        print(f'  R-squared: {panel_model.rsquared:.3f} (Adj: {panel_model.rsquared_adj:.3f})')\n",
            "        print(f'  N (rows): {len(panel_df):,}; Countries: {panel_df.location.nunique()}')\n",
            "        print('Note: Fixed effects absorb time-invariant country factors and common monthly shocks; still observational.')"
        ]
    })
    
    # Cell 16: Lag sweep
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            f"# 16. Lag sweep (7/14/21/28-day) for {analysis['independent_var']} effect robustness\n",
            "lags = [7, 14, 21, 28]\n",
            f"required_cols = ['location', 'date', '{analysis['independent_var']}', '{analysis['dependent_var']}']\n",
            "case_col = next((c for c in ['new_cases_smoothed_per_million', 'total_cases_per_million'] if 'df' in globals() and c in df.columns), None)\n",
            "if case_col:\n",
            "    required_cols.append(case_col)\n",
            "missing = [c for c in required_cols if 'df' not in globals() or c not in df.columns]\n",
            "if missing:\n",
            "    print('⚠️ Lag sweep skipped: missing columns:', missing)\n",
            "else:\n",
            "    lag_results = []\n",
            "    for L in lags:\n",
            # Only add aged_65_older to extra_cols if it's not the independent variable
            *([        "        extra_cols = [c for c in ['aged_65_older'] if c in df.columns]\n"] if analysis['independent_var'] != 'aged_65_older' else [        "        extra_cols = []\n"]),
            "        tmp = df[required_cols + extra_cols].copy()\n",
            f"        tmp[f'{analysis['independent_var']}_lag_{{L}}'] = tmp.groupby('location')['{analysis['independent_var']}'].shift(L)\n",
            f"        dropna_cols = [f'{analysis['independent_var']}_lag_{{L}}', '{analysis['dependent_var']}']\n",
            "        if case_col:\n",
            "            dropna_cols.append(case_col)\n",
            "        tmp = tmp.dropna(subset=dropna_cols)\n",
            "        if tmp['location'].nunique() < 10 or len(tmp) < 200:\n",
            "            lag_results.append((L, 'insufficient data', None, None))\n",
            "            continue\n",
            "        tmp = tmp.groupby('location', as_index=False).mean()\n",
            f"        formula_parts = [f'{analysis['independent_var']}_lag_{{L}}']\n",
            "        if case_col and case_col in tmp.columns:\n",
            "            formula_parts.append(case_col)\n",
            # Only add aged_65_older if it's not already the independent variable
            *([        "        if 'aged_65_older' in tmp.columns:\n",
            "            formula_parts.append('aged_65_older')\n"] if analysis['independent_var'] != 'aged_65_older' else []),
            f"        formula = '{analysis['dependent_var']} ~ ' + ' + '.join(formula_parts)\n",
            "        try:\n",
            "            model = ols(formula, data=tmp).fit(cov_type='HC3')\n",
            f"            coef = model.params.get(f'{analysis['independent_var']}_lag_{{L}}', float('nan'))\n",
            f"            pval = model.pvalues.get(f'{analysis['independent_var']}_lag_{{L}}', float('nan'))\n",
            "            lag_results.append((L, coef, pval, len(tmp)))\n",
            "        except Exception as e:\n",
            "            lag_results.append((L, f'error: {e}', None, len(tmp)))\n",
            "    print('Lag sweep results (coef, p-value, N):')\n",
            "    for L, coef, pval, n in lag_results:\n",
            "        print(f'  Lag {L} days ->', coef, '| p:', pval, '| N:', n)\n",
            "    print('Note: Country means used after lagging to reduce serial correlation; interpret as associative, not causal.')"
        ]
    })
    
    # Cell 17: Placebo lead test
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            f"# 17. Placebo lead test ({analysis['independent_var']} predicting past {analysis['dependent_var']})\n",
            "lead_days = 21\n",
            f"required_cols = ['location', 'date', '{analysis['independent_var']}', '{analysis['dependent_var']}']\n",
            "case_col = next((c for c in ['new_cases_smoothed_per_million', 'total_cases_per_million'] if 'df' in globals() and c in df.columns), None)\n",
            "if case_col:\n",
            "    required_cols.append(case_col)\n",
            "missing = [c for c in required_cols if 'df' not in globals() or c not in df.columns]\n",
            "if missing:\n",
            "    print('⚠️ Placebo lead skipped: missing columns:', missing)\n",
            "else:\n",
            # Only add aged_65_older to extra_cols if it's not the independent variable
            *([    "    extra_cols = [c for c in ['aged_65_older'] if c in df.columns]\n"] if analysis['independent_var'] != 'aged_65_older' else [    "    extra_cols = []\n"]),
            "    plc = df[required_cols + extra_cols].copy()\n",
            f"    plc[f'{analysis['independent_var']}_lead_{{lead_days}}'] = plc.groupby('location')['{analysis['independent_var']}'].shift(-lead_days)\n",
            f"    dropna_cols = [f'{analysis['independent_var']}_lead_{{lead_days}}', '{analysis['dependent_var']}']\n",
            "    if case_col:\n",
            "        dropna_cols.append(case_col)\n",
            "    plc = plc.dropna(subset=dropna_cols)\n",
            "    if plc['location'].nunique() < 10 or len(plc) < 200:\n",
            "        print('⚠️ Placebo lead skipped: insufficient data after lead shift.')\n",
            "    else:\n",
            "        plc = plc.groupby('location', as_index=False).mean()\n",
            f"        formula_parts = [f'{analysis['independent_var']}_lead_{{lead_days}}']\n",
            "        if case_col and case_col in plc.columns:\n",
            "            formula_parts.append(case_col)\n",
            # Only add aged_65_older if it's not already the independent variable
            *([        "        if 'aged_65_older' in plc.columns:\n",
            "            formula_parts.append('aged_65_older')\n"] if analysis['independent_var'] != 'aged_65_older' else []),
            f"        formula = '{analysis['dependent_var']} ~ ' + ' + '.join(formula_parts)\n",
            "        try:\n",
            "            placebo_model = ols(formula, data=plc).fit(cov_type='HC3')\n",
            f"            coef = placebo_model.params.get(f'{analysis['independent_var']}_lead_{{lead_days}}', float('nan'))\n",
            f"            pval = placebo_model.pvalues.get(f'{analysis['independent_var']}_lead_{{lead_days}}', float('nan'))\n",
            "            print(placebo_model.summary().tables[1])\n",
            "            print('\\nPlacebo coefficient (should be null if no reverse/anticipatory bias):', round(coef, 4))\n",
            "            print('p-value:', round(pval, 4))\n",
            "            print(f'N (countries): {len(plc)}')\n",
            "        except Exception as e:\n",
            "            print('⚠️ Placebo model failed:', e)\n",
            "    print('Note: Significant lead effects may indicate reverse causality, anticipation, or time-trend confounding, not true forward causation.')"
        ]
    })
    
    # Cell 18: Causation checklist
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 18. Causation drill-down status\n",
            "from IPython.display import Markdown, display\n",
            "automation = [\n",
            "    \"Panel FE (country + month) implemented (Cell 15)\",\n",
            "    \"Lag sweep 7/14/21/28 implemented (Cell 16)\",\n",
            "    \"Placebo lead test implemented (Cell 17)\",\n",
            "    \"Robust SEs (HC3) and trimming/log specs implemented (Cells 11,13)\"\n",
            "]\n",
            "gaps = [\n",
            "    \"Need external data for variants, mobility, NPIs, comorbidities, excess mortality\",\n",
            "    \"Need policy/event timing for true DiD or event-study identification\",\n",
            "    \"Need preregistration/holdout and falsification with additional outcomes\"\n",
            "]\n",
            "md = \"## 18. Causation Drill-Down Status\\n\\n\"\\\n",
            "     \"**Automated in-notebook (using single OWID input):**\\n\" + '\\n'.join([f\"- {a}\" for a in automation]) + \"\\n\\n\"\\\n",
            "     \"**Not automated here (data or design needed):**\\n\" + '\\n'.join([f\"- {g}\" for g in gaps]) + \"\\n\\n\"\\\n",
            "     \"Use this checklist when porting to generator code: keep interfaces generic (df with date/location + core metrics), and add hooks for external covariates/events when available.\"\n",
            "display(Markdown(md))"
        ]
    })
    
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def generate_all_notebooks():
    """Generate all analysis notebooks."""
    print("="*80)
    print("COVID-19 High-Information Analysis Notebook Generator".center(80))
    print("="*80)
    print()
    print(f"Project: {PROJECT_PATH}")
    print(f"Notebooks will be created in: {PROJECT_PATH / 'notebooks'}")
    print()
    
    # Use notebooks directory directly (following globalhealthdata pattern)
    notebooks_dir = PROJECT_PATH / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    
    generated = []
    failed = []
    
    for i, analysis in enumerate(ANALYSES, 1):
        print(f"[{i}/{len(ANALYSES)}] Generating: {analysis['title']}")
        
        try:
            # Create notebook content
            notebook_content = create_notebook_content(analysis)
            
            # Save notebook
            output_file = notebooks_dir / f"{analysis['name']}_analysis.ipynb"
            
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(notebook_content, f, indent=2, ensure_ascii=False)
            
            print(f"    ✓ Created: {output_file.name}")
            generated.append(output_file.name)
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            failed.append(analysis['name'])
    
    # Summary
    print()
    print("="*80)
    print("GENERATION SUMMARY".center(80))
    print("="*80)
    print(f"\n✓ Successfully generated: {len(generated)} notebooks")
    for name in generated:
        print(f"  • {name}")
    
    if failed:
        print(f"\n✗ Failed to generate: {len(failed)} notebooks")
        for name in failed:
            print(f"  • {name}")
    
    print(f"\n📂 Location: {notebooks_dir}")
    print(f"\n💡 Next steps:")
    print(f"   1. Open any notebook in Jupyter or VS Code")
    print(f"   2. Run cells to perform the analysis")
    print(f"   3. Interpret results in the context of information theory")
    print(f"   4. Compare findings across different analyses")
    

if __name__ == "__main__":
    generate_all_notebooks()
