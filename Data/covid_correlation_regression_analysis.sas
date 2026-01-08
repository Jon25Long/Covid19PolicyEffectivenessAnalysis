/*******************************************************************************
* Program: covid_correlation_regression_analysis.sas
* Purpose: Analyze COVID-19 data with correlation and regression analysis
* Date: December 31, 2025
* Description: This program performs:
*   1. Import owid-covid-data.csv
*   2. Remove missing values
*   3. Pearson correlation between stringency_index and reproduction_rate
*   4. Spearman correlation between stringency_index and reproduction_rate
*   5. Linear regression between stringency_index and reproduction_rate
*   6. Calculate variance for both variables
*   7. Export all results to Excel
*******************************************************************************/

/* Set options for cleaner output */
options nodate nonumber;

/* Define library and file paths */
%let datapath = /Users/jonlong/PublicHealthPortfolio/my-projects/policy-effectiveness-analysis/Data;
%let outputpath = /Users/jonlong/PublicHealthPortfolio/my-projects/policy-effectiveness-analysis/Data;

/* Create Data folder if it doesn't exist (you may need to create this manually) */
/* Check if output directory exists */

/* Step 1: Import the CSV file */
proc import datafile="&datapath./owid-covid-data.csv"
    out=work.covid_raw
    dbms=csv
    replace;
    guessingrows=5000;
run;

/* Check the imported data */
proc contents data=work.covid_raw;
run;

/* Step 2: Remove all missing values for the two variables of interest */
/* Keep only complete cases for stringency_index and reproduction_rate */
data work.covid_clean;
    set work.covid_raw;
    /* Keep only rows where both variables are non-missing */
    if not missing(stringency_index) and not missing(reproduction_rate);
run;

/* Display information about the cleaned dataset */
proc print data=work.covid_clean(obs=10);
    var stringency_index reproduction_rate;
    title "First 10 observations of cleaned data";
run;

/* Check sample size */
proc sql;
    select count(*) as n_observations
    from work.covid_clean;
quit;

/*******************************************************************************
* Step 3 & 4: Pearson and Spearman Correlations
*******************************************************************************/

/* Perform both Pearson and Spearman correlations */
ods output PearsonCorr=work.pearson_results
           SpearmanCorr=work.spearman_results;

proc corr data=work.covid_clean pearson spearman nosimple;
    var stringency_index reproduction_rate;
    title "Correlation Analysis: Stringency Index and Reproduction Rate";
run;

/* Create a combined correlation results table */
data work.correlation_results;
    length correlation_type $20 variable1 $30 variable2 $30;
    
    /* Pearson correlation */
    set work.pearson_results;
    if _N_ = 1 then do;
        correlation_type = "Pearson";
        variable1 = "stringency_index";
        variable2 = "reproduction_rate";
        correlation = stringency_index;
        p_value = Pstringency_index;
        output;
    end;
    
    keep correlation_type variable1 variable2 correlation p_value;
run;

data work.spearman_formatted;
    length correlation_type $20 variable1 $30 variable2 $30;
    set work.spearman_results;
    if _N_ = 1 then do;
        correlation_type = "Spearman";
        variable1 = "stringency_index";
        variable2 = "reproduction_rate";
        correlation = stringency_index;
        p_value = Pstringency_index;
        output;
    end;
    
    keep correlation_type variable1 variable2 correlation p_value;
run;

/* Combine both correlation results */
data work.all_correlations;
    set work.correlation_results work.spearman_formatted;
run;

proc print data=work.all_correlations;
    title "Summary of Correlation Results";
run;

/*******************************************************************************
* Step 5: Linear Regression
*******************************************************************************/

/* Perform linear regression with reproduction_rate as outcome */
/* and stringency_index as predictor */
ods output ParameterEstimates=work.regression_params
           FitStatistics=work.regression_fit
           ANOVA=work.regression_anova;

proc reg data=work.covid_clean;
    model reproduction_rate = stringency_index;
    title "Linear Regression: Reproduction Rate = f(Stringency Index)";
run;
quit;

/* Create a formatted regression results table */
data work.regression_results;
    set work.regression_params;
    label Variable = "Variable"
          Estimate = "Coefficient"
          StdErr = "Standard Error"
          tValue = "t Value"
          Probt = "P-value";
run;

proc print data=work.regression_results label;
    title "Regression Parameter Estimates";
run;

proc print data=work.regression_fit;
    title "Regression Model Fit Statistics";
run;

proc print data=work.regression_anova;
    title "Regression ANOVA Table";
run;

/*******************************************************************************
* Step 6: Calculate Variance
*******************************************************************************/

/* Calculate variance for both variables */
proc means data=work.covid_clean n mean std var;
    var stringency_index reproduction_rate;
    output out=work.variance_results
           n=n_stringency n_reproduction
           mean=mean_stringency mean_reproduction
           std=std_stringency std_reproduction
           var=var_stringency var_reproduction;
    title "Descriptive Statistics and Variance";
run;

/* Create a formatted variance table */
data work.variance_formatted;
    set work.variance_results;
    length variable $30;
    
    /* Row for stringency_index */
    variable = "stringency_index";
    n = n_stringency;
    mean = mean_stringency;
    std_dev = std_stringency;
    variance = var_stringency;
    output;
    
    /* Row for reproduction_rate */
    variable = "reproduction_rate";
    n = n_reproduction;
    mean = mean_reproduction;
    std_dev = std_reproduction;
    variance = var_reproduction;
    output;
    
    keep variable n mean std_dev variance;
run;

proc print data=work.variance_formatted;
    title "Variance and Descriptive Statistics by Variable";
run;

/*******************************************************************************
* Step 7: Export All Results to Excel
*******************************************************************************/

/* Export all results to a single Excel file with multiple sheets */
ods excel file="&outputpath./covid_analysis_results.xlsx"
    options(sheet_interval='proc' embedded_titles='yes');

/* Sheet 1: Data Summary */
proc print data=work.covid_clean(obs=100);
    var stringency_index reproduction_rate;
    title "Cleaned Data Sample (First 100 Observations)";
run;

/* Sheet 2: Descriptive Statistics and Variance */
proc print data=work.variance_formatted;
    title "Descriptive Statistics and Variance";
run;

/* Sheet 3: Correlation Results */
proc print data=work.all_correlations;
    title "Correlation Analysis Results";
run;

/* Sheet 4: Regression Parameter Estimates */
proc print data=work.regression_results label;
    title "Linear Regression - Parameter Estimates";
run;

/* Sheet 5: Regression Model Fit */
proc print data=work.regression_fit;
    title "Linear Regression - Model Fit Statistics";
run;

/* Sheet 6: Regression ANOVA */
proc print data=work.regression_anova;
    title "Linear Regression - ANOVA Table";
run;

ods excel close;

/* Print completion message */
data _null_;
    put "========================================";
    put "Analysis Complete!";
    put "Results exported to: &outputpath./covid_analysis_results.xlsx";
    put "========================================";
run;

/* Optional: Clean up intermediate datasets */
/*
proc datasets library=work nolist;
    delete covid_raw pearson_results spearman_results 
           correlation_results spearman_formatted;
quit;
*/
