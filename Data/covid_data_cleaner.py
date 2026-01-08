"""
COVID-19 Data Cleaner
This script imports the OWID COVID-19 dataset, removes all missing values,
and exports the cleaned data to an Excel file.
"""

import pandas as pd
import os

# Define file paths
data_folder = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(data_folder, "owid-covid-data.csv")
output_file = os.path.join(data_folder, "owid-covid-data-cleaned.xlsx")

# Step 1: Import the CSV file
print(f"Reading data from: {input_file}")
df = pd.read_csv(input_file)
print(f"Original data shape: {df.shape}")
print(f"Total missing values: {df.isnull().sum().sum()}")

# Step 2: Remove all missing values
print("\nAnalyzing missing data...")
# Calculate missing percentage for each column
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print(f"Columns with >50% missing: {(missing_pct > 50).sum()}")

# Drop columns that are more than 80% empty
threshold = 80
cols_to_keep = missing_pct[missing_pct <= threshold].index.tolist()
df_filtered = df[cols_to_keep]
print(f"Keeping {len(cols_to_keep)} columns with <={threshold}% missing values")

# Now remove rows with any remaining missing values
print("\nRemoving rows with any missing values...")
df_cleaned = df_filtered.dropna()
print(f"Cleaned data shape: {df_cleaned.shape}")
print(f"Rows removed: {len(df) - len(df_cleaned)}")

# Step 3: Export to Excel file
print(f"\nExporting cleaned data to: {output_file}")
df_cleaned.to_excel(output_file, index=False, engine='openpyxl')
print("Export complete!")

# Display summary statistics
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Original rows: {len(df)}")
print(f"Cleaned rows: {len(df_cleaned)}")
print(f"Columns: {len(df_cleaned.columns)}")
print(f"Output file: {output_file}")
