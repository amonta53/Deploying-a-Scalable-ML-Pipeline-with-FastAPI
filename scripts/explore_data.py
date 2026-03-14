# ---------------------------------------------------------------------
# Script: scripts.explore_data
# Lightweight exploratory analysis for the census dataset.
#
# This script is used during development to inspect dataset structure,
# identify missing values, check feature distributions, and detect
# potential data quality issues before building the ML pipeline.
#
# Responsibilities include:
# - loading the raw dataset
# - profiling dataset structure
# - identifying placeholder missing values
# - examining feature distributions
# ---------------------------------------------------------------------
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "census.csv"

# Load the census dataset used for the income prediction task.
# This dataset is the classic "Adult" dataset derived from the 1994
# U.S. Census. Our goal later will be to predict whether income exceeds $50K
# per year. Automatic whitespace cleanup detection is enabled to handle any
# extra spaces in the CSV file.
df = pd.read_csv(DATA_PATH, skipinitialspace=True)


# How big is the dataset?
# Expect ~32K rows in the training dataset version.  Interesting documentation
# said 48k ??
print("Shape:", df.shape)


# Always inspect column names first. Many versions of this dataset differ
# slightly. For example, some versions use "income" while others
# use "salary".
print("\nColumns:")
print(df.columns.tolist())


# Basic structure overview:
# - data types
# - non-null counts
# - memory footprint
# Useful for identifying categorical vs numeric features.
print("\nInfo:")
df.info()


# Check for missing values.
# That means pandas will not detect them here.
print("\nMissing values (NaN check only):")
print(df.isna().sum())

# Placeholder values often used instead of NaN
fillers = ["?", "NA", "N/A", "None", "null", "unknown", "Unknown", "", " "]

print("\nChecking for placeholder values:")
found_placeholder = False

# Loop through and look for any of the common placeholder values in the
# dataset.
# If found, print the counts for each column.
# This is important because these placeholders can cause issues during
# model training if not handled properly (e.g., imputation, removal, or
# treating as a separate category).
for filler in fillers:
    counts = (df == filler).sum()
    if counts.sum() > 0:
        found_placeholder = True
        print(f"\nPlaceholder '{filler}' detected:")
        print(counts[counts > 0])

if not found_placeholder:
    print("No placeholder values detected.")

# Examine the target variable distribution.
# In this dataset the target is 'salary', representing whether
# income > 50K. Expect roughly a 75/25 split between <=50K and >50K.
print("\nSalary distribution:")
print(df['salary'].value_counts())


# Summary statistics for numeric columns.
# This helps identify ranges, skewed values, and possible outliers.
# Pay special attention to:
# - capital-gain
# - capital-loss
# These are usually extremely skewed.
print("\nSummary statistics:")
print(df.describe())

# Show the first few rows to visually inspect the dataset.
# This helps confirm:
# - the file loaded correctly
# - column order and naming look right
# - categorical values appear as expected
# - no obvious formatting issues (extra spaces, weird chars)

print("\nFirst 5 rows:")
print(df.head())

# Display the data types for each column.
# Important for identifying which features are numeric vs categorical.
# Pandas labels most categorical fields as "object", which means they will
# likely need encoding (e.g., one-hot encoding) before training a model.
print("\nColumn data types:")
print(df.dtypes.value_counts())

# Check skewness of numeric features to understand distribution shape.
# Highly skewed values indicate most observations cluster near one value
# with a few extreme outliers. In this dataset, financial variables such as
# capital-gain and capital-loss are expected to be strongly right-skewed.
print("\nSkewness (numeric features):")
print(df.select_dtypes(include='number').skew())
