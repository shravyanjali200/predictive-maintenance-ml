import pandas as pd

# Load the dataset
df = pd.read_csv("data/ai4i2020.csv")

# Show basic information
print("Shape of dataset:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget column distribution:")
print(df["Machine failure"].value_counts())