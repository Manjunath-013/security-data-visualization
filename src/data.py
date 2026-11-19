import pandas as pd

# Load dataset
df = pd.read_csv("data/incidents.csv")

# Print column headings
print("DATASET COLUMN HEADINGS:")
print(df.columns.tolist())

# Print one complete row with column names
print("\nONE COMPLETE INCIDENT RECORD:")
print(df.iloc[0].to_string())