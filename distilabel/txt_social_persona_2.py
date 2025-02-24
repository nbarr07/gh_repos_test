import pandas as pd

# Load any of the CSVs
df = pd.read_csv("troll_dataset.csv")

# Quick analysis
print(f"Total interactions: {len(df)}")
print("\nSample of responses:")
print(df[['post', 'interaction_troll']].head())

# Basic statistics
print("\nResponse lengths:")
df['response_length'] = df['interaction_troll'].str.len()
print(df['response_length'].describe())

# Find empty or very short responses
print("\nPotentially problematic responses:")
print(df[df['response_length'] < 10][['post', 'interaction_troll']])