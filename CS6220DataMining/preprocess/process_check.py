import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the final processed CSV file
df = pd.read_csv('../data/Duolingo_dataset_outliers_capped.csv')

# Get descriptive statistics
print(df.describe())
# Plot histograms for all numerical features
df.hist(figsize=(12, 8), bins=50)
plt.show()

# Or boxplots for individual features
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    sns.boxplot(x=df[col])
    plt.show()

# Check for any NaN values across the dataset
print(df.isnull().sum())

print(df.sample(10))

# Load the original dataset
df_original = pd.read_csv('Duolingo_dataset.csv')

# Compare specific statistics or feature distributions
# For example, check the range of values for a particular feature
print("Original 'session_seen' range:", df_original['session_seen'].min(), "to", df_original['session_seen'].max())
print("Processed 'session_seen' range:", df['session_seen_capped'].min(), "to", df['session_seen_capped'].max())