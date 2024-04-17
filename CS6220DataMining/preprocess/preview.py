import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/Duolingo_dataset.csv')

# Set display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', 20)       # Show 20 rows
pd.set_option('display.width', 1000)        # Set the display width for the terminal
pd.set_option('display.max_colwidth', None) # Show the full content of each column

# Explore the first few rows of the dataset
# Check for missing values
print(df.isnull().sum())
# Summary statistics for numerical features
print(df.describe())

# Summary statistics for categorical features
print(df.describe(include=['O']))
print(df.head())
# Display data types of each column
print(df.dtypes)



# For numerical features, plot histograms
num_features = df.select_dtypes(include=['int64', 'float64']).columns
df[num_features].hist(bins=15, figsize=(15, 6), layout=(2, 4))
plt.show()

# For categorical features, plot bar charts
cat_features = df.select_dtypes(include=['object']).columns
for col in cat_features:
    df[col].value_counts().plot(kind='bar', title=col)
    plt.show()

# Save the DataFrame with missing values info to a new CSV file
missing_values_report = df.isnull().sum().to_frame(name='missing_values')
missing_values_report.to_csv('missing_values_report.csv')