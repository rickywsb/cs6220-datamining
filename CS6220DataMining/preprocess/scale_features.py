import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the one-hot encoded dataset
df = pd.read_csv('../data/Duolingo_dataset_encoded.csv')


# Define the numerical columns you want to scale
numerical_columns = ['delta', 'history_seen', 'history_correct', 'session_seen', 'session_correct']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and transform the numerical columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Save the dataframe with scaled numerical columns to a new CSV file
df.to_csv('Duolingo_dataset_scaled.csv', index=False)

# Print out a summary of the scaled data
print(df[numerical_columns].describe())
