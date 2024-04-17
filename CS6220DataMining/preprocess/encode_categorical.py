import pandas as pd

# Load the dataset with datetime features
df = pd.read_csv('../data/Duolingo_dataset_with_datetime.csv')

# Define the categorical columns you want to encode
categorical_columns = ['learning_language', 'ui_language']

# Apply one-hot encoding to these columns
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Optional: Drop the original datetime column if it's no longer needed
df_encoded.drop('datetime', axis=1, inplace=True)

# Save the dataframe with one-hot encoded columns to a new CSV file
df_encoded.to_csv('Duolingo_dataset_encoded.csv', index=False)

# Print out the new shape of the DataFrame
print(f'The shape of the updated DataFrame is: {df_encoded.shape}')
