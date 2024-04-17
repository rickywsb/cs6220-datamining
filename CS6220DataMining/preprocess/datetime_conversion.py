import pandas as pd

# Load dataset
df = pd.read_csv('Duolingo_dataset.csv')

# Convert 'timestamp' to datetime and extract features
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df['hour_of_day'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Save the dataframe with new features
df.to_csv('Duolingo_dataset_with_datetime.csv', index=False)
