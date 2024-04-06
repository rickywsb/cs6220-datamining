import pandas as pd

# Load the feature-hashed dataset
df = pd.read_csv('Duolingo_dataset_feature_hashed.csv')

# Specify the features to check for outliers and the percentile for capping
features_to_treat = ['delta', 'session_seen']
cap_percentile = 0.95

for feature in features_to_treat:
    # Determine the percentile value
    cap_value_upper = df[feature].quantile(cap_percentile)
    cap_value_lower = df[feature].quantile(1 - cap_percentile)

    # Cap outliers
    df[f'{feature}_capped'] = df[feature].clip(lower=cap_value_lower, upper=cap_value_upper)

    # Optionally, you can drop the original column if you don't need it anymore
    # df.drop(feature, axis=1, inplace=True)

# Save the dataframe with outliers capped
df.to_csv('Duolingo_dataset_outliers_capped.csv', index=False)
