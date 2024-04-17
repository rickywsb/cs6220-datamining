import pandas as pd
from sklearn.feature_extraction import FeatureHasher

# Load the dataset
df = pd.read_csv('../data/Duolingo_dataset_scaled.csv')

# Initialize the FeatureHasher
hasher = FeatureHasher(n_features=10, input_type='string')

# The fix: Ensure that the input to FeatureHasher is an iterable of iterables.
# We use .apply(lambda x: [str(x)]) to ensure each cell is converted to a list of strings
hashed_features_user = hasher.fit_transform(df['user_id'].apply(lambda x: [str(x)])).toarray()
hashed_features_lexeme = hasher.transform(df['lexeme_id'].apply(lambda x: [str(x)])).toarray()

# Proceed as before
hashed_feature_df_user = pd.DataFrame(hashed_features_user, columns=[f'user_hash_{i}' for i in range(10)])
hashed_feature_df_lexeme = pd.DataFrame(hashed_features_lexeme, columns=[f'lexeme_hash_{i}' for i in range(10)])

# Concatenate hashed features with the original dataframe
df = pd.concat([df.reset_index(drop=True), hashed_feature_df_user.reset_index(drop=True), hashed_feature_df_lexeme.reset_index(drop=True)], axis=1)


# Drop the original 'user_id', 'lexeme_id' and 'lexeme_string' columns
df.drop(['user_id', 'lexeme_id', 'lexeme_string'], axis=1, inplace=True)

# Save the dataframe with hashed and parsed features to a new CSV file
df.to_csv('Duolingo_dataset_feature_hashed.csv', index=False)
