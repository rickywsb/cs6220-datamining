import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Load the data
df = pd.read_csv('../Duolingo_dataset_outliers_capped.csv')

# Remove unnecessary columns
columns_to_drop = ['timestamp', 'user_id', 'session_correct', 'session_seen', 'delta', 'lexeme_id']
df.drop(columns_to_drop, axis=1, inplace=True)

# Select the features for PCA
features = [f'vec_{i}' for i in range(50)]  # vec_0 to vec_49
X = df[features]
y = df['p_recall']  # Assuming 'p_recall' is the column to predict

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)  # Preserve 95% of the variance
X_pca = pca.fit_transform(X_scaled)

# Convert the PCA-transformed data back to DataFrame
pca_columns = [f'pca_{i}' for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=pca_columns)

# Combine the PCA results with the original DataFrame (excluding the original vector features)
df_combined = pd.concat([df.drop(features, axis=1), df_pca], axis=1)

# Convert 'p_recall' to binary class
df_combined['p_recall'] = (df_combined['p_recall'] > 0.5).astype(int)

# Separate the classes
df_majority = df_combined[df_combined.p_recall==1]
df_minority = df_combined[df_combined.p_recall==0]

# Downsample the majority class
df_majority_downsampled = resample(df_majority,
                                   replace=False,    # sample without replacement
                                   n_samples=len(df_minority),     # to match minority class
                                   random_state=123) # reproducible results

# Combine the minority class with the downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

# Save the processed dataset
df_downsampled.to_csv('dataset_final_downsample.csv', index=False)

print("PCA application and downsampling of majority class complete. Final dataset saved.")
