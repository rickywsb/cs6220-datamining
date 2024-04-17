Preprocessing Steps
Step 1: Timestamp Conversion
Script: datetime_conversion.py
Description: Converts Unix timestamps to a more interpretable datetime format. This step facilitates the extraction of potentially influential features like the hour of the day and day of the week, which can affect user performance.
Output: Duolingo_dataset_with_datetime.csv
Purpose: Understanding time-based patterns in user behavior which could impact learning effectiveness.

Step 2: One-Hot Encoding of Categorical Variables
Script: encode_categorical.py
Description: Transforms categorical variables such as the languages being learned (learning_language) and user interface language (ui_language) into a binary matrix, making them suitable for machine learning algorithms that require numerical input.
Output: Duolingo_dataset_encoded.csv
Purpose: To eliminate any bias or misinterpretation that algorithms might derive from categorical text data.

Step 3: Feature Scaling
Script: scale_features.py
Description: Standardizes features like delta, history_seen, and session_seen to have zero mean and unit variance, ensuring that no variable dominates others due to scale differences.
Output: Duolingo_dataset_scaled.csv
Purpose: To normalize data range which helps in speeding up the convergence in algorithms that use gradient descent.

Step 4: Handling High Cardinality Features
Script: handle_cardinality.py
Description: Reduces the dimensionality of features with high cardinality. Applies natural language processing techniques to text data to extract meaningful patterns.
Output: Duolingo_dataset_nlp_processed.csv
Purpose: To capture the essence of textual data and reduce computational complexity in modeling.

Step 5: Outlier Detection and Treatment
Script: outlier_treatment.py
Description: Identifies and caps outliers in numeric features like delta and session_seen at their 95th percentile. This limits the impact of extreme value distortions on model training.
Output: Duolingo_dataset_outliers_capped.csv
Purpose: To prevent outliers from skewing the model results, leading to more robust predictions.

Step 6: Dimensionality Reduction and Class Imbalance Handling
Description: Applies Principal Component Analysis (PCA) to reduce feature dimensionality further and utilizes downsampling techniques to address class imbalances, ensuring the model does not bias towards the majority class.
Purpose: To enhance model training efficiency and improve the fairness of predictive outcomes across different classes.
