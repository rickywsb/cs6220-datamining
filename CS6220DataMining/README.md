
# Data Preprocessing Documentation

Overview
This README provides a detailed account of the data preprocessing steps undertaken for the Duolingo language learning session dataset. The process transforms raw data into a format suitable for performing machine learning tasks.

Raw Data
The raw dataset, original_Duolingo_dataset.csv, represents user interactions with the Duolingo application and includes features such as user activity timestamps, session details, and performance metrics.

Preprocessing Steps
Step 1: Timestamp Conversion
Script: datetime_conversion.py
Description: Converts Unix timestamps to human-readable datetime, extracting hour of the day and day of the week.
Output: Duolingo_dataset_with_datetime.csv

Step 2: One-Hot Encoding of Categorical Variables
Script: encode_categorical.py
Description: Applies one-hot encoding to the learning_language and ui_language columns to convert categorical variables into a numerical format.
Output: Duolingo_dataset_encoded.csv

Step 3: Feature Scaling
Script: scale_features.py
Description: Standardizes numerical features like delta, history_seen, session_seen, etc., to have a mean of zero and a standard deviation of one.
Output: Duolingo_dataset_scaled.csv

Step 4: Handling High Cardinality Features
Script: handle_cardinality.py
Description: Reduces dimensionality of high-cardinality features like user_id and lexeme_id using feature hashing.
Output: Duolingo_dataset_feature_hashed.csv

Step 5: Outlier Detection and Treatment
Script: outlier_treatment.py
Description: Caps outliers in features like delta and session_seen at the 95th percentile to limit the influence of extreme values.
Output: Duolingo_dataset_outliers_capped.csv

Step 6: Validation and Testing
Script: validate_outliers.py
Description: Visually compares feature distributions before and after outlier treatment and checks the final dataset's integrity.
Tools: Uses boxplots and statistical summaries for validation.

Final Dataset
The preprocessed dataset, Duolingo_dataset_outliers_capped.csv, is the final output ready for machine learning tasks. It reflects all preprocessing steps and is considered clean and structured for analysis.

