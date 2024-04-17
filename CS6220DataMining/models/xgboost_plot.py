import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('dataset_final_downsample.csv')

# Convert 'p_recall' to binary if it's not already
data['p_recall'] = (data['p_recall'] > 0.5).astype(int)

# Prepare features and target
features = ['session_seen_capped', 'delta_capped', 'pca_1', 'learning_language_fr',
            'history_seen', 'pca_9', 'history_correct', 'pca_13', 'pca_15', 'pca_17',
            'learning_language_it', 'ui_language_es', 'pca_12', 'pca_14', 'pca_7',
            'pca_5', 'pca_18', 'pca_3', 'pca_0', 'pca_4', 'pca_16', 'pca_10', 'pca_2',
            'pca_19', 'pca_8', 'ui_language_it', 'pca_11', 'pca_6', 'hour_of_day',
            'learning_language_es', 'learning_language_en', 'learning_language_pt']
X = data[features]
y = data['p_recall']

# Define and fit the model
model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, max_depth=3)
eval_set = [(X, y)]
model.fit(X, y, eval_metric="logloss", eval_set=eval_set, verbose=True)

# Accessing the training loss data
results = model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

# Plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.legend()

plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.savefig('xgboost_loss_curve.png')  # Save the figure
plt.show()

# Make predictions
predictions = model.predict(X)
probs = model.predict_proba(X)[:, 1]  # probabilities for the positive class

# Calculate metrics
accuracy = accuracy_score(y, predictions)
roc_auc = roc_auc_score(y, probs)
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
f1 = f1_score(y, predictions)

# Save the results to a text file
results_path = 'xgboost_classification_results_v2.txt'
with open(results_path, 'w') as file:
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Area Under ROC: {roc_auc}\n")
    file.write(f"Precision: {precision}\n")
    file.write(f"Recall: {recall}\n")
    file.write(f"F1 Score: {f1}\n")

print(f"Results saved to {results_path}")
