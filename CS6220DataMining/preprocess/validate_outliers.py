import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df_before = pd.read_csv('../data/Duolingo_dataset_feature_hashed.csv')  # Before outlier treatment
df_after = pd.read_csv('../data/Duolingo_dataset_outliers_capped.csv')  # After outlier treatment


def visualize_outlier_treatment(feature):
    plt.figure(figsize=(10, 4))

    # Adjust the following line if you kept the original feature name for the capped version
    capped_feature = feature + '_capped'  # If you renamed the capped column

    # Before treatment
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df_before[feature])
    plt.title(f'{feature} Before Treatment')

    # After treatment
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df_after[capped_feature])
    plt.title(f'{feature} After Treatment')

    plt.show()


# Example calls - replace 'delta' and 'session_seen' with actual columns you've treated
visualize_outlier_treatment('delta')
visualize_outlier_treatment('session_seen')