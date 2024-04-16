# Step 1: Download the Iris dataset and load it into a DataFrame
import pandas as pd

# Download the Iris dataset from the UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# Define column names
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
# Load the dataset into a DataFrame
df = pd.read_csv(url, names=column_names)

# Step 2: List down the features and their types
features = df.columns[:-1]  # Exclude the 'species' column
feature_types = df.dtypes[:-1]  # Exclude the 'species' column
print("Features and their types:")
for feature, ftype in zip(features, feature_types):
    print(f"{feature}: {ftype}")

# Step 3: Create histograms for each feature
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure and axes
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Plot histograms for each feature
for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    sns.histplot(df[feature], ax=axes[row, col], kde=True)
    axes[row, col].set_title(f'Histogram of {feature}')

# Adjust layout
plt.tight_layout()
plt.show()

# Step 4: Create boxplots for each feature
# Set up the figure and axes
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Plot boxplots for each feature
for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    sns.boxplot(x=df[feature], ax=axes[row, col])
    axes[row, col].set_title(f'Boxplot of {feature}')

# Adjust layout
plt.tight_layout()
plt.show()

# Step 5: Compare distributions and identify outliers
# We can visually inspect the histograms and boxplots to identify outliers and compare distributions.
# Outliers may appear as points outside the whiskers in the boxplots.
