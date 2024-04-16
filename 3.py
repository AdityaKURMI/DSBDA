# Step 1: Summary statistics grouped by a qualitative variable
import pandas as pd

# Load dataset (you can replace 'data.csv' with your dataset)
data = pd.read_csv('data.csv')

# Assuming 'gender' is the categorical variable and 'age' is the numeric variable
summary_stats = data.groupby('gender')['age'].describe()

# Display summary statistics
print("Summary statistics of age grouped by gender:")
print(summary_stats)
