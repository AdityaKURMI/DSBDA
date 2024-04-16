# Step 1: Create the dataset
import pandas as pd
import numpy as np

# Creating sample data for academic performance
data = {
    'student_id': [101, 102, 103, 104, 105],
    'exam_score': [85, 78, 92, 65, 88],
    'attendance': [90, 95, 80, np.nan, 85],  # Introducing missing value
    'study_hours': [4, 3, 5, 2, 6],
    'grade': ['A', 'B', 'A', 'C', 'B']
}

# Creating DataFrame
df = pd.DataFrame(data)

# Step 2: Handle missing values and inconsistencies
# Checking for missing values
missing_values = df.isnull().sum()

# Handling missing values by imputing with mean
df['attendance'].fillna(df['attendance'].mean(), inplace=True)

# Step 3: Handle outliers
# For simplicity, let's assume outliers in 'exam_score' as any score above 100
# Replace outliers with the maximum allowed score (100)
max_score = 100
df['exam_score'] = np.where(df['exam_score'] > max_score, max_score, df['exam_score'])

# Step 4: Data transformations
# Let's transform 'study_hours' variable to square root to decrease skewness
df['study_hours_sqrt'] = np.sqrt(df['study_hours'])

# Displaying the transformed DataFrame
print("Transformed DataFrame:")
print(df)

# Displaying summary of missing values
print("\nMissing Values:")
print(missing_values)
