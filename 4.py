# Step 2: Basic statistical details of species in iris dataset
import pandas as pd

# Load iris dataset
iris_data = pd.read_csv('iris.csv')

# Assuming 'species' is the categorical variable
# Selecting data for each species
setosa_stats = iris_data[iris_data['species'] == 'Iris-setosa'].describe()
versicolor_stats = iris_data[iris_data['species'] == 'Iris-versicolor'].describe()
virginica_stats = iris_data[iris_data['species'] == 'Iris-virginica'].describe()

# Display statistical details for each species
print("\nStatistical details for Iris-setosa:")
print(setosa_stats)
print("\nStatistical details for Iris-versicolor:")
print(versicolor_stats)
print("\nStatistical details for Iris-virginica:")
print(virginica_stats)
