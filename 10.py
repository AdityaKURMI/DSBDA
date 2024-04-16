# Step 1: Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_titanic

# Load Titanic dataset
titanic = load_titanic(as_frame=True)
df = titanic.frame

# Step 2: Plot box plot for distribution of age with respect to each gender and survival status
plt.figure(figsize=(10, 6))
sns.boxplot(x='Sex', y='Age', hue='Survived', data=df)
plt.title('Distribution of Age by Gender and Survival Status')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()
