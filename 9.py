# Step 1: Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_titanic

# Load Titanic dataset
titanic = load_titanic(as_frame=True)
df = titanic.frame

# Step 2: Visualize patterns in the data using Seaborn
# Let's start by plotting the count of survivors by class
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survivors by Class')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Step 3: Plot histogram to visualize the distribution of ticket prices
sns.histplot(df['Fare'], bins=20, kde=True)
plt.title('Distribution of Ticket Prices')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()
