# Import libraries
import pandas as pd  # data manipulation
import seaborn as sns  # creation of attractive graphics
import matplotlib.pyplot as plt  # data visualizations and plotting graphs

from sklearn.model_selection import train_test_split  # split dataset into training and testing
from sklearn.ensemble import RandomForestClassifier  # classification tasks
from sklearn.metrics import classification_report, confusion_matrix  # evaluate the performance of the model

# Load the Iris dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris_data = pd.read_csv(url, names=columns)

# Display a pairplot of the Iris dataset with species as hue
sns.pairplot(iris_data, hue='species')
plt.show()

# Splitting the data into features and target variable
X = iris_data.drop('species', axis=1)  # Features
y = iris_data['species']  # Target

# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Random Forest model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
from sklearn.metrics import confusion_matrix, classification_report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

