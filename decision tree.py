import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

#load the dataset
df = pd.read_csv("C:/Users/User/OneDrive/Documents/Desktop/miniproject2.csv") 

#convert 'smoker' column to binary (yes -> 1, no -> 0)
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

#one-hot encode categorical variables ('sex' and 'region')
df = pd.get_dummies(df, columns=['sex', 'region'], drop_first=True)

#define input (X) and output (y)
X = df.drop(columns=['smoker'])
y = df['smoker']

#standardize numeric features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#make predictions
y_pred = model.predict(X_test)

#evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

#create confusion matrix
cm = confusion_matrix(y_test, y_pred)

#display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Smoker', 'Smoker'])
disp.plot(cmap='Blues', values_format='d', xticks_rotation='horizontal')

#add title to the plot
plt.title("Confusion Matrix - Decision Tree", fontsize=12, fontweight="bold")
#show the plot
plt.show()

