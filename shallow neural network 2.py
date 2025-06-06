import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv("C:/Users/User/OneDrive/Documents/Desktop/miniproject2.csv") 
# Convert 'smoker' column to binary (yes -> 1, no -> 0)
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

# One-hot encode categorical variables ('sex' and 'region')
df = pd.get_dummies(df, columns=['sex', 'region'], drop_first=True)

# Define input (X) and output (y)
X = df.drop(columns=['smoker'])  # Features
y = df['smoker']  # Target variable

# Standardize numeric features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Shallow Neural Network model (Single hidden layer with 20 neurons)
model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Smoker', 'Smoker'])
disp.plot(cmap='Blues', values_format='d', xticks_rotation='horizontal')

# Add title to the plot
plt.title("Confusion Matrix - Shallow Neural Network Model", fontsize=12, fontweight="bold")

# Show the plot
plt.show()
