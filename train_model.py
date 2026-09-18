
import pickle
import os

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
digits = load_digits()

X = digits.data
y = digits.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = RandomForestClassifier(
    n_estimators=100, random_state=42
)

model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print("Model accuracy:", round(accuracy * 100, 2), "%")

# Create models folder
os.makedirs("models", exist_ok=True)

# Save the trained model
with open("models/digit_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")