
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)
import pickle
import matplotlib.pyplot as plt

# Load dataset
digits = load_digits()

X = digits.data
y = digits.target

# Use the same test split as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model
with open("models/digit_model.pkl", "rb") as file:
    model = pickle.load(file)

# Make predictions
predictions = model.predict(X_test)

# Display accuracy
print("Accuracy:", accuracy_score(y_test, predictions))

# Display detailed evaluation
print(classification_report(y_test, predictions))

# Create confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions
)

plt.title("Digit Recognition Confusion Matrix")
plt.show()