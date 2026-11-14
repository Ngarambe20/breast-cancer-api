# train_model.py

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("1. Loading dataset...")
# The dataset contains features of cell nuclei to predict if a tumor is malignant (0) or benign (1)
data = load_breast_cancer()
X = data.data
y = data.target

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Training the model...")
# We'll use a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("3. Evaluating the model...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("4. Saving the model...")
# Save the trained model to a file named 'model.joblib'
joblib.dump(model, 'model.joblib')
print("Model saved successfully as 'model.joblib'!")