# Import libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
# Load dataset
df = pd.read_csv(r"C:\Users\ADMIN\Desktop\46 Heart Disease Prediction Using Machine Learning Algorithms\Code\Heart1\heart.csv")

# Separate features and target variable
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = df['target']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Evaluate model performance
predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, 'random_forest_model.joblib')
print("Model saved successfully!")

# Load the model
model = joblib.load('random_forest_model.joblib')
print("Model loaded successfully!")

# Prepare a new data point (example)
import pandas as pd

new_data = pd.DataFrame({
    'age': 55,
    'sex': 1,  # Male
    'cp': 2,  # Atypical angina
    'trestbps': 140,
    'chol': 200,  # Assuming a more realistic cholesterol value
    'fbs': 0,
    'restecg': 150,
    'thalach': 200,  # Assuming a more realistic heart rate
    'exang': 0,  # No exercise-induced angina
    'oldpeak': 0.2,  # Slight ST depression
    'slope': 1,  # Upsloping ST segment
    'ca': 1,  # One major vessel colored by fluoroscopy
    'thal': 3  # Normal thallium defect
}, index=[0])  # Ensure index is set for single row DataFrame


predictions = model.predict(new_data)


if predictions[0] == 1:
    print("Predicted: Heart Disease")
else:
    print("Predicted: No Heart Disease")

