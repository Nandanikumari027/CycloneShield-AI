import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("data/cyclone_data.csv")

# Features
X = data[
    [
        "wind_speed",
        "rainfall",
        "distance_from_coast",
        "infrastructure_age",
        "population_density",
        "cyclone_intensity"
    ]
]

# Target
y = data["risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Test
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "model/cyclone_model.pkl")

print("Model saved successfully!")