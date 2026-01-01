import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os


# Load data
data = pd.read_csv(r"C:\Users\siddh\OneDrive\Desktop\Documents\croprecom\crop-recommendation-\ui\Crop_recommendation.csv")

# Check columns
print("Columns:", data.columns)

# TARGET COLUMN IS 'label'
X = data.drop("label", axis=1)
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl created successfully")
