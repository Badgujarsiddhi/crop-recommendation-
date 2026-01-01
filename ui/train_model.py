import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Path to CSV (same folder as this file)
DATA_PATH = os.path.join(os.path.dirname(__file__), "Crop_recommendation.csv")

data = pd.read_csv(DATA_PATH)

# Correct target column
X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model in SAME folder
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl created successfully")
