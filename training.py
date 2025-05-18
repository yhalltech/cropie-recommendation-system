import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load and preprocess data
df = pd.read_csv("Crop_recommendation.csv")
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

# Split features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Train final model on full dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

# Save model and preprocessing objects
joblib.dump(rf_model, 'crop_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

# ----------------------------------
# To make predictions (in a separate file):
# ----------------------------------
# Load saved artifacts
model = joblib.load('crop_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Sample input (N, P, K, temperature, humidity, ph, rainfall)
new_data = [[90, 42, 43, 20.88, 82.00, 6.50, 202.94]]  # Replace with your values

# Preprocess and predict
scaled_data = scaler.transform(new_data)
prediction = model.predict(scaled_data)

# Decode label
predicted_crop = le.inverse_transform(prediction)[0]
print(f"Recommended crop: {predicted_crop}")