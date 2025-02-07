import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
import xgboost as xgb

# Load the dataset
df = pd.read_csv("df_Insurance.csv")

# Rename columns to remove spaces
df.rename(columns={
    "Annual Income": "Annual_Income",
    "Credit Score": "Credit_Score",
    "Health Score": "Health_Score",
    "Insurance Duration": "Insurance_Duration"
}, inplace=True)

# Select top 5 important features
selected_features = ["Age", "Annual_Income", "Health_Score", "Credit_Score", "Insurance_Duration"]
target_column = "Premium Amount"

# Drop rows with missing values in selected features
df = df.dropna(subset=selected_features + [target_column])

# Prepare features (X) and target (y)
X = df[selected_features]
y = df[target_column]

# Scale numerical features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost model
xg_reg = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
xg_reg.fit(X_train, y_train)

# Save model and scaler
with open("xgboost_model_top5.pkl", "wb") as f:
    pickle.dump(xg_reg, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")