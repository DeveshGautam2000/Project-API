import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, RobustScaler

# Load the dataset
df = pd.read_csv("df_Insurance.csv")

# Clean the dataset (drop rows with missing critical columns)
df_new = df.copy()
df_new = df_new.dropna(subset=['Age', 'Annual Income', 'Marital Status'])

# Function to create sine-transformed features from 'Month' and 'Day'
def date(df_new):
    df_new['Month_sin'] = np.sin(2 * np.pi * df_new['Month'] / 12)
    df_new['Day_sin'] = np.sin(2 * np.pi * df_new['Day'] / 31)
    df_new.drop(['Month', 'Day'], axis=1, inplace=True)
    return df_new

# Apply the date function
df_new = date(df_new)

# Label encode categorical columns
categorical_columns = df_new.select_dtypes(include=['object']).columns
df_new_cat = df_new.copy()
for col in categorical_columns:
    df_new_cat[col] = LabelEncoder().fit_transform(df_new_cat[col])

# Fill missing values for numerical columns with mode
numerical_columns = df_new_cat.select_dtypes(exclude=['object']).columns
for col in numerical_columns:
    df_new_cat[col] = df_new_cat[col].fillna(df_new_cat[col].mode()[0])

# Scale numerical features
scaler = RobustScaler()
df_new_cat[numerical_columns] = scaler.fit_transform(df_new_cat[numerical_columns])

# Define the target column
target_column = 'Premium Amount'
X = df_new_cat.drop(columns=[target_column])
y = df_new_cat[target_column]

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Optuna optimization function for LightGBM
def objective(trial):
    param = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int("num_leaves", 200, 512),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-1),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 5, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "max_depth": trial.suggest_int("max_depth", -1, 16),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-4, 10.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-4, 10.0),
        "seed" : 42
    }

    # Create LightGBM dataset
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    # Train the model
    model = lgb.train(param, dtrain, valid_sets=[dval])
    
    # Predictions and calculate RMSE
    y_val_pred = model.predict(X_val)
    rmsle = root_mean_squared_log_error(y_val, np.maximum(y_val_pred, 0))
    return rmsle

# Run Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Retrieve the best hyperparameters
best_params = study.best_trial.params

# Train the final model with the best parameters
final_model = lgb.train(best_params, lgb.Dataset(X, label=y))

# Evaluate model performance
y_pred = final_model.predict(X)

# Calculate performance metrics
rmsle = root_mean_squared_log_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
mape = np.mean(np.abs((y - y_pred) / y)) * 100

# Display performance metrics
print(f"\nPerformance Metrics:\n{'-'*30}")
print(f"RMSLE: {rmsle:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Feature importance plot
importances = final_model.feature_importance(importance_type='split')
features = X.columns
sorted_indices = importances.argsort()[::-1]

importance_df = pd.DataFrame({
    'Feature': [features[i] for i in sorted_indices],
    'Importance': importances[sorted_indices]
})

# Plot top 10 feature importances
plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', palette="coolwarm")
plt.title("Top 10 Feature Importances", fontsize=16, fontweight='bold')
plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()

# Residual analysis (scatter plot)
residuals = y - y_pred

plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color="#007acc")
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.title("Residuals vs Predicted Values", fontsize=16, fontweight='bold')
plt.xlabel("Predicted Values", fontsize=12)
plt.ylabel("Residuals", fontsize=12)
plt.tight_layout()
plt.show()

# Residual distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True, color="#55a630")
plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
plt.title("Distribution of Residuals", fontsize=16, fontweight='bold')
plt.xlabel("Residuals", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.show()

# Save the model using pickle
import pickle
with open("final_model.pkl", "wb") as f:
    pickle.dump(final_model, f)
