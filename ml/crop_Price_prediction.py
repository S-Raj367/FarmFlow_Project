import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform
import joblib

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Load dataset
df = pd.read_csv("data/coconut.csv")  # Replace with your dataset path
print(df.head())

# Check for missing values
sns.heatmap(df.isnull(), cmap="coolwarm")
plt.show()

# Distribution plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df["Rainfall"], bins=15, color="purple", kde=True)  # Use "Rainfall" instead of "temperature"
plt.subplot(1, 2, 2)
sns.histplot(df['WPI'], bins=15, color="green", kde=True)  # Use "WPI" instead of "ph"
plt.show()

# Pairplot to see correlations
sns.pairplot(df)
plt.show()

# Heatmap for correlations
sns.heatmap(df.corr(), cmap="coolwarm")
plt.show()

# Features and target
X = df.iloc[:, :-1].values  # Features (all columns except the last one)
y = df.iloc[:, -1].values   # Target (last column, "WPI")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
try:
    joblib.dump(scaler, "data/coconut_scaler.joblib")
    print("Scaler saved successfully.")
except Exception as e:
    print(f"Error saving scaler: {e}")

# RandomForestRegressor Model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
print("RandomForest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("RandomForest R2 Score:", r2_score(y_test, y_pred))

# Hyperparameter tuning for RandomForestRegressor
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10),
    'max_features': ['sqrt', 'log2'],  # Use 'sqrt' or 'log2' instead of 'auto'
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10)
}
rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_dist, n_iter=50, cv=4, n_jobs=-1, random_state=42)
rf_search.fit(X_train_scaled, y_train)
print("Best RandomForest Parameters:", rf_search.best_params_)
print("Best RandomForest Score:", rf_search.best_score_)

# Final Model (RandomForest with best parameters)
final_model = rf_search.best_estimator_
final_model.fit(X_train_scaled, y_train)
y_pred_final = final_model.predict(X_test_scaled)
print("Final Model RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_final)))
print("Final Model R2 Score:", r2_score(y_test, y_pred_final))

# Save the final model
try:
    joblib.dump(final_model, "data/coconut_price_model.joblib")
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

# Make a prediction
x = [5, 2012, 47.5]  # Example input: [Month, Year, Rainfall]
x_scaled = scaler.transform(np.array(x).reshape(1, -1))
predicted_price = final_model.predict(x_scaled)
print(f"Predicted Price: {predicted_price[0]}")