import os
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
from scipy.stats import randint, uniform

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Load dataset
df = pd.read_csv("data/Crop_recommendation.csv")
print(df.head())

# Unique classes of plants
print(df["label"].unique())

# Heatmap to check null/missing values
sns.heatmap(df.isnull(), cmap="coolwarm")
plt.show()

# Distribution plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df["temperature"], bins=15, color="purple", kde=True)
plt.subplot(1, 2, 2)
sns.histplot(df['ph'], bins=15, color="green", kde=True)
plt.show()

# Countplot for labels
sns.countplot(y='label', data=df, palette="plasma_r")
plt.show()

# Pairplot to see correlations
sns.pairplot(df, hue='label')
plt.show()

# Heatmap for correlations
sns.heatmap(df.corr(), cmap="coolwarm")
plt.show()

# Convert labels to categorical codes
c = df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target'] = c.cat.codes

# Features and target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "crop_recommendation_scaler.joblib")

# KNN Model
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
print("KNN Accuracy:", knn.score(X_test_scaled, y_test))

# Confusion matrix for KNN
mat = confusion_matrix(y_test, knn.predict(X_test_scaled))
df_cm = pd.DataFrame(mat, list(targets.values()), list(targets.values()))
sns.set(font_scale=1.0)
plt.figure(figsize=(12, 8))
sns.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap="terrain")
plt.show()

# SVM Model
svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(X_train_scaled, y_train)
print("SVM RBF Kernel Accuracy:", svc_rbf.score(X_test_scaled, y_test))

# Hyperparameter tuning for SVM
param_dist = {
    'C': uniform(0.1, 100),
    'gamma': uniform(0.01, 10),
    'kernel': ['rbf', 'poly']
}
svm_search = RandomizedSearchCV(SVC(), param_distributions=param_dist, n_iter=50, cv=4, n_jobs=-1, random_state=1)
svm_search.fit(X_train_scaled, y_train)
print("Best SVM Parameters:", svm_search.best_params_)
print("Best SVM Score:", svm_search.best_score_)

# RandomForest Model
rf = RandomForestClassifier(random_state=1)
rf.fit(X_train_scaled, y_train)
print("RandomForest Accuracy:", rf.score(X_test_scaled, y_test))

# Hyperparameter tuning for RandomForest
param_dist_rf = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10),
    'max_features': ['auto', 'sqrt', 'log2']
}
rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=1), param_distributions=param_dist_rf, n_iter=50, cv=4, n_jobs=-1, random_state=1)
rf_search.fit(X_train_scaled, y_train)
print("Best RandomForest Parameters:", rf_search.best_params_)
print("Best RandomForest Score:", rf_search.best_score_)

# XGBoost Model
xgb_model = xgb.XGBClassifier(random_state=1)
xgb_model.fit(X_train_scaled, y_train)
print("XGBoost Accuracy:", xgb_model.score(X_test_scaled, y_test))

# Hyperparameter tuning for XGBoost
param_dist_xgb = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5)
}
xgb_search = RandomizedSearchCV(xgb.XGBClassifier(random_state=1), param_distributions=param_dist_xgb, n_iter=50, cv=4, n_jobs=-1, random_state=1)
xgb_search.fit(X_train_scaled, y_train)
print("Best XGBoost Parameters:", xgb_search.best_params_)
print("Best XGBoost Score:", xgb_search.best_score_)

# Final Model (SVM with best parameters)
final_model = svm_search.best_estimator_
final_model.fit(X_train_scaled, y_train)
print("Final Model Accuracy:", final_model.score(X_test_scaled, y_test))

# Save the final model
joblib.dump(final_model, "crop_recommendation_svm.joblib")