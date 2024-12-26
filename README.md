# codealpha-task-3
new repo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Step 1: Load Dataset
# Replace 'medical_data.csv' with your dataset file
data = pd.read_csv('medical_data.csv')

# Step 2: Data Preparation
# Identify features (X) and target (y)
X = data.drop(['disease'], axis=1)  # 'disease' is the target column
y = data['disease']

# Encode categorical features
categorical_features = X.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_features]))
X_encoded.columns = encoder.get_feature_names_out(categorical_features)

# Combine numerical and encoded categorical features
numerical_features = X.select_dtypes(exclude=['object'])
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(numerical_features), columns=numerical_features.columns)
X_prepared = pd.concat([X_scaled, X_encoded], axis=1)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42)

# Step 3: Model Development
# Train a Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
predictions = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# Metrics
print("Model Metrics:")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, average='weighted'))
print("Recall:", recall_score(y_test, predictions, average='weighted'))
print("F1 Score:", f1_score(y_test, predictions, average='weighted'))
print("ROC AUC:", roc_auc_score(pd.get_dummies(y_test).values, model.predict_proba(X_test), multi_class='ovr'))

# Step 5: Save the Model
import joblib
joblib.dump(model, 'disease_prediction_model.pkl')

# Notes:
# - Replace 'disease' with the actual target column name in your dataset.
# - Ensure the dataset is cleaned and preprocessed correctly before using the script.
# - For multiclass classification, ensure metrics and data preparation match the requirements.
