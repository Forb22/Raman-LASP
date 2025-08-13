import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

#Load and Prepare Data
datafile = '/Users/guy/Desktop/Sherbrooke_Lab_Data/Raman Data/ML DATA All.csv'
data = np.genfromtxt(datafile, delimiter=',', unpack=False, skip_header=1, usecols=range(1, 17), dtype=float)

X = data[:, 1:]
Y = data[:, 0]

# Prepare feature names for plotting
feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]

# Encode labels (change temperature (target) values from numbers to labels) and split data
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42)

#Define and Train the Final Model
best_params = {
    'subsample': 0.9,
    'n_estimators': 350,
    'min_child_weight': 4,
    'max_depth': 7,
    'learning_rate': 0.07,
    'gamma': 0.05,
    'colsample_bytree': 0.8,
    'objective': 'multi:softmax',
    'num_class': 4,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss'
}

final_model = xgb.XGBClassifier(**best_params)

# Train the model on the training data
print("Training the final model with best parameters...")
final_model.fit(X_train, Y_train)
print("Training complete.")

# Make predictions on the test set
y_pred = final_model.predict(X_test)

#Overall Accuracy
score = accuracy_score(Y_test, y_pred)
print(f"\nAccuracy on the test set: {score:.4f}")

#Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
# Use le.classes_ to get the original temperature names (500, 600, etc.)
print(classification_report(Y_test, y_pred, target_names=le.classes_.astype(str)))

#Confusion Matrix
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_.astype(str),
            yticklabels=le.classes_.astype(str))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Final Model')
plt.show()

#Feature Importance
print("\nGenerating Feature Importance Plot...")
importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importances)
plt.title('XGBoost Feature Importance')
plt.show()

print("\nTop 5 Most Important Features:")
print(importances.head(5))