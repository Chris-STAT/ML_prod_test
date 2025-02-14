import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import json

print(f"XGBoost version: {xgb.__version__}")

# SageMaker specific directories
prefix = '/opt/ml/'
input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')

# Load the Iris dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(os.path.join(input_path, 'training', 'input_data.csv'), header=None, names=names)

# Convert feature columns to numeric type
col_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
for column in col_names:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Drop any rows with NaN values that might have resulted from the conversion
df = df.dropna()

# Prepare the data
X = df.drop('class', axis=1)
y = df['class']

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y))
}

num_round = 100
model = xgb.train(params, dtrain, num_round)

# Make predictions
y_pred = model.predict(dtest)
y_pred_classes = np.argmax(y_pred, axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_classes)
classification_rep = classification_report(y_test, y_pred_classes, target_names=le.classes_)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)

# Save the model
os.makedirs(model_path, exist_ok=True)
model_file = os.path.join(model_path, 'xgboost-model')
model.save_model(model_file)

# Save feature column names
feature_columns = X.columns.tolist()
with open(os.path.join(model_path, 'feature_columns.json'), 'w') as f:
    json.dump(feature_columns, f)

# Save label encoder classes
with open(os.path.join(model_path, 'classes.json'), 'w') as f:
    json.dump(le.classes_.tolist(), f)

# Save metrics and evaluation results
os.makedirs(output_path, exist_ok=True)
metrics = {'accuracy': float(accuracy)}
with open(os.path.join(output_path, 'metrics.json'), 'w') as f:
    json.dump(metrics, f)

with open(os.path.join(output_path, 'evaluation_results.json'), 'w') as f:
    json.dump({'classification_report': classification_rep}, f)

print("Training completed. Model and results saved.")
