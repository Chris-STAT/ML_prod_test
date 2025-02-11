import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import boto3

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(url, names=names)

# Prepare the data
X = df.drop('class', axis=1)
y = df['class']

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=le.classes_)

# Save the model
joblib.dump(model, 'model.joblib')

# Create a tarball of the model
import tarfile
with tarfile.open('model.tar.gz', 'w:gz') as tar:
    tar.add('model.joblib')

# Save metrics and evaluation results
metrics = {'accuracy': accuracy}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

with open('evaluation_results.json', 'w') as f:
    json.dump({'classification_report': classification_rep}, f)

# Upload to S3
s3 = boto3.client('s3')
model_artifacts_bucket = os.environ['MODEL_ARTIFACTS_BUCKET']

s3.upload_file('model.tar.gz', model_artifacts_bucket, 'model.tar.gz')
s3.upload_file('metrics.json', model_artifacts_bucket, 'metrics.json')
s3.upload_file('evaluation_results.json', model_artifacts_bucket, 'evaluation_results.json')

print("Training completed. Model and results uploaded to S3.")