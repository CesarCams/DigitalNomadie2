import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json

features_all_train = np.load("features_all_train_miniLM.npy")
features_all_test = np.load("features_all_test_miniLM.npy")
labels_train = np.load("labels_train_miniLM.npy")

# Load your feature matrix and labels from previous steps
# feature_matrix: the matrix of averaged embeddings for each tweet
# labels: the corresponding labels (+1 for positive, -1 for negative)

# Split the data into training and test sets (e.g., 80% train, 20% test)
#   X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.0, random_state=42)

features = features_all_train
labels = labels_train

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


#X_test = X_test
# Normalize the feature matrix

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the SGDClassifier
model = SVC()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance using accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Test set accuracy:", accuracy)

# Evaluate the model's performance using F1 score
# f1 = f1_score(y_test, y_pred)
# print("Test set F1 score:", f1)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = {
    "model": "MiniLM",
    "classifier": "SVC",
    "accuracy": accuracy,
    "f1_score": f1
}

with open("model_results.json", "w") as f:
    json.dump(results, f)
