import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json

features_all_train = np.load("features_all_train_miniLM.npy")
features_all_test = np.load("features_all_test_miniLM.npy")
labels_train = np.load("labels_train_miniLM.npy")

features = features_all_train
labels = labels_train

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the SGDClassifier
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

y_pred_labels = np.where(y_pred ==1 , 1,-1)

accuracy = accuracy_score(y_test, y_pred_labels)
f1 = f1_score(y_test, y_pred_labels)

results = {
    "model": "MiniLM",
    "classifier": "Logistic",
    "accuracy": accuracy,
    "f1_score": f1
}

with open("model_results.json", "w") as f:
    json.dump(results, f)
