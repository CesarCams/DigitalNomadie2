import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import random 
# Chemins vers les fichiers
features_path = "features_output/features_miniLM.pt"
labels_path = "features_output/labels_miniLM.npy"  # Chemin vers le fichier contenant les labels

# Charger les features depuis le fichier .pt
features_distilbert = torch.load(features_path).cpu().numpy()

# Charger les labels depuis le fichier .npy
labels = np.load(labels_path)

# Vérifier que le nombre de features correspond au nombre de labels
assert features_distilbert.shape[0] == labels.shape[0], "Le nombre de features et de labels ne correspond pas."

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features_distilbert, labels, test_size=0.2, random_state=42)

# Normalisation des features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialiser et entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Convertir les prédictions en labels binaires (+1 ou -1)
y_pred_labels = np.where(y_pred >= 0, 1, -1)

# Évaluation des performances
accuracy = accuracy_score(y_test, y_pred_labels)
f1 = f1_score(y_test, y_pred_labels)

print("Test set accuracy:", accuracy)
print("Test set F1 score:", f1)
