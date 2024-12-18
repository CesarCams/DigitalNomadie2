import torch
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Chemins vers les fichiers
features_path = "features_output/features_miniLM.pt"
labels_path = "features_output/labels_miniLM.npy"  # Chemin vers le fichier contenant les labels

# Charger les features depuis le fichier .pt
features_twitter_roberta = torch.load(features_path).cpu().numpy()

# Charger les labels depuis le fichier .npy
labels = np.load(labels_path)

# Vérifier que le nombre de features correspond au nombre de labels
assert features_twitter_roberta.shape[0] == labels.shape[0], "Le nombre de features et de labels ne correspond pas."

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features_twitter_roberta, labels, test_size=0.2, random_state=42)

# Normalisation des features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Définir les paramètres pour la Grid Search
param_grid = {
    'C': [0.01,0.1, 1, 10,100,1000],  # Valeurs de régularisation (inverse de lambda)
    'penalty': ['l2'],  # Différents types de régularisation
}

# Initialiser le modèle de régression logistique
logistic_model = LogisticRegression(max_iter=1000)

# Configurer la Grid Search
grid_search = GridSearchCV(estimator=logistic_model, param_grid=param_grid, scoring='f1', cv=3, verbose=2, n_jobs=-1)

# Lancer la recherche sur les hyperparamètres
print("Démarrage de la Grid Search...")
grid_search.fit(X_train, y_train)

# Meilleurs paramètres et meilleur score
print("Meilleurs paramètres trouvés:", grid_search.best_params_)
print("Meilleur score (F1):", grid_search.best_score_)

# Évaluer sur l'ensemble de test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Test set accuracy:", accuracy)
print("Test set F1 score:", f1)
