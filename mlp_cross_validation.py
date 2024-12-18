import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
from itertools import product

# Chemins vers les fichiers
features_path = "features_output/features_glove.pt"
labels_path = "features_output/labels_glove.npy"  # Chemin vers le fichier contenant les labels

# Charger les features depuis le fichier .pt
features = torch.load(features_path).cpu().numpy()

# Charger les labels depuis le fichier .npy
labels = np.load(labels_path)

# Vérifier que le nombre de features correspond au nombre de labels
assert features.shape[0] == labels.shape[0], "Le nombre de features et de labels ne correspond pas."

# Normalisation des données
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Définir un modèle MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Couche cachée
            nn.ReLU(),  # Fonction d'activation ReLU
            nn.Linear(hidden_dim, output_dim),  # Couche de sortie
            nn.Sigmoid()  # Fonction d'activation pour la classification binaire
        )

    def forward(self, x):
        return self.model(x)

# Hyperparamètres pour la Grid Search
hidden_layer_sizes = [50, 100, 200]  # Tailles des couches cachées
learning_rates = [0.00001, 0.001, 0.1]  # Taux d'apprentissage
epoch_values = [100]  # Nombre d'époques

# Initialiser le meilleur score et les meilleurs paramètres
best_f1 = 0
best_params = {}

# Grid Search avec validation croisée
kf = KFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for hidden_dim, lr, epochs in product(hidden_layer_sizes, learning_rates, epoch_values):
    print(f"Testing configuration: Hidden Layer Size={hidden_dim}, Learning Rate={lr}, Epochs={epochs}")

    fold_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
        print(f"Fold {fold + 1} / 5")

        # Diviser les données en ensemble d'entraînement et de validation
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Convertir les données en tenseurs PyTorch
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

        # Initialiser le modèle, la fonction de perte et l'optimiseur
        input_dim = X_train.shape[1]
        output_dim = 1   # Sortie binaire (sigmoïde)

        model = MLP(input_dim, hidden_dim, output_dim).to(device)
        criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Entraînement du modèle
        for epoch in tqdm(range(epochs), desc=f"Training Fold {fold + 1} (Hidden={hidden_dim}, LR={lr}, Epochs={epochs})", unit="epoch"):
            model.train()
            optimizer.zero_grad()

            # Prédiction et calcul de la perte
            outputs = model(X_train_tensor).squeeze()
            loss = criterion(outputs, (y_train_tensor + 1) / 2)  # Transformer -1/+1 en 0/1

            # Backpropagation et optimisation
            loss.backward()
            optimizer.step()

        # Évaluation du modèle sur l'ensemble de validation
        model.eval()
        with torch.no_grad():
            y_val_pred_proba = model(X_val_tensor).squeeze()
            y_val_pred = torch.where(y_val_pred_proba >= 0.5, 1.0, -1.0)  # Seuil à 0.5 pour binariser les prédictions

        # Conversion en NumPy pour sklearn
        y_val_np = y_val_tensor.cpu().numpy()
        y_val_pred_np = y_val_pred.cpu().numpy()

        # Calcul du F1-score pour ce pli
        fold_f1 = f1_score(y_val_np, y_val_pred_np)
        fold_f1_scores.append(fold_f1)

    # Calcul du F1-score moyen pour les 5 plis
    mean_f1 = np.mean(fold_f1_scores)
    print(f"Configuration: Hidden={hidden_dim}, LR={lr}, Epochs={epochs} | Mean F1={mean_f1:.4f}")

    # Mise à jour des meilleurs paramètres
    if mean_f1 > best_f1:
        best_f1 = mean_f1
        best_params = {
            "hidden_dim": hidden_dim,
            "learning_rate": lr,
            "epochs": epochs
        }

print("Best F1 Score:", best_f1)
print("Best Parameters:", best_params)
