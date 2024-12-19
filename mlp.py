import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
from itertools import product

# Chemins vers les fichiers
features_path = "features_output/features_distilbert.pt"
labels_path = "features_output/labels_distilbert.npy"  # Chemin vers le fichier contenant les labels

# Charger les features depuis le fichier .pt
features_distilbert = torch.load(features_path).cpu().numpy()

# Charger les labels depuis le fichier .npy
labels = np.load(labels_path)

# Vérifier que le nombre de features correspond au nombre de labels
assert features_distilbert.shape[0] == labels.shape[0], "Le nombre de features et de labels ne correspond pas."

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features_distilbert, labels, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Conversion des données en tenseurs PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

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

# Grid Search
for hidden_dim, lr, epochs in product(hidden_layer_sizes, learning_rates, epoch_values):
    print(f"Testing configuration: Hidden Layer Size={hidden_dim}, Learning Rate={lr}, Epochs={epochs}")

    # Initialiser le modèle, la fonction de perte et l'optimiseur
    input_dim = X_train.shape[1]
    output_dim = 1   # Sortie binaire (sigmoïde)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Déplacer les données sur le device
    X_train_device = X_train.to(device)
    X_test_device = X_test.to(device)
    y_train_device = y_train.to(device)
    y_test_device = y_test.to(device)

    # Entraînement du modèle
    for epoch in tqdm(range(epochs), desc=f"Training Progress (Hidden={hidden_dim}, LR={lr}, Epochs={epochs})", unit="epoch"):
        model.train()
        optimizer.zero_grad()

        # Prédiction et calcul de la perte
        outputs = model(X_train_device).squeeze()
        loss = criterion(outputs, (y_train_device + 1) / 2)  # Transformer -1/+1 en 0/1

        # Backpropagation et optimisation
        loss.backward()
        optimizer.step()

    # Évaluation du modèle
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_test_device).squeeze()
        y_pred = torch.where(y_pred_proba >= 0.5, 1.0, -1.0)  # Seuil à 0.5 pour binariser les prédictions

    # Conversion en NumPy pour sklearn
    y_test_np = y_test_device.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    # Évaluation avec accuracy et F1-score
    accuracy = accuracy_score(y_test_np, y_pred_np)
    f1 = f1_score(y_test_np, y_pred_np)

    print(f"Configuration: Hidden={hidden_dim}, LR={lr}, Epochs={epochs} | Accuracy={accuracy:.4f}, F1={f1:.4f}")

    # Mise à jour des meilleurs paramètres
    if f1 > best_f1:
        best_f1 = f1
        best_params = {
            "hidden_dim": hidden_dim,
            "learning_rate": lr,
            "epochs": epochs
        }

print("Best F1 Score:", best_f1)
print("Best Parameters:", best_params)
