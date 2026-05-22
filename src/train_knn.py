import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from preprocessing import load_and_preprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==============================
# 1. CARICAMENTO DATI
# ==============================
print("=" * 50)
print(" AVVIO KNN - SOC DETECTION ")
print("=" * 50)

# Dataset preprocessato:
# - Cleaning
# - PCA
# - StandardScaler
# - SMOTE
X_train, X_test, y_train, y_test = load_and_preprocess(use_smote=False)

print("\nDataset caricato correttamente!")
print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")


# ==============================
# 3. INIZIALIZZAZIONE MODELLO
# ==============================
print("\nInizializzazione K-Nearest Neighbor...")

knn = KNeighborsClassifier(
    n_neighbors=5,
    metric='euclidean'
)

# ==============================
# 4. TRAINING
# ==============================
print("\nAddestramento KNN in corso...")

start_train = time.time()

knn.fit(X_train, y_train)

end_train = time.time()

train_time = end_train - start_train

print(f"Tempo training KNN: {train_time:.2f} secondi")

# ==============================
# 5. PREDIZIONE
# ==============================
print("\nPredizione in corso...")

start_pred = time.time()

pred = knn.predict(X_test)

end_pred = time.time()

pred_time = end_pred - start_pred

print(f"Tempo prediction KNN: {pred_time:.2f} secondi")

# ==============================
# 6. VALUTAZIONE
# ==============================
print("\n" + "=" * 50)
print(" RISULTATI KNN (BINARIO) ")
print("=" * 50)

accuracy = accuracy_score(y_test, pred)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        pred,
        target_names=["Normale (0)", "Attacco (1)"]
    )
)

# ==============================
# 7. CONFUSION MATRIX
# ==============================
print("\nGenerazione Confusion Matrix...")

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=["Normale (0)", "Attacco (1)"],
    yticklabels=["Normale (0)", "Attacco (1)"]
)

plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()

# ==============================
# 8. SIMULAZIONE SOC
# ==============================
def risk(x):

    if x == 1:
        return "🔴 HIGH (Attacco)"

    else:
        return "🟢 LOW (Normale)"

results = pd.DataFrame({
    "Prediction": pred
})

results["Risk_Level"] = results["Prediction"].apply(risk)

print("\n--- OUTPUT SOC (KNN Engine) ---")
print(results.head(15))

# ==============================
# 9. DISTRIBUZIONE ALERT
# ==============================
print("\nGenerazione grafico alert SOC...")

counts = results["Prediction"].value_counts()

plt.figure(figsize=(8, 5))

colori = [
    '#2ca02c' if label == 0 else '#d62728'
    for label in counts.index
]

counts.plot(
    kind='bar',
    color=colori,
    edgecolor='black'
)

plt.title("Distribuzione Alert SOC - KNN")

plt.xlabel("Classe")
plt.ylabel("Numero di rilevazioni")

plt.xticks(
    ticks=[0, 1],
    labels=["Normale (0)", "Attacco (1)"],
    rotation=0
)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# ==============================
# 10. PERFORMANCE COMPUTAZIONALI
# ==============================
print("\n" + "=" * 50)
print(" PERFORMANCE COMPUTAZIONALI ")
print("=" * 50)

performance_df = pd.DataFrame({
    "Algoritmo": ["K-Nearest Neighbor"],
    "Accuracy": [round(accuracy, 4)],
    "Tempo Training (s)": [round(train_time, 2)],
    "Tempo Prediction (s)": [round(pred_time, 2)]
})

print(performance_df)
