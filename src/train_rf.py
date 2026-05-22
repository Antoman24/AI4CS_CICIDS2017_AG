import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from preprocessing import load_and_preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# 1. CARICAMENTO DATI
# ==============================
print("=" * 50)
print(" AVVIO RANDOM FOREST - SOC DETECTION ")
print("=" * 50)


X_train, X_test, y_train, y_test = load_and_preprocess(use_pca=False)

print("\nDataset caricato correttamente!")
print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ==============================
# 2. ADDESTRAMENTO MODELLO
# ==============================
print("\nAddestramento Random Forest (Classificazione Binaria)...")

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# ==============================
# TIMER TRAINING
# ==============================
start_train = time.time()

rf.fit(X_train, y_train)

end_train = time.time()

train_time = end_train - start_train

print(f"\nTempo training RF: {train_time:.2f} secondi")

# ==============================
# 3. PREDIZIONE
# ==============================
print("\nPredizione in corso...")

# ==============================
# TIMER PREDICTION
# ==============================
start_pred = time.time()

pred = rf.predict(X_test)

end_pred = time.time()

pred_time = end_pred - start_pred

print(f"Tempo prediction RF: {pred_time:.2f} secondi")

# ==============================
# 4. VALUTAZIONE BINARIA
# ==============================
print("\n" + "=" * 50)
print(" RISULTATI RANDOM FOREST (BINARIO) ")
print("=" * 50)

accuracy = accuracy_score(y_test, pred)

print(f"\nAccuracy globale: {accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred,
        target_names=["Normale (0)", "Attacco (1)"]
    )
)

# ==============================
# 5. MATRICE DI CONFUSIONE
# ==============================
print("\nGenerazione della Matrice di Confusione...")

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=["Normale (0)", "Attacco (1)"],
    yticklabels=["Normale (0)", "Attacco (1)"]
)

plt.title("Matrice di Confusione - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()

# ==============================
# 6. FEATURE IMPORTANCE
# ==============================
print("\nAnalisi Feature Importance...")

importances = pd.Series(rf.feature_importances_, index=X_train.columns)

top_features = importances.nlargest(10)

plt.figure(figsize=(8, 5))

top_features.sort_values().plot(
    kind='barh',
    color='darkred'
)

plt.title("Top 10 Feature Importance")
plt.xlabel("Importanza")

plt.tight_layout()
plt.show()

# ==============================
# 7. SIMULAZIONE SOC
# ==============================
def valuta_rischio(label):

    # 0 = traffico normale
    if label == 0:
        return "🟢 LOW (Normale)"

    # 1 = attacco rilevato
    else:
        return "🔴 HIGH (Attacco Rilevato!)"

results = pd.DataFrame({
    "Valore Reale": y_test,
    "Predizione": pred
})

results["Risk_Level"] = results["Predizione"].apply(
    valuta_rischio
)

print("\n--- OUTPUT CRUSCOTTO SOC ---")
print(results.head(15))

# ==============================
# 8. DISTRIBUZIONE ALERT
# ==============================
print("\nGenerazione grafico alert SOC...")

counts = results["Predizione"].value_counts()

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

plt.title("Distribuzione Alert SOC")
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
# 9. REPORT PERFORMANCE
# ==============================
print("\n" + "=" * 50)
print(" PERFORMANCE COMPUTAZIONALI ")
print("=" * 50)

performance_df = pd.DataFrame({
    "Algoritmo": ["Random Forest"],
    
    "Accuracy": [round(accuracy, 4)],
    "Tempo Training (s)": [round(train_time, 2)],
    "Tempo Prediction (s)": [round(pred_time, 2)]
})

print(performance_df)

print("\nPipeline completata con successo!")