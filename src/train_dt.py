import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from preprocessing import load_and_preprocess

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==============================
# 1. CARICAMENTO DATI
# ==============================
print("=" * 50)
print(" AVVIO DECISION TREE - SOC DETECTION ")
print("=" * 50)

X_train, X_test, y_train, y_test = load_and_preprocess(use_pca=False, use_smote = False)

print("\nDataset caricato correttamente!")
print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ==============================
# 2. INIZIALIZZAZIONE MODELLO
# ==============================
print("\nInizializzazione Decision Tree...")

dt = DecisionTreeClassifier(
    max_depth=10,    
    min_samples_split=5, 
    random_state=42
)

# ==============================
# 3. TRAINING
# ==============================
print("\nAddestramento Decision Tree in corso...")

start_train = time.time()

dt.fit(X_train, y_train)

end_train = time.time()

train_time = end_train - start_train

print(f"Tempo training Decision Tree: {train_time:.2f} secondi")

# ==============================
# 4. PREDIZIONE
# ==============================
print("\nPredizione in corso...")

start_pred = time.time()

pred = dt.predict(X_test)

end_pred = time.time()

pred_time = end_pred - start_pred

print(f"Tempo prediction Decision Tree: {pred_time:.2f} secondi")

# ==============================
# 5. VALUTAZIONE
# ==============================
print("\n" + "=" * 50)
print(" RISULTATI DECISION TREE ")
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
# 6. CONFUSION MATRIX
# ==============================
print("\nGenerazione Confusion Matrix...")

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Normale (0)", "Attacco (1)"],
    yticklabels=["Normale (0)", "Attacco (1)"]
)

plt.title("Confusion Matrix - Decision Tree")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()

# ==============================
# 7. SIMULAZIONE SOC
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

print("\n--- OUTPUT SOC (Decision Tree Engine) ---")
print(results.head(15))

# ==============================
# 8. DISTRIBUZIONE ALERT
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

plt.title("Distribuzione Alert SOC - Decision Tree")

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
# 9. FEATURE IMPORTANCE
# ==============================
print("\nAnalisi Feature Importance...")

features_idx = None
if hasattr(X_train, 'columns'):
    features_idx = X_train.columns
else:
    features_idx = [f"Feature_{i}" for i in range(len(dt.feature_importances_))]

importances = pd.Series(dt.feature_importances_, index=features_idx)

top_features = importances.nlargest(10)

plt.figure(figsize=(8, 5))

top_features.sort_values().plot(
    kind='barh',
    color='darkgreen'
)

plt.title("Top 10 Feature Importance - Decision Tree")
plt.xlabel("Importanza")

plt.tight_layout()
plt.show()

# ==============================
# 10. VISUALIZZAZIONE ALBERO
# ==============================
print("\nGenerazione albero di decisione...")

plt.figure(figsize=(16, 8))

plot_tree(
    dt,
    max_depth=3,  # solo primi livelli per leggibilità
    class_names=["Normale", "Attacco"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title(
    "Rappresentazione Logica Decision Tree (Primi 3 Livelli)"
)

plt.tight_layout()
plt.show()

# ==============================
# 11. PERFORMANCE COMPUTAZIONALI
# ==============================
print("\n" + "=" * 50)
print(" PERFORMANCE COMPUTAZIONALI ")
print("=" * 50)

performance_df = pd.DataFrame({
    "Algoritmo": ["Decision Tree"],
    "Accuracy": [round(accuracy, 4)],
    "Tempo Training (s)": [round(train_time, 2)],
    "Tempo Prediction (s)": [round(pred_time, 2)]
})

print(performance_df)

