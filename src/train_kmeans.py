import pandas as pd
import matplotlib.pyplot as plt
import time

from preprocessing import load_and_preprocess
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==============================
# 1. CARICAMENTO DATI
# ==============================
print("=" * 50)
print(" AVVIO K-MEANS - SOC CLUSTERING ")
print("=" * 50)

_, X_test, _, y_test = load_and_preprocess(use_pca=True)

print("\nDataset caricato correttamente!")
print(f"Test set utilizzato per clustering: {X_test.shape}")

# ==============================
# 2. METODO ELBOW
# ==============================
print("\n--- FASE 1: METODO ELBOW ---")
print("Calcolo inerzia per diversi valori di K...")

KVect = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
dist = []

start_elbow = time.time()

for K in KVect:
    kmeans_temp = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans_temp.fit(X_test)
    dist.append(kmeans_temp.inertia_)

end_elbow = time.time()
elbow_time = end_elbow - start_elbow

print(f"Tempo calcolo Elbow: {elbow_time:.2f} secondi")

# ==============================
# GRAFICO ELBOW
# ==============================
plt.figure(figsize=(8, 5))
plt.plot(KVect, dist, marker='o', color='blue')
plt.title("Metodo Elbow per la stima del numero di Cluster")
plt.xlabel("Numero Cluster (K)")
plt.ylabel("Inerzia (SSE)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# 3. CLUSTERING DEFINITIVO
# ==============================
K_ottimo = 4

print(f"\n--- FASE 2: K-MEANS DEFINITIVO (K={K_ottimo}) ---")

kmeans = KMeans(n_clusters=K_ottimo, random_state=42, n_init=10)

start_train = time.time()
clusters = kmeans.fit_predict(X_test)
end_train = time.time()

train_time = end_train - start_train
print(f"Tempo clustering K-Means: {train_time:.2f} secondi")

centroids = kmeans.cluster_centers_

# ==============================
# 4. SILHOUETTE SCORE
# ==============================
print("\nCalcolo Silhouette Score...")

start_sil = time.time()

sil_score = silhouette_score(
    X_test,
    clusters,
    metric='euclidean',
    sample_size=min(10000, X_test.shape[0]),
    random_state=42
)

end_sil = time.time()
sil_time = end_sil - start_sil

print(f"Silhouette Score: {sil_score:.3f}")
print(f"Tempo calcolo silhouette: {sil_time:.2f} secondi")

# ==============================
# 5. COMPOSIZIONE CLUSTER
# ==============================
print("\n--- COMPOSIZIONE CLUSTER (numerosità) ---")

st_df = pd.DataFrame({
    "Cluster": clusters,
    "True_Label": y_test
})

counts  = st_df["Cluster"].value_counts().sort_index()
percnt  = st_df["Cluster"].value_counts(normalize=True).sort_index() * 100

stats = pd.DataFrame({
    "Conteggio":       counts,
    "Percentuale (%)": percnt
})

print(stats)

# ==============================
# 6. DISTRIBUZIONE TRUE_LABEL PER CLUSTER
# ==============================
print("\n--- DISTRIBUZIONE TRUE_LABEL PER CLUSTER ---")

# Tabella pivot: righe = cluster, colonne = label reali, valori = conteggi
pivot = (
    st_df.groupby(["Cluster", "True_Label"])
    .size()
    .unstack(fill_value=0)
)

# Label dominante per cluster (purezza)
pivot["Label_Dominante"] = pivot.idxmax(axis=1)
pivot["Purezza (%)"]     = (pivot.max(axis=1) / pivot.sum(axis=1) * 100).round(1)

print("\nConteggio campioni per (Cluster × True_Label):")
print(pivot.drop(columns=["Label_Dominante", "Purezza (%)"]).to_string())

print("\nLabel dominante e purezza per cluster:")
print(pivot[["Label_Dominante", "Purezza (%)"]].to_string())

# ==============================
# GRAFICO: HEATMAP CLUSTER vs TRUE_LABEL
# ==============================
print("\nGenerazione heatmap Cluster × True_Label...")

count_matrix = pivot.drop(columns=["Label_Dominante", "Purezza (%)"])

fig, ax = plt.subplots(figsize=(max(10, len(count_matrix.columns) * 1.2), 7))

im = ax.imshow(count_matrix.values, aspect="auto", cmap="YlOrRd")

ax.set_xticks(range(len(count_matrix.columns)))
ax.set_xticklabels(count_matrix.columns, rotation=45, ha="right", fontsize=8)

ax.set_yticks(range(len(count_matrix.index)))
ax.set_yticklabels([f"Cluster {i}" for i in count_matrix.index], fontsize=8)

# Annota ogni cella con il valore numerico
for row_idx in range(count_matrix.shape[0]):
    for col_idx in range(count_matrix.shape[1]):
        val = count_matrix.iloc[row_idx, col_idx]
        if val > 0:
            ax.text(
                col_idx, row_idx, str(val),
                ha="center", va="center",
                fontsize=7, color="black"
            )

plt.colorbar(im, ax=ax, label="Numero campioni")
ax.set_title("Distribuzione True_Label per Cluster (K-Means)")
ax.set_xlabel("True Label")
ax.set_ylabel("Cluster")
plt.tight_layout()
plt.show()

# ==============================
# 7. VISUALIZZAZIONE PCA 2D
# ==============================
print("\nGenerazione visualizzazione cluster PCA 2D...")

X_plot          = X_test[:, :2]
centroids_plot  = centroids[:, :2]

plt.figure(figsize=(9, 6))
colors = plt.colormaps.get_cmap("tab20")

for i in range(K_ottimo):
    cluster_color = colors(i % colors.N)
    plt.scatter(
        X_plot[clusters == i, 0],
        X_plot[clusters == i, 1],
        color=cluster_color,
        label=f"Cluster {i}",
        alpha=0.6,
        s=30
    )

plt.scatter(
    centroids_plot[:, 0],
    centroids_plot[:, 1],
    color='black',
    marker='X',
    s=250,
    edgecolor='white',
    label='Centroidi'
)

plt.title(f"K-Means Clustering - Analisi Traffico SOC (K={K_ottimo})")
plt.xlabel("Componente PCA 1")
plt.ylabel("Componente PCA 2")
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# 8. DISTRIBUZIONE CLUSTER (bar chart)
# ==============================
print("\nGenerazione grafico distribuzione cluster...")

cluster_counts = pd.Series(clusters).value_counts().sort_index()

plt.figure(figsize=(8, 5))
cluster_counts.plot(kind='bar', edgecolor='black')
plt.title("Distribuzione dei Cluster Identificati")
plt.xlabel("Cluster")
plt.ylabel("Numero Campioni")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==============================
# 9. PERFORMANCE COMPUTAZIONALI
# ==============================
print("\n" + "=" * 50)
print(" PERFORMANCE COMPUTAZIONALI ")
print("=" * 50)

performance_df = pd.DataFrame({
    "Algoritmo":            ["K-Means"],
    "Silhouette Score":     [round(sil_score, 3)],
    "Tempo Elbow (s)":      [round(elbow_time, 2)],
    "Tempo Clustering (s)": [round(train_time, 2)],
    "Tempo Silhouette (s)": [round(sil_time, 2)]
})

print(performance_df)