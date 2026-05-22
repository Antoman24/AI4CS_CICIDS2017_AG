# AI-Based Intrusion Detection System using Machine Learning

Progetto di Machine Learning applicato alla cybersecurity per il rilevamento automatico delle intrusioni in ambiente SOC (Security Operation Center).

Il progetto utilizza il dataset [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), sviluppato dal Canadian Institute for Cybersecurity, uno dei benchmark più utilizzati per la valutazione degli Intrusion Detection System (IDS).

---

# Obiettivo

Analizzare e confrontare diversi algoritmi di Machine Learning per il rilevamento di traffico malevolo e anomalie di rete.

## Algoritmi implementati

- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- K-Means Clustering

---

# Dataset

Il dataset CICIDS2017 contiene traffico benigno e diverse tipologie di attacco realistiche, tra cui:

- DoS / DDoS
- Brute Force
- Botnet
- Web Attacks
- Heartbleed
- Port Scanning

Il dataset include feature estratte dal traffico di rete tramite CICFlowMeter.

---

# Pipeline del progetto

1. Data Cleaning  
2. Feature Selection  
3. Standardizzazione  
4. Bilanciamento classi con SMOTE  
5. Training e valutazione modelli  
6. Analisi delle feature importance  
7. Clustering e analisi esplorativa  

---


# Struttura del progetto

```text
preprocessing.py
train_dt.py
train_rf.py
train_knn.py
train_kmeans.py
```

# Risultati

Gli algoritmi supervisionati hanno ottenuto performance molto elevate sul dataset, in particolare **Random Forest**, che ha mostrato il miglior equilibrio tra accuratezza, robustezza e capacità di generalizzazione.

L’analisi ha inoltre evidenziato alcune limitazioni del dataset, tra cui il possibile rischio di **data leakage** e la limitata rappresentatività rispetto a scenari di rete reali. Per questo motivo, i risultati ottenuti devono essere interpretati criticamente, considerando il contesto sperimentale del benchmark CICIDS2017.
