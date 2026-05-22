import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


def load_and_preprocess(use_pca: bool = True, sample_frac: float = 1, use_smote: bool = True):

    # ==============================
    # 1. CARICAMENTO
    # ==============================
    print("1. Caricamento dati in corso...")
    paths = [
        'data/Monday-WorkingHours.pcap_ISCX.csv',
        'data/Tuesday-WorkingHours.pcap_ISCX.csv',
        'data/Wednesday-workingHours.pcap_ISCX.csv',
        'data/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'data/Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
    ]

    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    # ==============================
    # 2. SAMPLING
    # ==============================
    df = df.sample(frac=sample_frac, random_state=42)
    print(f"Shape dopo sampling: {df.shape}")

    # ==============================
    # 3. NORMALIZZAZIONE COLONNE
    # ==============================
    df.columns = df.columns.str.strip()

    # Rinomina la colonna label in modo robusto
    for col in df.columns:
        if col.lower() == 'label':
            df.rename(columns={col: 'Label'}, inplace=True)
            break

    # ==============================
    # 4. RIMOZIONE COLONNE IDENTIFICATORE
    # ==============================
    print("2. Rimozione colonne identificatore (IP, porta, timestamp)...")
    id_keywords = [ 'Flow ID', 'Source IP', 'Destination IP', 'Source Port', 'Timestamp']
    cols_to_drop_id = [
        c for c in df.columns
        if any(kw in c.lower() for kw in id_keywords) and c != 'Label'
    ]
    if cols_to_drop_id:
        print(f"  Colonne identificatore rimosse: {cols_to_drop_id}")
        df.drop(columns=cols_to_drop_id, inplace=True, errors='ignore')
    else:
        print("  Nessuna colonna identificatore trovata.")

    # ==============================
    # 5. LABELING
    # ==============================
    print("3. Mappatura etichette...")
    attack_map = {
        'BENIGN': 'BENIGN',
        'DDoS': 'DDoS',
        'DoS Hulk': 'DoS',
        'DoS GoldenEye': 'DoS',
        'DoS slowloris': 'DoS',
        'DoS Slowhttptest': 'DoS',
        'PortScan': 'Port Scan',
        'Web Attack – Brute Force': 'Web Attack',
        'Web Attack – XSS': 'Web Attack',
        'Web Attack – Sql Injection': 'Web Attack',
        'Web Attack - Brute Force': 'Web Attack',
        'Web Attack - XSS': 'Web Attack',
        'Web Attack - Sql Injection': 'Web Attack'
    }

    df['Attack_Type'] = df['Label'].map(attack_map).fillna('Other Attack')
    df['Label_Binary'] = np.where(df['Attack_Type'] == 'BENIGN', 0, 1)

    # ==============================
    # 6. RIMOZIONE DUPLICATI
    # ==============================
    print("4. Rimozione duplicati...")
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Duplicati rimossi: {n_before - len(df)} | Shape: {df.shape}")

    # ==============================
    # 7. SOSTITUZIONE INFINITI CON NaN
    # ==============================
    print("5. Sostituzione valori infiniti con NaN...")
    numeric_cols = df.select_dtypes(include=np.number).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    print(f"  Valori infiniti trovati: {inf_count}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ==============================
    # 8. FEATURE / TARGET
    # ==============================
    X = df.drop(columns=['Label', 'Attack_Type', 'Label_Binary'])
    y = df['Label_Binary']

    # Mantieni solo colonne numeriche
    X = X.select_dtypes(include=np.number)

    # ==============================
    # 9. RIMOZIONE COLONNE A VARIANZA ZERO
    # ==============================
    num_unique = X.nunique()
    zero_var_cols = num_unique[num_unique <= 1].index.tolist()
    if zero_var_cols:
        print(f"  Colonne a varianza zero rimosse: {zero_var_cols}")
        X.drop(columns=zero_var_cols, inplace=True)

    # ==============================
    # 10. TRAIN / TEST SPLIT
    # ==============================
    print("6. Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    # Sanity check: nessun overlap tra indici
    overlap = len(set(X_train.index) & set(X_test.index))
    print(f"  Overlap train/test (deve essere 0): {overlap}")

    # ==============================
    # 11. FILLNA CON MEDIANA DEL TRAIN
    # ==============================
    print("7. Imputazione NaN con mediana del train...")
    train_median = X_train.median(numeric_only=True)
    missing_before = X_train.isna().sum().sum() + X_test.isna().sum().sum()
    X_train = X_train.fillna(train_median)
    X_test  = X_test.fillna(train_median)   # usa mediana del TRAIN, non del test
    print(f"  Valori mancanti imputati: {missing_before}")
    print(f"  Valori mancanti residui: {X_train.isna().sum().sum() + X_test.isna().sum().sum()}")

    # ==============================
    # 12. CORRELATION FILTER
    # ==============================
    print("8. Filtraggio feature altamente correlate (soglia 0.98)...")
    corr_matrix = X_train.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.98)]
    X_train = X_train.drop(columns=to_drop_corr)
    X_test  = X_test.drop(columns=to_drop_corr)
    print(f"  Feature rimosse per alta correlazione: {len(to_drop_corr)}")
    print(f"  Feature rimanenti: {X_train.shape[1]}")

    # ==============================
    # 13. SCALING
    # ==============================
    print("9. Standardizzazione (fit solo su train)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ==============================
    # 14. PCA
    # ==============================
    if use_pca:
        print("10. PCA dimensionality reduction (enabled)...")
        pca = PCA(n_components=0.90, random_state=42)
        X_train_trans = pca.fit_transform(X_train_scaled)
        X_test_trans  = pca.transform(X_test_scaled)
        print(f"  Feature ridotte da PCA: {X_train_scaled.shape[1]} → {X_train_trans.shape[1]}")
    else:
        print("10. PCA disabilitata - uso feature scalate.")
        X_train_trans = X_train_scaled
        X_test_trans  = X_test_scaled

    # ==============================
    # 15. SMOTE
    # ==============================
    if use_smote:
        print("11. Applicazione SMOTE sul train...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_trans, y_train)
        print(f"  Prima SMOTE  → {pd.Series(y_train).value_counts().to_dict()}")
        print(f"  Dopo  SMOTE  → {pd.Series(y_train_res).value_counts().to_dict()}")
    else:
        print("11. SMOTE disattivato.")
        X_train_res, y_train_res = X_train_trans, y_train

    # ==============================
    # 16. RIPRISTINO NOMI COLONNE (se PCA disabilitata)
    # ==============================
    if not use_pca:
        feature_names = X_train.columns.tolist()
        X_train_res  = pd.DataFrame(X_train_res,  columns=feature_names)
        X_test_trans = pd.DataFrame(X_test_trans, columns=feature_names)

    print("\nPreprocessing completato!")
    print(f"  X_train finale: {X_train_res.shape if hasattr(X_train_res, 'shape') else 'N/A'}")
    print(f"  X_test  finale: {X_test_trans.shape if hasattr(X_test_trans, 'shape') else 'N/A'}")

    return X_train_res, X_test_trans, y_train_res, y_test