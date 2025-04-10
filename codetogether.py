import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from scipy.sparse import lil_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def similarity_score(row1, row2):
    scores = []
    weights = []

    if pd.notnull(row1['Name']) and pd.notnull(row2['Name']):
        name_sim = fuzz.token_sort_ratio(str(row1['Name']), str(row2['Name'])) / 100.0
        scores.append(name_sim)
        weights.append(0.4)

    if pd.notnull(row1['Address']) and pd.notnull(row2['Address']):
        address_sim = fuzz.token_sort_ratio(str(row1['Address']), str(row2['Address'])) / 100.0
        scores.append(address_sim)
        weights.append(0.2)

    if pd.notnull(row1['Aadhaar/SSN']) and pd.notnull(row2['Aadhaar/SSN']):
        aadhaar_sim = fuzz.ratio(str(row1['Aadhaar/SSN']), str(row2['Aadhaar/SSN'])) / 100.0
        scores.append(aadhaar_sim)
        weights.append(0.3)

    if pd.notnull(row1['Bank Account']) and pd.notnull(row2['Bank Account']):
        bank_sim = 1.0 if row1['Bank Account'] == row2['Bank Account'] else 0.0
        scores.append(bank_sim)
        weights.append(0.1)

    if not scores:
        return 0.0
    return np.average(scores, weights=weights)

def red_flag(row1, row2):
    name_sim = fuzz.token_sort_ratio(str(row1['Name']), str(row2['Name'])) / 100.0
    aadhaar_match = str(row1['Aadhaar/SSN']) == str(row2['Aadhaar/SSN'])
    bank_match = str(row1['Bank Account']) == str(row2['Bank Account'])
    return name_sim > 0.85 and (not aadhaar_match or not bank_match)

def detect_fraud(df):
    df = df.copy()
    df['block_key'] = df['Name'].str[:3].str.lower() + '_' + df['Region'].str.lower()

    similar_pairs = []
    for block_key, group in tqdm(df.groupby('block_key'), desc="Processing blocks"):
        indices = group.index.tolist()
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]
                sim = similarity_score(df.loc[idx_i], df.loc[idx_j])
                if sim > 0.4:
                    similar_pairs.append((idx_i, idx_j, 1 - sim))
                elif red_flag(df.loc[idx_i], df.loc[idx_j]):
                    similar_pairs.append((idx_i, idx_j, 0.1))

    n = len(df)
    dist_matrix = lil_matrix((n, n))
    for i, j, dist in similar_pairs:
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist

    dbscan_model = DBSCAN(eps=0.9, min_samples=2, metric='precomputed')
    df['cluster_dbscan'] = dbscan_model.fit_predict(dist_matrix)

    cluster_features = df[df['cluster_dbscan'] != -1].groupby('cluster_dbscan').agg({
        'Pension Scheme': 'nunique',
        'Amount': 'sum',
        'Name': 'count'
    }).rename(columns={
        'Pension Scheme': 'scheme',
        'Amount': 'amount',
        'Name': 'entry_count'
    }).reset_index()

    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    cluster_features['anomaly'] = iso_forest.fit_predict(cluster_features[['scheme', 'amount', 'entry_count']])

    suspicious_clusters = cluster_features[cluster_features['anomaly'] == -1]

    df['Fraud_Flag'] = 'Legit'
    df.loc[df['cluster_dbscan'].isin(suspicious_clusters['cluster_dbscan']), 'Fraud_Flag'] = 'Suspicious'

    # Save separate legitimate users for convenience
    legit_users = df[df['Fraud_Flag'] != 'Suspicious']
    legit_users.to_csv("legitimate_users.csv", index=False)

    return df, cluster_features