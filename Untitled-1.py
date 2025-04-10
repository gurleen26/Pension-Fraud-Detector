# %%

import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from scipy.sparse import lil_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("processed_pension_dataset.csv")

# Blocking: reduce comparisons
df['block_key'] = df['Name'].str[:3].str.lower() + '_' + df['Region'].str.lower()

# Updated dynamic similarity scoring function
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

# Heuristic rule-based red flag
def red_flag(row1, row2):
    name_sim = fuzz.token_sort_ratio(str(row1['Name']), str(row2['Name'])) / 100.0
    aadhaar_match = str(row1['Aadhaar/SSN']) == str(row2['Aadhaar/SSN'])
    bank_match = str(row1['Bank Account']) == str(row2['Bank Account'])
    return name_sim > 0.85 and (not aadhaar_match or not bank_match)

# Generate pairwise distances
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
                similar_pairs.append((idx_i, idx_j, 0.1))  # Force similarity if red flag is triggered

# Sparse matrix for DBSCAN
n = len(df)
dist_matrix = lil_matrix((n, n))
for i, j, dist in similar_pairs:
    dist_matrix[i, j] = dist
    dist_matrix[j, i] = dist

# DBSCAN clustering
dbscan_model = DBSCAN(eps=0.9, min_samples=2, metric='precomputed')
df['cluster_dbscan'] = dbscan_model.fit_predict(dist_matrix)

# Summary of clusters
cluster_counts = df['cluster_dbscan'].value_counts()
print(f"\nUnique clusters (DBSCAN): {len(cluster_counts)}\n{cluster_counts}")

# Grouped cluster-level features
cluster_features = df[df['cluster_dbscan'] != -1].groupby('cluster_dbscan').agg({
    'Pension Scheme': 'nunique',
    'Amount': 'sum',
    'Name': 'count'
}).rename(columns={
    'Pension Scheme': 'scheme',
    'Amount': 'amount',
    'Name': 'entry_count'
}).reset_index()

# Isolation Forest to find anomalous clusters
iso_forest = IsolationForest(contamination=0.1, random_state=42)
cluster_features['anomaly'] = iso_forest.fit_predict(cluster_features[['scheme', 'amount', 'entry_count']])

# Get suspicious clusters
suspicious_clusters = cluster_features[cluster_features['anomaly'] == -1]
print("\nSuspicious clusters:\n", suspicious_clusters)
print(f"\nTotal suspicious clusters: {len(suspicious_clusters)}")

# Flag suspicious rows
df['Fraud_Flag'] = 'Legit'
df.loc[df['cluster_dbscan'].isin(suspicious_clusters['cluster_dbscan']), 'Fraud_Flag'] = 'Suspicious'

# Save results
df.to_csv("clustered_pension_dataset.csv", index=False)
df[df['Fraud_Flag'] == 'Suspicious'].to_csv("flagged_cases.csv", index=False)
print("✅ Data saved: 'clustered_pension_dataset.csv' and 'flagged_cases.csv'")


# %%
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=suspicious_clusters,
    x='entry_count',
    y='amount',
    hue='scheme',
    palette='tab10',
    s=100,
    alpha=0.8
)
plt.title("Suspicious Cluster Stats", fontsize=14)
plt.xlabel("Entry Count")
plt.ylabel("Total Amount")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Example: Weighted score (higher = more suspicious)
# Risk score
suspicious_clusters['risk_score'] = (
    0.5 * suspicious_clusters['amount'].rank(pct=True) +
    0.3 * suspicious_clusters['entry_count'].rank(pct=True) +
    0.2 * suspicious_clusters['scheme'].rank(pct=True)
)
suspicious_clusters.to_excel("suspicious_clusters_report.xlsx", index=False)


# %%

suspicious_clusters.to_pickle("suspicious_clusters.pkl")
df[df['Fraud_Flag'] == 'Suspicious'].to_pickle("flagged_individuals.pkl")


