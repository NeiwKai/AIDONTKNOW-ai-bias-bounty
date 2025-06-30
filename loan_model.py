import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

data = pd.read_csv("datasets/loan_access_dataset.csv")
binary_cols = ['Disability_Status', 'Criminal_Record']
if 'ID' in data.columns:
    data = data.drop(columns=['ID'])
data[binary_cols] = data[binary_cols].replace({'No': 0, 'Yes': 1})
# encode the data that does not numeric
categorical_cols = data.select_dtypes(include=['object', 'category']).columns
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
data_encoded = data_encoded.astype(int)
'''
print(data_encoded.info())
print(data_encoded.head())
'''
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_encoded)

# model initialize
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_scaled)

anomaly_labels = model.predict(X_scaled)

data['Anomaly'] = anomaly_labels

print(data['Anomaly'].value_counts())

print(data[data['Anomaly'] == -1].head())  # Top anomalous samples

# Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=anomaly_labels, cmap='coolwarm', s=50)
plt.title("Anomaly Detection with Isolation Forest")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

