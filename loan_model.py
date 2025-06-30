import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

data = pd.read_csv("datasets/loan_access_dataset.csv")
binary_cols = ['Disability_Status', 'Criminal_Record']
if 'ID' in data.columns:
    data = data.drop(columns=['ID'])
data[binary_cols] = data[binary_cols].replace({'No': 0, 'Yes': 1})
# Replace string values with integers
data['Gender'] = data['Gender'].replace({'Male': 1, 'Female': 0, 'Non-binary': 2}).astype(int)
data['Race'] = data['Race'].replace({'White': 0, 'Black': 1, 'Hispanic': 2, 'Multiracial': 3, 'Native American': 4, 'Asian': 5}).astype(int)

data = data.select_dtypes(include=[np.number])

print(data.columns)
print(data.head())

'''
# encode the data that does not numeric
categorical_cols = data.select_dtypes(include=['object', 'category']).columns
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
data_encoded = data_encoded.astype(int)

print(data_encoded.info())
#print(data_encoded.head())
'''

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# model initialize
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_scaled)

anomaly_labels = model.predict(X_scaled)

data['Anomaly'] = anomaly_labels

print(data['Anomaly'].value_counts())

print(data[data['Anomaly'] == -1].head())  # Top anomalous samples

num_non_criminal_anomalies = data[(data['Criminal_Record'] == 0) & (data['Anomaly'] == -1)].shape[0]
print(f"Number of anomalous applicants without a criminal record: {num_non_criminal_anomalies}")

num_non_criminal = data[(data['Criminal_Record'] == 0) & (data['Anomaly'] == 0)].shape[0]
print(f"Number of non anomalous applicants without a criminal record: {num_non_criminal}")

num_non_disability_anomalies = data[(data['Disability_Status'] == 0) & (data['Anomaly'] == -1)].shape[0]
print(f"Number of anomalous applicants without disability: {num_non_disability_anomalies}")

num_non_disability = data[(data['Disability_Status'] == 0) & (data['Anomaly'] == 0)].shape[0]
print(f"Number of non anomalous applicants without disability: {num_non_disability}")

'''
# Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=anomaly_labels, cmap='coolwarm', s=50)
plt.title("Anomaly Detection with Isolation Forest")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
'''
'''
df_aif = data.copy()

protected_attribute_names = ['Gender', 'Race']
label_name = 'Anomaly'

aif_dataset = BinaryLabelDataset(
    df=df_aif,
    label_names=[label_name],
    protected_attribute_names=protected_attribute_names
)

# Analyze fairness with respect to 'Gender'
metric = BinaryLabelDatasetMetric(aif_dataset, privileged_groups=[{'Gender': 1}], unprivileged_groups=[{'Gender': 0}])

print("Disparate Impact:", metric.disparate_impact())
print("Statistical Parity Difference:", metric.statistical_parity_difference())
'''

data['Gender_Male_priv'] = (data['Gender'] == 1).astype(int)  # Male as privileged group
data['Race_White_priv'] = (data['Race'] == 0).astype(int)  # White as privileged group

aif_dataset = BinaryLabelDataset(
    df=data,
    label_names=['Anomaly'],
    protected_attribute_names=['Gender_Male_priv', 'Race_White_priv'],
    favorable_label=1,
    unfavorable_label=-1  # Note: your anomaly labels are 1 and -1 from IsolationForest
)

aif_dataset.privileged_protected_attributes = np.array([[1]])

privileged_groups = [{'Gender_Male_priv': 1, 'Race_White_priv': 0}]
unprivileged_groups = [{'Gender_Male_priv': 0, 'Race_White_priv': 1}]

metric = BinaryLabelDatasetMetric(aif_dataset, 
                                  privileged_groups=privileged_groups, 
                                  unprivileged_groups=unprivileged_groups)

print("Disparate Impact:", metric.disparate_impact())
print("Statistical Parity Difference:", metric.statistical_parity_difference())
