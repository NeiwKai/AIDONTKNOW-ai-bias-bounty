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

if 'ID' in data.columns:
    data = data.drop(columns=['ID'])

# Replace string values with integers
binary_cols = ['Disability_Status', 'Criminal_Record']
data[binary_cols] = data[binary_cols].replace({'No': 0, 'Yes': 1})
data['Gender'] = data['Gender'].replace({'Male': 1, 'Female': 0, 'Non-binary': 2}).astype(int)
data['Race'] = data['Race'].replace({'White': 0, 'Black': 1, 'Hispanic': 2, 'Multiracial': 3, 'Native American': 4, 'Asian': 5}).astype(int)
data['Loan_Approved'] = data['Loan_Approved'].replace({'Approved': 1, 'Denied': 0}).astype(int)
data['Employment_Type'] = data['Employment_Type'].replace({'Unemployed': 0, 'Part-time': 1, 'Gig': 2, 'Self-employed': 3, 'Full-time': 4}).astype(int)
data['Education_Level'] = data['Education_Level'].replace({'High School': 0, 'Some College': 1, "Bachelor's": 2, 'Graduate': 3}).astype(int)

# accept only numeric data
data = data.select_dtypes(include=[np.number])

print(data.columns)
print(data.head())

# define Privilege
data['Gender_Male_priv'] = (data['Gender'] == 1).astype(int)
data['Gender_Female_priv'] = (data['Gender'] == 0).astype(int)
data['Gender_Non-binary_priv'] = (data['Gender'] == 2).astype(int) 
data['Race_White_priv'] = (data['Race'] == 0).astype(int)
data['Race_Black_priv'] = (data['Race'] == 1).astype(int)
data['Race_Asian_priv'] = (data['Race'] == 5).astype(int)
data['Race_Native-American_priv'] = (data['Race'] == 4).astype(int)

# Bias against Black
privileged_groups_black = [{'Race_Black_priv': 0}]
unprivileged_grpup_black = [{'Race_Black_priv': 1}]
alf_dataset_black = BinaryLabelDataset(
    df=data,
    label_names=['Loan_Approved'],
    protected_attribute_names=['Race_Black_priv'],
    favorable_label=1,
    unfavorable_label=0
)
metric_black = BinaryLabelDatasetMetric(alf_dataset_black,
                                        privileged_groups=privileged_groups_black,
                                        unprivileged_groups=unprivileged_grpup_black)
print("Bias against Black")
print("Disparate Impact", metric_black.disparate_impact())
print("Statistical Parity Difference:", metric_black.statistical_parity_difference())
print()

# Bias against Asian
privileged_groups_asian = [{'Race_Asian_priv': 0}]
unprivileged_grpup_asian = [{'Race_Asian_priv': 1}]
alf_dataset_asian = BinaryLabelDataset(
    df=data,
    label_names=['Loan_Approved'],
    protected_attribute_names=['Race_Asian_priv'],
    favorable_label=1,
    unfavorable_label=0
)
metric_asian = BinaryLabelDatasetMetric(alf_dataset_asian,
                                        privileged_groups=privileged_groups_asian,
                                        unprivileged_groups=unprivileged_grpup_asian)
print("Bias against Asian")
print("Disparate Impact", metric_asian.disparate_impact())
print("Statistical Parity Difference:", metric_asian.statistical_parity_difference())
print()
