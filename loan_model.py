import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

data = pd.read_csv("datasets/loan_access_dataset.csv")

if 'ID' in data.columns:
    data = data.drop(columns=['ID'])

# Replace string values with integer
binary_cols = ['Disability_Status', 'Criminal_Record']
data[binary_cols] = data[binary_cols].replace({'No': 0, 'Yes': 1})
data['Gender'] = data['Gender'].replace({'Male': 1, 'Female': 0, 'Non-binary': 2}).astype(int)
data['Race'] = data['Race'].replace({'White': 0, 'Black': 1, 'Hispanic': 2, 'Multiracial': 3, 'Native American': 4, 'Asian': 5}).astype(int)
data['Loan_Approved'] = data['Loan_Approved'].replace({'Approved': 1, 'Denied': 0}).astype(int)
data['Employment_Type'] = data['Employment_Type'].replace({'Unemployed': 0, 'Part-time': 1, 'Gig': 2, 'Self-employed': 3, 'Full-time': 4}).astype(int)
data['Education_Level'] = data['Education_Level'].replace({'High School': 0, 'Some College': 1, "Bachelor's": 2, 'Graduate': 3}).astype(int)
data['Zip_Code_Group'] = data['Zip_Code_Group'].replace({'Historically Redlined': 0, 'Rural': 1, 'Urban Professional': 2,
                                                         'Working Class Urban': 3, 'High-income Suburban': 4}).astype(int)
data['Citizenship_Status'] = data['Citizenship_Status'].replace({'Visa Holder': 0, 'Permanent Resident': 1, 'Citizen': 2}).astype(int)
data['Age_Group'] = data['Age_Group'].replace({'Over 60': 0, '25-60': 1, 'Under 25': 2}).astype(int)
data['Language_Proficiency'] = data['Language_Proficiency'].replace({'Limited': 0, 'Fluent': 1}).astype(int)

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
data['EmploymentType_Unemployed_priv'] = (data['Gender'] == 0).astype(int)

# Bias against Black
privileged_groups_black = [{'Race_Black_priv': 0}]
unprivileged_groups_black = [{'Race_Black_priv': 1}]
alf_dataset_black = BinaryLabelDataset(
    df=data,
    label_names=['Loan_Approved'],
    protected_attribute_names=['Race_Black_priv'],
    favorable_label=1,
    unfavorable_label=0
)
metric_black = BinaryLabelDatasetMetric(alf_dataset_black,
                                        privileged_groups=privileged_groups_black,
                                        unprivileged_groups=unprivileged_groups_black)
print("Bias against Black")
print("Disparate Impact:", metric_black.disparate_impact())
print("Statistical Parity Difference:", metric_black.statistical_parity_difference())
print()

# Bias against Asian
privileged_groups_asian = [{'Race_Asian_priv': 0}]
unprivileged_groups_asian = [{'Race_Asian_priv': 1}]
alf_dataset_asian = BinaryLabelDataset(
    df=data,
    label_names=['Loan_Approved'],
    protected_attribute_names=['Race_Asian_priv'],
    favorable_label=1,
    unfavorable_label=0
)
metric_asian = BinaryLabelDatasetMetric(alf_dataset_asian,
                                        privileged_groups=privileged_groups_asian,
                                        unprivileged_groups=unprivileged_groups_asian)
print("Bias against Asian")
print("Disparate Impact:", metric_asian.disparate_impact())
print("Statistical Parity Difference:", metric_asian.statistical_parity_difference())
print()

# Bias against Non-binary
privileged_groups_nonbinary = [{'Gender_Non-binary_priv': 0}]
unprivileged_groups_nonbinary = [{'Gender_Non-binary_priv': 1}]
alf_dataset_nonbinary = BinaryLabelDataset(
    df=data,
    label_names=['Loan_Approved'],
    protected_attribute_names=['Gender_Non-binary_priv'],
    favorable_label=1,
    unfavorable_label=0
)
metric_nonbinary = BinaryLabelDatasetMetric(alf_dataset_nonbinary,
                                            privileged_groups=privileged_groups_nonbinary,
                                            unprivileged_groups=unprivileged_groups_nonbinary)
print("Bias against Non-binary")
print("Disparate Impact:", metric_nonbinary.disparate_impact())
print("Statistical Parity Difference:", metric_nonbinary.statistical_parity_difference())
print()

# Bias against Female
privileged_groups_female = [{'Gender_Female_priv': 0}]
unprivileged_groups_female = [{'Gender_Female_priv': 1}]
alf_dataset_female = BinaryLabelDataset(
    df=data,
    label_names=['Loan_Approved'],
    protected_attribute_names=['Gender_Female_priv'],
    favorable_label=1,
    unfavorable_label=0
)
metric_female = BinaryLabelDatasetMetric(alf_dataset_female,
                                         privileged_groups=privileged_groups_female,
                                         unprivileged_groups=unprivileged_groups_female)
print("Bias against Female")
print("Disparate Impact:", metric_female.disparate_impact())
print("Statistical Parity Difference:", metric_female.statistical_parity_difference())
print()

# Bias against Unemployed
privileged_groups_unemployed = [{'EmploymentType_Unemployed_priv': 0}]
unprivileged_groups_unemployed = [{'EmploymentType_Unemployed_priv': 1}]
alf_dataset_unemployed = BinaryLabelDataset(
    df=data,
    label_names=['Loan_Approved'],
    protected_attribute_names=['EmploymentType_Unemployed_priv'],
    favorable_label=1,
    unfavorable_label=0
)
metric_unemployed = BinaryLabelDatasetMetric(alf_dataset_unemployed,
                                             privileged_groups=privileged_groups_unemployed,
                                             unprivileged_groups=unprivileged_groups_unemployed)
print("Bias against Unemployed")
print("Disparate Impact:", metric_unemployed.disparate_impact())
print("Statistical Parity Difference:", metric_unemployed.statistical_parity_difference())
print()

# Bias against Black Criminal_Record
privileged_groups_BlCr = [{'Race_Black_priv': 0, 'Criminal_Record': 1}]
unprivileged_groups_BlCr = [{'Race_Black_priv': 1, 'Criminal_Record': 1}]
alf_dataset_BlCr = BinaryLabelDataset(
    df=data,
    label_names=['Loan_Approved'],
    protected_attribute_names=['Race_Black_priv', 'Criminal_Record'],
    favorable_label=1,
    unfavorable_label=0
)
metric_BlCr = BinaryLabelDatasetMetric(alf_dataset_BlCr,
                                       privileged_groups=privileged_groups_BlCr,
                                       unprivileged_groups=unprivileged_groups_BlCr)
print("Bias against Black with Criminal Record")
print("Disparate Impact:", metric_BlCr.disparate_impact())
print("Statistical Parity Difference:", metric_BlCr.statistical_parity_difference())
print()

# Plot Loan Approval with Criminal Record between Black and Non-Black people
# Filter: only those with a criminal record
criminal_data = data[data['Criminal_Record'] == 1]

# Plot using mapping inline
sns.barplot(
    data=criminal_data,
    x=criminal_data['Race_Black_priv'].map({1: 'Black', 0: 'Non-Black'}),
    y='Loan_Approved',
    ci=None
)

plt.title('Loan Approval Rate: Black vs Non-Black (With Criminal Record)')
plt.xlabel('Race')
plt.ylabel('Approval Rate')
plt.ylim(0, 1)
plt.show()
