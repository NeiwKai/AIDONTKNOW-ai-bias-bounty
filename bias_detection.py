import matplotlib.pyplot as plt
import seaborn as sns

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

def bias_detection(data):
    # define Privilege
    data['Gender_Male_priv'] = (data['Gender'] == 1).astype(int)
    data['Gender_Female_priv'] = (data['Gender'] == 0).astype(int)
    data['Gender_Non-binary_priv'] = (data['Gender'] == 2).astype(int) 
    data['Race_White_priv'] = (data['Race'] == 0).astype(int)
    data['Race_Black_priv'] = (data['Race'] == 1).astype(int)
    data['Race_Asian_priv'] = (data['Race'] == 5).astype(int)
    data['Race_Native-American_priv'] = (data['Race'] == 4).astype(int)
    data['EmploymentType_Unemployed_priv'] = (data['Gender'] == 0).astype(int)
    data['CreditScore_HighCredit_priv'] = (data['Credit_Score'] >= 670).astype(int)
    data['Income_HighIncome_priv'] = (data['Income'] >= 45000).astype(int)

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

    # Bias against Non-White Criminal_Record
    privileged_groups_NonWhCr = [{'Race_White_priv': 1, 'Criminal_Record': 1}]
    unprivileged_groups_NonWhCr = [{'Race_White_priv': 0, 'Criminal_Record': 1}]
    alf_dataset_NonWhCr = BinaryLabelDataset(
        df=data,
        label_names=['Loan_Approved'],
        protected_attribute_names=['Race_White_priv', 'Criminal_Record'],
        favorable_label=1,
        unfavorable_label=0
    )
    metric_NonWhCr = BinaryLabelDatasetMetric(alf_dataset_NonWhCr,
                                           privileged_groups=privileged_groups_NonWhCr,
                                           unprivileged_groups=unprivileged_groups_NonWhCr)
    print("Bias against Non-White with Criminal Record")
    print("Disparate Impact:", metric_NonWhCr.disparate_impact())
    print("Statistical Parity Difference:", metric_NonWhCr.statistical_parity_difference())
    print()

    # Bias against Non-High Credit Score
    privileged_groups_NonHCredit = [{'CreditScore_HighCredit_priv': 1}]
    unprivileged_groups_NonHCredit = [{'CreditScore_HighCredit_priv': 0}]
    alf_dataset_NonHCredit = BinaryLabelDataset(
        df=data,
        label_names=['Loan_Approved'],
        protected_attribute_names=['CreditScore_HighCredit_priv'],
        favorable_label=1,
        unfavorable_label=0
    )
    metric_NonHCredit = BinaryLabelDatasetMetric(alf_dataset_NonHCredit,
                                                privileged_groups=privileged_groups_NonHCredit,
                                                unprivileged_groups=unprivileged_groups_NonHCredit)
    print("Bias against Non-High Credit Score")
    print("Disparate Impact:", metric_NonHCredit.disparate_impact())
    print("Statistical Parity Difference:", metric_NonHCredit.statistical_parity_difference())
    print()

    # Bias against Non-High Income
    privileged_groups_HighIncome = [{'Income_HighIncome_priv': 1}]
    unprivileged_groups_HighIncome = [{'Income_HighIncome_priv': 0}]
    alf_dataset_HighIncome = BinaryLabelDataset(
        df=data,
        label_names=['Loan_Approved'],
        protected_attribute_names=['Income_HighIncome_priv'],
        favorable_label=1,
        unfavorable_label=0
    )
    metric_HighIncome = BinaryLabelDatasetMetric(alf_dataset_HighIncome,
                                                 privileged_groups=privileged_groups_HighIncome,
                                                 unprivileged_groups=unprivileged_groups_HighIncome)
    print("Bias against Non-High Income")
    print("Disparate Impact:", metric_HighIncome.disparate_impact())
    print("Statistical Parity Difference:", metric_HighIncome.statistical_parity_difference())
    print()

    # Bias against Non-High Credit with Criminal Record
    privileged_groups_NonHCredCr = [{'CreditScore_HighCredit_priv': 1,
                                     'Criminal_Record': 1}]
    unprivileged_groups_NonHCredCr = [{'CreditScore_HighCredit_priv': 0,
                                       'Criminal_Record': 1}]
    alf_dataset_NonHCredCr = BinaryLabelDataset(
        df=data,
        label_names=['Loan_Approved'],
        protected_attribute_names=['CreditScore_HighCredit_priv', 'Criminal_Record'],
        favorable_label=1,
        unfavorable_label=0
    )
    metric_NonHCredCr = BinaryLabelDatasetMetric(alf_dataset_NonHCredCr,
                                                 privileged_groups=privileged_groups_NonHCredCr,
                                                 unprivileged_groups=unprivileged_groups_NonHCredCr)
    print("Bias against Non-High Credit with Criminal Record")
    print("Disparate Impact:", metric_NonHCredCr.disparate_impact())
    print("Statistical Parity Difference:", metric_NonHCredCr.statistical_parity_difference())
    print()

    # Bias against Non-High Income with Criminal Record
    privileged_groups_NonHInCr = [{'Income_HighIncome_priv': 1,
                                   'Criminal_Record': 1}]
    unprivileged_groups_NonHInCr = [{'Income_HighIncome_priv': 0,
                                     'Criminal_Record': 1}]
    alf_dataset_NonHInCr = BinaryLabelDataset(
        df=data,
        label_names=['Loan_Approved'],
        protected_attribute_names=['Income_HighIncome_priv', 'Criminal_Record'],
        favorable_label=1,
        unfavorable_label=0
    )
    metric_NonHInCr = BinaryLabelDatasetMetric(alf_dataset_NonHInCr,
                                               privileged_groups=privileged_groups_NonHInCr,
                                               unprivileged_groups=unprivileged_groups_NonHInCr)
    print("Bias against Non-High Income with Criminal Record")
    print("Disparate Impact:", metric_NonHInCr. disparate_impact())
    print("Statistical Parity Difference:", metric_NonHInCr.statistical_parity_difference())
    print()

    criminal_data = data[data['Criminal_Record'] == 1]

    # Plot Loan Approval with Criminal Record Between White and Non-White people
    sns.barplot(
        data=criminal_data,
        x=criminal_data['Race_White_priv'].map({1: 'White', 0: 'Non-White'}),
        y='Loan_Approved',
        ci=None
    )
    plt.title('Loan Approval Rate: White vs Non-White (With Criminal Record)')
    plt.xlabel('Race')
    plt.ylabel('Approval Rate')
    plt.ylim(0, 1)
    plt.show()
