import matplotlib.pyplot as plt
import seaborn as sns

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

def bias_detection(data):
    # define Privilege
    data['Gender_Non-binary_priv'] = (data['Gender'] == 2).astype(int) 
    data['Race_White_priv'] = (data['Race'] == 0).astype(int)
    data['Race_Black_priv'] = (data['Race'] == 1).astype(int)
    data['Race_Native-American_priv'] = (data['Race'] == 2).astype(int)
    data['EmploymentType_Unemployed_priv'] = (data['Gender'] == 0).astype(int)
    data['CreditScore_HighCredit_priv'] = (data['Credit_Score'] >= 670).astype(int)
    data['Income_HighIncome_priv'] = (data['Income'] >= 45000).astype(int)


    criminal_data = data[data['Criminal_Record'] == 1]

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
    sns.barplot(
        data=data,
        x=data['Race_Black_priv'].map({1: 'Black', 0: 'Non-Black'}),
        y='Loan_Approved',
        errorbar=None
    )
    plt.title('Loan Approval Rate: Black vs Non-Black')
    plt.xlabel('Race')
    plt.ylabel('Approval Rate')
    plt.ylim(0, 1)
    plt.show()



    # Bias against Non-White
    privileged_groups_white = [{'Race_White_priv': 1}]
    unprivileged_groups_white = [{'Race_White_priv': 0}]
    alf_dataset_white = BinaryLabelDataset(
        df=data,
        label_names=['Loan_Approved'],
        protected_attribute_names=['Race_White_priv'],
        favorable_label=1,
        unfavorable_label=0
    )
    metric_white = BinaryLabelDatasetMetric(alf_dataset_white,
                                            privileged_groups=privileged_groups_white,
                                            unprivileged_groups=unprivileged_groups_white)
    print("Bias against Non-White")
    print("Disparate Impact:", metric_white.disparate_impact())
    print("Statistical Parity Difference:", metric_white.statistical_parity_difference())
    print()
    sns.barplot(
        data=data,
        x=data['Race_White_priv'].map({1: 'White', 0: 'Non-White'}),
        y='Loan_Approved',
        errorbar=None
    )
    plt.title('Loan Approval Rate: White vs Non-White')
    plt.xlabel('Race')
    plt.ylabel('Approval Rate')
    plt.ylim(0, 1)
    plt.show()   



    # Bias against Not Native American
    privileged_groups_nna = [{'Race_Native-American_priv': 1}]
    unprivileged_groups_nna = [{'Race_Native-American_priv': 0}]
    alf_dataset_nna = BinaryLabelDataset(
        df=data,
        label_names=['Loan_Approved'],
        protected_attribute_names=['Race_Native-American_priv'],
        favorable_label=1,
        unfavorable_label=0
    )
    metric_nna = BinaryLabelDatasetMetric(alf_dataset_nna,
                                          privileged_groups=privileged_groups_nna,
                                          unprivileged_groups=unprivileged_groups_nna)
    print("Bias against Non-Native American")
    print("Disparate Impact:", metric_nna.disparate_impact())
    print("Statistical Parity Difference:", metric_nna.statistical_parity_difference())
    print()
    sns.barplot(
        data=data,
        x=data['Race_Native-American_priv'].map({1: 'Native American', 0: 'Non-Native American'}),
        y='Loan_Approved',
        errorbar=None
    )
    plt.title('Loan Approval Rate: Native American vs Non-Native American')
    plt.xlabel('Race')
    plt.ylabel('Approval Rate')
    plt.ylim(0, 1)
    plt.show()



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
    sns.barplot(
        data=data,
        x=data['Gender_Non-binary_priv'].map({1: 'Non-binary', 0: 'Binary'}),
        y='Loan_Approved',
        errorbar=None
    )
    plt.title('Loan Approval Rate: Non-binary vs Binary')
    plt.xlabel('Gender')
    plt.ylabel('Approval Rate')
    plt.ylim(0, 1)
    plt.show()



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
    sns.barplot(
        data=data,
        x=data['EmploymentType_Unemployed_priv'].map({1: 'Unemployed', 0: 'Employed'}),
        y='Loan_Approved',
        errorbar=None
    )
    plt.title('Loan Approval Rate: Unemployed vs Employed')
    plt.xlabel('Employment')
    plt.ylabel('Approval Rate')
    plt.ylim(0, 1)
    plt.show()



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
    sns.barplot(
        data=criminal_data,
        x=criminal_data['Race_White_priv'].map({1: 'White', 0: 'Non-White'}),
        y='Loan_Approved',
        errorbar=None
    )
    plt.title('Loan Approval Rate: White vs Non-White (With Criminal Record)')
    plt.xlabel('Race')
    plt.ylabel('Approval Rate')
    plt.ylim(0, 1)
    plt.show()



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
    sns.barplot(
        data=data,
        x=data['CreditScore_HighCredit_priv'].map({1: 'High Credit', 0: 'Not High Credit'}),
        y='Loan_Approved',
        errorbar=None
    )
    plt.title('Loan Approval Rate: High Credit vs Not High Credit')
    plt.xlabel('Credit')
    plt.ylabel('Approval Rate')
    plt.ylim(0, 1)
    plt.show()



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
    sns.barplot(
        data=data,
        x=data['Income_HighIncome_priv'].map({1: 'High Income', 0: 'Not High Income'}),
        y='Loan_Approved',
        errorbar=None
    )
    plt.title('Loan Approval Rate: High Income vs Not High Income')
    plt.xlabel('Credit')
    plt.ylabel('Approval Rate')
    plt.ylim(0, 1)
    plt.show()



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
    sns.barplot(
        data=criminal_data,
        x=criminal_data['Income_HighIncome_priv'].map({1: 'High Income', 0: 'Not High Income'}),
        y='Loan_Approved',
        errorbar=None
    )
    plt.title('Loan Approval Rate: High Income vs Not High Income (With Criminal Record)')
    plt.xlabel('Credit')
    plt.ylabel('Approval Rate')
    plt.ylim(0, 1)
    plt.show()


