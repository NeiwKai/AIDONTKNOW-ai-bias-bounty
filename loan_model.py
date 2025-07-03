import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# import other python module
from bias_detection import bias_detection
from dataset_loader import dataset_loader

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from scipy.stats import uniform, randint

from fairlearn.postprocessing import ThresholdOptimizer

# Create new Race label
def clean_race(race):
    return race if race in [0, 1, 2] else 99  # Keep top 3 groups, others → 99, for better fairness detection and Post processinc

def main():
    pd.set_option('future.no_silent_downcasting', True)
    data = pd.read_csv("datasets/loan_access_dataset.csv")

    # Ignore column ID
    if 'ID' in data.columns:
        data = data.drop(columns=['ID'])

    obj = (data.dtypes == 'object')
    print("Categorical variables:",len(list(obj[obj].index)))


    data = dataset_loader(data)
    data['Race'] = data['Race'].apply(clean_race)


    print(data.groupby(['Gender', 'Race']).size())

    # Bias Detection
    bias_detection(data) # use to detect the bias in dataset with aif360
    # Drop column that use to check bias
    drop_list = ['Gender_Non-binary_priv', 'Race_White_priv', 'Race_Black_priv', 'Race_Native-American_priv', 'EmploymentType_Unemployed_priv', 'CreditScore_HighCredit_priv', 'Income_HighIncome_priv']
    data = data.drop(columns=drop_list, errors='ignore')
    # --------------

    print(data.columns)
    print(data.head())
    

    # split the train and val dataset
    X = data.drop(['Loan_Approved'], axis=1)
    Y = data['Loan_Approved']
    X.shape, Y.shape
    X_train, X_val, Y_train, Y_val= train_test_split(X, Y, test_size=0.4, random_state=42)
    X_train.shape, X_val.shape, Y_train.shape, Y_val.shape



    # Define the CatBoost model
    cat = CatBoostClassifier(
        iterations=87,      
        learning_rate=0.2,   
        depth=6,              
        #verbose=0, 
        loss_function='Focal:focal_alpha=0.5;focal_gamma=1.5',
        random_seed=42 
    )
    cat_features = ['Gender', 'Race', 'Age_Group', 'Employment_Type', 'Education_Level', 'Citizenship_Status',
                    'Language_Proficiency', 'Disability_Status', 'Criminal_Record', 'Zip_Code_Group']
    cat.fit(X_train, Y_train)
    Y_pred = cat.predict(X_val)
    print("Classification Report:\n", classification_report(Y_val, Y_pred))





    sensitive_features = X_val[['Gender', 'Race']]
    print("Calculating Unfairness in Gender and Race...")
    mf = MetricFrame(
        metrics={
            'accuracy': accuracy_score,
            'selection_rate': selection_rate,
            'TPR': true_positive_rate,
            'FPR': false_positive_rate
        },
        y_true=Y_val,
        y_pred=Y_pred,
        sensitive_features=sensitive_features
    )

    print("\n✅ Overall metrics:")
    print(mf.overall)

    print("\n📊 Group-wise metrics:")
    print(mf.by_group)

    print("\n⚠️ Fairness Gaps:")
    print("Selection rate diff:", mf.difference()['selection_rate'])
    print("Equalized odds diff (max of TPR/FPR):", max(mf.difference()['TPR'], mf.difference()['FPR']))

    df = mf.by_group.reset_index()  # convert MetricFrame to DataFrame

    plt.figure(figsize=(12, 6))
    sns.barplot(x=df["Race"].map({0: 'White', 1: 'Black', 2:'Native American', 99: 'Other'}), y="selection_rate", hue=df["Gender"].map({0: 'Male', 1: 'Female', 2: 'Non-binary'}), data=df)
    plt.title("Selection Rate by Gender and Race")
    plt.ylabel("Selection Rate")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


    # Posprocessing the unfairness
    postprocessor = ThresholdOptimizer(
        estimator=cat,
        constraints="equalized_odds",  # or "demographic_parity"
        prefit=True
    )
    # Fit using validation set
    postprocessor.fit(X_val, Y_val, sensitive_features=X_val[['Gender']])

    # Predict with fairness constraints
    Y_pred_fair = postprocessor.predict(X_val, sensitive_features=X_val[['Gender']])

    mf_after = MetricFrame(
        metrics={
            'accuracy': accuracy_score,
            'selection_rate': selection_rate,
            'TPR': true_positive_rate,
            'FPR': false_positive_rate
        },
        y_true=Y_val,
        y_pred=Y_pred_fair,
        sensitive_features=X_val[['Gender', 'Race']]
    )

    print("\n✅ AFTER Mitigation:")
    print("Overall metrics:\n", mf_after.overall)
    print("\nGroup-wise:\n", mf_after.by_group)
    print("\n⚠️ Fairness Gaps:")
    print("Selection rate diff:", mf_after.difference()['selection_rate'])
    print("Equalized odds diff (max of TPR/FPR):", max(mf_after.difference()['TPR'], mf_after.difference()['FPR']))

    df = mf_after.by_group.reset_index()  # convert MetricFrame to DataFrame

    plt.figure(figsize=(12, 6))
    sns.barplot(x=df["Race"].map({0: 'White', 1: 'Black', 2:'Native American', 99: 'Other'}), y="selection_rate", hue=df["Gender"].map({0: 'Male', 1: 'Female', 2: 'Non-binary'}), data=df)
    plt.title("Selection Rate by Gender and Race (After Mitigation)")
    plt.ylabel("Selection Rate")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


    print("Begin testing... output->submission.csv")
    test_data = pd.read_csv("datasets/test.csv")
    test_data = dataset_loader(test_data)

    test_data['Race'] = test_data['Race'].apply(clean_race)   

    X_test = test_data.drop(['Loan_Approved'], axis=1, errors='ignore')
    sensitive_features_test = X_test[['Gender']]

    Y_test_pred_fair = postprocessor.predict(X_test, sensitive_features=sensitive_features_test)

    # Map predictions to strings
    mapping = {1: "Approved", 0: "Denied"}
    loan_approval_status = pd.Series(Y_test_pred_fair).map(mapping)

    submission = pd.DataFrame({
        'ID': test_data['ID'],            # make sure 'ID' is your actual ID column name
        'Loan_Approve': loan_approval_status
    })

    submission.to_csv('datasets/submission.csv', index=False)
    print("Submission saved to datasets/submission.csv")


if __name__ == "__main__":
    main()
