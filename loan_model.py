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

# Create new Race label
def clean_race(race):
    return race if race in [0, 1, 2] else 99  # Keep top 3 groups, others → 99

def main():
    pd.set_option('future.no_silent_downcasting', True)
    data = pd.read_csv("datasets/loan_access_dataset.csv")

    if 'ID' in data.columns:
        data = data.drop(columns=['ID'])

    data = dataset_loader(data)

    data['Race'] = data['Race'].apply(clean_race)


    print(data.groupby(['Gender', 'Race']).size())


    #bias_detection(data) # use to detect the bias in dataset
    # Drop unwanted column before training process
    '''
    ignore_cols = ['Gender', 'Race', 'Income', 'Loan_Amount', 'Age']#, 'Zip_Code_Group']
    data = data.drop(columns=ignore_cols)
    '''

    print(data.columns)
    print(data.head())

    #print(data.corr()['Loan_Approved'].sort_values())

    X = data.drop(['Loan_Approved'], axis=1)
    Y = data['Loan_Approved']
    X.shape, Y.shape
    X_train, X_val, Y_train, Y_val= train_test_split(X, Y, test_size=0.4, random_state=42)
    X_train.shape, X_val.shape, Y_train.shape, Y_val.shape



    # Define the base model
    cat = CatBoostClassifier(
        iterations=20,      
        learning_rate=0.1,   
        depth=6,              
        #verbose=0, 
        loss_function='Logloss', 
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
    sns.barplot(x="Race", y="selection_rate", hue="Gender", data=df)
    plt.title("Selection Rate by Gender and Race")
    plt.ylabel("Selection Rate")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


    from fairlearn.postprocessing import ThresholdOptimizer

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


    print("Begin the submission.csv")
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
