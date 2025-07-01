import pandas as pd

# import other python module
from bias_detection import bias_detection
from dataset_loader import dataset_loader

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate

def main():
    data = pd.read_csv("datasets/loan_access_dataset.csv")

    data = dataset_loader(data)


    #bias_detection(data) # use to detect the bias in dataset
    # Drop unwanted column before training process
    ignore_cols = ['Gender', 'Race', 'Age', 'Income', 'Credit_Score', 'Loan_Amount', 'Employment_Type', 'Zip_Code_Group']#, 'Zip_Code_Group']
    data = data.drop(columns=ignore_cols)

    print(data.columns)
    print(data.head())

    X = data.drop(['Loan_Approved'], axis=1)
    Y = data['Loan_Approved']
    X.shape, Y.shape
    X_train, X_val, Y_train, Y_val= train_test_split(X, Y, test_size=0.4, random_state=1)
    X_train.shape, X_val.shape, Y_train.shape, Y_val.shape

    sensitive_feature = X_val[['Credit_Level', 'Meets_LTI_Criteria']]


    # Define the base model
    rf = RandomForestClassifier(random_state=42)

    # Define hyperparameter space for tuning
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 10],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', None]
    }

    # Set up randomized search
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=30,                # Number of combinations to try
        cv=10,                     # 5-fold cross-validation
        verbose=1,
        scoring='accuracy',      # or 'f1', 'roc_auc', etc.
        random_state=42,
        n_jobs=-1
    )

    # Run the search
    random_search.fit(X_train, Y_train)

    # Best model
    best_rf = random_search.best_estimator_

    # Evaluate on validation set
    Y_pred = best_rf.predict(X_val)
    print("Best parameters:", random_search.best_params_)
    print("Validation Accuracy:", accuracy_score(Y_val, Y_pred))
    print("Classification Report:\n", classification_report(Y_val, Y_pred))

    mf = MetricFrame(
        metrics={
            'accuracy': accuracy_score,
            'selection_rate': selection_rate,
            'TPR': true_positive_rate,
            'FPR': false_positive_rate
        },
        y_true=Y_val,
        y_pred=Y_pred,
        sensitive_features=sensitive_feature
    )

    print("\n✅ Overall metrics:")
    print(mf.overall)

    print("\n📊 Group-wise metrics:")
    print(mf.by_group)

    print("\n⚠️ Fairness Gaps:")
    print("Selection rate diff:", mf.difference()['selection_rate'])
    print("Equalized odds diff (max of TPR/FPR):", max(mf.difference()['TPR'], mf.difference()['FPR']))

if __name__ == "__main__":
    main()
