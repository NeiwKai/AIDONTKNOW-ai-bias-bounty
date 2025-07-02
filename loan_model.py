import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# import other python module
from bias_detection import bias_detection
from dataset_loader import dataset_loader

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from fairlearn.postprocessing import ThresholdOptimizer


def main():
    pd.set_option('future.no_silent_downcasting', True)
    data = pd.read_csv("datasets/loan_access_dataset.csv")

    data = dataset_loader(data)

    #bias_detection(data) # use to detect the bias in dataset

    ignore_cols = ['Gender', 'Race', 'Income',
                   'Loan_Amount', 'Age', 'Language_Proficiency',
                   'Zip_Code_Group', 'Citizenship_Status', 'Credit_Score']#, 'Zip_Code_Group']
    data = data.drop(columns=ignore_cols)

    print(data.columns)
    print(data.head())

    X = data.drop(['Loan_Approved'], axis=1)
    Y = data['Loan_Approved']

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.4, random_state=42)

    # Define CatBoost model
    cat = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        random_seed=42,
        verbose=0
    )

    cat_features = ['Age_Group', 'Employment_Type', 'Education_Level',
                    'Disability_Status', 'Criminal_Record']

    # Train CatBoost
    cat.fit(X_train, Y_train, cat_features=cat_features)

    # Predict class labels on validation set
    Y_pred = cat.predict(X_val)
    print("Classification Report:\n", classification_report(Y_val, Y_pred))




if __name__ == "__main__":
    main()

