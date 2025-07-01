import pandas as pd

#import machine learning libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformerfrom sklearn.pipeline import Pipeline


# import other python module
from bias_detection import bias_detection
from dataset_loader import dataset_loader

def main():
    data = pd.read_csv("datasets/loan_access_dataset.csv")

    data = dataset_loader(data)

    '''
    print(data.columns)
    print(data.head())
    '''

    bias_detection(data)
    
    categorical_cols = data.drop(columns=['Loan_Approved']).select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],remainder='passthrough')
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))   
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=['Loan_Approved']),
        data['Loan_Approved'],
        test_size=0.3,
        random_state=42,
        stratify=data['Loan_Approved']
    )
    
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    
    output = pd.DataFrame({
    'ID': X_test.index,
    'Loan_Approved': predictions  # 1 = approved, 0 = not approved
    })
    output.to_csv("loan_approval_predictions.csv", index=False)
    print("Prediction complete. Output saved as 'loan_approval_predictions.csv'")


    

if __name__ == "__main__":
    main()
