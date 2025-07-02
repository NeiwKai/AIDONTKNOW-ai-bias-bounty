import pandas as pd
import numpy as np

def credit_score_level(score):
    if score >= 800:
        return 4
    elif score >= 740:
        return 3
    elif score >= 670:
        return 2
    elif score >= 580:
        return 1
    else:
        return 0

def dataset_loader(data):
    if 'Loan_Approved' in data.columns:
            data['Loan_Approved'] = data['Loan_Approved'].replace({'Approved': 1, 'Denied': 0}).astype(int)
    # Replace string values with integer
    binary_cols = ['Disability_Status', 'Criminal_Record']
    data[binary_cols] = data[binary_cols].replace({'No': 0, 'Yes': 1})
    
    #for bias_detection
    data['Gender'] = data['Gender'].replace({'Male': 1, 'Female': 0, 'Non-binary': 2}).astype(int)
    data['Race'] = data['Race'].replace({'White': 0, 'Black': 1, 'Hispanic': 2, 'Multiracial': 3, 'Native American': 4, 'Asian': 5}).astype(int)
    # -----------------

    '''
    data = pd.get_dummies(data, columns=['Gender', 'Race',
                                         'Employment_Type', 'Education_Level', 'Zip_Code_Group',
                                         'Citizenship_Status', 'Age_Group', 'Language_Proficiency'], drop_first=True)
    '''

    data['Employment_Type'] = data['Employment_Type'].replace({'Unemployed': 0, 'Part-time': 1, 'Gig': 2, 'Self-employed': 3, 'Full-time': 4}).astype(int)
    data['Education_Level'] = data['Education_Level'].replace({'High School': 0, 'Some College': 1, "Bachelor's": 2, 'Graduate': 3}).astype(int)
    data['Zip_Code_Group'] = data['Zip_Code_Group'].replace({'Historically Redlined': 0, 'Rural': 1, 'Urban Professional': 2,
                                                             'Working Class Urban': 3, 'High-income Suburban': 4}).astype(int)
    data['Citizenship_Status'] = data['Citizenship_Status'].replace({'Visa Holder': 0, 'Permanent Resident': 1, 'Citizen': 2}).astype(int)
    data['Age_Group'] = data['Age_Group'].replace({'Over 60': 0, '25-60': 1, 'Under 25': 2}).astype(int)
    data['Language_Proficiency'] = data['Language_Proficiency'].replace({'Limited': 0, 'Fluent': 1}).astype(int)

    # For further training
    data['Loan_to_Income'] = data['Loan_Amount'] / (data['Income'] + 1)  # +1 to avoid division by zero
    data['Meets_LTI_Criteria'] = (data['Loan_to_Income'] < 3).astype(int)
    data['Credit_Level'] = data['Credit_Score'].apply(credit_score_level)

    return data
