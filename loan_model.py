import pandas as pd

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

if __name__ == "__main__":
    main()
