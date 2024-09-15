import pandas as pd
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data

def main():
    # Load raw data
    train_df = load_data('data/raw/KDDTrain+.txt')
    test_df = load_data('data/raw/KDDTest+.txt')

    # Preprocess data
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # Save preprocessed data
    train_df.to_csv('data/processed/train_data_processed.csv', index=False)
    test_df.to_csv('data/processed/test_data_processed.csv', index=False)

if __name__ == '__main__':
    main()
