import pandas as pd
from src.data.load_data import load_data
from src.models.gbm import train_and_save_gbm
from src.processing.scaler import load_scaler

def main():
    # Load preprocessed data and scaler
    train_df = pd.read_csv('data/processed/train_data_processed.csv')
    scaler = load_scaler('models/scaler.pkl')

    # Define features and labels
    X_train = train_df.drop(columns=['attack'])
    Y_train = train_df['attack']

    # Standardize features
    X_train = scaler.transform(X_train)

    # Train and save GBM model
    train_and_save_gbm(X_train, Y_train, 'models/gbm_model.pkl')

if __name__ == '__main__':
    main()
