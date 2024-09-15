from src.data.load_data import load_data
from src.models.svm import train_and_save_svm
from src.processing.scaler import load_scaler

def main():
    # Load preprocessed data and scaler
    train_df = load_data('data/processed/train_data_processed.csv')
    scaler = load_scaler('models/scaler.pkl')

    # Define features and labels
    X_train = train_df.drop(columns=['attack'])
    Y_train = train_df['attack']

    # Standardize features
    X_train = scaler.transform(X_train)

    # Train and save SVM model
    train_and_save_svm(X_train, Y_train, 'models/svm_model.pkl')

    # Print sucess message
    print('!!SVM Model Training Successful!! \n Check SVM Model at /models/')

if __name__ == '__main__':
    main()
