import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data.load_data import load_data
from src.processing.scaler import load_scaler

def plot_cm(y_true, y_pred, title):
    """Plot confusion matrix."""
    figsize = (10, 10)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap="mako", annot=annot, fmt='', ax=ax)
    plt.show()

def evaluate_model(model_path, scaler_path, test_data_path):
    """Evaluate the model on test data."""
    # Load test data and scaler
    test_df = pd.read_csv(test_data_path)
    scaler = load_scaler(scaler_path)

    # Define features and labels
    X_test = test_df.drop(columns=['attack'])
    Y_test = test_df['attack']

    # Standardize features
    X_test = scaler.transform(X_test)

    # Load model
    model = joblib.load(model_path)

    # Make predictions
    Y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f'Testing accuracy: {accuracy}')

    # Print classification report
    report = classification_report(Y_test, Y_pred)
    print(report)

    # Plot confusion matrix
    plot_cm(Y_test, Y_pred, f'Confusion Matrix for {model_path}')

if __name__ == '__main__':
    # Evaluate all models
    evaluate_model('models/gbm_model.pkl', 'models/scaler.pkl', 'data/processed/test_data_processed.csv')
    evaluate_model('models/svm_model.pkl', 'models/scaler.pkl', 'data/processed/test_data_processed.csv')
    evaluate_model('models/random_forest_model.pkl', 'models/scaler.pkl', 'data/processed/test_data_processed.csv')
    evaluate_model('models/knn_model.pkl', 'models/scaler.pkl', 'data/processed/test_data_processed.csv')
