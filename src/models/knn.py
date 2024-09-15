from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import pickle

def train_and_save_knn(X_train, y_train, model_path):
    # KNN Classifier with GridSearchCV
    param_grid = {
        'n_neighbors': [5, 10, 20],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, verbose=2)
    grid.fit(X_train, y_train)

    # Save the trained model
    with open(model_path, 'wb') as f:
        pickle.dump(grid.best_estimator_, f)

    # Return the best model and best parameters for logging purposes
    return grid.best_estimator_, grid.best_params_

def load_knn(model_path):
    """Load the KNN model."""
    return joblib.load(model_path)
    