from sklearn.svm import SVC
import joblib

def train_and_save_svm(X_train, Y_train, model_path):
    """Train and save the SVM model."""
    model = SVC(kernel='rbf', random_state=40)
    model.fit(X_train, Y_train)
    joblib.dump(model, model_path)

def load_svm(model_path):
    """Load the SVM model."""
    return joblib.load(model_path)
