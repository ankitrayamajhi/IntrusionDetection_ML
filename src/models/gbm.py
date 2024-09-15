from sklearn.ensemble import GradientBoostingClassifier
import joblib

def train_and_save_gbm(X_train, Y_train, model_path):
    """Train and save the Gradient Boosting model."""
    model = GradientBoostingClassifier(random_state=40)
    model.fit(X_train, Y_train)
    joblib.dump(model, model_path)

def load_gbm(model_path):
    """Load the Gradient Boosting model."""
    return joblib.load(model_path)
