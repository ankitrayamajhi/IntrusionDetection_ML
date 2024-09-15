from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_rf(X_train, Y_train, model_path):
    """Train and save the Random Forest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    joblib.dump(model, model_path)

def load_rf(model_path):
    """Load the Random Forest model."""
    return joblib.load(model_path)
