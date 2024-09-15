from sklearn.preprocessing import StandardScaler
import joblib

def train_and_save_scaler(X_train, scaler_path):
    """Train and save the scaler."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, scaler_path)

def load_scaler(scaler_path):
    """Load the scaler."""
    return joblib.load(scaler_path)
