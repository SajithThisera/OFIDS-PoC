# train_model.py
from sklearn.ensemble import RandomForestClassifier

def train_model(features, labels):
    print("Initializing RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(features, labels)
    print("Model training completed.")
    return model
