# train_model.py (voting ensemble version)
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def train_model(features, labels):
    print("Initializing VotingClassifier ensemble...")
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
    lr = LogisticRegression(max_iter=1000)
    svm = SVC(probability=True)
    model = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('lr', lr),
            ('svm', svm)
        ],
        voting='soft'
    )
    model.fit(features, labels)
    print("Ensemble model training completed.")
    return model
