from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def evaluate_model(model, test_features, test_labels):
    # No need for model.eval() since RandomForestClassifier doesn’t require it
    print("Predicting on test set...")
    predictions = model.predict(test_features)

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    auc = roc_auc_score(test_labels, model.predict_proba(test_features)[:, 1]) if len(set(test_labels)) > 1 else None
    report = classification_report(test_labels, predictions, output_dict=True)

    # Output metrics
    metrics = {
        "accuracy": accuracy,
        "AUC": auc,
        "classification_report": report,
    }

    print("Evaluation metrics computed.")
    return metrics
