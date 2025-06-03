import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from modules.data_acquisition.video_loader import load_video_clips_with_videomae_features
from modules.model_training.train_model import train_model
from modules.model_training.evaluate_model import evaluate_model

# ================= XAI: Ensemble Feature Importance Function =================

def plot_ensemble_feature_importance(ensemble, feature_names=None, save_path=None):
    # Get tree-based models from the ensemble
    rf = ensemble.named_estimators_['rf']
    gb = ensemble.named_estimators_['gb']
    # Get feature importances and average them
    rf_importance = rf.feature_importances_
    gb_importance = gb.feature_importances_
    avg_importance = (rf_importance + gb_importance) / 2
    # Feature names for plot
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(avg_importance))]
    # Plot
    plt.figure(figsize=(8, 4))
    plt.bar(feature_names, avg_importance)
    plt.title('Average Feature Importance (Voting Ensemble)')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Feature importance plot saved to {save_path}")
    plt.show()

# =========================== MAIN PIPELINE ================================

# Paths to videos
video_files = [
    'I:/IIT/MSc/Data-PoC/Videos/A002_02_DDK_PATAKA_color.avi',
    'I:/IIT/MSc/Data-PoC/Videos/A006_02_DDK_PATAKA_color.avi',
    'I:/IIT/MSc/Data-PoC/Videos/A008_02_DDK_PATAKA_color.avi',
    'I:/IIT/MSc/Data-PoC/Videos/A009_02_DDK_PATAKA_color.avi',
    'I:/IIT/MSc/Data-PoC/Videos/A010_02_DDK_PATAKA_color.avi',
    'I:/IIT/MSc/Data-PoC/Videos/A015_02_DDK_PATAKA_color.avi',
    'I:/IIT/MSc/Data-PoC/Videos/N001_02_DDK_PATAKA_color.avi',
    'I:/IIT/MSc/Data-PoC/Videos/N002_02_DDK_PATAKA_color.avi',
    'I:/IIT/MSc/Data-PoC/Videos/N003_02_DDK_PATAKA_color.avi',
    # Add more video paths as needed
]
output_frames_dir = 'I:/IIT/MSc/Data-PoC/Output/Frames'

def main():
    features = []
    labels = []

    # Load videos, extract features, and assign labels
    for video_file in video_files:
        print(f"Processing {video_file}")
        video_features = load_video_clips_with_videomae_features(video_file, output_frames_dir)
        features.append(video_features)
        # Determine label based on filename
        label = 0 if os.path.basename(video_file).startswith('N') else 1
        labels.extend([label] * video_features.shape[0])

    # Flatten features and convert to numpy arrays for training
    features = np.vstack(features)
    labels = np.array(labels)
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # --------- Cross-Validated Ensemble Performance (Recommended for reporting) ---------
    from modules.model_training.train_model import train_model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ensemble_model = train_model(features, labels)
    scores = cross_val_score(ensemble_model, features, labels, cv=cv, scoring='accuracy')
    print("Voting ensemble cross-validated accuracy:", scores)
    print("Mean accuracy: {:.2f}% (+/- {:.2f}%)".format(scores.mean() * 100, scores.std() * 100))

    # --------- Standard Train/Test Split & Evaluation (for XAI and presentation) ---------
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Training the ensemble model...")
    trained_model = train_model(train_features, train_labels)
    print("Model training complete.")

    print("Evaluating the ensemble model...")
    metrics = evaluate_model(trained_model, test_features, test_labels)
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # --------- Plot and Save Feature Importance from Ensemble ---------
    plot_ensemble_feature_importance(
        trained_model,
        save_path='I:/IIT/MSc/Data-PoC/Output/ensemble_feature_importance.png'
    )

if __name__ == "__main__":
    main()
