import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from modules.data_acquisition.video_loader import load_video_clips_with_videomae_features
from modules.model_training.train_model import train_model
from modules.model_training.evaluate_model import evaluate_model

# Import the landmark-based XAI helpers from your separate module
from modules.xai_visualization.landmark_xai import (
    extract_mean_landmarks_from_video,
    train_rf_on_landmarks,
    plot_landmark_importance
)

# ========== XAI: Ensemble Feature Importance Function ==========
def plot_ensemble_feature_importance(ensemble, feature_names=None, save_path=None):
    rf = ensemble.named_estimators_['rf']
    gb = ensemble.named_estimators_['gb']
    rf_importance = rf.feature_importances_
    gb_importance = gb.feature_importances_
    avg_importance = (rf_importance + gb_importance) / 2
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(avg_importance))]
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
]

output_frames_dir = 'I:/IIT/MSc/Data-PoC/Output/Frames'

def main():
    # === VideoMAE + Ensemble pipeline ===
    features = []
    labels = []
    for video_file in video_files:
        print(f"Processing {video_file}")
        video_features = load_video_clips_with_videomae_features(video_file, output_frames_dir)
        features.append(video_features)
        label = 0 if os.path.basename(video_file).startswith('N') else 1
        labels.extend([label] * video_features.shape[0])

    features = np.vstack(features)
    labels = np.array(labels)
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # Cross-validated performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ensemble_model = train_model(features, labels)
    scores = cross_val_score(ensemble_model, features, labels, cv=cv, scoring='accuracy')
    print("Voting ensemble cross-validated accuracy:", scores)
    print("Mean accuracy: {:.2f}% (+/- {:.2f}%)".format(scores.mean() * 100, scores.std() * 100))

    # Train/test split for XAI/demo
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

    plot_ensemble_feature_importance(
        trained_model,
        save_path='I:/IIT/MSc/Data-PoC/Output/ensemble_feature_importance.png'
    )

    # === Landmark-based Explainability (from landmark_xai.py) ===
    print("\n=== Landmark-based XAI: Extracting and visualizing landmark importance ===")
    landmark_features = []
    landmark_labels = []
    for video_file in video_files:
        mean_landmarks = extract_mean_landmarks_from_video(video_file)
        if mean_landmarks is not None:
            landmark_features.append(mean_landmarks)
            label = 0 if os.path.basename(video_file).startswith('N') else 1
            landmark_labels.append(label)
        else:
            print(f"Landmarks NOT detected for: {video_file}")

    if len(landmark_features) >= 1:  # Ensure enough data for classifier
        landmark_features = np.array(landmark_features)
        landmark_labels = np.array(landmark_labels)
        print("Landmark features shape:", landmark_features.shape)
        rf_landmark = train_rf_on_landmarks(landmark_features, landmark_labels)
        template_landmarks = landmark_features[0]  # Or np.mean(landmark_features, axis=0)
        plot_landmark_importance(
            rf_landmark.feature_importances_,
            template_landmarks,
            save_path="I:/IIT/MSc/Data-PoC/Output/landmark_importance.png"
        )
    else:
        print("Not enough landmark data for landmark-based XAI. Skipping this step.")

if __name__ == "__main__":
    main()
