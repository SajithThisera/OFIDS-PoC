import os
import numpy as np
from sklearn.model_selection import train_test_split
from modules.data_acquisition.video_loader import load_video_clips_with_videomae_features
from modules.model_training.train_model import train_model
from modules.model_training.evaluate_model import evaluate_model
from modules.xai_visualization.xai_generator import generate_shap_explanations

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
    # Add more video paths here
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
        labels.extend([label] * len(video_features))

    # Flatten features and convert to numpy arrays for training
    features = np.vstack(features)
    labels = np.array(labels)
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # Split data into train and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Train model
    print("Training the model...")
    trained_model = train_model(train_features, train_labels)
    print("Model training complete.")

    # Evaluate model
    print("Evaluating the model...")
    metrics = evaluate_model(trained_model, test_features, test_labels)
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Generate SHAP explanations
    print("Generating SHAP explanations and saving plots...")
    generate_shap_explanations(trained_model, test_features, output_dir='I:/IIT/MSc/Data-PoC/Output/XAI')

if __name__ == "__main__":
    main()
