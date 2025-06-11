import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules.data_preprocessing.face_alignment_extractor import FaceAlignmentExtractor
from sklearn.ensemble import RandomForestClassifier

def extract_mean_landmarks_from_video(video_path):
    fa_extractor = FaceAlignmentExtractor()
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = fa_extractor.extract_landmarks_from_frame(frame)
        if landmarks is not None:
            landmarks_list.append(landmarks)
    cap.release()
    if not landmarks_list:
        return None
    landmarks_array = np.array(landmarks_list)
    mean_landmarks = np.mean(landmarks_array, axis=0)
    return mean_landmarks  # shape: (136,)

def train_rf_on_landmarks(landmark_features, labels):
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(landmark_features, labels)
    return rf

def plot_landmark_importance(importances, template_landmarks, save_path=None):
    num_points = importances.shape[0] // 2
    xs = template_landmarks[::2]
    ys = template_landmarks[1::2]
    scores = np.sqrt(importances[::2]**2 + importances[1::2]**2)
    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c=scores, cmap='hot', s=100)
    plt.title("Facial Landmark Importance")
    plt.colorbar(label="Importance")
    plt.gca().invert_yaxis()
    if save_path:
        plt.savefig(save_path)
        print(f"Landmark importance plot saved to {save_path}")
    plt.show()
