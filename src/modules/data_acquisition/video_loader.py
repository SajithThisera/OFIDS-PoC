import cv2
import os
import numpy as np
import torch
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification

def load_video_clips_with_videomae_features(video_path, output_frames_dir, frame_rate=1, max_frames=16):

    # Ensure the output directory exists
    os.makedirs(output_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return np.array([])

    frames = []
    frame_count = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_rate == 0:
            resized_frame = cv2.resize(frame, (224, 224))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            print(f"Processed frame {frame_count} for VideoMAE feature extraction.")
        frame_count += 1
    cap.release()

    # Check if we have the correct number of frames
    if len(frames) < max_frames:
        print(f"Warning: Video {video_path} has only {len(frames)} frames, which is less than the required {max_frames}.")
        frames += [frames[-1]] * (max_frames - len(frames))  # Repeat the last frame if needed

    # Initialize VideoMAE model and extractor
    extractor = VideoMAEFeatureExtractor()
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base").to("cpu")

    print("Extracting VideoMAE features...")
    inputs = extractor(frames, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model(**inputs)
    features = outputs.logits.cpu().numpy()
    print(f"Extracted VideoMAE features with shape: {features.shape}")

    return features
