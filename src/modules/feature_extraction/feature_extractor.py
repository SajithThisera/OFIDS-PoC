# feature_extractor.py
from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor
import torch

# Load the pre-trained VideoMAE model and feature extractor
video_mae_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base")
video_mae_feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
video_mae_model.eval()

def extract_videomae_features(clips):
    features = []
    for clip in clips:
        inputs = video_mae_feature_extractor(images=clip, return_tensors="pt")
        with torch.no_grad():
            outputs = video_mae_model(**inputs)
        features.append(outputs.logits)  # or select intermediate layers if needed
    return torch.cat(features, dim=0).numpy()
